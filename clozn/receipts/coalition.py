"""coalition.py -- batched coalition/Shapley causal credit (docs/PRODUCT_ROADMAP.md §8 tail engine-debt
item: "batched causal credit (coalition/Shapley over teacher-forced arms -- the seam is in
clozn/receipts/core.py, on top of the /v1/branch batched-decode primitive)"), built on top of core.py's
leave-one-out (LOO) machinery.

WHY THIS EXISTS (this project's own causal-tracer autopsy -- the measured motivation, not a hunch): solo
(leave-one-out) attribution systematically OVERCOUNTS a joint effect. The tracer study found a median
interaction_gap/sum_solo around -60% (-82% median on a larger model, Llama-3.1-8B): summing every influence's OWN solo
delta ran roughly 2.5x the size of what removing them all TOGETHER actually did. "Sum-of-singles != joint
ablation" cost a wrong reported result once already (the same project's own retrospective). This module
never reports solo deltas alone -- every report also runs pairwise coalitions, a joint (all-at-once) arm,
and states the resulting gap plainly, with that overcounting caveat attached.

WHAT'S COMPUTED, at what N (cost model -- every additional coalition is one more greedy (re-)generation,
extending core._APPROX_NOTE/_PERF_NOTE's existing accounting):
  * solo (leave-one-out): N arms -- REUSED from the caller's own already-computed LOO receipts, never
    regenerated a second time.
  * pairwise: all C(N,2) pairs for N <= EXHAUSTIVE_PAIRS_MAX_N (5); for larger N, only the top-K pairs by
    solo magnitude (K = min(C(N,2), top_k_pairs), stated in every report as `pairs_capped`/`k_pairs`).
  * joint (every influence ablated together): always exactly 1 arm, whatever N is -- what the interaction
    gap is measured against.
  * Shapley value: EXACT (the standard weighted-marginal-contribution formula over the full 2^N power set)
    when N <= EXACT_SHAPLEY_MAX_N (4) -- affordable because solo+pairs+joint already cover the full power
    set through N=3, and N=4 needs only 4 more (triple) arms beyond that. For N=5 (all pairs computed, but
    a full power set would need 26 more coalitions) and N>5 (only a pair sample), Shapley falls back to a
    documented, closed-form 2nd-order approximation -- the Shapley-Taylor interaction index (Sundararajan,
    Dhamdhere & Agarwal, "The Shapley Taylor Interaction Index", 2020): each influence's solo delta, adjusted
    by the average pairwise interaction term split evenly between the two influences in each pair. This is
    NEVER presented as exact; every report states which class it used and, for the approximate class, the
    exact sample count (how many coalitions were actually run) plus a bootstrap CI on the interaction term
    (reusing clozn.experiments.stats.paired_bootstrap_ci -- the same paired-resampling machinery Phase 4.4
    built for experiment reports, applied here to a single run's own set of pairwise measurements).

BATCHING (the /v1/branch engine primitive) -- read directly from engine/core/src/generate_ar.cpp's
`generate_ar_branched` (as shipped): it batches N branches from ONE shared checkpoint under a SINGLE shared
sampling config; branches differ ONLY by an RNG-seed offset, never by a distinct per-branch dial/steer
intervention. No substrate today can therefore batch HETEROGENEOUS coalition arms (each ablating a
different subset) through it -- that needs per-branch steering the shipped primitive does not have. This
module's batching hook (`sub.branch_coalitions`) is accordingly a forward-looking, duck-typed, OPT-IN seam
for a future substrate that adds real per-arm steering to /v1/branch. The repo's FP-landmine rule applies
without exception: a substrate is never trusted silently. See `_attempt_batched`'s docstring for the exact
three-way contract (`coalitions_batch`: "auto" | "off" | "approximate") this module enforces.
"""
from __future__ import annotations

from itertools import combinations
import math

from .core import _ablated_child
from .deltas import _merge_ablation_changes
from .metrics import receipt_metrics

EXACT_SHAPLEY_MAX_N = 4
EXHAUSTIVE_PAIRS_MAX_N = 5
DEFAULT_TOP_K_PAIRS = 10

OVERCOUNTING_CAVEAT = (
    "solo (leave-one-out) attribution characteristically OVERCOUNTS a joint effect: this project's own "
    "causal-tracing study measured a median interaction_gap/sum_solo around -60% (-82% median on a larger "
    "model, Llama-3.1-8B) -- summing every influence's solo delta ran roughly 2.5x the size of the actual joint effect. "
    "Read THIS run's own gap below; it is not guaranteed to match that figure."
)

_BATCH_NOTE = (
    "the shipped /v1/branch engine primitive batches N branches from one shared checkpoint under a SINGLE "
    "shared sampling config (branches differ only by an RNG-seed offset, per engine/core/src/generate_ar.cpp "
    "-- see module docstring); no substrate today can batch heterogeneous coalition arms through it, so this "
    "run took the sequential path."
)


# ============================================================================================ value function

def _arm_value(baseline_reply: str, reply: str) -> dict:
    """One coalition arm's measured effect: `value` is the word-type Jaccard-change fraction
    (receipts.metrics.receipt_metrics's `changed`/100, the same magnitude runner.judge_receipt already
    treats as an effect size) -- 0.0 when the reply exactly matches baseline (so value(no ablation) == 0,
    the Shapley convention), up to 1.0 for a fully disjoint word-type set. `has_effect` is the strict
    string-inequality boolean core.py's receipts already use."""
    metrics = receipt_metrics(baseline_reply, reply)
    return {"reply": reply, "value": round(metrics["changed"] / 100.0, 6), "has_effect": reply != baseline_reply}


def _run_subset(run: dict, sub, subset_infs: list, baseline_ref, baseline_reply: str) -> dict | None:
    """One coalition arm for an ARBITRARY subset of influences (solo, pair, triple, or the full joint set)
    -- generalizes core._ablated_child (which already early-stops on divergence from the baseline) to any
    subset size, so pairs/triples/joint are computed by the exact same, already-correct machinery as
    core.py's own leave-one-out arms. Returns None when the subset has no representable ablation change,
    or the ablated arm could not be generated -- never fabricates a value for a failed arm."""
    changes = _merge_ablation_changes(subset_infs)
    if not changes:
        return None
    child = _ablated_child(run, changes, sub, baseline_ref, baseline_reply)
    if child is None:
        return None
    out = _arm_value(baseline_reply, child.get("response") or "")
    out["changes_applied"] = changes
    return out


# ==================================================================================== coalition selection

def _all_pairs(keys: list) -> list:
    return [frozenset(p) for p in combinations(sorted(keys), 2)]


def _top_k_pairs(keys: list, solo_values: dict, k: int) -> list:
    """The k pairs ranked by the SUM of their two members' |solo value| (a cheap, honest proxy for "this
    pair is worth checking jointly" -- an influence with a large solo effect is the one whose interactions
    are most worth measuring). Ties broken by sorted key names, so selection is deterministic."""
    pairs = _all_pairs(keys)
    ranked = sorted(
        pairs,
        key=lambda pair: (-sum(abs(solo_values.get(key, 0.0)) for key in pair), sorted(pair)),
    )
    return ranked[:max(0, k)]


def _missing_subsets_for_exact_shapley(keys: list, already: set) -> list:
    """Every non-empty, non-full subset of `keys` not already in `already` -- what's still needed to reach
    the full 2^N power set for an EXACT Shapley computation. Only ever called for N <= EXACT_SHAPLEY_MAX_N,
    where this is cheap (N=4's only gap is its 4 triples; N<=3 needs nothing extra -- solo+pairs+joint
    already IS the full power set through N=3)."""
    n = len(keys)
    missing = []
    for size in range(1, n):
        for combo in combinations(sorted(keys), size):
            fs = frozenset(combo)
            if fs not in already:
                missing.append(fs)
    return missing


# ======================================================================================= Shapley estimators

def exact_shapley(keys: list, values: dict) -> dict:
    """The exact Shapley value from a COMPLETE power-set value function. `values` must carry every
    non-empty subset of `keys` (frozenset -> float); the empty coalition is defined as 0.0 (the Shapley
    convention: no ablation, no change) and is never read from `values`. Raises KeyError (never silently
    defaults to 0) if any required subset is missing -- an incomplete power set must never masquerade as a
    real exact answer."""
    n = len(keys)
    phi = {}
    for i in keys:
        others = [k for k in keys if k != i]
        total = 0.0
        for r in range(0, n):
            weight = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
            for combo in combinations(others, r):
                s_fs = frozenset(combo)
                v_s = 0.0 if not s_fs else values[s_fs]
                v_si = values[frozenset(s_fs | {i})]
                total += weight * (v_si - v_s)
        phi[i] = total
    return phi


def shapley_taylor(keys: list, solo_values: dict, pair_values: dict) -> dict:
    """The Shapley-Taylor 2nd-order interaction index (Sundararajan, Dhamdhere & Agarwal 2020): each
    influence's solo delta, adjusted by the average pairwise interaction term (pair - solo_i - solo_j)
    split evenly between the two influences in every pair that was actually evaluated. Reduces to the
    EXACT Shapley value whenever every pairwise interaction was measured and there are no 3-way-or-higher
    effects (in particular, for N<=2 this is always exact). An influence with NO evaluated pair (e.g. it
    wasn't selected among the top-K for a large N) falls back to its bare solo delta, explicitly documented
    per-influence via `n_partners`."""
    phi, n_partners = {}, {}
    for i in keys:
        partners = [j for j in keys if j != i]
        terms = []
        for j in partners:
            pair = pair_values.get(frozenset((i, j)))
            if pair is None:
                continue
            terms.append(pair["value"] - solo_values[i] - solo_values[j])
        phi[i] = solo_values[i] + (sum(terms) / len(terms) / 2.0 if terms else 0.0)
        n_partners[i] = len(terms)
    return phi, n_partners


def interaction_gap(joint_value: float, solo_values: dict) -> dict:
    """joint (all influences ablated together) vs the naive sum of every solo delta -- the number the
    causal-tracer autopsy warns is characteristically overcounted. `ratio` is None (never a fabricated
    0/0) when every solo delta is exactly 0."""
    sum_solo = sum(solo_values.values())
    gap = joint_value - sum_solo
    ratio = round(gap / sum_solo, 6) if sum_solo else None
    return {"joint_value": round(joint_value, 6), "sum_solo": round(sum_solo, 6), "gap": round(gap, 6),
           "ratio": ratio, "note": OVERCOUNTING_CAVEAT}


# ==================================================================================================== batching

def _attempt_batched(run: dict, sub, work: list, baseline_reply: str, coalitions_batch: str):
    """Try to batch `work` (a list of (frozenset_key, influences_list) not yet run) through a substrate's
    OPTIONAL `branch_coalitions(run, changes_list, baseline_reply=...)` capability -- returns
    `(results, batch_report)`. `results` is a `{frozenset: arm_result|None}` dict ONLY when it's safe to use
    directly; otherwise it's None and the caller must run `_run_subset` sequentially for every item in
    `work` (the trusted, always-correct path -- see module docstring on why no real substrate offers this
    today).

    `coalitions_batch` (the repo's FP-landmine rule -- never trust batching without an equality check):
      * "off"          -- never attempt batching; always sequential.
      * "auto" (default) -- attempt it; a substrate that self-certifies via a truthy
                            `branch_coalitions_verified_exact` attribute is trusted directly (`class:
                            "exact"`). One that does NOT self-certify is cross-checked THIS call against
                            the sequential path (never trusted on its say-so alone): `results` is None
                            (sequential runs), and the batched replies are compared afterward by the
                            caller, which fills in `batch_report["bit_exact"]`.
      * "approximate"  -- the caller's explicit opt-in to use an uncertified substrate's batched output
                          directly (no sequential cross-check that call, a real perf win at the caller's
                          own stated risk) -- `class: "approximate"`, always labeled as such.
    """
    branch_fn = getattr(sub, "branch_coalitions", None)
    if coalitions_batch == "off" or not callable(branch_fn) or not work:
        return None, {"attempted": False, "used": False, "class": None,
                      "reason": "batching off, or substrate has no branch_coalitions" if work
                                else "nothing left to run"}
    changes_list = [_merge_ablation_changes(infs) for _, infs in work]
    try:
        raw_batch = branch_fn(run, changes_list, baseline_reply=baseline_reply)
    except Exception as exc:
        return None, {"attempted": True, "used": False, "class": None,
                      "reason": f"branch_coalitions raised {type(exc).__name__}: {exc}"}
    if not isinstance(raw_batch, list) or len(raw_batch) != len(work):
        return None, {"attempted": True, "used": False, "class": None,
                      "reason": "branch_coalitions returned a malformed batch (missing/extra arms)"}
    batched: dict = {}
    for (subset_key, _infs), raw in zip(work, raw_batch):
        reply = raw.get("reply") if isinstance(raw, dict) else None
        batched[subset_key] = _arm_value(baseline_reply, reply) if isinstance(reply, str) else None

    if bool(getattr(sub, "branch_coalitions_verified_exact", False)):
        return batched, {"attempted": True, "used": True, "class": "exact",
                         "reason": "substrate self-certifies its batched output as bit-exact to sequential"}
    if coalitions_batch == "approximate":
        return batched, {"attempted": True, "used": True, "class": "approximate",
                         "reason": "substrate is not self-certified exact; used at the caller's explicit "
                                   "--coalitions-batch=approximate opt-in"}
    # "auto" and not self-certified: never trust it un-verified -- hand the pending batched replies back so
    # the caller can run sequential (the truth) and compare, filling in bit_exact/mismatched_subsets.
    return None, {"attempted": True, "used": False, "class": None, "pending_compare": batched,
                  "reason": "substrate offers batching but is not self-certified exact; cross-checking "
                            "against sequential this call rather than trusting it silently"}


# =================================================================================================== orchestrator

def coalition_report(run: dict, sub, *, solo_results: dict, baseline_reply: str, baseline_ref=None,
                     coalitions_batch: str = "auto", top_k_pairs: int = DEFAULT_TOP_K_PAIRS,
                     bootstrap_resamples: int = 2000, bootstrap_seed: int = 0) -> dict:
    """The Phase-8-tail coalition/Shapley report for one run's already-fired influences.

    `solo_results` is `{influence_key: {"reply", "value", "has_effect"}}` -- the caller's OWN already-
    computed leave-one-out arms (core._prove_all_regen builds exactly this shape from its own receipts),
    reused here rather than regenerated a second time. Everything else (pairs, joint, and -- for N<=4 --
    the extra triples an exact Shapley needs) is computed fresh, batched through `sub.branch_coalitions`
    when available and not opted out (see `_attempt_batched`), else sequentially via `_run_subset`.

    Returns `{"available": False, "reason": ...}` for N==0 (nothing fired) or if the joint arm itself could
    not be generated (there is no honest interaction gap without it). Never raises.
    """
    keys = sorted(solo_results)
    n = len(keys)
    if n == 0:
        return {"available": False, "reason": "no fired influences to build coalitions from"}
    solo_values = {k: solo_results[k]["value"] for k in keys}
    influences_by_key = {k: solo_results[k]["_influence"] for k in keys if solo_results[k].get("_influence")}
    # Prefer explicit influence specs if the caller attached them; otherwise reconstruct minimally from the
    # key convention (`_key` in deltas.py: "section:<name>").
    def _spec_for(key: str) -> dict:
        if key in influences_by_key:
            return influences_by_key[key]
        kind, _, ident = key.partition(":")
        if kind == "section":
            return {"section": ident}
        return {}

    if n == 1:
        # A single influence: the "pair"/"joint"/triple machinery is vacuous -- solo IS the whole story.
        joint_value = solo_values[keys[0]]
        return {
            "available": True, "n_influences": 1, "keys": keys, "solo": solo_values,
            "pairs_evaluated": [], "pairs_capped": False, "k_pairs": 0,
            "joint": {"value": round(joint_value, 6)},
            "shapley": {"class": "exact", "values": dict(solo_values), "estimator_note":
                       "a single influence's Shapley value is trivially its own solo delta"},
            "interaction_gap": interaction_gap(joint_value, solo_values),
            "batch_report": {"attempted": False, "used": False, "class": None,
                            "reason": "nothing to batch for a single influence"},
            "cost_note": "1 solo arm only (reused); no pairs/joint arm needed beyond it.",
        }

    # ---- decide which pairs to run ----
    pairs_capped = n > EXHAUSTIVE_PAIRS_MAX_N
    pair_keys = _top_k_pairs(keys, solo_values, top_k_pairs) if pairs_capped else _all_pairs(keys)
    k_pairs = len(pair_keys)

    want_exact = n <= EXACT_SHAPLEY_MAX_N
    full_key = frozenset(keys)
    # A dict keyed by subset dedupes the N=2 degenerate case, where "the pair" and "the joint" are the
    # SAME 2-element subset -- never compute the identical coalition twice.
    work_map: dict = {pair: [_spec_for(k) for k in pair] for pair in pair_keys}
    work_map.setdefault(full_key, [_spec_for(k) for k in keys])   # the joint arm, always
    extra_subsets: list = []
    if want_exact:
        already = {frozenset({k}) for k in keys} | set(work_map)
        extra_subsets = [fs for fs in _missing_subsets_for_exact_shapley(keys, already) if fs not in work_map]
        for fs in extra_subsets:
            work_map[fs] = [_spec_for(k) for k in fs]
    work: list = list(work_map.items())

    results: dict = {}
    batched, batch_report = _attempt_batched(run, sub, work, baseline_reply, coalitions_batch)
    if batched is not None:
        results.update(batched)
        remaining = [(subset, infs) for subset, infs in work if results.get(subset) is None]
    else:
        remaining = work
    pending = batch_report.pop("pending_compare", None)
    for subset, infs in remaining:
        results[subset] = _run_subset(run, sub, infs, baseline_ref, baseline_reply)
    if pending is not None:
        mismatched = sorted(
            (subset for subset, seq in results.items()
             if subset in pending and pending[subset] is not None and seq is not None
             and pending[subset]["reply"] != seq["reply"]),
            key=lambda fs: sorted(fs),
        )
        batch_report["bit_exact"] = not mismatched
        if mismatched:
            batch_report["mismatched_subsets"] = [sorted(fs) for fs in mismatched]
            batch_report["reason"] += (
                f"; {len(mismatched)} arm(s) disagreed with the batched reply -- the sequential result is "
                "the one reported (never trust batching without an equality check)"
            )

    if results.get(full_key) is None:
        return {"available": False,
               "reason": "the joint (all-influences-ablated) arm could not be generated -- no honest "
                         "interaction gap without it"}

    pair_values = {pair: results[pair] for pair in pair_keys if results.get(pair) is not None}
    joint_value = results[full_key]["value"]

    if want_exact:
        power_set = {frozenset({k}): solo_values[k] for k in keys}
        power_set.update({pair: v["value"] for pair, v in pair_values.items()})
        power_set.update({fs: results[fs]["value"] for fs in extra_subsets if results.get(fs) is not None})
        power_set[full_key] = joint_value
        try:
            shapley_values = exact_shapley(keys, power_set)
            shapley = {"class": "exact", "values": {k: round(v, 6) for k, v in shapley_values.items()},
                      "estimator_note": f"exact Shapley over the full {2 ** n}-coalition power set (N={n} "
                                        f"<= {EXACT_SHAPLEY_MAX_N})."}
        except KeyError as exc:
            shapley = {"class": None, "values": {}, "estimator_note":
                      f"exact Shapley could not be computed: coalition {exc} was not generated"}
    else:
        taylor_values, n_partners = shapley_taylor(keys, solo_values, pair_values)
        per_influence_ci = {}
        for k in keys:
            terms = {j: pair_values[frozenset((k, j))]["value"] - solo_values[k] - solo_values[j]
                     for j in keys if j != k and frozenset((k, j)) in pair_values}
            per_influence_ci[k] = _bootstrap_interaction_ci(terms, bootstrap_resamples, bootstrap_seed)
        shapley = {
            "class": "shapley_taylor_2nd_order",
            "values": {k: round(v, 6) for k, v in taylor_values.items()},
            "n_partners": n_partners,
            "per_influence_interaction_ci": per_influence_ci,
            "estimator_note": (
                f"NOT exact -- N={n} exceeds the exact-Shapley threshold ({EXACT_SHAPLEY_MAX_N}). This is "
                f"the Shapley-Taylor 2nd-order interaction index (Sundararajan/Dhamdhere/Agarwal 2020), "
                f"built from {n} solo + {k_pairs} pairwise coalitions "
                f"({'all pairs' if not pairs_capped else f'top {k_pairs} of {n * (n - 1) // 2} by solo magnitude'})"
                " -- it ignores any 3-way-or-higher interaction."
            ),
        }

    return {
        "available": True, "n_influences": n, "keys": keys, "solo": solo_values,
        "pairs_evaluated": [sorted(p) for p in pair_keys], "pairs_capped": pairs_capped, "k_pairs": k_pairs,
        "joint": {"value": round(joint_value, 6)},
        "shapley": shapley,
        "interaction_gap": interaction_gap(joint_value, solo_values),
        "batch_report": {**batch_report, "note": _BATCH_NOTE if not batch_report.get("used") else None},
        "cost_note": (f"{n} solo (reused) + {k_pairs} pair + 1 joint"
                     f"{f' + {len(extra_subsets)} extra (exact Shapley)' if extra_subsets else ''} arm(s) "
                     "run for this report."),
    }


def _bootstrap_interaction_ci(terms: dict, n_resamples: int, seed: int) -> dict:
    """A bootstrap CI on one influence's mean pairwise interaction term, reusing
    clozn.experiments.stats.paired_bootstrap_ci (Phase 4.4's paired-resampling machinery) over that
    influence's own set of measured pair interactions -- honestly reflecting how much the Shapley-Taylor
    estimate might move if a different sample of pairs had been chosen. Unavailable (never fabricated) with
    fewer than 2 partner pairs, mirroring paired_bootstrap_ci's own contract."""
    from clozn.experiments import stats as _stats

    return _stats.paired_bootstrap_ci(terms, n_resamples=n_resamples, seed=seed)


# ======================================================================================================= format

def format_report(report: dict) -> str:
    """Pure text rendering (JSON in, text out) -- no I/O, mirrors clozn.cli.commands.explain's
    format_explain/format_narrate convention exactly, so it's testable with a canned dict."""
    if not report.get("available"):
        return f"coalition/Shapley credit: unavailable -- {report.get('reason', 'unknown reason')}"
    lines = [f"coalition/Shapley credit -- {report['n_influences']} influence(s): {', '.join(report['keys'])}",
            f"  solo (leave-one-out): " + ", ".join(f"{k}={v:+.4f}" for k, v in report["solo"].items())]
    lines.append(f"  pairs evaluated: {report['k_pairs']}"
                + (f" (top-{report['k_pairs']} of {report['n_influences'] * (report['n_influences'] - 1) // 2} "
                   "by solo magnitude -- capped)" if report["pairs_capped"] else " (all pairs)"))
    lines.append(f"  joint (all ablated together): {report['joint']['value']:+.4f}")
    gap = report["interaction_gap"]
    ratio_s = f"{gap['ratio']:+.1%}" if gap["ratio"] is not None else "n/a (zero sum-of-solos)"
    lines.append(f"  interaction gap: joint {gap['joint_value']:+.4f} vs sum-of-solos {gap['sum_solo']:+.4f}"
                f"  ->  gap {gap['gap']:+.4f}  ratio {ratio_s}")
    lines.append(f"    {gap['note']}")
    shap = report["shapley"]
    lines.append(f"  Shapley ({shap.get('class') or 'unavailable'}): "
                + ", ".join(f"{k}={v:+.4f}" for k, v in (shap.get("values") or {}).items()))
    lines.append(f"    {shap.get('estimator_note', '')}")
    batch = report.get("batch_report") or {}
    if batch.get("attempted"):
        lines.append(f"  batching: class={batch.get('class')}  bit_exact={batch.get('bit_exact')}  "
                    f"{batch.get('reason', '')}")
    else:
        lines.append(f"  batching: not attempted -- {batch.get('reason', '')}")
    lines.append(f"  {report['cost_note']}")
    return "\n".join(lines)
