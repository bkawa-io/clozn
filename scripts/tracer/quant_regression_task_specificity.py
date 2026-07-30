"""quant_regression_task_specificity.py -- research/task-specific-recurrence: does ANY restoring site
restore one CATEGORY of quantization regression but not another, beyond chance?

WHY THIS EXPERIMENT (docs/research/QUANT_REGRESSION_POPULATION.md's own open question)
----------------------------------------------------------------------------------------
That study's Site-overlap analysis section found restoring ffn sites recur strongly (layers 24/25/26
hold 44% of all hits, vs ~11% uniform) but the recurrence looked TASK-INDEPENDENT: layer 25 alone
restored arithmetic, structured_json, multilingual AND multi_step_reasoning disagreements alike, across
both Q2_K and Q4_K_M. That read as a QUANTIZER-GEOMETRY signature (rounding damage accumulates toward
the output end) rather than a CIRCUIT signature (a layer doing one job). But per-category n was only
4-5 there -- "too small to exclude task-specific structure" (that doc's own words, in its Limits
section: "What would flip this back toward circuits: task-SPECIFIC recurrence ... Not observed here,
but per-category n is too small to exclude. That needs more bisects per category, i.e. a new GPU run.").
This IS that run.

DESIGN: THREE MAXIMALLY-DIFFERENT CATEGORIES, ~14 DISAGREEMENTS EACH (NOT A NEW BISECT ENGINE)
----------------------------------------------------------------------------------------------------
`arithmetic`, `multilingual`, `structured_json` -- chosen per the experiment brief: structured_json is
the outlier on every axis this project has measured (lowest Phase-1 disagreement rate at 6.2% of
positions, highest Phase-2 localization rate at 62.5%), arithmetic and multilingual are both common,
high-disagreement-rate categories that nonetheless differ in kind (symbolic computation vs. token
identity/script). This script does NOT reimplement the bisect search -- it imports
`scripts/tracer/quant_regression_bisect.py` by path (that file has no `__init__.py`, so it is not an
importable package; loading it via `importlib.util`, exactly as `quant_mixed_precision_bands.py` already
does) and calls its `build_pool`, `run_one_disagreement`, and `characterize` UNCHANGED. The only new
selection logic here is `select_category_balanced_sample`, a round-robin across exactly the three target
categories capped at `--per-category-n` (default 14) each, deterministic (no RNG), analogous to
`quant_regression_bisect.select_stratified_sample` but restricted to a caller-chosen category subset
instead of all seven.

Both fixes this project's own retraction produced are already the DEFAULT behavior of
`clozn.analysis.causal_bisect.run_bisect()`, inherited automatically through `qrb.run_one_disagreement`
(never re-specified or reimplemented here):
  * `head_site_selection` defaults to `"stratified_divergence"` (GAP 1: depth-stratified head-site
    capping, closing the late-band-only coverage artifact the population study's Limits section flagged).
  * Every single-site confirmation's random-control seed is independently derived via SHA-256 over
    canonical JSON (`sha256_canonical_json_uint64_be_v1`, `_SINGLE_SITE_SEED_DERIVATION` in
    causal_bisect.py) keyed by `{base_seed, source, hook, layer, head?}` -- `seed=entry_seed` (this
    script's per-disagreement `seed_base + sample_index`, mirroring quant_regression_bisect.py's own
    fix) is only the BASE; no single frozen `random.Random(seed)` direction is ever reused across sites.

THE MEASUREMENT: SITE x CATEGORY MATRIX, TESTED AGAINST A PERMUTATION NULL, NOT PER-CELL p-VALUES
-------------------------------------------------------------------------------------------------------
For every (disagreement, hook) bisect that produced a `single_site_test` with `ok=True` and
`transplant.analysis.reference_specific=True` (the SAME "beat control at this exact site" criterion
docs/research/QUANT_REGRESSION_POPULATION.md's own Site-overlap analysis used -- window-level
`distributed_restoration`/`localized_window` verdicts are reported for context but deliberately excluded
from the site-attribution matrix, since they do not pin a single layer), record a "hit":
`(disagreement_id, category, hook, layer, head_or_none)`. Aggregate into `M[(hook, layer)][category]`.

With ~28 layers x 2 hooks x 3 categories, most cells are a lot of multiple-comparisons exposure if tested
one at a time -- exactly the trap the experiment brief warns about. This script does NOT run per-cell
significance tests. It runs exactly two GLOBAL permutation tests, each summarizing the WHOLE matrix (or
its most extreme cell) in one statistic, with the null built by shuffling CATEGORY LABELS ACROSS
DISAGREEMENT IDS (never across individual hits -- an id's hits move together under a shuffle, preserving
the real within-id correlation structure a naive per-hit shuffle would break) many times and asking how
often a statistic that extreme arises by relabeling alone:

  1. CHI2_SUM (global structure): for every site with >=2 total hits, a per-site Pearson chi-square
     deviation from the observed GLOBAL category-hit-share, summed across qualifying sites. Large under
     the alternative "categories are non-uniformly distributed across sites, taken as a whole"; the
     permutation null already accounts for however many sites happen to qualify and however many hits
     each carries, since site-hit MEMBERSHIP (which ids hit where) is fixed and only category LABELS are
     shuffled.
  2. MAX_PURITY (single most category-specific site): for every site with >=3 total hits (n=2 sites are
     trivially 50%+ pure by construction and would dominate a max-statistic net of nothing but noise), the
     fraction belonging to its single most common category; the observed MAXIMUM across sites is the test
     statistic. This is the exact multiple-comparisons-correct way to ask "is there at least one layer
     that looks task-specific" -- taking the max under permutation automatically controls the family-wise
     error rate across every site searched, which a per-site p-value never does (see the experiment
     brief's "Multiple comparisons are the trap").

For comparison ONLY (never used as the actual decision rule), a naive Bonferroni bound
(alpha=0.05 / n_sites_tested) is also reported alongside -- to show explicitly how much more conservative
(and how much less informative, since it ignores the max-statistic's exact permutation distribution) that
approach would have been.

POWER: A MINIMUM-DETECTABLE-EFFECT SEARCH AGAINST THE REAL OBSERVED BACKGROUND
------------------------------------------------------------------------------------
A generic power formula would be dishonest here -- the real "noise" is this run's own messy multi-hit,
multi-category background, not a clean binomial. Instead: starting from the run's OWN observed hit
pattern (every real id and every real hit, unchanged), inject a SYNTHETIC, previously-unused site that is
hit by exactly `k` disagreements, ALL from one category (perfect, maximal specificity for that k) and
NONE from the others. For increasing `k` (1, 2, 3, ... up to that category's own `--per-category-n`),
rerun the SAME max-purity permutation test (recomputing the null fresh each time, since adding synthetic
hits changes which sites qualify) and report the smallest `k` at which p < 0.05. That `k` (and `k /
per_category_n` as a normalized effect size) is the honest answer to "what could this experiment's design
and this run's actual background noise have detected" -- not a textbook number, a number this exact
dataset and this exact test would have needed to see.

CHECKPOINTING (same discipline as quant_regression_bisect.py, its own EVERY-disagreement pattern)
---------------------------------------------------------------------------------------------------
`runs/experiments/_quant_task_specificity_checkpoint.json` accumulates one completed disagreement record
(both hooks' full bisect documents) at a time, atomic write-then-replace after EACH ONE -- not batched.
Re-running this script skips any selected disagreement id already present in the checkpoint. The
permutation test / MDE search are pure-CPU, seconds-scale, and re-run fresh on every invocation (never
checkpointed -- there is nothing expensive to save there).

OUTPUT
--------
runs/experiments/quant_task_specificity.json -- `clozn.quant_task_specificity.v1`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "engine", "client"))

# scripts/tracer has no __init__.py -- loaded by path so this script reuses
# quant_regression_bisect.py's pool-building / per-disagreement bisect / characterize functions
# VERBATIM rather than re-deriving them (same pattern quant_mixed_precision_bands.py already uses).
_qrb_spec = importlib.util.spec_from_file_location(
    "quant_regression_bisect", os.path.join(REPO, "scripts", "tracer", "quant_regression_bisect.py"))
qrb = importlib.util.module_from_spec(_qrb_spec)
_qrb_spec.loader.exec_module(qrb)

MINE_REPORT_PATH = qrb.MINE_REPORT_PATH
OUT_PATH = os.path.join(REPO, "runs", "experiments", "quant_task_specificity.json")
CHECKPOINT_PATH = os.path.join(REPO, "runs", "experiments", "_quant_task_specificity_checkpoint.json")

TARGET_CATEGORIES = ("arithmetic", "multilingual", "structured_json")
DEFAULT_PER_CATEGORY_N = 14
DEFAULT_SEED_BASE = 5001  # distinct from quant_regression_bisect.py's default (1) and
                          # quant_mixed_precision_bands.py's default (9001) -- visibly unrelated state.
DEFAULT_ALPHA = 0.05
DEFAULT_N_PERM = 100_000


def _atomic_write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ================================================================================ category-balanced sample

def select_category_balanced_sample(pool: dict, categories: "tuple[str, ...]", per_category_n: int) -> list:
    """Deterministic round-robin across EXACTLY `categories` (never the full CATEGORY_ORDER), each capped
    at `per_category_n` (or the category's own pool size, if smaller -- never padded, never silently
    widened). No RNG: re-running with the same mine report, categories, and per_category_n reproduces the
    identical sample. Mirrors quant_regression_bisect.select_stratified_sample's own round-robin discipline,
    restricted to a caller-chosen category subset with a PER-CATEGORY (not global) cap."""
    order = [c for c in categories if pool.get(c)]
    queues = {c: list(pool[c]) for c in order}
    counts = {c: 0 for c in order}
    selected = []
    i = 0
    while any(counts[c] < per_category_n and queues[c] for c in order):
        c = order[i % len(order)]
        i += 1
        if counts[c] < per_category_n and queues[c]:
            selected.append(queues[c].pop(0))
            counts[c] += 1
    return selected


# =================================================================================== site-hit extraction

def extract_site_hits(records: list) -> list:
    """One entry per (disagreement, hook) single_site_test that came back ok AND
    transplant.analysis.reference_specific=True -- the SAME per-site "beat control at this exact site"
    criterion docs/research/QUANT_REGRESSION_POPULATION.md's Site-overlap analysis section used. Window-
    level-only verdicts (distributed_restoration, localized_window with no single confirmed site) are
    deliberately NOT represented here -- they do not pin one layer, so they cannot honestly enter a
    site x category matrix; `characterization` (from qrb.characterize) still reports their counts."""
    hits = []
    for rec in records:
        for hook in ("ffn", "head"):
            res = rec.get(hook)
            if not res or not res.get("ok"):
                continue
            doc = res["document"]
            for s in doc.get("single_site_tests", []):
                if not s.get("ok"):
                    continue
                a = (s.get("transplant") or {}).get("analysis", {})
                if a.get("reference_specific"):
                    hits.append({
                        "id": rec["id"], "category": rec["category"], "candidate": rec["candidate"],
                        "hook": s["hook"], "layer": s.get("layer"), "head": s.get("head"),
                        "source": s.get("source"),
                    })
    return hits


def _site_key(hit: dict) -> tuple:
    return (hit["hook"], hit["layer"])  # pooled across head index -- matches the population study's own
                                        # layer-level granularity; raw (hook, layer, head) stays in hits.


def build_matrix(hits: list, categories: "tuple[str, ...]") -> dict:
    m: dict = defaultdict(lambda: {c: 0 for c in categories})
    for h in hits:
        m[_site_key(h)][h["category"]] += 1
    return {f"{k[0]}:L{k[1]}": v for k, v in sorted(m.items())}


# ==================================================================== permutation test (id-level shuffle)

def _id_category_map(hits: list) -> dict:
    m = {}
    for h in hits:
        m[h["id"]] = h["category"]
    return m


def _id_sites_map(hits: list) -> dict:
    m = defaultdict(list)
    for h in hits:
        m[h["id"]].append(_site_key(h))
    return dict(m)


def _matrix_from_labeling(id_sites: dict, id_labels: dict, categories: "tuple[str, ...]") -> dict:
    m: dict = defaultdict(lambda: {c: 0 for c in categories})
    for hid, sites in id_sites.items():
        cat = id_labels[hid]
        for site in sites:
            m[site][cat] += 1
    return m


def _chi2_sum_stat(matrix: dict, categories: "tuple[str, ...]", min_hits: int = 2) -> "tuple[float, int]":
    totals = {c: 0 for c in categories}
    grand = 0
    for counts in matrix.values():
        for c in categories:
            totals[c] += counts[c]
            grand += counts[c]
    if grand == 0:
        return 0.0, 0
    shares = {c: totals[c] / grand for c in categories}
    stat = 0.0
    n_qualifying = 0
    for counts in matrix.values():
        n_s = sum(counts[c] for c in categories)
        if n_s < min_hits:
            continue
        n_qualifying += 1
        for c in categories:
            expected = n_s * shares[c]
            if expected > 0:
                stat += (counts[c] - expected) ** 2 / expected
    return stat, n_qualifying


def _max_purity_stat(matrix: dict, categories: "tuple[str, ...]", min_hits: int = 3) -> "tuple[float, str, int]":
    best = 0.0
    best_site = None
    n_qualifying = 0
    for site, counts in matrix.items():
        n_s = sum(counts[c] for c in categories)
        if n_s < min_hits:
            continue
        n_qualifying += 1
        purity = max(counts[c] for c in categories) / n_s
        if purity > best:
            best = purity
            best_site = site
    return best, best_site, n_qualifying


def run_permutation_test(hits: list, sample: list, categories: "tuple[str, ...]", n_perm: int, seed: int,
                         chi2_min_hits: int = 2, purity_min_hits: int = 3) -> dict:
    """`sample` (every disagreement actually bisected, `{"id", "category", ...}`) is the exchangeability
    unit: category labels are shuffled across the FULL sample (its true per-category counts, e.g. 14/14/14
    here), never just the subset of ids that happened to produce a hit. An id with zero hits still occupies
    a slot in the true marginal and must remain eligible to receive any label under the null.

    A REAL BUG THIS FUNCTION HAD, FOUND AND FIXED BEFORE ANY RESULT WAS REPORTED: an earlier version built
    the label pool from `_id_category_map(hits)` -- i.e. only the ids that already had >=1 hit. On this
    run that was 17 of 42 sampled ids, with an observed category split (arithmetic 9 / multilingual 5 /
    structured_json 3) already skewed far from the true 14/14/14 sample. Shuffling within that skewed
    17-id pool builds a null that has quietly baked the very category-level hit-rate imbalance the test
    exists to interrogate into its own reference distribution -- and it starves any injected-effect check
    of ids to draw on (this is what made `mde_search`'s minimum-detectable-effect search return `None`
    even at k=14, the theoretically maximal, perfectly-pure injection: exposed by is own internal sanity
    check, not by a difference in the final headline p-values, which happened to still read as
    "no structure" either way here -- but a p-value that reads right for the wrong reason is exactly the
    kind of thing this project's own retraction history says to distrust, not credit)."""
    id_labels_true = {e["id"]: e["category"] for e in sample}
    id_sites = _id_sites_map(hits)  # only hit-bearing ids present; an id absent here simply owns no site
    ids = sorted(id_labels_true.keys())  # deterministic ordering, independent of dict/hit insertion order
    label_pool = [id_labels_true[i] for i in ids]  # multiset to shuffle, preserves TRUE category counts

    obs_matrix = _matrix_from_labeling(id_sites, id_labels_true, categories)
    chi2_obs, n_chi2_sites = _chi2_sum_stat(obs_matrix, categories, chi2_min_hits)
    purity_obs, purity_site_obs, n_purity_sites = _max_purity_stat(obs_matrix, categories, purity_min_hits)

    rng = random.Random(seed)
    chi2_ge = 0
    purity_ge = 0
    chi2_null_sample = []
    purity_null_sample = []
    for p in range(n_perm):
        shuffled = list(label_pool)
        rng.shuffle(shuffled)
        id_labels_perm = dict(zip(ids, shuffled))
        mtx = _matrix_from_labeling(id_sites, id_labels_perm, categories)
        c2, _ = _chi2_sum_stat(mtx, categories, chi2_min_hits)
        pu, _, _ = _max_purity_stat(mtx, categories, purity_min_hits)
        if c2 >= chi2_obs:
            chi2_ge += 1
        if pu >= purity_obs:
            purity_ge += 1
        if p < 2000:  # keep a bounded sample of the null for the artifact (never the full 100k -- that
                      # would bloat the JSON for no analytical benefit), never used for the p-value itself
            chi2_null_sample.append(round(c2, 4))
            purity_null_sample.append(round(pu, 4))

    # standard permutation-test p-value with the +1 correction (Davison & Hinkley, 1997) -- avoids ever
    # reporting p=0.0 from a finite number of permutations, which would misstate the test's own resolution.
    chi2_p = (chi2_ge + 1) / (n_perm + 1)
    purity_p = (purity_ge + 1) / (n_perm + 1)

    naive_bonferroni_alpha = (DEFAULT_ALPHA / n_purity_sites) if n_purity_sites else None
    n_ids_with_any_hit = len({h["id"] for h in hits})

    return {
        "n_ids_in_sample": len(ids),  # the permutation label pool -- the FULL sample, not just hit-bearing
                                      # ids (see this function's own docstring on why that fix mattered)
        "n_ids_with_any_hit": n_ids_with_any_hit, "n_hits_total": len(hits),
        "chi2_sum": {
            "description": "global structure test: sum of per-site chi-square deviations from the "
                           "observed overall category share, over every site with >= "
                           f"{chi2_min_hits} total hits.",
            "min_hits_per_site": chi2_min_hits, "n_qualifying_sites": n_chi2_sites,
            "observed_statistic": round(chi2_obs, 4), "n_permutations": n_perm,
            "n_permutations_at_least_as_extreme": chi2_ge, "p_value": round(chi2_p, 6),
            "null_sample_first_2000": chi2_null_sample,
        },
        "max_purity": {
            "description": "single most category-specific site test (multiple-comparisons-correct via "
                           "the max statistic itself): the largest, over every site with >= "
                           f"{purity_min_hits} total hits, of (most-common-category count / total hits "
                           "at that site).",
            "min_hits_per_site": purity_min_hits, "n_qualifying_sites": n_purity_sites,
            "observed_statistic": round(purity_obs, 4), "observed_site": purity_site_obs,
            "observed_site_breakdown": obs_matrix.get(purity_site_obs) if purity_site_obs else None,
            "n_permutations": n_perm, "n_permutations_at_least_as_extreme": purity_ge,
            "p_value": round(purity_p, 6),
            "null_sample_first_2000": purity_null_sample,
        },
        "naive_bonferroni_context": {
            "description": "NOT the decision rule used above -- reported only to show how much more "
                           "conservative (and how much less informative, since it ignores the exact "
                           "permutation distribution of the max statistic) a naive per-site-tested "
                           "Bonferroni correction would have been.",
            "n_sites_that_would_need_correcting": n_purity_sites,
            "bonferroni_alpha": round(naive_bonferroni_alpha, 6) if naive_bonferroni_alpha else None,
        },
        "observed_matrix": {f"{k[0]}:L{k[1]}": v for k, v in
                            sorted(obs_matrix.items(), key=lambda kv: (kv[0][0], kv[0][1]))},
    }


# ============================================================================== minimum detectable effect

def mde_search(hits: list, sample: list, categories: "tuple[str, ...]", per_category_n: int, n_perm: int,
               seed: int, alpha: float = DEFAULT_ALPHA, purity_min_hits: int = 3,
               focus_category: "str | None" = None) -> dict:
    """Inject a synthetic, previously-unused site hit EXCLUSIVELY by k disagreements of one category (all
    real ids/hits left exactly as observed), rerun the max-purity permutation test fresh for each k, and
    report the smallest k at which p < alpha. Answers "what effect size could this run's own background
    noise, at n=per_category_n per category, have let us detect" -- not a textbook power number.

    `focus_ids_unique` and the shuffled label pool are BOTH drawn from the FULL SAMPLE (every id actually
    bisected, `sample`, with its TRUE category -- e.g. the real 14/14/14 marginal here), never merely the
    subset of ids that happened to produce a hit elsewhere. An id with zero real hits is a perfectly valid,
    arguably the CLEANEST, injection target: `id_sites` (built from `hits` only) simply has no entry for
    it yet, and `.setdefault(i, [])` below adds one.

    A REAL BUG THIS FUNCTION HAD, FOUND AND FIXED BEFORE ANY RESULT WAS REPORTED (see
    `run_permutation_test`'s own matching docstring note for the full account): an earlier version built
    BOTH the injection candidate pool AND the permutation label pool from `_id_category_map(hits)` --
    ids that already had >=1 hit. Restricting the candidate pool that way silently caps how many ids of a
    thin-hit category are even available to inject into; restricting the PERMUTATION pool that way is
    worse -- it builds a null out of whatever skewed category composition the hit-bearing ids happen to
    have (here: arithmetic 9 / multilingual 5 / structured_json 3, nothing like the true 14/14/14), which
    can make even a maximally strong, perfectly pure synthetic effect at k=per_category_n fail to reach
    significance for a reason that has nothing to do with statistical power -- exactly what this run's
    first pass produced (`k_min_detected: null` even at k=14) before this fix."""
    id_sites = _id_sites_map(hits)  # only hit-bearing ids present; an id absent here simply owns no site
    id_labels_true_full = {e["id"]: e["category"] for e in sample}
    all_ids = sorted(id_labels_true_full.keys())
    label_pool_full = [id_labels_true_full[i] for i in all_ids]  # TRUE per-category counts, e.g. 14/14/14
    ids_by_category = {c: sorted({e["id"] for e in sample if e["category"] == c}) for c in categories}
    if focus_category is None:
        # conservative choice: the category with the FEWEST hit-bearing ids in the real data (ties broken
        # by category name for determinism), so the MDE is not flattered by picking whichever category
        # already has the most signal. Every target category is considered even if it produced zero
        # hits (defaultdict-style: Counter lookups on a missing key return 0, never a KeyError/omission).
        hit_counts = Counter(_id_category_map(hits).values())
        focus_category = min(categories, key=lambda c: (hit_counts.get(c, 0), c))

    focus_ids_unique = ids_by_category.get(focus_category, [])
    max_k = min(per_category_n, len(focus_ids_unique)) if focus_ids_unique else 0

    trace = []
    k_min = None
    rng = random.Random(seed)
    for k in range(1, max_k + 1):
        synthetic_site = ("ffn", -1000 - k)  # layer -1000-k can never collide with a real ffn/head layer
        aug_id_sites = {i: list(s) for i, s in id_sites.items()}
        for i in focus_ids_unique[:k]:
            aug_id_sites.setdefault(i, []).append(synthetic_site)
        # labels never change here: every injected id's TRUE category is already `focus_category` (drawn
        # from ids_by_category[focus_category], itself derived from the true sample) -- only its site
        # membership gains the synthetic entry.

        obs_matrix = _matrix_from_labeling(aug_id_sites, id_labels_true_full, categories)
        purity_obs, purity_site, n_qual = _max_purity_stat(obs_matrix, categories, purity_min_hits)

        ge = 0
        for _ in range(n_perm):
            shuffled = list(label_pool_full)
            rng.shuffle(shuffled)
            perm_labels = dict(zip(all_ids, shuffled))
            mtx = _matrix_from_labeling(aug_id_sites, perm_labels, categories)
            pu, _, _ = _max_purity_stat(mtx, categories, purity_min_hits)
            if pu >= purity_obs:
                ge += 1
        p = (ge + 1) / (n_perm + 1)
        trace.append({"k": k, "k_fraction_of_per_category_n": round(k / per_category_n, 4),
                      "observed_max_purity": round(purity_obs, 4), "p_value": round(p, 6)})
        if k_min is None and p < alpha:
            k_min = k

    return {
        "description": "smallest k (injected, category-EXCLUSIVE synthetic site hits, real background "
                       "otherwise unchanged) at which the max-purity permutation test rejects "
                       f"task-independence at alpha={alpha}.",
        "focus_category": focus_category,
        "focus_category_n_ids_available_in_sample": len(focus_ids_unique),
        "per_category_n": per_category_n, "alpha": alpha, "n_perm_per_k": n_perm,
        "k_min_detected": k_min,
        "k_min_fraction_of_per_category_n": round(k_min / per_category_n, 4) if k_min else None,
        "note_if_none": (None if k_min is not None else
                         "no k up to min(per_category_n, ids available) reached p<alpha -- the run "
                         "either lacked ids to inject into, or even a fully category-exclusive "
                         "synthetic site at the maximum testable k was not significant; see trace."),
        "trace": trace,
    }


# ======================================================================================================= cli

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mine-report", default=MINE_REPORT_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--categories", nargs="+", default=list(TARGET_CATEGORIES))
    ap.add_argument("--per-category-n", type=int, default=DEFAULT_PER_CATEGORY_N)
    ap.add_argument("--ffn-window-size", type=int, default=None, help="default: max(4, n_layer//4)")
    ap.add_argument("--ffn-max-windows", type=int, default=4)
    ap.add_argument("--head-window-size", type=int, default=4)
    ap.add_argument("--head-max-windows", type=int, default=4)
    ap.add_argument("--max-head-sites", type=int, default=16)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED_BASE)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=DEFAULT_N_PERM)
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--chi2-min-hits", type=int, default=2)
    ap.add_argument("--purity-min-hits", type=int, default=5,
                    help="min total hits a site needs to enter the max-purity statistic. Default 5, NOT "
                        "3, and this default matters: with 3 categories at a true 14/14/14 marginal, a "
                        "site with only 3 hits has ~9.5%% chance of looking perfectly category-pure by "
                        "PURE COMBINATORIAL LUCK alone (comb(14,3)/comb(42,3) x 3 categories) -- with "
                        "several such small sites in play, that noise floor is high enough that even a "
                        "maximal, perfectly-pure INJECTED effect at min_hits=3 never reaches p<0.05 on "
                        "this run's own background (see minimum_detectable_effect / the "
                        "purity_min_hits_sensitivity_sweep in the output report). At min_hits=5 the "
                        "by-chance rate drops to ~0.7%%/site and the test has real, demonstrated power. "
                        "--purity-min-hits only sets the PRIMARY reported value; 3/4/5 are always all "
                        "computed and reported in purity_min_hits_sensitivity_sweep regardless.")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test knob: only bisect the first N selected disagreements")
    ap.add_argument("--select-only", action="store_true",
                    help="print the category-balanced sample and exit -- no engine calls")
    ap.add_argument("--analyze-only", action="store_true",
                    help="skip bisecting -- load the checkpoint (must be complete for the requested "
                        "sample) and (re)run only the matrix/permutation/MDE analysis + report write")
    args = ap.parse_args()
    categories = tuple(args.categories)

    with open(args.mine_report, encoding="utf-8") as f:
        mine_report = json.load(f)

    pool = qrb.build_pool(mine_report)
    pool_sizes = {c: len(v) for c, v in pool.items()}
    print("pool sizes (all categories):", pool_sizes, flush=True)
    print("pool sizes (target categories):", {c: pool_sizes.get(c, 0) for c in categories}, flush=True)
    sample = select_category_balanced_sample(pool, categories, args.per_category_n)
    if args.limit is not None:
        sample = sample[:args.limit]
    print(f"selected {len(sample)} disagreements (target {args.per_category_n}/category x "
          f"{len(categories)} categories = {args.per_category_n * len(categories)}):", flush=True)
    cat_counts = Counter(e["category"] for e in sample)
    cand_counts = Counter(e["candidate"] for e in sample)
    print(f"  by category: {dict(cat_counts)}", flush=True)
    print(f"  by candidate: {dict(cand_counts)}", flush=True)
    for e in sample:
        print(f"    [{e['id']}] {e['category']:<14} {e['candidate']:<7} pos={e['first_disagree_index']} "
              f"{e['prompt'][:50]!r} ref={e['target_piece']!r} cand={e['candidate_top1_piece']!r}",
              flush=True)
    if args.select_only:
        return 0

    checkpoint = {}
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                checkpoint = json.load(f)
        except Exception:
            checkpoint = {}
    completed = checkpoint.get("completed", {})
    if completed:
        print(f"resuming: {len(completed)} disagreements already bisected in checkpoint", flush=True)

    bounds = {
        "target_categories": list(categories),
        "per_category_n_target": args.per_category_n,
        "per_category_n_actual": dict(cat_counts),
        "sample_size_actual": len(sample),
        "pool_sizes_target_categories": {c: pool_sizes.get(c, 0) for c in categories},
        "ffn_window_size": args.ffn_window_size or "max(4, n_layer // 4), per candidate",
        "ffn_max_windows": args.ffn_max_windows, "head_window_size": args.head_window_size,
        "head_max_windows": args.head_max_windows, "max_head_sites": args.max_head_sites,
        "head_site_selection": "stratified_divergence (causal_bisect.run_bisect's default -- depth-"
                               "banded head-site cap, closes the late-band-only coverage gap the "
                               "population study's Limits section flagged; never overridden here)",
        "seed_base": args.seed,
        "seed_scheme": "seed = seed_base + sample_index, VARIED per disagreement (matches "
                       "quant_regression_bisect.py's fix for the retracted seed confound); every "
                       "single-site leaf within a disagreement further derives its OWN independent seed "
                       "via causal_bisect's sha256_canonical_json_uint64_be_v1 strategy (module default, "
                       "not re-specified here).",
        "topk": args.topk, "store_tensors": False,
        "write_target": "first disagreement position only, per prompt",
        "permutation_test": {"n_perm": args.n_perm, "alpha": args.alpha,
                             "chi2_min_hits_per_site": args.chi2_min_hits,
                             "purity_min_hits_per_site": args.purity_min_hits},
    }

    records = []
    if not args.analyze_only:
        dims_by_candidate: dict = {}
        for sample_index, entry in enumerate(sample):
            eid = entry["id"]
            if eid in completed:
                records.append(completed[eid])
                continue
            cand_path = qrb.CANDIDATES[entry["candidate"]]
            if entry["candidate"] not in dims_by_candidate:
                dims_by_candidate[entry["candidate"]] = qrb._pair_and_dims(cand_path)
            dims = dims_by_candidate[entry["candidate"]]

            entry_seed = args.seed + sample_index
            t0 = time.monotonic()
            print(f"\n[bisect {eid}] category={entry['category']} candidate={entry['candidate']} "
                  f"seed={entry_seed} prompt={entry['prompt'][:60]!r}", flush=True)
            out = qrb.run_one_disagreement(
                entry, cand_path, dims, ffn_window_size=args.ffn_window_size,
                ffn_max_windows=args.ffn_max_windows, head_window_size=args.head_window_size,
                head_max_windows=args.head_max_windows, max_head_sites=args.max_head_sites,
                seed=entry_seed, topk=args.topk)
            out["seed"] = entry_seed
            elapsed = time.monotonic() - t0

            ffn_label = (out["ffn"]["document"]["verdict"]["label"] if out["ffn"].get("ok")
                        else f"ERROR: {out['ffn'].get('error')}")
            head_label = (out["head"]["document"]["verdict"]["label"] if out["head"].get("ok")
                         else f"ERROR: {out['head'].get('error')}")
            print(f"[bisect {eid}] ffn={ffn_label!r} head={head_label!r} ({elapsed:.1f}s)", flush=True)

            record = dict(entry)
            record.update(out)
            records.append(record)
            completed[eid] = record
            _atomic_write(CHECKPOINT_PATH, {"completed": completed})
    else:
        missing = [e["id"] for e in sample if e["id"] not in completed]
        if missing:
            print(f"--analyze-only requested but {len(missing)} disagreements missing from checkpoint: "
                  f"{missing[:10]}{'...' if len(missing) > 10 else ''}", flush=True)
            return 1
        records = [completed[e["id"]] for e in sample]

    characterization = qrb.characterize(records)
    hits = extract_site_hits(records)
    matrix = build_matrix(hits, categories)
    print(f"\n{len(hits)} site-level reference_specific hits extracted across {len(records)} records "
          f"(hook x layer keys: {len(matrix)})", flush=True)

    perm_t0 = time.monotonic()
    permutation = run_permutation_test(hits, sample, categories, args.n_perm, seed=args.seed,
                                       chi2_min_hits=args.chi2_min_hits,
                                       purity_min_hits=args.purity_min_hits)
    print(f"permutation test ({args.n_perm} perms x 2 statistics) in "
          f"{time.monotonic() - perm_t0:.1f}s", flush=True)

    mde_t0 = time.monotonic()
    mde = mde_search(hits, sample, categories, args.per_category_n, n_perm=max(2000, args.n_perm // 10),
                     seed=args.seed + 777, alpha=args.alpha, purity_min_hits=args.purity_min_hits)
    print(f"MDE search in {time.monotonic() - mde_t0:.1f}s", flush=True)

    # SENSITIVITY SWEEP over purity_min_hits -- added after an audit finding on this run's own data (not
    # a hypothetical): with 3 categories at the true 14/14/14 marginal, a qualifying site of exactly
    # `min_hits=3` has ~9.5% chance of looking PERFECTLY category-pure by pure combinatorial luck alone
    # (comb(14,3)/comb(42,3), times 3 categories -- see `purity_chance_by_pure_luck` below, computed the
    # same way this run's own audit computed it live before trusting the k=14 MDE result). With several
    # such small sites in the qualifying set, the CUMULATIVE chance that at least one of them coincidentally
    # matches or beats even a maximal, perfectly-pure injected effect is substantial -- which is exactly
    # why the min_hits=3 max-purity test alone should never be read in isolation. This sweep reruns BOTH
    # the observed max-purity test and the MDE search at min_hits in {3, 4, 5} so the read does not depend
    # on one arbitrarily chosen threshold, and reports the exact by-chance purity probability at each `n_s`
    # alongside it for direct comparison.
    def _purity_chance_by_luck(n_s: int, per_cat: int = args.per_category_n, n_total: "int | None" = None) -> float:
        n_total = n_total if n_total is not None else per_cat * len(categories)
        if n_s > per_cat:
            return None
        from math import comb
        return round(len(categories) * comb(per_cat, n_s) / comb(n_total, n_s), 6)

    sweep_t0 = time.monotonic()
    purity_sweep = {}
    for pmh in (3, 4, 5):
        perm_pmh = run_permutation_test(hits, sample, categories, args.n_perm, seed=args.seed,
                                        chi2_min_hits=args.chi2_min_hits, purity_min_hits=pmh)
        mde_pmh = mde_search(hits, sample, categories, args.per_category_n,
                             n_perm=max(2000, args.n_perm // 10), seed=args.seed + 777,
                             alpha=args.alpha, purity_min_hits=pmh)
        purity_sweep[str(pmh)] = {
            "min_hits": pmh,
            "purity_chance_by_pure_luck_at_this_n_s": _purity_chance_by_luck(pmh),
            "max_purity_observed": perm_pmh["max_purity"]["observed_statistic"],
            "max_purity_observed_site": perm_pmh["max_purity"]["observed_site"],
            "max_purity_n_qualifying_sites": perm_pmh["max_purity"]["n_qualifying_sites"],
            "max_purity_p_value": perm_pmh["max_purity"]["p_value"],
            "mde_k_min_detected": mde_pmh["k_min_detected"],
            "mde_k_min_fraction": mde_pmh["k_min_fraction_of_per_category_n"],
            "mde_focus_category": mde_pmh["focus_category"],
            "mde_trace": mde_pmh["trace"],  # full k=1..per_category_n p-value curve, not just the
                                            # summary -- self-contained enough to plot without rerunning.
        }
    print(f"purity_min_hits sensitivity sweep (3 thresholds) in {time.monotonic() - sweep_t0:.1f}s",
          flush=True)

    report = {
        "schema": "clozn.quant_task_specificity.v1",
        "generated_at": _now_iso(),
        "reference_model": qrb.os.path.basename(qrb.REFERENCE_MODEL),
        "candidates": {k: os.path.basename(v) for k, v in qrb.CANDIDATES.items()},
        "mine_report_used": os.path.relpath(args.mine_report, REPO),
        "pool_sizes_all_categories": pool_sizes,
        "bounds": bounds,
        "sample": [{"id": e["id"], "candidate": e["candidate"], "category": e["category"],
                    "prompt": e["prompt"], "first_disagree_index": e["first_disagree_index"],
                    "target_piece": e["target_piece"], "candidate_top1_piece": e["candidate_top1_piece"]}
                   for e in sample],
        "characterization": characterization,
        "site_by_category_matrix": matrix,
        "site_hits_raw": hits,
        "permutation_test": permutation,
        "minimum_detectable_effect": mde,
        "purity_min_hits_sensitivity_sweep": purity_sweep,
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _atomic_write(args.out, report)
    print(f"\nwrote {args.out}", flush=True)

    print("\n=== verdict histogram (overall) ===")
    for label, n in sorted(characterization["verdict_histogram_overall"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:<28} {n}")
    print("\n=== verdict histogram by hook x category ===")
    print(json.dumps(characterization["verdict_histogram_by_hook_and_category"], indent=2))
    print("\n=== instrument sanity ===")
    print(f"  {characterization['instrument_sanity']}")
    print("\n=== site x category matrix ===")
    for site, counts in matrix.items():
        print(f"  {site:<10} {counts}")
    print("\n=== permutation test ===")
    print(json.dumps({"chi2_sum": {k: v for k, v in permutation["chi2_sum"].items()
                                   if k != "null_sample_first_2000"},
                      "max_purity": {k: v for k, v in permutation["max_purity"].items()
                                    if k != "null_sample_first_2000"},
                      "naive_bonferroni_context": permutation["naive_bonferroni_context"]}, indent=2))
    print("\n=== minimum detectable effect ===")
    print(json.dumps({k: v for k, v in mde.items() if k != "trace"}, indent=2))
    print("\n=== purity_min_hits sensitivity sweep ===")
    print(json.dumps(purity_sweep, indent=2))

    if not args.analyze_only:
        # --analyze-only intentionally leaves the checkpoint in place: it exists precisely so the
        # (expensive, GPU-bound) bisect records can be reanalyzed repeatedly -- e.g. after fixing an
        # analysis bug, as happened on this run -- without ever re-running the engine.
        try:
            os.remove(CHECKPOINT_PATH)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
