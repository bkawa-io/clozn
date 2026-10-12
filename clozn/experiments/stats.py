"""Statistical rigor layer over :mod:`clozn.experiments.suite` case × variant × seed artifacts
(``clozn.experiment.result.v0``). This module never re-derives suite.py's own validation or per-cell
status; it only aggregates and analyzes an already-validated result (``suite.validate_result``).

What this gives an experiment report, and nothing more:

  * ``paired_bootstrap_ci`` -- a paired bootstrap confidence interval on per-case deltas (resample CASES,
    not raw cells, with replacement.
  * ``case_aggregates`` -- replicate/seed aggregation: one row per case, collapsed across every seed run
    for it, never one row per raw cell.
  * ``stats_report`` -- multiple-comparison honesty: it reports exactly how many (suite, variant)
    comparisons this report makes (``n_comparisons``), and every comparison carries BOTH its raw-alpha CI
    and a Bonferroni-adjusted CI side by side, labeled as such.
  * a predeclared ``primary_metric`` read verbatim off the manifest (never re-guessed), and which
    comparisons it names, tagged in the report.
  * ``report_replay_class`` -- a replay-honesty label (bit-identical greedy / re-prefilled / stochastic-
    sampled / unknown) derived from each cell's own recorded decode settings, never assumed.

HONESTY (non-negotiable, per the roadmap's explicit warning against a "one-size-fits-all significance
badge"): there is no derived significant/not-significant verdict anywhere in this module. A reader sees the
point estimate, both intervals, the case count, and the comparison count -- never a star rating or a
pass/fail badge standing in for them.
"""
from __future__ import annotations

import random

from clozn.experiments.suite import PRIMARY_METRIC_KINDS  # noqa: F401 -- re-exported; one source of truth

DEFAULT_RESAMPLES = 2000
DEFAULT_ALPHA = 0.05

REPLAY_CLASSES = ("bit_identical_greedy", "re_prefilled", "stochastic_sampled", "unknown")


# ============================================================================================ replay class

def replay_class_for_meta(meta: dict | None) -> str:
    """Classify ONE run's replay honesty from its own recorded decode settings -- never inferred from
    surrounding context, never assumed. `meta` is a run's ``meta`` dict (the field vocabulary is
    ``clozn.testkit.runner.REPRO_META_KEYS``; ``sampler_mode``/``temperature`` are set in
    ``clozn.server.app``'s decode-block construction).

      * A teacher-forced RE-SCORE of an already-fixed continuation -- no new tokens decoded at all -- is
        marked here by the caller setting
        ``meta["forced_rescore"] = True``. -> ``"re_prefilled"``: deterministic given the same fixed
        tokens (one more forward pass over an unchanged prefix), but this recomputes scores over EXISTING
        text, not a fresh decode.
      * ``sampler_mode == "greedy"`` (equivalently ``temperature == 0.0``) -> ``"bit_identical_greedy"``:
        ``clozn.server.app``'s own contract is that greedy decode ignores top_k/top_p/seed "so receipts and
        forced-greedy replay stay bit-exact" -- replaying the same prompt under an unchanged build/quant
        reproduces the same output deterministically.
      * ``sampler_mode == "sample"`` -> ``"stochastic_sampled"``: token picks vary run to run for reasons
        that have nothing to do with a regression (``clozn.testkit.ci``'s own documented REPRODUCIBILITY
        CAVEAT). A recorded ``seed`` is provenance metadata only -- this codebase never demonstrates that
        replaying the same nominal seed reproduces the same sample, so it is never treated as one here.
      * missing/malformed meta, or a ``sampler_mode`` this module doesn't recognize -> ``"unknown"`` --
        never defaults to the strongest (bit-identical) claim just because the weaker signal is absent.
    """
    meta = meta if isinstance(meta, dict) else {}
    if meta.get("forced_rescore"):
        return "re_prefilled"
    mode = meta.get("sampler_mode")
    mode = str(mode).strip().lower() if isinstance(mode, str) else ""
    temp = meta.get("temperature")
    is_zero_temp = isinstance(temp, (int, float)) and not isinstance(temp, bool) and float(temp) == 0.0
    if mode == "greedy" or (mode == "" and is_zero_temp):
        return "bit_identical_greedy"
    if mode == "sample":
        return "stochastic_sampled"
    return "unknown"


def replay_class_for_cell(cell: dict) -> str:
    """``replay_class_for_meta`` applied to one suite.py result cell's embedded run record."""
    run = cell.get("run") if isinstance(cell, dict) else None
    meta = run.get("meta") if isinstance(run, dict) else None
    return replay_class_for_meta(meta)


def report_replay_class(cells: list) -> dict:
    """Aggregate ``replay_class_for_cell`` over every cell that actually carries run evidence (an
    ``error`` cell has none, per ``suite.validate_result``'s own contract, and is skipped here rather than
    guessed at). Returns ``{"classes": {class: count}, "overall": <the single class every cell agrees on,
    or "mixed">, "n_cells_with_run": int}`` -- ``overall`` is only ever a single named class when every
    counted cell actually agrees; it is never picked from a majority or a first-seen value."""
    counts = {name: 0 for name in REPLAY_CLASSES}
    n = 0
    for cell in cells or []:
        run = cell.get("run") if isinstance(cell, dict) else None
        if not isinstance(run, dict):
            continue
        n += 1
        counts[replay_class_for_cell(cell)] += 1
    present = [name for name, count in counts.items() if count]
    if len(present) == 1:
        overall = present[0]
    elif len(present) > 1:
        overall = "mixed"
    else:
        overall = "unknown"
    return {"classes": counts, "overall": overall, "n_cells_with_run": n}


# =================================================================================== seed/replicate aggregation

def _case_score(cells: list) -> float | None:
    """Mean pass/fail indicator (1.0 pass, 0.0 fail) over one case's seed replicates. ``error``/
    ``unscored`` cells are excluded from both numerator and denominator -- mirrors
    ``suite._summarize``'s own ``pass_rate`` definition exactly, so this module's per-case scores are
    never a silently different notion of "scored". ``None`` when nothing in the set was scored."""
    scored = [c for c in cells if c.get("status") in ("pass", "fail")]
    if not scored:
        return None
    return sum(1.0 for c in scored if c["status"] == "pass") / len(scored)


def case_aggregates(cells: list, *, suite: str, variant: str) -> dict:
    """Replicate/seed aggregation for one (suite, variant): ``{case_name: {"n_seeds", "counts",
    "pass_rate", "seeds"}}`` -- one row per case, already collapsed across every seed run for it. This is
    what ``paired_case_deltas``/the bootstrap below draw their per-case scores from."""
    by_case: dict = {}
    for cell in cells or []:
        if cell.get("suite") != suite or cell.get("variant") != variant:
            continue
        by_case.setdefault(cell.get("case"), []).append(cell)
    out = {}
    for case, case_cells in by_case.items():
        counts = {s: sum(1 for c in case_cells if c.get("status") == s)
                 for s in ("pass", "fail", "error", "unscored")}
        out[case] = {"n_seeds": len(case_cells), "counts": counts, "pass_rate": _case_score(case_cells),
                     "seeds": sorted(c.get("seed") for c in case_cells if c.get("seed") is not None)}
    return out


def paired_case_deltas(cells: list, *, suite: str, baseline: str, variant: str) -> dict:
    """Per-case (candidate - baseline) ``pass_rate`` deltas for one suite, paired by case name. A case
    contributes a delta only when BOTH variants have at least one scored (pass/fail) seed for it --
    unpaired or fully-unscored cases are listed in ``skipped_cases``, never imputed to 0."""
    base_agg = case_aggregates(cells, suite=suite, variant=baseline)
    cand_agg = case_aggregates(cells, suite=suite, variant=variant)
    deltas: dict = {}
    skipped = []
    for case in sorted(set(base_agg) | set(cand_agg), key=str):
        b_rate = base_agg[case]["pass_rate"] if case in base_agg else None
        c_rate = cand_agg[case]["pass_rate"] if case in cand_agg else None
        if b_rate is None or c_rate is None:
            skipped.append(case)
            continue
        deltas[case] = c_rate - b_rate
    return {"deltas": deltas, "skipped_cases": skipped,
           "baseline_aggregates": base_agg, "candidate_aggregates": cand_agg}


# ======================================================================================= paired bootstrap CI

def paired_bootstrap_ci(deltas: dict, *, n_resamples: int = DEFAULT_RESAMPLES, alpha: float = DEFAULT_ALPHA,
                        seed: int = 0) -> dict:
    """Percentile paired bootstrap over per-case deltas: resample CASES with replacement `n_resamples`
    times, take the mean delta each time, and report the ``alpha/2 .. 1 - alpha/2`` percentiles as the CI.
    Deterministic given `seed` (a private ``random.Random``, never the shared global RNG) -- a report is
    exactly reproducible, never flaky. Fewer than 2 paired cases returns an unavailable result rather than
    a meaningless point estimate."""
    values = list(deltas.values())
    n = len(values)
    if n < 2:
        return {"available": False, "reason": f"need at least 2 paired cases, got {n}", "n_cases": n,
               "alpha": alpha}
    point = sum(values) / n
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_resamples))

    def _percentile(p: float) -> float:
        idx = min(n_resamples - 1, max(0, round(p * (n_resamples - 1))))
        return means[idx]

    lo, hi = _percentile(alpha / 2), _percentile(1 - alpha / 2)
    return {"available": True, "n_cases": n, "point_estimate": round(point, 6), "alpha": alpha,
           "ci": [round(lo, 6), round(hi, 6)], "n_resamples": n_resamples, "seed": seed}


# ============================================================================ multiple-comparison honesty

def compare_variant(cells: list, *, suite: str, baseline: str, variant: str,
                    n_resamples: int = DEFAULT_RESAMPLES, alpha: float = DEFAULT_ALPHA,
                    bonferroni_m: int = 1, seed: int = 0) -> dict:
    """One (suite, baseline, variant) comparison: the paired case deltas, a raw-alpha CI, and a
    Bonferroni-adjusted CI at ``alpha / bonferroni_m`` -- reported side by side and labeled, never
    collapsed into a single pass/fail verdict."""
    paired = paired_case_deltas(cells, suite=suite, baseline=baseline, variant=variant)
    raw = paired_bootstrap_ci(paired["deltas"], n_resamples=n_resamples, alpha=alpha, seed=seed)
    bonferroni_alpha = alpha / bonferroni_m if bonferroni_m > 0 else alpha
    adjusted = paired_bootstrap_ci(paired["deltas"], n_resamples=n_resamples, alpha=bonferroni_alpha, seed=seed)
    return {"suite": suite, "baseline": baseline, "variant": variant, "skipped_cases": paired["skipped_cases"],
           "ci_raw": raw, "ci_bonferroni": adjusted, "bonferroni_alpha": bonferroni_alpha,
           "bonferroni_m": bonferroni_m}


def stats_report(result: dict, *, alpha: float = DEFAULT_ALPHA, n_resamples: int = DEFAULT_RESAMPLES,
                 seed: int = 0) -> dict:
    """The Phase 4.4 statistics report over one ``clozn.experiment.result.v0`` artifact. Callers should
    validate `result` with ``suite.validate_result`` first -- this module trusts that shape rather than
    re-deriving it.

    Reports one paired-bootstrap comparison per (suite, non-baseline variant) pair; `n_comparisons` is
    exactly how many such pairs this report makes, and every comparison's Bonferroni adjustment divides
    `alpha` by that same count -- so the honesty count and the adjustment can never silently drift apart.
    """
    manifest = result.get("manifest") or {}
    variants = [v["name"] for v in manifest.get("variants") or [] if isinstance(v, dict) and v.get("name")]
    baseline = manifest.get("baseline_variant")
    cells = result.get("cells") or []
    suite_names = ("target", "guard")
    others = [v for v in variants if v != baseline]
    n_comparisons = max(len(others) * len(suite_names), 1)

    comparisons = []
    for suite_name in suite_names:
        for variant in others:
            comparisons.append(compare_variant(cells, suite=suite_name, baseline=baseline, variant=variant,
                                               n_resamples=n_resamples, alpha=alpha,
                                               bonferroni_m=n_comparisons, seed=seed))

    primary_metric = manifest.get("primary_metric")
    primary_comparisons = None
    if isinstance(primary_metric, dict):
        primary_comparisons = [c for c in comparisons if c["suite"] == primary_metric.get("suite")]

    return {
        "experiment_id": result.get("experiment_id"), "name": result.get("name"),
        "baseline_variant": baseline, "alpha": alpha, "n_comparisons": n_comparisons,
        "comparisons": comparisons, "primary_metric": primary_metric,
        "primary_comparisons": primary_comparisons, "replay_class": report_replay_class(cells),
        "case_replicate_aggregates": {
            suite_name: {variant: case_aggregates(cells, suite=suite_name, variant=variant)
                        for variant in variants}
            for suite_name in suite_names
        },
    }


# ==================================================================================================== format

def format_stats_report(report: dict) -> str:
    """Human-readable rendering. NEVER prints a significance badge/star rating -- every comparison line
    prints its own case count, point estimate, and both intervals side by side (see module docstring)."""
    lines = [f"clozn experiment stats {report.get('experiment_id')}  {report.get('name')}",
            f"  baseline: {report.get('baseline_variant')}  comparisons made: {report['n_comparisons']} "
            f"(raw alpha={report['alpha']:g}; multiple-comparison honesty means reading the "
            "Bonferroni-adjusted interval below alongside the raw one, not a single verdict)"]
    replay = report["replay_class"]
    lines.append(f"  replay class: {replay['overall']}  ({replay['n_cells_with_run']} cell(s) with run "
                f"evidence; breakdown {replay['classes']})")
    primary = report.get("primary_metric")
    if primary:
        lines.append(f"  predeclared primary metric: suite={primary['suite']}  metric={primary['metric']}")
    else:
        lines.append("  predeclared primary metric: none declared in the manifest -- every "
                    "comparison below is exploratory")
    for comp in report["comparisons"]:
        raw, adjusted = comp["ci_raw"], comp["ci_bonferroni"]
        tag = " [PRIMARY]" if primary and comp["suite"] == primary.get("suite") else ""
        label = f"  {comp['suite']}/{comp['variant']} vs {comp['baseline']}{tag}"
        if not raw["available"]:
            lines.append(f"{label}: {raw['reason']}")
            continue
        lines.append(f"{label}: n_cases={raw['n_cases']}  delta={raw['point_estimate']:+.4f}  "
                    f"ci_raw@alpha={raw['alpha']:g}={raw['ci']}  "
                    f"ci_bonferroni@alpha={adjusted['alpha']:.4g}={adjusted['ci']}")
        if comp["skipped_cases"]:
            lines.append(f"    skipped (unpaired/unscored): {comp['skipped_cases']}")
    return "\n".join(lines)
