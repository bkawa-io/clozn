"""depth_vs_leverage_analysis.py -- ANALYSIS ONLY, no engine calls, no new GPU work.

Reads runs/experiments/quant_mixed_precision_bands.json (clozn.quant_mixed_precision_bands.v1) and
answers, using existing per-arm continuous metrics already recorded by that script's own
`_run_window`/`_movement_results` calls (not re-derived, not re-scored): does the late-band advantage
in docs/research/QUANT_MIXED_PRECISION_BANDS.md look like content-specificity or raw depth leverage?

WHY THIS SCRIPT, NOT JUST THE ALREADY-PUBLISHED `characterization` BLOCK
--------------------------------------------------------------------------
The report's own `characterization.by_band` only carries the BINARY outcome per arm (`moved` /
`random_moved`, both computed by the SAME function `causal_bisect._flipped_to_target` -- confirmed by
reading `causal_bisect.py` directly, not inferred from field names). That binary answers "did the
random-equal-norm control land EXACTLY on the target token," but not "how hard did it try" or "which
direction did it move in and by how much." Every record's `band_results[band]["movement_metrics"]`
already carries a CONTINUOUS score for both the reference_transplant and random_equal_norm arms
(`reference_token_logprob_recovery`: how much closer to the reference model's own logprob for the
target token the treated logprob moved, signed, `direction_vs_reference` in {toward_reference,
away_from_reference}) -- computed once by `_movement_results` inside `_run_window` and stored verbatim
in the JSON already on disk. This script only READS and aggregates that pre-computed field; it applies
no new scoring rule.

This directly tests the "random-at-late destroys the output so thoroughly it can never land on target
by chance" alternative explanation: if that were true, random-at-late's movement should be LARGER in
magnitude and more skewed toward "moved to some THIRD token, not baseline's own top-1" than
random-at-mid's. If instead random-at-late's movement looks like random-at-mid's (similar magnitude,
similar direction split, similarly often a no-op on top-1), the "destroys the output" story does not
fit the record-level data, and the null result at late (0/25 random-beats) reads as evidence of
direction-specificity, not raw destructive leverage.

USAGE
-------
python scripts/tracer/depth_vs_leverage_analysis.py [--report PATH]
Prints a plain-text table; writes nothing. Exits 0 always (analysis, not a test).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as stats
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REPORT = os.path.join(REPO, "runs", "experiments", "quant_mixed_precision_bands.json")

BANDS = ("early", "mid", "late_matched")
ARMS = ("reference_transplant", "candidate_self_transplant", "random_equal_norm", "shuffled_window")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iter_band_results(data: dict):
    """Yields (record_id, candidate, category, band_name, band_result_dict) for every band_result that
    was actually tested (not skipped -- i.e. has 'arms')."""
    for rec in data["records"]:
        cr = rec.get("candidate_result")
        if not cr or not cr.get("ok"):
            continue
        for band_name, br in (cr.get("band_results") or {}).items():
            if band_name not in BANDS:
                continue
            if not br or br.get("skipped") or not br.get("arms"):
                continue
            yield rec["id"], rec["candidate"], rec["category"], band_name, br


def site_count_audit(data: dict) -> None:
    """Confirms, from the RECORDS (not the bounds/method prose), that every record's early/mid/late_matched
    band has the same site count -- the 'are the counts genuinely matched' question the caller asked to
    verify directly rather than trust the doc's prose."""
    counts = defaultdict(Counter)
    for rec in data["records"]:
        cr = rec.get("candidate_result")
        if not cr or not cr.get("ok"):
            continue
        bands = cr.get("bands") or {}
        for b in BANDS:
            n = len(bands.get(b) or [])
            counts[b][n] += 1
    print("=== Site-count audit (from records[*].candidate_result.bands, not from bounds/method prose) ===")
    for b in BANDS:
        print(f"  {b:<14} n_sites histogram: {dict(counts[b])}")
    all_nine = all(set(counts[b].keys()) == {9} for b in BANDS if counts[b])
    print(f"  every record has exactly 9 sites in every band: {all_nine}")
    print()


def arm_measurement_symmetry_audit() -> None:
    print("=== Measurement-symmetry audit (from causal_bisect.py source, not inferred from field names) ===")
    print("  causal_bisect._run_window computes BOTH 'moved' (reference_transplant) and 'random_moved'")
    print("  (random_equal_norm) via the SAME function: _flipped_to_target(baseline_metrics, arm_metrics) =")
    print("  (not baseline_hit) and arm_hit. Same top1_is_target field, same equality check, both arms.")
    print("  beat_control = bool(reference_moved and not random_moved) -- a joint rule over both arms'")
    print("  identically-computed flip flags, not two differently-scored criteria compared post hoc.")
    print("  Reference and random are NOT scored differently. Verified by reading the function, not assumed.")
    print()


def outcome_breakdown_by_band_and_arm(data: dict) -> dict:
    """For each (band, arm): partitions every tested (band present, instrument_sane) record's arm result
    into three buckets using the SAME top1_token_id field every arm already carries:
      - 'restored'  : top1_token_id == target_token_id (the flip-to-target event; matches 'moved'/'random_moved')
      - 'no_op'     : top1_token_id == baseline top1_token_id (write changed nothing observable at top-1)
      - 'diverted'  : top1_token_id is neither baseline's own top1 nor the target -- moved to a THIRD token.
    'diverted' is the direct empirical signature the 'destroys the output' hypothesis predicts should be
    LARGER at late than at mid for the random arm, if that hypothesis is what explains late's null random
    rate. Only counts records where baseline was NOT already-correct (mirrors _band_stats' own exclusion,
    read from 'moved' is not None as the sentinel -- a band_result with moved present had a live
    disagreement to test)."""
    out: dict = {b: {a: Counter() for a in ARMS} for b in BANDS}
    for _id, _cand, _cat, band, br in _iter_band_results(data):
        if br.get("moved") is None:
            continue  # already-correct-at-baseline or not evaluable -- excluded, same as _band_stats
        baseline_top1 = None
        arms = br.get("arms") or {}
        self_arm = arms.get("candidate_self_transplant") or {}
        baseline_top1 = self_arm.get("top1_token_id")  # self-transplant is a confirmed no-op -> == baseline top1
        target_hits = {a: (arms.get(a) or {}).get("top1_is_target") for a in ARMS}
        for a in ARMS:
            am = arms.get(a)
            if am is None:
                continue
            t1 = am.get("top1_token_id")
            if target_hits.get(a) is True:
                out[band][a]["restored"] += 1
            elif t1 is not None and baseline_top1 is not None and t1 == baseline_top1:
                out[band][a]["no_op"] += 1
            elif t1 is not None:
                out[band][a]["diverted"] += 1
            else:
                out[band][a]["not_evaluable"] += 1
    return out


def movement_stats_by_band_and_arm(data: dict) -> dict:
    """Pulls the PRE-COMPUTED, continuous movement_metrics.<arm>.result.movement (and
    direction_vs_reference) already stored per band_result -- signed logprob-space movement of the
    target token toward (positive) or away from (negative) the reference model's own logprob for it.
    Aggregates mean/median/stdev per (band, arm) plus a toward/away count, restricted to disagreements
    where the metric reached 'selected' state (both baseline and reference_target_logprob available)."""
    out: dict = {b: {a: [] for a in ("reference_transplant", "random_equal_norm")} for b in BANDS}
    directions: dict = {b: {a: Counter() for a in ("reference_transplant", "random_equal_norm")} for b in BANDS}
    for _id, _cand, _cat, band, br in _iter_band_results(data):
        mm = br.get("movement_metrics") or {}
        for a in ("reference_transplant", "random_equal_norm"):
            entry = mm.get(a) or {}
            if entry.get("state") != "selected":
                continue
            res = entry.get("result") or {}
            if res.get("state") != "measurable":
                continue
            mv = res.get("movement")
            if isinstance(mv, (int, float)):
                out[band][a].append(mv)
                directions[band][a][res.get("direction_vs_reference")] += 1
    return {"movement": out, "direction": directions}


def _summary(values: list) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(stats.mean(values), 3),
        "median": round(stats.median(values), 3),
        "stdev": round(stats.stdev(values), 3) if len(values) > 1 else 0.0,
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "abs_mean": round(stats.mean(abs(v) for v in values), 3),
    }


def gap_closed_fraction_by_band(data: dict, candidate_filter: "str | None" = None) -> dict:
    """gap_closed_fraction = movement / gap, where gap = |reference_target_logprob - baseline_logprob| --
    a property of the DISAGREEMENT, not the band (same baseline_metrics and reference_target_logprob are
    reused across all 4 band calls for a given disagreement -- confirmed in run_candidate_disagreement:
    ctx['reference_target_logprob'] and baseline_metrics are computed ONCE per disagreement, before the
    BAND_ORDER loop). This makes gap_closed_fraction, unlike raw 'movement', directly comparable ACROSS
    bands for the same disagreement: it measures what fraction of the SAME target distance each band's
    write closed, not raw nats (which trivially scales with how large that disagreement's own gap was)."""
    out = {b: {"reference_transplant": [], "random_equal_norm": []} for b in BANDS}
    for rec in data["records"]:
        if candidate_filter and rec.get("candidate") != candidate_filter:
            continue
        cr = rec.get("candidate_result")
        if not cr or not cr.get("ok"):
            continue
        for b in BANDS:
            br = (cr.get("band_results") or {}).get(b) or {}
            mm = br.get("movement_metrics") or {}
            for arm in ("reference_transplant", "random_equal_norm"):
                e = mm.get(arm) or {}
                if e.get("state") != "selected":
                    continue
                res = e.get("result") or {}
                if res.get("state") != "measurable":
                    continue
                g = res.get("gap_closed_fraction")
                if isinstance(g, (int, float)):
                    out[b][arm].append(g)
    return out


def shuffled_window_destination_audit(data: dict) -> dict:
    """`_pick_shuffled_sites(sites, usable_sites)` (causal_bisect.py) returns the FIRST len(sites) entries
    of `usable_sites` disjoint from the band's own `sites`, in ASCENDING layer order -- NOT a same-band
    shuffle. Recomputed here (deterministically, from each record's own `usable_layers` + `bands`, not
    guessed) to state exactly what shuffled_window tests for each band, since misreading this control
    would be exactly the kind of error this analysis was asked to guard against."""
    dest_by_band = {b: Counter() for b in BANDS}
    for rec in data["records"]:
        cr = rec.get("candidate_result")
        if not cr or not cr.get("ok"):
            continue
        usable = cr["usable_layers"]
        bands = cr["bands"]
        for b in BANDS:
            sites = bands[b]
            br = (cr.get("band_results") or {}).get(b) or {}
            if not br.get("arms") or "shuffled_window" not in (br.get("arms") or {}):
                continue
            pool = [s for s in usable if s not in sites]
            dest = tuple(pool[:len(sites)])
            dest_by_band[b][dest] += 1
    return dest_by_band


def print_report(path: str) -> None:
    data = _load(path)
    print(f"Loaded {path}")
    print(f"  schema={data.get('schema')} generated_at={data.get('generated_at')} "
          f"reference_model={data.get('reference_model')} candidates={data.get('candidates')}")
    print()

    site_count_audit(data)
    arm_measurement_symmetry_audit()

    print("=== Outcome breakdown by band x arm (from arms[*].top1_token_id / top1_is_target, record-level) ===")
    breakdown = outcome_breakdown_by_band_and_arm(data)
    header = f"{'band':<14}{'arm':<24}{'restored':>9}{'no_op':>7}{'diverted':>9}{'not_eval':>9}{'n':>5}"
    print(header)
    for b in BANDS:
        for a in ARMS:
            c = breakdown[b][a]
            n = sum(c.values())
            print(f"{b:<14}{a:<24}{c['restored']:>9}{c['no_op']:>7}{c['diverted']:>9}"
                  f"{c.get('not_evaluable', 0):>9}{n:>5}")
    print()
    print("  Reading guide: 'restored' matches characterization.by_band[band].n_moved_reference_arm /")
    print("  n_random_control_also_moved exactly (same flip flag) -- this table exists to break it open")
    print("  further into no_op vs diverted, which the published characterization block does not carry.")
    print()

    print("=== Continuous movement (movement_metrics[<arm>].result.movement, logprob-space, signed) ===")
    mv = movement_stats_by_band_and_arm(data)
    for band in BANDS:
        print(f"  -- {band} --")
        for arm in ("reference_transplant", "random_equal_norm"):
            s = _summary(mv["movement"][band][arm])
            dirs = dict(mv["direction"][band][arm])
            print(f"    {arm:<22} {s}  direction_vs_reference={dirs}")
    print()
    print("  If 'random-at-late destroys the output' (a leverage story) were driving the 0/25 random-beat")
    print("  rate at late, random_equal_norm's movement at late should show LARGER magnitude (abs_mean) and/or")
    print("  a stronger away_from_reference skew than at mid. If magnitude/direction are comparable across")
    print("  bands for the random arm, the null late-random rate is not explained by 'more chaotic at late'.")
    print()

    print("=== Reference-arm movement magnitude by band (does correctly-aimed leverage rise with depth?) ===")
    for band in BANDS:
        s = _summary(mv["movement"][band]["reference_transplant"])
        print(f"  {band:<14} reference_transplant movement: {s}")
    print("  A rising abs_mean/median here across early->mid->late, even restricted to the SAME correctly-")
    print("  aimed reference direction, is exactly what a pure depth-leverage story predicts (a correctly-")
    print("  aimed intervention moves logits more per site simply by being closer to the unembedding) --")
    print("  this table quantifies that component using the study's own recorded numbers, without yet")
    print("  separating it from a content-specific damage-concentration story (see analysis writeup).")
    print()

    print("=== gap_closed_fraction by band x arm (normalized: movement / this disagreement's own gap; ===")
    print("=== the gap is identical across bands for a given disagreement, so THIS is the fair cross-band ===")
    print("=== comparison, unlike raw movement above, which trivially scales with each disagreement's own gap) ===")
    gcf_pooled = gap_closed_fraction_by_band(data)
    for band in BANDS:
        for arm in ("reference_transplant", "random_equal_norm"):
            s = _summary(gcf_pooled[band][arm])
            print(f"  {band:<14}{arm:<22} {s}")
    print()
    print("  -- same table, split by candidate (robustness check: does the pattern survive stratification, --")
    print("  -- the exact check this project's own history says to run before trusting a clean-looking result) --")
    for cand in sorted({r["candidate"] for r in data["records"]}):
        print(f"  -- candidate={cand} --")
        gcf_c = gap_closed_fraction_by_band(data, candidate_filter=cand)
        for band in BANDS:
            for arm in ("reference_transplant", "random_equal_norm"):
                s = _summary(gcf_c[band][arm])
                print(f"    {band:<14}{arm:<22} {s}")
    print()

    print("=== shuffled_window destination-site audit (what does this control ACTUALLY test per band?) ===")
    dest = shuffled_window_destination_audit(data)
    for b in BANDS:
        print(f"  {b:<14} shuffled_window destination sites used: {dict(dest[b])}")
    print("  shuffled_window does NOT shuffle WITHIN a band -- _pick_shuffled_sites always returns the")
    print("  numerically-LOWEST disjoint usable sites. Concretely: mid's and late_matched's shuffled_window")
    print("  BOTH write their own band's real reference vectors into the EARLY band's site indices (0-8).")
    print("  This means comparing mid's vs late's shuffled_window restoration rate holds DESTINATION DEPTH")
    print("  constant (both land at layers 0-8) and varies only WHICH layer the real content was captured")
    print("  at -- a genuine (if accidental) content-vs-destination-depth separation already latent in this")
    print("  artifact. See analysis writeup for the restoration-rate comparison and its own caveats (raw")
    print("  vector norm by source depth is not verifiable from this artifact: store_tensors=False).")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    args = ap.parse_args()
    if not os.path.exists(args.report):
        print(f"NOT FOUND: {args.report} -- this artifact is gitignored and was not present locally. "
              f"No analysis run; see docs/research/DEPTH_VS_LEVERAGE.md for what was available instead.")
        return 0
    print_report(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
