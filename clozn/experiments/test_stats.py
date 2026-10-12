"""Tests for clozn.experiments.stats (Phase 4.4: paired bootstrap CIs, seed aggregation, multiple-
comparison honesty, predeclared primary metric, and replay-honesty labels).
Model-free: everything here is pure over hand-built / suite.validate_result-shaped dicts, never a live
gateway."""
from __future__ import annotations

import pytest

from clozn.experiments import stats, suite


# ======================================================================================== replay class

def test_replay_class_for_meta_greedy_is_bit_identical():
    assert stats.replay_class_for_meta({"sampler_mode": "greedy", "temperature": 0.0}) == "bit_identical_greedy"
    # temperature 0.0 alone (no explicit sampler_mode) reaches the same conclusion, never a stronger one
    # than the recorded fields actually support.
    assert stats.replay_class_for_meta({"temperature": 0.0}) == "bit_identical_greedy"


def test_replay_class_for_meta_sample_is_stochastic():
    assert stats.replay_class_for_meta({"sampler_mode": "sample", "temperature": 0.8}) == "stochastic_sampled"


def test_replay_class_for_meta_forced_rescore_is_re_prefilled_regardless_of_sampler_mode():
    assert stats.replay_class_for_meta({"sampler_mode": "sample", "forced_rescore": True}) == "re_prefilled"
    assert stats.replay_class_for_meta({"sampler_mode": "greedy", "forced_rescore": True}) == "re_prefilled"


def test_replay_class_for_meta_never_guesses_on_missing_or_malformed_fields():
    assert stats.replay_class_for_meta(None) == "unknown"
    assert stats.replay_class_for_meta({}) == "unknown"
    assert stats.replay_class_for_meta({"sampler_mode": "quantum"}) == "unknown"
    assert stats.replay_class_for_meta({"temperature": True}) == "unknown"   # bool is not a real temperature


def test_report_replay_class_agrees_when_every_cell_agrees():
    cells = [{"run": {"meta": {"sampler_mode": "greedy", "temperature": 0.0}}},
            {"run": {"meta": {"sampler_mode": "greedy", "temperature": 0.0}}}]
    out = stats.report_replay_class(cells)
    assert out["overall"] == "bit_identical_greedy"
    assert out["classes"]["bit_identical_greedy"] == 2
    assert out["n_cells_with_run"] == 2


def test_report_replay_class_is_mixed_when_cells_disagree():
    cells = [{"run": {"meta": {"sampler_mode": "greedy", "temperature": 0.0}}},
            {"run": {"meta": {"sampler_mode": "sample", "temperature": 0.9}}},
            {"status": "error", "run": None}]                # error cell has no run -- excluded, not guessed
    out = stats.report_replay_class(cells)
    assert out["overall"] == "mixed"
    assert out["n_cells_with_run"] == 2                       # the error cell never counted


def test_report_replay_class_unknown_when_no_run_evidence_at_all():
    assert stats.report_replay_class([]) == {"classes": {c: 0 for c in stats.REPLAY_CLASSES},
                                             "overall": "unknown", "n_cells_with_run": 0}


# ================================================================================ seed/replicate aggregation

def _cell(suite_name, case, variant, seed, status):
    return {"suite": suite_name, "case": case, "variant": variant, "seed": seed, "status": status}


def test_case_aggregates_collapses_seeds_and_excludes_error_unscored_from_pass_rate():
    cells = [
        _cell("target", "c1", "base", 0, "pass"), _cell("target", "c1", "base", 1, "fail"),
        _cell("target", "c1", "base", 2, "error"), _cell("target", "c1", "base", 3, "unscored"),
    ]
    agg = stats.case_aggregates(cells, suite="target", variant="base")
    assert agg["c1"]["n_seeds"] == 4
    assert agg["c1"]["pass_rate"] == pytest.approx(0.5)       # 1 pass / (1 pass + 1 fail) -- error/unscored excluded
    assert agg["c1"]["counts"] == {"pass": 1, "fail": 1, "error": 1, "unscored": 1}
    assert agg["c1"]["seeds"] == [0, 1, 2, 3]


def test_case_aggregates_pass_rate_is_none_when_nothing_was_scored():
    cells = [_cell("target", "c1", "base", 0, "error"), _cell("target", "c1", "base", 1, "unscored")]
    agg = stats.case_aggregates(cells, suite="target", variant="base")
    assert agg["c1"]["pass_rate"] is None


def test_paired_case_deltas_pairs_by_case_and_skips_unpaired_or_unscored():
    cells = [
        _cell("target", "shared", "base", 0, "fail"), _cell("target", "shared", "base", 1, "fail"),
        _cell("target", "shared", "cand", 0, "pass"), _cell("target", "shared", "cand", 1, "pass"),
        # "only-in-cand" has no baseline cell at all -- must be skipped, never imputed to a 0 baseline.
        _cell("target", "only-in-cand", "cand", 0, "pass"),
        # "unscored-both" is present in both variants but never scored -- also skipped.
        _cell("target", "unscored-both", "base", 0, "unscored"),
        _cell("target", "unscored-both", "cand", 0, "unscored"),
    ]
    out = stats.paired_case_deltas(cells, suite="target", baseline="base", variant="cand")
    assert out["deltas"] == {"shared": pytest.approx(1.0)}     # 1.0 (cand) - 0.0 (base)
    assert set(out["skipped_cases"]) == {"only-in-cand", "unscored-both"}


# ============================================================================================= bootstrap CI

def test_paired_bootstrap_ci_needs_at_least_two_paired_cases():
    out = stats.paired_bootstrap_ci({"only-one": 0.3})
    assert out["available"] is False and out["n_cases"] == 1


def test_paired_bootstrap_ci_is_deterministic_given_a_seed():
    deltas = {"a": 0.4, "b": -0.2, "c": 0.1, "d": 0.3, "e": -0.1}
    one = stats.paired_bootstrap_ci(deltas, n_resamples=500, seed=7)
    two = stats.paired_bootstrap_ci(deltas, n_resamples=500, seed=7)
    assert one == two


def test_paired_bootstrap_ci_zero_variance_deltas_collapse_to_a_point_interval():
    # every case has the identical delta -> every bootstrap resample's mean is exactly that value.
    deltas = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.5}
    out = stats.paired_bootstrap_ci(deltas, n_resamples=200, seed=0)
    assert out["available"] is True
    assert out["point_estimate"] == pytest.approx(0.5)
    assert out["ci"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert out["n_cases"] == 4


def test_paired_bootstrap_ci_reports_alpha_and_resamples_used():
    out = stats.paired_bootstrap_ci({"a": 0.1, "b": 0.2, "c": 0.3}, n_resamples=300, alpha=0.2, seed=1)
    assert out["alpha"] == 0.2 and out["n_resamples"] == 300 and out["seed"] == 1


# =============================================================================== manifest primary_metric

def _base_manifest(**overrides):
    manifest = {
        "schema_version": suite.MANIFEST_SCHEMA, "name": "primary-metric-check", "seeds": [0],
        "defaults": {}, "baseline_variant": "base",
        "variants": [{"name": "base", "kind": "base"}, {"name": "cand", "kind": "tuned"}],
        "suites": {"target": {"cases": [{"name": "c1", "prompt": "hi"}]},
                  "guard": {"cases": [{"name": "g1", "prompt": "hi"}]}},
    }
    manifest.update(overrides)
    return manifest


def test_primary_metric_is_optional_and_absent_by_default():
    manifest = suite.validate_manifest(_base_manifest())
    assert "primary_metric" not in manifest


def test_primary_metric_normalizes_a_valid_declaration():
    manifest = suite.validate_manifest(_base_manifest(primary_metric={"suite": "target", "metric": "pass_rate"}))
    assert manifest["primary_metric"] == {"suite": "target", "metric": "pass_rate"}


@pytest.mark.parametrize("bad", [
    {"suite": "bogus", "metric": "pass_rate"},
    {"suite": "target", "metric": "not_a_real_metric"},
    "not-a-dict",
])
def test_primary_metric_rejects_malformed_declarations(bad):
    with pytest.raises(suite.ManifestError, match="primary_metric"):
        suite.validate_manifest(_base_manifest(primary_metric=bad))


def test_manifest_validation_is_digest_idempotent_without_a_primary_metric():
    """Repeated validation must not change the canonical manifest digest."""
    manifest = _base_manifest()
    before = suite._manifest_digest(suite.validate_manifest(manifest))
    # Re-validating (as validate_result does on every load) must be idempotent and stable.
    after = suite._manifest_digest(suite.validate_manifest(suite.validate_manifest(manifest)))
    assert before == after


# ======================================================================================= full stats_report

def _result_with_three_variants():
    """A minimal, fully suite.validate_result-shaped artifact: 3 variants (base + two non-baseline
    candidates), 2 seeds, one case per suite -- enough to exercise n_comparisons scaling (2 candidates x
    2 suites = 4) and Bonferroni division by that count."""
    manifest = suite.validate_manifest({
        "schema_version": suite.MANIFEST_SCHEMA, "name": "three-variant-check",
        "primary_metric": {"suite": "target", "metric": "pass_rate"},
        "seeds": [0, 1], "defaults": {"temperature": 0}, "baseline_variant": "base",
        "variants": [{"name": "base", "kind": "base"}, {"name": "cand-a", "kind": "tuned"},
                    {"name": "cand-b", "kind": "tuned"}],
        "suites": {
            "target": {"cases": [{"name": "capital", "prompt": "Capital of France?",
                                    "expect": {"contains": "Paris"}},
                                 {"name": "capital-2", "prompt": "Capital of Japan?",
                                  "expect": {"contains": "Tokyo"}}]},
            "guard": {"cases": [{"name": "colors", "prompt": "Name a color", "expect": {"contains": "red"}},
                               {"name": "colors-2", "prompt": "Name another color",
                                "expect": {"contains": "blue"}}]},
        },
    })
    # base: always fails target, always passes guard. cand-a: always passes both (a real target gain, no
    # guard regression). cand-b: passes target but regresses guard (a real guard regression) -- gives the
    # comparisons genuine, non-degenerate separation instead of every delta being identical. Two cases per
    # suite (both following the SAME pattern) so paired_bootstrap_ci has >= 2 paired cases to work with.
    status_by_variant_suite = {
        ("base", "target"): "fail", ("base", "guard"): "pass",
        ("cand-a", "target"): "pass", ("cand-a", "guard"): "pass",
        ("cand-b", "target"): "pass", ("cand-b", "guard"): "fail",
    }
    cells = []
    for variant in manifest["variants"]:
        name = variant["name"]
        for suite_name, case in (("target", "capital"), ("target", "capital-2"),
                                 ("guard", "colors"), ("guard", "colors-2")):
            for seed in manifest["seeds"]:
                status = status_by_variant_suite[(name, suite_name)]
                run_id = f"run-{name}-{suite_name}-{seed}"
                cells.append({
                    "suite": suite_name, "case": case, "variant": name, "variant_kind": variant["kind"],
                    "seed": seed, "status": status, "run_id": run_id,
                    "response": "Paris" if suite_name == "target" else "red",
                    "assertions": [], "min_confidence": None, "receipts": None, "error": None,
                    "run": {"id": run_id, "model": "clozn", "meta": {"sampler_mode": "greedy", "temperature": 0.0}},
                })
    result = {
        "schema_version": suite.RESULT_SCHEMA, "experiment_id": "exp_three_variant", "name": manifest["name"],
        "created_at": "2026-07-22T00:00:00Z", "manifest_sha256": suite._manifest_digest(manifest),
        "suite_fingerprint": suite.suite_fingerprint(manifest, seeds=manifest["seeds"]),
        "manifest": manifest, "seeds": manifest["seeds"], "cells": cells,
        "summary": suite._summarize(cells, "base", [v["name"] for v in manifest["variants"]]),
    }
    return suite.validate_result(result)


def test_stats_report_counts_comparisons_and_scales_bonferroni_alpha():
    result = _result_with_three_variants()
    report = stats.stats_report(result, alpha=0.05, n_resamples=300, seed=0)
    assert report["n_comparisons"] == 4                        # 2 non-baseline variants x 2 suites
    for comp in report["comparisons"]:
        assert comp["bonferroni_alpha"] == pytest.approx(0.05 / 4)
        assert comp["ci_bonferroni"]["alpha"] == pytest.approx(0.05 / 4)
        assert comp["ci_raw"]["alpha"] == 0.05


def test_stats_report_reflects_the_real_pass_fail_pattern():
    result = _result_with_three_variants()
    report = stats.stats_report(result, n_resamples=300, seed=0)
    by_key = {(c["suite"], c["variant"]): c for c in report["comparisons"]}
    # cand-a fixes the target failure -> a positive delta, zero-variance (every case/seed agrees) CI.
    target_a = by_key[("target", "cand-a")]["ci_raw"]
    assert target_a["point_estimate"] == pytest.approx(1.0)
    assert target_a["ci"] == [pytest.approx(1.0), pytest.approx(1.0)]
    # cand-b regresses guard -> a negative delta.
    guard_b = by_key[("guard", "cand-b")]["ci_raw"]
    assert guard_b["point_estimate"] == pytest.approx(-1.0)


def test_stats_report_tags_only_the_predeclared_primary_suite():
    result = _result_with_three_variants()
    report = stats.stats_report(result, n_resamples=200, seed=0)
    assert report["primary_metric"] == {"suite": "target", "metric": "pass_rate"}
    assert {c["suite"] for c in report["primary_comparisons"]} == {"target"}
    assert len(report["primary_comparisons"]) == 2              # both non-baseline variants, target suite only


def test_stats_report_primary_comparisons_is_none_when_manifest_declares_nothing():
    result = _result_with_three_variants()
    result["manifest"].pop("primary_metric", None)
    report = stats.stats_report(result, n_resamples=200, seed=0)
    assert report["primary_metric"] is None
    assert report["primary_comparisons"] is None


def test_stats_report_replay_class_is_bit_identical_when_every_cell_is_greedy():
    result = _result_with_three_variants()
    report = stats.stats_report(result, n_resamples=100, seed=0)
    assert report["replay_class"]["overall"] == "bit_identical_greedy"
    assert report["replay_class"]["n_cells_with_run"] == len(result["cells"])


def test_format_stats_report_never_prints_a_significance_badge():
    result = _result_with_three_variants()
    report = stats.stats_report(result, n_resamples=200, seed=0)
    text = stats.format_stats_report(report)
    assert "significant" not in text.lower()
    assert "Bonferroni" in text
    assert "comparisons made: 4" in text
    assert "[PRIMARY]" in text
    assert "replay class: bit_identical_greedy" in text
