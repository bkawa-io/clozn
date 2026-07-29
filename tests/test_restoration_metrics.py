"""test_restoration_metrics -- clozn/analysis/restoration_metrics.py, slice 3.6.

Model-free throughout (roadmap rule 8): every function under test is a pure computation over plain
floats, plain token-id lists, or hand-built `clozn.experiments.suite`-shaped cell dicts. No engine, no
GPU, no network, no filesystem -- this module doesn't touch disk, so there is nothing for the
`_never_write_the_real_user_data` tripwire in conftest.py to catch, but tests still avoid the real
`~/.clozn` directory as a matter of course.
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from clozn.analysis import restoration_metrics as rm  # noqa: E402


# ============================================================================================ metric 1/2/3
# reference_token_logprob_recovery / candidate_token_suppression / sequence_nll_movement all share the
# `_movement()` primitive; exercise the shared states through metric 1 and spot-check the others.

def test_recovery_normal_case_reports_gap_closed_toward_reference():
    r = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0,
                                            treated_logprob=-2.0)
    assert r["metric"] == "reference_token_logprob_recovery"
    assert r["state"] == "measurable"
    assert r["movement"] == pytest.approx(3.0)
    assert r["movement_sign"] == "increased"
    assert r["direction_vs_reference"] == "toward_reference"
    assert r["gap"] == pytest.approx(4.9)
    assert r["gap_closed_fraction"] == pytest.approx(3.0 / 4.9)
    assert r["omitted"] == []


def test_recovery_wrong_direction_is_reported_as_away_not_toward():
    """Direction and magnitude must be distinguishable from noise: a move in the WRONG direction is its
    own explicit state, never silently folded into "no movement" or a small positive number."""
    r = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0,
                                            treated_logprob=-6.0)
    assert r["state"] == "measurable"
    assert r["movement"] == pytest.approx(-1.0)
    assert r["movement_sign"] == "decreased"
    assert r["direction_vs_reference"] == "away_from_reference"
    assert r["gap_closed_fraction"] < 0          # negative fraction: unambiguously the wrong way


def test_recovery_no_movement_is_its_own_state():
    r = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0,
                                            treated_logprob=-5.0)
    assert r["movement"] == 0.0
    assert r["movement_sign"] == "unchanged"
    assert r["direction_vs_reference"] == "unmoved"


def test_degenerate_baseline_refuses_a_fabricated_ratio():
    """The central normalization-honesty requirement: reference and baseline start within noise of each
    other, so 'percent of gap closed' is refused rather than exploding or being silently 0/1."""
    r = rm.reference_token_logprob_recovery(reference_logprob=-2.0000000001, baseline_logprob=-2.0,
                                            treated_logprob=-1.0)
    assert r["state"] == "degenerate_gap"
    assert "gap_closed_fraction" not in r
    assert r["gap"] == pytest.approx(0.0, abs=1e-6)
    # the raw values are still honestly reported -- nothing here is hidden, only the fabricated ratio is
    assert r["baseline_value"] == -2.0
    assert r["treated_value"] == -1.0
    assert r["movement"] == pytest.approx(1.0)
    reasons = {o["field"]: o["reason"] for o in r["omitted"]}
    assert "gap_closed_fraction" in reasons
    assert "divide by ~zero" in reasons["gap_closed_fraction"]


def test_degenerate_baseline_with_zero_treated_movement_is_at_reference_not_away():
    """When baseline == reference AND treated == baseline, nothing moved at all -- distinct from the
    degenerate-gap-but-treated-moved case above, and must not be misreported as 'away'."""
    r = rm.reference_token_logprob_recovery(reference_logprob=-2.0, baseline_logprob=-2.0,
                                            treated_logprob=-2.0)
    assert r["state"] == "degenerate_gap"
    assert r["direction_vs_reference"] == "unmoved"
    assert r["movement_sign"] == "unchanged"


def test_metric_cannot_be_computed_at_all_is_omitted_never_zero():
    """A metric with no usable inputs must be an absent key with a reason, never a fabricated `0.0`
    (docs/SEAMS.md rule 2) -- a `0.0` here would read as 'no movement', a different and false claim."""
    r = rm.reference_token_logprob_recovery(reference_logprob=None, baseline_logprob=None,
                                            treated_logprob=-1.0)
    assert r["state"] == "not_computable"
    assert "movement" not in r
    assert "gap_closed_fraction" not in r
    assert "baseline_value" not in r
    assert r["treated_value"] == -1.0    # what WAS observable is still reported
    fields = {o["field"] for o in r["omitted"]}
    assert "baseline_value" in fields
    assert "reference_value" in fields


def test_boolean_is_never_mistaken_for_a_logprob():
    r = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=True,
                                            treated_logprob=-1.0)
    assert r["state"] == "not_computable"
    assert "baseline_value" not in r


def test_suppression_reference_unknown_still_reports_raw_movement():
    """Metric 2's `reference_logprob` is optional and frequently unobservable in practice; the drop must
    still be measurable from the two candidate-side values alone."""
    r = rm.candidate_token_suppression(baseline_logprob=-1.0, treated_logprob=-4.0)
    assert r["metric"] == "candidate_token_suppression"
    assert r["state"] == "reference_unknown"
    assert r["movement"] == pytest.approx(-3.0)
    assert r["movement_sign"] == "decreased"
    assert "direction_vs_reference" not in r
    assert "gap_closed_fraction" not in r
    reasons = {o["field"] for o in r["omitted"]}
    assert {"direction_vs_reference", "gap", "gap_closed_fraction"} <= reasons


def test_suppression_with_known_reference_is_a_plain_movement_result():
    r = rm.candidate_token_suppression(baseline_logprob=-1.0, treated_logprob=-4.0, reference_logprob=-6.0)
    assert r["state"] == "measurable"
    assert r["direction_vs_reference"] == "toward_reference"


# ==================================================================================== metric 3 (sequence)

def test_sequence_nll_movement_normal_case():
    r = rm.sequence_nll_movement(reference_logprobs=[-0.1, -0.2, -0.1], baseline_logprobs=[-3, -3, -3],
                                 treated_logprobs=[-1, -1, -1])
    assert r["positions_total"] == 3
    assert r["positions_used"] == 3
    assert "positions_skipped" not in r
    assert r["baseline_value"] == pytest.approx(3.0)      # mean NLL = -mean(logprob)
    assert r["treated_value"] == pytest.approx(1.0)
    assert r["state"] == "measurable"
    assert r["direction_vs_reference"] == "toward_reference"


def test_sequence_nll_movement_mismatched_lengths_is_not_computable():
    r = rm.sequence_nll_movement(reference_logprobs=[-0.1, -0.2], baseline_logprobs=[-3],
                                 treated_logprobs=[-1, -1])
    assert r["state"] == "not_computable"
    assert "movement" not in r
    assert r["positions_total"] == 2


def test_sequence_nll_movement_skips_incomplete_positions_never_pads_with_zero():
    # position 1 is missing on the reference side -- it must be dropped from ALL three sums, not
    # treated as a 0.0 logprob (which would fabricate a very confident, wrong contribution).
    r = rm.sequence_nll_movement(reference_logprobs=[-0.1, None, -0.3],
                                 baseline_logprobs=[-3.0, -3.0, -3.0],
                                 treated_logprobs=[-1.0, -1.0, -1.0])
    assert r["positions_total"] == 3
    assert r["positions_used"] == 2
    assert r["positions_skipped"] == 1
    assert r["baseline_value"] == pytest.approx(3.0)
    assert r["reference_value"] == pytest.approx((0.1 + 0.3) / 2)


def test_sequence_nll_movement_all_positions_missing_is_not_computable():
    r = rm.sequence_nll_movement(reference_logprobs=[None, None], baseline_logprobs=[-1.0, -1.0],
                                 treated_logprobs=[-1.0, -1.0])
    assert r["state"] == "not_computable"
    assert r["positions_used"] == 0
    assert r["positions_skipped"] == 2


# ======================================================================================== metric 4 cells

def _cell(status, *, suite="target", case="c1", variant="base", seed=0):
    return {"suite": suite, "case": case, "variant": variant, "seed": seed, "status": status,
            "assertions": []}


def test_assertion_restoration_fail_to_pass_is_restored():
    r = rm.assertion_restoration(baseline_cell=_cell("fail", variant="base"),
                                 treated_cell=_cell("pass", variant="restored"))
    assert r["metric"] == "assertion_restoration"
    assert r["restoration_state"] == "restored"
    assert r["coordinates"]["baseline"] == ("target", "c1", "base", 0)
    assert r["coordinates"]["treated"] == ("target", "c1", "restored", 0)


def test_assertion_restoration_pass_to_fail_is_regressed_not_restored():
    r = rm.assertion_restoration(baseline_cell=_cell("pass"), treated_cell=_cell("fail"))
    assert r["restoration_state"] == "regressed"


def test_assertion_restoration_unchanged_states():
    still_pass = rm.assertion_restoration(baseline_cell=_cell("pass"), treated_cell=_cell("pass"))
    still_fail = rm.assertion_restoration(baseline_cell=_cell("fail"), treated_cell=_cell("fail"))
    assert still_pass["restoration_state"] == "unchanged_pass"
    assert still_fail["restoration_state"] == "unchanged_fail"


def test_assertion_restoration_error_side_is_not_measurable_not_a_fabricated_verdict():
    r = rm.assertion_restoration(baseline_cell=_cell("fail"), treated_cell=_cell("error"))
    assert r["state"] == "not_measurable"
    assert r["restoration_state"] == "not_measurable"
    assert r["omitted"][0]["field"] == "restoration_state"


def test_assertion_restoration_unscored_side_is_not_measurable():
    r = rm.assertion_restoration(baseline_cell=_cell("unscored"), treated_cell=_cell("pass"))
    assert r["restoration_state"] == "not_measurable"


def test_assertion_restoration_malformed_cell_is_not_computable_not_a_crash():
    r = rm.assertion_restoration(baseline_cell={"not": "a cell"}, treated_cell=_cell("pass"))
    assert r["state"] == "not_computable"
    assert r["coordinates"]["baseline"] == (None, None, None, None)


def test_assertion_restoration_non_mapping_cell_coordinate_is_none_not_a_crash():
    r = rm.assertion_restoration(baseline_cell="not even a dict", treated_cell=_cell("pass"))
    assert r["state"] == "not_computable"
    assert r["coordinates"]["baseline"] is None


# ================================================================================ metric 5 (structured IO)

def test_structured_output_validity_restored():
    r = rm.structured_output_validity_restoration(baseline_output="not json at all",
                                                   treated_output='{"a": 1}')
    assert r["metric"] == "structured_output_validity_restoration"
    assert r["baseline_valid"] is False
    assert r["treated_valid"] is True
    assert r["restoration_state"] == "restored"
    assert "baseline_invalid_reason" in r
    assert "treated_invalid_reason" not in r


def test_structured_output_validity_regressed():
    r = rm.structured_output_validity_restoration(baseline_output='{"a": 1}', treated_output="{broken")
    assert r["restoration_state"] == "regressed"


def test_structured_output_validity_unchanged_states():
    valid = rm.structured_output_validity_restoration(baseline_output='{"a": 1}', treated_output='{"a": 2}')
    invalid = rm.structured_output_validity_restoration(baseline_output="nope", treated_output="still nope")
    assert valid["restoration_state"] == "unchanged_valid"
    assert invalid["restoration_state"] == "unchanged_invalid"


def test_structured_output_validity_with_schema_reuses_structured_io_validation():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}
    # valid JSON, but does not satisfy the schema (wrong type) -- still "invalid" under this metric,
    # proving it reuses structured_io's schema check and not a bare json.loads().
    r = rm.structured_output_validity_restoration(baseline_output='{"a": "not-an-int"}',
                                                   treated_output='{"a": 7}', json_schema=schema)
    assert r["baseline_valid"] is False
    assert r["treated_valid"] is True
    assert r["restoration_state"] == "restored"


def test_structured_output_validity_bad_schema_itself_is_not_computable():
    r = rm.structured_output_validity_restoration(baseline_output="{}", treated_output="{}",
                                                   json_schema={"type": "not-a-real-type"})
    assert r["state"] == "not_computable"
    assert "restoration_state" not in r


def test_structured_output_validity_non_string_output_is_invalid_not_a_crash():
    r = rm.structured_output_validity_restoration(baseline_output=None, treated_output='{"a": 1}')
    assert r["baseline_valid"] is False
    assert r["restoration_state"] == "restored"


# ======================================================================================= metric 6 greedy

def test_greedy_suffix_match_restored():
    r = rm.greedy_suffix_match(reference_ids=[1, 2, 3], baseline_candidate_ids=[1, 9, 9],
                               treated_candidate_ids=[1, 2, 3])
    assert r["restoration_state"] == "restored"
    assert r["treated"]["exact_match"] is True
    assert r["baseline"]["exact_match"] is False
    assert r["movement"]["direction_vs_reference"] == "toward_reference"


def test_greedy_suffix_match_regressed():
    r = rm.greedy_suffix_match(reference_ids=[1, 2, 3], baseline_candidate_ids=[1, 2, 3],
                               treated_candidate_ids=[1, 2, 9])
    assert r["restoration_state"] == "regressed"


def test_greedy_suffix_match_unchanged_states():
    still_match = rm.greedy_suffix_match(reference_ids=[1, 2, 3], baseline_candidate_ids=[1, 2, 3],
                                         treated_candidate_ids=[1, 2, 3])
    still_no_match = rm.greedy_suffix_match(reference_ids=[1, 2, 3], baseline_candidate_ids=[9, 9, 9],
                                            treated_candidate_ids=[1, 8, 8])
    assert still_match["restoration_state"] == "unchanged_match"
    assert still_no_match["restoration_state"] == "unchanged_no_match"


def test_greedy_suffix_match_longer_candidate_with_matching_prefix_is_not_exact():
    r = rm.greedy_suffix_match(reference_ids=[1, 2, 3], treated_candidate_ids=[1, 2, 3, 4])
    assert r["treated"]["match_length"] == 3
    assert r["treated"]["match_fraction"] == 1.0
    assert r["treated"]["exact_match"] is False    # trailing extra tokens: not truly a reproduction


def test_greedy_suffix_match_baseline_unknown_state():
    r = rm.greedy_suffix_match(reference_ids=[1, 2, 3], treated_candidate_ids=[1, 2, 3])
    assert r["state"] == "baseline_unknown"
    assert r["treated"]["exact_match"] is True
    assert "baseline" not in r


def test_greedy_suffix_match_empty_reference_is_not_computable():
    r = rm.greedy_suffix_match(reference_ids=[], treated_candidate_ids=[1, 2])
    assert r["state"] == "not_computable"


# ============================================================================================= select_primary

def test_select_primary_returns_the_named_result_and_the_rest_separately():
    a = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0, treated_logprob=-2.0)
    b = rm.candidate_token_suppression(baseline_logprob=-1.0, treated_logprob=-4.0)
    picked = rm.select_primary({"reference_token_logprob_recovery": a, "candidate_token_suppression": b},
                               primary_metric="candidate_token_suppression")
    assert picked["state"] == "selected"
    assert picked["result"] is b
    assert "candidate_token_suppression" not in picked["other_metrics"]
    assert "reference_token_logprob_recovery" in picked["other_metrics"]


def test_select_primary_refuses_unknown_metric_name_rather_than_defaulting():
    """The central design requirement: nothing here silently falls back to a fixed metric like
    next-token argmax when the caller's requested primary metric is bogus."""
    picked = rm.select_primary({}, primary_metric="next_token_argmax")
    assert picked["state"] == "unknown_primary_metric"
    assert "result" not in picked


def test_select_primary_refuses_a_metric_not_actually_computed():
    a = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0, treated_logprob=-2.0)
    picked = rm.select_primary({"reference_token_logprob_recovery": a}, primary_metric="greedy_suffix_match")
    assert picked["state"] == "primary_metric_not_computed"
    assert "result" not in picked


def test_metric_kinds_matches_every_metric_field_this_module_emits():
    a = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0, treated_logprob=-2.0)
    b = rm.candidate_token_suppression(baseline_logprob=-1.0, treated_logprob=-4.0)
    c = rm.sequence_nll_movement(reference_logprobs=[-0.1], baseline_logprobs=[-3], treated_logprobs=[-1])
    d = rm.assertion_restoration(baseline_cell=_cell("fail"), treated_cell=_cell("pass"))
    e = rm.structured_output_validity_restoration(baseline_output="x", treated_output="{}")
    f = rm.greedy_suffix_match(reference_ids=[1], treated_candidate_ids=[1])
    for result in (a, b, c, d, e, f):
        assert result["metric"] in rm.METRIC_KINDS


# ================================================================================================ beat_control

def test_beat_control_reference_arm_beats_random_control():
    """Callers must be able to answer 'did the reference arm beat the random control' by reading fields
    already on the two `movement()` results -- without re-deriving a gap or a fraction."""
    treatment = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0,
                                                     treated_logprob=-2.0)          # big recovery
    control = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0,
                                                   treated_logprob=-4.9)            # barely moved
    out = rm.beat_control(treatment, control)
    assert out["state"] == "comparable"
    assert out["arm_beat_control"] is True
    assert out["gap_closed_fraction_margin"] > 0
    assert out["movement_margin"] == pytest.approx(treatment["movement"] - control["movement"])


def test_beat_control_control_arm_can_win_too():
    treatment = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0,
                                                     treated_logprob=-4.9)
    control = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0,
                                                   treated_logprob=-1.0)
    out = rm.beat_control(treatment, control)
    assert out["arm_beat_control"] is False


def test_beat_control_degenerate_gap_falls_back_to_movement_only_not_a_fake_verdict():
    treatment = rm.reference_token_logprob_recovery(reference_logprob=-2.0000000001, baseline_logprob=-2.0,
                                                     treated_logprob=-1.0)
    control = rm.reference_token_logprob_recovery(reference_logprob=-2.0000000001, baseline_logprob=-2.0,
                                                   treated_logprob=-1.5)
    out = rm.beat_control(treatment, control)
    assert out["state"] == "movement_only"
    assert "arm_beat_control" not in out
    assert out["movement_margin"] == pytest.approx(0.5)


def test_beat_control_mismatched_metrics_is_incomparable():
    a = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0, treated_logprob=-2.0)
    b = rm.candidate_token_suppression(baseline_logprob=-1.0, treated_logprob=-4.0)
    out = rm.beat_control(a, b)
    assert out["state"] == "incomparable_arms"


def test_beat_control_neither_side_computable_is_not_comparable():
    a = rm.reference_token_logprob_recovery(reference_logprob=None, baseline_logprob=None, treated_logprob=None)
    b = rm.reference_token_logprob_recovery(reference_logprob=None, baseline_logprob=None, treated_logprob=None)
    out = rm.beat_control(a, b)
    assert out["state"] == "not_comparable"


# ======================================================================== cross-metric: raw movement != assertion

def test_raw_numbers_can_move_while_the_assertion_still_fails():
    """The whole point of composable, non-argmax-only metrics: a continuous metric can show real
    movement toward the reference while the strict, discrete metrics (an assertion, or full greedy
    reproduction) still say no. Neither metric is wrong; they measure different things and must not be
    conflated into one verdict.
    """
    logprob_result = rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-8.0,
                                                          treated_logprob=-3.0)
    assert logprob_result["direction_vs_reference"] == "toward_reference"
    assert logprob_result["gap_closed_fraction"] > 0.5      # real, substantial movement

    suffix_result = rm.greedy_suffix_match(reference_ids=[10, 11, 12], baseline_candidate_ids=[99, 99, 99],
                                           treated_candidate_ids=[10, 11, 99])     # still one token off
    assert suffix_result["restoration_state"] == "unchanged_no_match"

    assertion_result = rm.assertion_restoration(baseline_cell=_cell("fail"), treated_cell=_cell("fail"))
    assert assertion_result["restoration_state"] == "unchanged_fail"

    # select_primary demonstrates the resolution: which of these "wins" for this case is a choice the
    # caller makes explicitly, never one this module makes for them.
    results = {"reference_token_logprob_recovery": logprob_result, "greedy_suffix_match": suffix_result,
              "assertion_restoration": assertion_result}
    by_logprob = rm.select_primary(results, primary_metric="reference_token_logprob_recovery")
    by_assertion = rm.select_primary(results, primary_metric="assertion_restoration")
    assert by_logprob["result"]["direction_vs_reference"] == "toward_reference"
    assert by_assertion["result"]["restoration_state"] == "unchanged_fail"


# =================================================================================== no causal vocabulary

_BANNED = ("caused", "because", "responsible for", "localiz")   # localize/localized/localization


def test_module_source_never_contains_causal_vocabulary():
    """This module describes MOVEMENT, not what produced it -- mirrors
    clozn.analysis.mechanistic_diff's own guard (test_mechanistic_diff.py) at the source-file level, so
    any string this module could ever emit into a caller's output is covered, not just one sampled run.

    The module docstring legitimately NAMES the banned words, in quotes, to state the rule against using
    them. This test removes exactly those one-time quoted mentions, then asserts the bare words appear
    nowhere else in the file.
    """
    path = os.path.join(REPO_ROOT, "clozn", "analysis", "restoration_metrics.py")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    quoted_mentions = ('"caused"', '"because"', '"responsible for"', '"localized"')
    remainder = text
    for mention in quoted_mentions:
        assert mention in remainder, f"expected the module docstring to name {mention} as banned vocabulary"
        remainder = remainder.replace(mention, "", 1)
    lowered = remainder.lower()
    for word in _BANNED:
        assert word not in lowered, f"causal vocabulary {word!r} used outside its one quoted mention"


def test_computed_results_never_contain_causal_vocabulary():
    """A second, complementary guard: scan actual RETURNED data (not just source text) for the same
    vocabulary, the way test_mechanistic_diff.py scans one produced document."""
    blob = {
        "recovery": rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0,
                                                         treated_logprob=-2.0),
        "suppression": rm.candidate_token_suppression(baseline_logprob=-1.0, treated_logprob=-4.0),
        "sequence": rm.sequence_nll_movement(reference_logprobs=[-0.1], baseline_logprobs=[-3],
                                             treated_logprobs=[-1]),
        "assertion": rm.assertion_restoration(baseline_cell=_cell("fail"), treated_cell=_cell("error")),
        "structured": rm.structured_output_validity_restoration(baseline_output="nope",
                                                                 treated_output="{broken"),
        "greedy": rm.greedy_suffix_match(reference_ids=[1, 2], treated_candidate_ids=[1]),
        "primary_unknown": rm.select_primary({}, primary_metric="bogus"),
        "beat_control_mismatch": rm.beat_control(
            rm.reference_token_logprob_recovery(reference_logprob=-0.1, baseline_logprob=-5.0, treated_logprob=-2.0),
            rm.candidate_token_suppression(baseline_logprob=-1.0, treated_logprob=-4.0)),
    }
    text = json.dumps(blob, default=str).lower()
    for word in _BANNED:
        assert word not in text, f"causal vocabulary {word!r} leaked into a computed result"
