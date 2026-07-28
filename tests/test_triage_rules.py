"""Unit tests for clozn.triage.rules.classify: the rule engine, per notes/agent_roadmap/
05-automatic-regression-triage.md's own fixture list ("Build synthetic fixtures for...").

The one rule that matters, confirmed by the feature owner: `classification` is `"undetermined"` UNLESS
exactly one step reached `causally_supported`, in which case it is that step's own `kind`, verbatim.
"""
from __future__ import annotations

import pytest

from clozn.triage.rules import classify


def _step(kind, status, **extra):
    step = {"kind": kind, "status": status, "inputs": {}, "observations": [],
           "artifact_refs": [], "cost": {"model_runs": 0}}
    step.update(extra)
    return step


def test_no_steps_is_undetermined_and_not_run():
    summary = classify([])
    assert summary["classification"] == "undetermined"
    assert summary["confidence_basis"] == "not_run"
    assert "no steps were executed" in summary["caveats"]


def test_only_model_hash_changes_stays_undetermined_without_a_causal_step():
    """'Only model hash changes' (spec fixture): a mismatch alone is `observed`, never itself proof --
    classification must not name a cause without an intervention."""
    steps = [
        _step("identity_diff:model", "mismatched"),
        _step("identity_diff:template", "matched"),
        _step("context_diff:rendered_prompt", "matched"),
    ]
    summary = classify(steps)
    assert summary["classification"] == "undetermined"
    assert summary["confidence_basis"] == "observed"
    assert summary["observed"] == ["identity_diff:model"]
    assert summary["eliminated"] == ["context_diff:rendered_prompt", "identity_diff:template"]
    assert any("identity_diff:model" in c for c in summary["caveats"])


def test_all_matched_is_undetermined_with_eliminated_confidence_basis():
    steps = [_step("identity_diff:model", "matched"), _step("identity_diff:template", "matched")]
    summary = classify(steps)
    assert summary["classification"] == "undetermined"
    assert summary["confidence_basis"] == "eliminated"
    assert summary["eliminated"] == ["identity_diff:model", "identity_diff:template"]
    assert summary["observed"] == []


def test_template_change_reproduces_and_reverses_failure_names_the_causal_step():
    """'Template change reproduces and reverses failure' (spec fixture), written against a synthetic
    already-recorded intervention step -- exactly what the spec's own fixture list asks for, since this
    build does not execute controlled replays itself."""
    steps = [
        _step("identity_diff:model", "matched"),
        _step("identity_diff:template", "mismatched",
             caveats=["template_fingerprint hashes tokenizer and template together"]),
        _step("template_swap", "causally_supported", cost={"model_runs": 2}),
    ]
    summary = classify(steps)
    assert summary["classification"] == "template_swap"
    assert summary["confidence_basis"] == "causally_supported"
    assert summary["causally_supported"] == ["template_swap"]
    # the step-level caveat must survive into the artifact's summary, not just live on the step.
    assert any("tokenizer" in c for c in summary["caveats"])


def test_multiple_entangled_causally_supported_steps_remain_undetermined():
    """'Multiple entangled changes remain inconclusive' (spec fixture): even two *causally_supported*
    findings must not be allowed to silently pick a winner."""
    steps = [
        _step("template_swap", "causally_supported"),
        _step("sampling_swap", "causally_supported"),
    ]
    summary = classify(steps)
    assert summary["classification"] == "undetermined"
    assert summary["confidence_basis"] == "causally_supported"
    assert set(summary["causally_supported"]) == {"template_swap", "sampling_swap"}
    assert any("multiple steps reached causally_supported" in c for c in summary["caveats"])


def test_context_truncation_removes_needed_instruction_is_observed_not_causal():
    steps = [_step("context_diff:assembled_messages", "mismatched"),
             _step("identity_diff:model", "matched")]
    summary = classify(steps)
    assert summary["classification"] == "undetermined"
    assert summary["confidence_basis"] == "observed"


def test_adapter_accepted_but_no_op_reads_as_matched_execution_fingerprint():
    """An adapter that loaded but made no measurable difference is an ELIMINATED hypothesis at the
    identity-diff layer (its execution fingerprint matched base) -- not proof of anything else."""
    steps = [_step("identity_diff:ext.adapter", "matched")]
    summary = classify(steps)
    assert summary["classification"] == "undetermined"
    assert summary["eliminated"] == ["identity_diff:ext.adapter"]


def test_correlated_evidence_never_promotes_itself_to_a_classification():
    steps = [_step("context_diff:assembled_messages", "correlated")]
    summary = classify(steps)
    assert summary["classification"] == "undetermined"
    assert summary["confidence_basis"] == "correlated"
    assert any("correlated" in c for c in summary["caveats"])


def test_deep_step_unsupported_on_an_unqualified_model_is_visible():
    steps = [_step("internal_localization", "unsupported", reason="model has no qualified J-lens")]
    summary = classify(steps)
    assert summary["unsupported"] == ["internal_localization"]
    assert summary["classification"] == "undetermined"


def test_precedence_causally_supported_beats_everything_else():
    steps = [
        _step("a", "mismatched"), _step("b", "correlated"), _step("c", "inconclusive"),
        _step("d", "causally_supported"),
    ]
    summary = classify(steps)
    assert summary["classification"] == "d"
    assert summary["confidence_basis"] == "causally_supported"


def test_a_step_with_an_invalid_status_is_refused_not_silently_classified():
    with pytest.raises(ValueError):
        classify([_step("x", "root_cause")])


def test_caveats_are_deduplicated_across_steps():
    steps = [
        _step("identity_diff:template", "mismatched", caveats=["same caveat"]),
        _step("identity_diff:ext.adapter", "mismatched", caveats=["same caveat"]),
    ]
    summary = classify(steps)
    assert summary["caveats"].count("same caveat") == 1
