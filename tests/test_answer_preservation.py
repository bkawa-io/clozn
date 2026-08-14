"""Adversarial, model-free tests for exact recorded-answer preservation."""
from __future__ import annotations

import copy

import pytest

from clozn.runs.answer_preservation import (
    assess_exact_eligibility,
    classify_reference_match,
)
from clozn.runs.minimal_context import (
    MinimalContextError,
    run_minimal_context_search,
)
from clozn.server import app as cs


CONTRACT = {
    "decode_mode": "greedy",
    "sampling": None,
    "max_new": 4,
    "stop": [],
    "expected_termination": {"reason": "eos", "reason_raw": "eos"},
}


@pytest.mark.parametrize(
    ("actual", "termination", "expected", "kind"),
    [
        ([10, 11, 12], {"kind": "eos"}, {"reason": "eos", "reason_raw": "eos"}, None),
        ([99, 11, 12], {"kind": "eos"}, {"reason": "eos", "reason_raw": "eos"}, "token_mismatch"),
        ([10, 99, 12], {"kind": "eos"}, {"reason": "eos", "reason_raw": "eos"}, "token_mismatch"),
        ([10, 11], {"kind": "eos"}, {"reason": "eos", "reason_raw": "eos"}, "early_termination"),
        ([10, 11, 12, 13], {"kind": "eos"}, {"reason": "eos", "reason_raw": "eos"}, "extra_token_after_reference"),
        ([10, 11, 12], {"kind": "length"}, {"reason": "eos", "reason_raw": "eos"}, "termination_mismatch"),
        ([10, 11, 12], {"kind": "steps_exhausted"}, {"reason": "max_tokens", "reason_raw": "max_tokens"}, None),
    ],
)
def test_reference_match_classifier_distinguishes_token_and_boundary_failures(
    actual, termination, expected, kind
):
    result = classify_reference_match(
        [10, 11, 12], actual, termination=termination,
        expected_termination=expected, max_new=4,
    )
    assert result["status"] == ("matched" if kind is None else "diverged")
    assert result["divergence_kind"] == kind


def _probe_substrate(monkeypatch, generated, *, termination="eos", finish="stop", diverged=False, at=-1):
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = object()
    sub.steer = None

    monkeypatch.setattr(cs, "_engine_tmpl", lambda engine, messages, **kwargs: "PROMPT")

    def complete(engine, prompt, max_tokens, kw, *, sample=None, usage_out=None, stop=None):
        if usage_out is not None:
            usage_out["termination"] = {"kind": termination}
        steps = [{"token_id": token, "piece": str(token)} for token in generated]
        return "".join(str(token) for token in generated), steps, finish, (diverged, at)

    monkeypatch.setattr(cs, "_engine_complete_traced", complete)
    return sub


def test_low_level_probe_is_non_journaling_and_uses_explicit_contract(monkeypatch):
    sub = _probe_substrate(monkeypatch, [10, 11, 12])
    before = dict(sub.__dict__)
    result = sub.probe_reference_match(
        [{"role": "user", "content": "hello"}], [10, 11, 12],
        generation_contract=CONTRACT,
        explicit_conditions={"steer_strengths": {"warm": 0.9}},
    )
    assert result["status"] == "matched"
    assert result["termination_match"] is True
    assert sub.__dict__ == before


def _identity(template="b" * 32):
    return {
        "model_sha256": "a" * 64,
        "template_fingerprint": template,
        "engine_build": "test-engine",
        "context_size": 4096,
        "backend": "cpu",
        "white_box_flags": {"sae": False, "jlens": False, "attn_knockout": False},
    }


def _eligible_run():
    return {
        "id": "run-exact",
        "model": "fixture",
        "response": "abc",
        "messages": [{"role": "system", "content": "context"}],
        "trace": {
            "steps": [
                {"token_id": 10, "piece": "a"},
                {"token_id": 11, "piece": "b"},
                {"token_id": 12, "piece": "c"},
            ]
        },
        "generation_contract": copy.deepcopy(CONTRACT),
        "identity": _identity(),
        "meta": {"n_ctx": 4096, "device": "cpu"},
    }


class _CurrentRuntime:
    def __init__(self, template="b" * 32):
        self.template = template

    def identity_meta(self):
        return _identity(self.template)

    def run_meta(self):
        return {"n_ctx": 4096, "device": "cpu"}


def test_eligibility_rejects_missing_ids_retokenization_and_identity_drift():
    run = _eligible_run()
    missing = copy.deepcopy(run)
    missing["trace"] = {"steps": [{"piece": "abc"}]}
    assert "missing_exact_recorded_token_ids" in assess_exact_eligibility(
        missing, _CurrentRuntime()
    )["reasons"]

    retokenized = _eligible_run()
    retokenized["trace"]["retokenized"] = True
    assert "retokenized_continuation" in assess_exact_eligibility(
        retokenized, _CurrentRuntime()
    )["reasons"]

    mismatch = assess_exact_eligibility(run, _CurrentRuntime("c" * 32))
    assert mismatch["eligible"] is False
    assert "template_mismatch" in mismatch["reasons"]


def _probe_record(removed, status):
    return {
        "schema_version": "clozn.reference-match-probe.v1",
        "probe_id": "rmp_" + ("a" * 24),
        "run_id": "run-exact",
        "removed_source_ids": list(removed),
        "exact_removed_ranges": [],
        "basis_digest": "0" * 64,
        "intervened_context_digest": "1" * 64,
        "reference_token_ids_sha256": "2" * 64,
        "reference_token_count": 3,
        "generation_contract": CONTRACT,
        "result": {"status": status},
        "provenance": "direct_generation_probe",
    }


def test_exact_minimal_context_accepts_only_direct_probe_evidence():
    def measure(removed):
        return _probe_record(removed, "matched" if set(removed) == {"A"} else "diverged")

    result = run_minimal_context_search(
        ["A", "B"], measure, tolerance_nats=0.0,
        search_probe_budget=10, certification_probe_budget=10,
        candidate_retained_source_sets=[["B"]],
        preservation={"kind": "exact_recorded_output", "target": "whole_recorded_continuation"},
    )
    assert result["candidate"]["result_status"] == "matched"
    assert result["candidate"]["provenance"] == "direct_generation_probe"
    assert result["certificate"]["kind"] == "exact_minimum"
    assert "delta_nats" not in result["candidate"]


def test_likelihood_evidence_cannot_fill_exact_proof_rows():
    with pytest.raises(MinimalContextError, match="direct exact evidence"):
        run_minimal_context_search(
            ["A", "B"], lambda removed: {"experiment_id": "x", "removed_source_ids": list(removed),
                                             "delta_nats": 0.0, "provenance": "measured"},
            tolerance_nats=0.0, search_probe_budget=1, certification_probe_budget=1,
            preservation={"kind": "exact_recorded_output", "target": "whole_recorded_continuation"},
        )
