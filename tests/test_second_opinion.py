"""Model-free coverage for `clozn.runs.second_opinion` (E4) -- the pure-ish comparison logic behind
`POST /runs/<id>/second-opinion`. No engine, no GPU: `run_second_opinion_arm` is exercised against a fake
selection whose `.sub.chat` is a plain Python callable, mirroring `quant_check.FakeEngine`'s own
model-free discipline elsewhere in this codebase.

Route-layer coverage (worker resolution, HTTP status codes, the managed-router gate) lives in
tests/test_second_opinion_route.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import clozn.runs.store as runlog
from clozn import schemas
from clozn.runs import second_opinion as so


# ============================================================================================ fixtures

@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _selection(model_id="beta", *, chat, template_fingerprint="tmpl-b", model_sha256="b" * 64,
              context_size=8192, worker_id="gen-b", worker_generation=1, last_finish_reason=None):
    sub = SimpleNamespace(chat=chat)
    if last_finish_reason is not None:
        sub.last_finish_reason = lambda: last_finish_reason
    runtime_key = {
        "template_fingerprint": template_fingerprint, "gguf_artifact_sha256": model_sha256,
        "engine_build": "build-b", "backend": "cpu", "context_size": context_size,
    }
    worker_identity = {"worker_id": worker_id, "worker_generation": worker_generation}
    return SimpleNamespace(model_id=model_id, sub=sub, runtime_key=runtime_key,
                           worker_identity=worker_identity)


def _run(**overrides):
    values = {
        "source": "engine_chat", "client": "studio", "model": "alpha", "substrate": "engine",
        "messages": [{"role": "user", "content": "What year was the bridge built?"}],
        "response": "The bridge was built in 1920.", "finish_reason": "stop",
        "identity": {"model_sha256": "a" * 64, "template_fingerprint": "tmpl-a", "engine_build": "build-a"},
    }
    values.update(overrides)
    run_id = runlog.record(**values)
    assert run_id
    return runlog.get_run(run_id)


# ==================================================================================== build_anchor_arm

def test_anchor_arm_ok_never_touches_a_worker(iso):
    run = _run()
    arm = so.build_anchor_arm(run)
    assert arm["role"] == "anchor"
    assert arm["status"] == "ok"
    assert arm["response_text"] == "The bridge was built in 1920."
    assert arm["model_id"] == "alpha"
    assert arm["worker_identity"]["template_fingerprint"] == "tmpl-a"
    assert arm["finish_reason"] == "stop"
    assert "latency_ms" in arm   # timing.duration_ms is always set by runlog.record


def test_anchor_arm_empty_response(iso):
    run = _run(response="")
    arm = so.build_anchor_arm(run)
    assert arm["status"] == "empty"
    assert "response_text" not in arm


def test_anchor_arm_redacted(iso):
    run = _run()
    run = dict(run)
    run["redaction"] = {"status": "redacted"}
    arm = so.build_anchor_arm(run)
    assert arm["status"] == "redacted"
    assert "response_text" not in arm


def test_anchor_arm_unavailable_when_response_missing(iso):
    run = _run()
    run = dict(run)
    run["response"] = None
    arm = so.build_anchor_arm(run)
    assert arm["status"] == "unavailable"
    assert "response_text" not in arm


# =========================================================================== run_second_opinion_arm

def test_second_opinion_arm_success_records_latency_identity_and_tokens(iso):
    def fake_chat(messages, max_new=256, sample=True, trace_out=None, **_kw):
        if trace_out is not None:
            trace_out.extend([{"id": 1}, {"id": 2}, {"id": 3}])
        return "Construction finished in 1920."

    selection = _selection(chat=fake_chat, last_finish_reason=lambda: None)
    arm = so.run_second_opinion_arm(
        selection, requested_model_id="beta", messages=[{"role": "user", "content": "hi"}], budget=64)

    assert arm["status"] == "ok"
    assert arm["response_text"] == "Construction finished in 1920."
    assert arm["model_id"] == "beta"
    assert arm["requested_model_id"] == "beta"
    assert arm["worker_identity"]["template_fingerprint"] == "tmpl-b"
    assert arm["worker_identity"]["context_size"] == 8192
    assert arm["generated_tokens"] == 3
    assert arm["latency_ms"] >= 0
    assert "refusal" not in arm


def test_second_opinion_arm_generation_error_never_raises(iso):
    def failing_chat(*_a, **_k):
        raise RuntimeError("worker socket closed")

    selection = _selection(chat=failing_chat)
    arm = so.run_second_opinion_arm(
        selection, requested_model_id="beta", messages=[{"role": "user", "content": "hi"}], budget=64)

    assert arm["status"] == "generation_error"
    assert arm["refusal"]["code"] == "generation_error"
    assert "worker socket closed" in arm["refusal"]["message"]
    assert "response_text" not in arm


def test_second_opinion_arm_refuses_with_no_delivered_messages(iso):
    calls = []

    def chat_should_never_run(*_a, **_k):
        calls.append(1)
        return "should not happen"

    selection = _selection(chat=chat_should_never_run)
    arm = so.run_second_opinion_arm(
        selection, requested_model_id="beta", messages=[], budget=64)

    assert arm["status"] == "refused"
    assert arm["refusal"]["code"] == "no_delivered_messages"
    assert not calls   # chat() must never be called on an empty delivered-input request


def test_second_opinion_arm_tolerates_a_fake_chat_without_trace_out(iso):
    """Mirrors clozn.replay.replay.replay's own progressive-degrade discipline: a substrate whose
    chat() predates trace_out must still produce a full, honest arm_b (just no generated_tokens)."""
    def narrow_chat(messages, max_new=256, sample=True):
        return "a reply"

    selection = _selection(chat=narrow_chat)
    arm = so.run_second_opinion_arm(
        selection, requested_model_id="beta", messages=[{"role": "user", "content": "hi"}], budget=64)

    assert arm["status"] == "ok"
    assert arm["response_text"] == "a reply"
    assert "generated_tokens" not in arm


# ========================================================================================== compatibility

def test_chat_template_compat_differs(iso):
    run = _run()
    arm_a = so.build_anchor_arm(run)
    arm_b = {"worker_identity": {"template_fingerprint": "tmpl-b"}}
    compat = so._template_compat(arm_a, arm_b)
    assert compat["state"] == "differs"
    assert "caveat" in compat


def test_chat_template_compat_same(iso):
    run = _run(identity={"model_sha256": "a" * 64, "template_fingerprint": "shared-tmpl"})
    arm_a = so.build_anchor_arm(run)
    arm_b = {"worker_identity": {"template_fingerprint": "shared-tmpl"}}
    compat = so._template_compat(arm_a, arm_b)
    assert compat["state"] == "same"
    assert "caveat" not in compat


def test_chat_template_compat_unknown_when_either_side_missing(iso):
    run = _run(identity={})
    arm_a = so.build_anchor_arm(run)
    compat = so._template_compat(arm_a, {})
    assert compat["state"] == "unknown"


def test_context_limit_exceeds_estimate(iso):
    run = _run()
    run = dict(run)
    run["context_receipt"] = {"limits": {"prompt_tokens": 9000}}
    arm_a = so.build_anchor_arm(run)
    arm_b = {"worker_identity": {"context_size": 4096}}
    compat = so._context_limit_compat(arm_a, arm_b)
    assert compat["state"] == "exceeds_estimate"
    assert compat["arm_a_prompt_tokens_estimate"] == 9000
    assert compat["arm_b_context_window_tokens"] == 4096


def test_context_limit_within_estimate(iso):
    run = _run()
    run = dict(run)
    run["context_receipt"] = {"limits": {"prompt_tokens": 100}}
    arm_a = so.build_anchor_arm(run)
    arm_b = {"worker_identity": {"context_size": 4096}}
    compat = so._context_limit_compat(arm_a, arm_b)
    assert compat["state"] == "within_estimate"


def test_tools_schema_none_used_by_default(iso):
    run = _run()
    assert so._tools_schema_compat(run) == {"state": "none_used"}


def test_tools_schema_used_not_replayed(iso):
    run = _run(output_contract={"mode": "tools"})
    compat = so._tools_schema_compat(run)
    assert compat["state"] == "used_not_replayed"
    assert compat["requested_mode"] == "tools"
    assert "caveat" in compat


def test_qualified_evidence_is_always_anchor_only(iso):
    # Not parametrized on run state: this is a structural fact of the v1 milestone, not a per-request
    # measurement (see clozn.runs.second_opinion's own module docstring).
    document = so.build_second_opinion(
        _run(), _selection(chat=lambda *_a, **_k: "ok"), requested_model_id="beta")
    assert document["compatibility"]["qualified_evidence"]["state"] == "anchor_only"


# ================================================================================== build_second_opinion

def test_build_second_opinion_both_arms_ok_produces_a_valid_comparison(iso):
    def fake_chat(messages, max_new=256, sample=True, trace_out=None, **_kw):
        return "Construction on the bridge finished in 1920."

    run = _run()
    selection = _selection(chat=fake_chat)
    document = so.build_second_opinion(run, selection, requested_model_id="beta")

    schemas.validate(document)   # build_second_opinion already validates; re-asserted here explicitly
    assert document["run_id"] == run["id"]
    assert document["arm_a"]["status"] == "ok"
    assert document["arm_b"]["status"] == "ok"
    assert "comparison" in document
    assert document["comparison"]["agreement"]["method"] == "lexical_overlap_heuristic"
    assert 0 <= document["comparison"]["agreement"]["lexical_difference_percent"] <= 100
    assert document["delivered_input"]["identical_across_arms"] is True
    assert document["delivered_input"]["message_count"] == 1


def test_build_second_opinion_arm_b_failure_never_drops_arm_a(iso):
    """The core honesty requirement, proved at the document level: arm_b's failure never removes or
    degrades arm_a's already-recorded evidence, and the document as a whole still validates."""
    def failing_chat(*_a, **_k):
        raise RuntimeError("connection reset")

    run = _run()
    selection = _selection(chat=failing_chat)
    document = so.build_second_opinion(run, selection, requested_model_id="beta")

    schemas.validate(document)
    assert document["arm_a"]["status"] == "ok"
    assert document["arm_a"]["response_text"] == run["response"]
    assert document["arm_b"]["status"] == "generation_error"
    assert "comparison" not in document   # omitted, never a fabricated comparison against nothing


def test_build_second_opinion_omits_comparison_when_anchor_has_no_text(iso):
    run = _run(response="")
    selection = _selection(chat=lambda *_a, **_k: "a fresh answer")
    document = so.build_second_opinion(run, selection, requested_model_id="beta")

    schemas.validate(document)
    assert document["arm_a"]["status"] == "empty"
    assert document["arm_b"]["status"] == "ok"
    assert "comparison" not in document


def test_build_second_opinion_never_carries_a_token_probability_anywhere(iso):
    """A structural, load-bearing regression: no key anywhere in the document is a per-token
    probability/logprob (owner's spec, bolded: never present token probabilities from different
    models as calibrated confidence)."""
    run = _run()
    selection = _selection(chat=lambda *_a, **_k: "a fresh answer")
    document = so.build_second_opinion(run, selection, requested_model_id="beta")

    def _walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert "logprob" not in key.lower() and "probability" not in key.lower(), (
                    f"found a probability-shaped field {key!r} in the second-opinion document")
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(document)
