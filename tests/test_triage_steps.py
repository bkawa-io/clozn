"""Unit tests for clozn.triage.steps: model-free identity diff (step 1) and context diff (step 2)."""
from __future__ import annotations

from clozn.triage.status import STATES
from clozn.triage.steps import context_diff_steps, identity_diff_steps


def _by_kind(steps):
    return {s["kind"]: s for s in steps}


# ============================================================================================= step 1 ===

def test_identical_identity_matches_on_every_scalar_field():
    identity = {"model_sha256": "a" * 64, "template_fingerprint": "b" * 16,
                "engine_build": "eng-1", "clozn_version": "1.2.3"}
    baseline = {"identity": dict(identity)}
    candidate = {"identity": dict(identity)}
    steps = _by_kind(identity_diff_steps(baseline, candidate))
    for kind in ("identity_diff:model", "identity_diff:template",
                "identity_diff:engine_build", "identity_diff:clozn_version"):
        assert steps[kind]["status"] == "matched", kind


def test_different_model_sha_is_mismatched():
    baseline = {"identity": {"model_sha256": "a" * 64}}
    candidate = {"identity": {"model_sha256": "b" * 64}}
    step = _by_kind(identity_diff_steps(baseline, candidate))["identity_diff:model"]
    assert step["status"] == "mismatched"
    assert step["observations"][0]["baseline"] == "a" * 64
    assert step["observations"][0]["candidate"] == "b" * 64


def test_missing_on_one_side_is_inconclusive_not_a_guess():
    baseline = {"identity": {"model_sha256": "a" * 64}}
    candidate = {"identity": {}}
    step = _by_kind(identity_diff_steps(baseline, candidate))["identity_diff:model"]
    assert step["status"] == "inconclusive"
    assert "candidate" in step["reason"]


def test_missing_on_both_sides_is_not_run():
    baseline = {"identity": {}}
    candidate = {"identity": {}}
    step = _by_kind(identity_diff_steps(baseline, candidate))["identity_diff:engine_build"]
    assert step["status"] == "not_run"


def test_tokenizer_is_always_an_explicit_not_run_with_a_stated_reason():
    step = _by_kind(identity_diff_steps({"identity": {}}, {"identity": {}}))["identity_diff:tokenizer"]
    assert step["status"] == "not_run"
    assert "template_fingerprint" in step["reason"]


def test_template_step_carries_the_tokenizer_conflation_caveat():
    baseline = {"identity": {"template_fingerprint": "a" * 16}}
    candidate = {"identity": {"template_fingerprint": "b" * 16}}
    step = _by_kind(identity_diff_steps(baseline, candidate))["identity_diff:template"]
    assert step["status"] == "mismatched"
    assert step["caveats"], "the standing tokenizer/template conflation caveat must be visible on the artifact"
    assert "tokenizer" in step["caveats"][0]


def test_ext_namespaces_are_diffed_generically():
    baseline = {"identity": {"ext": {"adapter": {"sha256": "a" * 64}}}}
    candidate = {"identity": {"ext": {"adapter": {"sha256": "b" * 64}}}}
    steps = _by_kind(identity_diff_steps(baseline, candidate))
    assert "identity_diff:ext.adapter" in steps
    assert steps["identity_diff:ext.adapter"]["status"] == "mismatched"


def test_no_ext_namespace_present_produces_no_ext_steps():
    steps = identity_diff_steps({"identity": {}}, {"identity": {}})
    assert not [s for s in steps if s["kind"].startswith("identity_diff:ext.")]


def test_ext_namespace_only_on_one_side_is_inconclusive():
    baseline = {"identity": {"ext": {"machine": {"cpu": "x"}}}}
    candidate = {"identity": {}}
    step = _by_kind(identity_diff_steps(baseline, candidate))["identity_diff:ext.machine"]
    assert step["status"] == "inconclusive"


def test_every_identity_step_status_is_in_the_controlled_enum():
    baseline = {"identity": {"model_sha256": "a" * 64, "ext": {"adapter": {"x": 1}}}}
    candidate = {"identity": {"model_sha256": "b" * 64}}
    for step in identity_diff_steps(baseline, candidate):
        assert step["status"] in STATES
        assert step["cost"]["model_runs"] == 0, "identity diff is model-free"


# ============================================================================================= step 2 ===

def _run_with_context_receipt(*, delivered=None, assembled=None, final_prompt=None):
    receipt = {}
    if delivered is not None:
        receipt["delivered"] = {"messages": delivered}
    if assembled is not None or final_prompt is not None:
        receipt["survived"] = {}
        if assembled is not None:
            receipt["survived"]["assembled_messages"] = assembled
        if final_prompt is not None:
            receipt["survived"]["final_prompt"] = final_prompt
    return {"context_receipt": receipt}


def test_identical_rendered_prompt_matches():
    baseline = _run_with_context_receipt(final_prompt="hello world")
    candidate = _run_with_context_receipt(final_prompt="hello world")
    step = _by_kind(context_diff_steps(baseline, candidate))["context_diff:rendered_prompt"]
    assert step["status"] == "matched"
    assert step["observations"][0]["baseline_sha256"] == step["observations"][0]["candidate_sha256"]


def test_different_rendered_prompt_mismatches_and_never_embeds_raw_text():
    baseline = _run_with_context_receipt(final_prompt="hello world")
    candidate = _run_with_context_receipt(final_prompt="hello world, extra tokens appended here")
    step = _by_kind(context_diff_steps(baseline, candidate))["context_diff:rendered_prompt"]
    assert step["status"] == "mismatched"
    dumped = str(step)
    assert "hello world" not in dumped, "raw prompt content must never be embedded in the artifact"
    assert step["observations"][0]["baseline_length"] == len("hello world")


def test_assembled_messages_digest_is_order_and_content_sensitive():
    msgs_a = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    msgs_b = [{"role": "user", "content": "u"}, {"role": "system", "content": "s"}]
    baseline = _run_with_context_receipt(assembled=msgs_a)
    candidate = _run_with_context_receipt(assembled=msgs_b)
    step = _by_kind(context_diff_steps(baseline, candidate))["context_diff:assembled_messages"]
    assert step["status"] == "mismatched"


def test_missing_context_receipt_on_both_sides_is_not_run():
    step = _by_kind(context_diff_steps({}, {}))["context_diff:rendered_prompt"]
    assert step["status"] == "not_run"


def test_omissions_and_special_tokens_are_always_explicit_not_run_pending_feature_06():
    steps = _by_kind(context_diff_steps({}, {}))
    for kind in ("context_diff:omissions", "context_diff:special_tokens"):
        assert steps[kind]["status"] == "not_run"
        assert "feature 06" in steps[kind]["reason"]


def test_every_context_step_status_is_in_the_controlled_enum():
    baseline = _run_with_context_receipt(final_prompt="a", assembled=[{"role": "user", "content": "a"}])
    candidate = _run_with_context_receipt(final_prompt="b", assembled=[{"role": "user", "content": "b"}])
    for step in context_diff_steps(baseline, candidate):
        assert step["status"] in STATES
        assert step["cost"]["model_runs"] == 0, "context diff is model-free"
