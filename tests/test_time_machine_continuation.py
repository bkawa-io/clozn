"""Model-free contract tests for ADR-010 exact appended-turn continuation."""
from __future__ import annotations

import hashlib
import json
import struct

import pytest

from clozn import schemas
from clozn.replay.time_machine_continuation import (
    WORKER_CONTINUE_ENDPOINT,
    AppendDerivationError,
    ClosedRequestError,
    build_worker_request,
    derive_append_tokens,
    orchestrate_continuation,
    parse_continuation_request,
    sampler_config_sha256,
    sampler_state_sha256,
    token_ids_sha256,
)


def _sha(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


HISTORY = [1, 2, 3]
TOKENIZER = "a" * 64
SETTINGS = "b" * 64
MODEL = "c" * 64
RUNTIME = "d" * 64
TEMPLATE = "e" * 16
PAYLOAD = "f" * 64
SAMPLER_CONFIG = "1" * 64
SAMPLER_STATE = "2" * 64


def _source():
    return {
        "status": "resolved",
        "source_run_id": "run_source",
        "source_turn": 0,
        "resolution": "exact_latest_run",
    }


def _checkpoint():
    return {
        "status": "available",
        "provenance": "live_worker_checkpoint",
        "capture_regime": "organic_live_kv",
        "restart_safe": False,
        "source_run_id": "run_source",
        "checkpoint_reference_id": "checkpoint-ref-1",
        "checkpoint_id": "checkpoint-1",
        "source_worker_generation_id": "generation-1",
        "executing_worker_generation_id": "generation-1",
        "prompt_tokens": 2,
        "n_past": len(HISTORY),
        "token_history_sha256": token_ids_sha256(HISTORY),
        "payload_sha256": PAYLOAD,
    }


def _identity():
    return {
        "status": "matched",
        "source_runtime_key_sha256": RUNTIME,
        "selected_runtime_key_sha256": RUNTIME,
        "model_sha256": MODEL,
        "tokenizer_sha256": TOKENIZER,
        "template_fingerprint": TEMPLATE,
        "generation_settings_sha256": SETTINGS,
        "engine_build": "fixture-build",
        "context_size": 128,
        "backend": "cpu",
        "adapter": {
            "present": False,
            "identity_sha256": None,
            "artifact_sha256": None,
            "scale": None,
        },
        "worker": {
            "worker_id": "worker-1",
            "worker_generation_id": "generation-1",
            "protocol_version": "1.0",
        },
        "checks": {
            "runtime": True,
            "model": True,
            "tokenizer": True,
            "template": True,
            "adapter": True,
            "settings": True,
            "checkpoint": True,
        },
    }


def _sampler():
    return {
        "status": "preserved",
        "mode": "sample",
        "source": "checkpoint",
        "config_sha256": SAMPLER_CONFIG,
        "state_sha256": SAMPLER_STATE,
        "rng_draws_before_append": 7,
    }


def _render(_request):
    return {
        "full_render_token_ids": HISTORY + [9, 10],
        "rendered_append": "<user>next question</user><assistant>",
        "template_fingerprint": TEMPLATE,
        "tokenizer_sha256": TOKENIZER,
        "generation_prefix_token_count": 1,
    }


def _worker_reply(_endpoint, body):
    assert _endpoint == WORKER_CONTINUE_ENDPOINT
    assert body["append_token_ids"] == [9, 10]
    assert body["expected_checkpoint_payload_sha256"] == PAYLOAD
    assert "adapter_sha256" not in body
    assert "model_sha256" not in body
    return {
        "status": "completed",
        "request_id": body["request_id"],
        "worker_generation_id": "generation-1",
        "checkpoint_id": "checkpoint-1",
        "checkpoint_payload_sha256": PAYLOAD,
        "restore_mode": "live_checkpoint",
        "n_past_restored": 3,
        "n_past_after_append": 5,
        "append_token_count": 2,
        "append_token_ids_sha256": body["append_token_ids_sha256"],
        "tokens": [55, 56],
        "token_pieces": ["a new", " answer"],
        "text": "a new answer",
        "finish_reason": "stop",
        "cancelled": False,
        "sampler": {
            "source": "checkpoint",
            "mode": "sample",
            "config_sha256": SAMPLER_CONFIG,
            "state_sha256": SAMPLER_STATE,
            "rng_draws_before_append": 7,
        },
        "sampler_state_preserved": True,
        "steering_state_preserved": True,
        "historical_prefix_recomputed": False,
        "historical_prefix_retokenized_for_execution": False,
        "append_only_execution": True,
        "append_decode_regime": "sequential_single_token",
        "native_grammar_constraints_applied": False,
        "additional_stop_constraints_applied": False,
        "adapter_state_mutated": False,
    }


def _call(**overrides):
    values = {
        "requested_run_id": "run_requested",
        "request_id": "request-1",
        "generation_config_sha256": SETTINGS,
        "source": _source(),
        "source_checkpoint": _checkpoint(),
        "identity": _identity(),
        "sampler": _sampler(),
        "historical_token_ids": HISTORY,
        "render_append": _render,
        "worker_post": _worker_reply,
        "persist_child": lambda payload: {"child_run_id": payload["child_run_id"]},
        "clock": lambda: 100.0,
        "continuation_id": "tmc_0123456789abcdef0123",
        "child_run_id_factory": lambda: "run_child_1",
    }
    values.update(overrides)
    return orchestrate_continuation(
        {"turn": 0, "user": {"content": "next question"}, "max_tokens": 16}, **values)


def test_request_is_closed_and_does_not_admit_runtime_overrides():
    parsed = parse_continuation_request(
        {"turn": 0, "user": {"content": "hello"}, "max_tokens": 4})
    assert parsed.turn == 0 and parsed.user_content == "hello" and parsed.max_tokens == 4
    with pytest.raises(ClosedRequestError):
        parse_continuation_request({
            "turn": 0, "user": {"content": "hello"}, "max_tokens": 4, "temperature": 0.1,
        })
    with pytest.raises(ClosedRequestError):
        parse_continuation_request({"turn": 0, "user": {"content": "hello", "role": "user"}, "max_tokens": 4})


def test_append_derivation_returns_only_suffix_and_fails_closed_on_boundary_mismatch():
    append = derive_append_tokens(
        HISTORY, HISTORY + [9], rendered_append="<u>x", template_fingerprint=TEMPLATE,
        tokenizer_sha256=TOKENIZER)
    assert append.append_token_ids == (9,)
    assert append.receipt()["historical_token_ids_sha256"] == token_ids_sha256(HISTORY)
    with pytest.raises(AppendDerivationError, match="exact prefix"):
        derive_append_tokens(
            HISTORY, [1, 99, 3, 9], rendered_append="<u>x", template_fingerprint=TEMPLATE,
            tokenizer_sha256=TOKENIZER)


def test_token_hash_uses_the_worker_binary_domain_not_json_encoding():
    expected = hashlib.sha256(
        b"clozn.time-machine.token-ids.v1\0" + struct.pack("<I", 3) + struct.pack("<III", 1, 2, 3)
    ).hexdigest()
    assert token_ids_sha256([1, 2, 3]) == expected


def test_sampler_hashes_use_the_worker_binary_domains():
    config_wire = (
        b"clozn.time-machine.sampler-config.v1\0"
        + b"\x01"
        + struct.pack("<ddId", 0.8, 1.1, 40, 0.9)
    )
    state_wire = b"clozn.time-machine.sampler-state.v1\0" + struct.pack("<QQ", 7, 3)
    assert sampler_config_sha256(
        has_sampler=True, temperature=0.8, repeat_penalty=1.1, top_k=40, top_p=0.9,
    ) == hashlib.sha256(config_wire).hexdigest()
    assert sampler_state_sha256(seed=7, rng_draws=3) == hashlib.sha256(state_wire).hexdigest()


def test_success_calls_only_private_append_endpoint_and_persists_closed_receipt():
    seen = {}

    def persist(payload):
        seen.update(payload)
        schemas.validate(payload["receipt"])
        assert payload["user"] == {"content": "next question"}
        assert payload["receipt"]["append"]["append_token_ids"] == [9, 10]
        assert "next question" not in json.dumps(payload["receipt"])
        return {"child_run_id": payload["child_run_id"]}

    receipt = _call(persist_child=persist)

    schemas.validate(receipt)
    assert receipt["status"] == "completed"
    assert receipt["exactness"]["historical_prefix_retokenized_for_execution"] is False
    assert receipt["exactness"]["append_only_execution"] is True
    assert receipt["exactness"]["source_capture_regime"] == "organic_live_kv"
    assert receipt["child_lineage"]["child_run_id"] == "run_child_1"
    assert seen["receipt"] == receipt
    assert "sampler_rng_advanced_for_new_generation" in receipt["unavoidable_differences"]


def test_missing_checkpoint_is_unavailable_without_render_or_worker_call():
    calls = []

    receipt = _call(
        source_checkpoint=None,
        render_append=lambda _request: calls.append("render") or _render(_request),
        worker_post=lambda *_args: calls.append("worker"),
    )

    assert receipt["status"] == "unavailable"
    assert receipt["failure"]["code"] == "checkpoint_unavailable"
    assert receipt["child_lineage"]["status"] == "not_created"
    assert calls == []


def test_prefix_mismatch_is_terminal_and_worker_is_not_called():
    calls = []

    receipt = _call(
        render_append=lambda _request: {**_render(_request), "full_render_token_ids": [1, 8, 3, 9]},
        worker_post=lambda *_args: calls.append("worker"),
    )

    assert receipt["status"] == "failed"
    assert receipt["failure"]["stage"] == "append_derivation"
    assert receipt["failure"]["code"] == "append_prefix_mismatch"
    assert calls == []


def test_cancellation_before_persistence_creates_no_child():
    checks = iter([False, False, True])
    persisted = []

    receipt = _call(
        cancel_check=lambda: next(checks),
        persist_child=lambda payload: persisted.append(payload),
    )

    assert receipt["status"] == "cancelled"
    assert receipt["failure"]["code"] == "request_cancelled"
    assert receipt["worker"]["status"] == "cancelled"
    assert receipt["child_lineage"]["status"] == "cancelled"
    assert persisted == []


def test_persistence_failure_cannot_be_reported_as_completion():
    receipt = _call(persist_child=lambda _payload: None)

    schemas.validate(receipt)
    assert receipt["status"] == "failed"
    assert receipt["failure"] == {
        "stage": "persistence",
        "code": "child_persistence_failed",
        "message": "could not durably create the continuation child and terminal receipt",
        "retryable": True,
    }
    assert receipt["child_lineage"]["status"] == "failed"


def test_source_constraints_the_worker_cannot_apply_fail_closed_before_invocation():
    calls = []
    receipt = _call(
        source_generation_settings={"native_grammar_constraints_required": True},
        worker_post=lambda *_args: calls.append("worker"),
    )

    assert receipt["status"] == "unavailable"
    assert receipt["failure"]["code"] == "worker_capability_missing"
    assert calls == []


def test_durable_import_keeps_public_provenance_while_worker_uses_live_checkpoint():
    checkpoint = {
        **_checkpoint(),
        "provenance": "durable_pin_import",
        "capture_regime": "verified_prompt_boundary_reprefill",
        "restart_safe": True,
        "pin_id": "pin_0123456789abcdef0123",
        "pin_blob_sha256": "3" * 64,
        "source_worker_generation_id": "generation-before-restart",
    }
    receipt = _call(source_checkpoint=checkpoint)

    assert receipt["status"] == "completed"
    assert receipt["worker"]["restore_mode"] == "durable_import"
    assert receipt["exactness"]["historical_state_source"] == "durable_import"
    assert receipt["exactness"]["source_capture_regime"] == "verified_prompt_boundary_reprefill"
    assert "worker_process_generation_changed_after_durable_import" in receipt["unavoidable_differences"]


def test_worker_wire_requires_payload_digest_and_never_carries_gateway_runtime_identity():
    request = parse_continuation_request(
        {"turn": 0, "user": {"content": "hello"}, "max_tokens": 4}).receipt(
        request_id="request-x", generation_config_sha256=SETTINGS)
    append = derive_append_tokens(
        HISTORY, HISTORY + [4], rendered_append="x", template_fingerprint=TEMPLATE,
        tokenizer_sha256=TOKENIZER)
    body = build_worker_request(
        request=request, source_checkpoint=_checkpoint(), append=append, sampler=_sampler())
    assert body["expected_checkpoint_payload_sha256"] == PAYLOAD
    assert set(body) == {
        "checkpoint_id", "worker_generation_id", "expected_checkpoint_payload_sha256",
        "expected_n_past", "expected_token_history_sha256", "append_token_ids",
        "append_token_ids_sha256", "max_tokens", "request_id",
    }
    with_checkpoint = build_worker_request(
        request=request, source_checkpoint=_checkpoint(), append=append, sampler=_sampler(),
        checkpoint_on_finish=True)
    assert with_checkpoint == {**body, "checkpoint_on_finish": True}
