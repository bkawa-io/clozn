"""Model-free FORK-CKPT-01 recorded-parent checkpoint capture coverage."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from clozn import schemas
from clozn.cli.worker_registry import AdapterRuntimeIdentity, RuntimeKey
from clozn.replay.checkpoint_capture import capture_parent_checkpoint
from clozn.replay.execution_fork import parent_runtime_projection
import clozn.runs.store as runlog
from clozn.server.model_routing import PreloadedModelBinding, PreloadedModelRouter
from clozn.server.routes import execution_fork as route


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "test-build",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {
        "present": False,
        "identity_sha256": None,
        "artifact_sha256": None,
        "scale": None,
    },
    "white_box_flags": {},
}
WORKER = {
    "worker_id": "generation-a",
    "worker_generation_id": "generation-a",
    "protocol_version": "1.1",
}


def _sha(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_managed_parent_runtime_accepts_coarse_gpu_and_partial_legacy_adapter():
    white_box = {"sae": False, "jlens": False, "attn_knockout": False}
    adapter = AdapterRuntimeIdentity(
        present=True,
        identity_sha256="8" * 64,
        artifact_sha256="9" * 64,
        scale=0.5,
    )
    key = RuntimeKey(
        gguf_artifact_sha256="a" * 64,
        context_size=4096,
        backend="gpu",
        adapter=adapter,
        template_fingerprint="b" * 16,
        engine_build="gpu-build",
        white_box_flags=white_box,
    ).as_dict()

    class Engine:
        def health(self):
            return {
                "status": "ok",
                "worker_generation_id": "gpu-worker",
                "protocol_version": "1.1",
                "model_sha256": "a" * 64,
                "n_ctx": 4096,
                "device": "cuda",
                "engine_build": "gpu-build",
                "template_fingerprint": "b" * 16,
                "capabilities": dict(white_box),
                "lora": {"scale": 0.5},
            }

    engine = Engine()
    sub = type("Sub", (), {"engine": engine})()
    binding = PreloadedModelBinding(
        model_id="gpu-model",
        resolved_artifact={
            "model_id": "gpu-model",
            "format": "gguf",
            "artifact_sha256": "a" * 64,
        },
        runtime_key=key,
        adapter=adapter.as_dict(),
        state="ready",
        worker_identity={
            "worker_id": "gpu-worker",
            "worker_generation_id": "gpu-worker",
            "worker_generation": 1,
            "runtime_key_sha256": key["key_sha256"],
            "protocol_version": "1.1",
            "engine_build": "gpu-build",
            "backend": "gpu",
        },
        sub=sub,
        engine=engine,
    )
    router = PreloadedModelRouter(
        [binding],
        default_model_id="gpu-model",
        preload_model_ids=["gpu-model"],
        max_loaded_workers=1,
    )
    routing = router.select(
        "gpu-model",
        field_present=True,
        surface="openai",
        route="/v1/chat/completions",
    ).artifact
    parent = {
        "id": "run_gpu",
        "model": "gpu-model",
        "identity": {
            "model_sha256": "a" * 64,
            "template_fingerprint": "b" * 16,
            "ext": {
                "adapter": {
                    "path": "adapter.gguf",
                    "scale": 0.5,
                    "meta": {"general.architecture": "lora"},
                }
            },
        },
        "meta": {
            "n_ctx": 4096,
            "device": "cuda",
            # EngineSubstrate journals a friendly/upstream worker identity here. It is intentionally
            # a different namespace from the gateway's canonical model ID in parent["model"].
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "white_box_flags": dict(white_box),
            "model_routing": routing,
        },
    }

    projected = parent_runtime_projection(parent)
    assert projected is not None
    assert projected["backend"] == "gpu"
    assert projected["adapter"] == adapter.as_dict()

    contradicted = deepcopy(parent)
    contradicted["identity"]["ext"]["adapter"]["scale"] = 0.75
    assert parent_runtime_projection(contradicted) is None


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _parent(*, sampled=True, active_dials=None, steering=None, **overrides):
    decode = (
        {
            "mode": "sample",
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "seed": 7,
        }
        if sampled
        else {"mode": "greedy", "temperature": 0.0, "seed": 0}
    )
    meta = {
        "n_ctx": 4096,
        "device": "cpu",
        "prompt_tokens": 2,
        "stream": False,
        "decode": decode,
    }
    if steering is not None:
        meta["execution_fork_steering"] = steering
    values = {
        "source": "engine_chat",
        "client": "studio",
        "model": "fixture-model",
        "substrate": "engine",
        "messages": [{"role": "user", "content": "count"}],
        "assembled_messages": [{"role": "user", "content": "count"}],
        "response": "one two three",
        "final_prompt": "<prompt>",
        "trace": {"tokens": ["one", " two", " three"], "token_ids": [11, 22, 33]},
        "behavior": {"active_dials": dict(active_dials or {})},
        "meta": meta,
        "identity": {
            "model_sha256": "a" * 64,
            "template_fingerprint": "b" * 16,
            "engine_build": "test-build",
            "white_box_flags": {},
        },
    }
    values.update(overrides)
    run_id = runlog.record(**values)
    assert run_id
    return runlog.get_run(run_id)


class CaptureEngine:
    def __init__(self, *, prompt_ids=None, control_tokens=None,
                 control_text=None, checkpoint_error=None, finish_reason="stop"):
        self.prompt_ids = list(prompt_ids or [1, 2])
        self.control_tokens = list(control_tokens or [11, 22, 33])
        self.control_text = control_text if control_text is not None else "one two three"
        self.checkpoint_error = checkpoint_error
        self.finish_reason = finish_reason
        self.calls = []

    def score(self, **kwargs):
        self.calls.append(("score", deepcopy(kwargs)))
        continuation = list(kwargs["continuation_ids"])
        return {
            "n_prompt": len(self.prompt_ids),
            "n_cont": len(continuation),
            "prompt_ids": list(self.prompt_ids),
            "tokens": [
                {"id": token_id, "piece": str(token_id), "logprob": -0.1}
                for token_id in continuation
            ],
            "sum_logprob": -0.1 * len(continuation),
        }

    def create_checkpoint(self, tokens, **kwargs):
        self.calls.append(("checkpoint", {"tokens": list(tokens), **deepcopy(kwargs)}))
        if self.checkpoint_error is not None:
            raise self.checkpoint_error
        return {
            "checkpoint_id": "ckpt-generation-a-9",
            "worker_generation_id": "generation-a",
            "n_past": len(tokens),
            "n_tokens": len(tokens),
            "size_bytes": 987654,
        }

    def import_checkpoint(self, envelope):
        self.calls.append(("import_checkpoint", deepcopy(envelope)))
        state = envelope["state"]
        return {
            "checkpoint_id": "ckpt-generation-a-imported",
            "worker_generation_id": "generation-a",
            "n_past": state["n_past"],
            "size_bytes": 987654,
        }

    def execution_fork(self, **kwargs):
        self.calls.append(("execution_fork", deepcopy(kwargs)))
        assert kwargs["intervention"] == {"type": "none"}
        return {
            "worker_generation_id": "generation-a",
            "text": self.control_text,
            "tokens": list(self.control_tokens),
            "prompt_len": 2,
            "n_past_restored": 2,
            "restore_mode": "reprefill",
            "exactness": {
                "source": "reprefill",
                "truncation_regime": "prompt_boundary",
                "boundary_shape_true": True,
            },
            "sampler_source": "checkpoint",
            "steer_source": "none",
            "intervention_applied": {"type": "none"},
            "finish_reason": self.finish_reason,
        }

    def health(self):
        return {
            "worker_generation_id": "generation-a",
            "protocol_version": "1.1",
        }


def test_capture_reconstructs_and_proves_ephemeral_reference_without_mutating_parent(store):
    parent = _parent()
    before = deepcopy(parent)
    before_ids = {run["id"] for run in runlog.iter_runs()}
    engine = CaptureEngine()

    artifact = capture_parent_checkpoint(
        parent, engine,
        runtime_identity=RUNTIME,
        worker_identity=WORKER,
        clock=lambda: 123.5,
    )

    schemas.validate(artifact)
    assert artifact["status"] == "available"
    assert artifact["captured_ts"] == 123.5
    assert artifact["lifecycle"] == {
        "storage": "worker_memory",
        "durability": "ephemeral",
        "pinned": False,
        "eviction_policy": "bounded_fifo",
        "validity_scope": "worker_process_generation",
        "observed_state": "available",
        "size_bytes": 987654,
        "expires_when": ["worker_restart", "fifo_eviction", "gateway_shutdown"],
    }
    reference = artifact["checkpoint_reference"]
    assert reference["size_bytes"] == 987654
    assert reference["prompt_tokens"] == 2
    assert reference["n_past"] == 5
    assert artifact["proof"]["status"] == "matched"
    assert artifact["proof"]["exactness_regime"] == "prompt_boundary_reprefill"
    assert [name for name, _call in engine.calls] == [
        "score", "checkpoint", "execution_fork"]
    score = engine.calls[0][1]
    assert score == {
        "prompt": "<prompt>",
        "continuation_ids": [11, 22, 33],
        "topk": 0,
    }
    checkpoint = engine.calls[1][1]
    assert checkpoint["tokens"] == [1, 2, 11, 22, 33]
    assert checkpoint["prefill_to"] == 2
    assert checkpoint["n_past"] == 5
    assert checkpoint["worker_generation_id"] == "generation-a"
    assert checkpoint["sampler"] == {
        "seed": 7,
        "rng_draws": 3,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
        "rep_penalty": 1.1,
    }
    assert runlog.get_run(parent["id"]) == before
    assert {run["id"] for run in runlog.iter_runs()} == before_ids


def test_greedy_capture_omits_sampler_state_and_records_zero_draws(store):
    parent = _parent(sampled=False)
    engine = CaptureEngine()

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "available"
    assert artifact["sampler"]["mode"] == "greedy"
    assert artifact["sampler"]["rng_draws"] == 0
    checkpoint = next(call for name, call in engine.calls if name == "checkpoint")
    assert "sampler" not in checkpoint


def test_schema_binds_sampler_steering_and_lifecycle_to_their_modes(store):
    artifact = capture_parent_checkpoint(
        _parent(), CaptureEngine(),
        runtime_identity=RUNTIME, worker_identity=WORKER)

    missing_seed = deepcopy(artifact)
    missing_seed["sampler"].pop("seed")
    with pytest.raises(schemas.ValidationError):
        schemas.validate(missing_seed)

    cross_mode = deepcopy(artifact)
    cross_mode["steering"]["vector_sha256"] = "f" * 64
    with pytest.raises(schemas.ValidationError):
        schemas.validate(cross_mode)

    wrong_lifecycle = deepcopy(artifact)
    wrong_lifecycle["lifecycle"]["observed_state"] = "not_created"
    with pytest.raises(schemas.ValidationError):
        schemas.validate(wrong_lifecycle)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda run: run["meta"].pop("prompt_tokens"), "missing_prompt_boundary"),
        (lambda run: run["meta"]["decode"].pop("seed"), "sampler_provenance_missing"),
        (lambda run: run["meta"].pop("stream"), "unsupported_execution_shape"),
        (lambda run: run["behavior"].update(active_dials={"warm": 0.5}),
         "steering_provenance_missing"),
        (lambda run: run.update(reasoning={"private_text": "not retained"}),
         "unsupported_execution_shape"),
        (lambda run: run.update(output_contract={"activated": True}),
         "unsupported_execution_shape"),
    ],
)
def test_missing_execution_provenance_fails_before_any_worker_call(
    store, mutate, reason,
):
    parent = _parent()
    mutate(parent)
    engine = CaptureEngine()

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "unavailable"
    assert artifact["reasons"][0]["code"] == reason
    assert artifact["proof"] == {"status": "not_run"}
    assert engine.calls == []


def test_prompt_ids_require_worker_evidence_and_original_recorded_count(store):
    parent = _parent()
    engine = CaptureEngine(prompt_ids=[1, 2, 3])

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "failed"
    assert artifact["reasons"][0]["code"] == "prompt_tokenization_failed"
    assert [name for name, _call in engine.calls] == ["score"]
    assert "checkpoint_reference" not in artifact


def test_runtime_mismatch_is_planner_unavailable_before_tokenization(store):
    parent = _parent()
    engine = CaptureEngine()
    wrong = {**RUNTIME, "backend": "cuda"}

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=wrong, worker_identity=WORKER)

    assert artifact["status"] == "unavailable"
    assert artifact["reasons"][0]["code"] == "runtime_identity_mismatch"
    assert engine.calls == []


def test_exact_raw_steering_provenance_is_replayed_but_dial_names_are_not_rederived(store):
    dials = {"warm": 0.5}
    steering = {
        "source": "recorded_raw_vector",
        "steer_vec": [0.25, -0.5],
        "steer_layer": 4,
        "steer_coef": 1.25,
        "active_dials_sha256": _sha(dials),
    }
    parent = _parent(active_dials=dials, steering=steering)
    engine = CaptureEngine()

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "available"
    assert artifact["steering"]["mode"] == "raw_vector"
    checkpoint = next(call for name, call in engine.calls if name == "checkpoint")
    assert checkpoint["steer_vec"] == [0.25, -0.5]
    assert checkpoint["steer_layer"] == 4
    assert checkpoint["steer_coef"] == 1.25


def test_diverged_control_leaves_real_checkpoint_visible_but_unusable(store):
    parent = _parent()
    engine = CaptureEngine(control_tokens=[11, 999, 33], control_text="one wrong three")

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "failed"
    assert artifact["reasons"][0]["code"] == "unchanged_control_diverged"
    assert artifact["proof"]["status"] == "diverged"
    assert artifact["checkpoint_reference"]["size_bytes"] == 987654
    assert artifact["lifecycle"]["observed_state"] == "unusable"
    assert artifact["lifecycle"]["pinned"] is False


# ============================================================================================
# Boundary stop-token exemption (gateway/prove_unchanged_control divergence characterized live
# against qwen2.5-0.5b-instruct-q4_k_m.gguf: scripts/smoke/gateway_eos_boundary_battery.py).
#
# The recorded parent trace (native chat-io's transcript) includes the chat turn's terminal
# stop/EOS-class token as an explicit trailing entry whenever the run finished via
# finish_reason == "stop". The raw engine's own generation loop (generate_ar, what
# EngineClient.execution_fork replays) NEVER returns that token as part of `tokens` -- sampling it
# TERMINATES the loop; it is a stop signal there, not committed output (see finish_reason()'s
# "eos"/"stop" -> "stop" mapping in engine/core/serve/server_shared.hpp). Every EOS-terminated chat
# run therefore has a parent trace exactly ONE token longer than anything a raw replay can ever
# produce -- a representational mismatch between two independently-correct conventions, not a real
# generation divergence. prove_unchanged_control's _boundary_stop_token_exempt narrowly exempts
# EXACTLY this shape (see that function's own docstring for the four required conditions); every
# test below proves one edge of that boundary, so a fix that becomes even slightly more permissive
# than intended fails one of them.
# ============================================================================================

def _parent_eos_terminated(*, extra_piece=""):
    """A parent whose recorded trace carries ONE trailing token beyond `[11, 22, 33]` -- decoding
    to `extra_piece` (empty by default, matching a real EOS/chat-end control token) -- and whose
    own finish_reason is "stop", mirroring an ordinary EOS-terminated chat completion exactly."""
    return _parent(
        trace={"tokens": ["one", " two", " three", extra_piece], "token_ids": [11, 22, 33, 999]},
        finish_reason="stop",
    )


def test_boundary_stop_token_is_exempted_when_both_sides_agree_finish_reason_stop(store):
    parent = _parent_eos_terminated()
    engine = CaptureEngine(control_tokens=[11, 22, 33], control_text="one two three",
                           finish_reason="stop")

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "available"
    assert artifact["proof"]["status"] == "matched"
    assert artifact["proof"]["control_result"]["exact_match"] is True
    note = artifact["proof"]["control_result"]["note"]
    assert "stop token" in note and "exempted" in note
    # The exemption is disclosed, not hidden: the two suffix hashes genuinely differ (the parent's
    # recorded suffix really is one token longer) even though exact_match is True.
    assert (artifact["proof"]["control_result"]["parent_suffix_sha256"]
            != artifact["proof"]["control_result"]["control_suffix_sha256"])


def test_boundary_exemption_does_not_cover_an_earlier_real_divergence(store):
    """Same off-by-one LENGTH shape (parent 4 tokens, control 3) -- but the middle token is wrong
    too. The exemption must never mask an actual content divergence sitting behind it."""
    parent = _parent_eos_terminated()
    engine = CaptureEngine(control_tokens=[11, 999, 33], control_text="one wrong three",
                           finish_reason="stop")

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "failed"
    assert artifact["reasons"][0]["code"] == "unchanged_control_diverged"
    assert artifact["proof"]["control_result"]["exact_match"] is False


def test_boundary_exemption_requires_parent_finish_reason_stop(store):
    """If the parent itself did NOT record finish_reason=='stop' (e.g. it was truncated at
    max_tokens), a length-off-by-one control reply is a genuine divergence, not a boundary quirk --
    there is no reason a max_tokens-truncated parent's trace would carry an extra token at all."""
    parent = _parent(
        trace={"tokens": ["one", " two", " three", ""], "token_ids": [11, 22, 33, 999]},
        finish_reason="length",
    )
    engine = CaptureEngine(control_tokens=[11, 22, 33], control_text="one two three",
                           finish_reason="stop")

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "failed"
    assert artifact["reasons"][0]["code"] == "unchanged_control_diverged"
    assert artifact["proof"]["control_result"]["exact_match"] is False


def test_boundary_exemption_requires_worker_finish_reason_stop(store):
    """If the REPLAY did not itself report finish_reason=='stop' (e.g. it hit its own max_tokens
    cap instead of naturally stopping), the two runs have not independently agreed on WHY they
    each ended one token apart -- fail closed rather than assume."""
    parent = _parent_eos_terminated()
    engine = CaptureEngine(control_tokens=[11, 22, 33], control_text="one two three",
                           finish_reason="length")

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "failed"
    assert artifact["reasons"][0]["code"] == "unchanged_control_diverged"
    assert artifact["proof"]["control_result"]["exact_match"] is False


def test_boundary_exemption_requires_exact_text_match(store):
    """The exemption never overrides a TEXT mismatch -- only ever forgives the one known-missing
    trailing token id when the decoded text is already identical (as it always is for a control
    token whose piece truly is empty)."""
    parent = _parent_eos_terminated()
    engine = CaptureEngine(control_tokens=[11, 22, 33], control_text="one two threeX",
                           finish_reason="stop")

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "failed"
    assert artifact["reasons"][0]["code"] == "unchanged_control_diverged"
    assert artifact["proof"]["control_result"]["exact_match"] is False


def test_boundary_exemption_requires_exactly_one_missing_token(store):
    """Two (or zero) missing tokens is not the known EOS-bookkeeping shape -- still a divergence."""
    parent = _parent_eos_terminated()
    engine = CaptureEngine(control_tokens=[11], control_text="one", finish_reason="stop")

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "failed"
    assert artifact["reasons"][0]["code"] == "unchanged_control_diverged"
    assert artifact["proof"]["control_result"]["exact_match"] is False


def test_capture_failure_never_fabricates_checkpoint_size_or_reference(store):
    parent = _parent()
    engine = CaptureEngine(checkpoint_error=RuntimeError("worker full"))

    artifact = capture_parent_checkpoint(
        parent, engine, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert artifact["status"] == "failed"
    assert artifact["reasons"][0]["code"] == "checkpoint_capture_failed"
    assert artifact["lifecycle"]["observed_state"] == "not_created"
    assert "size_bytes" not in artifact["lifecycle"]
    assert "checkpoint_reference" not in artifact


class Handler:
    def __init__(self, sub):
        self._inj_sub = sub
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


class FakeSub:
    def __init__(self, engine, *, runtime=RUNTIME, worker=WORKER):
        self.engine = engine
        self.runtime_identity = lambda: deepcopy(runtime)
        self.worker_identity = lambda: deepcopy(worker)


def test_legacy_engine_substrate_shape_uses_observed_executable_sha(store, monkeypatch):
    from clozn.server import app as server

    executable_sha = "e" * 64
    engine_build = f"sha256:{executable_sha}"

    class OrganicEngine(CaptureEngine):
        def health(self):
            return {
                "status": "ok",
                "worker_generation_id": "generation-a",
                "protocol_version": "1.1",
                "model_sha256": "a" * 64,
                "n_ctx": 4096,
                "device": "cpu",
                "capabilities": {},
            }

    class OrganicSub:
        def __init__(self, engine):
            self.engine = engine

        def identity_meta(self):
            return {
                "model_sha256": "a" * 64,
                "template_fingerprint": "b" * 16,
                "ext": {
                    "engine_artifact": {
                        "discovery_source": "managed",
                        "artifact_sha256": executable_sha,
                        "protocol_version": "1.1",
                    },
                },
            }

        def run_meta(self):
            return {
                "n_ctx": 4096,
                "device": "cpu",
                "model_id": "Qwen/Qwen2.5-7B-Instruct",
                "white_box_flags": {},
            }

    identity = {
        "model_sha256": "a" * 64,
        "template_fingerprint": "b" * 16,
        "ext": {
            "engine_artifact": {
                "discovery_source": "managed",
                "artifact_sha256": executable_sha,
                "protocol_version": "1.1",
            },
        },
        "white_box_flags": {},
    }
    parent = _parent(identity=identity)
    engine = OrganicEngine()
    monkeypatch.setattr(server, "MODEL_ROUTER", None)
    handler = Handler(OrganicSub(engine))

    assert route.try_post(
        handler,
        f"/runs/{parent['id']}/execution-fork/checkpoint",
        {},
    )
    assert handler.status == 201
    assert handler.body["status"] == "available"
    projected = parent_runtime_projection(parent)
    assert projected is not None
    assert projected["engine_build"] == engine_build
    assert (
        handler.body["identity"]["selected_runtime_key_sha256"]
        == projected["runtime_key_sha256"]
    )

    contradicted = deepcopy(parent)
    contradicted["identity"]["engine_build"] = "named-build-that-is-not-the-observed-sha"
    refused_engine = OrganicEngine()
    refused = capture_parent_checkpoint(
        contradicted,
        refused_engine,
        runtime_identity={
            "model_sha256": "a" * 64,
            "template_fingerprint": "b" * 16,
            "engine_build": engine_build,
            "context_size": 4096,
            "backend": "cpu",
            "adapter": {
                "present": False,
                "identity_sha256": None,
                "artifact_sha256": None,
                "scale": None,
            },
            "white_box_flags": {},
        },
        worker_identity=WORKER,
    )
    assert refused["status"] == "unavailable"
    assert refused["reasons"][0]["code"] == "runtime_identity_unavailable"
    assert refused_engine.calls == []


def test_gateway_checkpoint_route_returns_public_artifact_and_rejects_v1_options(store):
    parent = _parent()
    engine = CaptureEngine()
    handler = Handler(FakeSub(engine))

    assert route.try_post(
        handler,
        f"/runs/{parent['id']}/execution-fork/checkpoint",
        {},
    )
    assert handler.status == 201
    assert handler.body["status"] == "available"
    assert handler.body["schema_version"] == "clozn.checkpoint-reference.v1"

    rejected = Handler(FakeSub(CaptureEngine()))
    assert route.try_post(
        rejected,
        f"/runs/{parent['id']}/execution-fork/checkpoint",
        {"pin": True},
    )
    assert rejected.status == 400
    assert rejected.body["code"] == "checkpoint_capture_options_unsupported"


def test_gateway_checkpoint_route_can_explicitly_hydrate_a_durable_pin(store, monkeypatch):
    parent = _parent()
    engine = CaptureEngine()
    envelope = {
        "envelope_version": "clozn.checkpoint-export.v1",
        "identity": {"model_sha256": "a" * 64},
        "state": {
            "tokens": [1, 2, 11, 22, 33],
            "n_tokens": 5,
            "n_past": 5,
            "prompt_tokens": 2,
        },
        "payload_sha256": "c" * 64,
    }
    monkeypatch.setattr(
        "clozn.replay.checkpoint_pin_store.resolve_pin",
        lambda run_id: {"ok": True, "manifest": {"run_id": run_id}, "envelope": envelope},
    )
    handler = Handler(FakeSub(engine))

    assert route.try_post(
        handler,
        f"/runs/{parent['id']}/execution-fork/checkpoint",
        {"pinned": True},
    )
    assert handler.status == 201
    assert handler.body["status"] == "available"
    assert [name for name, _call in engine.calls] == [
        "score", "import_checkpoint", "execution_fork"
    ]


def test_capture_can_hydrate_a_resolved_pin_and_still_prove_the_same_control(store):
    parent = _parent()
    engine = CaptureEngine()
    envelope = {
        "envelope_version": "clozn.checkpoint-export.v1",
        "identity": {"model_sha256": "a" * 64},
        "state": {
            "tokens": [1, 2, 11, 22, 33],
            "n_tokens": 5,
            "n_past": 5,
            "prompt_tokens": 2,
        },
        "payload_sha256": "c" * 64,
    }

    artifact = capture_parent_checkpoint(
        parent,
        engine,
        runtime_identity=RUNTIME,
        worker_identity=WORKER,
        checkpoint_envelope=envelope,
    )

    schemas.validate(artifact)
    assert artifact["status"] == "available"
    assert artifact["proof"]["status"] == "matched"
    assert [name for name, _call in engine.calls] == [
        "score", "import_checkpoint", "execution_fork"
    ]
    assert engine.calls[1][1] == envelope


def test_capture_refuses_a_pin_whose_token_history_is_not_the_parent(store):
    parent = _parent()
    engine = CaptureEngine()
    artifact = capture_parent_checkpoint(
        parent,
        engine,
        runtime_identity=RUNTIME,
        worker_identity=WORKER,
        checkpoint_envelope={
            "state": {
                "tokens": [1, 2, 99],
                "n_tokens": 3,
                "n_past": 3,
                "prompt_tokens": 2,
            }
        },
    )
    assert artifact["status"] == "unavailable"
    assert artifact["reasons"][0]["code"] == "pinned_checkpoint_parent_mismatch"
    assert [name for name, _call in engine.calls] == ["score"]


def test_gateway_capture_selects_non_default_parent_model_worker(
    store, monkeypatch,
):
    from clozn.server import app as server

    white_box = {"sae": False, "jlens": False, "attn_knockout": False}
    absent = AdapterRuntimeIdentity.absent()

    class RoutedEngine(CaptureEngine):
        def __init__(self, *, digest, template, build, generation):
            super().__init__()
            self.digest = digest
            self.template = template
            self.build = build
            self.generation = generation

        def health(self):
            return {
                "status": "ok",
                "worker_generation_id": self.generation,
                "protocol_version": "1.1",
                "model_sha256": self.digest,
                "n_ctx": 4096,
                "device": "cpu",
                "engine_build": self.build,
                "template_fingerprint": self.template,
                "capabilities": dict(white_box),
            }

    def binding(model_id, digest, template, build, generation):
        runtime_key = RuntimeKey(
            gguf_artifact_sha256=digest,
            context_size=4096,
            backend="cpu",
            adapter=absent,
            template_fingerprint=template,
            engine_build=build,
            white_box_flags=white_box,
        )
        runtime = {
            "model_sha256": digest,
            "template_fingerprint": template,
            "engine_build": build,
            "context_size": 4096,
            "backend": "cpu",
            "adapter": absent.as_dict(),
            "white_box_flags": dict(white_box),
        }
        worker = {
            "worker_id": generation,
            "worker_generation_id": generation,
            "protocol_version": "1.1",
        }
        engine = RoutedEngine(
            digest=digest, template=template, build=build, generation=generation)
        sub = FakeSub(engine, runtime=runtime, worker=worker)
        key = runtime_key.as_dict()
        return PreloadedModelBinding(
            model_id=model_id,
            resolved_artifact={
                "model_id": model_id,
                "format": "gguf",
                "artifact_sha256": digest,
            },
            runtime_key=key,
            adapter=absent.as_dict(),
            state="ready",
            worker_identity={
                **worker,
                "worker_generation": 1,
                "runtime_key_sha256": key["key_sha256"],
                "engine_build": build,
                "backend": "cpu",
            },
            sub=sub,
            engine=engine,
        ), engine, runtime

    alpha, alpha_engine, _alpha_runtime = binding(
        "alpha", "c" * 64, "d" * 16, "alpha-build", "generation-alpha")
    beta, beta_engine, beta_runtime = binding(
        "beta", "a" * 64, "b" * 16, "test-build", "generation-a")
    router = PreloadedModelRouter(
        [alpha, beta],
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
    )
    persisted_routing = router.select(
        "beta",
        field_present=True,
        surface="openai",
        route="/v1/chat/completions",
    ).artifact
    parent = _parent(
        model="beta",
        identity={
            "model_sha256": beta_runtime["model_sha256"],
            "template_fingerprint": beta_runtime["template_fingerprint"],
            "white_box_flags": dict(white_box),
        },
        meta={
            "n_ctx": 4096,
            "device": "cpu",
            "prompt_tokens": 2,
            "stream": False,
            "decode": {
                "mode": "sample",
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 40,
                "repeat_penalty": 1.1,
                "seed": 7,
            },
            "white_box_flags": dict(white_box),
            "model_id": "beta",
            "model_routing": persisted_routing,
        },
    )
    monkeypatch.setattr(server, "MODEL_ROUTER", router)
    handler = Handler(alpha.sub)

    assert route.try_post(
        handler,
        f"/runs/{parent['id']}/execution-fork/checkpoint",
        {},
    )

    assert handler.status == 201
    assert handler.body["status"] == "available"
    assert [name for name, _call in beta_engine.calls] == [
        "score", "checkpoint", "execution_fork"]
    assert alpha_engine.calls == []
    assert handler.body["identity"]["worker_generation_id"] == "generation-a"

    contradicted = deepcopy(parent)
    contradicted["identity"]["model_sha256"] = "f" * 64
    fresh_engine = CaptureEngine()
    refused = capture_parent_checkpoint(
        contradicted,
        fresh_engine,
        runtime_identity=beta_runtime,
        worker_identity=WORKER,
    )
    assert refused["status"] == "unavailable"
    assert refused["reasons"][0]["code"] == "runtime_identity_unavailable"
    assert fresh_engine.calls == []
