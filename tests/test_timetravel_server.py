"""test_timetravel_server -- the time-travel debugger's studio endpoints (MODEL-FREE).

No model, no GPU. Drives the REAL clozn_server do_GET/do_POST handlers (the object.__new__(H) no-socket
trick, same as test_facts_server) against a FAKE substrate whose .chat() echoes the transcript it saw, so
we exercise ALL the wiring end to end:
  * POST /runs/<id>/branch -> a CHILD run (parent_run_id + changes_applied noting the branch turn), from
    the TRUNCATED transcript; greedy by default; alt_user substitution; validation (bad/absent turn,
    missing run, no-substrate 503).
  * GET/POST /timetravel/mode -> the on/off gate (DEFAULT OFF -- the RAM rule) + ring config + honest
    store stats, persisted through studio_settings.json.
  * the branch never mutates the live studio (dials/strength restored).
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs   # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402

import clozn.runs.store as runlog               # noqa: E402
import clozn.replay.timetravel as timetravel           # noqa: E402
from clozn import schemas                         # noqa: E402


# --- fake substrate: .chat() echoes the transcript it saw; dials/strength are restorable ---------------
class _FakeSteer:
    def __init__(self, strength=None):
        self.strength = dict(strength or {})
        self.saved = False

    def active(self):
        return {k: v for k, v in self.strength.items() if v}

    def save_state(self, path):
        self.saved = True


class _FakeMem:
    def __init__(self, strength=1.0):
        self.memory_strength = float(strength)
        self.prefix = None


class FakeSub:
    name = "qwen"

    def __init__(self, mem=None, steer=None):
        self.memory = mem if mem is not None else _FakeMem()
        self._mem = self.memory
        self.steer = steer if steer is not None else _FakeSteer()
        self.seen = None

    def chat(self, messages, max_new=256, sample=True):
        self.seen = {"messages": [dict(m) for m in messages], "sample": sample}
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return f"reply[{last_user}|{len(messages)}]"


# --- driving the real handler without a socket (mirrors test_facts_server) ----------------------------
def _dispatch(method, path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _post(path, body_obj=None):
    return _dispatch("POST", path, body_obj)


def _get(path):
    return _dispatch("GET", path)


CONV = [
    {"role": "user", "content": "u0"},
    {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "a2"},
]


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Isolate settings + the run store; install a fake substrate + a fresh snapshot store. The gate
    starts OFF (the real default)."""
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    sub = FakeSub()
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "qwen")
    monkeypatch.setattr(cs, "SNAPSHOTS", None)      # fresh store each test
    return tmp_path


def _seed_parent():
    """Record a parent run whose transcript is CONV; return its id."""
    return runlog.record(source="studio_chat", client="studio", model="clozn-qwen",
                         substrate="QwenSubstrate", messages=CONV, response="a2")


def _seed_session_history():
    """Three organic session runs whose message prefixes identify turns 0, 1, and 2."""
    session = "session_0123456789abcdef01234567"
    # Match the real OpenAI shape: each request persists messages through its user turn and stores the
    # completed assistant text in the separate immutable `response` field.
    prefixes = [CONV[:1], CONV[:3], CONV[:5]]
    ids = []
    for index, (messages, response) in enumerate(zip(prefixes, ("a0", "a1", "a2"))):
        ids.append(runlog.record(
            source="studio_chat", client="studio", model="clozn-qwen", substrate="engine",
            messages=messages, response=response, session_key=session,
            started=100.0 + index,
        ))
    return ids[-1], ids[0], ids[1]


_EXACT_MODEL_SHA = "a" * 64
_EXACT_TOKENIZER_SHA = "b" * 64
_EXACT_TEMPLATE = "c" * 16
_EXACT_PAYLOAD = "d" * 64
_EXACT_RUNTIME = {
    "gguf_artifact_sha256": _EXACT_MODEL_SHA,
    "template_fingerprint": _EXACT_TEMPLATE,
    "engine_build": "fixture-build",
    "context_size": 128,
    "backend": "cpu",
    "adapter": {
        "present": False,
        "identity_sha256": None,
        "artifact_sha256": None,
        "scale": None,
    },
    "white_box_flags": {},
}
_EXACT_WORKER = {
    "worker_id": "generation-exact",
    "worker_generation_id": "generation-exact",
    "protocol_version": "1.1",
}


class ExactContinuationEngine:
    def __init__(self, *, prefix_matches=True):
        self.prefix_matches = prefix_matches
        self.continuations = []

    def health(self):
        return {
            "status": "ok",
            "worker_generation_id": "generation-exact",
            "protocol_version": "1.1",
            "tokenizer_sha256": _EXACT_TOKENIZER_SHA,
            "capabilities": {"time_machine_continuation": True},
        }

    def apply_template_info(self, messages, add_assistant=True):
        assert messages[-1] == {"role": "user", "content": "new question"}
        prompt = "<prompt>answer<user>new question</user>"
        if add_assistant:
            prompt += "<assistant>"
        return {"prompt": prompt, "prompt_tokens": 5 if add_assistant else 4}

    def score(self, **kwargs):
        ids = [1, 2, 3, 9, 10] if self.prefix_matches else [1, 8, 3, 9, 10]
        return {"prompt_ids": ids, "n_prompt": len(ids)}

    def time_machine_continue(self, **body):
        from clozn.replay.time_machine_continuation import sampler_config_sha256

        self.continuations.append(dict(body))
        return {
            "status": "completed",
            "code": "continuation_completed",
            "request_id": body["request_id"],
            "worker_generation_id": "generation-exact",
            "checkpoint_id": "checkpoint-imported",
            "checkpoint_payload_sha256": _EXACT_PAYLOAD,
            "restore_mode": "live_checkpoint",
            "n_past_restored": 3,
            "n_past_after_append": 5,
            "append_token_count": 2,
            "append_token_ids_sha256": body["append_token_ids_sha256"],
            "generated_token_count": 1,
            "tokens": [55],
            "token_pieces": ["next answer"],
            "text": "next answer",
            "finish_reason": "stop",
            "cancelled": False,
            "historical_prefix_recomputed": False,
            "historical_prefix_retokenized_for_execution": False,
            "append_only_execution": True,
            "append_decode_regime": "sequential_single_token",
            "sampler": {
                "source": "checkpoint",
                "mode": "greedy",
                "config_sha256": sampler_config_sha256(has_sampler=False),
                "state_sha256": None,
                "rng_draws_before_append": 0,
            },
            "sampler_state_preserved": True,
            "steering_state_preserved": True,
            "native_grammar_constraints_applied": False,
            "additional_stop_constraints_applied": False,
            "adapter_state_mutated": False,
            "final_checkpoint_id": "checkpoint-child",
        }


class ExactContinuationSub:
    name = "engine"

    def __init__(self, engine):
        self.engine = engine
        self.runtime_identity = dict(_EXACT_RUNTIME)
        self.worker_identity = dict(_EXACT_WORKER)


def _seed_exact_continuation_source():
    return runlog.record(
        source="studio_chat",
        client="studio",
        model="fixture-exact-model",
        substrate="engine",
        messages=[{"role": "user", "content": "original question"}],
        response="answer",
        final_prompt="<prompt>",
        trace={"tokens": ["answer"], "token_ids": [3]},
        meta={
            "prompt_tokens": 2,
            "stream": False,
            "decode": {"mode": "greedy", "temperature": 0.0, "seed": 0},
            "sampler_mode": "greedy",
            "n_ctx": 128,
            "device": "cpu",
            "white_box_flags": {},
        },
        identity={
            "model_sha256": _EXACT_MODEL_SHA,
            "tokenizer_sha256": _EXACT_TOKENIZER_SHA,
            "template_fingerprint": _EXACT_TEMPLATE,
            "engine_build": "fixture-build",
            "context_size": 128,
            "backend": "cpu",
            "white_box_flags": {},
        },
    )


def _install_exact_continuation_fakes(monkeypatch, tmp_path, engine):
    from clozn.replay import checkpoint_capture, checkpoint_pin_store
    from clozn.replay import time_machine_continuation_results

    monkeypatch.setattr(cs, "SUB", ExactContinuationSub(engine))
    monkeypatch.setattr(time_machine_continuation_results, "RESULTS_DIR", str(tmp_path / "tm-results"))
    monkeypatch.setattr(checkpoint_pin_store, "resolve_pin", lambda _run_id: {
        "ok": True,
        "manifest": {
            "pin_id": "pin_0123456789abcdef0123",
            "source": {"worker_generation_id": "generation-source"},
            "blob": {"sha256": "e" * 64},
        },
        "envelope": {"payload_sha256": _EXACT_PAYLOAD},
    })

    def capture(source, _engine, *, material_out, **_kwargs):
        material_out.update({
            "historical_token_ids": [1, 2, 3],
            "checkpoint_response": {"payload_sha256": _EXACT_PAYLOAD},
            "capture_regime": "verified_prompt_boundary_reprefill",
            "checkpoint_provenance": "durable_pin_import",
            "sampler": {"mode": "greedy", "config_sha256": "ignored", "rng_draws": 0},
            "sampler_wire": None,
            "steering": {"mode": "none", "provenance": "recorded_none"},
        })
        return {
            "status": "available",
            "checkpoint_reference_id": "checkpoint_ref_fixture",
            "checkpoint_reference": {
                "checkpoint_id": "checkpoint-imported",
                "worker_generation_id": "generation-exact",
                "prompt_tokens": 2,
                "n_past": 3,
            },
        }

    monkeypatch.setattr(checkpoint_capture, "capture_parent_checkpoint", capture)


# ===================================================================================================
# the gate -- default OFF, config round-trips, honest stats
# ===================================================================================================
def test_mode_defaults_off(iso):
    out = _get("/timetravel/mode")
    assert out["enabled"] is False
    assert out["cap"] == timetravel.DEFAULT_CAP
    assert out["budget_mb"] == timetravel.DEFAULT_BUDGET_MB


def test_mode_toggle_persists(iso):
    out = _post("/timetravel/mode", {"enabled": True})
    assert out["enabled"] is True and out["changed"] is True
    assert timetravel.enabled() is True
    assert _get("/timetravel/mode")["enabled"] is True
    out = _post("/timetravel/mode", {"enabled": False})
    assert out["enabled"] is False
    assert timetravel.enabled() is False


def test_mode_sets_ring_config(iso):
    out = _post("/timetravel/mode", {"cap": 4, "budget_mb": 64})
    assert out["cap"] == 4 and out["budget_mb"] == 64
    assert timetravel.get_config() == {"cap": 4, "budget_mb": 64}


def test_mode_config_change_reconfigures_live_store(iso):
    # seed the live store with 5 snapshots, then shrink the cap via the endpoint -> evicts to the last 3
    store = cs._snap_store()
    for turn in range(5):
        store.snapshot_turn("run_x", turn, n_tok=10, nbytes=100)
    out = _post("/timetravel/mode", {"cap": 3})
    assert out["cap"] == 3
    assert out["store"]["cap"] == 3                       # the LIVE store adopted the new cap
    assert cs._snap_store().turns_for("run_x") == [2, 3, 4]


def test_stats_reports_empty_store(iso):
    out = _post("/timetravel/stats", {})
    assert out["enabled"] is False
    assert out["snapshots"] == 0 and out["mb"] == 0.0


def test_mode_reports_store_stats(iso):
    # put a couple of snapshots in the process store, then read them back through the endpoint
    store = cs._snap_store()
    store.snapshot_turn("run_x", 0, n_tok=10, nbytes=1048576)
    store.snapshot_turn("run_x", 1, n_tok=10, nbytes=1048576)
    out = _get("/timetravel/mode")
    assert out["store"]["snapshots"] == 2
    assert out["store"]["mb"] == 2.0


def test_time_machine_reports_structural_replay_and_turns(iso):
    rid = _seed_parent()
    out = _get("/runs/" + rid + "/time-machine")
    assert out["schema_version"] == "clozn.time-machine-eligibility.v1"
    assert out["state"] == "structurally_reproducible"
    assert out["eligible"] is True
    assert out["exact_replay"]["eligible"] is False
    assert [turn["turn"] for turn in out["turns"]] == [0, 1, 2]
    assert all(turn["branch_eligible"] for turn in out["turns"])
    assert all(turn["replay_fidelity"] == "structurally_reproducible" for turn in out["turns"])
    assert out["turns"][0]["source"]["status"] == "unavailable"
    assert out["turns"][-1]["source"] == {
        "status": "available",
        "run_id": rid,
        "scope": "full_run_prompt_boundary",
        "source_turn": 2,
        "durable_pin": {
            "status": "unavailable",
            "reason": {
                "code": "durable_pin_missing",
                "message": (
                    "no durable checkpoint is recorded for this source run; restart-safe hydration "
                    "is unavailable until an explicit pin succeeds"),
            },
        },
        "reasons": [{
            "code": "requested_run_source_resolved",
            "message": "the requested immutable run backs its latest completed prompt boundary",
        }],
    }
    schemas.validate(out, "clozn.time-machine-eligibility.v1")


def test_time_machine_reports_snapshot_without_overclaiming_exactness(iso):
    rid = _seed_parent()
    cs._snap_store().snapshot_turn(rid, 1, n_tok=10, kv=((_FakeT(), _FakeT()),))
    out = _get("/runs/" + rid + "/time-machine")
    turn = out["turns"][1]
    assert turn["snapshot"]["has_cache"] is True
    assert turn["exact_replay_eligible"] is False
    assert any(reason["code"] == "exact_restore_pending" for reason in out["reasons"])


def test_resolve_turn_source_run_requires_an_exact_session_prefix(iso):
    rid, turn_zero_id, turn_one_id = _seed_session_history()
    parent = runlog.get_run(rid)
    assert timetravel.resolve_turn_source_run(parent, 0)["id"] == turn_zero_id
    assert timetravel.resolve_turn_source_run(parent, 1)["id"] == turn_one_id
    assert timetravel.resolve_turn_source_run(parent, 2) is None


def test_time_machine_projects_each_exact_source_and_restart_safe_pin_state(iso, monkeypatch):
    rid, turn_zero_id, turn_one_id = _seed_session_history()
    from clozn.replay import checkpoint_pin_store
    monkeypatch.setattr(
        checkpoint_pin_store,
        "get_pin",
        lambda run_id: {
            "pin_id": "pin_source_turn_zero",
            "pinned_at": "2026-08-02T00:00:00Z",
            "blob": {"kv_bytes": 2048, "envelope_bytes": 3072},
        } if run_id == turn_zero_id else None,
    )
    out = _get("/runs/" + rid + "/time-machine")
    first, second, latest = out["turns"]
    assert first["source"]["status"] == "available"
    assert first["source"]["run_id"] == turn_zero_id
    assert first["source"]["scope"] == "session_turn_prompt_boundary"
    assert first["source"]["durable_pin"] == {
        "status": "stored",
        "reason": {
            "code": "durable_pin_recorded",
            "message": (
                "a durable checkpoint is recorded for this source run; the next exact action will "
                "re-read its bytes and fail closed if integrity or runtime compatibility does not match"),
        },
        "pin": {
            "pin_id": "pin_source_turn_zero",
            "pinned_at": "2026-08-02T00:00:00Z",
            "kv_bytes": 2048,
            "envelope_bytes": 3072,
        },
    }
    assert second["source"]["run_id"] == turn_one_id
    assert second["source"]["durable_pin"]["status"] == "unavailable"
    assert latest["source"]["run_id"] == rid
    assert latest["source"]["scope"] == "full_run_prompt_boundary"
    schemas.validate(out, "clozn.time-machine-eligibility.v1")


def test_time_machine_projects_separately_stored_latest_response_as_completed_turn(iso):
    import clozn.runs.store as runlog

    rid = runlog.record(
        source="openai",
        client="fixture",
        model="fixture-model",
        messages=[{"role": "user", "content": "latest question"}],
        response="latest answer",
        session_key="session_latest_response",
    )
    run = runlog.get_run(rid)
    assert run["messages"] == [{"role": "user", "content": "latest question"}]
    assert run["response"] == "latest answer"

    out = _get("/runs/" + rid + "/time-machine")
    assert len(out["turns"]) == 1
    assert out["turns"][0]["turn"] == 0
    assert out["turns"][0]["source"]["status"] == "available"
    assert out["turns"][0]["source"]["run_id"] == rid
    schemas.validate(out, "clozn.time-machine-eligibility.v1")


def test_exact_appended_turn_continuation_persists_child_and_closed_receipt(iso, monkeypatch):
    engine = ExactContinuationEngine()
    _install_exact_continuation_fakes(monkeypatch, iso, engine)
    parent_id = _seed_exact_continuation_source()
    parent_before = runlog.get_run(parent_id)

    out = _post(f"/runs/{parent_id}/time-machine/continue", {
        "turn": 0,
        "user": {"content": "new question"},
        "max_tokens": 12,
    })

    assert out["status"] == "completed"
    assert out["exactness"]["append_only_execution"] is True
    assert out["exactness"]["historical_prefix_recomputed"] is False
    assert out["source_checkpoint"]["provenance"] == "durable_pin_import"
    assert out["source_checkpoint"]["source_worker_generation_id"] == "generation-source"
    assert out["source_checkpoint"]["executing_worker_generation_id"] == "generation-exact"
    assert out["append"]["append_token_ids"] == [9, 10]
    assert "new question" not in json.dumps(out)
    schemas.validate(out, "clozn.time-machine-continuation.v1")

    child = runlog.get_run(out["child_lineage"]["child_run_id"])
    assert child["parent_run_id"] == parent_id
    assert child["messages"][-1] == {"role": "user", "content": "new question"}
    assert child["response"] == "next answer"
    assert child["trace"]["token_ids"] == [55]
    assert child["trace"]["tokens"] == ["next answer"]
    assert child["time_machine_continuation"] == out
    assert runlog.get_run(parent_id) == parent_before
    assert engine.continuations[0]["append_token_ids"] == [9, 10]
    assert set(engine.continuations[0]) == {
        "checkpoint_id", "worker_generation_id", "expected_n_past",
        "expected_token_history_sha256", "expected_checkpoint_payload_sha256",
        "append_token_ids", "append_token_ids_sha256", "max_tokens", "request_id",
        "checkpoint_on_finish",
    }


def test_exact_appended_turn_requires_a_valid_durable_pin(iso, monkeypatch):
    from clozn.replay import checkpoint_capture, checkpoint_pin_store
    from clozn.replay import time_machine_continuation_results

    engine = ExactContinuationEngine()
    monkeypatch.setattr(cs, "SUB", ExactContinuationSub(engine))
    monkeypatch.setattr(time_machine_continuation_results, "RESULTS_DIR", str(iso / "tm-results"))
    monkeypatch.setattr(
        checkpoint_pin_store, "resolve_pin",
        lambda _run_id: {"unavailable": "pinned checkpoint blob corrupt (digest mismatch)"},
    )
    captures = []
    monkeypatch.setattr(
        checkpoint_capture, "capture_parent_checkpoint",
        lambda *_args, **_kwargs: captures.append(True),
    )
    parent_id = _seed_exact_continuation_source()

    out = _post(f"/runs/{parent_id}/time-machine/continue", {
        "turn": 0,
        "user": {"content": "new question"},
        "max_tokens": 12,
    })

    assert out["status"] == "unavailable"
    assert out["failure"]["stage"] == "checkpoint"
    assert out["failure"]["code"] == "checkpoint_corrupt"
    assert captures == []
    assert engine.continuations == []
    schemas.validate(out, "clozn.time-machine-continuation.v1")


def test_exact_appended_turn_fails_on_token_prefix_drift_without_child(iso, monkeypatch):
    engine = ExactContinuationEngine(prefix_matches=False)
    _install_exact_continuation_fakes(monkeypatch, iso, engine)
    parent_id = _seed_exact_continuation_source()
    before_ids = {run["id"] for run in runlog.iter_runs()}

    out = _post(f"/runs/{parent_id}/time-machine/continue", {
        "turn": 0,
        "user": {"content": "new question"},
        "max_tokens": 12,
    })

    assert out["status"] == "failed"
    assert out["failure"]["stage"] == "append_derivation"
    assert out["failure"]["code"] == "append_prefix_mismatch"
    assert {run["id"] for run in runlog.iter_runs()} == before_ids
    assert engine.continuations == []
    schemas.validate(out, "clozn.time-machine-continuation.v1")


def test_time_machine_verifies_an_earlier_session_turn(iso, monkeypatch):
    rid, turn_zero_id, _ = _seed_session_history()
    from clozn.replay import checkpoint_capture, timetravel_results
    monkeypatch.setattr(timetravel_results, "RESULTS_DIR", str(iso / "time-machine-results"))
    monkeypatch.setattr(
        checkpoint_capture,
        "capture_parent_checkpoint",
        lambda parent, *args, **kwargs: {
            "status": "available",
            "parent_run_id": parent["id"],
            "parent_fingerprint_sha256": "c" * 64,
            "checkpoint_reference_id": "checkpoint_ref_session",
            "proof": {"status": "matched"},
            "reasons": [{"code": "exact_checkpoint_captured", "message": "matched"}],
        },
    )
    from clozn.server.routes import execution_fork
    monkeypatch.setattr(execution_fork, "_parent_sub_facts", lambda *args: ({}, {}, object()))
    out = _post("/runs/" + rid + "/time-machine/verify", {"turn": 0})
    assert out["status"] == "verified"
    assert out["scope"] == "session_turn_prompt_boundary"
    assert out["requested_run_id"] == rid
    assert out["source_run_id"] == turn_zero_id
    assert out["parent_run_id"] == turn_zero_id
    assert timetravel_results.latest_for_run(rid, 0)["verification_id"] == out["verification_id"]
    schemas.validate(out, "clozn.time-machine-verification.v1")


def test_time_machine_verification_hydrates_a_durable_source_pin(iso, monkeypatch):
    rid, turn_zero_id, _ = _seed_session_history()
    from clozn.replay import checkpoint_capture, checkpoint_pin_store, timetravel_results
    monkeypatch.setattr(timetravel_results, "RESULTS_DIR", str(iso / "time-machine-results"))
    seen = {}

    def capture(parent, *args, **kwargs):
        seen["parent_id"] = parent["id"]
        seen["envelope"] = kwargs.get("checkpoint_envelope")
        return {
            "status": "available",
            "parent_run_id": parent["id"],
            "parent_fingerprint_sha256": "d" * 64,
            "checkpoint_reference_id": "checkpoint_ref_pinned",
            "proof": {"status": "matched"},
            "reasons": [{"code": "exact_checkpoint_captured", "message": "hydrated"}],
        }

    monkeypatch.setattr(checkpoint_capture, "capture_parent_checkpoint", capture)
    monkeypatch.setattr(
        checkpoint_pin_store,
        "resolve_pin",
        lambda run_id: {
            "ok": True,
            "envelope": {"envelope_version": "clozn.checkpoint-export.v1", "state": {}},
        } if run_id == turn_zero_id else {"unavailable": "no pin"},
    )
    from clozn.server.routes import execution_fork
    monkeypatch.setattr(execution_fork, "_parent_sub_facts", lambda *args: ({}, {}, object()))
    out = _post("/runs/" + rid + "/time-machine/verify", {"turn": 0})
    assert out["status"] == "verified"
    assert seen == {
        "parent_id": turn_zero_id,
        "envelope": {"envelope_version": "clozn.checkpoint-export.v1", "state": {}},
    }
    schemas.validate(out, "clozn.time-machine-verification.v1")


def test_time_machine_exact_branch_rejects_an_edited_question(iso):
    rid = _seed_parent()
    out = _post("/runs/" + rid + "/time-machine/branch", {"turn": 0, "alt_user": "different"})
    assert out["code"] == "time_machine_branch_options_unsupported"


def test_time_machine_exact_branch_reports_missing_historical_source(iso):
    rid = _seed_parent()
    out = _post("/runs/" + rid + "/time-machine/branch", {"turn": 0})
    assert out["schema_version"] == "clozn.time-machine-branch.v1"
    assert out["status"] == "unavailable"
    assert out["exact_replay"] is False
    schemas.validate(out, "clozn.time-machine-branch.v1")


def test_time_machine_exact_branch_persists_a_same_prompt_child(iso, monkeypatch):
    rid = _seed_parent()
    from clozn.replay import checkpoint_capture, execution_fork, execution_fork_execute
    monkeypatch.setattr(
        checkpoint_capture,
        "capture_parent_checkpoint",
        lambda parent, *args, **kwargs: {
            "status": "available",
            "parent_run_id": parent["id"],
            "checkpoint_reference": {
                "checkpoint_id": "ckpt-test",
                "worker_generation_id": "worker-generation-test",
                "state": "available",
                "parent_run_id": parent["id"],
                "prompt_tokens": 3,
                "n_past": 6,
            },
            "reasons": [{"code": "exact_checkpoint_captured", "message": "matched"}],
        },
    )
    monkeypatch.setattr(
        execution_fork,
        "plan_execution_fork",
        lambda *args, **kwargs: {
            "classification": "exact_execution_fork",
            "plan_id": "fork_plan_0123456789abcdef0123",
        },
    )
    monkeypatch.setattr(
        execution_fork_execute,
        "execute_exact_fork",
        lambda *args, **kwargs: {
            "receipt": {
                "phase": "completed",
                "execution_id": "fork_exec_0123456789abcdef0123",
            },
            "child": {"id": "run_exact_child_0123456789"},
        },
    )
    from clozn.server.routes import execution_fork as execution_fork_route
    monkeypatch.setattr(
        execution_fork_route,
        "_parent_sub_facts",
        lambda *args: ({"runtime": "facts"}, {"worker": "facts"}, object()),
    )
    out = _post("/runs/" + rid + "/time-machine/branch", {"turn": 2})
    assert out["status"] == "completed"
    assert out["exact_replay"] is True
    assert out["source_run_id"] == rid
    assert out["child_run_id"] == "run_exact_child_0123456789"
    assert out["execution_fork_execution_id"] == "fork_exec_0123456789abcdef0123"
    schemas.validate(out, "clozn.time-machine-branch.v1")


def test_time_machine_missing_run_404s(iso):
    out = _get("/runs/run_nope/time-machine")
    assert "error" in out and "not found" in out["error"]


def test_time_machine_verification_reports_earlier_turn_unavailable(iso):
    rid = _seed_parent()
    out = _post("/runs/" + rid + "/time-machine/verify", {"turn": 0})
    assert out["schema_version"] == "clozn.time-machine-verification.v1"
    assert out["status"] == "unavailable"
    assert out["exact_replay"] is False
    assert out["reasons"][0]["code"] == "turn_state_not_available"
    schemas.validate(out, "clozn.time-machine-verification.v1")


def test_time_machine_verification_persists_verified_prompt_boundary(iso, monkeypatch):
    rid = _seed_parent()
    from clozn.replay import checkpoint_capture, timetravel_results
    monkeypatch.setattr(timetravel_results, "RESULTS_DIR", str(iso / "time-machine-results"))
    monkeypatch.setattr(
        checkpoint_capture,
        "capture_parent_checkpoint",
        lambda *args, **kwargs: {
            "status": "available",
            "parent_fingerprint_sha256": "a" * 64,
            "checkpoint_reference_id": "checkpoint_ref_test",
            "proof": {"status": "matched"},
            "reasons": [{"code": "exact_checkpoint_captured", "message": "matched"}],
        },
    )
    from clozn.server.routes import execution_fork
    monkeypatch.setattr(execution_fork, "_parent_sub_facts", lambda *args: ({}, {}, object()))
    out = _post("/runs/" + rid + "/time-machine/verify", {"turn": 2})
    assert out["status"] == "verified"
    assert out["exact_replay"] is True
    assert out["fidelity"] == "exact_replay_eligible"
    assert timetravel_results.latest_for_run(rid, 2)["verification_id"] == out["verification_id"]
    schemas.validate(out, "clozn.time-machine-verification.v1")


def test_time_machine_eligibility_exposes_prior_proof_without_overclaiming(iso, monkeypatch):
    rid = _seed_parent()
    from clozn.replay import checkpoint_capture, timetravel_results
    monkeypatch.setattr(timetravel_results, "RESULTS_DIR", str(iso / "time-machine-results"))
    monkeypatch.setattr(
        checkpoint_capture,
        "capture_parent_checkpoint",
        lambda *args, **kwargs: {
            "status": "available",
            "parent_fingerprint_sha256": "b" * 64,
            "checkpoint_reference_id": "checkpoint_ref_test",
            "proof": {"status": "matched"},
            "reasons": [{"code": "exact_checkpoint_captured", "message": "matched"}],
        },
    )
    from clozn.server.routes import execution_fork
    monkeypatch.setattr(execution_fork, "_parent_sub_facts", lambda *args: ({}, {}, object()))
    verified = _post("/runs/" + rid + "/time-machine/verify", {"turn": 2})
    out = _get("/runs/" + rid + "/time-machine")
    turn = out["turns"][2]
    assert turn["last_verification"]["verification_id"] == verified["verification_id"]
    assert turn["last_verification"]["exact_replay"] is True
    assert turn["exact_replay_eligible"] is False
    schemas.validate(out, "clozn.time-machine-eligibility.v1")


def test_unknown_timetravel_route_404s(iso):
    out = _post("/timetravel/bogus", {})
    assert "error" in out


# ===================================================================================================
# branch -> child run
# ===================================================================================================
def test_branch_records_child_from_truncated_transcript(iso):
    rid = _seed_parent()
    child = _post("/runs/" + rid + "/branch", {"turn": 1})
    assert child["parent_run_id"] == rid
    assert child["changes_applied"]["branch_turn"] == 1
    assert child["changes_applied"]["edited_user"] is False
    assert child["source"] == "branch"
    assert "replayed" in child["flags"]
    # the fake saw the TRUNCATED transcript (turn 1's user, nothing later)
    assert cs.SUB.seen["messages"][-1]["content"] == "u1"
    assert len(cs.SUB.seen["messages"]) == 3
    assert cs.SUB.seen["sample"] is False        # greedy by default (the receipt path)


def test_branch_with_alt_user(iso):
    rid = _seed_parent()
    child = _post("/runs/" + rid + "/branch", {"turn": 1, "alt_user": "something else entirely"})
    assert child["changes_applied"]["edited_user"] is True
    assert child["changes_applied"]["alt_user"] == "something else entirely"
    assert cs.SUB.seen["messages"][-1] == {"role": "user", "content": "something else entirely"}


def test_branch_flags_kv_snapshot_when_present(iso):
    rid = _seed_parent()
    # a real-cache snapshot for (this run, turn 1) flips the kv_snapshot note
    cs._snap_store().snapshot_turn(rid, 1, n_tok=10, kv=((_FakeT(), _FakeT()),))
    child = _post("/runs/" + rid + "/branch", {"turn": 1})
    assert child["changes_applied"]["kv_snapshot"] is True


def test_branch_sample_true_when_requested(iso):
    rid = _seed_parent()
    _post("/runs/" + rid + "/branch", {"turn": 0, "sample": True})
    assert cs.SUB.seen["sample"] is True


def test_branch_missing_run_404(iso):
    out = _post("/runs/run_nope/branch", {"turn": 0})
    assert "error" in out and "not found" in out["error"]


def test_branch_absent_turn_400(iso):
    rid = _seed_parent()
    out = _post("/runs/" + rid + "/branch", {})
    assert "error" in out and "turn" in out["error"]


def test_branch_bad_turn_type_400(iso):
    rid = _seed_parent()
    out = _post("/runs/" + rid + "/branch", {"turn": "abc"})
    assert "error" in out


def test_branch_out_of_range_400(iso):
    rid = _seed_parent()
    out = _post("/runs/" + rid + "/branch", {"turn": 99})
    assert "error" in out                      # timetravel.branch returns None -> 400


def test_branch_no_substrate_503(iso, monkeypatch):
    rid = _seed_parent()
    monkeypatch.setattr(cs, "SUB", None)
    out = _post("/runs/" + rid + "/branch", {"turn": 0})
    assert "error" in out and "worker" in out["error"]


def test_branch_does_not_mutate_live_studio(iso):
    steer = _FakeSteer({"concise": 0.4})
    mem = _FakeMem(strength=1.3)
    cs.SUB.steer = steer
    cs.SUB.memory = mem
    cs.SUB._mem = mem
    rid = _seed_parent()
    _post("/runs/" + rid + "/branch", {"turn": 1})
    assert steer.saved is False
    assert steer.strength == {"concise": 0.4}   # restored
    assert mem.memory_strength == 1.3


def test_branch_child_is_fetchable_via_runs_endpoint(iso):
    rid = _seed_parent()
    child = _post("/runs/" + rid + "/branch", {"turn": 1})
    fetched = _get("/runs/" + child["id"])
    assert fetched["parent_run_id"] == rid
    assert fetched["source"] == "branch"


class _FakeT:
    """A tiny duck-typed tensor so a snapshot payload sizes to nonzero bytes without torch."""

    def element_size(self):
        return 2

    def nelement(self):
        return 100


# ===================================================================================================
# per-turn snapshot registration in the chat log path (_maybe_snapshot_turn) -- gated OFF by default
# ===================================================================================================
def _handler():
    """A no-socket handler instance to call _maybe_snapshot_turn on directly."""
    H = cs.make_handler()
    h = object.__new__(H)
    h.headers = {"User-Agent": "pytest"}
    return h


def test_no_snapshot_registered_when_gate_off(iso):
    h = _handler()
    trace = [{"piece": "a"}, {"piece": "b"}]              # 2 tokens
    h._maybe_snapshot_turn("run_z", CONV, trace, None)
    assert cs._snap_store().count() == 0                  # gate OFF -> nothing recorded


def test_descriptor_snapshot_registered_when_gate_on(iso):
    timetravel.set_enabled(True)
    h = _handler()
    trace = [{"piece": "a"}, {"piece": "b"}, {"piece": "c"}]   # 3 tokens
    h._maybe_snapshot_turn("run_z", CONV, trace, None)
    store = cs._snap_store()
    assert store.count() == 1
    snap = store.latest("run_z")
    assert snap.turn == 2                                 # CONV has 3 turns -> this reply is turn index 2
    assert snap.n_tok == 3                                # from the raw step-list length
    assert snap.has_cache is False and snap.nbytes == 0  # stateless path -> descriptor only
    assert snap.descriptor().get("stateless") is True


def test_snapshot_skipped_on_error_run(iso):
    timetravel.set_enabled(True)
    h = _handler()
    h._maybe_snapshot_turn("run_z", CONV, [{"piece": "a"}], "boom")
    assert cs._snap_store().count() == 0                  # an errored run isn't snapshotted


def test_snapshot_ring_is_bounded_in_the_log_path(iso):
    timetravel.set_enabled(True)
    timetravel.set_config(cap=3)
    cs.SNAPSHOTS = None                                   # rebuild the store with the new cap
    h = _handler()
    for turn in range(6):
        msgs = []
        for k in range(turn + 1):                        # a transcript with turn+1 user/assistant pairs
            msgs += [{"role": "user", "content": f"u{k}"}, {"role": "assistant", "content": f"a{k}"}]
        h._maybe_snapshot_turn("run_ring", msgs, [{"piece": "x"}], None)
    store = cs._snap_store()
    assert store.turns_for("run_ring") == [3, 4, 5]      # only the last 3 turns survive (cap=3)
