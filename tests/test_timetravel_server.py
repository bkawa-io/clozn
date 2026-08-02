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
