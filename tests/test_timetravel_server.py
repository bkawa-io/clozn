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
