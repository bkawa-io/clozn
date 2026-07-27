"""No-model tests for research/replay.py (roadmap issue F1 / I3).

Exercise the replay engine against a FAKE substrate (no 7B, no PyTorch): a stub with a canned .chat(), a
mutable .memory.memory_strength, and a .steer whose .strength dict is driven exactly like the real
SteeringControl (per-axis caps + set/clear/active). The store is isolated to a pytest tmp dir via
runlog.RUNS_DIR (a module global) -- same trick as test_runlog.py.

Asserts the two invariants that keep the LIVE studio safe:
  (a) memory_off drives memory_strength to 0 DURING chat, and restores it exactly AFTER;
  (b) behavior_off clears the dials DURING chat, and restores them exactly AFTER;
plus that a child run is recorded with parent_run_id + changes_applied, and that save_state is never called.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/ on path
import clozn.memory.mode as memory_mode  # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
from clozn import replay  # noqa: E402
import clozn.runs.store as runlog  # noqa: E402


# --- a fake substrate that mirrors the real surfaces replay() touches -------------------------------------

AX_MAX = {"concise": 1.5, "warm": 1.5, "candid": 0.45}   # a couple of per-axis caps like steering.AXES


class FakeSteer:
    """Mimics SteeringControl's dial surface: .strength dict, set() (capped), clear(), active(), and a
    save_state() we assert is NEVER called during replay (persisting temp dials would corrupt personality)."""

    def __init__(self, strength=None):
        self.strength = dict(strength or {})
        self.saved = False                                 # flips true if save_state is (wrongly) called

    def set(self, name, value):
        mx = AX_MAX.get(name, 1.5)
        self.strength[name] = max(-mx, min(mx, float(value)))

    def clear(self):
        self.strength = {}

    def active(self):
        return {k: v for k, v in self.strength.items() if v}

    def save_state(self, path):
        self.saved = True


class FakeMem:
    def __init__(self, strength=1.0, rules=None, prefix="PFX"):
        self.memory_strength = float(strength)
        self.rules = list(rules or [])
        self.prefix = prefix


class FakeSub:
    """A stub substrate. .chat() records the state it SAW at call time (so we can assert the change was live
    during generation) and returns a reply that echoes that state (so replies differ by change)."""

    def __init__(self, mem=None, steer=None):
        self.memory = mem if mem is not None else FakeMem()
        self.steer = steer if steer is not None else FakeSteer()
        self.seen = {}                                     # snapshot of what chat observed
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls += 1
        self.seen = {"memory_strength": self.memory.memory_strength,
                     "dials": dict(self.steer.strength), "max_new": max_new, "sample": sample}
        if trace_out is not None:                          # mirror the real chat: fill the per-token trace
            trace_out.extend([{"piece": "re", "conf": 0.9, "alts": []},
                              {"piece": "ply", "conf": 0.7, "alts": []}])
        return (f"reply mem={self.memory.memory_strength} "
                f"dials={sorted(self.steer.strength.items())}")


RUN = {"id": "run_parent0", "model": "clozn-qwen", "substrate": "QwenSubstrate",
       "messages": [{"role": "user", "content": "hello there"}]}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Isolated run log + memory mode PINNED to internalized: this suite asserts the prefix-era replay
    semantics (whole-memory suppression; per-card ids an honest "not applied" note), which the mode swap
    keeps intact. Prompt-mode replay (REAL per-card ablation) is covered in test_memory_mode."""
    monkeypatch.setenv("CLOZN_RUNTIME_KIND", "lab")   # internalized/soft-prefix memory is a LAB feature now
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    assert memory_mode.set_mode("internalized")
    original = runlog.RUNS_DIR
    runlog.RUNS_DIR = str(tmp_path / "runs")
    try:
        yield runlog
    finally:
        runlog.RUNS_DIR = original


# --- (a) memory_off: strength 0 during chat, restored after -----------------------------------------------

def test_memory_off_suppresses_then_restores(store):
    sub = FakeSub(mem=FakeMem(strength=1.0))
    child = replay.replay(RUN, {"memory_off": True}, sub)
    assert child is not None
    assert sub.seen["memory_strength"] == 0.0              # suppressed DURING generation
    assert sub.memory.memory_strength == 1.0               # restored EXACTLY afterward
    assert child["parent_run_id"] == "run_parent0"
    assert child["changes_applied"] == {"memory_off": True}
    assert child["memory"]["strength"] == 0.0
    assert child["memory"]["cards_applied"] == []


def test_memory_off_restores_nonstandard_strength(store):
    sub = FakeSub(mem=FakeMem(strength=1.7))               # a non-default live strength must come back verbatim
    replay.replay(RUN, {"memory_off": True}, sub)
    assert sub.memory.memory_strength == 1.7


# --- (b) behavior_off: dials cleared during chat, restored after ------------------------------------------

def test_behavior_off_clears_then_restores(store):
    sub = FakeSub(steer=FakeSteer({"concise": 0.8, "warm": 0.3}))
    child = replay.replay(RUN, {"behavior_off": True}, sub)
    assert child is not None
    assert sub.seen["dials"] == {}                         # neutral DURING generation
    assert sub.steer.strength == {"concise": 0.8, "warm": 0.3}   # restored EXACTLY afterward
    assert child["behavior"]["active_dials"] == {}
    assert sub.steer.saved is False                        # temp dials were NOT persisted


# --- nudge: bump one dial toward + pole, capped, on top of current, then restore --------------------------

def test_nudge_bumps_capped_and_restores(store):
    sub = FakeSub(steer=FakeSteer({"concise": 0.3}))
    child = replay.replay(RUN, {"nudge": "concise"}, sub)
    assert sub.seen["dials"]["concise"] == pytest.approx(0.8)    # 0.3 + 0.5 step
    assert child["behavior"]["active_dials"]["concise"] == pytest.approx(0.8)
    assert sub.steer.strength == {"concise": 0.3}          # restored


def test_nudge_respects_axis_cap(store):
    sub = FakeSub(steer=FakeSteer({"candid": 0.4}))        # candid caps at 0.45
    child = replay.replay(RUN, {"nudge": "candid"}, sub)
    assert sub.seen["dials"]["candid"] == pytest.approx(0.45)    # 0.4 + 0.5 -> capped to 0.45


# --- behavior_overrides: set specific dials for this run, restore after -----------------------------------

def test_behavior_overrides_set_then_restore(store):
    sub = FakeSub(steer=FakeSteer({"warm": 0.2}))
    child = replay.replay(RUN, {"behavior_overrides": {"concise": 0.8}}, sub)
    assert sub.seen["dials"]["concise"] == pytest.approx(0.8)
    assert child["behavior"]["active_dials"]["concise"] == pytest.approx(0.8)
    assert sub.steer.strength == {"warm": 0.2}             # restored to the ORIGINAL, override gone


# --- plain re-roll: unchanged, still a child run ----------------------------------------------------------

def test_plain_reroll_records_child(store):
    sub = FakeSub(mem=FakeMem(strength=1.0), steer=FakeSteer({"concise": 0.4}))
    child = replay.replay(RUN, {}, sub)
    assert child is not None
    assert sub.seen["memory_strength"] == 1.0             # unchanged
    assert sub.seen["dials"] == {"concise": 0.4}          # unchanged
    assert child["parent_run_id"] == "run_parent0"
    assert child["changes_applied"] == {}


class _ScopedSub(FakeSub):
    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None,
             memory_scope=None):
        self.seen_scope = memory_scope
        return super().chat(messages, max_new=max_new, sample=sample, trace_out=trace_out,
                            mem_out=mem_out)


def test_replay_reuses_only_explicit_client_scope_and_inherits_project_on_child(store):
    parent = {
        **RUN,
        "client_key": "client_0123456789abcdef01234567",
        "client_key_source": "header",
        "project_key": "project_0123456789abcdef01234567",
    }
    sub = _ScopedSub()

    child = replay.replay(parent, {}, sub)

    assert sub.seen_scope.app_key == parent["client_key"]
    assert sub.seen_scope.project_key == parent["project_key"]
    assert child["client_key"] == parent["client_key"]
    assert child["client_key_source"] == "header"
    assert child["project_key"] == parent["project_key"]


def test_replay_does_not_activate_user_agent_client_scope(store):
    parent = {**RUN, "client_key": "client_ua_fingerprint", "client_key_source": "user_agent"}
    sub = _ScopedSub()

    child = replay.replay(parent, {}, sub)

    assert child is not None
    assert sub.seen_scope.app_key is None


# --- best-effort card toggles: no-op with a note ----------------------------------------------------------

def test_disabled_memory_ids_is_noted_stub(store):
    sub = FakeSub(mem=FakeMem(strength=1.0))
    child = replay.replay(RUN, {"disabled_memory_ids": ["mem_1"]}, sub)
    assert child is not None
    assert sub.seen["memory_strength"] == 1.0            # unchanged (string-based cards -> best-effort no-op)
    assert "disabled_memory_ids" in child["memory"].get("notes", {})


# --- linkage + child is a real, fetchable run --------------------------------------------------------------

def test_child_is_persisted_and_flagged_replayed(store):
    sub = FakeSub()
    child = replay.replay(RUN, {"memory_off": True}, sub)
    fetched = store.get_run(child["id"])
    assert fetched is not None
    assert fetched["parent_run_id"] == "run_parent0"
    assert fetched["source"] == "replay"
    assert fetched["client"] == "studio"
    assert fetched["model"] == "clozn-qwen"
    assert "replayed" in fetched["flags"]                 # runlog._flags sets this from parent_run_id


def test_reply_differs_when_memory_toggled(store):
    """The whole point: a change yields a different reply (here the fake echoes the state it saw)."""
    plain = replay.replay(RUN, {}, FakeSub(mem=FakeMem(strength=1.0)))
    off = replay.replay(RUN, {"memory_off": True}, FakeSub(mem=FakeMem(strength=1.0)))
    assert plain["response"] != off["response"]


# --- per-token trace + repro fields on the child run (the data a baseline-vs-replay diff needs) -----------

def test_replay_captures_the_per_token_trace(store):
    """The gap this closes: replay now passes trace_out to chat and records it, so the child run carries a
    token timeline. Previously replays stored an empty trace and the token diff had nothing to compare."""
    child = replay.replay(RUN, {}, FakeSub())
    assert child["trace"]["tokens"] == ["re", "ply"]
    assert child["trace"]["confidence"] == [0.9, 0.7]


class _MetaSub(FakeSub):
    """A substrate exposing the engine's post-generation stashes (last_finish_reason / run_meta)."""

    def last_finish_reason(self):
        return "length"

    def run_meta(self):
        return {"quant": "Q4_K_M", "sampling": "greedy"}


def test_replay_records_finish_reason_and_meta_when_available(store):
    child = replay.replay(RUN, {}, _MetaSub())
    assert child["finish_reason"] == "length"
    assert child["meta"]["quant"] == "Q4_K_M" and child["meta"]["sampling"] == "greedy"
    assert child["meta"]["capture_tier"] == "standard"     # the tier rides every run's meta
    assert "truncated" in child["flags"]                   # length -> truncated, exactly like a live run
    assert child["warnings"][0]["code"] == "output_truncated"


class _ContextSub(FakeSub):
    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        reply = super().chat(messages, max_new=max_new, sample=sample, trace_out=trace_out,
                             mem_out=mem_out)
        if mem_out is not None:
            mem_out["assembled_messages"] = [{"role": "system", "content": "replay memory"}] + list(messages)
            mem_out["final_prompt"] = "REPLAY EXACT PROMPT"
        return reply


def test_replay_child_retains_its_own_survived_prompt(store):
    child = replay.replay(RUN, {}, _ContextSub())
    assert child["assembled_messages"][0] == {"role": "system", "content": "replay memory"}
    assert child["final_prompt"] == "REPLAY EXACT PROMPT"
    assert child["context_receipt"]["survived"]["final_prompt"] == "REPLAY EXACT PROMPT"


def test_replay_at_light_tier_drops_the_trace(store, monkeypatch):
    """Light tier stores text-only -> the child run's trace is empty even though chat produced one, and the
    tier is recorded. (End-to-end proof of the capture policy through a real runlog.record, no model.)"""
    from clozn.runs import capture_mode
    monkeypatch.setattr(capture_mode, "tier", lambda: "light")
    child = replay.replay(RUN, {}, FakeSub())
    assert child["trace"] == {}
    assert child["meta"]["capture_tier"] == "light"


def test_replay_without_engine_stashes_records_none(store):
    """The HF stub has no last_finish_reason / run_meta -> finish_reason None and no model_file/quant; the
    capture tier is still recorded (it applies to every run, engine or not)."""
    child = replay.replay(RUN, {}, FakeSub())
    assert child["finish_reason"] is None
    assert "model_file" not in child["meta"] and "quant" not in child["meta"]
    assert child["meta"]["capture_tier"] == "standard"


# --- never raises; returns None on a broken substrate -----------------------------------------------------

def test_returns_none_when_no_chat(store):
    class NoChat:
        memory = FakeMem()
        steer = FakeSteer()
    assert replay.replay(RUN, {"memory_off": True}, NoChat()) is None


def test_returns_none_on_chat_exception_and_restores(store):
    class Boom(FakeSub):
        def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
            raise RuntimeError("model exploded")
    sub = Boom(mem=FakeMem(strength=1.3), steer=FakeSteer({"concise": 0.5}))
    assert replay.replay(RUN, {"memory_off": True, "behavior_off": True}, sub) is None
    # even on failure, the live state must be restored (the finally ran)
    assert sub.memory.memory_strength == 1.3
    assert sub.steer.strength == {"concise": 0.5}


def test_returns_none_on_empty_run(store):
    assert replay.replay(None, {}, FakeSub()) is None
