"""test_timetravel -- MODEL-FREE tests for the time-travel debugger.

No model, no GPU, no torch: research/timetravel.py imports only stdlib (torch is lazy, inside the one
live-cache method we don't exercise here). Covers the bounded snapshot store's bookkeeping (ring cap,
byte budget, eviction actually frees the payload, honest accounting), the pure rewind/branch transcript
transforms (message_turns / branch_messages), the on/off gate (DEFAULT OFF -- the RAM rule), and the
branch()->child-run recorder against a FAKE substrate (the replay.py test pattern), asserting the child
carries parent_run_id + changes_applied noting the branch turn and NEVER persists the live knobs.

The determinism RECEIPT (a branch byte-matches a fresh recompute) is proven separately, on the real KV
mechanism, in the gated test_timetravel_determinism.py (@pytest.mark.model).
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.memory.mode as memory_mode  # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
import clozn.runs.store as runlog  # noqa: E402
import clozn.replay.timetravel as tt  # noqa: E402


# ===================================================================================================
# the gate -- default OFF (the RAM rule), round-trips, garbage reads as off
# ===================================================================================================
@pytest.fixture
def iso_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


def test_gate_defaults_off(iso_settings):
    assert tt.enabled() is False


def test_gate_round_trips(iso_settings):
    assert tt.set_enabled(True)
    assert tt.enabled() is True
    assert tt.set_enabled(False)
    assert tt.enabled() is False


def test_gate_garbage_reads_off(iso_settings):
    memory_mode.set_setting(tt._ENABLED_KEY, "banana")
    assert tt.enabled() is False


def test_gate_string_truthy(iso_settings):
    for v in ("on", "true", "1", "yes", "On", "TRUE"):
        memory_mode.set_setting(tt._ENABLED_KEY, v)
        assert tt.enabled() is True, v
    for v in ("off", "false", "0", "no", ""):
        memory_mode.set_setting(tt._ENABLED_KEY, v)
        assert tt.enabled() is False, v


def test_config_defaults_and_clamp(iso_settings):
    cfg = tt.get_config()
    assert cfg == {"cap": tt.DEFAULT_CAP, "budget_mb": tt.DEFAULT_BUDGET_MB}
    # garbage / out-of-range persisted config clamps to the sane band, never breaks the store
    memory_mode.set_setting("timetravel_cap", 9999)
    memory_mode.set_setting("timetravel_budget_mb", 1)
    cfg = tt.get_config()
    assert cfg["cap"] == 128            # clamped to the max
    assert cfg["budget_mb"] == 8        # clamped to the min
    memory_mode.set_setting("timetravel_cap", "nope")
    assert tt.get_config()["cap"] == tt.DEFAULT_CAP   # unparseable -> default


def test_set_config_only_writes_given_keys(iso_settings):
    assert tt.set_config(cap=4)
    assert tt.get_config()["cap"] == 4
    assert tt.get_config()["budget_mb"] == tt.DEFAULT_BUDGET_MB   # untouched
    assert tt.set_config(budget_mb=64)
    assert tt.get_config() == {"cap": 4, "budget_mb": 64}


# ===================================================================================================
# snapshot byte accounting -- the "measure and report it" number, pure
# ===================================================================================================
def test_kv_snapshot_bytes_qwen7b():
    # Qwen2.5-7B-Instruct: 28 layers, 4 KV heads, head_dim 128, bf16 KV -> ~7 MB / 128 tok.
    b = tt.kv_snapshot_bytes(128, n_layers=28, n_kv_heads=4, head_dim=128, bytes_per_elt=2)
    assert b == 28 * 128 * (2 * 4 * 128) * 2
    assert round(b / 1048576, 2) == 7.0
    # linear in tokens
    assert tt.kv_snapshot_bytes(256, 28, 4, 128) == 2 * tt.kv_snapshot_bytes(128, 28, 4, 128)


def test_kv_snapshot_bytes_qwen1p5b():
    # the determinism-test model: 28 layers, 2 KV heads, head_dim 128 -> half the 7B's per-token cost.
    b = tt.kv_snapshot_bytes(256, n_layers=28, n_kv_heads=2, head_dim=128, bytes_per_elt=2)
    assert round(b / 1048576, 2) == 7.0


# ===================================================================================================
# a fake offloaded kv payload (duck-typed tensors) -- exercises _sizeof_kv with no torch
# ===================================================================================================
class FakeTensor:
    """Quacks like a torch tensor for the store's size accounting: element_size() * nelement() bytes."""

    def __init__(self, n_elem: int, elt_bytes: int = 2):
        self._n = int(n_elem)
        self._b = int(elt_bytes)

    def element_size(self):
        return self._b

    def nelement(self):
        return self._n


def fake_kv(n_layers: int, elems_per_tensor: int) -> tuple:
    """A tuple-of-(keys, values) per layer, each a FakeTensor -- the offload_cache output shape."""
    return tuple((FakeTensor(elems_per_tensor), FakeTensor(elems_per_tensor)) for _ in range(n_layers))


def test_sizeof_kv_sums_all_layer_tensors():
    kv = fake_kv(3, 100)                                  # 3 layers x (k+v) x 100 elems x 2 bytes
    assert tt._sizeof_kv(kv) == 3 * 2 * 100 * 2
    assert tt._sizeof_kv(()) == 0
    assert tt._sizeof_kv(None) == 0


def test_sizeof_kv_defensive_on_junk():
    # a payload with non-tensor junk contributes 0 for the junk, never raises
    assert tt._sizeof_kv([(object(), object())]) == 0


# ===================================================================================================
# the bounded ring: per-run cap eviction (oldest turn), payload freed
# ===================================================================================================
def test_store_put_and_read_back():
    s = tt.SnapshotStore(cap=8, budget_mb=512)
    s.snapshot_turn("run_a", 0, n_tok=100, nbytes=1000)
    s.snapshot_turn("run_a", 1, n_tok=200, nbytes=2000)
    assert s.count() == 2
    assert s.total_bytes == 3000
    assert s.get("run_a", 1).n_tok == 200
    assert s.latest("run_a").turn == 1
    assert s.turns_for("run_a") == [0, 1]
    assert s.get("run_a", 99) is None
    assert s.latest("run_missing") is None


def test_store_evicts_oldest_turn_over_cap():
    s = tt.SnapshotStore(cap=3, budget_mb=512)
    kept_refs = []
    for turn in range(6):                                 # 6 turns into a cap-3 window
        kept_refs.append(s.snapshot_turn("run_a", turn, n_tok=10, nbytes=100, kv=fake_kv(1, 10)))
    assert s.turns_for("run_a") == [3, 4, 5]              # only the last 3 turns survive
    assert s.count() == 3
    assert s.total_bytes == 3 * 100
    # the evicted snapshots had their payloads dropped (RAM reclaimed)
    for evicted in kept_refs[:3]:
        assert evicted.kv is None
    for alive in kept_refs[3:]:
        assert alive.kv is not None


def test_store_per_run_windows_are_independent():
    s = tt.SnapshotStore(cap=2, budget_mb=512)
    for turn in range(3):
        s.snapshot_turn("run_a", turn, n_tok=10, nbytes=100)
        s.snapshot_turn("run_b", turn, n_tok=10, nbytes=100)
    assert s.turns_for("run_a") == [1, 2]                 # each run keeps its own last-2
    assert s.turns_for("run_b") == [1, 2]
    assert s.count() == 4


# ===================================================================================================
# the global byte budget: evict globally-oldest until under, across runs
# ===================================================================================================
def test_store_evicts_over_byte_budget():
    # budget 1 MB; each snapshot ~0.4 MB -> only 2 fit, the 3rd evicts the globally-oldest.
    one_mb = 1048576
    s = tt.SnapshotStore(cap=100, budget_mb=1)            # cap huge so ONLY the byte budget bites
    a = s.snapshot_turn("run_a", 0, n_tok=1, nbytes=int(0.4 * one_mb))
    b = s.snapshot_turn("run_b", 0, n_tok=1, nbytes=int(0.4 * one_mb))
    assert s.count() == 2 and s.total_bytes <= one_mb
    c = s.snapshot_turn("run_c", 0, n_tok=1, nbytes=int(0.4 * one_mb))
    assert s.total_bytes <= one_mb                        # stayed under the ceiling
    assert s.count() == 2
    assert a.kv is None or s.get("run_a", 0) is None      # the oldest (a) was evicted
    assert s.get("run_c", 0) is c                         # the newest survived


def test_store_stats_reports_honest_totals():
    s = tt.SnapshotStore(cap=8, budget_mb=256)
    s.snapshot_turn("run_a", 0, n_tok=100, nbytes=1048576)       # exactly 1 MB
    s.snapshot_turn("run_a", 1, n_tok=100, nbytes=1048576)
    st = s.stats()
    assert st["snapshots"] == 2
    assert st["runs"] == 1
    assert st["bytes"] == 2 * 1048576
    assert st["mb"] == 2.0
    assert st["cap"] == 8
    assert st["budget_mb"] == 256.0


def test_store_reconfigure_shrinks_live():
    # a live store with 5 turns; shrinking the cap to 2 evicts down to the last 2 immediately
    s = tt.SnapshotStore(cap=8, budget_mb=512)
    for turn in range(5):
        s.snapshot_turn("run_a", turn, n_tok=10, nbytes=100)
    s.reconfigure(cap=2)
    assert s.turns_for("run_a") == [3, 4]
    assert s.count() == 2
    # tightening the byte budget evicts too
    s.reconfigure(budget_mb=1)                            # (100 bytes each, well under 1 MB) -> both stay
    assert s.count() == 2
    s.reconfigure(cap=1)
    assert s.turns_for("run_a") == [4]


def test_store_clear_run_frees_bytes():
    s = tt.SnapshotStore(cap=8, budget_mb=512)
    s.snapshot_turn("run_a", 0, n_tok=10, nbytes=500, kv=fake_kv(1, 10))
    s.snapshot_turn("run_a", 1, n_tok=10, nbytes=500, kv=fake_kv(1, 10))
    s.clear_run("run_a")
    assert s.count() == 0
    assert s.total_bytes == 0
    assert s.turns_for("run_a") == []


def test_descriptor_only_snapshot_has_no_cache():
    # the stateless studio path: n_tok known, no reusable cache -> descriptor-only, zero bytes.
    s = tt.SnapshotStore()
    snap = s.snapshot_turn("run_a", 0, n_tok=42)          # kv omitted
    assert snap.has_cache is False
    assert snap.nbytes == 0
    d = snap.descriptor()
    assert d["has_cache"] is False and d["n_tok"] == 42 and d["mb"] == 0.0


def test_snapshot_with_real_payload_reports_cache_and_bytes():
    s = tt.SnapshotStore()
    snap = s.snapshot_turn("run_a", 0, n_tok=10, kv=fake_kv(2, 50))   # nbytes inferred from payload
    assert snap.has_cache is True
    assert snap.nbytes == 2 * 2 * 50 * 2                 # layers x (k+v) x elems x 2 bytes
    assert snap.descriptor()["has_cache"] is True


# ===================================================================================================
# rewind/branch transcript transforms -- pure
# ===================================================================================================
CONV = [
    {"role": "user", "content": "u0"},
    {"role": "assistant", "content": "a0"},
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "a2"},
]


def test_message_turns_folds_pairs():
    turns = tt.message_turns(CONV)
    assert len(turns) == 3
    assert turns[0] == {"turn": 0, "user": "u0", "assistant": "a0", "user_idx": 0, "assistant_idx": 1}
    assert turns[2]["user"] == "u2" and turns[2]["assistant"] == "a2"


def test_message_turns_dangling_final_user():
    conv = CONV[:5]                                       # ends on u2 with no reply
    turns = tt.message_turns(conv)
    assert len(turns) == 3
    assert turns[2]["assistant"] is None
    assert turns[2]["assistant_idx"] is None


def test_message_turns_leading_system_rides_along():
    conv = [{"role": "system", "content": "sys"}] + CONV[:2]
    turns = tt.message_turns(conv)
    assert len(turns) == 1
    assert turns[0]["user"] == "u0"                       # system doesn't start a turn


def test_branch_messages_truncates_and_drops_reply():
    # branch at turn 1, no alt -> keep [u0,a0,u1], drop a1 and everything after
    b = tt.branch_messages(CONV, 1)
    assert b == [{"role": "user", "content": "u0"},
                 {"role": "assistant", "content": "a0"},
                 {"role": "user", "content": "u1"}]


def test_branch_messages_with_alt_user_replaces_turn():
    b = tt.branch_messages(CONV, 1, alt_user="different question")
    assert b[-1] == {"role": "user", "content": "different question"}
    assert b[:-1] == [{"role": "user", "content": "u0"}, {"role": "assistant", "content": "a0"}]


def test_branch_messages_turn0_reroll_from_scratch():
    b = tt.branch_messages(CONV, 0)
    assert b == [{"role": "user", "content": "u0"}]       # just the first user turn


def test_branch_messages_blank_alt_is_ignored():
    assert tt.branch_messages(CONV, 1, alt_user="   ") == tt.branch_messages(CONV, 1)


def test_branch_messages_out_of_range_raises():
    with pytest.raises(ValueError):
        tt.branch_messages(CONV, 99)
    with pytest.raises(ValueError):
        tt.branch_messages(CONV, -1)
    with pytest.raises(ValueError):
        tt.branch_messages([], 0)


# ===================================================================================================
# branch() -> child run, against a FAKE substrate (the replay.py test pattern)
# ===================================================================================================
class FakeSteer:
    def __init__(self, strength=None):
        self.strength = dict(strength or {})
        self.saved = False

    def active(self):
        return {k: v for k, v in self.strength.items() if v}

    def save_state(self, path):
        self.saved = True


class FakeMem:
    def __init__(self, strength=1.0, prefix="PFX"):
        self.memory_strength = float(strength)
        self.prefix = prefix


class FakeSub:
    """Echoes the transcript it SAW so a branch's reply reflects the truncated history."""

    def __init__(self, mem=None, steer=None):
        self.memory = mem if mem is not None else FakeMem()
        self.steer = steer if steer is not None else FakeSteer()
        self.seen = None
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True):
        self.calls += 1
        self.seen = {"messages": [dict(m) for m in messages], "sample": sample, "max_new": max_new}
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return f"reply-to[{last_user}] over {len(messages)} msgs"


PARENT = {"id": "run_parent0", "model": "clozn-qwen", "substrate": "QwenSubstrate",
          "messages": CONV}


@pytest.fixture
def store(tmp_path):
    original = runlog.RUNS_DIR
    runlog.RUNS_DIR = str(tmp_path / "runs")
    try:
        yield runlog
    finally:
        runlog.RUNS_DIR = original


def test_branch_records_child_with_parent_and_turn(store):
    sub = FakeSub()
    child = tt.branch(PARENT, 1, sub, sample=False)
    assert child is not None
    assert child["parent_run_id"] == "run_parent0"
    assert child["changes_applied"]["branch_turn"] == 1
    assert child["changes_applied"]["edited_user"] is False
    assert child["source"] == "branch"
    assert child["client"] == "studio"
    assert "replayed" in child["flags"]                  # runlog._flags sets this from parent_run_id
    # it re-generated from the TRUNCATED transcript (turn 1's user, no later turns)
    assert sub.seen["messages"][-1]["content"] == "u1"
    assert len(sub.seen["messages"]) == 3
    assert sub.seen["sample"] is False                   # greedy by default (the receipt path)


def test_branch_live_regeneration_inherits_exact_memory_scope_and_associations(store):
    parent = {
        **PARENT,
        "session_key": "session_0123456789abcdef01234567",
        "client_key": "client_0123456789abcdef01234567",
        "client_key_source": "header",
        "project_key": "project_0123456789abcdef01234567",
    }

    class ScopedSub(FakeSub):
        def chat(self, messages, max_new=256, sample=True, memory_scope=None):
            self.seen_scope = memory_scope
            return super().chat(messages, max_new=max_new, sample=sample)

    sub = ScopedSub()
    child = tt.branch(parent, 1, sub)

    assert sub.seen_scope.app_key == parent["client_key"]
    assert sub.seen_scope.project_key == parent["project_key"]
    assert child["session_key"] == parent["session_key"]
    assert child["client_key"] == parent["client_key"]
    assert child["client_key_source"] == "header"
    assert child["project_key"] == parent["project_key"]


def test_branch_with_alt_user_notes_edit(store):
    sub = FakeSub()
    child = tt.branch(PARENT, 1, sub, alt_user="ask something else")
    assert child["changes_applied"]["edited_user"] is True
    assert child["changes_applied"]["alt_user"] == "ask something else"
    assert sub.seen["messages"][-1] == {"role": "user", "content": "ask something else"}


def test_branch_records_kv_snapshot_flag(store):
    # a real-cache snapshot present for (run, turn) flips the kv_snapshot note; absent -> False
    s = tt.SnapshotStore()
    s.snapshot_turn("run_parent0", 1, n_tok=10, kv=fake_kv(1, 10))
    child = tt.branch(PARENT, 1, FakeSub(), store=s)
    assert child["changes_applied"]["kv_snapshot"] is True
    child2 = tt.branch(PARENT, 2, FakeSub(), store=s)    # no snapshot for turn 2
    assert child2["changes_applied"]["kv_snapshot"] is False


def test_branch_never_persists_and_restores_knobs(store):
    steer = FakeSteer({"concise": 0.4})
    mem = FakeMem(strength=1.3)
    sub = FakeSub(mem=mem, steer=steer)
    tt.branch(PARENT, 0, sub)
    assert steer.saved is False                          # temp knobs NEVER persisted
    assert steer.strength == {"concise": 0.4}            # restored EXACTLY
    assert mem.memory_strength == 1.3


def test_branch_child_is_fetchable(store):
    child = tt.branch(PARENT, 1, FakeSub())
    fetched = store.get_run(child["id"])
    assert fetched is not None
    assert fetched["parent_run_id"] == "run_parent0"
    assert fetched["source"] == "branch"


def test_branch_reply_differs_by_branch_point(store):
    c0 = tt.branch(PARENT, 0, FakeSub())
    c2 = tt.branch(PARENT, 2, FakeSub())
    assert c0["response"] != c2["response"]              # different histories -> different replies


def test_branch_out_of_range_returns_none(store):
    assert tt.branch(PARENT, 99, FakeSub()) is None
    assert tt.branch(PARENT, -1, FakeSub()) is None


def test_branch_returns_none_without_chat(store):
    class NoChat:
        memory = FakeMem()
        steer = FakeSteer()
    assert tt.branch(PARENT, 0, NoChat()) is None


def test_branch_returns_none_on_chat_exception_and_restores(store):
    class Boom(FakeSub):
        def chat(self, messages, max_new=256, sample=True):
            raise RuntimeError("model exploded")
    sub = Boom(mem=FakeMem(strength=1.7), steer=FakeSteer({"warm": 0.5}))
    assert tt.branch(PARENT, 1, sub) is None
    assert sub.memory.memory_strength == 1.7             # finally restored on failure
    assert sub.steer.strength == {"warm": 0.5}


def test_branch_returns_none_on_empty_run(store):
    assert tt.branch(None, 0, FakeSub()) is None
    assert tt.branch({}, 0, FakeSub()) is None
