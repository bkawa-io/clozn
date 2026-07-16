"""test_async_retrain -- memory-card retrains no longer block the HTTP handler (async retrain).

Mutating a memory card retrains the soft-prefix via consolidate(), which is ~4-5 min on the 4-bit 7B.
That used to run SYNCHRONOUSLY inside the request handler, hanging the UI (and timing out clients). The
fix runs consolidate() on a background daemon thread, exposes an in-flight signal the UI polls, and makes
the chat/generate paths serialize behind the retrain via a shared _TRAIN_LOCK.

No model, no GPU: a SLOW FakeMem (consolidate == time.sleep) stands in for SelfTeach, so we can assert
    * a card mutation RETURNS FAST (doesn't wait out the ~fake~ 4-5 min consolidate),
    * the retrain signal flips active -> idle (with the right card_id/action while in flight),
    * /memory/cards + /memory/retrain-status expose that signal,
    * a chat acquiring _TRAIN_LOCK BLOCKS until the retrain finishes (serialized, never races),
    * a retrain that ERRORS still clears the in-flight flag (the poll always terminates),
    * a no-op transition spawns no thread.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs      # noqa: E402
import clozn.lab.substrates as lab_substrates       # noqa: E402  (the _InternalizedRetrain mixin lives here)
import clozn.memory.cards as memory_cards            # noqa: E402
import clozn.memory.mode as memory_mode             # noqa: E402
import clozn.runs.store as runlog                  # noqa: E402


class SlowMem:
    """Stand-in for SelfTeach/DreamMemory whose consolidate() is a slow no-op (time.sleep), so we can prove
    the endpoint returned BEFORE the retrain finished. Records call order + threads for the assertions."""

    def __init__(self, rules=None, delay=0.5, boom=False):
        self.rules = list(rules or [])
        self.prefix = "PREFIX" if self.rules else None
        self.memory_strength = 1.0
        self.delay = delay
        self.boom = boom
        self.consolidate_calls: list[list[str]] = []
        self.reset_calls = 0
        self.consolidate_thread = None
        self.finished_at = None

    def consolidate(self, rules):
        self.consolidate_thread = threading.current_thread()
        time.sleep(self.delay)                            # the slow bit (stand-in for 120 steps x 8 probes)
        if self.boom:
            raise RuntimeError("consolidate blew up")
        self.consolidate_calls.append(list(rules))
        self.rules = list(rules)
        self.prefix = "PREFIX"
        self.finished_at = time.time()
        return {"ok": True, "rules": list(rules)}

    def reset(self):
        time.sleep(self.delay)
        self.reset_calls += 1
        self.prefix = None
        self.rules = []
        self.finished_at = time.time()
        return {"ok": True}


class _LabSub(lab_substrates._InternalizedRetrain, cs.Substrate):
    """A lab substrate: the base studio surface + the per-instance internalized retrain machinery (same
    MRO as QwenSubstrate/DreamSubstrate), so /memory/* dispatch runs through the REAL async-retrain path
    now that the retrain state moved off the app module onto the substrate."""


_LIVE_SUBS: list = []   # substrates handed out this test -> joined at teardown (retrain state is per-instance)


def _substrate(mem):
    sub = object.__new__(_LabSub)
    sub._mem = mem
    _LIVE_SUBS.append(sub)
    return sub


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Isolate the card store + run log AND reset the module-level retrain singletons so tests don't leak
    an in-flight flag / a held lock into each other. Memory mode is PINNED to internalized: async retrain
    IS the internalized path (prompt mode short-circuits it entirely -- covered in test_memory_mode)."""
    monkeypatch.setenv("CLOZN_RUNTIME_KIND", "lab")   # internalized/soft-prefix retrain is a LAB feature now
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    assert memory_mode.set_mode("internalized")
    # retrain state is PER-SUBSTRATE now (moved off the app module onto _InternalizedRetrain); each test
    # builds its own sub via _substrate(), so there's nothing process-global to reset -- just make sure no
    # daemon retrain outlives the test by joining every substrate we handed out.
    _LIVE_SUBS.clear()
    yield tmp_path
    while _LIVE_SUBS:
        _LIVE_SUBS.pop()._join_retrain(timeout=10.0)      # never leave a daemon retrain running past a test


# ---- the endpoint returns fast; the retrain runs in the background --------------------------------

def test_approve_returns_immediately_and_retrains_in_background(iso):
    mem = SlowMem(["likes tea"], delay=0.6)
    sub = _substrate(mem)
    card = sub._memory("/memory/add", {"text": "wants bullet points"})

    t0 = time.time()
    res = sub._memory("/memory/approve", {"id": card["id"]})
    dt = time.time() - t0

    # returned WAY before the 0.6s fake-consolidate finished
    assert dt < 0.2, f"approve blocked for {dt:.2f}s (should be immediate)"
    assert res["status"] == "active"                     # card flipped to its FINAL status synchronously
    assert res["resync"]["retraining"] is True           # ...and reported a retrain is now in flight
    # the retrain hasn't applied yet: rules still the old set right after the call
    assert mem.consolidate_calls == []

    assert sub._retrain_in_flight() is True               # the poll signal is live
    st = sub._retrain_status()
    assert st["active"] is True and st["card_id"] == card["id"] and st["action"] == "approve"

    assert sub._join_retrain(timeout=5.0)                 # await the background consolidate
    assert set(mem.rules) == {"likes tea", "wants bullet points"}
    assert mem.consolidate_calls[-1] and set(mem.consolidate_calls[-1]) == {"likes tea", "wants bullet points"}
    assert sub._retrain_in_flight() is False              # flag cleared on finish
    assert sub._retrain_status()["active"] is False
    # it really ran off the calling thread
    assert mem.consolidate_thread is not None and mem.consolidate_thread is not threading.current_thread()


# ---- the in-flight signal is exposed on the poll endpoints ----------------------------------------

def test_retrain_status_endpoint_and_cards_expose_the_flag(iso):
    mem = SlowMem(["likes tea"], delay=0.5)
    sub = _substrate(mem)
    card = sub._memory("/memory/add", {"text": "wants bullet points"})

    # idle before any mutation
    assert sub._memory("/memory/retrain-status", {})["active"] is False
    assert sub._memory("/memory/cards", {})["retraining"]["active"] is False

    sub._memory("/memory/approve", {"id": card["id"]})

    # both surfaces now report the in-flight retrain (fold-in + dedicated endpoint)
    live = sub._memory("/memory/retrain-status", {})
    assert live["active"] is True and live["action"] == "approve"
    folded = sub._memory("/memory/cards", {})["retraining"]
    assert folded["active"] is True and folded["card_id"] == card["id"]

    assert sub._join_retrain(timeout=5.0)
    assert sub._memory("/memory/retrain-status", {})["active"] is False
    assert sub._memory("/memory/cards", {})["retraining"]["active"] is False


# ---- CONCURRENCY: a chat serializes behind the retrain (waits, never races) -----------------------

def test_chat_blocks_until_retrain_finishes(iso):
    """The chat/generate paths acquire the substrate's self._train_lock, so a chat that arrives mid-retrain
    must wait out the consolidate rather than racing the shared model. We stand in for 'a chat' with the
    SAME lock the real QwenSubstrate.chat / chat_stream / _say paths take, and assert the ordering: the chat
    can't proceed until the background retrain releases the lock."""
    mem = SlowMem(["likes tea"], delay=0.7)
    sub = _substrate(mem)
    card = sub._memory("/memory/add", {"text": "wants bullet points"})

    order = []

    def chat():
        # mirrors `with self._train_lock:` in QwenSubstrate.chat / _say / DreamSubstrate /denoise
        with sub._train_lock:
            order.append(("chat_ran", time.time(), bool(mem.consolidate_calls)))

    sub._memory("/memory/approve", {"id": card["id"]})   # kicks off the ~0.7s background retrain
    assert sub._retrain_in_flight() is True

    t = threading.Thread(target=chat)
    t0 = time.time()
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), "chat never completed"

    assert len(order) == 1
    label, when, saw_consolidate = order[0]
    # the chat waited ~the retrain duration before it got the lock
    assert (when - t0) >= 0.5, "chat did not wait for the retrain to release the lock"
    # and by the time it ran, the retrain had committed (consolidate recorded) -> it saw a consistent model
    assert saw_consolidate is True
    assert sub._retrain_in_flight() is False


# ---- a failed retrain still clears the in-flight flag ---------------------------------------------

def test_failed_retrain_clears_the_flag(iso):
    mem = SlowMem(["likes tea"], delay=0.3, boom=True)   # consolidate() raises
    sub = _substrate(mem)
    card = sub._memory("/memory/add", {"text": "wants bullet points"})

    res = sub._memory("/memory/approve", {"id": card["id"]})
    assert res["resync"]["retraining"] is True

    assert sub._join_retrain(timeout=5.0)                 # the worker's finally: must clear the flag even on error
    st = sub._retrain_status()
    assert st["active"] is False                          # poll terminates
    assert st["error"] and "RuntimeError" in st["error"]  # the error is surfaced for the UI


# ---- a no-op transition never spawns a thread -----------------------------------------------------

def test_noop_transition_does_not_retrain(iso):
    """Approving a card whose text is already active doesn't move the active set -> no thread, no flag."""
    mem = SlowMem(["likes tea"], delay=0.3)
    sub = _substrate(mem)
    sub._memory("/memory/cards", {})                     # migrate 'likes tea' as active
    only = memory_cards.list_cards()[0]

    # re-approving an already-active card is a no-op transition
    res = sub._memory("/memory/approve", {"id": only["id"]})
    assert res["resync"]["retraining"] is False
    assert res["resync"]["changed"] is False
    assert sub._retrain_in_flight() is False
    assert mem.consolidate_calls == []


# ---- a second retrain is refused while one is in flight (no stacking) -----------------------------

def test_second_retrain_is_refused_while_one_runs(iso):
    mem = SlowMem(["likes tea"], delay=0.8)
    sub = _substrate(mem)
    a = sub._memory("/memory/add", {"text": "wants bullet points"})
    b = sub._memory("/memory/add", {"text": "keeps a garden"})

    first = sub._memory("/memory/approve", {"id": a["id"]})
    assert first["resync"]["retraining"] is True
    assert sub._retrain_in_flight() is True

    # a second mutation arriving mid-retrain flips its card synchronously but does NOT stack a 2nd thread
    second = sub._memory("/memory/approve", {"id": b["id"]})
    assert second["status"] == "active"                  # status still flips (fast + synchronous)
    assert second["resync"].get("busy") is True          # ...but the retrain was refused as busy
    assert second["resync"]["retraining"] is True

    assert sub._join_retrain(timeout=5.0)
    assert sub._retrain_in_flight() is False
