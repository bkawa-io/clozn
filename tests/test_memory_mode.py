"""test_memory_mode -- the memory-mode swap, model-free.

No model, no GPU, no embedder. Three layers under test:

  * memory_mode.py        -- mode persistence (round-trip, the fresh-install default, the existing-.pt
                             migration rule) and compile_prompt_block (exact sys_rule wording, ordering,
                             empty -> no block).
  * clozn_server helpers  -- _prompt_block_for's per-turn decision (gate-out omission, strength 0,
                             per-card exclusion), _inject_block, and the PROMPT-mode short-circuit of the
                             card-mutation retrain machinery (instant, no consolidate, no thread), plus
                             the force retrain used by the toggle-back-to-internalized catch-up.
  * replay.py + endpoints -- disabled_memory_ids as REAL per-card ablation in prompt mode (applied during
                             the one generation, restored after, no "not applied" note), memory.mode on
                             every recorded run, and the GET/POST /memory/mode endpoint round-trip over a
                             real (model-less) HTTP server.

The topic gate is STUBBED everywhere it matters (a fixed scalar) so no sentence-transformer loads and the
gate-in/gate-out branches are deterministic.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs      # noqa: E402
import clozn.lab.substrates as lab_substrates       # noqa: E402  (the _InternalizedRetrain mixin lives here)
import clozn.memory.cards as memory_cards            # noqa: E402
import clozn.memory.mode as memory_mode             # noqa: E402
from clozn import replay                  # noqa: E402
import clozn.runs.store as runlog                  # noqa: E402
import clozn.memory.topic_gate as topic_gate              # noqa: E402


# ---- fakes (mirror test_memory_wiring / test_replay -- each suite carries its own) -----------------

class FakeMem:
    """Stand-in for SelfTeach: the minimal surface the mode wiring touches. Records consolidate/reset
    so a test can assert the prefix machinery was (or was NOT) invoked."""

    def __init__(self, rules=None):
        self.rules = list(rules or [])
        self.prefix = "PREFIX" if self.rules else None
        self.memory_strength = 1.0
        self.consolidate_calls: list[list[str]] = []
        self.reset_calls = 0

    def consolidate(self, rules):
        self.consolidate_calls.append(list(rules))
        self.rules = list(rules)
        self.prefix = "PREFIX"
        return {"ok": True, "rules": list(rules)}

    def reset(self):
        self.reset_calls += 1
        self.prefix = None
        self.rules = []
        return {"ok": True}


class FakeSub:
    """Replay-facing stub: chat() snapshots the state it SAW at call time (incl. the per-card exclusion
    attr) so tests can assert the ablation was live DURING generation and gone after."""

    def __init__(self, mem=None):
        self.memory = mem if mem is not None else FakeMem()
        self.steer = None
        self.seen = {}

    def chat(self, messages, max_new=256, sample=True):
        self.seen = {"memory_strength": self.memory.memory_strength,
                     "exclude": list(getattr(self.memory, "_exclude_card_ids", None) or []),
                     "sample": sample}
        return f"reply excl={self.seen['exclude']}"


class StubGate:
    def __init__(self, value):
        self.value = float(value)

    def scalar(self, prompt, texts):
        return self.value


class _LabSub(lab_substrates._InternalizedRetrain, cs.Substrate):
    """A lab substrate: the base studio surface + the per-instance internalized retrain machinery (same
    MRO as QwenSubstrate/DreamSubstrate). In PROMPT mode _start_retrain short-circuits exactly like the
    base; in INTERNALIZED mode it runs the REAL background consolidate -- both paths are exercised here."""


_LIVE_SUBS: list = []   # substrates handed out this test -> joined at teardown (retrain state is per-instance)


def _substrate(mem):
    sub = object.__new__(_LabSub)
    sub._mem = mem
    _LIVE_SUBS.append(sub)
    return sub


def _gate(monkeypatch, value):
    """Pin the topic gate to a fixed scalar (no embedder load, deterministic gate-in/out)."""
    monkeypatch.setattr(topic_gate, "get_gate", lambda: StubGate(value))


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Isolate every store this suite touches (cards, runs, settings, the legacy-.pt migration probe)
    and reset the retrain singletons. Mode starts UNSET -- each test picks its own via set_mode.
    Runs under the LAB runtime kind: internalized is a lab-only mode, so set_mode('internalized') is
    only permitted here (a product process refuses it -- see test_product_refuses_internalized_mode)."""
    monkeypatch.setenv("CLOZN_RUNTIME_KIND", "lab")
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "legacy.pt")])
    # retrain state is PER-SUBSTRATE now (moved off the app module onto _InternalizedRetrain); each test
    # builds its own sub via _substrate(), so there's nothing process-global to reset -- just make sure no
    # daemon retrain outlives the test by joining every substrate we handed out.
    _LIVE_SUBS.clear()
    yield tmp_path
    while _LIVE_SUBS:
        _LIVE_SUBS.pop()._join_retrain(timeout=10.0)


# ---- memory_mode: mode persistence + migration rule -------------------------------------------------

def test_fresh_install_defaults_to_prompt(iso):
    assert memory_mode.get_mode() == "prompt"


def test_existing_trained_prefix_keeps_internalized(iso):
    # the migration rule: a live personality on disk is NOT silently swapped to prompt carriage
    open(memory_mode.LEGACY_PREFIX_PATHS[0], "w").close()
    assert memory_mode.get_mode() == "internalized"


def test_explicit_choice_beats_the_migration_default(iso):
    open(memory_mode.LEGACY_PREFIX_PATHS[0], "w").close()   # legacy .pt present...
    assert memory_mode.set_mode("prompt")                   # ...but the user toggled
    assert memory_mode.get_mode() == "prompt"


def test_mode_round_trips_through_the_settings_file(iso):
    assert memory_mode.set_mode("internalized")
    assert memory_mode.get_mode() == "internalized"
    assert memory_mode.set_mode("prompt")
    assert memory_mode.get_mode() == "prompt"
    with open(memory_mode.SETTINGS_PATH, encoding="utf-8") as f:
        assert json.load(f)["memory_mode"] == "prompt"


def test_invalid_mode_is_refused(iso):
    assert memory_mode.set_mode("slots") is False           # future tier, not wired yet
    assert memory_mode.set_mode("") is False


def test_settings_merge_preserves_other_keys(iso):
    assert memory_mode.set_setting("memory_strength", 0.7)
    assert memory_mode.set_mode("internalized")
    assert memory_mode.get_setting("memory_strength") == 0.7


def test_set_setting_bad_value_raises_cleanly_and_prior_settings_survive(iso):
    """Round-2 pressure test #1 (HIGH): set_setting used to open(SETTINGS_PATH, "w") then json.dump,
    truncating the file BEFORE a non-serializable value could raise -- so one bad call wiped every other
    persisted setting (memory_mode, block_style, memory_facts, active_profile, ...). set_setting's
    contract (False on failure, never raise) is unchanged; what must change is that the failure no
    longer destroys anything already on disk."""
    assert memory_mode.set_mode("internalized")
    assert memory_mode.set_block_style("strict")
    assert memory_mode.set_setting("memory_strength", 0.7)

    ok = memory_mode.set_setting("bad_key", {1, 2, 3})       # a set -- json can't serialize it
    assert ok is False

    assert memory_mode.get_mode() == "internalized"          # every prior setting survives
    assert memory_mode.get_block_style() == "strict"
    assert memory_mode.get_setting("memory_strength") == 0.7
    assert memory_mode.get_setting("bad_key") is None        # the bad write never landed
    with open(memory_mode.SETTINGS_PATH, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == {"memory_mode": "internalized", "block_style": "strict", "memory_strength": 0.7}


# ---- memory_mode: block_style setting ------------------------------------------------------------------

def test_block_style_defaults_to_soft(iso):
    assert memory_mode.get_block_style() == "soft"
    assert memory_mode.DEFAULT_BLOCK_STYLE == "soft"


def test_block_style_round_trips_through_the_settings_file(iso):
    assert memory_mode.set_block_style("strict")
    assert memory_mode.get_block_style() == "strict"
    assert memory_mode.set_block_style("soft")
    assert memory_mode.get_block_style() == "soft"
    with open(memory_mode.SETTINGS_PATH, encoding="utf-8") as f:
        assert json.load(f)["block_style"] == "soft"


def test_invalid_block_style_is_refused_and_leaves_the_default(iso):
    assert memory_mode.set_block_style("loud") is False
    assert memory_mode.set_block_style("") is False
    assert memory_mode.get_block_style() == "soft"           # refused write never landed


def test_garbage_in_settings_file_degrades_to_default_block_style(iso, monkeypatch):
    # IO/parsing never raises (module ethos): a corrupt or unexpected value degrades, not crashes.
    monkeypatch.setattr(memory_mode, "_load_settings", lambda: {"block_style": "shout"})
    assert memory_mode.get_block_style() == "soft"


def test_block_style_setting_is_independent_of_memory_mode(iso):
    assert memory_mode.set_mode("internalized")
    assert memory_mode.set_block_style("strict")
    assert memory_mode.get_mode() == "internalized"
    assert memory_mode.get_block_style() == "strict"
    assert memory_mode.set_mode("prompt")
    assert memory_mode.get_block_style() == "strict"          # switching mode doesn't touch block_style


def test_block_style_setting_survives_alongside_other_settings(iso):
    assert memory_mode.set_setting("memory_strength", 0.7)
    assert memory_mode.set_block_style("strict")
    assert memory_mode.get_setting("memory_strength") == 0.7
    assert memory_mode.get_block_style() == "strict"


# ---- memory_mode: block compilation ------------------------------------------------------------------

def test_block_wording_is_exactly_consolidates_sys_rule(iso):
    # The load-bearing string: the block MUST be the same distillation target the prefix trains toward
    # (SelfTeach.consolidate's sys_rule), or the two modes stop being behaviourally comparable.
    expected = ("You are a helpful assistant talking with a returning user. Here is what you know "
                "about them; use it naturally to tailor how you respond:\n"
                "- likes tea\n"
                "- prefers bullet points")
    assert memory_mode.compile_prompt_block(["likes tea", "prefers bullet points"]) == expected


def test_block_wording_still_matches_self_teach_source(iso):
    # drift guard from the other end: if consolidate's sys_rule literal is ever reworded, this fails and
    # forces a lockstep update of compile_prompt_block (and vice versa via the exact-output test above).
    src = open(os.path.join(RESEARCH, "clozn", "substrates", "self_teach.py"), encoding="utf-8").read()
    assert "You are a helpful assistant talking with a returning user. Here is what you know " in src
    assert "about them; use it naturally to tailor how you respond:" in src


def test_block_preserves_order_and_skips_blanks(iso):
    block = memory_mode.compile_prompt_block(["b", "", "a", "   "])
    assert block.endswith("- b\n- a")


def test_empty_texts_compile_to_no_block(iso):
    assert memory_mode.compile_prompt_block([]) == ""
    assert memory_mode.compile_prompt_block(["", "  "]) == ""


# ---- memory_mode: strict block variant -------------------------------------------------------------------

def test_no_style_arg_defaults_to_soft_and_is_unchanged(iso):
    # every EXISTING call site (clozn_server.py, phantom_kv.py) calls compile_prompt_block(texts) with no
    # second arg -- this is the back-compat contract: omitting style must behave exactly as before.
    expected_soft = ("You are a helpful assistant talking with a returning user. Here is what you know "
                     "about them; use it naturally to tailor how you respond:\n- likes tea")
    assert memory_mode.compile_prompt_block(["likes tea"]) == expected_soft
    assert memory_mode.compile_prompt_block(["likes tea"], style=None) == expected_soft


def test_explicit_soft_style_matches_the_default(iso):
    assert (memory_mode.compile_prompt_block(["likes tea"], style="soft")
            == memory_mode.compile_prompt_block(["likes tea"]))


def test_strict_style_is_a_direct_imperative_not_the_soft_hedge(iso):
    block = memory_mode.compile_prompt_block(["likes tea", "prefers bullet points"], style="strict")
    assert "use it naturally to tailor" not in block          # the soft hedge must be gone
    assert "- likes tea" in block and "- prefers bullet points" in block
    assert block != memory_mode.compile_prompt_block(["likes tea", "prefers bullet points"], style="soft")


def test_strict_style_preserves_order_and_skips_blanks_like_soft(iso):
    block = memory_mode.compile_prompt_block(["b", "", "a", "   "], style="strict")
    assert block.endswith("- b\n- a")


def test_strict_style_empty_texts_still_compile_to_no_block(iso):
    assert memory_mode.compile_prompt_block([], style="strict") == ""
    assert memory_mode.compile_prompt_block(["", "  "], style="strict") == ""


def test_explicit_style_overrides_the_persisted_setting(iso):
    memory_mode.set_block_style("strict")
    # even with "strict" persisted, an explicit style="soft" call (e.g. the A/B rig) gets soft wording
    soft = memory_mode.compile_prompt_block(["likes tea"], style="soft")
    assert "use it naturally to tailor" in soft
    memory_mode.set_block_style("soft")
    # and with "soft" persisted, an explicit style="strict" call still gets strict wording
    strict = memory_mode.compile_prompt_block(["likes tea"], style="strict")
    assert "use it naturally to tailor" not in strict


def test_no_style_arg_honors_the_persisted_setting(iso):
    # this is the part that makes existing call sites configurable WITHOUT editing them: they pass no
    # style, so compile_prompt_block falls through to get_block_style() and picks up whatever a user
    # (or the toggle in memory.js) has set.
    memory_mode.set_block_style("strict")
    assert "use it naturally to tailor" not in memory_mode.compile_prompt_block(["likes tea"])
    memory_mode.set_block_style("soft")
    assert "use it naturally to tailor" in memory_mode.compile_prompt_block(["likes tea"])


def test_unknown_style_string_falls_back_to_soft_wording(iso):
    # never raise on a garbage explicit style either (module ethos) -- degrade to the safe default.
    assert (memory_mode.compile_prompt_block(["likes tea"], style="shout")
            == memory_mode.compile_prompt_block(["likes tea"], style="soft"))


def test_active_cards_reads_only_active_and_honors_exclusions(iso):
    a = memory_cards.create("likes tea", status="active")
    b = memory_cards.create("keeps a garden", status="active")
    memory_cards.create("pending one", status="pending")
    memory_cards.create("disabled one", status="disabled")
    cards = memory_mode.active_cards()
    assert {c["text"] for c in cards} == {"likes tea", "keeps a garden"}
    minus_a = memory_mode.active_cards(exclude_ids=[a["id"]])
    assert [c["text"] for c in minus_a] == [c["text"] for c in cards if c["id"] != a["id"]]
    assert all(c["id"] != a["id"] for c in minus_a)
    assert any(c["id"] == b["id"] for c in minus_a)


# ---- clozn_server: the per-turn block decision -------------------------------------------------------

def test_block_matches_store_order(iso, monkeypatch):
    _gate(monkeypatch, 1.0)
    memory_cards.create("likes tea", status="active")
    memory_cards.create("keeps a garden", status="active")
    mem = FakeMem()
    block, applied, gate = cs._prompt_block_for(mem, "tell me about tea")
    order = [c["text"] for c in memory_mode.active_cards()]
    assert block == memory_mode.compile_prompt_block(order)
    assert [c["text"] for c in applied] == order
    assert gate == 1.0


def test_gate_out_omits_the_block_entirely(iso, monkeypatch):
    _gate(monkeypatch, 0.0)                                  # off-topic turn
    memory_cards.create("likes tea", status="active")
    block, applied, gate = cs._prompt_block_for(FakeMem(), "summarize this contract")
    assert block is None and applied == [] and gate == 0.0


def test_gate_below_threshold_omits_block(iso, monkeypatch):
    _gate(monkeypatch, cs.PROMPT_GATE_MIN / 2)               # ~0 per the spec -> omit
    memory_cards.create("likes tea", status="active")
    block, applied, _ = cs._prompt_block_for(FakeMem(), "anything")
    assert block is None and applied == []


def test_strength_zero_never_injects(iso, monkeypatch):
    _gate(monkeypatch, 1.0)
    memory_cards.create("likes tea", status="active")
    mem = FakeMem()
    mem.memory_strength = 0.0                                # the dial maps to on/off in prompt mode
    block, applied, _ = cs._prompt_block_for(mem, "tell me about tea")
    assert block is None and applied == []


def test_no_active_cards_means_no_block(iso, monkeypatch):
    _gate(monkeypatch, 1.0)
    memory_cards.create("pending only", status="pending")
    block, applied, _ = cs._prompt_block_for(FakeMem(), "hello")
    assert block is None and applied == []


def test_exclusion_attr_removes_exactly_that_card(iso, monkeypatch):
    _gate(monkeypatch, 1.0)
    a = memory_cards.create("likes tea", status="active")
    memory_cards.create("keeps a garden", status="active")
    mem = FakeMem()
    mem._exclude_card_ids = [a["id"]]                        # what replay sets for a per-card receipt
    block, applied, _ = cs._prompt_block_for(mem, "tea and gardens")
    assert "keeps a garden" in block and "likes tea" not in block
    assert [c["text"] for c in applied] == ["keeps a garden"]


def test_inject_block_prepends_or_merges_system(iso):
    msgs = [{"role": "user", "content": "hi"}]
    out = cs._inject_block(msgs, "BLOCK")
    assert out[0] == {"role": "system", "content": "BLOCK"} and out[1]["content"] == "hi"
    assert msgs == [{"role": "user", "content": "hi"}]       # caller's list untouched
    merged = cs._inject_block([{"role": "system", "content": "client rules"}, msgs[0]], "BLOCK")
    assert merged[0]["content"] == "client rules\n\nBLOCK"
    assert cs._inject_block(msgs, None) == msgs              # no block -> unchanged


# ---- clozn_server: card mutations are INSTANT in prompt mode (no consolidate, no thread) -------------

def test_prompt_mode_approve_is_instant_and_never_consolidates(iso):
    memory_mode.set_mode("prompt")
    mem = FakeMem(["likes tea"])
    sub = _substrate(mem)
    sub._memory("/memory/cards", {})                         # one-time migration seeds 'likes tea' active
    card = sub._memory("/memory/add", {"text": "wants bullet points"})

    res = sub._memory("/memory/approve", {"id": card["id"]})
    assert res["status"] == "active"
    assert res["resync"] == {"retraining": False, "changed": True, "mode": "prompt"}
    assert mem.consolidate_calls == []                       # the prefix machinery never ran
    assert sub._retrain_in_flight() is False                 # and no thread was spawned
    assert set(mem.rules) == {"likes tea", "wants bullet points"}   # bookkeeping still syncs


def test_prompt_mode_remove_and_edit_never_touch_the_prefix(iso):
    memory_mode.set_mode("prompt")
    mem = FakeMem(["likes tea", "keeps a garden"])
    sub = _substrate(mem)
    sub._memory("/memory/cards", {})
    target = next(c for c in memory_cards.list_cards() if c["text"] == "keeps a garden")

    sub._memory("/memory/edit", {"id": target["id"], "text": "keeps a rooftop garden"})
    assert "keeps a rooftop garden" in mem.rules
    sub._memory("/memory/remove", {"id": target["id"]})
    assert mem.rules == ["likes tea"]
    assert mem.consolidate_calls == [] and mem.reset_calls == 0
    assert mem.prefix == "PREFIX"                            # the trained artifact is preserved verbatim


def test_prompt_mode_disable_last_card_keeps_the_prefix(iso):
    # internalized would reset() here; prompt mode must NOT destroy the (unused) trained artifact.
    memory_mode.set_mode("prompt")
    mem = FakeMem(["only rule"])
    sub = _substrate(mem)
    sub._memory("/memory/cards", {})
    only = memory_cards.list_cards()[0]
    sub._memory("/memory/disable", {"id": only["id"]})
    assert mem.rules == []
    assert mem.reset_calls == 0 and mem.prefix == "PREFIX"


def test_prompt_mode_retrain_status_is_constant_idle(iso):
    memory_mode.set_mode("prompt")
    sub = _substrate(FakeMem(["likes tea"]))
    assert sub._memory("/memory/retrain-status", {}) == {"active": False, "mode": "prompt"}
    out = sub._memory("/memory/cards", {})
    assert out["mode"] == "prompt"
    assert out["retraining"] == {"active": False, "mode": "prompt"}


def test_internalized_cards_endpoint_reports_its_mode(iso):
    memory_mode.set_mode("internalized")
    out = _substrate(FakeMem(["likes tea"]))._memory("/memory/cards", {})
    assert out["mode"] == "internalized"
    assert out["retraining"]["active"] is False and out["retraining"]["mode"] == "internalized"


def test_force_retrain_consolidates_even_when_rules_are_synced(iso):
    # the toggle-back catch-up: rules == active set (prompt mode kept them synced) but the PREFIX is
    # stale -- force=True must consolidate anyway.
    memory_mode.set_mode("internalized")
    mem = FakeMem(["likes tea"])
    cs._mem_migrate(mem)                                     # store == rules -> the no-op pre-check would skip
    sub = _substrate(mem)
    res = sub._start_retrain(mem, "mode-switch", None, force=True)
    assert res["retraining"] is True
    assert sub._join_retrain(timeout=5.0)
    assert mem.consolidate_calls == [["likes tea"]]


# ---- replay: disabled_memory_ids is REAL per-card ablation in prompt mode ----------------------------

RUN = {"id": "run_parent0", "model": "clozn-qwen", "substrate": "QwenSubstrate",
       "messages": [{"role": "user", "content": "hello there"}]}


def test_replay_prompt_per_card_ablation_is_real_and_restored(iso):
    memory_mode.set_mode("prompt")
    a = memory_cards.create("likes tea", status="active")
    memory_cards.create("keeps a garden", status="active")
    sub = FakeSub(FakeMem(["likes tea", "keeps a garden"]))

    child = replay.replay(RUN, {"disabled_memory_ids": [a["id"]], "greedy": True}, sub)
    assert child is not None
    assert sub.seen["exclude"] == [a["id"]]                  # ablation was LIVE during generation
    assert not hasattr(sub.memory, "_exclude_card_ids")      # ...and fully removed afterward
    # the child's memory summary: exactly that card's text is gone; the id list matches; NO stub note
    assert "likes tea" not in child["memory"]["cards_applied"]
    assert "keeps a garden" in child["memory"]["cards_applied"]
    assert a["id"] not in child["memory"]["applied_ids"]
    assert "disabled_memory_ids" not in (child["memory"].get("notes") or {})
    assert child["memory"]["mode"] == "prompt"
    assert child["changes_applied"]["disabled_memory_ids"] == [a["id"]]


def test_replay_internalized_keeps_the_honest_note(iso):
    memory_mode.set_mode("internalized")
    a = memory_cards.create("likes tea", status="active")
    sub = FakeSub(FakeMem(["likes tea"]))
    child = replay.replay(RUN, {"disabled_memory_ids": [a["id"]]}, sub)
    assert sub.seen["exclude"] == []                         # never applied on the fused prefix
    assert "disabled_memory_ids" in child["memory"]["notes"]
    assert child["memory"]["mode"] == "internalized"


def test_replay_prompt_memory_off_still_suppresses_everything(iso):
    memory_mode.set_mode("prompt")
    memory_cards.create("likes tea", status="active")
    sub = FakeSub(FakeMem(["likes tea"]))
    child = replay.replay(RUN, {"memory_off": True}, sub)
    assert sub.seen["memory_strength"] == 0.0                # strength 0 == never inject the block
    assert sub.memory.memory_strength == 1.0                 # restored
    assert child["memory"]["cards_applied"] == [] and child["memory"]["applied_ids"] == []


def test_replay_restores_exclusions_even_when_chat_raises(iso):
    memory_mode.set_mode("prompt")
    a = memory_cards.create("likes tea", status="active")

    class Boom(FakeSub):
        def chat(self, messages, max_new=256, sample=True):
            raise RuntimeError("model exploded")

    sub = Boom(FakeMem(["likes tea"]))
    assert replay.replay(RUN, {"disabled_memory_ids": [a["id"]]}, sub) is None
    assert not hasattr(sub.memory, "_exclude_card_ids")      # the finally cleaned up


def test_replay_children_record_the_mode_in_the_runlog(iso):
    # "runlog records memory.mode": the child run is a real runlog record -- fetch it back and check.
    memory_mode.set_mode("prompt")
    child = replay.replay(RUN, {}, FakeSub(FakeMem([])))
    assert runlog.get_run(child["id"])["memory"]["mode"] == "prompt"
    memory_mode.set_mode("internalized")
    child2 = replay.replay(RUN, {}, FakeSub(FakeMem([])))
    assert runlog.get_run(child2["id"])["memory"]["mode"] == "internalized"


# ---- the /memory/mode endpoint over a real (model-less) HTTP server ----------------------------------

@pytest.fixture()
def server(iso, monkeypatch):
    """A live clozn_server HTTP handler with NO substrate loaded (SUB=None -- the mode endpoints are
    server-level and must work on any substrate). Ephemeral port; torn down after the test."""
    from http.server import ThreadingHTTPServer
    monkeypatch.setattr(cs, "SUB", None)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), cs.make_handler())
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        yield base
    finally:
        srv.shutdown()
        srv.server_close()


def _http(base, path, body=None):
    if body is None:
        req = urllib.request.Request(base + path)
    else:
        req = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def test_mode_endpoint_round_trip(server):
    """Product gateway: prompt is the ONLY selectable memory mechanism. GET lists just prompt; POST
    prompt round-trips + persists; internalized soft-prefix memory is lab-only, so the product endpoint
    refuses it with 410 (the lab-server variant of this toggle is covered by the catch-up tests below)."""
    code, out = _http(server, "/memory/mode")                # GET: fresh install -> prompt
    assert code == 200 and out["mode"] == "prompt" and set(out["modes"]) == {"prompt"}

    code, out = _http(server, "/memory/mode", {"mode": "prompt"})
    assert code == 200 and out["ok"] is True and out["mode"] == "prompt"
    assert memory_mode.get_mode() == "prompt"                # and it landed in the settings file

    code, out = _http(server, "/memory/mode", {"mode": "internalized"})
    assert code == 410 and "lab-only" in out["error"]        # soft-prefix memory is a lab-only mechanism


def test_state_reports_the_memory_mode(server):
    code, out = _http(server, "/state")
    assert code == 200 and out["memory_mode"] == "prompt"


def test_toggle_to_internalized_kicks_the_catchup_retrain(server, monkeypatch):
    # Internalized toggling is a LAB behavior now: a product server 410s it, a lab server allows it and
    # runs the catch-up retrain. Pin the module runtime kind to lab so the /memory/mode route accepts it.
    monkeypatch.setattr(cs, "RUNTIME_KIND", "lab")
    # cards were edited in prompt mode (no consolidate) -> the prefix is stale -> toggling back must
    # retrain in the background so chats don't serve a personality the cards no longer describe.
    memory_cards.create("likes tea", status="active")
    mem = FakeMem(["likes tea"])                             # rules synced with the store...
    mem._trained_rules = ["something older"]                 # ...but the prefix embodies an OLD set
    monkeypatch.setattr(cs, "SUB", _substrate(mem))
    code, out = _http(server, "/memory/mode", {"mode": "internalized"})
    assert code == 200 and out["ok"] is True
    assert out["resync"]["retraining"] is True
    assert cs.SUB._join_retrain(timeout=5.0)                 # retrain ran on the active substrate (cs.SUB)
    assert mem.consolidate_calls == [["likes tea"]]          # the prefix caught up to the cards


def test_toggle_with_fresh_prefix_does_not_retrain(server, monkeypatch):
    monkeypatch.setattr(cs, "RUNTIME_KIND", "lab")           # internalized toggle is lab-only (see above)
    memory_cards.create("likes tea", status="active")
    mem = FakeMem(["likes tea"])
    mem._trained_rules = ["likes tea"]                       # prefix already embodies the active set
    monkeypatch.setattr(cs, "SUB", _substrate(mem))
    code, out = _http(server, "/memory/mode", {"mode": "internalized"})
    assert code == 200 and "resync" not in out               # nothing stale -> no retrain kicked
    assert mem.consolidate_calls == []
