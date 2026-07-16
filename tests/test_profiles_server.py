"""test_profiles_server -- the /profiles/* studio endpoints (profiles studio UI).

No model, no GPU. We drive the REAL clozn_server.do_GET/do_POST handler (the same object.__new__(H)
no-socket trick test_propose_memory.py uses) against:
  * a FAKE substrate exposing ._mem (rules/consolidate/reset, like test_memory_wiring.FakeMem) and
    .steer (clear/set/add_custom, like SteeringControl's duck type profiles.apply_dials expects);
  * an isolated profiles.DEFAULT_DIR, memory_cards.CARDS_PATH, memory_mode.SETTINGS_PATH, runlog.RUNS_DIR.

The load-bearing invariant under test: **profiles.switch() semantics, server-side.** A switch REPLACES
the studio's active cards (never merges -- disjoint personas must not bleed), replaces dial values via
apply_dials (clear then set), resyncs the memory mechanism (instant in prompt mode; a backgrounded
consolidate in internalized mode -- the existing _start_retrain machinery, not a reimplementation), and
names the item-5 seam (facts travel in the bundle but are not compiled anywhere -- no live slot store is
wired into the server yet) rather than silently dropping them.
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

from clozn.server import app as cs      # noqa: E402
import clozn.lab.substrates as lab_substrates       # noqa: E402  (the _InternalizedRetrain mixin lives here)
import clozn.memory.cards as memory_cards            # noqa: E402
import clozn.memory.mode as memory_mode             # noqa: E402
from clozn.profiles import store as P           # noqa: E402
import clozn.runs.store as runlog                  # noqa: E402


# --- fakes: mirror the surfaces the endpoint touches (no 7B, no PyTorch) ----------------------------

class FakeMem:
    """Stand-in for SelfTeach/DreamMemory -- exactly test_memory_wiring.FakeMem's surface, so the profile
    switch tests exercise the SAME _start_retrain contract every other card mutation already does."""

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


class FakeSteer:
    """Mirrors test_profiles.FakeSteer (the profiles.py unit tests) -- clear/set/add_custom, plus a
    save_state/save_custom log so we can assert the switched-to persona is persisted like a manual dial."""

    def __init__(self):
        self.vecs, self.strength, self.customs = {}, {}, []
        self.cleared = 0
        self.saved_state_paths: list[str] = []
        self.saved_custom_paths: list[str] = []

    def clear(self):
        self.cleared += 1
        self.strength = {}

    def set(self, name, value):
        self.strength[name] = value

    def add_custom(self, name, pos, neg, mx):
        self.vecs[name] = "dir"
        self.customs.append((name, pos, neg, mx))

    def save_state(self, path):
        self.saved_state_paths.append(path)

    def save_custom(self, path):
        self.saved_custom_paths.append(path)


class FakeSub(lab_substrates._InternalizedRetrain, cs.Substrate):
    """A qwen-like substrate: ._mem for cards/rules, .steer for dials. Subclasses the real cs.Substrate
    (zero-arg, __init__ skipped) so /memory/* still runs through the REAL Substrate._memory dispatch if a
    test needs it -- mirrors test_propose_memory.FakeSub. Mixes in _InternalizedRetrain (same MRO as the
    real QwenSubstrate) so the internalized-mode switch goes through the REAL background-consolidate path."""
    name = "qwen"

    def __init__(self, mem=None, steer=None):
        self._mem = mem if mem is not None else FakeMem()
        self.steer = steer if steer is not None else FakeSteer()

    def handle(self, path, body):
        if path.startswith("/memory/"):
            return self._memory(path, body)
        return None


# --- driving the real do_GET/do_POST handlers without a socket ---------------------------------------

class _FakeRequest:
    def __init__(self, path, body_obj=None):
        self.path = path
        raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Length": str(len(raw))}


def _dispatch(method, path, body_obj=None):
    """Invoke the real clozn_server GET/POST handler for `path`, return the decoded JSON response body."""
    H = cs.make_handler()
    h = object.__new__(H)
    req = _FakeRequest(path, body_obj)
    h.path, h.rfile, h.wfile, h.headers = req.path, req.rfile, req.wfile, req.headers
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    raw = req.wfile.getvalue()
    _, _, payload = raw.partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _get(path):
    return _dispatch("GET", path)


def _post(path, body_obj=None):
    return _dispatch("POST", path, body_obj)


# --- isolation ----------------------------------------------------------------------------------------

@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Point every module-level store at tmp files/dirs: profiles, cards, the memory-mode settings, and
    the run log. Defaults memory mode to "prompt" (the fresh-install default and the mode the "switch
    instantly" contract is about) -- internalized-mode behavior gets its own test using set_mode.

    profiles.ProfileStore's root is a CONSTRUCTOR DEFAULT (`root: str = DEFAULT_DIR`), baked in at
    def-time -- patching the module attribute P.DEFAULT_DIR has no effect on calls already using the
    default, since the server (correctly, matching real usage) always calls bare ProfileStore(). Rebind
    the bound default tuple itself so every no-arg ProfileStore() this test creates lands in tmp_path;
    monkeypatch restores it automatically."""
    monkeypatch.setattr(P.ProfileStore.__init__, "__defaults__", (str(tmp_path / "profiles"),))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    assert memory_mode.set_mode("prompt")
    monkeypatch.setattr(cs, "SUBNAME", "qwen")
    return tmp_path


def _mk_store():
    """A ProfileStore pointed at the (already monkeypatched) default dir -- what the endpoints use."""
    return P.ProfileStore()


def _friend_bundle():
    p = P.new_profile("friend", "the sci-fi one")
    p["cards"] = [{"text": "Loves sci-fi", "status": "active"}]
    p["dials"] = {"warm": 0.7}
    p["facts"] = [{"cue": "The user's dog is named", "answer": " Biscuit"}]
    return p


def _work_bundle():
    p = P.new_profile("work", "strictly business")
    p["cards"] = [{"text": "Prefers bullet points", "status": "active"}]
    p["dials"] = {"concise": 0.8}
    p["custom_dials"] = [{"name": "brisk", "pos": "curt and efficient", "neg": "rambling", "max": 0.4}]
    p["facts"] = [{"cue": "The user's boss is", "answer": " Marta"}]
    return p


# ---- /profiles/list ----------------------------------------------------------------------------------

def test_list_is_empty_before_anything_is_saved(iso):
    out = _get("/profiles/list")
    assert out["profiles"] == []
    assert out["active"] is None


def test_list_returns_saved_bundles_and_the_active_name(iso, monkeypatch):
    _mk_store().save(_friend_bundle())
    _mk_store().save(_work_bundle())
    sub = FakeSub()
    monkeypatch.setattr(cs, "SUB", sub)
    _post("/profiles/switch", {"name": "work"})

    out = _get("/profiles/list")
    assert {p["name"] for p in out["profiles"]} == {"friend", "work"}
    assert out["active"] == "work"


# ---- /profiles/save -----------------------------------------------------------------------------------

def test_save_creates_a_bundle_round_trippable_from_list(iso):
    out = _post("/profiles/save", _friend_bundle())
    assert out["ok"] is True
    assert out["profile"]["name"] == "friend"
    assert out["profile"]["dials"]["warm"] == 0.7

    listed = _get("/profiles/list")["profiles"]
    assert len(listed) == 1 and listed[0]["name"] == "friend"


def test_save_rejects_a_non_slug_name(iso):
    bad = _friend_bundle()
    bad["name"] = "Not A Slug!"
    out = _post("/profiles/save", bad)
    assert "error" in out
    assert _get("/profiles/list")["profiles"] == []


def test_save_is_an_update_when_the_name_already_exists(iso):
    _post("/profiles/save", _friend_bundle())
    again = _friend_bundle()
    again["cards"].append({"text": "Also loves cats", "status": "active"})
    out = _post("/profiles/save", again)
    assert out["ok"] is True
    assert len(out["profile"]["cards"]) == 2
    assert len(_get("/profiles/list")["profiles"]) == 1        # updated in place, not duplicated


# ---- /profiles/switch: THE persona switch (cards replace, dials replace, prompt mode = instant) -------

def test_switch_replaces_cards_and_dials_and_is_instant_in_prompt_mode(iso, monkeypatch):
    _mk_store().save(_friend_bundle())
    mem = FakeMem(["some stale rule"])
    steer = FakeSteer()
    sub = FakeSub(mem=mem, steer=steer)
    monkeypatch.setattr(cs, "SUB", sub)

    out = _post("/profiles/switch", {"name": "friend"})
    assert out["ok"] is True
    assert out["name"] == "friend"
    assert "sci-fi" in out["prompt_block"]

    # cards: the new active set is EXACTLY the profile's cards
    assert {c["text"] for c in memory_cards.list_cards(status="active")} == {"Loves sci-fi"}
    assert out["cards"]["added"] == 1

    # dials: cleared then set to exactly the profile's values
    assert steer.cleared == 1
    assert steer.strength == {"warm": 0.7}
    assert steer.saved_state_paths                          # persisted, like a manual /steer/set

    # PROMPT MODE: instant -- no retrain, no consolidate call
    assert out["resync"]["retraining"] is False
    assert out["resync"]["mode"] == "prompt"
    assert mem.consolidate_calls == []

    # facts: the item-5 seam is NAMED, not silently dropped. With the facts tier OFF (the default in
    # this fixture) the note explains the bundle's facts travel but aren't compiled until the tier is on
    # (the seam is now CLOSED -- see test_facts_server for the on-path that actually compiles them).
    assert out["facts_note"] is not None
    assert "fact" in out["facts_note"] and ("tier is off" in out["facts_note"] or "store" in out["facts_note"])

    # the active-profile name is now readable back
    assert cs._active_profile_name() == "friend"


def test_switching_between_two_disjoint_personas_never_bleeds(iso, monkeypatch):
    """The done-criterion: two personas with disjoint cards/dials switch instantly with zero carry-over."""
    _mk_store().save(_friend_bundle())
    _mk_store().save(_work_bundle())
    mem, steer = FakeMem(), FakeSteer()
    sub = FakeSub(mem=mem, steer=steer)
    monkeypatch.setattr(cs, "SUB", sub)

    r1 = _post("/profiles/switch", {"name": "friend"})
    assert {c["text"] for c in memory_cards.list_cards(status="active")} == {"Loves sci-fi"}
    assert steer.strength == {"warm": 0.7}
    assert "sci-fi" in r1["prompt_block"] and "bullet points" not in r1["prompt_block"]

    r2 = _post("/profiles/switch", {"name": "work"})
    # friend's card is GONE (not just disabled) -- disjoint personas keep disjoint cards
    active_texts = {c["text"] for c in memory_cards.list_cards(status="active")}
    assert active_texts == {"Prefers bullet points"}
    assert "Loves sci-fi" not in {c["text"] for c in memory_cards.list_cards()}
    # dials fully replaced -- warm is gone, concise is set, and the custom recipe compiled
    assert steer.strength == {"concise": 0.8}
    assert "warm" not in steer.strength
    assert steer.customs == [("brisk", "curt and efficient", "rambling", 0.4)]
    assert "bullet points" in r2["prompt_block"] and "sci-fi" not in r2["prompt_block"]
    # both switches cleared the dials (never blended) and neither retrained (prompt mode both times)
    assert steer.cleared == 2
    assert mem.consolidate_calls == []
    assert cs._active_profile_name() == "work"


def test_switch_in_internalized_mode_kicks_the_normal_background_retrain(iso, monkeypatch):
    """Off the "instant" contract on purpose: internalized mode still goes through the SAME async
    consolidate() path a card add/remove already uses -- not reimplemented, not skipped."""
    monkeypatch.setenv("CLOZN_RUNTIME_KIND", "lab")
    assert memory_mode.set_mode("internalized")
    _mk_store().save(_friend_bundle())
    mem, steer = FakeMem(["stale"]), FakeSteer()
    sub = FakeSub(mem=mem, steer=steer)
    monkeypatch.setattr(cs, "SUB", sub)
    # retrain state is per-substrate now; this fresh sub starts idle, so there's nothing to reset.

    out = _post("/profiles/switch", {"name": "friend"})
    assert out["resync"]["retraining"] is True
    assert sub._join_retrain(timeout=5.0)                    # await the background consolidate
    assert mem.consolidate_calls and mem.consolidate_calls[-1] == ["Loves sci-fi"]


def test_switch_unknown_profile_is_a_clean_404(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub())
    out = _post("/profiles/switch", {"name": "nonexistent"})
    assert "error" in out


def test_switch_without_a_name_is_a_400(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub())
    out = _post("/profiles/switch", {})
    assert "error" in out


def test_switch_with_no_substrate_loaded_is_a_503(iso, monkeypatch):
    _mk_store().save(_friend_bundle())
    monkeypatch.setattr(cs, "SUB", None)
    out = _post("/profiles/switch", {"name": "friend"})
    assert "error" in out


def test_switch_a_profile_with_no_cards_empties_the_active_set(iso, monkeypatch):
    empty = P.new_profile("blank")
    _mk_store().save(empty)
    mem, steer = FakeMem(["stale one", "stale two"]), FakeSteer()
    sub = FakeSub(mem=mem, steer=steer)
    monkeypatch.setattr(cs, "SUB", sub)

    out = _post("/profiles/switch", {"name": "blank"})
    assert memory_cards.list_cards(status="active") == []
    assert out["prompt_block"] == ""
    assert out["facts_note"] is None                          # no facts in this bundle -> nothing to note


# ---- /profiles/export + /profiles/import -----------------------------------------------------------

def test_export_returns_the_bundle_as_json(iso):
    _post("/profiles/save", _friend_bundle())
    out = _post("/profiles/export", {"name": "friend"})
    assert out["ok"] is True
    assert out["profile"]["name"] == "friend"
    assert out["profile"]["custom_dials"] == []


def test_export_unknown_profile_is_a_clean_404(iso):
    out = _post("/profiles/export", {"name": "ghost"})
    assert "error" in out


def test_import_round_trips_an_exported_bundle(iso):
    _post("/profiles/save", _work_bundle())
    exported = _post("/profiles/export", {"name": "work"})["profile"]

    imported = _post("/profiles/import", {"profile": exported})
    assert imported["ok"] is True
    assert imported["profile"]["name"] == "work"
    assert imported["profile"]["custom_dials"][0]["pos"] == "curt and efficient"   # the RECIPE travels

    listed = {p["name"] for p in _get("/profiles/list")["profiles"]}
    assert "work" in listed


def test_import_can_rename_on_the_way_in(iso):
    _post("/profiles/save", _friend_bundle())
    exported = _post("/profiles/export", {"name": "friend"})["profile"]

    imported = _post("/profiles/import", {"profile": exported, "rename": "friend2"})
    assert imported["ok"] is True
    assert imported["profile"]["name"] == "friend2"

    listed = {p["name"] for p in _get("/profiles/list")["profiles"]}
    assert listed == {"friend", "friend2"}                    # the original is untouched


def test_import_rejects_a_malformed_bundle(iso):
    out = _post("/profiles/import", {"profile": {"name": "bad name here"}})
    assert "error" in out
    assert _get("/profiles/list")["profiles"] == []
