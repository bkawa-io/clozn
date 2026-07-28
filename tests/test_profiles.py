"""Model-free tests for profiles.py -- the portable persona bundles (CRUD, export/import, compile)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clozn.profiles import store as P


def mk(tmp_path):
    return P.ProfileStore(root=str(tmp_path / "profiles"))


def test_round_trip_and_listing(tmp_path):
    st = mk(tmp_path)
    p = P.new_profile("work", "strictly business")
    p["cards"].append({"text": "Prefers concise technical answers", "status": "active"})
    p["dials"] = {"concise": 0.8, "warm": -0.2}
    p["facts"].append({"cue": "The user's team standup is at", "answer": " nine"})
    st.save(p)
    q = st.load("work")
    assert q["name"] == "work" and q["dials"]["concise"] == 0.8
    assert q["facts"][0]["answer"] == " nine"
    assert [x["name"] for x in st.list()] == ["work"]
    assert st.delete("work") and st.list() == []


def test_names_are_slug_safe():
    with pytest.raises(ValueError):
        P.new_profile("Not A Slug!")
    with pytest.raises(ValueError):
        P.validate({"name": "../evil"})


def test_export_import_is_the_portability_contract(tmp_path):
    st = mk(tmp_path)
    p = P.new_profile("friend")
    p["cards"].append({"text": "Loves sci-fi", "status": "active"})
    p["custom_dials"].append({"name": "cozy", "pos": "warm and homey", "neg": "cold and formal", "max": 0.6})
    st.save(p)
    dest = str(tmp_path / "friend_bundle.json")
    st.export("friend", dest)
    st2 = P.ProfileStore(root=str(tmp_path / "other_machine"))
    q = st2.import_(dest, rename="friend2")
    assert q["name"] == "friend2"
    assert q["custom_dials"][0]["pos"] == "warm and homey"   # the RECIPE travels, not the vector


def test_prompt_block_active_only():
    p = P.new_profile("x")
    p["cards"] = [{"text": "A", "status": "active"}, {"text": "B", "status": "disabled"},
                  {"text": "C", "status": "active"}]
    block = P.prompt_block(P.validate(p))
    assert "- A" in block and "- C" in block and "B" not in block
    p["cards"] = []
    assert P.prompt_block(P.validate(p)) == ""


class FakeSteer:
    def __init__(self):
        self.vecs, self.strength, self.customs = {}, {}, []
        self.cleared = 0

    def clear(self):
        self.cleared += 1
        self.strength = {}

    def set(self, name, value):
        self.strength[name] = value

    def add_custom(self, name, pos, neg, mx):
        self.vecs[name] = "dir"
        self.customs.append((name, pos, neg, mx))


class FakeMem:
    def __init__(self):
        self.writes, self.calibrated = [], False

    def write(self, cue, answer, gate=True):
        self.writes.append((cue, answer, gate))
        return {"written": True}

    def calibrate_gate(self):
        self.calibrated = True


def test_switch_applies_everything_and_isolates():
    friend = P.validate({**P.new_profile("friend"), "dials": {"warm": 0.7},
                         "cards": [{"text": "Loves sci-fi"}],
                         "facts": [{"cue": "The user's dog is named", "answer": " Biscuit"}]})
    work = P.validate({**P.new_profile("work"), "dials": {"concise": 0.8},
                       "custom_dials": [{"name": "brisk", "pos": "p", "neg": "n", "max": 0.4}],
                       "cards": [{"text": "Prefers bullet points"}],
                       "facts": [{"cue": "The user's boss is", "answer": " Marta"}]})
    steer, mem_f, mem_w = FakeSteer(), FakeMem(), FakeMem()

    r1 = P.switch(friend, steer=steer, slotmem=mem_f)
    assert steer.strength == {"warm": 0.7} and "sci-fi" in r1["prompt_block"]
    assert mem_f.writes == [("The user's dog is named", " Biscuit", False)] and mem_f.calibrated

    r2 = P.switch(work, steer=steer, slotmem=mem_w)
    # switching REPLACES: dials cleared then set; customs computed from the recipe; stores isolated.
    assert steer.cleared == 2 and steer.strength == {"concise": 0.8}
    assert steer.customs == [("brisk", "p", "n", 0.4)]
    assert mem_w.writes[0][0] == "The user's boss is"
    assert all("dog" not in w[0] for w in mem_w.writes)      # friend facts never leak into work
    assert "bullet points" in r2["prompt_block"] and "sci-fi" not in r2["prompt_block"]


# ---- round-2 pressure test #1 (HIGH): atomic writes -- a bad save must never destroy a prior bundle ---

def test_save_bad_bundle_raises_and_prior_bundle_survives_on_disk(tmp_path):
    """profiles are user data too, same defect as memory cards / settings. `description` is NOT coerced
    by validate() (only cards/dials/custom_dials/facts are), so a non-JSON-serializable value can reach
    ProfileStore.save()'s write. save() is documented to propagate write failures (the server routes
    catch ValueError/KeyError/TypeError) -- the FIX is that the failure must happen before the on-disk
    file is touched, so a bundle already saved at that name survives a later bad save attempt intact."""
    st = mk(tmp_path)
    good = P.new_profile("work", "strictly business")
    good["cards"].append({"text": "Prefers concise answers", "status": "active"})
    st.save(good)

    bad = P.new_profile("work")
    bad["description"] = {1, 2, 3}                     # a set -- not coerced by validate(), not serializable
    with pytest.raises(TypeError):
        st.save(bad)

    reloaded = st.load("work")                          # the prior good bundle is untouched
    assert reloaded["description"] == "strictly business"
    assert reloaded["cards"][0]["text"] == "Prefers concise answers"


# ---- round-2 pressure test #2 (MEDIUM): validate() type-guards dials/custom_dials/facts consistently --

def test_validate_drops_non_dict_dials_container():
    p = P.new_profile("x")
    p["dials"] = "not-a-dict"
    assert P.validate(p)["dials"] == {}


def test_validate_drops_non_numeric_dial_values():
    p = P.new_profile("x")
    p["dials"] = {"concise": 0.8, "warm": "not-a-number", "chatty": None}
    assert P.validate(p)["dials"] == {"concise": 0.8}


def test_validate_skips_malformed_custom_dials_entries():
    p = P.new_profile("x")
    p["custom_dials"] = [
        {"name": "good", "pos": "p", "neg": "n", "max": 0.5},
        {"name": "missing_pos_and_neg"},                # would have KeyError'd before the fix
        "not-a-dict-at-all",                             # would have AttributeError'd before the fix
        {"name": "", "pos": "p", "neg": "n"},            # falsy name
    ]
    out = P.validate(p)
    assert [d["name"] for d in out["custom_dials"]] == ["good"]


def test_validate_drops_non_list_custom_dials_container():
    p = P.new_profile("x")
    p["custom_dials"] = {"oops": "a dict, not a list"}
    assert P.validate(p)["custom_dials"] == []


def test_validate_skips_malformed_facts_entries():
    p = P.new_profile("x")
    p["facts"] = [
        {"cue": "good cue", "answer": "good answer"},
        {"cue": "missing answer"},                       # would have KeyError'd before the fix
        "not-a-dict",                                     # would have AttributeError'd before the fix
        42,                                                # would have AttributeError'd before the fix
    ]
    out = P.validate(p)
    assert [f["cue"] for f in out["facts"]] == ["good cue"]


def test_validate_drops_non_list_facts_container():
    p = P.new_profile("x")
    p["facts"] = "not-a-list"
    assert P.validate(p)["facts"] == []


def test_validate_never_raises_on_a_fully_hand_edited_junk_bundle():
    """The concrete reachable path: /profiles/import and /profiles/save feed hand-editable bundle JSON
    straight into validate(). A malformed bundle must degrade field-by-field, never raise
    AttributeError/KeyError/TypeError (only a clean ValueError for a bad top-level name/version)."""
    junk = {"name": "junk",
            "dials": ["not", "a", "dict"],
            "custom_dials": [None, 5, {"name": "ok", "pos": "p", "neg": "n"}],
            "facts": [None, {"cue": "c"}, {"cue": "c2", "answer": "a2"}]}
    out = P.validate(junk)
    assert out["dials"] == {}
    assert [d["name"] for d in out["custom_dials"]] == ["ok"]
    assert [(f["cue"], f["answer"]) for f in out["facts"]] == [("c2", "a2")]


# ---- Seam 2 retrofit: the bundle IS clozn.behavior-profile.v1 (docs/SEAMS.md) -------------------

def test_new_profile_and_validate_carry_the_schema_version():
    p = P.new_profile("work")
    assert p["schema_version"] == "clozn.behavior-profile.v1"
    assert P.validate({"name": "work"})["schema_version"] == "clozn.behavior-profile.v1"


def test_saved_bundle_validates_against_its_own_schema(tmp_path):
    from clozn import schemas
    st = mk(tmp_path)
    p = P.new_profile("work", "strictly business")
    p["dials"] = {"concise": 0.8}
    p["response_policies"] = ["less-verbose"]
    st.save(p)
    schemas.validate(st.load("work"))     # infers the schema from the document's own schema_version


def test_a_legacy_bundle_with_no_schema_version_still_loads_and_gets_one(tmp_path):
    """A bundle saved before this retrofit has no `schema_version` key on disk. Loading it must not
    reject it -- validate() setdefaults the field so an old bundle upgrades in place on next load."""
    import json
    st = mk(tmp_path)
    legacy_path = os.path.join(st.root, "legacy.json")
    os.makedirs(st.root, exist_ok=True)
    with open(legacy_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "name": "legacy", "description": "", "cards": [], "dials": {},
                   "response_policies": [], "custom_dials": [], "facts": [],
                   "created_at": 1.0, "updated_at": 1.0}, f)
    loaded = st.load("legacy")
    assert loaded["schema_version"] == "clozn.behavior-profile.v1"


def test_load_rejects_path_traversal(tmp_path):
    """load()/switch/export took an unvalidated name, so `../config` escaped the profiles dir to
    ~/.clozn/*.json. _path() now validates every name -> traversal raises ValueError (the routes turn
    that into a clean 400/404)."""
    st = mk(tmp_path)
    for bad in ("../config", "/etc/passwd", "foo/../bar", "..", "a/b", "UPPER", ".hidden"):
        with pytest.raises(ValueError):
            st._path(bad)
        with pytest.raises(ValueError):
            st.load(bad)
