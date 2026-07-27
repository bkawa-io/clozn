"""test_experiment_server -- POST /runs/<id>/experiment and GET /experiments/types, the HTTP wiring for
clozn/experiments/experiment.py's ONE experiment primitive (clozn/server/routes/receipts.py).

No model, no GPU: drives the REAL clozn_server do_GET/do_POST handler (the object.__new__(H) no-socket
trick used by test_receipts_server.py / test_explain_server.py / test_timetravel_server.py) against an
isolated runlog store + memory_cards store + memory_mode settings, with a FAKE substrate standing in for
the qwen one. experiment.run_experiment()'s own dispatch/envelope logic is exhaustively unit-tested
model-free in test_experiment.py; this file only proves the THIN endpoint wiring: the route matches, a
missing run is a clean 404, an unknown/malformed change is a clean 400, the wrong substrate is a clean 503
(mirroring each op's OWN substrate-gate wording), and a real request's experiment envelope comes back over
HTTP with the honesty fields (and Python True/None -> JSON true/null) intact.
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

from clozn.server import app as cs                    # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
import clozn.experiments.experiment as clozn_experiment  # noqa: E402
import clozn.memory.cards as memory_cards         # noqa: E402
import clozn.memory.mode as memory_mode          # noqa: E402
import clozn.runs.store as runlog                # noqa: E402


class FakeSteer:
    def __init__(self, strength=None):
        self.strength = dict(strength or {})

    def set(self, name, value):
        self.strength[str(name)] = float(value)

    def clear(self):
        self.strength = {}

    def active(self):
        return {k: v for k, v in self.strength.items() if v}


class FakeMem:
    def __init__(self, strength=1.0):
        self.memory_strength = float(strength)
        self.rules = []
        self.prefix = "PFX"


class FakeSub:
    name = "qwen"

    def __init__(self, mem=None, steer=None, concise_card_ids=()):
        self.memory = mem if mem is not None else FakeMem()
        self._mem = self.memory
        self.steer = steer if steer is not None else FakeSteer()
        self.concise_card_ids = {str(i) for i in concise_card_ids}
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True):
        self.calls += 1
        excluded = {str(i) for i in (getattr(self.memory, "_exclude_card_ids", None) or [])}
        if self.memory.memory_strength <= 0:
            return "Generic reply, memory off."
        concise_active = self.concise_card_ids - excluded
        concise_dial = float(self.steer.strength.get("concise", 0.0) or 0.0)
        base = "Short answer." if (concise_active or concise_dial > 0) else "A much longer rambling reply."
        if float(self.steer.strength.get("warm", 0.0) or 0.0) > 0:
            base += " Warmly!"
        return base


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


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    monkeypatch.setattr(cs, "SUB", FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5})))
    return tmp_path


def _seed_run():
    return runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="QwenSubstrate",
                         messages=[{"role": "user", "content": "hi there"}],
                         response="THE SAMPLED reply -- must never come back as a baseline",
                         behavior={"active_dials": {"warm": 0.5}})


# ==================================================================================================== GET

def test_get_experiment_types_returns_the_whole_registry(iso):
    out = _get("/experiments/types")
    assert set(out["types"]) == set(clozn_experiment.REGISTRY)
    for ctype, entry in out["types"].items():
        assert set(entry.keys()) == {"label", "needs", "cost_hint", "substrate", "op", "control"}


# =================================================================================================== POST

def test_experiment_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/experiment", {"change": {"type": "reroll"}})
    assert out == {"error": "run not found"}


def test_experiment_missing_change_is_a_clean_400(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment", {})
    assert "error" in out
    out2 = _post(f"/runs/{rid}/experiment", {"change": "not-a-dict"})
    assert "error" in out2


def test_experiment_unknown_change_type_is_a_clean_400(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment", {"change": {"type": "nonsense"}})
    assert "error" in out and "unknown change.type" in out["error"]


def test_experiment_needs_the_qwen_substrate_503_for_chat_backed_types(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment", {"change": {"type": "reroll"}})
    assert out == {"error": "experiment requires a ready product model worker"}


def test_experiment_needs_the_engine_substrate_503_for_swap_concept(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub())   # a chat-only substrate has no .engine/.jlens
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment", {"change": {"type": "swap_concept", "to_concept": "ocean"}})
    assert out == {"error": "experiment requires the product worker with J-lens enabled"}


def test_experiment_bad_required_field_is_a_clean_400(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment", {"change": {"type": "ablate_card"}})   # no card_id
    assert "error" in out


def test_experiment_bad_method_is_a_clean_400(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment",
               {"change": {"type": "ablate_dial", "dial": "warm"}, "method": "bogus"})
    assert "error" in out


def test_experiment_happy_path_ablate_dial_over_http(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment", {"change": {"type": "ablate_dial", "dial": "warm"}})
    assert "error" not in out
    assert out["run_id"] == rid
    assert out["change"] == {"type": "ablate_dial", "target": "warm", "label": "zeroing the 'warm' dial"}
    assert out["method"] == "receipt:regen"
    assert out["result"]["has_effect"] is True                # JSON true, round-tripped from Python True
    assert out["result"]["causal_verified"] is True
    assert out["result"]["null"] is None                      # JSON null -- regen mode has no null control
    assert out["baseline"]["reply"] == "A much longer rambling reply. Warmly!"
    assert out["result"]["changed_reply"] == "A much longer rambling reply."
    assert "cost_note" not in out["cost"]                      # cost shape is {passes, note[, est_seconds]}
    assert out["cost"]["passes"] == 2
    assert "note" in out["cost"]


def test_experiment_happy_path_set_dial_over_http(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment", {"change": {"type": "set_dial", "dial": "warm", "value": 0.0}})
    assert "error" not in out
    assert out["method"] == "counterfactual"
    assert out["result"]["null"] is None
    assert out["result"]["changed_reply"] == "A much longer rambling reply."


def test_experiment_happy_path_reroll_over_http(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment", {"change": {"type": "reroll"}})
    assert "error" not in out
    assert out["method"] == "replay:reroll"
    assert out["result"]["has_effect"] is None                 # replay computes no verdict -- never invented
    assert out["result"]["causal_verified"] is None
    assert out["result"]["changed_reply"] is not None


def test_experiment_happy_path_edit_turn_over_http(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment", {"change": {"type": "edit_turn", "turn": 0}})
    assert "error" not in out
    assert out["method"] == "branch:greedy"                    # branch defaults to greedy, not sampling dice
    assert out["change"]["target"] == 0


def test_experiment_swap_concept_over_http_with_a_stubbed_underlying_op(iso, monkeypatch):
    class EngineJlensSub:
        engine = object()

        def jlens(self, *a, **k):
            return {"available": False}

    monkeypatch.setattr(cs, "SUB", EngineJlensSub())

    def fake_swap(run, from_hint, to_concept, sub):
        return {"mode": "swap_receipt", "causal_verified": True, "run_id": run.get("id"),
               "disposed": {"hint": from_hint, "jlens_available": False, "jlens_layer": 21,
                           "jlens_top1": None, "jlens_top5": None, "jlens_reason": "stub",
                           "baseline_lean": "paris"},
               "swapped_to": {"concept": to_concept, "layer": 21, "strength": 6.0, "token_id": 5,
                             "coef": 0.5},
               "baseline_reply": "Paris.", "swapped_reply": "The ocean.", "null_reply": "Paris, a city.",
               "targeted_shift": True, "null_control_available": True,
               "lexicon_hits": {"baseline": 0, "swap": 1, "null": 0},
               "logprob_shift": {"baseline": -4.0, "swap": -0.5, "null": -3.8,
                                "swap_over_baseline_nat": 3.5, "swap_over_null_nat": 3.3},
               "coherent": True, "coherence_score": 0.9, "null_note": "...", "lexicon_note": "...",
               "blocked": None, "note": None}

    monkeypatch.setattr(clozn_experiment, "_swap_receipt", fake_swap)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/experiment",
               {"change": {"type": "swap_concept", "to_concept": "ocean", "from_hint": "Paris"}})
    assert "error" not in out
    assert out["method"] == "swap_receipt"
    assert out["result"]["has_effect"] is None                 # never invented from targeted_shift
    assert out["result"]["causal_verified"] is True
    assert out["result"]["null"]["available"] is True
    assert out["result"]["null"]["reply"] == "Paris, a city."
    assert out["result"]["receipt"]["targeted_shift"] is True   # full raw receipt, no info loss
