"""test_prove_cli -- model-free tests for `clozn prove` (docs/PRODUCT_ROADMAP.md §8 tail: batched
coalition/Shapley causal credit). Mirrors test_narrate_cli.py's layout exactly:

  * canned-dict tests drive format_prove() directly against hand-built /receipts-shaped fixtures -- no
    running Studio, no model, no GPU.
  * a wire-format check drives the REAL clozn_server /runs/<id>/receipts endpoint in-process (the same
    no-socket object.__new__(H) trick), proving format_prove() renders a genuine server response
    (including the opt-in `coalitions` report) end to end.
  * _fetch_prove()'s honest failure path against a guaranteed-closed local port.
  * the --coalitions/--coalitions-batch flags: opt-in (default False/"auto"), and cmd_prove threads them
    into _fetch_prove's request body exactly.

`POST /runs/<id>/receipts` previously had no CLI front door at all (only the Studio UI or a raw curl could
reach it) -- `clozn prove` is the new one this task adds.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import clozn.cli.main as clozn_cli                 # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
import clozn.cli.commands.explain as explain_cmd    # noqa: E402
from clozn.server import app as cs                  # noqa: E402
import clozn.memory.cards as memory_cards           # noqa: E402
import clozn.memory.mode as memory_mode             # noqa: E402
import clozn.runs.store as runlog                   # noqa: E402


# ---------------------------------------------------------------------------------- canned-dict fixtures

HAPPY_PATH = {
    "run_id": "run_1",
    "receipts": [
        {"influence": {"card_id": "c1"}, "has_effect": True, "causal_verified": True},
        {"influence": {"dial": "warm"}, "has_effect": False, "causal_verified": True},
    ],
    "skipped": [],
    "redundant_pairs": [],
    "approximation_note": "leave-one-out over every fired card/dial, plus a pairwise redundancy guard.",
    "perf_note": "sequential, not batched.",
}


def test_happy_path_renders_every_receipt_and_the_run_id():
    out = clozn_cli.format_prove(HAPPY_PATH)
    assert "run_1" in out
    assert "card c1" in out and "changed" in out
    assert "dial warm" in out and "no effect" in out


def test_no_fired_influences_is_an_honest_empty_result_not_an_error():
    out = explain_cmd.format_prove({"run_id": "run_2", "receipts": [], "redundant_pairs": []})
    assert "no fired influences" in out
    assert "error" not in out.lower()


def test_redundant_pairs_render_as_their_own_line():
    obj = dict(HAPPY_PATH)
    obj["redundant_pairs"] = [{"redundant": ["card:c1", "card:c2"],
                              "note": "together they drive this; individually neither is load-bearing"}]
    out = explain_cmd.format_prove(obj)
    assert "redundant pair" in out
    assert "card:c1" in out and "card:c2" in out


def test_coalitions_report_renders_when_present():
    obj = dict(HAPPY_PATH)
    obj["coalitions"] = {
        "available": True, "n_influences": 2, "keys": ["card:c1", "dial:warm"],
        "solo": {"card:c1": 0.3, "dial:warm": 0.1}, "pairs_evaluated": [["card:c1", "dial:warm"]],
        "pairs_capped": False, "k_pairs": 1, "joint": {"value": 0.35},
        "shapley": {"class": "exact", "values": {"card:c1": 0.275, "dial:warm": 0.075},
                   "estimator_note": "exact Shapley over the full 4-coalition power set (N=2 <= 4)."},
        "interaction_gap": {"joint_value": 0.35, "sum_solo": 0.4, "gap": -0.05, "ratio": -0.125,
                           "note": "solo attribution characteristically OVERCOUNTS a joint effect ..."},
        "batch_report": {"attempted": False, "used": False, "class": None, "reason": "batching off"},
        "cost_note": "2 solo (reused) + 1 pair + 1 joint arm(s) run for this report.",
    }
    out = explain_cmd.format_prove(obj)
    assert "coalition/Shapley credit" in out
    assert "Shapley (exact)" in out
    assert "interaction gap" in out


def test_coalitions_absent_never_renders_a_coalition_section():
    out = explain_cmd.format_prove(HAPPY_PATH)
    assert "coalition/Shapley" not in out


@pytest.mark.parametrize("garbage", [None, "not a dict", 42, [], {"receipts": "nope"}])
def test_never_raises_on_malformed_input(garbage):
    out = explain_cmd.format_prove(garbage)
    assert isinstance(out, str) and out


# --------------------------------------------------------------------------------- wire-format compatibility

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


class ProveFakeSub:
    name = "qwen"

    def __init__(self):
        self.memory = FakeMem(1.0)
        self._mem = self.memory
        self.steer = FakeSteer({"warm": 0.4})

    def chat(self, messages, max_new=256, sample=True):
        base = "A much longer rambling reply." if float(self.steer.strength.get("warm", 0.0) or 0.0) <= 0 \
            else "Short warm answer."
        return base


def _post(path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    monkeypatch.setattr(cs, "SUB", ProveFakeSub())
    return tmp_path


def _seed_run():
    return runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="QwenSubstrate",
                         messages=[{"role": "user", "content": "hi"}],
                         response="SAMPLED, never a baseline", behavior={"active_dials": {"warm": 0.4}})


def test_format_prove_renders_a_genuine_server_response(iso):
    rid = _seed_run()
    out_json = _post(f"/runs/{rid}/receipts", {})
    assert "error" not in out_json
    out = explain_cmd.format_prove(out_json)
    assert isinstance(out, str) and out
    assert rid in out
    assert "dial warm" in out


def test_format_prove_renders_the_coalitions_report_over_a_genuine_response(iso):
    rid = _seed_run()
    out_json = _post(f"/runs/{rid}/receipts", {"coalitions": True})
    assert "error" not in out_json and out_json["coalitions"]["available"] is True
    out = explain_cmd.format_prove(out_json)
    assert "coalition/Shapley credit" in out


def test_format_prove_renders_a_genuine_404_shape(iso):
    out_json = _post("/runs/run_does_not_exist/receipts", {})
    assert out_json == {"error": "run not found"}
    out = explain_cmd.format_prove(out_json)
    assert isinstance(out, str) and out


def test_format_prove_renders_a_genuine_503_shape(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out_json = _post(f"/runs/{rid}/receipts", {})
    assert out_json == {"error": "receipts require a ready product model worker"}
    out = explain_cmd.format_prove(out_json)
    assert isinstance(out, str) and out


# ------------------------------------------------------------------------------------------- _fetch_prove

def test_fetch_prove_is_an_honest_cloznerror_when_studio_is_down():
    port = clozn_cli._free_port()
    with pytest.raises(clozn_cli.CloznError):
        explain_cmd._fetch_prove(port, "run_x", mode="regen", coalitions=False, coalitions_batch="auto")


# -------------------------------------------------------------------------------- CLI wiring (--coalitions)

def test_prove_is_registered_with_the_expected_defaults():
    args = clozn_cli.build_parser().parse_args(["prove", "run_x"])
    assert args.fn is clozn_cli.cmd_prove
    assert args.run_id == "run_x" and args.last is False
    assert args.mode == "regen" and args.coalitions is False and args.coalitions_batch == "auto"
    assert args.json is False


def test_prove_accepts_coalitions_flags():
    args = clozn_cli.build_parser().parse_args(
        ["prove", "run_x", "--coalitions", "--coalitions-batch", "approximate", "--mode", "both"])
    assert args.coalitions is True and args.coalitions_batch == "approximate" and args.mode == "both"


def test_prove_needs_a_run_id_or_last():
    args = clozn_cli.build_parser().parse_args(["prove"])
    with pytest.raises(clozn_cli.CloznError):
        clozn_cli.cmd_prove(args)


def test_cmd_prove_threads_coalitions_flags_into_fetch_prove(monkeypatch, capsys):
    seen = {}

    def _fake_fetch(port, rid, *, mode, coalitions, coalitions_batch):
        seen.update(port=port, rid=rid, mode=mode, coalitions=coalitions, coalitions_batch=coalitions_batch)
        return dict(HAPPY_PATH)

    monkeypatch.setattr(explain_cmd, "_fetch_prove", _fake_fetch)
    args = clozn_cli.build_parser().parse_args(
        ["prove", "run_9", "--coalitions", "--coalitions-batch", "approximate", "--port", "9999"])
    clozn_cli.cmd_prove(args)
    assert seen == {"port": 9999, "rid": "run_9", "mode": "regen", "coalitions": True,
                    "coalitions_batch": "approximate"}
    assert "run_1" in capsys.readouterr().out


def test_cmd_prove_json_flag_prints_raw_json(monkeypatch, capsys):
    monkeypatch.setattr(explain_cmd, "_fetch_prove", lambda *a, **k: dict(HAPPY_PATH))
    args = clozn_cli.build_parser().parse_args(["prove", "run_1", "--json"])
    clozn_cli.cmd_prove(args)
    printed = json.loads(capsys.readouterr().out)
    assert printed == HAPPY_PATH
