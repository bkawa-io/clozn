"""test_jlens_server -- the J3 studio seam feeding the Run Inspector's J-lens panel.

Two layers, both model-free (no engine, no GPU):
  * EngineSubstrate.jlens(text, layer, topk) -- the /jlens HTTP proxy (mirrors score_tokens' /score proxy):
    availability from /health's jlens block, the engine payload + available_layers, EngineError -> clean
    surface, graceful absence.
  * POST /jlens and POST /runs/<id>/jlens -- the two backend routes returning the exact frontend contract
    (available/run_id/layer/available_layers/text_source/tokens/readouts/provenance), driven through the
    REAL do_POST handler with no socket (object.__new__(H)), mirroring test_rederive_server.py.
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
sys.path.insert(0, os.path.join(REPO, "engine", "client"))

from clozn.server import app as cs   # noqa: E402
import clozn.runs.store as runlog                # noqa: E402


# =========================================================== EngineSubstrate.jlens (the /jlens proxy)

class FakeEngineJlens:
    """Stands in for clozn_engine.EngineClient inside EngineSubstrate.jlens: .health() (advertises the
    jlens block) + .jlens()."""

    PAYLOAD = {"layer": 25, "n_tokens": 2, "tokens": ["The", " boot"],
               "readouts": [[{"id": 1, "piece": " Italy", "score": 19.6}],
                            [{"id": 2, "piece": " pasta", "score": 9.1}]]}

    def __init__(self, jlens_layers=(2, 14, 21, 25), raise_unknown=False):
        self._layers = list(jlens_layers)
        self._raise = raise_unknown
        self.calls = []

    def health(self):
        h = {"status": "ok", "model": "qwen.gguf"}
        if self._layers:
            h["jlens"] = {"layers": self._layers, "default_layer": self._layers[0]}
        return h

    def jlens(self, text, layer=None, topk=5):
        self.calls.append({"text": text, "layer": layer, "topk": topk})
        if self._raise:
            raise cs.EngineError("POST /jlens -> 400: no J-lens sidecar for that layer")
        return dict(self.PAYLOAD)


def _bare_engine_sub(engine):
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = engine
    return sub


def test_engine_substrate_jlens_success_returns_payload_plus_available_layers():
    eng = FakeEngineJlens()
    sub = _bare_engine_sub(eng)
    out = sub.jlens("the country shaped like a boot is", layer=25, topk=5)
    assert out["available"] is True
    assert out["layer"] == 25
    assert out["available_layers"] == [2, 14, 21, 25]
    assert out["tokens"] == ["The", " boot"]
    assert out["readouts"][0][0]["piece"] == " Italy"
    assert eng.calls[-1] == {"text": "the country shaped like a boot is", "layer": 25, "topk": 5}


def test_engine_substrate_jlens_unavailable_when_engine_has_no_jlens_block():
    sub = _bare_engine_sub(FakeEngineJlens(jlens_layers=()))    # engine started WITHOUT --jlens
    out = sub.jlens("hi", layer=None)
    assert out == {"available": False, "reason": "the engine was started without --jlens"}


class FakeEngineDown:
    """.health() always raises -- the engine is FULLY unreachable, as opposed to
    FakeEngineJlens(jlens_layers=()) above, which simulates a REACHABLE engine that simply wasn't started
    with --jlens. Fix #5 (engine-down pressure test): these two used to collapse into the identical wrong
    "started without --jlens" reason -- .jlens() must never even be reached once health() itself fails."""

    def health(self):
        raise OSError("connection refused")

    def jlens(self, text, layer=None, topk=5):
        raise AssertionError("must never be called once health() itself failed")


def test_engine_substrate_jlens_distinguishes_engine_down_from_no_jlens_flag(monkeypatch):
    down = FakeEngineDown()
    down.base = "http://127.0.0.1:8080"
    monkeypatch.setattr(cs, "ENGINE", down)     # ctx._engine_unreachable_message() reads ENGINE.base
    sub = _bare_engine_sub(down)
    out = sub.jlens("hi", layer=None)
    assert out == {"available": False,
                   "reason": "engine not reachable at http://127.0.0.1:8080 -- is it running?"}


def test_engine_substrate_jlens_unknown_layer_surfaces_cleanly():
    sub = _bare_engine_sub(FakeEngineJlens(raise_unknown=True))
    out = sub.jlens("hi", layer=99)
    assert out["available"] is True                 # the lens IS loaded; just not that layer
    assert "error" in out
    assert out["available_layers"] == [2, 14, 21, 25]
    assert out["readouts"] == []


# =========================================================== the two backend routes (no-socket handler)

class FakeJlensSub:
    """A substrate exposing just .jlens (what the routes call). Returns a canned NORMALIZED dict (as
    EngineSubstrate.jlens would)."""
    name = "engine"

    DEFAULT = {"available": True, "layer": 25, "available_layers": [2, 14, 21, 25],
               "n_tokens": 2, "tokens": ["The", " boot"],
               "readouts": [[{"id": 1, "piece": " Italy", "score": 19.6}],
                            [{"id": 2, "piece": " pasta", "score": 9.1}]]}

    def __init__(self, res=None):
        self._res = res if res is not None else dict(self.DEFAULT)
        self.calls = []

    def jlens(self, text, layer=None, topk=5):
        self.calls.append({"text": text, "layer": layer, "topk": topk})
        return dict(self._res)


def _post(path, body_obj):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(cs, "SUB", FakeJlensSub())
    # a manifest so the provenance builder is exercised (fresh cache per test)
    jl = tmp_path / "jlens"
    jl.mkdir()
    (jl / "manifest.json").write_text(json.dumps({
        "model": "Qwen/Qwen2.5-7B-Instruct", "layers": [2, 14, 21, 25],
        "fitted_on": {"quant": "nf4", "n_prompts": 100}}), encoding="utf-8")
    cs._jlens_provenance._cached = None
    return tmp_path


def _seed_run(response="Italy is shaped like a boot."):
    return runlog.record(source="openai_api", client="studio", model="clozn-engine", substrate="engine",
                         messages=[{"role": "user", "content": "which country is boot-shaped?"}],
                         response=response)


# ----- POST /jlens (general passthrough) -----

def test_jlens_route_happy_path_matches_the_contract(iso):
    head, out = _post("/jlens", {"text": "the country shaped like a boot is", "layer": 25})
    assert "200" in head
    assert out["available"] is True
    assert out["run_id"] is None
    assert out["layer"] == 25
    assert out["available_layers"] == [2, 14, 21, 25]
    assert out["text_source"] == "input"
    assert out["tokens"] == ["The", " boot"]
    assert out["readouts"][0][0] == {"id": 1, "piece": " Italy", "score": 19.6}
    prov = out["provenance"]
    assert prov["kind"] == "jacobian_lens"
    assert prov["fit_model"] == "Qwen2.5-7B (HF, nf4, 100 prompts)"
    assert prov["layers"] == [2, 14, 21, 25]
    assert "NOT the model's literal thought" in prov["note"]     # the honest, verbatim caveat


def test_jlens_route_needs_text(iso):
    head, out = _post("/jlens", {})
    assert "400" in head
    assert out == {"error": "need a 'text' to read"}


def test_jlens_route_degrades_when_substrate_has_no_jlens(iso, monkeypatch):
    class NoJlens:  # e.g. the qwen/dream substrate
        name = "qwen"
    monkeypatch.setattr(cs, "SUB", NoJlens())
    head, out = _post("/jlens", {"text": "hi"})
    assert "200" in head
    assert out["available"] is False
    assert "no J-lens" in out["reason"]


def test_jlens_route_protocol_emits_workspace_readouts(iso):
    _, out = _post("/jlens", {"text": "hi", "protocol": True})
    wr = out["workspace_readouts"]
    assert len(wr) == 2
    assert wr[0]["type"] == "workspace_readout"
    assert wr[0]["provider_type"] == "jacobian_lens"
    assert wr[0]["readout_kind"] == "token"
    assert wr[0]["provider"] == "jacobian_lens_l25"
    assert wr[0]["top_readouts"][0] == {"label": " Italy", "score": 19.6}


# ----- POST /runs/<id>/jlens (the inspector feed) -----

def test_runs_jlens_reads_the_stored_response(iso):
    rid = _seed_run()
    head, out = _post(f"/runs/{rid}/jlens", {"layer": 25})
    assert "200" in head
    assert out["available"] is True
    assert out["run_id"] == rid
    assert out["text_source"] == "response"                      # read the reply, and SAY so
    assert out["provenance"]["fit_model"] == "Qwen2.5-7B (HF, nf4, 100 prompts)"
    # the substrate was asked to read the run's stored response text
    assert cs.SUB.calls[-1]["text"] == "Italy is shaped like a boot."


def test_runs_jlens_falls_back_to_last_user_when_no_response(iso):
    rid = runlog.record(source="openai_api", substrate="engine",
                        messages=[{"role": "user", "content": "boot country?"}], response="")
    _, out = _post(f"/runs/{rid}/jlens", {})
    assert out["available"] is True
    assert out["text_source"] == "last_user_message"
    assert cs.SUB.calls[-1]["text"] == "boot country?"


def test_runs_jlens_unknown_run_is_404(iso):
    head, out = _post("/runs/run_nope/jlens", {})
    assert "404" in head
    assert out == {"error": "run not found"}


def test_runs_jlens_degrades_when_substrate_has_no_jlens(iso, monkeypatch):
    rid = _seed_run()
    monkeypatch.setattr(cs, "SUB", None)
    head, out = _post(f"/runs/{rid}/jlens", {})
    assert "200" in head
    assert out == {"available": False, "run_id": rid,
                   "reason": "the product model worker has no J-lens"}


def test_runs_jlens_unknown_layer_surfaces_error_and_available_layers(iso, monkeypatch):
    rid = _seed_run()
    monkeypatch.setattr(cs, "SUB", FakeJlensSub(res={
        "available": True, "error": "POST /jlens -> 400: no J-lens sidecar for that layer",
        "available_layers": [2, 14, 21, 25], "layer": 99, "n_tokens": 0, "tokens": [], "readouts": []}))
    head, out = _post(f"/runs/{rid}/jlens", {"layer": 99})
    assert "200" in head
    assert out["available"] is True
    assert "error" in out
    assert out["available_layers"] == [2, 14, 21, 25]
