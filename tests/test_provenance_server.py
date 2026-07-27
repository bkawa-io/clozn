"""HTTP-level tests for POST /runs/<id>/provenance. Drives the real do_POST handler with the no-socket
object.__new__(H) trick used throughout this test suite (mirrors tests/test_influence_map_server.py).
trace_provenance itself is monkeypatched -- these tests never touch a live engine."""
from __future__ import annotations

import io
import json
import types

import pytest

import clozn.analysis.provenance as prov
from clozn.server import app as cs
import clozn.runs.store as runlog


def _post(path, body=None):
    raw = json.dumps(body if body is not None else {}).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    handler.requestline = f"POST {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "POST"
    handler.do_POST()
    head, _, payload = handler.wfile.getvalue().partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload.decode("utf-8"))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    monkeypatch.setattr(cs, "ENGINE", None)
    return tmp_path


def _seed(*, final_prompt="EXACT RENDERED PROMPT", response="Tokyo"):
    return runlog.record(
        source="openai_api", messages=[{"role": "user", "content": "what is the capital?"}],
        response=response, final_prompt=final_prompt,
    )


_RECEIPT = {
    "ok": True, "answer": "Tokyo", "verdict": "CONTEXT_CARRIED", "dependence": 0.9,
    "best_control_ratio": 20.0, "span": [1], "span_tokens": ["Japan"],
}

_BLOCKED = {"ok": False, "blocked": "engine lacks attn_knockout; start clozn-server with --no-flash-attn"}


# --------------------------------------------------------------------------------------------- 404 / 400

def test_run_not_found_is_404(isolated):
    status, out = _post("/runs/missing/provenance")
    assert status == 404 and out == {"error": "run not found"}


def test_run_without_final_prompt_is_200_ok_false(isolated):
    rid = _seed(final_prompt=None)
    status, out = _post(f"/runs/{rid}/provenance")
    assert status == 200
    assert out["ok"] is False
    assert "final_prompt" in out["blocked"]


def test_run_without_recorded_response_is_200_ok_false(isolated):
    rid = _seed(response="")
    status, out = _post(f"/runs/{rid}/provenance")
    assert status == 200
    assert out["ok"] is False
    assert "response" in out["blocked"]


def test_invalid_focus_is_400(isolated, monkeypatch):
    rid = _seed()
    monkeypatch.setattr(prov, "trace_provenance", lambda *a, **k: pytest.fail("must not be called"))
    status, out = _post(f"/runs/{rid}/provenance", {"focus": [1]})
    assert status == 400 and "focus" in out["error"]
    status, out = _post(f"/runs/{rid}/provenance", {"focus": ["a", "b"]})
    assert status == 400 and "focus" in out["error"]


def test_invalid_seed_is_400(isolated, monkeypatch):
    rid = _seed()
    monkeypatch.setattr(prov, "trace_provenance", lambda *a, **k: pytest.fail("must not be called"))
    status, out = _post(f"/runs/{rid}/provenance", {"seed": "not-an-int"})
    assert status == 400 and "seed" in out["error"]


# ------------------------------------------------------------------------------- dispatch to the analysis

def test_calls_trace_provenance_with_the_runs_own_prompt_and_recorded_answer(isolated, monkeypatch):
    rid = _seed(final_prompt="EXACT RENDERED PROMPT", response="Tokyo")
    captured = {}

    def fake(prompt, continuation, **kwargs):
        captured["prompt"] = prompt
        captured["continuation"] = continuation
        captured["kwargs"] = kwargs
        return dict(_RECEIPT)

    monkeypatch.setattr(prov, "trace_provenance", fake)
    status, out = _post(f"/runs/{rid}/provenance")
    assert status == 200
    assert out == _RECEIPT
    assert captured["prompt"] == "EXACT RENDERED PROMPT"
    assert captured["continuation"] == "Tokyo"
    assert captured["kwargs"]["focus"] is None
    assert captured["kwargs"]["seed"] == 0


def test_forwards_focus_and_seed_from_the_body(isolated, monkeypatch):
    rid = _seed()
    captured = {}

    def fake(prompt, continuation, **kwargs):
        captured["kwargs"] = kwargs
        return dict(_RECEIPT)

    monkeypatch.setattr(prov, "trace_provenance", fake)
    status, out = _post(f"/runs/{rid}/provenance", {"focus": [2, 5], "seed": 9})
    assert status == 200
    assert captured["kwargs"]["focus"] == (2, 5)
    assert captured["kwargs"]["seed"] == 9


def test_blocked_dict_passes_through_as_200_not_an_error(isolated, monkeypatch):
    """The capability-unavailable choice: trace_provenance's own {"ok": False, "blocked": ...} dict
    (e.g. the engine lacks attn_knockout) ships as a 200 body, never a 500/503 -- see
    clozn/server/routes/provenance.py's module docstring for the reasoning."""
    rid = _seed()
    monkeypatch.setattr(prov, "trace_provenance", lambda *a, **k: dict(_BLOCKED))
    status, out = _post(f"/runs/{rid}/provenance")
    assert status == 200
    assert out == _BLOCKED


def test_uses_the_active_substrates_engine_base_url(isolated, monkeypatch):
    rid = _seed()
    captured = {}
    fake_sub = types.SimpleNamespace(engine=types.SimpleNamespace(base="http://sub-engine:1234"))
    monkeypatch.setattr(cs, "SUB", fake_sub)

    def fake(prompt, continuation, **kwargs):
        captured["kwargs"] = kwargs
        return dict(_RECEIPT)

    monkeypatch.setattr(prov, "trace_provenance", fake)
    _post(f"/runs/{rid}/provenance")
    assert captured["kwargs"]["engine_url"] == "http://sub-engine:1234"


def test_falls_back_to_the_module_level_engine_when_sub_has_none(isolated, monkeypatch):
    rid = _seed()
    captured = {}
    monkeypatch.setattr(cs, "SUB", types.SimpleNamespace())   # no .engine attribute at all
    monkeypatch.setattr(cs, "ENGINE", types.SimpleNamespace(base="http://module-engine:80"))

    def fake(prompt, continuation, **kwargs):
        captured["kwargs"] = kwargs
        return dict(_RECEIPT)

    monkeypatch.setattr(prov, "trace_provenance", fake)
    _post(f"/runs/{rid}/provenance")
    assert captured["kwargs"]["engine_url"] == "http://module-engine:80"


def test_omits_engine_url_kwarg_when_nothing_configured_letting_trace_provenance_use_its_own_default(
    isolated, monkeypatch,
):
    rid = _seed()
    captured = {}

    def fake(prompt, continuation, **kwargs):
        captured["kwargs"] = kwargs
        return dict(_RECEIPT)

    monkeypatch.setattr(prov, "trace_provenance", fake)
    _post(f"/runs/{rid}/provenance")
    assert "engine_url" not in captured["kwargs"]
