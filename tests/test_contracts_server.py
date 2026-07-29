"""HTTP-level tests for GET /contracts/hooks and POST /contracts/replay (roadmap Phase 4.2). Drives
the REAL clozn_server do_GET/do_POST handler with the no-socket object.__new__(H) trick used
throughout this test suite (mirrors tests/test_influence_map_server.py)."""
from __future__ import annotations

import io
import json

import pytest

from clozn.server import app as cs


def _post(path, body=None):
    raw = json.dumps(body or {}).encode("utf-8")
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


def _get(path):
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(b"")
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "GET"
    handler.do_GET()
    head, _, payload = handler.wfile.getvalue().partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload.decode("utf-8"))


class FakeEngine:
    def __init__(self, *, health=None, raise_on_score=None):
        self.calls: list[dict] = []
        self._health = health if health is not None else {"capabilities": {"attn_knockout": True}}
        self._raise = raise_on_score

    def health(self) -> dict:
        return self._health

    def score(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        if self._raise is not None:
            raise self._raise
        return {"n_prompt": 2, "n_cont": 1, "tokens": [{"id": 1, "piece": "x", "logprob": -0.5}],
                "sum_logprob": -0.5}


class FakeSub:
    def __init__(self, engine=None):
        self.engine = engine


@pytest.fixture
def isolated(monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    return None


_MANIFEST = {
    "schema": "clozn.intervention_manifest.v1",
    "name": "http demo",
    "request": {"prompt": "The sky is", "continuation_ids": [101, 102]},
    "arms": [{"name": "steer only", "steer_vec": [0.1, 0.2]}],
}


# --------------------------------------------------------------------------------------- GET /contracts/hooks

def test_get_hooks_returns_the_versioned_document(isolated):
    status, out = _get("/contracts/hooks")
    assert status == 200
    assert out["schema"] == "clozn.hook_vocabulary.v1"
    assert {hook["name"] for hook in out["hooks"]} == {
        "l_out-<il>", "kq_soft_max-<il>", "kqv_out-<il>", "ffn_out-<il>",
    }


def test_get_hooks_works_with_no_substrate_at_all(isolated):
    """The vocabulary is static, code-derived documentation -- it must be servable even with SUB=None
    (no engine anywhere), unlike POST /contracts/replay."""
    status, _out = _get("/contracts/hooks")
    assert status == 200


# ------------------------------------------------------------------------------------- POST /contracts/replay

def test_post_replay_requires_a_manifest_object(isolated):
    status, out = _post("/contracts/replay", {})
    assert status == 400
    assert "manifest" in out["error"]

    status, out = _post("/contracts/replay", {"manifest": "not an object"})
    assert status == 400


def test_post_replay_requires_an_engine_backed_substrate(isolated, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    status, out = _post("/contracts/replay", {"manifest": _MANIFEST})
    assert status == 503
    assert "engine" in out["error"]


def test_post_replay_requires_an_engine_backed_substrate_when_sub_has_no_engine_attr(isolated, monkeypatch):
    monkeypatch.setattr(cs, "SUB", object())
    status, out = _post("/contracts/replay", {"manifest": _MANIFEST})
    assert status == 503


def test_post_replay_structural_validation_error_is_400(isolated, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub(FakeEngine()))
    bad_manifest = dict(_MANIFEST)
    bad_manifest["arms"] = []
    status, out = _post("/contracts/replay", {"manifest": bad_manifest})
    assert status == 400
    assert "non-empty array" in out["error"]


def test_post_replay_capability_refusal_is_409_typed(isolated, monkeypatch):
    engine = FakeEngine(health={"capabilities": {"attn_knockout": False}})
    monkeypatch.setattr(cs, "SUB", FakeSub(engine))
    manifest = dict(_MANIFEST)
    manifest["arms"] = [{"name": "cut", "attention_knockout": [
        {"layer": 0, "queries": [0], "keys": [0]},
    ]}]
    status, out = _post("/contracts/replay", {"manifest": manifest})
    assert status == 409
    assert out["performed"] is False
    assert out["error"]["code"] == "capability_unavailable"
    assert engine.calls == []  # refused before ever touching the engine


def test_post_replay_success_returns_full_result(isolated, monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(cs, "SUB", FakeSub(engine))
    status, out = _post("/contracts/replay", {"manifest": _MANIFEST})
    assert status == 200
    assert out["performed"] is True
    assert out["schema"] == "clozn.intervention_replay.v1"
    assert len(out["arms"]) == 1
    assert out["arms"][0]["name"] == "steer only"
    assert len(engine.calls) == 2  # baseline + 1 arm


def test_post_replay_unexpected_engine_exception_is_500_not_silently_dropped(isolated, monkeypatch):
    engine = FakeEngine(raise_on_score=RuntimeError("engine exploded"))
    monkeypatch.setattr(cs, "SUB", FakeSub(engine))
    status, out = _post("/contracts/replay", {"manifest": _MANIFEST})
    assert status == 500
    assert "engine exploded" in out["error"]


def test_post_replay_health_fetch_failure_degrades_to_empty_health_not_a_crash(isolated, monkeypatch):
    class ExplodingHealthEngine(FakeEngine):
        def health(self):
            raise RuntimeError("health unreachable")

    engine = ExplodingHealthEngine()
    monkeypatch.setattr(cs, "SUB", FakeSub(engine))
    # No attention_knockout arm -> no capability required -> should still succeed even though health()
    # itself raised (degrades to {} health, not a 500).
    status, out = _post("/contracts/replay", {"manifest": _MANIFEST})
    assert status == 200
    assert out["performed"] is True
    assert out["identity"] == {
        "model": None, "model_sha256": None, "architecture": None,
        "n_layer": None, "n_embd": None, "protocol_version": None, "mode": None,
    }
