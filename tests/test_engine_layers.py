"""harvest_layers: the per-layer activation-summary client method (clozn_engine.EngineClient) + its parsing.

The engine's /harvest/layers (GgmlAdapter::layer_summary) returns per-token L2 norms at EVERY layer from one
forward -- the depth x position "MRI" map. This checks the thin client wrapper parses that shape and
degrades cleanly. Model-free: _post is monkeypatched with a canned engine response (no server, no GPU).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "engine", "client"))

from clozn_engine import EngineClient  # noqa: E402


def test_harvest_layers_parses_the_summary(monkeypatch):
    canned = {"tokens": ["The", " cat"], "n_tokens": 2, "n_layer": 3,
              "norms": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], "layer_mean": [1.5, 3.5, 5.5]}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: canned if path == "/harvest/layers" else {})
    r = ec.harvest_layers("The cat")
    assert r["n_layer"] == 3 and r["n_tokens"] == 2
    assert r["tokens"] == ["The", " cat"]
    assert r["norms"] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]     # [n_layer][n_tokens]
    assert r["layer_mean"] == [1.5, 3.5, 5.5]                     # [n_layer]


def test_harvest_layers_sends_the_text(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {})
    ec.harvest_layers("hello world")
    assert seen["path"] == "/harvest/layers"
    assert seen["body"] == {"text": "hello world"}


def test_harvest_layers_tolerates_missing_fields(monkeypatch):
    """A degraded / empty engine reply must not KeyError -- the wrapper fills a clean empty summary."""
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: {})
    assert ec.harvest_layers("x") == {"tokens": [], "n_tokens": 0, "n_layer": 0, "norms": [], "layer_mean": []}
