"""Model-free contracts for raw engine harvest/observe.

The real gateway handler is driven without a socket. Fake numpy activations prove the transform is
applied at the selected position. No worker, model, Torch forward, or GPU is involved.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from clozn.server import app as cs


class FakeHarvest:
    def __init__(self):
        self.tokens = ["The", " capital", " is"]
        self.activations = np.array([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
        self.layer = 7
        self.n_embd = 2


class FakeObservation:
    moved_l2 = 0.625
    baseline_top = [{"token": " Paris", "prob": 0.6}, {"token": " Rome", "prob": 0.2}]
    edited_top = [{"token": " Rome", "prob": 0.55}, {"token": " Paris", "prob": 0.3}]

    def summary(self):
        return {"changed": True}

    def shifted(self):
        return True


class FakeEngine:
    def __init__(self):
        self.harvest_texts = []
        self.observe_call = None

    def harvest(self, text):
        self.harvest_texts.append(text)
        return FakeHarvest()

    def edit_and_observe(self, text, transform, positions):
        before = FakeHarvest()
        after = transform(before.activations)
        self.observe_call = {"text": text, "positions": positions, "after": after}
        return before, FakeObservation()


def _dispatch(path, body=None):
    raw = json.dumps(body or {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    status = int(head.splitlines()[0].split()[1])
    return status, json.loads(payload.decode("utf-8"))


@pytest.fixture()
def fake_runtime(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(cs, "ENGINE", engine)
    return engine


def test_engine_harvest_returns_raw_norms_and_metadata(fake_runtime):
    engine = fake_runtime

    status, out = _dispatch("/engine/harvest", {"text": "The capital is"})

    assert status == 200
    assert engine.harvest_texts == ["The capital is"]
    assert out == {"tokens": ["The", " capital", " is"], "layer": 7, "n_embd": 2,
                   "norms": [5.0, 2.0, 1.0]}


def test_engine_observe_scales_only_the_selected_position(fake_runtime):
    engine = fake_runtime

    status, out = _dispatch("/engine/observe", {"text": "The capital is", "position": 1, "scale": 3})

    assert status == 200
    np.testing.assert_array_equal(engine.observe_call["after"],
                                  np.array([[3.0, 4.0], [0.0, 6.0], [1.0, 0.0]], dtype=np.float32))
    assert engine.observe_call["positions"] == [1]
    assert out["shifted"] is True
    assert out["moved_l2"] == 0.625
    assert out["position"] == 1 and out["scale"] == 3.0
    assert out["baseline_top"][0]["token"] == " Paris"
    assert out["edited_top"][0]["token"] == " Rome"

