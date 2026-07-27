"""Offline tests for the engine SDK — no server, no checkpoint (CPU-only, CI-safe).

Covers the two things that can silently corrupt the read -> edit -> write -> observe loop:
the base64 LE-float32 wire codec, and the request shapes EngineClient sends. The transport
is stubbed (the client's _request is replaced), so these never open a socket.

Run with pytest (`pytest test_clozn_engine.py -q`) or directly (`python test_clozn_engine.py`).
"""

from __future__ import annotations

import base64

import numpy as np

from clozn_engine import EngineClient, EngineError, Observation, decode_tensor, flatten_values


def _wire(arr: np.ndarray) -> dict:
    """Encode a numpy array the way clozn_server.cpp's tensor_json_f32 does."""
    a = np.ascontiguousarray(arr, dtype="<f4")
    return {"dtype": "float32", "shape": list(a.shape),
            "data": base64.b64encode(a.tobytes()).decode("ascii")}


# --------------------------------------------------------------------------- wire codec

def test_decode_tensor_roundtrip_exact():
    a = (np.arange(24, dtype="<f4").reshape(4, 6) * 0.5 - 3.25)
    b = decode_tensor(_wire(a))
    assert b.shape == (4, 6)
    assert np.array_equal(a, b), "tensor codec is not bit-exact"


def test_decode_tensor_shape_guard():
    a = np.arange(24, dtype="<f4").reshape(4, 6)
    bad = {"dtype": "float32", "shape": [4, 7], "data": _wire(a)["data"]}  # 24 floats, claims 28
    try:
        decode_tensor(bad)
    except ValueError:
        return
    raise AssertionError("decode_tensor accepted a shape/byte-count mismatch")


def test_decode_tensor_rejects_non_float32():
    try:
        decode_tensor({"dtype": "float64", "shape": [1], "data": ""})
    except ValueError:
        return
    raise AssertionError("decode_tensor accepted a non-float32 dtype")


def test_flatten_values_row_major():
    a = (np.arange(21, dtype="<f4").reshape(3, 7))
    flat = flatten_values(a)
    assert len(flat) == 21
    # row i of the slice must land at offset i * n_embd (position-major, the server's contract).
    assert abs(flat[0] - float(a[0, 0])) < 1e-6
    assert abs(flat[7] - float(a[1, 0])) < 1e-6
    assert abs(flat[14] - float(a[2, 0])) < 1e-6


# --------------------------------------------------------------------------- client wiring

class _Stub(EngineClient):
    """An EngineClient whose transport is a canned-response stub. Records every call as
    (method, path, body) in .calls and returns responses.pop(0) for each request."""

    def __init__(self, responses):
        super().__init__()
        self.calls = []
        self._responses = list(responses)

    def _request(self, method, path, body=None):
        self.calls.append((method, path, body))
        return self._responses.pop(0)


def test_harvest_parses_response():
    acts = (np.arange(10, dtype="<f4").reshape(2, 5) - 4.0)
    eng = _Stub([{"tokens": ["a", "b"], "layer": 3, "n_tokens": 2, "n_embd": 5,
                  "activations": _wire(acts)}])
    h = eng.harvest("ab", layer=3)
    assert h.tokens == ["a", "b"]
    assert h.layer == 3 and h.n_tokens == 2 and h.n_embd == 5
    assert np.array_equal(h.activations, acts)
    # the request carried the layer override
    method, path, body = eng.calls[0]
    assert method == "POST" and path == "/harvest" and body["layer"] == 3


def test_score_forwards_attn_knockout_verbatim():
    """Phase 4.2 (roadmap §7 item 2): score() must pass attn_knockout specs through to the wire body
    unmodified -- the engine validates/refuses them, this client does not second-guess the shape."""
    eng = _Stub([{"n_prompt": 2, "n_cont": 1, "tokens": [], "sum_logprob": -0.5}])
    knockouts = [{"layer": 2, "queries": [3, 4], "keys": [0, 1], "renormalize": True}]
    eng.score(prompt="hi", continuation_ids=[7], attn_knockout=knockouts)
    method, path, body = eng.calls[0]
    assert method == "POST" and path == "/score"
    assert body["attn_knockout"] == knockouts
    # a copy, not the same list object -- the caller's list must not be aliased into the wire body
    assert body["attn_knockout"] is not knockouts
    assert body["attn_knockout"][0] is not knockouts[0]


def test_score_omits_attn_knockout_when_absent():
    eng = _Stub([{"n_prompt": 2, "n_cont": 1, "tokens": [], "sum_logprob": -0.5}])
    eng.score(prompt="hi", continuation_ids=[7])
    _method, _path, body = eng.calls[0]
    assert "attn_knockout" not in body


def test_unembed_row_sends_token_id_and_parses():
    eng = _Stub([{"token_id": 42, "piece": " ocean", "d_model": 4, "vector": [1.0, 2.0, 3.0, 4.0]}])
    r = eng.unembed_row(42)
    method, path, body = eng.calls[0]
    assert method == "POST" and path == "/jlens/unembed_row" and body == {"token_id": 42}
    assert r["vector"] == [1.0, 2.0, 3.0, 4.0]
    assert r["d_model"] == 4


def test_write_state_sends_flat_values_and_parses():
    rows = np.ones((2, 4), dtype="<f4")
    eng = _Stub([{"applied": True, "layer": 6, "moved_l2": 12.5,
                  "baseline_top": [{"token": " is", "prob": 0.7}],
                  "edited_top": [{"token": " of", "prob": 0.5}]}])
    obs = eng.write_state("hello", 6, positions=[1, 3], values=rows)
    method, path, body = eng.calls[0]
    assert path == "/state"
    assert body["positions"] == [1, 3]
    assert len(body["values"]) == 2 * 4, "values must be flattened to positions*n_embd"
    assert obs.applied and obs.moved_l2 == 12.5
    assert obs.shifted(), "top-1 changed ' is' -> ' of', shifted() should be True"


def test_write_state_rejection_surfaces_error():
    eng = _Stub([{"applied": False, "moved_l2": 0.0, "error": "bad layer"}])
    obs = eng.write_state("x", 999, positions=[0], values=np.zeros((1, 4), dtype="<f4"))
    assert not obs.applied and not obs.shifted()
    assert "bad layer" in obs.summary()


def test_edit_and_observe_writes_changed_rows_at_harvest_layer():
    acts = np.zeros((3, 4), dtype="<f4")
    eng = _Stub([
        {"tokens": ["a", "b", "c"], "layer": 5, "n_tokens": 3, "n_embd": 4,
         "activations": _wire(acts)},                       # the harvest
        {"applied": True, "layer": 5, "moved_l2": 1.0, "baseline_top": [], "edited_top": []},  # the write
    ])

    def bump_row1(a):
        a[1] += 1.0          # change only the middle row
        return a

    h, obs = eng.edit_and_observe("abc", transform=bump_row1)
    assert obs.applied
    # the write call must target ONLY the changed row, at the SAME layer the harvest read (5).
    _, path, body = eng.calls[1]
    assert path == "/state"
    assert body["layer"] == 5
    assert body["positions"] == [1], "only the row the transform changed should be written"
    assert len(body["values"]) == 1 * 4


def test_observation_shifted_requires_applied():
    assert not Observation(applied=False, layer=1, moved_l2=0.0).shifted()
    same = Observation(applied=True, layer=1, moved_l2=1.0,
                       baseline_top=[{"token": " a", "prob": 0.5}],
                       edited_top=[{"token": " a", "prob": 0.4}])
    assert not same.shifted(), "same top-1 token is not a shift"


def test_http_error_becomes_engine_error(monkeypatch):
    import io
    import urllib.error

    def boom(*_a, **_k):
        raise urllib.error.HTTPError("http://x/state", 400, "Bad Request", {},
                                     io.BytesIO(b'{"error": "need values"}'))

    monkeypatch.setattr("urllib.request.urlopen", boom)
    eng = EngineClient()
    try:
        eng.health()
    except EngineError as e:
        assert "need values" in str(e)
        return
    raise AssertionError("a 400 response should raise EngineError with the server message")


# --------------------------------------------------------------------------- direct runner

def _run_all() -> int:
    """Run every test_* in this module without pytest (CI fallback). Supplies a tiny
    monkeypatch shim for the one test that takes it."""
    class _MP:
        def __init__(self):
            self._undo = []

        def setattr(self, target, value):
            mod, _, attr = target.rpartition(".")
            import importlib
            obj = importlib.import_module(mod)
            self._undo.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, value)

        def undo(self):
            for obj, attr, old in reversed(self._undo):
                setattr(obj, attr, old)

    import inspect
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = 0
    for name, fn in fns:
        mp = _MP()
        try:
            if "monkeypatch" in inspect.signature(fn).parameters:
                fn(mp)
            else:
                fn()
            passed += 1
            print(f"  ok  {name}")
        finally:
            mp.undo()
    print(f"\n{passed}/{len(fns)} passed")
    return 0 if passed == len(fns) else 1


if __name__ == "__main__":
    raise SystemExit(_run_all())
