"""tests/test_generation_guard_server.py -- the closed-loop disposition guardrail
(clozn/server/generation_guard.py) wired into POST /v1/chat/completions as the OPT-IN, DEFAULT-OFF
`clozn_guard` extension field (FRONTIER_BETS section 9.1 / experiment A1.1), including the per-model
guard-threshold calibration path (calibrated concept -> trigger-set signal; uncalibrated -> annotate-only)
and layer selection (never a hardcoded, possibly-invalid layer).

Model-free: drives the REAL clozn_server do_POST handler with no socket (object.__new__(H)), isolated
runlog/cards/settings/eval stores, and a FAKE engine client (never a live one). Concept resolution
(resolve_token_id / dir(c)) runs through the REAL clozn.behavior.steering.concept_dir math against tiny
on-disk J-lens/unembed FIXTURE files (mirrors tests/test_concept_dir.py's own fixture convention) -- only
the raw engine HTTP calls (.score/.complete/.intervene/.jlens/.apply_template/.health) are faked. The
per-model guard calibration file itself is faked too, via monkeypatching generation_guard.guard_calibration_
path into a tmp_path (mirrors tests/test_concept_dir.py's own concept_calibration_path monkeypatch trick).
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from clozn.server import app as cs                # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
from clozn.server import generation_guard as gg    # noqa: E402


import clozn.runs.store as runlog                  # noqa: E402


MODEL = "fake-clozn-model"
MODEL_SHA256 = "f" * 64
D_MODEL = 32
# NOTE: the REAL 7B guard-threshold calibration (guard_signal_calibrate.py) found its signal at layer 14 --
# but layer 14 has no entry in concept_dir.VALIDATED_MEDIAN_RESID_NORM (only 16/21/25 do; see item 2's own
# per-model concept-dial calibration gap), so a fake ConceptSteer.steer_toward() at layer 14 would fail on
# an UNRELATED missing-injection-magnitude error, not anything this test suite is about. These tests use
# layer 21 (one of the globally validated dir(c) injection layers) for the calibrated-firing scenarios so
# the guard-threshold-calibration mechanism is exercised in isolation; the layer-DISCOVERY test below picks
# its own local layer=14 fixture deliberately, since nothing ever fires there (no median_norm needed).
LAYER = 21
TRIGGER_TOKEN_ID = 5      # must be < the fixture's vocab size (D_MODEL) -- see _write_unembed_fixture
TRIGGER_MARKER = "BANNED_TRIGGER"


# ==================================================================================== fixtures (dir(c) plumbing)

def _orthogonal(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def _write_jlens_fixture(tmp_path, *, d_model=D_MODEL, layer=LAYER, seed=1):
    jdir = tmp_path / "jlens"
    jdir.mkdir()
    manifest = {"model": "fixture", "d_model": d_model, "vocab": d_model, "layers": [layer],
               "engine_default_tap_layer": layer}
    (jdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    J = _orthogonal(seed, d_model).astype(np.float32)
    J.astype("<f2").tofile(str(jdir / f"J_layer{layer}.f16"))
    return str(jdir)


def _write_unembed_fixture(tmp_path, *, d_model=D_MODEL, vocab=D_MODEL, seed=2):
    udir = tmp_path / "unembed"
    udir.mkdir()
    q = _orthogonal(seed, d_model)[:vocab].astype(np.float32)
    np.save(str(udir / "norm_weight.npy"), np.ones(d_model, dtype=np.float32))
    np.save(str(udir / "lm_head_weight.npy"), q)
    (udir / "unembed_meta.json").write_text(json.dumps({"rms_norm_eps": 1e-6}), encoding="utf-8")
    return str(udir)


# ==================================================================================== fake engine + substrate

class FakeGuardEngine:
    """Stands in for clozn_engine.EngineClient for the guard's production adapter: `.health` (jlens fitted
    layers -- LAYER SELECTION reads this), `.score` (token resolution), `.complete`/`.intervene` (chunked
    generation, scripted per-call-order), `.jlens` (disposition read, driven by a marker substring so tests
    stay simple), `.apply_template` (prompt render)."""

    def __init__(self, *, vocab, complete_pieces=(), intervene_pieces=(),
                trigger_token_id=TRIGGER_TOKEN_ID, marker=TRIGGER_MARKER,
                jlens_layers=(2, 14, 21, 25)):
        self.vocab = dict(vocab)
        self._complete_pieces = list(complete_pieces)
        self._intervene_pieces = list(intervene_pieces)
        self.complete_calls = []
        self.intervene_calls = []
        self.jlens_calls = []
        self._trigger_token_id = trigger_token_id
        self._marker = marker
        self._jlens_layers = list(jlens_layers)

    def health(self):
        return {"model": "fixture", "model_sha256": MODEL_SHA256,
                "jlens": {"on": True, "layers": self._jlens_layers}}

    def apply_template(self, messages):
        return "PROMPT:" + "".join(m.get("content", "") for m in messages)

    def score(self, prompt=None, continuation=None, topk=0, **kw):
        ids = self.vocab.get(continuation)
        if ids is None:
            raise AssertionError(f"unexpected /score continuation in test: {continuation!r}")
        return {"tokens": [{"id": i, "piece": continuation} for i in ids]}

    def complete(self, prompt, max_tokens=None, **kw):
        idx = len(self.complete_calls)
        self.complete_calls.append({"prompt": prompt, "max_tokens": max_tokens, "kw": kw})
        return {"choices": [{"text": self._complete_pieces[idx], "finish_reason": "length"}]}

    def intervene(self, prompt, vector=None, coef=None, layer=None, max_tokens=None, **kw):
        idx = len(self.intervene_calls)
        self.intervene_calls.append({"prompt": prompt, "vector": vector, "coef": coef, "layer": layer,
                                     "max_tokens": max_tokens, "kw": kw})
        return {"choices": [{"text": self._intervene_pieces[idx], "finish_reason": "length"}]}

    def jlens(self, text, layer=None, topk=5):
        self.jlens_calls.append({"text": text, "layer": layer, "topk": topk})
        if self._marker in text:
            return {"readouts": [[{"id": self._trigger_token_id, "score": 9.0}]]}
        return {"readouts": [[]]}


class FakeGuardSub:
    """The minimal substrate surface the guard path needs: `.chat` only to satisfy try_post's worker-
    availability gate (never actually called on the guard path -- would be a real bug if it were),
    `.engine` (the raw client the production adapter drives directly), and `.model_sha256` (the per-model
    guard-calibration lookup key)."""
    name = "engine"

    def __init__(self, engine, model_sha256=MODEL_SHA256):
        self.engine = engine
        self.model_sha256 = model_sha256

    def chat(self, *a, **kw):
        raise AssertionError("the guard path must never fall through to sub.chat()")


class TraceSub:
    """A qwen-shaped substrate for the OFF path -- mirrors test_selective_generation_server.py's TraceSub,
    proving the guard's off-switch leaves the ordinary chat() pipeline completely untouched."""
    name = "qwen"

    def __init__(self, reply="An ordinary ungated reply."):
        class _Mem:
            memory_strength = 1.0
            rules: list = []
            prefix = None
        self.memory = _Mem()
        self._mem = self.memory
        self._reply = reply
        self._run_meta = {"model_id": MODEL, "sampler_mode": "greedy", "sampling": "greedy",
                          "temperature": 0.0}

        class _Steer:
            strength: dict = {}

            def active(self):
                return {}
        self.steer = _Steer()

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        if trace_out is not None:
            trace_out.extend([{"piece": "ordinary", "conf": 0.9}])
        return self._reply

    def last_finish_reason(self):
        return "stop"

    def run_meta(self):
        return dict(self._run_meta)


def _dispatch(path, body_obj):
    raw = json.dumps(body_obj).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    return h.wfile.getvalue()


def _post(path, body_obj):
    raw = _dispatch(path, body_obj)
    status = int(raw.split(b" ", 2)[1])
    _, _, payload = raw.partition(b"\r\n\r\n")
    return status, json.loads(payload.decode("utf-8"))


def _body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "tell me something"}], **extra}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.delenv("CLOZN_JLENS_DIR", raising=False)
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    return tmp_path


@pytest.fixture
def dirc_fixtures(tmp_path, monkeypatch):
    """Real, tiny on-disk J-lens + unembed exports so ConceptSteer.compute() succeeds for real -- see the
    module docstring. Used by every test where concept resolution must actually succeed; the fail-closed
    test deliberately configures neither."""
    jdir = _write_jlens_fixture(tmp_path)
    udir = _write_unembed_fixture(tmp_path)
    monkeypatch.setenv("CLOZN_JLENS_DIR", jdir)
    monkeypatch.setenv("CLOZN_DIRC_UNEMBED_DIR", udir)
    return tmp_path


def _write_guard_calibration(tmp_path, monkeypatch, *, concept="violence", layer=LAYER, threshold=1.0,
                            trigger_ids=(TRIGGER_TOKEN_ID,), model_sha256=MODEL_SHA256):
    """A real on-disk guard_threshold_calibration.json, redirected into tmp_path via monkeypatching
    generation_guard.guard_calibration_path -- mirrors tests/test_concept_dir.py's own
    concept_calibration_path monkeypatch trick, never touching the real ~/.clozn."""
    cal_path = tmp_path / "guard_threshold_calibration.json"
    payload = {
        "schema_version": gg.GUARD_CALIBRATION_SCHEMA, "model_sha256": model_sha256,
        "default_layer": layer,
        "concepts": {concept: {
            "layer": layer, "threshold": threshold, "trigger_ids": list(trigger_ids),
            "trigger_pieces": [TRIGGER_MARKER], "catch": 1.0, "fp": 0.0, "n_battery": 12,
            "note": "small-battery calibration (6 banned + 6 clean); presence-separated.",
        }},
    }
    cal_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(gg, "guard_calibration_path", lambda model_sha256=None: str(cal_path))
    return str(cal_path)


# ==================================================================================== OFF = byte-identical

def test_field_absent_is_byte_identical_to_the_ordinary_chat_pipeline(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body())
    assert status == 200
    assert "clozn_guard_receipt" not in out
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "An ordinary ungated reply."}
    assert set(out.keys()) == {"id", "object", "created", "model", "choices", "clozn_run_id"}


def test_field_explicitly_empty_is_also_off(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body(clozn_guard={}))
    assert status == 200
    assert "clozn_guard_receipt" not in out


def test_server_setting_off_by_default(iso, monkeypatch):
    """No server-wide default ever saved -> parse_guard_spec must read its documented default (off), not
    silently opt in."""
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body())
    assert status == 200
    assert "clozn_guard_receipt" not in out


# ==================================================================================== malformed body -> 400

def test_malformed_guard_body_is_a_400(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body(clozn_guard={"concepts": "violence"}))
    assert status == 400
    assert out["error"]["param"] == "clozn_guard"


# ==================================================================================== streaming conflict -> refused

def test_guard_with_streaming_is_refused_not_silently_ignored(iso, dirc_fixtures, monkeypatch):
    engine = FakeGuardEngine(vocab={" violence": [TRIGGER_TOKEN_ID]})
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    status, out = _post("/v1/chat/completions",
                        _body(stream=True, clozn_guard={"concepts": ["violence"]}))
    assert status == 422
    assert out["error"]["code"] == "guard_streaming_unsupported"
    assert out["error"]["param"] == "clozn_guard"


# ==================================================================================== FAIL CLOSED

def test_fail_closed_when_no_valid_jlens_layer_at_all(iso, monkeypatch):
    """The engine reports NO fitted J-lens layers at all -- LAYER SELECTION has nothing to fall back to,
    so the whole request must be refused before any generation is attempted."""
    engine = FakeGuardEngine(vocab={" violence": [TRIGGER_TOKEN_ID]}, jlens_layers=())
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    status, out = _post("/v1/chat/completions",
                        _body(clozn_guard={"concepts": ["violence"]}, max_tokens=8))
    assert status == 422
    assert out["error"]["code"] == "guard_unavailable"
    assert "no valid j-lens layer" in out["error"]["message"].lower()
    assert len(engine.complete_calls) == 0


def test_fail_closed_when_calibrated_concept_cannot_be_resolved_to_dirc(iso, monkeypatch):
    """A calibration exists for 'violence' (so it's meant to be CORRECTABLE), but neither a lab unembed
    export NOR a working engine unembed_row is configured -- compute() degrades to 'unembed_unavailable',
    and the WHOLE request must be refused, never silently generate an unguarded reply."""
    engine = FakeGuardEngine(vocab={" violence": [TRIGGER_TOKEN_ID]})
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    _write_guard_calibration(iso, monkeypatch)   # calibrated -> compute() is required to succeed
    status, out = _post("/v1/chat/completions",
                        _body(clozn_guard={"concepts": ["violence"]}, max_tokens=8))
    assert status == 422
    assert out["error"]["code"] == "guard_unavailable"
    assert "violence" in out["error"]["message"]
    assert "unavailable" in out["error"]["message"]
    assert len(engine.complete_calls) == 0   # never generated anything under a promise it couldn't keep


# ==================================================================================== ON + fire (CALIBRATED)

def test_on_fire_uses_the_calibrated_trigger_set_and_builds_a_receipt(iso, dirc_fixtures, monkeypatch):
    engine = FakeGuardEngine(
        vocab={" violence": [TRIGGER_TOKEN_ID]},
        complete_pieces=[f"{TRIGGER_MARKER} content ", "clean chunk two"],
        intervene_pieces=["safe corrected content "],
    )
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    _write_guard_calibration(iso, monkeypatch, threshold=1.0)
    status, out = _post("/v1/chat/completions", _body(
        clozn_guard={"concepts": ["violence"], "chunk_tokens": 8},
        max_tokens=16,
    ))
    assert status == 200
    receipt = out["clozn_guard_receipt"]
    assert receipt["n_fires"] == 1
    assert receipt["cap_reached"] is False
    assert receipt["layer"] == LAYER
    assert receipt["layer_source"] == "calibration_default"
    assert receipt["topk"] == gg.CALIBRATION_TOPK_FLOOR   # raised even though the request never asked for it
    entry = receipt["concepts"]["violence"]
    assert entry["calibrated"] is True
    assert entry["layer"] == LAYER
    assert entry["threshold"] == 1.0
    assert entry["trigger_ids"] == [TRIGGER_TOKEN_ID]
    assert entry["catch"] == 1.0 and entry["fp"] == 0.0
    fire = receipt["fires"][0]
    assert fire["concept"] == "violence"
    assert fire["pre_activation"] == pytest.approx(9.0)
    assert fire["counter_strength"] == gg.DEFAULT_COUNTER_STRENGTH
    assert fire["calibrated"] is True
    assert receipt["caveat"] == gg.GUARD_CAVEAT
    assert TRIGGER_MARKER not in out["choices"][0]["message"]["content"]
    assert out["choices"][0]["message"]["content"] == "safe corrected content clean chunk two"
    assert len(engine.intervene_calls) == 1
    assert engine.intervene_calls[0]["layer"] == LAYER
    from clozn.behavior.steering.concept_dir import VALIDATED_MEDIAN_RESID_NORM
    assert engine.intervene_calls[0]["coef"] == pytest.approx(
        gg.DEFAULT_COUNTER_STRENGTH * VALIDATED_MEDIAN_RESID_NORM[LAYER])
    assert engine.intervene_calls[0]["coef"] < 0
    assert "clozn_run_id" in out
    stored = runlog.get_run(out["clozn_run_id"])
    assert stored["meta"]["clozn_guard"]["n_fires"] == 1
    # the poll actually used the raised topk floor, not the tiny default -- proves resolve_guard_topk's
    # effect reaches the real engine call, not just the receipt.
    assert all(c["topk"] == gg.CALIBRATION_TOPK_FLOOR for c in engine.jlens_calls)


def test_on_no_fire_never_calls_intervene(iso, dirc_fixtures, monkeypatch):
    engine = FakeGuardEngine(
        vocab={" violence": [TRIGGER_TOKEN_ID]},
        complete_pieces=["clean chunk one ", "clean chunk two"],
    )
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    _write_guard_calibration(iso, monkeypatch, threshold=1.0)
    status, out = _post("/v1/chat/completions", _body(
        clozn_guard={"concepts": ["violence"], "chunk_tokens": 8},
        max_tokens=16,
    ))
    assert status == 200
    receipt = out["clozn_guard_receipt"]
    assert receipt["n_fires"] == 0
    assert receipt["fires"] == []
    assert receipt["cap_reached"] is False
    assert out["choices"][0]["message"]["content"] == "clean chunk one clean chunk two"
    assert len(engine.intervene_calls) == 0


def test_cap_reached_is_labeled_honestly(iso, dirc_fixtures, monkeypatch):
    engine = FakeGuardEngine(
        vocab={" violence": [TRIGGER_TOKEN_ID]},
        complete_pieces=[f"{TRIGGER_MARKER} one", "corrected one",
                         f"{TRIGGER_MARKER} two", "plain rest"],
        intervene_pieces=["corrected one"],
    )
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    _write_guard_calibration(iso, monkeypatch, threshold=1.0)
    status, out = _post("/v1/chat/completions", _body(
        clozn_guard={"concepts": ["violence"], "chunk_tokens": 8, "max_fires": 1},
        max_tokens=24,
    ))
    assert status == 200
    receipt = out["clozn_guard_receipt"]
    assert receipt["n_fires"] == 1
    assert receipt["cap_reached"] is True
    assert receipt["cap_note"] == gg.GUARD_CAP_NOTE
    # the second trigger, past the cap, survives uncorrected in the final reply -- honest, not hidden
    assert TRIGGER_MARKER in out["choices"][0]["message"]["content"]
    assert len(engine.intervene_calls) == 1   # never re-steers past the cap


# ==================================================================================== UNCALIBRATED = annotate only

def test_uncalibrated_concept_never_fires_but_is_still_annotated(iso, dirc_fixtures, monkeypatch):
    """No calibration file at all for this model -- 'violence' falls back to the concept-WORD-token
    signal, and per the documented decision (UNCALIBRATED CONCEPTS DO NOT FIRE), it can NEVER trigger a
    correction, even though the exact same marker/activation would have fired if calibrated (see
    test_on_fire_uses_the_calibrated_trigger_set_and_builds_a_receipt above)."""
    engine = FakeGuardEngine(
        vocab={" violence": [TRIGGER_TOKEN_ID]},
        complete_pieces=[f"{TRIGGER_MARKER} content ", "clean chunk two"],
    )
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    # No _write_guard_calibration() call -- deliberately uncalibrated.
    status, out = _post("/v1/chat/completions", _body(
        clozn_guard={"concepts": ["violence"], "chunk_tokens": 8, "layer": LAYER},
        max_tokens=16,
    ))
    assert status == 200
    receipt = out["clozn_guard_receipt"]
    assert receipt["n_fires"] == 0
    assert receipt["fires"] == []
    assert receipt["layer_source"] == "explicit"
    entry = receipt["concepts"]["violence"]
    assert entry["calibrated"] is False
    assert entry["note"] == gg.UNCALIBRATED_NOTE
    assert entry["trigger_ids"] == [TRIGGER_TOKEN_ID]   # the word-token fallback still resolved
    assert entry["max_observed_activation"] == pytest.approx(9.0)   # observed, but never actionable
    assert "catch" not in entry
    # the marker survives UNCORRECTED in the final reply -- annotate-only means exactly that
    assert TRIGGER_MARKER in out["choices"][0]["message"]["content"]
    assert len(engine.intervene_calls) == 0


# ==================================================================================== layer selection end-to-end

def test_layer_16_invalid_on_this_engine_falls_through_to_a_valid_discovered_layer(iso, monkeypatch):
    """No explicit layer, no calibration -- the engine's fitted layers are [2, 14, 21, 25] (16 is NOT
    among them, exactly like the real 7B) -- LAYER SELECTION must discover 14 (nearest to the legacy 16),
    never hand a hardcoded 16 to the engine."""
    jdir = _write_jlens_fixture(iso, layer=14)
    udir = _write_unembed_fixture(iso)
    monkeypatch.setenv("CLOZN_JLENS_DIR", jdir)
    monkeypatch.setenv("CLOZN_DIRC_UNEMBED_DIR", udir)
    engine = FakeGuardEngine(
        vocab={" violence": [TRIGGER_TOKEN_ID]}, complete_pieces=["clean text one", "clean text two"],
        jlens_layers=(2, 14, 21, 25),
    )
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    status, out = _post("/v1/chat/completions", _body(
        clozn_guard={"concepts": ["violence"], "chunk_tokens": 8}, max_tokens=16,
    ))
    assert status == 200
    receipt = out["clozn_guard_receipt"]
    assert receipt["layer"] == 14
    assert receipt["layer_source"] == "discovered_valid"
    assert all(c["layer"] == 14 for c in engine.jlens_calls)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
