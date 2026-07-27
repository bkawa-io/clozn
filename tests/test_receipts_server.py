"""test_receipts_server -- POST /runs/<id>/receipt and /runs/<id>/receipts, the M2 endpoint wiring
(EXPLAIN_THIS_ANSWER_SPEC.md).

No model, no GPU: drives the REAL clozn_server do_POST handler (the object.__new__(H) no-socket trick used
by test_explain_server.py / test_profiles_server.py / test_timetravel_server.py) against an isolated
runlog store + memory_cards store + memory_mode settings, with a FAKE substrate standing in for the qwen
one. receipts.py itself (both-arms-greedy generation, the metric math, the redundancy guard) is exhaustively
unit-tested in test_receipts.py against fixture dicts; this file only proves the THIN endpoint wiring: the
routes match, a missing run is a clean 404, no substrate is a clean 503 (both endpoints regenerate, so --
unlike /runs/<id>/explain -- they need the live substrate), a malformed influence body is a clean 400, and a
real request's receipt/prove-all comes back over HTTP with the fields (and Python True -> JSON true) intact.
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs   # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
import clozn.memory.cards as memory_cards         # noqa: E402
import clozn.memory.mode as memory_mode          # noqa: E402
import clozn.runs.store as runlog                # noqa: E402


# --- a fake substrate: deterministic chat() keyed on which influence is currently ablated -------------------

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


# --- driving the real handler without a socket (mirrors test_explain_server / test_timetravel_server) ------

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


def _post_status(path, body_obj=None):
    """Like _post, but also returns the HTTP status (mirrors test_rewrite_route.py's _post) -- needed for
    the engine-not-reachable-vs-bad-request tests below, where the STATUS CODE is the thing under test,
    not just the message."""
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
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate the run/card/settings stores; SUB starts as a FakeSub (tests that want the 503 path
    override it to None explicitly)."""
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


# ============================================================================================ /receipt (one)

def test_receipt_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/receipt", {"influence": {"behavior_off": True}})
    assert out == {"error": "run not found"}


def test_receipt_needs_the_substrate_503(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipt", {"influence": {"behavior_off": True}})
    assert out == {"error": "receipt requires a ready product model worker"}


def test_receipt_rejects_a_missing_influence_spec_with_400(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipt", {})
    assert "error" in out
    out2 = _post(f"/runs/{rid}/receipt", {"influence": "not-a-dict"})
    assert "error" in out2


def test_receipt_rejects_an_unrecognized_influence_shape(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipt", {"influence": {"nonsense": True}})
    assert "error" in out          # _ablation_changes resolves to None -> receipts.receipt() -> None -> 500


# ============================================================================== engine-not-reachable vs bad request
# (engine-down pressure test finding #3): receipts.receipt() (and everything under it -- replay.py) is
# documented to NEVER raise, so a dead-engine URLError collapses into the SAME ambiguous None a bad
# influence spec produces (the test above). The route now probes the substrate's own engine directly on
# that ambiguous path (only there, never on the happy path) to tell the two apart.

class DownEngineSub:
    """A substrate whose OWN engine is unreachable: .chat() fails exactly like a live EngineSubstrate.chat
    would against a dead C++ worker, and .engine.health() fails too (what ctx._engine_reachable() probes)."""
    name = "engine"

    def __init__(self):
        self.memory = FakeMem()
        self.steer = FakeSteer()
        self.base = "http://127.0.0.1:8080"
        self.engine = self                 # simplest double: this object IS its own "engine client"

    def health(self):
        raise OSError("connection refused")

    def chat(self, messages, max_new=256, sample=True):
        raise OSError("connection refused")


def test_receipt_reports_engine_not_reachable_distinctly_from_a_bad_spec(iso, monkeypatch):
    sub = DownEngineSub()
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "ENGINE", sub)     # ctx._engine_unreachable_message() reads ENGINE.base
    rid = _seed_run()
    status, out = _post_status(f"/runs/{rid}/receipt", {"influence": {"dial": "warm"}})
    assert status == 502
    assert out == {"error": "engine not reachable at http://127.0.0.1:8080 -- is it running?"}


def test_receipt_bad_spec_keeps_the_generic_500_when_the_engine_is_fine(iso, monkeypatch):
    """Control: the SAME ambiguous-None path, but with a substrate that has no .engine at all (like this
    file's ordinary FakeSub) -- ctx._engine_reachable() can't blame connectivity, so the original generic
    message survives unchanged, at its original 500."""
    rid = _seed_run()
    status, out = _post_status(f"/runs/{rid}/receipt", {"influence": {"nonsense": True}})
    assert status == 500
    assert out == {"error": "receipt failed (bad influence spec, or the replay could not be generated)"}


def test_receipt_happy_path_dial_ablation_over_http(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipt", {"influence": {"dial": "warm"}})
    assert "error" not in out
    assert out["causal_verified"] is True                 # JSON true, round-tripped from Python True
    assert out["has_effect"] is True
    assert out["baseline_reply"] == "A much longer rambling reply. Warmly!"
    assert out["ablated_reply"] == "A much longer rambling reply."
    assert out["changes_applied"] == {"behavior_overrides": {"warm": 0.0}}
    # the stored sampled reply never shows up as either arm, and the receipt says so
    stored = runlog.get_run(rid)["response"]
    assert stored not in (out["baseline_reply"], out["ablated_reply"])
    assert "sampled" in out["note"].lower() and "baseline" in out["note"].lower()
    assert "cost_note" in out


def test_receipt_memory_off_ablation_over_http(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipt", {"influence": {"memory_off": True}})
    assert out["causal_verified"] is True
    assert out["ablated_reply"] == "Generic reply, memory off."


# =========================================================================================== /receipts (all)

def test_receipts_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/receipts", {})
    assert out == {"error": "run not found"}


def test_receipts_needs_the_substrate_503(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipts", {})
    assert out == {"error": "receipts require a ready product model worker"}


def test_receipts_prove_all_happy_path_over_http_finds_the_redundant_pair(iso, monkeypatch):
    memory_mode.set_mode("prompt")
    card_a, card_b = "mem_a", "mem_b"
    monkeypatch.setattr(cs, "SUB", FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}),
                                          concise_card_ids=[card_a, card_b]))
    rid = runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="QwenSubstrate",
                        messages=[{"role": "user", "content": "how's it going"}],
                        response="SAMPLED, never a baseline",
                        memory={"cards_applied": ["Be concise.", "Keep it short."],
                               "applied_ids": [card_a, card_b], "mode": "prompt", "gate": 0.8},
                        behavior={"active_dials": {"warm": 0.5}})
    out = _post(f"/runs/{rid}/receipts", {})
    assert "error" not in out
    assert out["run_id"] == rid
    assert len(out["receipts"]) == 3                       # card_a, card_b, warm
    assert all(r["causal_verified"] is True for r in out["receipts"])
    assert len(out["redundant_pairs"]) == 1
    assert set(out["redundant_pairs"][0]["redundant"]) == {f"card:{card_a}", f"card:{card_b}"}
    assert out["redundant_pairs"][0]["note"] == "together they drive this; individually neither is load-bearing"
    assert "approximation_note" in out and "perf_note" in out   # the documented-approximation SAY-SO


def test_receipts_omits_coalitions_key_by_default(iso, monkeypatch):
    """The opt-in flag (docs/PRODUCT_ROADMAP.md §8 tail) must never change the default response shape."""
    memory_mode.set_mode("prompt")
    card_a, card_b = "mem_a", "mem_b"
    monkeypatch.setattr(cs, "SUB", FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}),
                                          concise_card_ids=[card_a, card_b]))
    rid = runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="QwenSubstrate",
                        messages=[{"role": "user", "content": "how's it going"}],
                        response="SAMPLED, never a baseline",
                        memory={"cards_applied": ["Be concise.", "Keep it short."],
                               "applied_ids": [card_a, card_b], "mode": "prompt", "gate": 0.8},
                        behavior={"active_dials": {"warm": 0.5}})
    out = _post(f"/runs/{rid}/receipts", {})
    assert "coalitions" not in out


def test_receipts_coalitions_opt_in_returns_a_report_over_http(iso, monkeypatch):
    memory_mode.set_mode("prompt")
    card_a, card_b = "mem_a", "mem_b"
    monkeypatch.setattr(cs, "SUB", FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}),
                                          concise_card_ids=[card_a, card_b]))
    rid = runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="QwenSubstrate",
                        messages=[{"role": "user", "content": "how's it going"}],
                        response="SAMPLED, never a baseline",
                        memory={"cards_applied": ["Be concise.", "Keep it short."],
                               "applied_ids": [card_a, card_b], "mode": "prompt", "gate": 0.8},
                        behavior={"active_dials": {"warm": 0.5}})
    out = _post(f"/runs/{rid}/receipts", {"coalitions": True})
    assert "error" not in out
    coalitions = out["coalitions"]
    assert coalitions["available"] is True
    assert coalitions["n_influences"] == 3
    assert set(coalitions["keys"]) == {f"card:{card_a}", f"card:{card_b}", "dial:warm"}
    assert coalitions["shapley"]["class"] == "exact"          # N=3 <= EXACT_SHAPLEY_MAX_N
    assert "interaction_gap" in coalitions and "note" in coalitions["interaction_gap"]
    assert coalitions["batch_report"]["attempted"] is False    # FakeSub has no branch_coalitions


def test_receipts_rejects_bad_coalitions_batch_value(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipts", {"coalitions_batch": "bogus"})
    assert out == {"error": "coalitions_batch must be one of auto|off|approximate"}


def test_receipts_no_fired_influences_is_a_clean_empty_200(iso):
    rid = _seed_run()                                       # no memory cards, dials ARE active though
    # strip the dial too, for a maximally bare manifest
    rid2 = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey")
    out = _post(f"/runs/{rid2}/receipts", {})
    assert out["receipts"] == []
    assert out["run_id"] == rid2


# ============================================================================================================
# ================================================ S3: mode= (regen | forced | both) endpoint wiring =========
# ============================================================================================================
# regen (default/omitted) is asserted byte-identical above (those tests never pass `mode` at all -- the
# whole pre-S3 suite is itself the regression test). Here: the new `mode` plumbing -- a bad string 400s, a
# forced-only request needs no qwen substrate (no .chat gate), and an engine-shaped fake that ALSO exposes
# .score_tokens can drive forced/both end to end over HTTP.

class FakeEngineSub:
    """A substrate double exposing BOTH .chat() (regen) and .score_tokens() (forced/rederive) -- what a
    real EngineSubstrate looks like from these endpoints' point of view."""
    name = "engine"

    def __init__(self, reply="hi", tokens=None):
        self.memory = FakeMem()
        self.steer = FakeSteer()
        self.reply = reply
        self._tokens = tokens if tokens is not None else [{"id": 1, "piece": "hi", "logprob": -0.1}]
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True):
        self.calls += 1
        return self.reply

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        return self._tokens


def test_receipt_rejects_an_unrecognized_mode_with_400(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipt", {"influence": {"memory_off": True}, "mode": "bogus"})
    assert "error" in out


def test_receipts_rejects_an_unrecognized_mode_with_400(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipts", {"mode": "bogus"})
    assert "error" in out


def test_receipt_mode_forced_does_not_need_the_qwen_substrate_gate(iso, monkeypatch):
    """forced mode never regenerates -- unlike regen/both, it must not 503 just because SUB is None."""
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipt", {"influence": {"memory_off": True}, "mode": "forced"})
    assert "error" not in out
    assert out["causal_verified"] is False                 # honestly degrades: no substrate to score with
    assert "score_tokens" in out["note"]


def test_receipts_mode_forced_does_not_need_the_qwen_substrate_gate(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/receipts", {"mode": "forced"})
    assert "error" not in out
    assert out["mode"] == "forced"


def test_receipt_mode_forced_happy_path_over_http(iso, monkeypatch):
    fake = FakeEngineSub(tokens=[{"id": 1, "piece": "hi", "logprob": -0.1},
                                {"id": 2, "piece": " there", "logprob": -0.2}])
    monkeypatch.setattr(cs, "SUB", fake)
    rid = runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="engine",
                        messages=[{"role": "user", "content": "hi"}], response="hi there",
                        behavior={"active_dials": {"warm": 0.5}}, trace={"token_ids": [1, 2]})
    out = _post(f"/runs/{rid}/receipt", {"influence": {"dial": "warm"}, "mode": "forced"})
    assert "error" not in out
    assert out["mode"] == "forced"
    assert out["causal_verified"] is True
    assert out["deltas"] == [0.0, 0.0]                      # WITH == WITHOUT here (fake ignores steer args)
    assert fake.calls == 0                                  # forced mode never called .chat()


def test_receipt_mode_both_over_http_includes_forced_and_regen_fields(iso, monkeypatch):
    fake = FakeEngineSub(reply="Warmly!", tokens=[{"id": 1, "piece": "Warmly", "logprob": -0.1},
                                                  {"id": 2, "piece": "!", "logprob": -0.2}])
    monkeypatch.setattr(cs, "SUB", fake)
    rid = runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="engine",
                        messages=[{"role": "user", "content": "hi"}], response="Warmly!",
                        behavior={"active_dials": {"warm": 0.5}}, trace={"token_ids": [1, 2]})
    out = _post(f"/runs/{rid}/receipt", {"influence": {"dial": "warm"}, "mode": "both"})
    assert "error" not in out
    assert out["mode"] == "both"
    assert "baseline_reply" in out                          # regen fields present at the top level
    assert "forced" in out and out["forced"]["mode"] == "forced"
    assert "silent_influence" in out


# ============================================================================================================
# ============================================================= POST /runs/<id>/swap_receipt ==================
# ============================================================================================================
# The THIN endpoint wiring only -- clozn.receipts.swap_receipt's own math/degrade-paths are exhaustively
# unit-tested model-free in test_swap_receipt.py. Here: the route matches, a missing run is a clean 404, no
# engine/jlens substrate is a clean 503, a missing to_concept is a clean 400, and a real request's swap
# receipt comes back over HTTP with `causal_verified` intact. dir(c) needs a REAL (tiny, on-disk, fixture)
# J-lens + unembed export -- mirrors test_swap_receipt.py / test_concept_dir.py's own orthogonal-J /
# orthonormal-W_U construction -- pointed at via CLOZN_JLENS_DIR / CLOZN_DIRC_UNEMBED_DIR (the route itself
# builds a bare concept_dir.ConceptSteer(engine, layer=...) with no fixture wiring of its own, exactly as
# swap_receipt(run, from_hint, to_concept, ctx.SUB) is documented to be called -- see swap_receipt.py).

def _orthogonal(seed, n):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def _write_jlens_fixture(tmp_path, *, d_model=32, layer=21, seed=1):
    jdir = tmp_path / "jlens"
    jdir.mkdir()
    manifest = {"model": "fixture", "d_model": d_model, "vocab": d_model, "layers": [layer],
               "engine_default_tap_layer": layer}
    (jdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    J = _orthogonal(seed, d_model).astype(np.float32)
    J.astype("<f2").tofile(str(jdir / f"J_layer{layer}.f16"))
    return str(jdir)


def _write_unembed_fixture(tmp_path, *, d_model=32, vocab=32, seed=2):
    udir = tmp_path / "unembed"
    udir.mkdir()
    q = _orthogonal(seed, d_model)[:vocab].astype(np.float32)
    np.save(str(udir / "norm_weight.npy"), np.ones(d_model, dtype=np.float32))
    np.save(str(udir / "lm_head_weight.npy"), q)
    (udir / "unembed_meta.json").write_text(json.dumps({"rms_norm_eps": 1e-6}), encoding="utf-8")
    return str(udir)


class FakeSwapEngine:
    """apply_template/complete/intervene/score -- mirrors test_swap_receipt.py's FakeEngineClient, pared
    down to what a THIN route test needs (one swap arm, one null arm, both canned)."""

    def __init__(self):
        self.vocab = {}
        self._next_id = 0      # fixture vocab is d_model=32 rows -- stay well inside range
        self.intervene_calls = 0

    def apply_template(self, messages, add_assistant=True):
        return "PROMPT::" + " | ".join(str(m.get("content", "")) for m in messages)

    def complete(self, prompt, max_tokens=64):
        return {"choices": [{"text": "the sky is calm and blue today"}]}

    def intervene(self, prompt, vector=None, coef=None, layer=None, max_tokens=64):
        self.intervene_calls += 1
        if self.intervene_calls == 1:                 # the real swap arm
            return {"choices": [{"text": "a vast ocean wave of deep ocean water"}]}
        return {"choices": [{"text": "xk garble zzzz repeated repeated repeated"}]}   # the null arm

    def score(self, prompt=None, continuation_ids=None, continuation=None, topk=0, steer=None, steer_vec=None):
        if continuation is not None:                   # token-resolution path (ConceptSteer.resolve_token_id)
            word = continuation.strip()
            tid = self.vocab.get(word)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self.vocab[word] = tid
            return {"tokens": [{"id": tid, "piece": word}]}
        tid = continuation_ids[0] if continuation_ids else 0
        lp = -0.3 if steer_vec is not None else -2.0
        return {"tokens": [{"id": tid, "piece": "x", "logprob": lp}]}


class FakeSwapSub:
    """The minimal duck-typed substrate swap_receipt needs: .engine + .jlens -- mirrors
    clozn.server.app.EngineSubstrate's own shape."""

    def __init__(self, engine):
        self.engine = engine

    def jlens(self, text, layer=None, topk=5):
        return {"available": False, "reason": "test fake has no jlens sidecar"}


class FakeSwapSubNoJlens:
    def __init__(self, engine):
        self.engine = engine


class FakeSwapSubNoEngine:
    pass


@pytest.fixture
def jlens_env(tmp_path, monkeypatch):
    """CLOZN_JLENS_DIR / CLOZN_DIRC_UNEMBED_DIR -> a tiny fixture export -- the DEFAULT concept_dir.
    ConceptDirSource() the route builds (no fixture wiring of its own) picks these up via env vars."""
    jdir = _write_jlens_fixture(tmp_path)
    udir = _write_unembed_fixture(tmp_path)
    monkeypatch.setenv("CLOZN_JLENS_DIR", jdir)
    monkeypatch.setenv("CLOZN_DIRC_UNEMBED_DIR", udir)
    return tmp_path


def test_swap_receipt_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/swap_receipt", {"to_concept": "ocean"})
    assert out == {"error": "run not found"}


def test_swap_receipt_needs_the_engine_substrate_503_when_sub_is_none(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/swap_receipt", {"to_concept": "ocean"})
    assert out == {"error": "swap_receipt requires the product worker with J-lens enabled"}


def test_swap_receipt_needs_the_engine_substrate_503_when_sub_has_no_engine(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSwapSubNoEngine())
    rid = _seed_run()
    out = _post(f"/runs/{rid}/swap_receipt", {"to_concept": "ocean"})
    assert out == {"error": "swap_receipt requires the product worker with J-lens enabled"}


def test_swap_receipt_needs_the_engine_substrate_503_when_sub_has_no_jlens(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSwapSubNoJlens(FakeSwapEngine()))
    rid = _seed_run()
    out = _post(f"/runs/{rid}/swap_receipt", {"to_concept": "ocean"})
    assert out == {"error": "swap_receipt requires the product worker with J-lens enabled"}


def test_swap_receipt_rejects_a_missing_to_concept_with_400(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSwapSub(FakeSwapEngine()))
    rid = _seed_run()
    out = _post(f"/runs/{rid}/swap_receipt", {})
    assert out == {"error": "need a 'to_concept' to swap in"}


def test_swap_receipt_happy_path_over_http(iso, monkeypatch, jlens_env):
    monkeypatch.setattr(cs, "SUB", FakeSwapSub(FakeSwapEngine()))
    rid = _seed_run()
    out = _post(f"/runs/{rid}/swap_receipt", {"to_concept": "ocean", "from_hint": "Paris"})
    assert "error" not in out
    assert out["mode"] == "swap_receipt"
    assert out["run_id"] == rid
    assert out["causal_verified"] is True
    assert out["swapped_to"]["concept"] == "ocean"
    assert out["disposed"]["hint"] == "Paris"
    assert out["baseline_reply"] == "the sky is calm and blue today"
    assert out["swapped_reply"] == "a vast ocean wave of deep ocean water"
    assert "lexicon_hits" in out and "logprob_shift" in out


# ============================================================================== fix #4: total failure is non-2xx
# (engine-down pressure test finding #4): swap_receipt() never raises -- a total failure (no engine,
# template render, generation, ...) degrades to causal_verified:false + blocked/note explaining why, but
# used to still ship as plain HTTP 200, so a caller checking response.ok alone would read a complete
# failure as a success. Body stays intact either way (swap_receipt's own blocked/note already self-
# describe the failure) -- only the status code changes.

class DownSwapEngine:
    """A fully unreachable engine: apply_template (swap_receipt's first engine call) and health() (what
    ctx._engine_reachable() probes) both fail with a connection-style error. complete/intervene exist
    (swap_receipt's own hasattr gate needs them present) but are never reached -- apply_template fails first."""

    def __init__(self):
        self.base = "http://127.0.0.1:8080"

    def health(self):
        raise OSError("connection refused")

    def apply_template(self, messages, add_assistant=True):
        raise OSError("connection refused")

    def complete(self, prompt, max_tokens=64):
        raise OSError("connection refused")

    def intervene(self, prompt, vector=None, coef=None, layer=None, max_tokens=64):
        raise OSError("connection refused")


def test_swap_receipt_total_failure_is_a_clean_502_when_the_engine_is_down(iso, monkeypatch):
    down_engine = DownSwapEngine()
    monkeypatch.setattr(cs, "SUB", FakeSwapSub(down_engine))
    monkeypatch.setattr(cs, "ENGINE", down_engine)     # ctx._engine_unreachable_message() reads ENGINE.base
    rid = _seed_run()
    status, out = _post_status(f"/runs/{rid}/swap_receipt", {"to_concept": "ocean"})
    assert status == 502
    assert out["causal_verified"] is False
    assert out["blocked"] == "template_render"
    assert "connection refused" in out["note"]


def test_swap_receipt_total_failure_is_a_clean_500_when_not_engine_related(iso, monkeypatch):
    """A different total failure -- a run with no messages to reconstruct a prompt from -- has nothing to
    do with connectivity: ctx._engine_reachable() can't blame FakeSwapEngine (no .health method at all --
    see app._engine_reachable's docstring), so it lands on the plain 500 sibling routes use."""
    monkeypatch.setattr(cs, "SUB", FakeSwapSub(FakeSwapEngine()))
    rid = runlog.record(source="cli", messages=[], response="")
    status, out = _post_status(f"/runs/{rid}/swap_receipt", {"to_concept": "ocean"})
    assert status == 500
    assert out["causal_verified"] is False
    assert out["note"] == "run has no messages to reconstruct a prompt from"
