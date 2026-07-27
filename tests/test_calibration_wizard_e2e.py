"""tests/test_calibration_wizard_e2e.py -- Phase 3.6 (docs/PRODUCT_ROADMAP.md §6 item 6): proves the
guided calibration wizard's output actually drives the live ask/abstain metadata, not just that each piece
works in isolation. Every other test in this repo either (a) exercises `clozn eval --wizard` with
`eval_store.save_profile` MOCKED (tests/test_eval_cmd.py), or (b) exercises the live route with a
hand-authored `SAVED` dict written via `eval_store.save` directly (tests/test_ask_band_server.py) --
neither ever runs the wizard's own `policy.recommend` fit AND the live route's own `eval_store.load_profile`
read against the SAME file on disk. This test does exactly that, with only the one piece that inherently
needs a live gateway (the probe run itself, `bench.bench`) faked; everything else -- the wizard's prompt
flow, `policy.recommend`'s threshold fit, `eval_store.save_profile`'s write, and the live route's
`eval_store.load_profile` read + `eval_policy.classify_run` classification -- is the real production code.

Model-free: no engine, no socket (`object.__new__(H)`, mirrors test_ask_band_server.py's `_dispatch`).
"""
from __future__ import annotations

import builtins
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from clozn.cli.main import build_parser              # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
from clozn.cli.commands.eval import cmd_eval          # noqa: E402
from clozn.eval import bench, policy, store as eval_store  # noqa: E402
from clozn.server import app as cs                    # noqa: E402


import clozn.runs.store as runlog                      # noqa: E402


MODEL = "clozn-e2e-model"
TASK_ANSWER = "  Arith   Word Problems "
NORMALIZED_TASK = "arith word problems"

# A synthetic calibration set with real separation between confident-correct, mixed-mid, and
# confident-wrong tokens, so policy.recommend(target_error=0.1) fits a real two-threshold band
# (verified by the `test_fixture_pairs_fit_a_real_two_band_policy` guard below) instead of a degenerate
# answer-everything/answer-nothing edge case.
PAIRS = [
    (0.98, True), (0.95, True), (0.93, True), (0.90, True), (0.88, True),
    (0.85, True), (0.83, True), (0.80, True), (0.78, True), (0.75, True),
    (0.65, True), (0.62, True), (0.60, False), (0.58, True), (0.55, False),
    (0.45, False), (0.40, True), (0.35, False), (0.20, False), (0.10, False),
]


def _fake_out():
    return {"n": len(PAIRS), "unmatched": 0, "model": MODEL, "pairs": list(PAIRS), "rows": [],
            "report": {"available": True}}


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
    _, _, payload = _dispatch(path, body_obj).partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


class FakeSteer:
    def active(self):
        return {}


class FakeMem:
    def __init__(self):
        self.memory_strength = 1.0
        self.rules = []
        self.prefix = None


class TraceSub:
    """A qwen-shaped substrate whose chat() fills trace_out with a real per-token trace, mirroring
    test_ask_band_server.py's TraceSub."""
    name = "qwen"

    def __init__(self, steps, reply="The answer."):
        self.memory = FakeMem()
        self._mem = self.memory
        self.steer = FakeSteer()
        self._steps = steps
        self._reply = reply
        self._run_meta = {"model_id": MODEL, "sampler_mode": "greedy", "sampling": "greedy", "temperature": 0.0}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        if trace_out is not None:
            trace_out.extend([dict(s) for s in self._steps])
        return self._reply

    def last_finish_reason(self):
        return "stop"

    def run_meta(self):
        return dict(self._run_meta)


def _body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "what's 12 times 12?"}], **extra}


def test_fixture_pairs_fit_a_real_two_band_policy():
    """Guard on the fixture itself: policy.recommend must fit a genuine achievable answer/ask split
    (not an edge case where everything is answered or nothing is), so the assertions below actually
    exercise the 'ask' and 'abstain' bands, not a degenerate always/never verdict."""
    expected = policy.recommend(PAIRS, target_error=0.1)
    assert expected["achievable"] is True
    assert 0.0 < expected["ask_at"] < expected["answer_at"] < 1.0
    assert expected["summary"]["n_ask"] > 0


def test_wizard_fit_and_save_then_a_live_reply_carries_the_same_thresholds(tmp_path, monkeypatch):
    # --- isolate every store this touches -- never the real ~/.clozn ---
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "eval_report.json"))

    # --- the ONE necessarily-faked piece: the live probe run itself needs a real gateway. Everything
    # downstream of `pairs` (the fit, the save, the live read) is real production code. ---
    called = {}
    monkeypatch.setattr(bench, "bench", lambda url, which, score:
                        called.update(url=url, which=which, score=score) or _fake_out())
    monkeypatch.setattr(bench, "_print", lambda *_args: None)

    # --- drive the real wizard prompt flow: task, probe set (default), score (default), target_error,
    # save? -- exactly clozn.cli.commands.eval._wizard's own prompt order. ---
    answers = iter([TASK_ANSWER, "", "", "0.1", "yes"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(answers))

    ns = build_parser().parse_args(["eval", "--wizard"])
    assert cmd_eval(ns) == 0
    assert called["which"] == "arith" and called["score"] == "min"          # kept the CLI defaults

    # The wizard's own fit, read back from the real profile it just wrote to disk.
    saved_profile = eval_store.load_profile(MODEL, NORMALIZED_TASK, str(tmp_path / "eval_report.json"))
    assert saved_profile is not None and saved_profile["model"] == MODEL and saved_profile["task"] == NORMALIZED_TASK

    # An independent oracle call to the exact same policy fit, so the live assertions below are never
    # circular against the wizard's own saved numbers alone.
    expected = policy.recommend(PAIRS, target_error=0.1)
    answer_at, ask_at = expected["answer_at"], expected["ask_at"]
    assert saved_profile["policy"]["answer_at"] == answer_at
    assert saved_profile["policy"]["ask_at"] == ask_at

    # --- now a genuinely live-shaped POST /v1/chat/completions, reading that SAME saved profile through
    # eval_store.load_profile (never mocked) + eval_policy.classify_run + generation_gateway.policy_signal. ---
    ask_score = round((answer_at + ask_at) / 2, 4)             # strictly inside [ask_at, answer_at)
    monkeypatch.setattr(cs, "SUB", TraceSub(steps=[{"piece": "The", "conf": 0.99},
                                                   {"piece": " answer", "conf": ask_score}]))
    out = _post("/v1/chat/completions", _body(clozn_task="  arith   word   problems  "))
    assert out["clozn_policy"]["band"] == "ask"
    assert out["clozn_policy"]["answer_at"] == round(answer_at, 4)
    assert out["clozn_policy"]["ask_at"] == round(ask_at, 4)
    assert out["clozn_policy"]["calibration_model"] == MODEL
    assert out["clozn_policy"]["calibration_task"] == NORMALIZED_TASK

    abstain_score = round(max(0.0, ask_at - 0.05), 4)
    monkeypatch.setattr(cs, "SUB", TraceSub(steps=[{"piece": "The", "conf": 0.99},
                                                   {"piece": " answer", "conf": abstain_score}]))
    out_abstain = _post("/v1/chat/completions", _body(clozn_task="arith word problems"))
    assert out_abstain["clozn_policy"]["band"] == "abstain"
    assert "likely wrong" in out_abstain["clozn_policy"]["note"]

    # A confidently-correct-shaped reply gets no metadata at all -- the wizard's fit never fabricates a
    # verdict outside its own bands.
    monkeypatch.setattr(cs, "SUB", TraceSub(steps=[{"piece": "The", "conf": 0.99},
                                                   {"piece": " answer", "conf": min(0.999, answer_at + 0.01)}]))
    out_answer = _post("/v1/chat/completions", _body(clozn_task="arith word problems"))
    assert "clozn_policy" not in out_answer


def test_wizard_prints_the_token_probability_and_hard_tail_labels(tmp_path, monkeypatch, capsys):
    """The CRITICAL honesty requirement (docs/RESEARCH_ROADMAP.md Killed: the deployed signal is
    bit-identical to black-box logprobs): the wizard's own printed scope must label the signal as
    token-probability based and print the hard-tail band limitation -- never imply a white-box advantage."""
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "eval_report.json"))
    monkeypatch.setattr(bench, "bench", lambda url, which, score: _fake_out())
    monkeypatch.setattr(bench, "_print", lambda *_args: None)
    answers = iter([TASK_ANSWER, "", "", "0.1", "no"])
    monkeypatch.setattr(builtins, "input", lambda _prompt: next(answers))

    ns = build_parser().parse_args(["eval", "--wizard"])
    assert cmd_eval(ns) == 0
    text = capsys.readouterr().out
    assert "token-probability based" in text
    assert "NOT an internal/white-box signal" in text
    assert "hard-tail" in text
