"""test_quant_check -- clozn/cli/commands/quant_check.py (Tier 1 wrap):
the `clozn quant-check` CLI verb around the already-validated clozn/receipts/quant_receipts.py.

Model-free / GPU-free throughout, mirroring tests/test_quant_receipts.py's own discipline: no real
engine, no C++ server, no GPU. `_EngineScoreSub` and the fresh-generation helpers only ever call
`sub.engine.apply_template/.score/.complete`, so a FakeEngine stands in perfectly for a real
`EngineClient`; `build_receipts` is exercised against a FakeScoreSub (mirrors test_quant_receipts.py's
own fake exactly); `aggregate_receipts`/`format_ladder` are exercised on receipts built with the real,
already-tested `quant_receipts.diff_quant_scores` over fixture score arrays -- no new fixture shape
invented. `add_subparser`'s argparse wiring is exercised on a throwaway parser, never touching
clozn/cli/main.py.

DEFERRED (not covered here, by design): `cmd_quant_check` itself -- the real two-engine boot
(spawn_engine + a real EngineClient over the wire) -- needs a live engine/GPU and is out of scope for
this task (see quant_check.py's module docstring). Everything upstream of "actually boot a process and
open a socket" is covered.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.cli.commands.quant_check as qc  # noqa: E402
import clozn.receipts.quant_receipts as qr  # noqa: E402


# ==================================================================================== fixtures / fakes

def _tok(id_, piece, logprob, topk=None):
    t = {"id": id_, "piece": piece, "logprob": logprob}
    if topk is not None:
        t["topk"] = topk
    return t


class FakeScoreSub:
    """Mirrors tests/test_quant_receipts.py's own fake exactly -- exposes only `.score_tokens`."""

    def __init__(self, tokens=None, raises=False):
        self.calls = []
        self._tokens = tokens if tokens is not None else []
        self._raises = raises

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        if self._raises:
            raise RuntimeError("boom")
        self.calls.append({"messages": messages, "continuation_ids": continuation_ids, "topk": topk,
                           "block": block, "steer_strengths": steer_strengths})
        return self._tokens


class FakeEngine:
    """Stands in for engine/client/clozn_engine.py's EngineClient: exposes apply_template/score/complete
    with the SAME call shapes _EngineScoreSub and the fresh-generation helpers use. Records every call so
    tests can assert on what was actually sent (prompt string, topk, continuation vs continuation_ids)."""

    def __init__(self, *, template="TEMPLATED", score_tokens=None, completion_text="Paris"):
        self.template = template
        self._score_tokens = score_tokens if score_tokens is not None else []
        self._completion_text = completion_text
        self.apply_template_calls = []
        self.score_calls = []
        self.complete_calls = []

    def apply_template(self, messages):
        self.apply_template_calls.append(list(messages))
        return self.template

    def score(self, prompt=None, **kw):
        self.score_calls.append({"prompt": prompt, **kw})
        return {"tokens": self._score_tokens}

    def complete(self, prompt, **params):
        self.complete_calls.append({"prompt": prompt, **params})
        return {"choices": [{"text": self._completion_text}]}


# ==================================================================================== _EngineScoreSub

def test_engine_score_sub_score_tokens_uses_apply_template_and_score():
    eng = FakeEngine(template="RENDERED", score_tokens=[_tok(1, "a", -0.1)])
    sub = qc._EngineScoreSub(eng)
    messages = [{"role": "user", "content": "hi"}]
    out = sub.score_tokens(messages, [1], topk=5)
    assert out == [_tok(1, "a", -0.1)]
    assert eng.apply_template_calls[-1] == messages
    assert eng.score_calls[-1]["prompt"] == "RENDERED"
    assert eng.score_calls[-1]["topk"] == 5
    assert eng.score_calls[-1]["continuation_ids"] == [1]
    assert "continuation" not in eng.score_calls[-1]


def test_engine_score_sub_prefers_continuation_ids_over_continuation_text():
    eng = FakeEngine()
    sub = qc._EngineScoreSub(eng)
    sub.score_tokens([{"role": "user", "content": "hi"}], [7, 8], continuation="ignored text", topk=0)
    assert eng.score_calls[-1]["continuation_ids"] == [7, 8]
    assert "continuation" not in eng.score_calls[-1]


def test_engine_score_sub_falls_back_to_continuation_text_when_no_ids():
    eng = FakeEngine()
    sub = qc._EngineScoreSub(eng)
    sub.score_tokens([{"role": "user", "content": "hi"}], None, continuation="the answer", topk=2)
    assert eng.score_calls[-1]["continuation"] == "the answer"
    assert "continuation_ids" not in eng.score_calls[-1]


def test_engine_score_sub_passes_steer_vec_through():
    eng = FakeEngine()
    sub = qc._EngineScoreSub(eng)
    sub.score_tokens([{"role": "user", "content": "hi"}], [1], steer_vec=[0.1, 0.2], topk=0)
    assert eng.score_calls[-1]["steer_vec"] == [0.1, 0.2]


def test_inject_block_appends_to_existing_system_message():
    messages = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
    out = qc._EngineScoreSub._inject_block(messages, "remember: X")
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "be nice\n\nremember: X"
    assert out[1] == {"role": "user", "content": "hi"}
    # original untouched
    assert messages[0]["content"] == "be nice"


def test_inject_block_prepends_new_system_message_when_none_exists():
    messages = [{"role": "user", "content": "hi"}]
    out = qc._EngineScoreSub._inject_block(messages, "remember: X")
    assert out[0] == {"role": "system", "content": "remember: X"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_inject_block_noop_on_falsy_block():
    messages = [{"role": "user", "content": "hi"}]
    out = qc._EngineScoreSub._inject_block(messages, None)
    assert out == messages
    assert out is not messages   # still a copy


def test_engine_score_sub_folds_block_into_prompt():
    eng = FakeEngine()
    sub = qc._EngineScoreSub(eng)
    sub.score_tokens([{"role": "user", "content": "hi"}], [1], block="a memory card", topk=0)
    sent = eng.apply_template_calls[-1]
    assert sent[0] == {"role": "system", "content": "a memory card"}


# ==================================================================================== _completion_text

def test_completion_text_from_plain_completions_shape():
    assert qc._completion_text({"choices": [{"text": "hello"}]}) == "hello"


def test_completion_text_from_chat_message_shape():
    assert qc._completion_text({"choices": [{"message": {"content": "hi there"}}]}) == "hi there"


def test_completion_text_handles_garbage_without_raising():
    assert qc._completion_text({}) == ""
    assert qc._completion_text({"choices": []}) == ""
    assert qc._completion_text({"choices": [{}]}) == ""
    assert qc._completion_text({"choices": "not a list"}) == ""


# ==================================================================================== fresh-run generation

def test_generate_fresh_run_happy_path():
    eng = FakeEngine(template="PROMPT", completion_text="Paris is the capital.",
                     score_tokens=[_tok(11, "Paris", -0.05), _tok(12, " is", -0.02)])
    sub = qc._EngineScoreSub(eng)
    run = qc.generate_fresh_run(sub, "factual_qa", "What is the capital of France?", max_tokens=50, topk=8)
    assert run is not None
    assert run["category"] == "factual_qa"
    assert run["response"] == "Paris is the capital."
    assert run["trace"]["token_ids"] == [11, 12]
    assert run["messages"] == [{"role": "user", "content": "What is the capital of France?"}]
    assert run["behavior"] == {"active_dials": {}}
    # generation used greedy (temperature 0) and the fixing /score reused the SAME rendered prompt
    assert eng.complete_calls[-1]["temperature"] == 0.0
    assert eng.score_calls[-1]["continuation"] == "Paris is the capital."


def test_generate_fresh_run_returns_none_on_empty_completion():
    eng = FakeEngine(completion_text="   ")
    sub = qc._EngineScoreSub(eng)
    assert qc.generate_fresh_run(sub, "c", "p") is None


def test_generate_fresh_run_returns_none_when_a_token_id_is_missing():
    eng = FakeEngine(completion_text="ok", score_tokens=[_tok(1, "o", -0.1), {"piece": "k", "logprob": -0.1}])
    sub = qc._EngineScoreSub(eng)
    assert qc.generate_fresh_run(sub, "c", "p") is None


def test_generate_fresh_run_returns_none_when_score_yields_no_tokens():
    eng = FakeEngine(completion_text="ok", score_tokens=[])
    sub = qc._EngineScoreSub(eng)
    assert qc.generate_fresh_run(sub, "c", "p") is None


def test_generate_fresh_run_never_raises_when_engine_blows_up():
    class BoomEngine(FakeEngine):
        def complete(self, prompt, **params):
            raise RuntimeError("engine down")

    sub = qc._EngineScoreSub(BoomEngine())
    assert qc.generate_fresh_run(sub, "c", "p") is None


def test_gather_fresh_runs_caps_at_default_prompt_table_size():
    eng = FakeEngine(completion_text="x", score_tokens=[_tok(1, "x", -0.1)])
    sub = qc._EngineScoreSub(eng)
    runs = qc.gather_fresh_runs(sub, 10_000)
    assert len(runs) == len(qc._DEFAULT_PROMPTS)


def test_gather_fresh_runs_drops_failed_prompts_but_keeps_going():
    calls = {"n": 0}

    class FlakyEngine(FakeEngine):
        def complete(self, prompt, **params):
            calls["n"] += 1
            if calls["n"] == 2:                     # the second prompt fails to generate
                return {"choices": [{"text": "   "}]}
            return super().complete(prompt, **params)

    eng = FlakyEngine(completion_text="ok", score_tokens=[_tok(1, "o", -0.1)])
    sub = qc._EngineScoreSub(eng)
    runs = qc.gather_fresh_runs(sub, 3)
    assert len(runs) == 2   # 3 requested, 1 dropped


def test_default_prompts_are_well_formed():
    assert len(qc._DEFAULT_PROMPTS) > 0
    for category, prompt in qc._DEFAULT_PROMPTS:
        assert isinstance(category, str) and category
        assert isinstance(prompt, str) and prompt


# ==================================================================================== gather_from_log_runs

def test_gather_from_log_runs_reads_the_run_journal(monkeypatch):
    rows = [{"id": "run_a"}, {"id": "run_b"}, {"id": "missing"}]
    runs_by_id = {"run_a": {"id": "run_a", "response": "A"}, "run_b": {"id": "run_b", "response": "B"}}

    import clozn.runs.store as runlog

    def fake_list_runs(limit=50, include_replays=True):
        assert include_replays is False
        return rows[:limit]

    def fake_get_run(rid):
        return runs_by_id.get(rid)

    monkeypatch.setattr(runlog, "list_runs", fake_list_runs)
    monkeypatch.setattr(runlog, "get_run", fake_get_run)

    out = qc.gather_from_log_runs(3)
    assert [r["id"] for r in out] == ["run_a", "run_b"]   # "missing" silently skipped


def test_gather_from_log_runs_empty_journal(monkeypatch):
    import clozn.runs.store as runlog
    monkeypatch.setattr(runlog, "list_runs", lambda limit=50, include_replays=True: [])
    assert qc.gather_from_log_runs(5) == []


# ==================================================================================================== build_receipts

RUN_A = {"id": "run_1", "category": "factual_qa",
        "messages": [{"role": "user", "content": "hi"}],
        "assembled_messages": [{"role": "user", "content": "hi"}],
        "response": "ok", "behavior": {"active_dials": {}}, "trace": {"token_ids": [1, 2]}}

TOKENS_A = [_tok(1, "o", -0.1, [_tok(1, "o", -0.1)]), _tok(2, "k", -0.1, [_tok(2, "k", -0.1)])]
TOKENS_B = [_tok(1, "o", -0.2, [_tok(1, "o", -0.2)]), _tok(2, "k", -0.2, [_tok(2, "k", -0.2)])]


def test_build_receipts_delegates_and_stamps_run_id_and_category():
    sub_a = FakeScoreSub(tokens=TOKENS_A)
    sub_b = FakeScoreSub(tokens=TOKENS_B)
    out = qc.build_receipts([RUN_A], sub_a, sub_b, label_a="Q8_0", label_b="Q4_K_M", topk=8)
    assert len(out) == 1
    r = out[0]
    assert r["causal_verified"] is True
    assert r["run_id"] == "run_1"
    assert r["category"] == "factual_qa"
    assert r["label_a"] == "Q8_0" and r["label_b"] == "Q4_K_M"
    assert r["n_tokens"] == 2


def test_build_receipts_handles_bad_run_gracefully():
    out = qc.build_receipts([None, {}], FakeScoreSub(), FakeScoreSub(), label_a="a", label_b="b")
    assert len(out) == 2
    for r in out:
        assert r["causal_verified"] is False
        assert r["run_id"] is None


def test_build_receipts_never_raises_when_a_substrate_raises():
    out = qc.build_receipts([RUN_A], FakeScoreSub(raises=True), FakeScoreSub(tokens=TOKENS_B),
                            label_a="a", label_b="b")
    assert out[0]["causal_verified"] is False
    assert out[0]["run_id"] == "run_1"


# ==================================================================================================== aggregate_receipts

def _receipt(run_id, category, answer, tokens_a, tokens_b, label_a="Q8_0", label_b="Q4_K_M"):
    r = qr.diff_quant_scores(answer, tokens_a, tokens_b, label_a=label_a, label_b=label_b)
    r["run_id"] = run_id
    r["category"] = category
    return r


def _skip(run_id, category, note="no continuation ids"):
    return {"causal_verified": False, "run_id": run_id, "category": category, "note": note}


def test_aggregate_receipts_totals_across_runs():
    r1 = _receipt("run_1", "factual_qa", [1, 2],
                  [_tok(1, "a", -0.1, [_tok(1, "a", -0.1)]), _tok(2, "b", -0.1, [_tok(2, "b", -0.1)])],
                  [_tok(1, "a", -0.2, [_tok(1, "a", -0.2)]), _tok(2, "b", -0.2, [_tok(2, "b", -0.2)])])
    r2 = _receipt("run_2", "reasoning", [5],
                  [_tok(5, "4", -0.1, [_tok(5, "4", -0.1)])],
                  [_tok(5, "4", -1.0, [_tok(9, "5", -0.1), _tok(5, "4", -1.0)])])   # flips
    agg = qc.aggregate_receipts([r1, r2], label_a="Q8_0", label_b="Q4_K_M")
    assert agg["n_runs"] == 2
    assert agg["n_verified"] == 2
    assert agg["n_skipped"] == 0
    assert agg["total_tokens"] == 3
    assert agg["total_preserved"] == 2
    assert agg["total_flipped"] == 1
    assert agg["pct_preserved"] == round(100.0 * 2 / 3, 1)
    assert len(agg["per_run"]) == 2
    assert agg["caveat"] == qr._QUANT_CAVEAT
    # the one flip is surfaced in top_flips, tagged with its run/category
    assert len(agg["top_flips"]) == 1
    assert agg["top_flips"][0]["run_id"] == "run_2"
    assert agg["top_flips"][0]["category"] == "reasoning"


def test_aggregate_receipts_excludes_skipped_runs_from_totals():
    r1 = _receipt("run_1", "code", [1],
                  [_tok(1, "a", -0.1, [_tok(1, "a", -0.1)])],
                  [_tok(1, "a", -0.1, [_tok(1, "a", -0.1)])])
    skipped = _skip("run_2", "arithmetic")
    agg = qc.aggregate_receipts([r1, skipped], label_a="Q8_0", label_b="Q4_K_M")
    assert agg["n_runs"] == 2
    assert agg["n_verified"] == 1
    assert agg["n_skipped"] == 1
    assert agg["total_tokens"] == 1   # only the verified run counts
    skip_row = next(row for row in agg["per_run"] if row["run_id"] == "run_2")
    assert skip_row["verified"] is False
    assert skip_row["note"] == "no continuation ids"


def test_aggregate_receipts_all_skipped_reports_zero_tokens_no_crash():
    agg = qc.aggregate_receipts([_skip("run_1", "x"), _skip("run_2", "y")], label_a="a", label_b="b")
    assert agg["total_tokens"] == 0
    assert agg["pct_preserved"] is None
    assert agg["n_verified"] == 0
    assert agg["top_flips"] == []


def test_aggregate_receipts_top_flips_sorted_by_abs_delta_across_runs():
    small = _receipt("run_1", "a", [1],
                     [_tok(1, "x", -0.1, [_tok(1, "x", -0.1)])],
                     [_tok(1, "x", -0.5, [_tok(9, "y", -0.05), _tok(1, "x", -0.5)])])   # smaller |delta|
    big = _receipt("run_2", "b", [1],
                   [_tok(1, "x", -0.1, [_tok(1, "x", -0.1)])],
                   [_tok(1, "x", -5.0, [_tok(9, "y", -0.05), _tok(1, "x", -5.0)])])     # bigger |delta|
    agg = qc.aggregate_receipts([small, big], label_a="Q8_0", label_b="Q4_K_M")
    assert [f["run_id"] for f in agg["top_flips"]] == ["run_2", "run_1"]


def test_aggregate_receipts_flip_total_is_not_reduced_by_per_run_detail_cap():
    # Receipt summaries retain only the first 20 flip details, but their n_flipped count is complete.
    # The aggregate's "top N of M" denominator must name the complete count, not len(details).
    receipt = {
        "causal_verified": True,
        "run_id": "run_many",
        "category": "reasoning",
        "n_tokens": 25,
        "summary": {
            "n_preserved": 0,
            "n_flipped": 25,
            "n_unknown": 0,
            "flipped_detail": [
                {"index": i, "piece": str(i), "delta_nats": float(i)} for i in range(20)
            ],
        },
    }

    agg = qc.aggregate_receipts([receipt], label_a="base", label_b="tuned")

    assert agg["total_flipped"] == 25
    assert agg["n_flips_total"] == 25
    assert len(agg["top_flips"]) == qc._TOP_FLIPS_ACROSS_RUNS


def test_aggregate_receipts_handles_malformed_receipt_entries():
    agg = qc.aggregate_receipts([None, "not a dict", {}], label_a="a", label_b="b")
    assert agg["n_verified"] == 0
    assert agg["n_skipped"] == 3


# ==================================================================================================== format_ladder

def test_format_ladder_renders_header_totals_and_caveat():
    r1 = _receipt("run_1", "factual_qa", [1, 2],
                  [_tok(1, "a", -0.1, [_tok(1, "a", -0.1)]), _tok(2, "b", -0.1, [_tok(2, "b", -0.1)])],
                  [_tok(1, "a", -0.2, [_tok(1, "a", -0.2)]), _tok(2, "b", -0.2, [_tok(2, "b", -0.2)])])
    agg = qc.aggregate_receipts([r1], label_a="Q8_0", label_b="Q4_K_M")
    out = qc.format_ladder(agg)
    assert "quant-check: Q8_0 vs Q4_K_M" in out
    assert "2/2 tokens preserved" in out
    assert "run_1" in out and "factual_qa" in out
    assert qr._QUANT_CAVEAT in out


def test_format_ladder_shows_flip_detail_with_both_arms_named():
    r1 = _receipt("run_1", "reasoning", [5],
                 [_tok(5, "4", -0.1, [_tok(5, "4", -0.1)])],
                 [_tok(5, "4", -1.0, [_tok(9, "5", -0.1), _tok(5, "4", -1.0)])],
                 label_a="Q8_0", label_b="Q2_K")
    agg = qc.aggregate_receipts([r1], label_a="Q8_0", label_b="Q2_K")
    out = qc.format_ladder(agg)
    assert "most-changed behaviors" in out
    assert "Q8_0->'4'" in out
    assert "Q2_K->'5'" in out


def test_format_ladder_handles_no_verified_runs_without_crashing():
    agg = qc.aggregate_receipts([_skip("run_1", "x")], label_a="a", label_b="b")
    out = qc.format_ladder(agg)
    assert "no verified runs" in out
    assert "0/1" in out


def test_format_ladder_marks_skipped_runs():
    r1 = _receipt("run_1", "code", [1],
                  [_tok(1, "a", -0.1, [_tok(1, "a", -0.1)])],
                  [_tok(1, "a", -0.1, [_tok(1, "a", -0.1)])])
    skipped = _skip("run_2", "arithmetic", note="engine could not score")
    agg = qc.aggregate_receipts([r1, skipped], label_a="Q8_0", label_b="Q4_K_M")
    out = qc.format_ladder(agg)
    assert "run_2" in out
    assert "engine could not score" in out


# ==================================================================================================== add_subparser / argparse

def _build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    qc.add_subparser(sub)
    return p


def test_add_subparser_defaults():
    p = _build_parser()
    args = p.parse_args(["quant-check", "A.gguf", "B.gguf"])
    assert args.model_a == "A.gguf"
    assert args.model_b == "B.gguf"
    assert args.runs == 8
    assert args.from_log is False
    assert args.topk == 8
    assert args.max_tokens == 200
    assert args.port_a == 0
    assert args.port_b == 0
    assert args.cpu is False
    assert args.json is False
    assert args.fn is qc.cmd_quant_check


def test_add_subparser_parses_overrides():
    p = _build_parser()
    args = p.parse_args(["quant-check", "A.gguf", "B.gguf", "--runs", "20", "--from-log", "--topk", "16",
                        "--max-tokens", "64", "--port-a", "9001", "--port-b", "9002", "--cpu", "--json"])
    assert args.runs == 20
    assert args.from_log is True
    assert args.topk == 16
    assert args.max_tokens == 64
    assert args.port_a == 9001
    assert args.port_b == 9002
    assert args.cpu is True
    assert args.json is True


def test_add_subparser_requires_both_models():
    import contextlib
    import io
    p = _build_parser()
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            p.parse_args(["quant-check", "only_one.gguf"])
            raised = False
        except SystemExit:
            raised = True
    assert raised
