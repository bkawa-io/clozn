"""Model-free CLI coverage for ``clozn provenance``: arg parsing, the human/JSON render of fixture
receipts (all four verdicts + a blocked dict), and the 'last' journal-read wiring (a fake journal, a
monkeypatched trace_provenance -- never a live engine)."""
from __future__ import annotations

import json

import pytest

from clozn.analysis import provenance as prov
from clozn.cli import main as cli
from clozn.cli.commands.provenance import format_provenance
import clozn.runs.store as runlog


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))


def _record(*, final_prompt="RENDERED PROMPT", response="Tokyo", started=1000.0, source="openai_api"):
    return runlog.record(
        source=source, messages=[{"role": "user", "content": "what is the capital?"}],
        response=response, final_prompt=final_prompt, started=started, ended=started + 0.01,
    )


# --------------------------------------------------------------------------------------- arg parsing

def test_parser_defaults():
    args = cli.build_parser().parse_args(["provenance", "hello"])
    assert args.prompt == "hello"
    assert args.continuation is None
    assert args.focus is None
    assert args.engine == prov.DEFAULT_ENGINE
    assert args.seed == 0
    assert args.json is False
    assert args.fn.__name__ == "cmd_provenance"


def test_parser_accepts_all_flags():
    args = cli.build_parser().parse_args([
        "provenance", "hello world", "--continuation", "Paris",
        "--focus", "2", "5", "--engine", "http://x:9", "--seed", "7", "--json",
    ])
    assert args.prompt == "hello world"
    assert args.continuation == "Paris"
    assert args.focus == [2, 5]
    assert args.engine == "http://x:9"
    assert args.seed == 7
    assert args.json is True


def test_parser_accepts_last_as_the_prompt_positional():
    args = cli.build_parser().parse_args(["provenance", "last"])
    assert args.prompt == "last"


# -------------------------------------------------------------------------------- fixture receipt render

_BLOCKED = {"ok": False, "blocked": "engine lacks attn_knockout; start clozn-server with --no-flash-attn "
                                    "(flash attention fuses the softmax, so the attention weights never "
                                    "materialize)"}

_CONTEXT_CARRIED = {
    "ok": True, "answer": "Paris", "baseline_logprob": -0.05, "cut_logprob": -6.2,
    "span": [3, 4], "span_tokens": ["France", "is"], "delta": 6.15, "dependence": 0.95,
    "best_control_ratio": 12.3, "verdict": "CONTEXT_CARRIED",
    "top3_after_cut": [], "best_single": {"pos": 3, "token": "'France'", "delta": 5.9},
    "trace": [], "focus": None, "focus_trim": None, "focus_null": None,
    "config": {"renormalize": True, "n_layer": 28, "seed": 0},
}

_MIXED = {**_CONTEXT_CARRIED, "verdict": "MIXED", "dependence": 0.5, "best_control_ratio": 10.0}

_PARAMETRIC_WITH_SPAN = {**_CONTEXT_CARRIED, "verdict": "PARAMETRIC", "dependence": 0.1,
                        "best_control_ratio": 18.0}

_PARAMETRIC_NO_SPAN = {
    "ok": True, "answer": "Tokyo", "span": [], "dependence": 0.0, "verdict": "PARAMETRIC",
    "note": "no context position changed the answer", "focus": None, "baseline_logprob": -0.4,
}

_INCONCLUSIVE = {**_CONTEXT_CARRIED, "verdict": "INCONCLUSIVE", "dependence": 0.99,
                 "best_control_ratio": 2.9}

_FOCUS_RECEIPT = {
    **_CONTEXT_CARRIED, "focus": [2, 5],
    "focus_null": {"p_value": 0.0833, "n_draws": 12, "control_deltas": [], "pool": "outside-focus"},
}


@pytest.mark.parametrize("receipt,expected_verdict", [
    (_CONTEXT_CARRIED, "CONTEXT_CARRIED"),
    (_MIXED, "MIXED"),
    (_PARAMETRIC_WITH_SPAN, "PARAMETRIC"),
    (_INCONCLUSIVE, "INCONCLUSIVE"),
])
def test_format_provenance_renders_each_verdict(receipt, expected_verdict):
    out = format_provenance(receipt)
    assert out.startswith(prov.SCOPE_NOTE)
    assert f"verdict: {expected_verdict}" in out
    assert "dependence" in out and "best control ratio" in out
    assert "'France'" in out and "'is'" in out   # carrying span tokens, repr'd


def test_format_provenance_renders_parametric_with_no_span_honestly():
    out = format_provenance(_PARAMETRIC_NO_SPAN)
    assert "verdict: PARAMETRIC" in out
    assert "carrying span: (none)" in out
    assert "no context position changed the answer" in out


def test_format_provenance_renders_focus_null_p_value():
    out = format_provenance(_FOCUS_RECEIPT)
    assert "focus: prompt token span (2, 5)" in out
    assert "focus_null p-value: 0.0833" in out
    assert "not yet folded into the verdict" in out


def test_format_provenance_prints_blocked_message_verbatim_never_crashes():
    out = format_provenance(_BLOCKED)
    assert out.startswith(prov.SCOPE_NOTE)
    assert _BLOCKED["blocked"] in out
    assert "provenance blocked:" in out


# ---------------------------------------------------------------------------------------- cmd wiring

def test_cmd_provenance_prints_json_with_scope_and_returns_1_when_blocked(monkeypatch, capsys):
    monkeypatch.setattr(prov, "trace_provenance", lambda *a, **k: dict(_BLOCKED))
    rc = cli.main(["provenance", "some prompt", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["scope"] == prov.SCOPE_NOTE


def test_cmd_provenance_returns_0_and_renders_human_output_when_ok(monkeypatch, capsys):
    monkeypatch.setattr(prov, "trace_provenance", lambda *a, **k: dict(_CONTEXT_CARRIED))
    rc = cli.main(["provenance", "some prompt"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "CONTEXT_CARRIED" in out


def test_cmd_provenance_forwards_prompt_and_defaults_to_greedy_continuation(monkeypatch):
    captured = {}

    def fake(prompt, continuation, **kwargs):
        captured["prompt"] = prompt
        captured["continuation"] = continuation
        captured["kwargs"] = kwargs
        return dict(_CONTEXT_CARRIED)

    monkeypatch.setattr(prov, "trace_provenance", fake)
    cli.main(["provenance", "raw prompt text", "--focus", "1", "3", "--seed", "5"])
    assert captured["prompt"] == "raw prompt text"
    assert captured["continuation"] is None
    assert captured["kwargs"]["focus"] == (1, 3)
    assert captured["kwargs"]["seed"] == 5


def test_cmd_provenance_continuation_flag_overrides_greedy(monkeypatch):
    captured = {}
    monkeypatch.setattr(prov, "trace_provenance",
                        lambda prompt, continuation, **k: captured.update(continuation=continuation)
                        or dict(_CONTEXT_CARRIED))
    cli.main(["provenance", "raw prompt", "--continuation", "explicit answer"])
    assert captured["continuation"] == "explicit answer"


# -------------------------------------------------------------------------------- 'last' journal wiring

def test_last_uses_the_latest_runs_final_prompt_and_recorded_answer(isolated, monkeypatch):
    _record(final_prompt="EXACT RENDERED PROMPT", response="Tokyo")
    captured = {}

    def fake(prompt, continuation, **kwargs):
        captured["prompt"] = prompt
        captured["continuation"] = continuation
        return dict(_CONTEXT_CARRIED)

    monkeypatch.setattr(prov, "trace_provenance", fake)
    rc = cli.main(["provenance", "last"])
    assert rc == 0
    assert captured["prompt"] == "EXACT RENDERED PROMPT"
    assert captured["continuation"] == "Tokyo"


def test_last_continuation_flag_overrides_the_runs_recorded_answer(isolated, monkeypatch):
    _record(final_prompt="EXACT RENDERED PROMPT", response="Tokyo")
    captured = {}

    def fake(prompt, continuation, **kwargs):
        captured["continuation"] = continuation
        return dict(_CONTEXT_CARRIED)

    monkeypatch.setattr(prov, "trace_provenance", fake)
    cli.main(["provenance", "last", "--continuation", "Osaka"])
    assert captured["continuation"] == "Osaka"


def test_last_excludes_derived_runs(isolated, monkeypatch):
    organic = _record(final_prompt="ORGANIC PROMPT", response="organic answer", started=1.0)
    runlog.record(source="replay", parent_run_id=organic, messages=[],
                 response="derived answer", final_prompt="DERIVED PROMPT", started=2.0)
    captured = {}
    monkeypatch.setattr(prov, "trace_provenance",
                        lambda prompt, continuation, **k: captured.update(prompt=prompt) or
                        dict(_CONTEXT_CARRIED))
    cli.main(["provenance", "last"])
    assert captured["prompt"] == "ORGANIC PROMPT"


def test_last_with_no_recorded_runs_is_a_clean_cloznerror(isolated, capsys):
    rc = cli.main(["provenance", "last"])
    assert rc == 1
    assert "no recorded run found" in capsys.readouterr().err


def test_last_with_no_final_prompt_is_a_clean_cloznerror(isolated, capsys):
    _record(final_prompt=None, response="Tokyo")
    rc = cli.main(["provenance", "last"])
    assert rc == 1
    assert "no recorded final_prompt" in capsys.readouterr().err
