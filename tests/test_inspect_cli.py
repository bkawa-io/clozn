"""`clozn inspect`: local-journal-first any-client bridge; no server/model/GPU by default."""
import json

import pytest

from clozn.cli import main as cli
from clozn.cli.commands import explain as explain_cmd

import clozn.runs.store as runlog


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _record():
    return runlog.record(
        source="openai_api",
        messages=[{"role": "user", "content": "Capital of France?"}],
        response="Paris.",
        trace={"tokens": ["Paris", "."], "confidence": [.91, .99], "alternatives": [[], []]},
        started=1000.0,
    )


def test_parser_registers_inspect_as_zero_generation_command():
    args = cli.build_parser().parse_args(["inspect", "run_x"])
    assert args.fn is cli.cmd_inspect
    assert args.run_id == "run_x" and args.last is False and args.json is False
    assert not hasattr(args, "why")


def test_inspect_reads_local_journal_without_touching_gateway(iso, monkeypatch, capsys):
    rid = _record()
    monkeypatch.setattr(explain_cmd, "_fetch_explain",
                        lambda *_: (_ for _ in ()).throw(AssertionError("gateway must not be called")))

    cli.cmd_inspect(cli.build_parser().parse_args(["inspect", rid]))

    out = capsys.readouterr().out
    assert f"run {rid}" in out
    assert "confidence" in out and "influences active" in out


def test_inspect_json_is_the_exact_local_explanation(iso, monkeypatch, capsys):
    rid = _record()
    monkeypatch.setattr(explain_cmd, "_fetch_explain",
                        lambda *_: (_ for _ in ()).throw(AssertionError("gateway must not be called")))
    cli.cmd_inspect(cli.build_parser().parse_args(["inspect", rid, "--json"]))
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"] == rid
    assert out["confidence"]["n_tokens"] == 2


def test_inspect_last_uses_latest_non_replay_run(iso, monkeypatch, capsys):
    rid = _record()
    runlog.record(source="replay", messages=[], response="arm", started=2000.0, parent_run_id=rid)
    monkeypatch.setattr(explain_cmd, "_fetch_explain",
                        lambda *_: (_ for _ in ()).throw(AssertionError("gateway must not be called")))
    cli.cmd_inspect(cli.build_parser().parse_args(["inspect", "--last", "--json"]))
    assert json.loads(capsys.readouterr().out)["run_id"] == rid


def test_missing_local_id_falls_back_to_requested_gateway(iso, monkeypatch, capsys):
    seen = {}
    def fake(port, rid):
        seen.update(port=port, rid=rid)
        return {"run_id": rid, "confidence": {"available": False},
                "influences_active": {}, "concepts": {"available": False}}
    monkeypatch.setattr(explain_cmd, "_fetch_explain", fake)

    cli.cmd_inspect(cli.build_parser().parse_args(["inspect", "remote_run", "--port", "9090"]))
    assert seen == {"port": 9090, "rid": "remote_run"}
    assert "run remote_run" in capsys.readouterr().out


def test_inspect_requires_id_or_last(iso):
    with pytest.raises(cli.CloznError, match="clozn_run_id"):
        cli.cmd_inspect(cli.build_parser().parse_args(["inspect"]))
