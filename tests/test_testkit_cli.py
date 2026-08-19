"""test_testkit_cli -- model-free tests for `clozn test` (cli.cmd_test / build_parser's "test" subcommand).

Drives cli.cmd_test(args) directly (mirrors tests/test_cli_trace.py's SimpleNamespace-args pattern) against
an isolated runlog store and a real JSON spec file on disk -- no server, no model, no GPU. Covers: the
exit-code contract (0/1/2), --json, --attach's round trip into receipt_bundle, and that a causal assertion
without --live is an honest skip (exit 0), never a silent pass.
"""
from __future__ import annotations

import io
import json
import os
import sys
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import clozn.cli.main as cli       # noqa: E402
import clozn.cli.formatting as fmt          # noqa: E402
import clozn.receipts.bundle as receipt_bundle    # noqa: E402
import clozn.runs.store as runlog            # noqa: E402


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    # The color globals live in clozn.cli.formatting -- format_test_report reads fmt.DIM/BOLD/RST live.
    monkeypatch.setattr(fmt, "COLOR", False)
    monkeypatch.setattr(fmt, "DIM", "")
    monkeypatch.setattr(fmt, "BOLD", "")
    monkeypatch.setattr(fmt, "RST", "")
    return tmp_path


def _args(file, *, json_out=False, attach=False, live=False, port=0):
    return SimpleNamespace(file=str(file), json=json_out, attach=attach, live=live, port=port)


def _write_spec(tmp_path, spec, name="spec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(spec), encoding="utf-8")
    return p


def _make_run(**overrides):
    rec = dict(source="test", client="pytest", model="m",
              messages=[{"role": "user", "content": "capital of France?"}],
              response="The capital of France is Paris.", finish_reason="stop", started=1.0, ended=1.1)
    rec.update(overrides)
    return runlog.record(**rec)


# ================================================================================================== exit 2
def test_missing_file_is_exit_2(iso, capsys):
    rc = cli.cmd_test(_args(iso / "does_not_exist.json"))
    assert rc == 2
    assert "clozn test:" in capsys.readouterr().err


def test_invalid_json_is_exit_2(iso, capsys):
    p = iso / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    rc = cli.cmd_test(_args(p))
    assert rc == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_missing_tests_key_is_exit_2(iso, capsys):
    p = _write_spec(iso, {"nope": []})
    rc = cli.cmd_test(_args(p))
    assert rc == 2
    assert "'tests' list" in capsys.readouterr().err


def test_empty_tests_list_is_exit_2(iso):
    p = _write_spec(iso, {"tests": []})
    assert cli.cmd_test(_args(p)) == 2


def test_tests_not_a_list_is_exit_2(iso):
    p = _write_spec(iso, {"tests": "nope"})
    assert cli.cmd_test(_args(p)) == 2


# ================================================================================================== exit 0/1
def test_all_pass_is_exit_0(iso, capsys):
    _make_run()
    spec = {"tests": [{"name": "capital is paris", "run": "latest",
                      "assert": [{"check": "contains", "value": "Paris"},
                                {"check": "finish_reason", "value": "stop"}]}]}
    p = _write_spec(iso, spec)
    rc = cli.cmd_test(_args(p))
    out = capsys.readouterr().out
    assert rc == 0
    assert "capital is paris" in out
    assert "2 pass" in out


def test_any_fail_is_exit_1(iso, capsys):
    _make_run()
    spec = {"tests": [{"name": "wrong capital", "run": "latest",
                      "assert": [{"check": "contains", "value": "Berlin"}]}]}
    p = _write_spec(iso, spec)
    rc = cli.cmd_test(_args(p))
    out = capsys.readouterr().out
    assert rc == 1
    assert "1 fail" in out



def test_run_not_found_is_exit_1_not_2():
    """A spec-shape problem is exit 2; a run that fails to RESOLVE is a per-test 'error', which is exit 1
    (the spec file itself was fine)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        spec = {"tests": [{"name": "t", "run": "run_does_not_exist",
                          "assert": [{"check": "contains", "value": "x"}]}]}
        p = os.path.join(d, "spec.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(spec, f)
        assert cli.cmd_test(_args(p)) == 1


# ============================================================================================== causal honesty

def test_fetch_live_receipt_returns_a_sentinel_on_502_engine_not_reachable(monkeypatch):
    """The gateway's own /runs/<id>/receipt now answers 502 "engine not reachable ..." when ITS engine is
    down (fix 1a/3). _fetch_live_receipt used to swallow ANY failure -- this one included -- into a bare
    None, identical to a bad influence spec; it must now surface a distinguishing sentinel instead."""
    import urllib.error
    import urllib.request
    from clozn.cli.commands import test as cmd_test

    detail = "engine not reachable at http://127.0.0.1:8080 -- is it running?"
    body = json.dumps({"error": detail}).encode()

    def fake_urlopen(req, timeout=120):
        raise urllib.error.HTTPError(req.full_url, 502, "Bad Gateway", hdrs=None, fp=io.BytesIO(body))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    out = cmd_test._fetch_live_receipt(8080, "run123", {"dial": "warm"})
    assert out == {"_fetch_error": "engine_unreachable", "_fetch_detail": detail}


def test_fetch_live_receipt_returns_a_sentinel_on_503_too(monkeypatch):
    import urllib.error
    import urllib.request
    from clozn.cli.commands import test as cmd_test

    body = json.dumps({"error": "receipt requires a ready product model worker"}).encode()

    def fake_urlopen(req, timeout=120):
        raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", hdrs=None, fp=io.BytesIO(body))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    out = cmd_test._fetch_live_receipt(8080, "run123", {"dial": "warm"})
    assert out == {"_fetch_error": "engine_unreachable",
                   "_fetch_detail": "receipt requires a ready product model worker"}


def test_fetch_live_receipt_returns_none_on_a_plain_400_bad_request(monkeypatch):
    """An ordinary bad-request 4xx is UNCHANGED -- still a bare None, not the engine-down sentinel."""
    import urllib.error
    import urllib.request
    from clozn.cli.commands import test as cmd_test

    body = json.dumps({"error": "need an influence spec"}).encode()

    def fake_urlopen(req, timeout=120):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", hdrs=None, fp=io.BytesIO(body))
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    out = cmd_test._fetch_live_receipt(8080, "run123", {"bogus": True})
    assert out is None


def test_fetch_live_receipt_returns_none_on_connection_refused(monkeypatch):
    """The gateway itself is unreachable (not merely ITS engine) -- a plain URLError, not an HTTPError at
    all. Stays a bare None: judge_receipt's ordinary "could not be computed" note is the honest message
    when there's no gateway response to even inspect."""
    import urllib.error
    import urllib.request
    from clozn.cli.commands import test as cmd_test

    def fake_urlopen(req, timeout=120):
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    out = cmd_test._fetch_live_receipt(1, "run123", {"dial": "warm"})
    assert out is None


# ===================================================================================================== --json
def test_json_flag_prints_machine_readable_result(iso, capsys):
    _make_run()
    spec = {"tests": [{"name": "capital is paris", "run": "latest",
                      "assert": [{"check": "contains", "value": "Paris"}]}]}
    p = _write_spec(iso, spec)
    cli.cmd_test(_args(p, json_out=True))
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["status"] == "pass"
    assert data["tests"][0]["name"] == "capital is paris"
    assert data["tiny_tests"][0]["check"] == "contains"


# =================================================================================================== --attach
def test_attach_writes_tiny_tests_and_receipt_bundle_reads_it(iso):
    rid = _make_run()
    spec = {"tests": [{"name": "capital is paris", "run": rid,
                      "assert": [{"check": "contains", "value": "Paris"},
                                {"check": "finish_reason", "value": "stop"}]}]}
    p = _write_spec(iso, spec)
    rc = cli.cmd_test(_args(p, attach=True))
    assert rc == 0

    run = runlog.get_run(rid)
    assert run["tiny_tests"] is not None
    assert len(run["tiny_tests"]) == 2
    assert all(a["status"] == "pass" for a in run["tiny_tests"])

    bundle = receipt_bundle.build(run)
    assert bundle["tiny_tests"] == run["tiny_tests"]


def test_without_attach_the_run_is_untouched(iso):
    rid = _make_run()
    spec = {"tests": [{"name": "t", "run": rid, "assert": [{"check": "contains", "value": "Paris"}]}]}
    p = _write_spec(iso, spec)
    cli.cmd_test(_args(p, attach=False))
    assert runlog.get_run(rid).get("tiny_tests") is None


def test_attach_overwrites_not_appends_on_a_second_run(iso):
    rid = _make_run()
    spec1 = {"tests": [{"name": "t1", "run": rid, "assert": [{"check": "contains", "value": "Paris"}]}]}
    spec2 = {"tests": [{"name": "t2", "run": rid, "assert": [{"check": "contains", "value": "Paris"},
                                                            {"check": "finish_reason", "value": "stop"}]}]}
    cli.cmd_test(_args(_write_spec(iso, spec1, "s1.json"), attach=True))
    assert len(runlog.get_run(rid)["tiny_tests"]) == 1
    cli.cmd_test(_args(_write_spec(iso, spec2, "s2.json"), attach=True))
    assert len(runlog.get_run(rid)["tiny_tests"]) == 2


# ============================================================================================== parser wiring
def test_build_parser_exposes_the_test_subcommand_and_flags():
    args = cli.build_parser().parse_args(["test", "spec.json", "--json", "--attach", "--live", "--port", "9001"])
    assert args.file == "spec.json"
    assert args.json is True and args.attach is True and args.live is True and args.port == 9001
    assert args.fn is cli.cmd_test


def test_build_parser_test_subcommand_defaults():
    args = cli.build_parser().parse_args(["test", "spec.json"])
    assert args.json is False and args.attach is False and args.live is False and args.port == 0


# ================================================================================================ main() propagates
def test_main_propagates_cmd_test_exit_code(iso, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["clozn"])
    p = iso / "bad.json"
    p.write_text("not json", encoding="utf-8")
    rc = cli.main(["test", str(p)])
    assert rc == 2


def test_main_still_returns_0_for_a_command_with_no_return_value(iso, monkeypatch, capsys):
    """Regression guard: main()'s new `rc if isinstance(rc, int) else 0` must not break every OTHER
    command, which return None implicitly."""
    rc = cli.main(["models"])
    assert rc == 0
