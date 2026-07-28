"""Model-free contract tests for `clozn diff-adapter` (the live GPU path is not exercised here)."""
from __future__ import annotations

import argparse

import pytest

from clozn.cli import main as cli_main
from clozn.cli.commands import diff_adapter


def _parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    diff_adapter.add_subparser(sub)
    return p


def test_registered_on_the_real_cli_via_autoload():
    """It must reach the actual command tree, not merely be importable."""
    parser = cli_main.build_parser()
    sub = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    assert "diff-adapter" in sub.choices


def test_defaults():
    args = _parser().parse_args(["diff-adapter", "base.gguf", "adapter.gguf"])
    assert args.model == "base.gguf" and args.adapter == "adapter.gguf"
    assert args.adapter_scale is None       # resolved to 1.0 in the command, not baked into the parser
    assert args.runs == 8 and args.topk == 8 and args.max_tokens == 200
    assert args.from_log is False and args.both is False and args.cpu is False and args.json is False
    assert args.fn is diff_adapter.cmd_diff_adapter


def test_own_templates_is_deliberately_absent():
    """Both arms are the same model file and therefore the same chat template. Offering the flag would
    imply a degree of freedom this comparison does not have."""
    with pytest.raises(SystemExit):
        _parser().parse_args(["diff-adapter", "b.gguf", "a.gguf", "--own-templates"])


def test_missing_adapter_fails_before_booting_anything(monkeypatch, tmp_path):
    """Two model loads is an expensive way to discover a typo."""
    booted = []
    monkeypatch.setattr(diff_adapter, "spawn_engine",
                        lambda *a, **k: booted.append(a) or (None, {}, False))
    monkeypatch.setattr(diff_adapter, "resolve_model", lambda m: str(tmp_path / "base.gguf"))
    monkeypatch.setattr(diff_adapter.qc, "_import_engine_client", lambda: object)

    args = _parser().parse_args(["diff-adapter", "base.gguf", str(tmp_path / "nope.gguf")])
    with pytest.raises(Exception, match="adapter not found"):
        diff_adapter.cmd_diff_adapter(args)
    assert booted == [], "an engine was booted before the adapter path was validated"


def test_both_arms_get_identical_flags_except_the_adapter(monkeypatch, tmp_path):
    """The whole claim of this command rests on the two arms differing ONLY by the adapter. If the flag
    dicts diverged anywhere else, the comparison would be measuring something it does not name."""
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    adapter = tmp_path / "a.gguf"
    adapter.write_bytes(b"x")

    seen = []

    def fake_spawn(model, port, flags, prefer_gpu=True):
        seen.append((model, dict(flags)))
        return (None, {}, False)

    monkeypatch.setattr(diff_adapter, "spawn_engine", fake_spawn)
    monkeypatch.setattr(diff_adapter, "resolve_model", lambda m: str(base))
    monkeypatch.setattr(diff_adapter, "_flags_for", lambda m: {"ctx": 4096, "shared": "yes"})
    monkeypatch.setattr(diff_adapter.qc, "_import_engine_client", lambda: (lambda **kw: object()))
    monkeypatch.setattr(diff_adapter, "run_diff_model", lambda *a, **k: {"ladder": {}})

    args = _parser().parse_args(["diff-adapter", str(base), str(adapter), "--json"])
    diff_adapter.cmd_diff_adapter(args)

    assert len(seen) == 2
    (model_a, flags_a), (model_b, flags_b) = seen
    assert model_a == model_b, "both arms must serve the same model file"
    assert flags_b.pop("adapter") == str(adapter)
    assert flags_b.pop("adapter_scale") == 1.0
    assert flags_a == flags_b, f"arms differ beyond the adapter: {flags_a} vs {flags_b}"


def test_scale_is_threaded_and_labelled(monkeypatch, tmp_path, capsys):
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    adapter = tmp_path / "a.gguf"
    adapter.write_bytes(b"x")

    captured = {}

    def fake_run(eng_a, eng_b, args, *, label_a, label_b):
        captured["label_a"] = label_a
        captured["label_b"] = label_b
        return {"ladder": {}}

    monkeypatch.setattr(diff_adapter, "spawn_engine", lambda *a, **k: (None, {}, False))
    monkeypatch.setattr(diff_adapter, "resolve_model", lambda m: str(base))
    monkeypatch.setattr(diff_adapter, "_flags_for", lambda m: {})
    monkeypatch.setattr(diff_adapter.qc, "_import_engine_client", lambda: (lambda **kw: object()))
    monkeypatch.setattr(diff_adapter, "run_diff_model", fake_run)

    args = _parser().parse_args(["diff-adapter", str(base), str(adapter),
                                 "--adapter-scale", "0.5", "--json"])
    diff_adapter.cmd_diff_adapter(args)

    assert "base" in captured["label_a"]
    assert "@0.5" in captured["label_b"], f"scale not surfaced in the label: {captured['label_b']}"

    result = capsys.readouterr().out
    assert '"adapter_scale": 0.5' in result
    assert '"sole_difference": "adapter_weight_delta"' in result


def test_result_records_what_was_held_constant(monkeypatch, tmp_path, capsys):
    """The report's central claim is that the confounds diff-model must check for are absent here by
    construction. That has to be in the machine-readable result, not only in the prose."""
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    adapter = tmp_path / "a.gguf"
    adapter.write_bytes(b"x")

    monkeypatch.setattr(diff_adapter, "spawn_engine", lambda *a, **k: (None, {}, False))
    monkeypatch.setattr(diff_adapter, "resolve_model", lambda m: str(base))
    monkeypatch.setattr(diff_adapter, "_flags_for", lambda m: {})
    monkeypatch.setattr(diff_adapter.qc, "_import_engine_client", lambda: (lambda **kw: object()))
    monkeypatch.setattr(diff_adapter, "run_diff_model", lambda *a, **k: {"ladder": {}})

    args = _parser().parse_args(["diff-adapter", str(base), str(adapter), "--json"])
    diff_adapter.cmd_diff_adapter(args)

    import json
    out = json.loads(capsys.readouterr().out)
    held = out["comparison"]["held_constant_by_construction"]
    for expected in ("model_file", "tokenizer", "chat_template", "quantization", "engine_build"):
        assert expected in held
    assert out["comparison"]["kind"] == "base_vs_adapter"


def test_engines_are_torn_down_even_when_the_ladder_raises(monkeypatch, tmp_path):
    """A tokenizer-preflight refusal raises through the finally; two leaked GPU processes would be a
    genuinely expensive bug."""
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    adapter = tmp_path / "a.gguf"
    adapter.write_bytes(b"x")

    class FakeProc:
        def __init__(self):
            self.terminated = False

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            self.terminated = True

    procs = []

    def fake_spawn(*a, **k):
        p = FakeProc()
        procs.append(p)
        return (p, {}, False)

    monkeypatch.setattr(diff_adapter, "spawn_engine", fake_spawn)
    monkeypatch.setattr(diff_adapter, "resolve_model", lambda m: str(base))
    monkeypatch.setattr(diff_adapter, "_flags_for", lambda m: {})
    monkeypatch.setattr(diff_adapter.qc, "_import_engine_client", lambda: (lambda **kw: object()))

    def boom(*a, **k):
        raise RuntimeError("tokenizer preflight refused")

    monkeypatch.setattr(diff_adapter, "run_diff_model", boom)

    args = _parser().parse_args(["diff-adapter", str(base), str(adapter)])
    with pytest.raises(RuntimeError, match="tokenizer preflight refused"):
        diff_adapter.cmd_diff_adapter(args)

    assert len(procs) == 2
    assert all(p.terminated for p in procs), "an engine process leaked on the error path"
