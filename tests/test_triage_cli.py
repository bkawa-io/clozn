"""tests/test_triage_cli.py -- `clozn triage` (clozn/cli/commands/triage.py, roadmap feature 05).

Model-free throughout: `_resolve_run_pair_from_ids` is tested against a real, isolated run store
(mirrors tests/test_ci_check.py's `iso` fixture); `_resolve_run_pair_from_experiment` is tested against a
real `clozn.experiment.result.v0` object built the same way tests/test_experiment_suite_cmd.py does
(`suite.validate_manifest`/`_manifest_digest`/`_summarize`/`validate_result`) -- no monkeypatching needed
for either path, since both are pure data lookups over already-recorded evidence.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import clozn.cli.commands.triage as trg
import clozn.cli.formatting as fmt
import clozn.runs.identity as identity
import clozn.runs.store as runlog
from clozn.cli.main import CloznError, build_parser
from clozn.experiments import suite


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(identity, "_CACHE_PATH", str(tmp_path / "model_hashes.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(fmt, "COLOR", False)
    return tmp_path


def _make_run(**overrides):
    rec = dict(source="test", client="pytest", model="m",
              messages=[{"role": "user", "content": "hi"}], response="hello",
              finish_reason="stop", started=1.0, ended=1.1)
    rec.update(overrides)
    return runlog.record(**rec)


def _base_args(**overrides):
    base = dict(experiment_result=None, case=None, variant=None, seed=None,
               baseline_run=None, candidate_run=None, steps=None, deep=False,
               max_runs=None, max_seconds=None, out=None, force=False, json=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def _write_experiment_result(tmp_path, *, seeds=(0,), baseline_status="fail", candidate_status="pass",
                             baseline_identity=None, candidate_identity=None,
                             baseline_context=None, candidate_context=None, extra_variant=False):
    variants = [{"name": "base", "kind": "base"}, {"name": "cand", "kind": "tuned"}]
    if extra_variant:
        variants.append({"name": "cand2", "kind": "tuned"})
    manifest = suite.validate_manifest({
        "schema_version": suite.MANIFEST_SCHEMA, "name": "triage-cli-check", "seeds": list(seeds),
        "defaults": {}, "baseline_variant": "base", "variants": variants,
        "suites": {
            "target": {"cases": [{"name": "greeting", "prompt": "hi"}]},
            "guard": {"cases": [{"name": "g1", "prompt": "g"}]},
        },
    })
    cells = []
    for variant in manifest["variants"]:
        name = variant["name"]
        if name == "base":
            status = baseline_status
        elif name == "cand":
            status = candidate_status
        else:
            status = "pass"
        for suite_name, case in (("target", "greeting"), ("guard", "g1")):
            for seed in manifest["seeds"]:
                run_id = f"run-{name}-{suite_name}-{case}-{seed}"
                run = {"id": run_id, "model": "clozn"}
                if name == "base" and baseline_identity is not None:
                    run["identity"] = baseline_identity
                if name == "cand" and candidate_identity is not None:
                    run["identity"] = candidate_identity
                if name == "base" and baseline_context is not None:
                    run["context_receipt"] = baseline_context
                if name == "cand" and candidate_context is not None:
                    run["context_receipt"] = candidate_context
                cells.append({"suite": suite_name, "case": case, "variant": name,
                             "variant_kind": variant["kind"], "seed": seed, "status": status,
                             "run_id": run_id, "response": "x", "assertions": [],
                             "min_confidence": None, "receipts": None, "error": None, "run": run})
    result = suite.validate_result({
        "schema_version": suite.RESULT_SCHEMA, "experiment_id": "exp_triage_cli", "name": manifest["name"],
        "created_at": "2026-07-27T00:00:00Z", "manifest_sha256": suite._manifest_digest(manifest),
        "manifest": manifest, "seeds": manifest["seeds"], "cells": cells,
        "summary": suite._summarize(cells, "base", [v["name"] for v in manifest["variants"]]),
    })
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return str(path)


# =========================================================================================== argparse ====

def _subparser_choices(p):
    for action in p._actions:
        if getattr(action, "choices", None) and "triage" in action.choices:
            return action.choices
    return {}


def test_triage_is_registered_via_autoload():
    assert "triage" in _subparser_choices(build_parser())


def test_triage_defaults():
    ns = build_parser().parse_args(["triage", "--baseline-run", "run_b", "--candidate-run", "run_c"])
    assert ns.experiment_result is None
    assert ns.case is None and ns.variant is None and ns.seed is None
    assert ns.deep is False and ns.json is False and ns.force is False
    assert ns.out is None and ns.steps is None
    assert ns.plan is False and ns.dry_run is False


def test_triage_does_not_collide_with_the_existing_diagnose_command():
    """`clozn diagnose` (latency/cutoff, clozn/cli/commands/diagnose.py) is unrelated and pre-existing;
    this feature deliberately used a different verb rather than renaming or touching it."""
    choices = build_parser()._subparsers._group_actions[0].choices
    assert "diagnose" in choices
    assert "triage" in choices
    assert choices["diagnose"] is not choices["triage"]


# ========================================================================================== validation ===

def test_validate_args_rejects_both_sources():
    with pytest.raises(CloznError):
        trg._validate_args(_base_args(experiment_result="r.json", case="c",
                                      baseline_run="b", candidate_run="c"))


def test_validate_args_rejects_neither_source():
    with pytest.raises(CloznError):
        trg._validate_args(_base_args())


def test_validate_args_requires_case_with_an_experiment_result():
    with pytest.raises(CloznError):
        trg._validate_args(_base_args(experiment_result="r.json"))


def test_validate_args_requires_both_run_ids():
    with pytest.raises(CloznError):
        trg._validate_args(_base_args(baseline_run="run_b"))


def test_validate_args_rejects_case_flags_with_a_run_pair():
    with pytest.raises(CloznError):
        trg._validate_args(_base_args(baseline_run="run_b", candidate_run="run_c", seed=0))


def test_validate_args_accepts_a_well_formed_run_pair():
    trg._validate_args(_base_args(baseline_run="run_b", candidate_run="run_c"))


def test_validate_args_accepts_a_well_formed_experiment_selection():
    trg._validate_args(_base_args(experiment_result="r.json", case="greeting"))


# ============================================================================== run-pair resolution ======

def test_resolve_run_pair_from_ids_reads_real_recorded_runs(iso):
    b_id = _make_run(identity={"model_sha256": "a" * 64})
    c_id = _make_run(identity={"model_sha256": "b" * 64})
    baseline, candidate, source_experiment_id, case_id = trg._resolve_run_pair_from_ids(
        _base_args(baseline_run=b_id, candidate_run=c_id))
    assert baseline["id"] == b_id and candidate["id"] == c_id
    assert source_experiment_id is None and case_id is None


def test_resolve_run_pair_from_ids_raises_on_unknown_run(iso):
    b_id = _make_run()
    with pytest.raises(CloznError):
        trg._resolve_run_pair_from_ids(_base_args(baseline_run=b_id, candidate_run="run_does_not_exist"))


def test_resolve_run_pair_from_experiment_picks_the_regressed_seed(tmp_path):
    path = _write_experiment_result(tmp_path, seeds=(0, 1), baseline_status="pass",
                                    candidate_status="fail")
    baseline, candidate, experiment_id, case_id = trg._resolve_run_pair_from_experiment(
        _base_args(experiment_result=path, case="greeting"))
    assert experiment_id == "exp_triage_cli"
    assert case_id == "greeting"
    assert baseline["id"] == "run-base-target-greeting-0"
    assert candidate["id"] == "run-cand-target-greeting-0"


def test_resolve_run_pair_from_experiment_requires_variant_when_ambiguous(tmp_path):
    path = _write_experiment_result(tmp_path, extra_variant=True)
    with pytest.raises(CloznError):
        trg._resolve_run_pair_from_experiment(_base_args(experiment_result=path, case="greeting"))
    baseline, candidate, _, _ = trg._resolve_run_pair_from_experiment(
        _base_args(experiment_result=path, case="greeting", variant="cand"))
    assert candidate["id"].startswith("run-cand-")


def test_resolve_run_pair_from_experiment_rejects_unknown_case(tmp_path):
    path = _write_experiment_result(tmp_path)
    with pytest.raises(CloznError):
        trg._resolve_run_pair_from_experiment(_base_args(experiment_result=path, case="not-a-case"))


def test_resolve_run_pair_from_experiment_rejects_unknown_seed(tmp_path):
    path = _write_experiment_result(tmp_path, seeds=(0,))
    with pytest.raises(CloznError):
        trg._resolve_run_pair_from_experiment(
            _base_args(experiment_result=path, case="greeting", seed=99))


# ========================================================================================= end to end ====

def test_cmd_triage_json_output_is_a_valid_artifact(iso, capsys):
    b_id = _make_run(identity={"model_sha256": "a" * 64})
    c_id = _make_run(identity={"model_sha256": "a" * 64})
    rc = trg.cmd_triage(_base_args(baseline_run=b_id, candidate_run=c_id, json=True))
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["schema_version"] == "clozn.triage.v1"
    assert doc["summary"]["classification"] == "undetermined"
    assert "identity_diff:model" in doc["summary"]["eliminated"]

    from clozn import schemas
    schemas.validate(doc, "clozn.triage.v1")


def test_cmd_triage_human_output_headlines_undetermined_without_causal_evidence(iso, capsys):
    b_id = _make_run(identity={"model_sha256": "a" * 64})
    c_id = _make_run(identity={"model_sha256": "b" * 64})
    rc = trg.cmd_triage(_base_args(baseline_run=b_id, candidate_run=c_id))
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("UNDETERMINED")
    assert "OBSERVED" in out
    assert "identity_diff:model" in out


def test_cmd_triage_reports_deferred_steps_as_explicit_not_run(iso, capsys):
    b_id = _make_run()
    c_id = _make_run()
    rc = trg.cmd_triage(_base_args(baseline_run=b_id, candidate_run=c_id))
    assert rc == 0
    out = capsys.readouterr().out
    assert "NOT RUN" in out
    assert "template_swap" in out


def test_cmd_triage_narrow_steps_filter_excludes_placeholders(iso, capsys):
    b_id = _make_run()
    c_id = _make_run()
    rc = trg.cmd_triage(_base_args(baseline_run=b_id, candidate_run=c_id, steps="identity", json=True))
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert all(s["kind"].startswith("identity_diff:") for s in doc["steps"])


def test_cmd_triage_rejects_unknown_steps_value(iso):
    b_id = _make_run()
    c_id = _make_run()
    with pytest.raises(CloznError):
        trg.cmd_triage(_base_args(baseline_run=b_id, candidate_run=c_id, steps="not_a_family"))


def test_cmd_triage_writes_out_and_refuses_to_overwrite_without_force(iso, capsys, tmp_path):
    b_id = _make_run()
    c_id = _make_run()
    out_path = str(tmp_path / "triage.json")
    rc = trg.cmd_triage(_base_args(baseline_run=b_id, candidate_run=c_id, out=out_path))
    assert rc == 0
    with open(out_path, encoding="utf-8") as handle:
        doc = json.load(handle)
    assert doc["schema_version"] == "clozn.triage.v1"

    with pytest.raises(CloznError):
        trg.cmd_triage(_base_args(baseline_run=b_id, candidate_run=c_id, out=out_path))

    rc = trg.cmd_triage(_base_args(baseline_run=b_id, candidate_run=c_id, out=out_path, force=True))
    assert rc == 0


def test_cmd_triage_from_experiment_result_end_to_end(tmp_path, capsys):
    path = _write_experiment_result(
        tmp_path, baseline_status="pass", candidate_status="fail",
        baseline_identity={"model_sha256": "a" * 64, "template_fingerprint": "a" * 16},
        candidate_identity={"model_sha256": "a" * 64, "template_fingerprint": "b" * 16})
    rc = trg.cmd_triage(_base_args(experiment_result=path, case="greeting", json=True))
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["source_experiment_id"] == "exp_triage_cli"
    assert doc["case_id"] == "greeting"
    assert "identity_diff:template" in doc["summary"]["observed"]
    assert "identity_diff:model" in doc["summary"]["eliminated"]


def test_cmd_triage_plan_is_zero_run_and_embeds_controlled_artifact(iso, capsys):
    b_id = _make_run(
        messages=[{"role": "user", "content": "full"}], response="good",
        meta={"sampling": "greedy", "temperature": 0.0})
    c_id = _make_run(
        messages=[{"role": "user", "content": "short"}], response="bad",
        meta={"sampling": "greedy", "temperature": 0.0})
    rc = trg.cmd_triage(_base_args(
        baseline_run=b_id, candidate_run=c_id, plan=True, dry_run=False,
        match="exact_output", port=0, json=True,
    ))
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["execution_status"] == "planned"
    assert doc["controlled_tests"]["budget"]["runs_used"] == 0
    context = next(s for s in doc["steps"] if s["kind"] == "context_swap")
    assert context["ran"] is False
    assert context["stop_reason"] == "planned"


def test_format_triage_never_names_a_classification_without_a_causal_step():
    document = {
        "baseline_run_id": "b", "candidate_run_id": "c",
        "steps": [{"kind": "identity_diff:model", "status": "mismatched", "reason": None}],
        "summary": {"classification": "undetermined", "confidence_basis": "observed",
                   "caveats": [], "observed": ["identity_diff:model"], "eliminated": [],
                   "correlated": [], "causally_supported": [], "inconclusive": [], "not_run": []},
    }
    text = trg.format_triage(document)
    assert text.startswith("UNDETERMINED")
    assert "LIKELY" not in text


def test_format_triage_headlines_a_causal_classification():
    document = {
        "baseline_run_id": "b", "candidate_run_id": "c",
        "steps": [{"kind": "template_swap", "status": "causally_supported", "reason": None}],
        "summary": {"classification": "template_swap", "confidence_basis": "causally_supported",
                   "caveats": [], "observed": [], "eliminated": [],
                   "correlated": [], "causally_supported": ["template_swap"],
                   "inconclusive": [], "not_run": []},
    }
    text = trg.format_triage(document)
    assert text.startswith("LIKELY TEMPLATE SWAP")
