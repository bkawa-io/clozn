"""test_mech_target -- clozn/analysis/mech_target.py (`clozn.mechanistic-target.v1`, MECH-CASE-00): the
behavioral-delta target resolver several later mechanistic-analysis surfaces (MECH-CLI-01's
`diff-model --mechanistic` / `experiment explain-cell`, and eventually clozn.analysis.mechanistic_diff
itself) consume instead of scanning a model pair for "interesting" differences.

Model-free / GPU-free throughout: every fixture below is a hand-built dict shaped like a real
`clozn.experiment.result.v0` cell (clozn/experiments/suite.py), a real `clozn.pair-compatibility.v1`
document (clozn/analysis/pair_compatibility.py), or a real `diff_quant_scores()` position entry
(clozn/receipts/quant_receipts.py) -- no engine, no GPU, no file I/O (this module never opens a file).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from clozn import schemas  # noqa: E402
from clozn.analysis import mech_target as mt  # noqa: E402
from clozn.experiments import suite  # noqa: E402


# ==================================================================================== fixture builders

REF_SHA = "a" * 64
CAND_SHA = "b" * 64


def _pair_compat(*, per_token_permitted=True, model_a_sha=REF_SHA, model_b_sha=CAND_SHA):
    model_a = {"filename": "a.gguf"}
    if model_a_sha is not None:
        model_a["sha256"] = model_a_sha
    model_b = {"filename": "b.gguf"}
    if model_b_sha is not None:
        model_b["sha256"] = model_b_sha
    return {
        "schema_version": "clozn.pair-compatibility.v1", "generated_at": "2026-07-29T00:00:00Z",
        "model_a": model_a, "model_b": model_b,
        "tokenizer": {"state": "exact" if per_token_permitted else "differs", "method": "hash"},
        "template": {"state": "same", "method": "hash"},
        "architecture": {"state": "same"}, "layer_count": {"state": "same"},
        "hidden_size": {"state": "same"}, "vocab_size": {"state": "same"},
        "writable_layers": {},
        "verdict": {
            "overall": "compatible" if per_token_permitted else "incompatible", "reasons": [],
            "operations": {
                "per_token_comparison": {"permitted": per_token_permitted,
                                         "reason": "tokenizers match exactly" if per_token_permitted
                                         else "tokenizers differ"},
                "residual_transplant": {"permitted": True, "reason": "hidden_size matches exactly"},
            },
        },
    }


def _run(run_id, response, sha, *, token_ids=None, no_identity=False):
    run = {"id": run_id, "model": "x", "response": response,
          "messages": [{"role": "user", "content": "p"}]}
    if not no_identity:
        run["identity"] = {"model_sha256": sha, "captured_at": "now"}
    if token_ids is not None:
        run["trace"] = {"tokens": list(response), "token_ids": token_ids}
    return run


def _cell(*, suite_name, case, variant, seed, status, run=None, assertions=None, response=""):
    return {"suite": suite_name, "case": case, "variant": variant, "variant_kind": "base", "seed": seed,
           "status": status, "run_id": run.get("id") if run else None, "response": response,
           "assertions": assertions or [], "min_confidence": None, "receipts": None, "error": None,
           "run": run}


def _manifest():
    return suite.validate_manifest({
        "schema_version": suite.MANIFEST_SCHEMA, "name": "mech-target-fixture", "seeds": [0],
        "defaults": {}, "baseline_variant": "base",
        "variants": [{"name": "base", "kind": "base"}, {"name": "candidate", "kind": "tuned"}],
        "suites": {
            "target": {"cases": [{"name": "c1", "prompt": "p1"}]},
            "guard": {"cases": [{"name": "g1", "prompt": "p2"}]},
        },
    })


def _result(cells, *, manifest=None, experiment_id="exp_fixture"):
    manifest = manifest or _manifest()
    return suite.validate_result({
        "schema_version": suite.RESULT_SCHEMA, "experiment_id": experiment_id, "name": manifest["name"],
        "created_at": "2026-07-29T00:00:00Z", "manifest_sha256": suite._manifest_digest(manifest),
        "manifest": manifest, "seeds": [0], "cells": cells,
        "summary": suite._summarize(cells, manifest["baseline_variant"],
                                    [v["name"] for v in manifest["variants"]]),
    })


def _every_cell(cells):
    """Fill in every (suite, case, variant, seed) combination the manifest requires with a bland
    unscored/pass cell, so tests can override only the coordinates they care about and still pass
    validate_result's completeness check."""
    manifest = _manifest()
    have = {(c["suite"], c["case"], c["variant"], c["seed"]) for c in cells}
    out = list(cells)
    for suite_name in ("target", "guard"):
        for case in manifest["suites"][suite_name]["cases"]:
            for variant in manifest["variants"]:
                for seed in manifest["seeds"]:
                    key = (suite_name, case["name"], variant["name"], seed)
                    if key not in have:
                        out.append(_cell(suite_name=suite_name, case=case["name"], variant=variant["name"],
                                         seed=seed, status="unscored",
                                         run=_run(f"run-filler-{'-'.join(map(str, key))}", "", REF_SHA)))
    return out


def _standard_result(*, candidate_status="fail", candidate_sha=CAND_SHA, reference_sha=REF_SHA,
                     candidate_run=None, reference_run=None, assertions=None,
                     candidate_response="abx", reference_response="abc"):
    reference_run = reference_run if reference_run is not None else \
        _run("run-base-target", reference_response, reference_sha, token_ids=list(range(len(reference_response))))
    candidate_run = candidate_run if candidate_run is not None else \
        _run("run-candidate-target", candidate_response, candidate_sha,
            token_ids=list(range(len(candidate_response))))
    assertions = assertions if assertions is not None else [
        {"name": "t", "check": "equals", "target": "text", "expected": reference_response,
         "actual": candidate_response, "status": "fail", "note": None},
    ]
    cells = [
        _cell(suite_name="target", case="c1", variant="base", seed=0, status="pass",
             run=reference_run, response=reference_response),
        _cell(suite_name="target", case="c1", variant="candidate", seed=0, status=candidate_status,
             run=candidate_run, assertions=assertions, response=candidate_response),
    ]
    return _result(_every_cell(cells))


# =============================================================================== resolve_from_experiment_cell

def test_happy_path_builds_a_valid_target_with_token_divergence():
    result = _standard_result()
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is True
    target = outcome["target"]
    schemas.validate(target)   # round-trips through the real schema, not just this test's own opinion
    assert target["schema_version"] == mt.SCHEMA_VERSION
    assert target["origin"] == {
        "kind": "experiment_cell", "experiment_id": "exp_fixture", "suite": "target", "case": "c1",
        "seed": 0, "reference_variant": "base", "candidate_variant": "candidate",
        "reference_run_id": "run-base-target", "candidate_run_id": "run-candidate-target",
    }
    assert target["answer_position"] == {"kind": "token_index", "index": 2}
    assert target["reference_token"] == {"id": 2, "piece": "c"}
    assert target["candidate_token"] == {"id": 2, "piece": "x"}
    assert target["reference_model"]["sha256"] == REF_SHA
    assert target["candidate_model"]["sha256"] == CAND_SHA
    assert len(target["behavioral_delta"]["failed_assertions"]) == 1
    assert target["target_id"].startswith("mechtarget_")


def test_default_reference_variant_is_the_manifest_baseline():
    result = _standard_result()
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is True
    assert outcome["target"]["origin"]["reference_variant"] == "base"


def test_identical_responses_fall_back_to_final_response_answer_position():
    result = _standard_result(candidate_response="abc", reference_response="abc",
                              assertions=[{"name": "t", "check": "min_confidence", "target": None,
                                         "status": "fail", "note": "confidence too low"}])
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is True
    target = outcome["target"]
    assert target["answer_position"]["kind"] == "final_response"
    assert "token-identical" in target["answer_position"]["note"]
    assert "reference_token" not in target
    assert "candidate_token" not in target


# ------------------------------------------------------------------------------------------- refusals

def test_refuses_when_cell_not_found():
    result = _standard_result()
    outcome = mt.resolve_from_experiment_cell(result, case="does-not-exist", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "cell_not_found"


def test_refuses_when_candidate_is_not_failing():
    result = _standard_result(candidate_status="pass")
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "candidate_not_failing"


def test_refuses_when_candidate_has_no_run_error_status():
    # An "error" cell has no run (the generation itself threw) -- select_cells still finds it (it has
    # the right coordinate) but it carries no run at all.
    manifest = _manifest()
    cells = [
        _cell(suite_name="target", case="c1", variant="base", seed=0, status="pass",
             run=_run("run-base-target", "abc", REF_SHA)),
        _cell(suite_name="target", case="c1", variant="candidate", seed=0, status="error", run=None),
    ]
    result = _result(_every_cell(cells), manifest=manifest)
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "candidate_has_no_run"


def test_refuses_when_reference_cell_has_no_run():
    cells = [
        _cell(suite_name="target", case="c1", variant="base", seed=0, status="error", run=None),
        _cell(suite_name="target", case="c1", variant="candidate", seed=0, status="fail",
             run=_run("run-candidate-target", "abx", CAND_SHA),
             assertions=[{"name": "t", "check": "equals", "status": "fail"}]),
    ]
    result = _result(_every_cell(cells))
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "reference_has_no_run"


def test_refuses_when_variant_equals_reference_variant():
    result = _standard_result()
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="base", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "candidate_equals_reference"


def test_refuses_when_per_token_comparison_not_permitted():
    result = _standard_result()
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat(per_token_permitted=False))
    assert outcome["ok"] is False
    assert outcome["reason"] == "per_token_comparison_not_permitted"


def test_refuses_when_run_identity_sha_is_missing():
    result = _standard_result(reference_run=_run("run-base-target", "abc", REF_SHA, no_identity=True))
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "identity_unknown"


def test_refuses_when_run_identity_sha_mismatches_pair_compat():
    # The candidate run actually recorded a DIFFERENT model than pair_compat.model_b claims.
    result = _standard_result(candidate_sha="c" * 64)
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "identity_mismatch"
    assert "c" * 64 in outcome["error"] and CAND_SHA in outcome["error"]


def test_refuses_on_invalid_pair_compat_shape():
    result = _standard_result()
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat="not-a-dict")
    assert outcome["ok"] is False
    assert outcome["reason"] == "invalid_pair_compat"


def test_refuses_on_invalid_suite():
    result = _standard_result()
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              suite="bogus", pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "invalid_suite"


def test_refuses_on_malformed_result():
    outcome = mt.resolve_from_experiment_cell({"not": "a result"}, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "invalid_result"


def test_never_raises_on_an_internal_failure(monkeypatch):
    """A caller's malformed-but-dict-shaped pair_compat (missing model_a/model_b entirely) must not
    crash this resolver even where an internals helper might otherwise raise."""
    def _boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(mt.model_diff, "diff_runs", _boom)
    result = _standard_result()
    outcome = mt.resolve_from_experiment_cell(result, case="c1", variant="candidate", seed=0,
                                              pair_compat=_pair_compat())
    assert outcome["ok"] is False
    assert outcome["reason"] == "internal_error"
    assert "boom" in outcome["error"]


# ======================================================================= resolve_from_diff_model_position

def _position(*, status="flipped", index=7):
    return {"index": index, "token_id": 42, "piece": "the", "logprob_a": -0.1, "logprob_b": -1.2,
           "delta_nats": -1.1, "rank_a": 0, "rank_b": 3,
           "argmax_a_id": 42 if status != "unknown" else None,
           "argmax_a_piece": "the" if status != "unknown" else None,
           "argmax_b_id": 99 if status == "flipped" else (42 if status == "preserved" else None),
           "argmax_b_piece": "a" if status == "flipped" else ("the" if status == "preserved" else None),
           "argmax_flip": status == "flipped" if status != "unknown" else None, "status": status}


def test_diff_model_position_happy_path():
    run = _run("run-abc", "irrelevant", REF_SHA)
    outcome = mt.resolve_from_diff_model_position(run=run, position=_position(), pair_compat=_pair_compat(),
                                                  anchor="reference")
    assert outcome["ok"] is True
    target = outcome["target"]
    schemas.validate(target)
    assert target["origin"] == {"kind": "diff_model_position", "run_id": "run-abc", "anchor": "reference",
                                "position_index": 7, "label_a": "reference", "label_b": "candidate"}
    assert target["answer_position"] == {"kind": "token_index", "index": 7}
    assert target["reference_token"] == {"id": 42, "piece": "the"}
    assert target["candidate_token"] == {"id": 99, "piece": "a"}


def test_diff_model_position_candidate_anchor_checks_model_b():
    run = _run("run-abc", "irrelevant", CAND_SHA)
    outcome = mt.resolve_from_diff_model_position(run=run, position=_position(), pair_compat=_pair_compat(),
                                                  anchor="candidate")
    assert outcome["ok"] is True
    assert outcome["target"]["origin"]["anchor"] == "candidate"


def test_diff_model_position_refuses_on_preserved_status():
    run = _run("run-abc", "irrelevant", REF_SHA)
    outcome = mt.resolve_from_diff_model_position(run=run, position=_position(status="preserved"),
                                                  pair_compat=_pair_compat(), anchor="reference")
    assert outcome["ok"] is False
    assert outcome["reason"] == "position_not_flipped"


def test_diff_model_position_refuses_on_unknown_flip_status():
    run = _run("run-abc", "irrelevant", REF_SHA)
    outcome = mt.resolve_from_diff_model_position(run=run, position=_position(status="unknown"),
                                                  pair_compat=_pair_compat(), anchor="reference")
    assert outcome["ok"] is False
    assert outcome["reason"] == "position_flip_unknown"


def test_diff_model_position_refuses_on_identity_mismatch():
    run = _run("run-abc", "irrelevant", "c" * 64)   # neither model_a nor model_b's sha
    outcome = mt.resolve_from_diff_model_position(run=run, position=_position(), pair_compat=_pair_compat(),
                                                  anchor="reference")
    assert outcome["ok"] is False
    assert outcome["reason"] == "identity_mismatch"


def test_diff_model_position_refuses_on_invalid_anchor():
    run = _run("run-abc", "irrelevant", REF_SHA)
    outcome = mt.resolve_from_diff_model_position(run=run, position=_position(), pair_compat=_pair_compat(),
                                                  anchor="bogus")
    assert outcome["ok"] is False
    assert outcome["reason"] == "invalid_anchor"


def test_diff_model_position_refuses_on_invalid_run():
    outcome = mt.resolve_from_diff_model_position(run={"no": "id"}, position=_position(),
                                                  pair_compat=_pair_compat(), anchor="reference")
    assert outcome["ok"] is False
    assert outcome["reason"] == "invalid_run"


def test_diff_model_position_refuses_on_invalid_position_shape():
    run = _run("run-abc", "irrelevant", REF_SHA)
    outcome = mt.resolve_from_diff_model_position(run=run, position="not-a-dict",
                                                  pair_compat=_pair_compat(), anchor="reference")
    assert outcome["ok"] is False
    assert outcome["reason"] == "invalid_position"


def test_diff_model_position_refuses_when_per_token_comparison_not_permitted():
    run = _run("run-abc", "irrelevant", REF_SHA)
    outcome = mt.resolve_from_diff_model_position(run=run, position=_position(),
                                                  pair_compat=_pair_compat(per_token_permitted=False),
                                                  anchor="reference")
    assert outcome["ok"] is False
    assert outcome["reason"] == "per_token_comparison_not_permitted"


# ==================================================================================== file-addressable IO

def test_targets_directory_and_default_target_path(monkeypatch, tmp_path):
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path) if p == "~/.clozn" else p)
    directory = mt.targets_directory()
    assert directory == str(tmp_path / "mechanistic-targets") or directory.endswith("mechanistic-targets")
    target = {"target_id": "mechtarget_deadbeef00000000"}
    path = mt.default_target_path(target, directory="/tmp/somewhere")
    assert path.endswith(os.path.join("somewhere", "mechtarget_deadbeef00000000.json")) or \
        path == os.path.join("/tmp/somewhere", "mechtarget_deadbeef00000000.json")


def test_default_target_path_generates_an_id_when_missing():
    path = mt.default_target_path({}, directory="/tmp/somewhere")
    assert os.path.basename(path).startswith("mechtarget_")
    assert path.endswith(".json")


# ==================================================================================== no causal language
# Mirrors tests/test_mechanistic_diff.py's own guard exactly -- see clozn/analysis/mechanistic_diff.py.

_BANNED = ("caused", "because", "responsible for", "localiz")   # localize/localized/localization


def test_documents_never_contain_causal_vocabulary():
    for outcome in (
        mt.resolve_from_experiment_cell(_standard_result(), case="c1", variant="candidate", seed=0,
                                        pair_compat=_pair_compat()),
        mt.resolve_from_diff_model_position(run=_run("run-abc", "irrelevant", REF_SHA),
                                            position=_position(), pair_compat=_pair_compat(),
                                            anchor="reference"),
    ):
        assert outcome["ok"] is True
        import json
        text = json.dumps(outcome["target"]).lower()
        for word in _BANNED:
            assert word not in text, f"causal vocabulary {word!r} leaked into the target artifact"


def test_module_source_never_contains_causal_vocabulary():
    path = os.path.join(REPO_ROOT, "clozn", "analysis", "mech_target.py")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    quoted_mentions = ('"caused"', '"because"', '"responsible for"', '"localized"')
    remainder = text
    for mention in quoted_mentions:
        assert mention in remainder, f"expected the module docstring to name {mention} as banned vocabulary"
        remainder = remainder.replace(mention, "", 1)
    lowered = remainder.lower()
    for word in _BANNED:
        assert word not in lowered, f"causal vocabulary {word!r} used outside its one quoted mention"


# ==================================================================================== refusal shape

def test_every_refusal_reason_is_registered():
    """A refusal reason not in REFUSAL_REASONS would slip past `_refuse`'s own assert only if the assert
    were disabled; this is the same invariant checked with -O off, as an ordinary regression test."""
    result = _standard_result()
    for outcome in (
        mt.resolve_from_experiment_cell(result, case="nope", variant="candidate", seed=0,
                                        pair_compat=_pair_compat()),
        mt.resolve_from_experiment_cell(result, case="c1", variant="base", seed=0,
                                        pair_compat=_pair_compat()),
    ):
        assert outcome["reason"] in mt.REFUSAL_REASONS
