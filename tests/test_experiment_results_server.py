"""Focused HTTP wiring tests for GET /experiment-results[...] (clozn/server/routes/experiment_results.py).

Model-free: drives the REAL clozn_server do_GET handler (the object.__new__(H) no-socket trick used by
test_run_diagnosis_server.py / test_experiment_server.py / test_receipts_server.py) against a directory
of hand-built clozn.experiment.result.v0 fixtures on disk. clozn.experiments.suite's own validation is
exhaustively covered elsewhere (test_experiment_suite_cmd.py, clozn/experiments/test_stats.py); this file
only proves the route wiring: list/detail/cells/artifacts shapes, pagination, thin-vs-full cell payloads,
a broken file staying visible rather than vanishing, and that the existing GET /experiments/types drawer
catalog (clozn/server/routes/receipts.py) is untouched by the new /experiment-results namespace.
"""
from __future__ import annotations

import io
import json

import pytest

from clozn.server import app as cs
from clozn.experiments import suite


def _get(path):
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(b"")
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"GET {path} HTTP/1.1", "HTTP/1.1", "GET"
    h.do_GET()
    head, _, body = h.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), json.loads(body)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "results_directory", lambda: str(tmp_path))
    return tmp_path


def _write_result(directory, *, experiment_id, name, created_at, extra_error_cell=False):
    manifest = suite.validate_manifest({
        "schema_version": suite.MANIFEST_SCHEMA, "name": name, "seeds": [0, 1],
        "defaults": {}, "baseline_variant": "base",
        "variants": [{"name": "base", "kind": "base"}, {"name": "cand", "kind": "tuned"}],
        "suites": {
            "target": {"cases": [{"name": "c1", "prompt": "p1"}]},
            "guard": {"cases": [{"name": "g1", "prompt": "p2"}]},
        },
    })
    cells = []
    for variant in manifest["variants"]:
        vname, vkind = variant["name"], variant["kind"]
        for suite_name, case_name in (("target", "c1"), ("guard", "g1")):
            for seed in manifest["seeds"]:
                if extra_error_cell and vname == "cand" and suite_name == "guard" and seed == 1:
                    cells.append(suite._cell_error(suite_name, {"name": case_name}, variant, seed,
                                                    RuntimeError("boom")))
                    continue
                run_id = f"run-{vname}-{suite_name}-{case_name}-{seed}"
                status = "pass" if vname == "cand" else "fail"
                cells.append({
                    "suite": suite_name, "case": case_name, "variant": vname, "variant_kind": vkind,
                    "seed": seed, "status": status, "run_id": run_id, "response": f"reply-{vname}",
                    "assertions": [{"status": status, "check": case_name}], "min_confidence": 0.5,
                    "receipts": {"mode": "regen"}, "error": None,
                    "run": {"id": run_id, "model": "clozn"},
                })
    result = suite.validate_result({
        "schema_version": suite.RESULT_SCHEMA, "experiment_id": experiment_id, "name": name,
        "created_at": created_at, "manifest_sha256": suite._manifest_digest(manifest),
        "manifest": manifest, "seeds": manifest["seeds"], "cells": cells,
        "summary": suite._summarize(cells, "base", ["base", "cand"]),
    })
    path = directory / f"{experiment_id}.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


# --------------------------------------------------------------------------------------- GET /experiment-results

def test_empty_directory_is_an_empty_list_not_an_error(iso):
    head, body = _get("/experiment-results")
    assert "200" in head
    assert body == {"experiments": [], "total": 0, "limit": 50, "offset": 0, "broken": []}


def test_list_returns_thin_entries_newest_first(iso):
    _write_result(iso, experiment_id="exp_older", name="older-run", created_at="2026-07-01T00:00:00Z")
    _write_result(iso, experiment_id="exp_newer", name="newer-run", created_at="2026-07-20T00:00:00Z")

    head, body = _get("/experiment-results")
    assert "200" in head
    assert body["total"] == 2
    ids = [entry["experiment_id"] for entry in body["experiments"]]
    assert ids == ["exp_newer", "exp_older"]              # newest first

    newer = body["experiments"][0]
    assert newer["name"] == "newer-run"
    assert newer["baseline_variant"] == "base"
    assert sorted(newer["variants"]) == ["base", "cand"]
    assert newer["cell_count"] == 8                        # 2 variants x 2 suite/case pairs x 2 seeds
    assert "cand" in newer["aggregates"] and "target" in newer["aggregates"]["cand"]
    assert "cells" not in newer                            # the list entry is thin: no cell payload at all


def test_a_broken_result_file_is_reported_not_dropped(iso):
    _write_result(iso, experiment_id="exp_good", name="good", created_at="2026-07-10T00:00:00Z")
    (iso / "exp_corrupt.json").write_text("{not valid json", encoding="utf-8")

    head, body = _get("/experiment-results")
    assert "200" in head
    assert body["total"] == 1
    assert [e["experiment_id"] for e in body["experiments"]] == ["exp_good"]
    assert len(body["broken"]) == 1
    assert body["broken"][0]["path"].endswith("exp_corrupt.json")
    assert "could not read" in body["broken"][0]["error"] or "JSON" in body["broken"][0]["error"]


def test_pagination_limit_and_offset(iso):
    for i in range(3):
        _write_result(iso, experiment_id=f"exp_{i}", name=f"run-{i}",
                      created_at=f"2026-07-0{i + 1}T00:00:00Z")
    head, body = _get("/experiment-results?limit=1&offset=1")
    assert "200" in head
    assert body["total"] == 3
    assert body["limit"] == 1 and body["offset"] == 1
    assert len(body["experiments"]) == 1
    # newest first: exp_2, exp_1, exp_0 -- offset=1,limit=1 selects exp_1
    assert body["experiments"][0]["experiment_id"] == "exp_1"


@pytest.mark.parametrize("query", ["?limit=0", "?limit=501", "?offset=-1", "?limit=notanumber"])
def test_invalid_pagination_params_are_a_clean_400(iso, query):
    head, body = _get(f"/experiment-results{query}")
    assert "400" in head
    assert "error" in body


# --------------------------------------------------------------------------------- GET /experiment-results/<id>

def test_detail_thins_cells_but_keeps_manifest_and_summary(iso):
    _write_result(iso, experiment_id="exp_detail", name="detail-check", created_at="2026-07-05T00:00:00Z")
    head, body = _get("/experiment-results/exp_detail")
    assert "200" in head
    assert body["experiment_id"] == "exp_detail"
    assert body["manifest"]["name"] == "detail-check"
    assert body["summary"]["baseline_variant"] == "base"
    assert len(body["cells"]) == 8                          # 2 variants x 2 suite/case pairs x 2 seeds
    for cell in body["cells"]:
        assert set(cell) == {"suite", "case", "variant", "variant_kind", "seed", "status", "run_id",
                             "assertions", "min_confidence", "error"}
        assert "response" not in cell and "receipts" not in cell and "run" not in cell


def test_detail_missing_id_is_a_clean_404(iso):
    head, body = _get("/experiment-results/exp_nope")
    assert "404" in head
    assert body == {"error": "no experiment result found for id 'exp_nope'"}


# --------------------------------------------------------------------------------- GET /experiment-results/<id>/cells

def test_cells_endpoint_returns_full_payload_filtered(iso):
    _write_result(iso, experiment_id="exp_cells", name="cells-check", created_at="2026-07-06T00:00:00Z")
    head, body = _get("/experiment-results/exp_cells/cells?variant=cand&suite=target")
    assert "200" in head
    assert body["experiment_id"] == "exp_cells"
    assert len(body["cells"]) == 2                          # target/c1, seeds 0 and 1
    for cell in body["cells"]:
        assert cell["variant"] == "cand" and cell["suite"] == "target"
        assert cell["response"] is not None                  # full payload, unlike the detail route
        assert cell["run"] is not None
        assert cell["receipts"] is not None


def test_cells_endpoint_filters_by_seed(iso):
    _write_result(iso, experiment_id="exp_seed", name="seed-check", created_at="2026-07-07T00:00:00Z")
    head, body = _get("/experiment-results/exp_seed/cells?seed=0")
    assert "200" in head
    assert len(body["cells"]) == 4                           # 2 variants x 2 cases, seed 0 only
    assert all(cell["seed"] == 0 for cell in body["cells"])


def test_cells_endpoint_rejects_non_integer_seed(iso):
    _write_result(iso, experiment_id="exp_bad_seed", name="x", created_at="2026-07-08T00:00:00Z")
    head, body = _get("/experiment-results/exp_bad_seed/cells?seed=abc")
    assert "400" in head
    assert "seed" in body["error"]


def test_cells_endpoint_missing_experiment_is_a_clean_404(iso):
    head, body = _get("/experiment-results/exp_missing/cells")
    assert "404" in head
    assert body == {"error": "no experiment result found for id 'exp_missing'"}


def test_error_status_cell_is_visible_with_null_run(iso):
    _write_result(iso, experiment_id="exp_err", name="err-check", created_at="2026-07-09T00:00:00Z",
                  extra_error_cell=True)
    head, body = _get("/experiment-results/exp_err/cells?variant=cand&suite=guard&seed=1")
    assert "200" in head
    assert len(body["cells"]) == 1
    cell = body["cells"][0]
    assert cell["status"] == "error"
    assert cell["run"] is None and cell["run_id"] is None
    assert cell["error"] == "RuntimeError: boom"


# ----------------------------------------------------------------------------- GET /experiment-results/<id>/artifacts

def test_artifacts_route_is_an_explicit_not_implemented_not_a_silent_pass(iso):
    _write_result(iso, experiment_id="exp_art", name="art-check", created_at="2026-07-11T00:00:00Z")
    head, body = _get("/experiment-results/exp_art/artifacts/whatever.json")
    assert "404" in head
    assert "no artifact bundle" in body["error"]


def test_artifacts_route_missing_experiment_is_still_experiment_not_found(iso):
    head, body = _get("/experiment-results/exp_nope/artifacts/whatever.json")
    assert "404" in head
    assert body == {"error": "no experiment result found for id 'exp_nope'"}


# --------------------------------------------------------------------------- the existing drawer catalog is untouched

def test_experiments_types_drawer_catalog_is_unaffected_by_the_new_namespace(iso):
    """Regression guard for the one explicit constraint on this route family: do not repurpose
    GET /experiments/types (the single-run 'change one thing' drawer catalog, clozn/experiments/
    experiment.py + clozn/server/routes/receipts.py). /experiment-results is a fully distinct top-level
    segment, so this must still answer exactly as it did before this module existed."""
    head, body = _get("/experiments/types")
    assert "200" in head
    assert "types" in body
    assert "ablate_dial" in body["types"]
