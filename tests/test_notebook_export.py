"""Model-free tests for the generated ``reproduce.ipynb`` (roadmap Phase 4.5 "Open export").

No jupyter/nbformat dependency is exercised here (nor by the generator) -- these tests check the notebook
is valid plain JSON shaped like nbformat v4, that every code cell is syntactically valid Python
(``compile()``, matching the acceptance criterion literally), and that exactly one cell -- the clearly
labeled, opt-in live-reproduction cell -- is the one place stdlib networking (``urllib``) appears; every
other cell must be committed to running fully offline.
"""
from __future__ import annotations

import json

from clozn.runs import notebook_export


def _manifest(**overrides):
    manifest = {
        "schema": "clozn.export_bundle.v1",
        "run_id": "run_0000000000abc_123456",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "clozn_version": "0.1.0",
        "identity": {"model_sha256": "a" * 64},
        "method": {"name": "teacher_forced_matched_context_replacement", "version": "1"},
        "scope": "messages",
        "engine_url": None,
        "artifacts": [
            {"path": "receipt_bundle.json", "kind": "receipt_bundle", "sha256": "b" * 64, "bytes": 10},
            {"path": "trace.json", "kind": "trace_evidence", "sha256": "c" * 64, "bytes": 20},
        ],
        "honesty": {
            "note": "some honest note",
            "hash_verified_offline": ["receipt_bundle.json", "trace.json"],
            "live_reproduction": {
                "claim": "teacher_forced_sum_logprob_matches_recorded_baseline",
                "requires": ["a reachable clozn engine (POST /score)"],
                "proven_offline": False,
            },
        },
    }
    manifest.update(overrides)
    return manifest


def _receipt(**overrides):
    receipt = {
        "run": {"id": "run_0000000000abc_123456", "model": "qwen2.5-0.5b",
                "messages": [{"role": "user", "content": "hi"}], "response": "hello",
                "final_prompt": "hi", "source": "cli", "client": "sdk", "finish_reason": "stop"},
        "identity": {"model_sha256": "a" * 64},
    }
    receipt.update(overrides)
    return receipt


def _code_sources(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


# ================================================================================================ shape

def test_notebook_is_valid_nbformat_v4_structure():
    notebook = notebook_export.build_reproduction_notebook(_manifest(), _receipt())
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["nbformat_minor"], int)
    assert isinstance(notebook["cells"], list) and notebook["cells"]
    assert "kernelspec" in notebook["metadata"]
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("markdown", "code")
        assert isinstance(cell["source"], list)
        assert all(isinstance(line, str) for line in cell["source"])
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_notebook_is_json_serializable_round_trip():
    notebook = notebook_export.build_reproduction_notebook(_manifest(), _receipt())
    text = json.dumps(notebook)
    reloaded = json.loads(text)
    assert reloaded["nbformat"] == 4
    assert len(reloaded["cells"]) == len(notebook["cells"])


def test_notebook_has_both_markdown_and_code_cells():
    notebook = notebook_export.build_reproduction_notebook(_manifest(), _receipt())
    kinds = {cell["cell_type"] for cell in notebook["cells"]}
    assert kinds == {"markdown", "code"}


# ================================================================================================ compile

def test_every_code_cell_compiles():
    notebook = notebook_export.build_reproduction_notebook(_manifest(), _receipt())
    for source in _code_sources(notebook):
        compile(source, "<cell>", "exec")  # raises SyntaxError on failure -- that IS the assertion


def test_code_cells_compile_without_a_receipt_too():
    """build_reproduction_notebook must not assume `receipt` is present."""
    notebook = notebook_export.build_reproduction_notebook(_manifest(), None)
    for source in _code_sources(notebook):
        compile(source, "<cell>", "exec")


def test_code_cells_compile_with_no_influence_map_or_tensors_in_manifest():
    manifest = _manifest(method=None, scope=None,
                         artifacts=[{"path": "receipt_bundle.json", "kind": "receipt_bundle",
                                    "sha256": "b" * 64, "bytes": 10}])
    notebook = notebook_export.build_reproduction_notebook(manifest, _receipt())
    for source in _code_sources(notebook):
        compile(source, "<cell>", "exec")


# =============================================================================== network isolation

def test_exactly_one_code_cell_touches_the_network():
    notebook = notebook_export.build_reproduction_notebook(_manifest(), _receipt())
    sources = _code_sources(notebook)
    hits = [source for source in sources if "urllib" in source]
    assert len(hits) == 1, "exactly one cell (the opt-in live-reproduction check) may use urllib"


def test_the_one_networking_cell_is_clearly_labeled_optional_in_the_preceding_markdown():
    notebook = notebook_export.build_reproduction_notebook(_manifest(), _receipt())
    cells = notebook["cells"]
    net_index = next(i for i, cell in enumerate(cells)
                     if cell["cell_type"] == "code" and "urllib" in "".join(cell["source"]))
    preceding = "".join(cells[net_index - 1]["source"]) if net_index > 0 else ""
    assert "OPTIONAL" in preceding.upper()


def test_no_other_code_cell_mentions_urllib_or_requests():
    notebook = notebook_export.build_reproduction_notebook(_manifest(), _receipt())
    sources = _code_sources(notebook)
    non_network_sources = [s for s in sources if "urllib" not in s]
    for source in non_network_sources:
        assert "requests" not in source
        assert "socket" not in source


# =========================================================================================== content sanity

def test_manifest_fields_appear_verbatim_in_the_intro_markdown():
    manifest = _manifest()
    notebook = notebook_export.build_reproduction_notebook(manifest, _receipt())
    intro = "".join(notebook["cells"][0]["source"])
    assert manifest["run_id"] in intro
    assert manifest["method"]["name"] in intro
    assert manifest["scope"] in intro


def test_setup_cell_loads_manifest_json_from_bundle_dir():
    notebook = notebook_export.build_reproduction_notebook(_manifest(), _receipt())
    setup_source = _code_sources(notebook)[0]
    assert "manifest.json" in setup_source
    assert "BUNDLE_DIR" in setup_source


def test_verification_cell_reads_sha256_and_compares_against_manifest():
    notebook = notebook_export.build_reproduction_notebook(_manifest(), _receipt())
    verify_source = _code_sources(notebook)[1]
    assert "sha256" in verify_source
    assert "artifact['sha256']" in verify_source or 'artifact["sha256"]' in verify_source
