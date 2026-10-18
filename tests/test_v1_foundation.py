"""Small v1 foundation contracts: isolated state and exact runtime identity."""
from __future__ import annotations

import importlib

from clozn.experiments.execution_facts import runtime_projection
from clozn.paths import DEFAULT_DATA_ROOT, data_path, data_root
from clozn.runs import store as run_store


def test_v1_data_root_defaults_to_an_isolated_directory(monkeypatch):
    monkeypatch.delenv("CLOZN_DATA_ROOT", raising=False)
    assert data_root() == DEFAULT_DATA_ROOT


def test_v1_data_root_accepts_an_explicit_test_root(monkeypatch, tmp_path):
    monkeypatch.setenv("CLOZN_DATA_ROOT", str(tmp_path / "clozn-data"))
    assert data_root() == str(tmp_path / "clozn-data")


def test_run_store_defaults_below_the_v1_data_root(monkeypatch):
    monkeypatch.delenv("CLOZN_DATA_ROOT", raising=False)
    importlib.reload(run_store)
    assert run_store.RUNS_DIR == data_path("runs")


def test_runtime_projection_accepts_build_label_plus_exact_engine_artifact():
    projected = runtime_projection({
        "model_sha256": "a" * 64,
        "template_fingerprint": "b" * 16,
        "engine_build": "clozn-server 1.2.3",
        "context_size": 4096,
        "backend": "cpu",
        "ext": {"engine_artifact": {"artifact_sha256": "c" * 64}},
    })
    assert projected is not None
    assert projected["engine_build"] == f"sha256:{'c' * 64}"


def test_runtime_projection_rejects_conflicting_explicit_engine_digests():
    value = {
        "model_sha256": "a" * 64,
        "template_fingerprint": "b" * 16,
        "engine_build": f"sha256:{'d' * 64}",
        "context_size": 4096,
        "backend": "cpu",
        "ext": {"engine_artifact": {"artifact_sha256": "c" * 64}},
    }
    assert runtime_projection(value) is None
