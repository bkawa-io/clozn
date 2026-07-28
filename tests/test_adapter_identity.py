"""Contract tests for the fine-tune adapter identity facet (clozn/runs/identity_providers/adapter.py)."""
from __future__ import annotations

import pytest

from clozn.runs import identity_ext
from clozn.runs.identity_providers import adapter


@pytest.fixture(autouse=True)
def _isolate():
    identity_ext.reset_cache()
    yield
    identity_ext.reset_cache()


_HEALTH_WITH_ADAPTER = {
    "capabilities": {"lora": True},
    "lora": {
        "path": "C:\\models\\my-finetune.gguf",
        "scale": 1.0,
        "meta": {
            "general.type": "adapter",
            "general.architecture": "qwen2",
            "adapter.type": "lora",
            "adapter.lora.alpha": "8.000000",
        },
    },
}


def test_absent_when_no_adapter_attached():
    """The common case. Must be an absent namespace, not an empty-but-present one."""
    assert adapter.identity({"engine_health": {"capabilities": {"lora": True}}}) == {}


def test_absent_when_engine_predates_adapters():
    assert adapter.identity({"engine_health": {"capabilities": {}}}) == {}


@pytest.mark.parametrize("bad", [None, {}, {"engine_health": None}, {"engine_health": "nope"},
                                 {"engine_health": {"lora": "nope"}}, {"engine_health": {"lora": {}}}])
def test_malformed_input_yields_no_namespace(bad):
    assert adapter.identity(bad) == {}


def test_reports_path_scale_and_promoted_metadata():
    got = adapter.identity({"engine_health": _HEALTH_WITH_ADAPTER})
    assert got["path"] == "C:\\models\\my-finetune.gguf"
    assert got["scale"] == 1.0
    assert got["alpha"] == "8.000000"
    assert got["architecture"] == "qwen2"


def test_raw_meta_is_preserved_not_just_the_promoted_keys():
    """An adapter may declare fields _PROMOTED has not learned about; dropping them would lose identity
    a future reader might need."""
    got = adapter.identity({"engine_health": _HEALTH_WITH_ADAPTER})
    assert got["meta"]["adapter.type"] == "lora"
    assert got["meta"]["general.type"] == "adapter"


def test_scale_zero_is_recorded_not_erased():
    """'Attached, contributing nothing' is a materially different run from 'no adapter' -- it is the
    identity control an experiment uses. A falsy check here would collapse the two."""
    health = {"lora": {"path": "a.gguf", "scale": 0.0}}
    got = adapter.identity({"engine_health": health})
    assert got["scale"] == 0.0
    assert got != {}


def test_a_scale_of_zero_still_differs_from_no_adapter_at_all():
    attached = adapter.identity({"engine_health": {"lora": {"path": "a.gguf", "scale": 0.0}}})
    none = adapter.identity({"engine_health": {}})
    assert attached != none


def test_the_facet_is_discovered_through_the_seam():
    """It must actually be reachable via identity_ext.collect(), not merely importable."""
    got = identity_ext.collect({"engine_health": _HEALTH_WITH_ADAPTER})
    assert got.get("adapter", {}).get("path") == "C:\\models\\my-finetune.gguf"
    assert identity_ext.COLLECT_FAILURES == []


def test_identity_reads_the_engine_not_the_request():
    """The provider must describe what the engine LOADED, never what a caller asked for -- an adapter
    key in extra_context must not be able to manufacture an identity the engine never confirmed."""
    got = adapter.identity({"adapter": "requested-but-not-loaded.gguf", "engine_health": {}})
    assert got == {}
