"""The v1 process exposes only its declared commands, routes, and text request fields."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pytest

from clozn.cli import main as cli
from clozn.server import app
from clozn.server import openai_compat
from clozn.server.routes import health, openai, runs


ROOT = Path(__file__).resolve().parents[1]


def _command_names() -> tuple[str, ...]:
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return tuple(subparsers.choices)


def test_exact_public_cli_boundary():
    assert _command_names() == (
        "run", "serve", "models", "pull", "studio", "ps", "stop", "version", "doctor",
    )


def test_removed_watch_and_mechanistic_helpers_are_not_shipped():
    from clozn.cli import formatting
    from clozn.cli.commands import doctor
    from clozn.runs import store

    assert not hasattr(store, "runs_after")
    assert not hasattr(store, "cursor_for_run")
    assert not hasattr(doctor, "_check_white_box_qualification")
    assert not hasattr(formatting, "_paint_sparkline")
    assert not hasattr(formatting, "_heatmap_lines")


@pytest.mark.parametrize("name", [
    "plan", "smoke", "trace", "inspect", "eval", "experiment", "suite", "snapshot",
])
def test_removed_cli_products_are_not_registered(name):
    assert name not in _command_names()


@pytest.mark.parametrize("flag", [
    "--models-config", "--structured", "--sae", "--adapter", "--no-flash-attn",
])
def test_removed_serve_modes_are_rejected_by_argparse(flag):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "qwen.gguf", flag, "value"])


def test_exact_gateway_module_boundary():
    get_names = [getattr(module, "__name__", "runs_fallback") for module in app._GET_ROUTES]
    post_names = [module.__name__ for module in app._POST_ROUTES]
    assert get_names == [
        "clozn.server.static",
        "clozn.server.routes.health",
        "clozn.server.routes.openai",
        "clozn.server.routes.runs",
        "clozn.server.routes.context_investigation_v1",
        "clozn.server.routes.context_receipt",
        "clozn.server.routes.minimal_context",
        "clozn.server.routes.run_compare",
        "runs_fallback",
    ]
    assert post_names == [
        "clozn.server.routes.openai",
        "clozn.server.routes.context_investigation_v1",
        "clozn.server.routes.engine",
        "clozn.server.routes.minimal_context",
    ]
    assert app._DELETE_ROUTES == [runs]


@pytest.mark.parametrize("path", [
    "/v1/responses", "/api/chat", "/api/generate", "/api/clozn/generate",
    "/capture/tier", "/engine/health",
])
def test_removed_protocol_and_runtime_routes_are_not_claimed(path):
    class Handler:
        pass

    handler = Handler()
    assert openai.try_post(handler, path, {}) is False
    assert health.try_get(handler, path) is False


@pytest.mark.parametrize("field", [
    "tools", "tool_choice", "response_format", "clozn_guard", "clozn_trust", "clozn_task",
])
def test_removed_behavior_fields_fail_visibly(field):
    value = {
        "tools": [],
        "tool_choice": "none",
        "response_format": {"type": "json_object"},
        "clozn_guard": {},
        "clozn_trust": False,
        "clozn_task": "default",
    }[field]
    with pytest.raises(openai_compat.CompatibilityError) as caught:
        openai_compat.normalize_chat_request({
            "model": "qwen",
            "messages": [{"role": "user", "content": "hello"}],
            field: value,
        })
    assert caught.value.param == field
    assert caught.value.code == "unsupported_parameter"


class _Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status = status
        self.body = body


def test_unknown_run_suffix_is_not_treated_as_a_run_id():
    handler = _Handler()
    assert runs.try_get_fallback(handler, "/runs/run_123/timeline") is True
    assert handler.status == 404
    assert handler.body["code"] == "unknown_run_subroute"


def test_v1_run_delete_uses_the_transactional_mutation(monkeypatch):
    from clozn.runs import mutations

    monkeypatch.setattr(mutations, "delete_run", lambda run_id: {
        "ok": True,
        "action": "delete",
        "run_id": run_id,
        "deleted_run_ids": [run_id],
    })
    handler = _Handler()
    assert runs.try_delete(handler, "/runs/run_123") is True
    assert handler.status == 200
    assert handler.body["deleted_run_ids"] == ["run_123"]


def test_private_worker_advertises_only_v1_capabilities():
    fixture = json.loads((ROOT / "protocol/fixtures/handshake.json").read_text(encoding="utf-8"))
    assert fixture["capabilities"] == [
        "streaming", "sampling", "reference_match_arms",
        "persistent_parent_session", "performance_timing", "prompt_prefix_cache",
        "score_prefix_cache",
    ]

    source = (ROOT / "engine/core/serve/server_main.cpp").read_text(encoding="utf-8")
    block = re.search(r"json capabilities\{(.*?)\};", source, re.DOTALL)
    assert block is not None
    removed = {
        "state_stream", "steering", "infill", "revise", "sae", "jlens", "readout",
        "attn_knockout", "lora", "score_arms", "score_variants", "checkpoint",
        "execution_fork", "checkpoint_pin", "time_machine_continuation",
    }
    advertised = set(re.findall(r'\{"(\w+)",', block.group(1)))
    assert advertised.isdisjoint(removed)


def test_private_worker_does_not_link_removed_research_routes_or_assets():
    cmake = (ROOT / "engine/core/CMakeLists.txt").read_text(encoding="utf-8")
    source = (ROOT / "engine/core/serve/server_main.cpp").read_text(encoding="utf-8")
    for stem in ("routes_jlens", "routes_state", "routes_score_variants"):
        assert stem not in cmake
        assert not (ROOT / f"engine/core/serve/{stem}.cpp").exists()
    assert 'svr.Post("/v1/infill"' not in source
    assert "VIZ_HTML" not in source
    assert not (ROOT / "engine/core/serve/viz_html.hpp").exists()
    for removed_option in ("CLOZN_BUILD_SAE", "CLOZN_BUILD_CUDA"):
        assert removed_option not in cmake

    removed_native = (
        "engine/core/serve/checkpoint_codec.hpp",
        "engine/core/serve/checkpoint_store.hpp",
        "engine/core/serve/time_machine_continuation.hpp",
        "engine/core/serve/readout_plane.hpp",
        "engine/core/serve/variant_score_checkpoint_executor.hpp",
        "engine/core/include/clozn/sae.hpp",
        "engine/core/include/clozn/score_variants_protocol.hpp",
        "engine/core/include/clozn/variant_score_plan.hpp",
        "engine/core/include/clozn/variant_score_checkpoint_reuse.hpp",
        "engine/core/src/sae_encoder.cu",
        "engine/core/tools/clozn_ar.cpp",
        "engine/core/tools/clozn_bench.cpp",
        "engine/core/tools/clozn_probe_sweep.cpp",
    )
    assert all(not (ROOT / path).exists() for path in removed_native)


def test_removed_native_variant_scoring_has_no_python_or_protocol_surface():
    client = (ROOT / "engine/client/clozn_engine/__init__.py").read_text(encoding="utf-8")
    substrate = (ROOT / "clozn/server/substrates.py").read_text(encoding="utf-8")
    assert "score_variants" not in client
    assert "score_variants" not in substrate
    for path in (
        "engine/client/clozn_engine/_score_variants_contract.py",
        "protocol/SCORE_VARIANTS.md",
        "protocol/fixtures/score_variants.json",
        "scripts/protocol/generate_score_variants_contract.py",
        "scripts/smoke/score_variants_parity.py",
    ):
        assert not (ROOT / path).exists()


def test_removed_research_products_have_no_shippable_python_modules():
    removed = (
        "engine/client/probe_and_patch.py",
        "clozn/cli/commands/explain.py",
        "clozn/cli/commands/smoke.py",
        "clozn/cli/commands/adopt.py",
        "clozn/cli/commands/context.py",
        "clozn/cli/commands/model_lock.py",
        "clozn/cli/commands/qualification.py",
        "clozn/cli/commands/runs_privacy.py",
        "clozn/cli/commands/setup_engine.py",
        "clozn/cli/commands/watch.py",
        "clozn/cli/throughput_predictor.py",
        "clozn/cli/trace_io.py",
        "clozn/runs/telemetry.py",
        "clozn/runs/sections.py",
        "clozn/runs/lineage.py",
        "tests/test_sections.py",
        "tests/test_runlog_lineage.py",
        "tests/test_receipt_privacy_transactions.py",
        "clozn/models/fetch.py",
        "clozn/models/lockfile.py",
        "clozn/experiments/exact_execution.py",
        "clozn/experiments/generation.py",
        "clozn/experiments/observation_comparison.py",
        "clozn/experiments/reconstructed_execution.py",
        "clozn/experiments/state_ref.py",
        "clozn/experiments/stats.py",
        "clozn/replay/test_span_bridge.py",
        "clozn/analysis/model_diff.py",
        "clozn/analysis/run_diff.py",
        "clozn/analysis/provenance.py",
        "clozn/receipts/coalition.py",
        "clozn/receipts/hook_vocabulary.py",
        "clozn/receipts/intervention_manifest.py",
        "clozn/receipts/forced.py",
        "clozn/server/routes/contracts.py",
        "clozn/server/routes/receipts.py",
        "clozn/server/routes/card.py",
        "clozn/server/routes/diff.py",
        "clozn/server/routes/investigation.py",
        "clozn/server/routes/models.py",
        "clozn/server/routes/ollama.py",
        "clozn/server/routes/performance.py",
        "clozn/server/routes/replay.py",
        "clozn/server/routes/sessions.py",
        "clozn/server/routes/span_addresses.py",
        "clozn/server/card_html.py",
        "clozn/server/ndjson.py",
        "clozn/qualification/__init__.py",
        "clozn/readouts/__init__.py",
        "clozn/readouts/atlas_concepts.py",
        "clozn/readouts/brain.py",
        "clozn/readouts/sae7b.py",
        "clozn/readouts/workspace_lens.py",
        "scripts/bench/batched_receipts.py",
        "scripts/demo/minimal_context_50_to_3.py",
        "scripts/smoke/engine_substrate.py",
        "scripts/smoke/product_runtime.py",
        "tests/clients/package.json",
        "tests/test_aider_client_compat.py",
        "engine/kernels/confidence_select/CMakeLists.txt",
        "engine/kernels/confidence_select/confidence_select.cu",
        "engine/kernels/confidence_select/reference.py",
        "studio/atlas_emergent.json",
        "studio/casting",
        "studio/ext",
        "studio/workspace_lens_trace.jsonl",
        "clozn/schemas/defs/clozn.context-answer-influence-export.v1.json",
        "tests/fixtures/schemas/clozn.context-answer-influence-export.v1",
        "clozn/schemas/defs/clozn.receipt-privacy-transaction.v1.json",
        "tests/fixtures/schemas/clozn.receipt-privacy-transaction.v1",
        "clozn/schemas/defs/clozn.checkpoint-reference.v2.json",
        "clozn/schemas/defs/clozn.experiment-generated-observation.v2.json",
    )
    assert all(not (ROOT / path).exists() for path in removed)
