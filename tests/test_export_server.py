"""test_export_server -- GET /runs/<id>/export, the one-call run bundle (JSON + Markdown).

The endpoint does zero generation (it reuses M1 explain, a pure read), so like /explain it needs no
substrate. Drives the REAL clozn_server do_GET handler with the no-socket object.__new__(H) trick
(mirrors test_explain_server.py) against an isolated runlog.RUNS_DIR. Also unit-tests _export_markdown
(pure) directly, since the readable receipt is where the new fields -- finish_reason, per-card relevance,
metadata -- have to actually show up.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs   # noqa: E402
import clozn.receipts.bundle as receipt_bundle        # noqa: E402
import clozn.runs.store as runlog                # noqa: E402


def _get(path):
    """Drive do_GET without a socket; return (raw header block, raw body bytes)."""
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(b"")
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"GET {path} HTTP/1.1", "HTTP/1.1", "GET"
    h.do_GET()
    head, _, body = h.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), body


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)                    # export must not need a substrate
    return tmp_path


def _a_run():
    return runlog.record(
        source="engine_chat", model="clozn-qwen (engine)",
        messages=[{"role": "user", "content": "explain gravity"}],
        response="Mass attracts mass.",
        trace={"tokens": ["Mass", " attracts", " mass", "."], "confidence": [0.95, 0.2, 0.9, 0.99],
               "alternatives": [[], [{"piece": " pulls", "prob": 0.4}], [], []],
               "workspace_readouts": [{
                   "type": "workspace_readout", "provider": "engine_concepts",
                   "token_index": 1, "token_text": " attracts", "layer": 15, "position": 1,
                   "top_readouts": [{"label": "force_relation", "score": 0.74}],
                   "entropy": 0.31,
               }]},
        memory={"cards_applied": ["Keep it brief."], "applied_ids": ["c1"], "relevance": [0.81],
                "gate": 0.77, "strength": 1.0, "mode": "prompt"},
        behavior={"active_dials": {"concise": 0.5}},
        finish_reason="length",
        meta={"model_file": "qwen2.5-0.5b-instruct-q4_k_m.gguf", "quant": "Q4_K_M",
              "mode": "autoregressive", "sampler_mode": "greedy", "sampling": "greedy",
              "temperature": 0.0, "repetition_penalty": 1.0, "max_tokens": 64, "seed": 0,
              "n_ctx": 4096, "device": "cuda", "gpu_layers": 99,
              "finish_reason_source": "substrate", "build_git_commit": "abc123"},
    )


# --- the route ---------------------------------------------------------------------------------------------

def test_export_missing_run_is_a_clean_404(iso):
    head, body = _get("/runs/run_nope/export")
    assert "404" in head
    assert json.loads(body) == {"error": "run not found"}


def test_export_json_bundles_run_and_explain(iso):
    rid = _a_run()
    head, body = _get(f"/runs/{rid}/export")
    data = json.loads(body)
    assert data["schema_version"] == "receipt_bundle.v1"
    assert data["run"]["id"] == rid
    assert data["run"]["finish_reason"] == "length"
    assert data["run"]["meta"]["quant"] == "Q4_K_M"
    assert data["repro"]["run_id"] == rid
    assert data["repro"]["temperature"] == 0.0
    assert data["repro"]["max_tokens"] == 64
    assert data["repro"]["meta"]["quant"] == "Q4_K_M"
    assert data["trace"]["tokens"][1] == " attracts"
    assert data["memory"]["cards_applied"] == ["Keep it brief."]
    assert data["explain"]["run_id"] == rid               # M1 explain rides along (the receipts summary)
    assert data["receipts"] is None
    assert data["workspace_readouts"][0]["provider"] == "engine_concepts"
    assert data["workspace_readouts"][0]["provider_type"] == "engine_concepts"
    assert data["workspace_readouts"][0]["readout_kind"] == "concept"
    assert data["workspace_readouts"][0]["run_id"] == rid
    assert data["tiny_tests"] is None
    assert 'filename="' + rid + '.json"' in head          # download disposition


def test_export_markdown_variant_renders_the_receipt(iso):
    rid = _a_run()
    head, body = _get(f"/runs/{rid}/export?format=md")
    assert "text/markdown" in head
    md = body.decode("utf-8")
    assert md.startswith("# Run ")
    assert "explain gravity" in md and "Mass attracts mass." in md
    assert "Keep it brief." in md and "relevance 0.81" in md
    assert "truncated" in md                               # finish_reason == length
    assert "Q4_K_M" in md and "temperature=0.0" in md and "max_tokens=64" in md
    assert "concise: 0.5" in md
    assert "Workspace readouts" in md and "engine_concepts" in md


# --- _export_markdown (pure) -------------------------------------------------------------------------------

def test_markdown_minimal_run_never_crashes():
    md = cs._export_markdown({"id": "r1", "messages": [{"role": "user", "content": "hi"}], "response": "yo"}, None)
    assert "# Run r1" in md and "**user:** hi" in md and "**assistant:** yo" in md


def test_markdown_omits_empty_sections():
    """No memory / no dials / no stop cause -> those sections are simply absent (never empty headers)."""
    md = cs._export_markdown({"id": "r2", "messages": [], "response": "x"}, None)
    assert "Memory applied" not in md and "Behavior dials" not in md and "stop:" not in md


def test_markdown_renders_prompt_mode_assembled_messages():
    md = cs._export_markdown({
        "id": "r3",
        "messages": [{"role": "user", "content": "hi"}],
        "response": "yo",
        "assembled_messages": [{"role": "system", "content": "MEMORY BLOCK"},
                               {"role": "user", "content": "hi"}],
        "memory": {"mode": "prompt", "prompt_block": "MEMORY BLOCK"},
    }, None)
    assert "Assembled prompt/messages" in md
    assert "**system:** MEMORY BLOCK" in md
    assert "Memory-injected section" in md


def test_markdown_internalized_mode_stays_honest_about_soft_prefix():
    md = cs._export_markdown({
        "id": "r4",
        "messages": [{"role": "user", "content": "hi"}],
        "response": "yo",
        "memory": {"mode": "internalized", "has_prefix": True},
    }, None)
    assert "Memory injected as soft prefix; no literal prompt string." in md
    assert "MEMORY BLOCK" not in md


def test_markdown_survives_non_numeric_memory_strength_and_gate():
    """Bug #6 repro: `mem['strength']`/`mem['gate']` were guarded only by `is not None`, unlike
    `relevance` right above (which checks `isinstance(rel, (int, float))` first) -- a non-numeric value
    (e.g. a stray string from a malformed/legacy record) hit `float(mem['strength'])` and raised
    ValueError, taking down the whole GET /runs/<id>/export?format=md response with it."""
    md = cs._export_markdown({
        "id": "r5",
        "messages": [{"role": "user", "content": "hi"}],
        "response": "yo",
        "memory": {"cards_applied": ["Keep it brief."], "strength": "not-a-number", "gate": "also-bad",
                  "mode": "prompt"},
    }, None)
    assert "Keep it brief." in md
    assert "strength" not in md.split("## Memory applied", 1)[1]     # the bad field is simply omitted
    assert "gate" not in md.split("## Memory applied", 1)[1]
    assert "prompt mode" in md                                       # the still-valid sibling field renders


def test_markdown_renders_numeric_memory_strength_and_gate():
    """The happy path stays intact: real numeric strength/gate still render, formatted to 2 decimals."""
    md = cs._export_markdown({
        "id": "r6",
        "messages": [{"role": "user", "content": "hi"}],
        "response": "yo",
        "memory": {"cards_applied": ["Keep it brief."], "strength": 1.0, "gate": 0.77, "mode": "prompt"},
    }, None)
    assert "strength 1.00" in md
    assert "gate 0.77" in md


def test_markdown_is_robust_to_a_non_dict_run():
    assert cs._export_markdown(None, None).startswith("# Run ?")


def test_receipt_bundle_unknowns_are_null_or_empty():
    bundle = receipt_bundle.build({"id": "r1"})
    assert bundle["schema_version"] == "receipt_bundle.v1"
    assert bundle["repro"]["run_id"] == "r1"
    assert bundle["repro"]["temperature"] is None
    assert bundle["trace"] == {}
    assert bundle["memory"] == {}
    assert bundle["receipts"] is None
    assert bundle["workspace_readouts"] is None
    assert bundle["concepts"] is None
    assert bundle["tiny_tests"] is None
    assert bundle["influence_map"] is None


def test_receipt_bundle_omits_local_association_fingerprints():
    bundle = receipt_bundle.build({"id": "r1", "client_key": "client_deadbeef",
                                   "client_key_source": "header", "session_key": "session_deadbeef",
                                   "project_key": "project_deadbeef"})
    assert "client_key" not in bundle["run"]
    assert "client_key_source" not in bundle["run"]
    assert "session_key" not in bundle["run"]
    assert "project_key" not in bundle["run"]


def test_receipt_bundle_keeps_structured_evidence_once_without_widening_private_fields():
    contract = {
        "schema": "clozn.structured_io.v1",
        "mode": "tools",
        "raw_output": "LOCAL STRUCTURED EVIDENCE",
        "outcome": {"status": "parsed", "kind": "tool_call", "tool_name": "weather"},
    }
    source = {
        "id": "r_structured",
        "client_key": "client_deadbeef",
        "client_key_source": "header",
        "session_key": "session_deadbeef",
        "messages": [{"role": "user", "content": "weather?"}],
        "response": "",
        "output_contract": contract,
    }
    bundle = receipt_bundle.build(source)

    assert bundle["run"]["output_contract"] == contract
    assert "output_contract" not in {key for key in bundle if key != "run"}
    assert "client_key" not in bundle["run"]
    assert "client_key_source" not in bundle["run"]
    assert "session_key" not in bundle["run"]
    # The readable receipt does not grow a second raw-output disclosure; the evidence remains available
    # in the explicit JSON run document.
    assert "LOCAL STRUCTURED EVIDENCE" not in receipt_bundle.to_markdown(bundle)


def test_receipt_bundle_drops_malformed_output_contract_without_mutating_source():
    source = {"id": "r_bad_contract", "output_contract": ["bad"]}
    bundle = receipt_bundle.build(source)
    assert bundle["run"]["output_contract"] == {}
    assert source["output_contract"] == ["bad"]


# ---------------------------------------------------------------------------------------------------
# feature 06: the exported markdown states termination and omission/redaction status explicitly
# ---------------------------------------------------------------------------------------------------

def test_markdown_states_termination_and_privacy_for_a_new_shaped_receipt():
    from clozn.runs.context_receipt import build_context_receipt
    receipt = build_context_receipt(
        messages=[{"role": "user", "content": "hi"}], finish_reason="length",
        meta={"max_tokens": 4, "n_ctx": 4096, "prompt_tokens": 10},
        trace={"tokens": ["a", "b", "c", "d"]}, run_id="r_ctx",
    )
    bundle = receipt_bundle.build({"id": "r_ctx", "context_receipt": receipt, "messages": []})
    md = receipt_bundle.to_markdown(bundle)
    assert "## Context receipt" in md
    assert "termination: **max_tokens**" in md
    assert "privacy tier: **full**" in md


def test_markdown_states_content_withheld_by_privacy_tier_not_omitted_by_oversight():
    from clozn.runs.context_receipt import build_context_receipt
    receipt = build_context_receipt(
        messages=[{"role": "user", "content": "hi"}], final_prompt="secret prompt",
        run_id="r_priv", privacy="metadata_only",
    )
    bundle = receipt_bundle.build({"id": "r_priv", "context_receipt": receipt, "messages": []})
    md = receipt_bundle.to_markdown(bundle)
    assert "withheld by receipt privacy tier 'metadata_only'" in md
    assert "secret prompt" not in md


def test_markdown_labels_a_legacy_shaped_receipt_plainly():
    legacy = {
        "schema": "clozn.context_receipt.v1",
        "delivered": {"label": "delivered", "meaning": "m", "messages": []},
        "survived": {"label": "survived", "meaning": "m", "assembled_messages": None,
                    "final_prompt": None},
        "limits": {}, "warnings": [], "input_truncated": False, "input_policy": "p",
        "output_cut_off": False,
    }
    bundle = receipt_bundle.build({"id": "r_legacy", "context_receipt": legacy, "messages": []})
    md = receipt_bundle.to_markdown(bundle)
    assert "pre-2026-07-27 receipt shape" in md


def test_markdown_has_no_context_receipt_section_when_absent():
    bundle = receipt_bundle.build({"id": "r_none", "messages": []})
    assert "## Context receipt" not in receipt_bundle.to_markdown(bundle)


def test_receipt_bundle_preserves_actual_stored_receipts_and_tiny_tests():
    rec = {"influence": {"dial": "concise"}, "has_effect": True}
    tiny = [{"name": "reply_mentions_mass", "status": "pass"}]
    bundle = receipt_bundle.build({"id": "r2", "receipts": {"receipts": [rec]}, "tiny_tests": tiny})
    assert bundle["receipts"]["run_id"] == "r2"
    assert bundle["receipts"]["receipts"] == [rec]
    assert bundle["tiny_tests"] == tiny


def test_receipt_bundle_preserves_precomputed_context_answer_influence_map():
    influence_map = {
        "schema": "clozn.context_answer_influence.v1",
        "run_id": "r_map",
        "prompt_spans": [{"id": "p0", "text": "Use the account context"}],
        "answer_spans": [{"id": "a0", "text": "Your account is active."}],
        "links": [{"prompt_span_id": "p0", "answer_span_id": "a0", "score_nats": 0.8}],
    }
    source = {"id": "r_map", "influence_map": influence_map}

    bundle = receipt_bundle.build(source)

    assert bundle["influence_map"] == influence_map
    assert bundle["influence_map"] is not influence_map
    assert "influence_map" not in bundle["run"]
