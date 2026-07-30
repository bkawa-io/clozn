"""Model-free contract tests for stable text-span address projection."""
from __future__ import annotations

import copy
import hashlib

from clozn import schemas
from clozn.runs import text_span_addresses as spans


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _context_run(run_id: str = "run_context", *, parent: str | None = None) -> dict:
    system = "Stay concise."
    source = "Résumé 🌎"
    rendered = f"<system>{system}</system><user>{source}</user>"
    run = {
        "id": run_id,
        "parent_run_id": parent,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": source, "source_id": "doc-1"},
        ],
        "final_prompt": rendered,
        "context_receipt": {
            "schema_version": "clozn.context-receipt.v1",
            "run_id": run_id,
            "privacy": "full",
            "delivered": [
                {
                    "segment_id": "seg_1111111111111111",
                    "source_type": "message",
                    "source_label": "system",
                    "original_order": 0,
                    "content_hash": _sha(system)[:16],
                },
                {
                    "segment_id": "seg_2222222222222222",
                    "source_type": "message",
                    "source_label": "notes",
                    "client_source_id": "doc-1",
                    "original_order": 1,
                    "content_hash": _sha(source)[:16],
                },
            ],
            "rendered": {"sha256": _sha(rendered), "bytes": len(rendered.encode("utf-8"))},
        },
    }
    return run


def _metadata_influence() -> dict:
    prompt = "Alpha beta"
    answer = "yes"
    return {
        "schema_version": "clozn.context-answer-influence-export.v1",
        "source_schema": "clozn.context_answer_influence.v1",
        "source_artifact_sha256": "f" * 64,
        "privacy": "metadata_only",
        "status": "ok",
        "available": True,
        "method": {
            "name": "context_answer_influence",
            "version": "1",
            "mode": "forced_score_intervention",
            "claim_limit": "this exact intervention only",
            "caveat": "does not prove correctness or reveal hidden reasoning",
        },
        "identity": {},
        "cache_identity": {},
        "thresholds": {"cell_abs_delta_nats": 0.05},
        "prompt_sources": [{
            "id": "p.m000",
            "source_kind": "assembled_message",
            "segment_id": "seg_3333333333333333",
            "client_source_id": "doc-alpha",
            "source_label": "alpha notes",
            "selected": True,
            "start": 0,
            "end": len(prompt),
            "text_sha256": _sha(prompt),
            "text_bytes": len(prompt.encode("utf-8")),
        }],
        "selection": {
            "selected_source_ids": ["p.m000"],
            "omitted_source_ids": [],
        },
        "prompt_spans": [{
            "id": "p.m000.c000",
            "parent_id": "p.m000",
            "level": "coarse",
            "segment_id": "seg_3333333333333333",
            "client_source_id": "doc-alpha",
            "start": 0,
            "end": 5,
            "text_sha256": _sha("Alpha"),
            "text_bytes": 5,
        }],
        "answer_spans": [{
            "id": "a.t0000",
            "level": "token",
            "token_index": 0,
            "start": 0,
            "end": 3,
            "text_sha256": _sha(answer),
            "text_bytes": 3,
        }],
        "answer_hashes": {
            "scored_text": {
                "text_sha256": _sha(answer),
                "text_bytes": 3,
            },
        },
        "links": [{
            "context_span_id": "p.m000.c000",
            "answer_span_id": "a.t0000",
            "delta_nats": 0.2,
            "evidence_state": "causally_supported",
        }],
        "summary": {},
    }


def _canonical(address: dict) -> dict:
    return address["resolution"]["canonical"]


def test_context_projection_has_exact_codepoint_offsets_and_full_hashes():
    run = _context_run()
    document = spans.build_text_span_addresses(run, privacy="full")
    schemas.validate(document)

    assert [item["kind"] for item in document["addresses"]] == [
        "delivered_message", "attached_source_span", "rendered_prompt_segment",
    ]
    source_address = document["addresses"][1]
    canonical = _canonical(source_address)
    assert canonical["start"] == 0
    assert canonical["end"] == len("Résumé 🌎")
    assert canonical["span_code_points"] == len("Résumé 🌎")
    assert canonical["span_utf8_bytes"] == len("Résumé 🌎".encode("utf-8"))
    assert canonical["basis_sha256"] == _sha("Résumé 🌎")
    assert canonical["span_sha256"] == _sha("Résumé 🌎")
    assert canonical["text"] == "Résumé 🌎"
    assert source_address["native_ref"]["client_source_id"] == "doc-1"
    assert source_address["resolution"]["state"] == "exact"


def test_metadata_projection_never_carries_text_and_keeps_the_same_address_ids():
    run = _context_run()
    full = spans.build_text_span_addresses(run, privacy="full")
    metadata = spans.build_text_span_addresses(run, privacy="metadata_only")

    assert [item["address_id"] for item in full["addresses"]] == [
        item["address_id"] for item in metadata["addresses"]
    ]
    assert all(item["resolution"]["state"] == "metadata_only" for item in metadata["addresses"])
    assert all("text" not in _canonical(item) for item in metadata["addresses"])
    assert "Résumé 🌎" not in str(metadata)
    schemas.validate(metadata)


def test_context_projection_marks_recorded_hash_drift_instead_of_rewriting_it():
    run = _context_run()
    run["context_receipt"]["delivered"][0]["content_hash"] = "0" * 16
    address = spans.project_context_addresses(run, privacy="full")[0]

    assert address["resolution"]["state"] == "drifted"
    assert address["resolution"]["reason"] == "native_content_hash_mismatch"
    # Even a full projection does not disclose text whose native receipt hash
    # is in dispute. Drift is always a closed, metadata-only canonical shape.
    assert "text" not in _canonical(address)
    assert address["native_ref"]["recorded_hash"] == {
        "algorithm": "sha256_truncated_64bit",
        "value": "0" * 16,
        "scope": "canonical_basis",
    }


def test_redacted_context_keeps_native_reference_but_invents_no_offsets():
    run = _context_run()
    run["messages"] = []
    run["final_prompt"] = None
    run["redaction"] = {"status": "redacted"}
    run["flags"] = ["redacted"]
    addresses = spans.project_context_addresses(run)

    assert len(addresses) == 3
    assert all(address["resolution"]["state"] == "redacted" for address in addresses)
    assert all("canonical" not in address["resolution"] for address in addresses)
    assert addresses[0]["resolution"]["reason"] == "source_text_redacted"
    assert addresses[0]["native_ref"]["segment_id"] == "seg_1111111111111111"


def test_legacy_run_messages_are_projected_without_mutating_the_run():
    run = {
        "id": "run_legacy",
        "messages": [{"role": "user", "content": "legacy prompt"}],
        "context_receipt": {
            "schema": "clozn.context_receipt.v1",
            "delivered": {"messages": [{"role": "user", "content": "legacy prompt"}]},
        },
    }
    before = copy.deepcopy(run)
    document = spans.build_text_span_addresses(run)

    assert run == before
    assert document["addresses"][0]["native_ref"]["collection"] == "run.messages"
    assert _canonical(document["addresses"][0])["span_sha256"] == _sha("legacy prompt")


def test_metadata_only_influence_projection_preserves_native_method_and_all_span_kinds():
    influence = _metadata_influence()
    document = spans.build_text_span_addresses(
        {"id": "run_influence", "messages": []},
        influence=influence,
        privacy="metadata_only",
    )
    influence_source = document["source_artifacts"][1]
    projected = [
        item for item in document["addresses"]
        if item["native_ref"]["artifact_schema"] == influence["schema_version"]
    ]

    assert influence_source["native_status"] == "ok"
    assert influence_source["available"] is True
    assert influence_source["method"] == influence["method"]
    assert [item["native_ref"]["collection"] for item in projected] == [
        "influence.prompt_sources",
        "influence.prompt_spans",
        "influence.answer_spans",
    ]
    assert [item["kind"] for item in projected] == [
        "attached_source_span", "attached_source_span", "answer_span",
    ]
    assert all(item["resolution"]["state"] == "metadata_only" for item in projected)
    assert all("text" not in _canonical(item) for item in projected)
    # Numeric links and thresholds remain in the native artifact. This address
    # projection neither copies them nor invents a cross-method score.
    assert "links" not in document and "thresholds" not in document


def test_generic_claim_address_only_addresses_a_caller_supplied_span():
    answer = "Paris is in France."
    address = spans.make_text_span_address(
        run_id="run_claim",
        kind="claim",
        native_ref={
            "artifact_schema": "clozn.claims.v1",
            "collection": "derived.claims",
            "id": "claim-0",
        },
        relation_anchor={"claim_id": "claim-0"},
        basis="recorded_answer",
        start=0,
        end=18,
        basis_text=answer,
        privacy="full",
    )

    assert address["kind"] == "claim"
    assert _canonical(address)["text"] == "Paris is in France"
    assert _canonical(address)["span_sha256"] == _sha("Paris is in France")


def test_lineage_maps_only_exact_hash_and_offset_inheritance():
    parent_run = {
        "id": "run_parent",
        "messages": [{"role": "user", "content": "same"}],
    }
    parent = spans.build_text_span_addresses(parent_run)
    unchanged = spans.build_text_span_addresses(
        {
            "id": "run_child_same",
            "parent_run_id": "run_parent",
            "messages": [{"role": "user", "content": "same"}],
        },
        parent_document=parent,
    )
    changed = spans.build_text_span_addresses(
        {
            "id": "run_child_changed",
            "parent_run_id": "run_parent",
            "messages": [{"role": "user", "content": "diff"}],
        },
        parent_document=parent,
    )

    assert unchanged["lineage"]["mappings"] == [{
        "relation_key": parent["addresses"][0]["relation_key"],
        "parent_address_id": parent["addresses"][0]["address_id"],
        "child_address_id": unchanged["addresses"][0]["address_id"],
        "state": "inherited",
        "reason": "exact_text_and_hashes_unchanged",
    }]
    assert changed["lineage"]["mappings"][0]["state"] == "drifted"
    assert changed["lineage"]["mappings"][0]["reason"] == "canonical_basis_hash_changed"


def test_lineage_reports_missing_and_unresolved_children_without_guessing():
    parent = spans.build_text_span_addresses({
        "id": "run_parent",
        "messages": [{"role": "user", "content": "same"}],
        "context_receipt": {
            "schema_version": "clozn.context-receipt.v1",
            "delivered": [{
                "segment_id": "seg_bbbbbbbbbbbbbbbb",
                "source_type": "message",
                "original_order": 0,
                "content_hash": _sha("same")[:16],
            }],
        },
    })
    missing = spans.build_text_span_addresses({
        "id": "run_missing",
        "parent_run_id": "run_parent",
        "messages": [],
    }, parent_document=parent)
    unresolved = spans.build_text_span_addresses({
        "id": "run_redacted",
        "parent_run_id": "run_parent",
        "messages": [],
        "context_receipt": {
            "schema_version": "clozn.context-receipt.v1",
            "delivered": [{
                "segment_id": "seg_aaaaaaaaaaaaaaaa",
                "source_type": "message",
                "original_order": 0,
                "content_hash": "a" * 16,
            }],
        },
        "redaction": {"status": "redacted"},
    }, parent_document=parent)

    assert missing["lineage"]["mappings"][0]["reason"] == "missing_in_child"
    # The native relation anchor is the same original_order position, so the
    # unresolved child is explicitly distinguished from a missing child.
    assert unresolved["lineage"]["mappings"][0]["reason"] == "child_unresolved"
