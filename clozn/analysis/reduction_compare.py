"""Narrow comparison contract for a saved Minimal Context child.

This module intentionally does not provide arbitrary run-to-run comparison.
It accepts only a direct child produced by a delete-source intervention and
projects the evidence the v1 product needs: retained/removed context, rendered
prompt-token counts, exact answer/termination preservation, and generation
identity parity.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


SCHEMA_VERSION = "clozn.reduction-comparison.v1"

_IDENTITY_FIELDS = (
    "model_path",
    "model_sha256",
    "template_fingerprint",
    "engine_build",
    "clozn_version",
)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string_ids(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(dict.fromkeys(item for item in value if isinstance(item, str) and item))


def _prompt_tokens(run: Mapping[str, Any]) -> int | None:
    receipt = _mapping(run.get("context_receipt"))
    rendered = _mapping(receipt.get("rendered"))
    value = rendered.get("tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _context_ids(run: Mapping[str, Any]) -> list[str]:
    manifest = _mapping(run.get("context_units"))
    return _string_ids(manifest.get("default_source_ids"))


def _identity(run: Mapping[str, Any]) -> dict[str, Any]:
    recorded = _mapping(run.get("identity"))
    return {field: deepcopy(recorded[field]) for field in _IDENTITY_FIELDS if field in recorded}


def compare_reduction(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Compare one parent with its direct, source-reduced child.

    Invalid or unrelated pairs return a small typed error document instead of
    being treated as a general comparison request.
    """
    parent_id = parent.get("id") if isinstance(parent.get("id"), str) else ""
    child_id = child.get("id") if isinstance(child.get("id"), str) else ""
    if not parent_id or not child_id:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "error": {"code": "invalid_run", "message": "both runs need non-empty ids"},
        }
    if child.get("parent_run_id") != parent_id:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "parent_run_id": parent_id,
            "child_run_id": child_id,
            "error": {
                "code": "not_reduction_child",
                "message": "candidate run is not a direct child of the requested parent",
            },
        }

    experiment = _mapping(_mapping(child.get("changes_applied")).get("experiment"))
    intervention = _mapping(experiment.get("intervention"))
    if intervention.get("kind") != "delete_source":
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "parent_run_id": parent_id,
            "child_run_id": child_id,
            "error": {
                "code": "not_reduction_child",
                "message": "candidate child has no delete-source intervention",
            },
        }

    removed = _string_ids(intervention.get("source_ids"))
    provenance = _mapping(_mapping(experiment.get("derived_provenance")).get("minimal_context"))
    retained = _string_ids(provenance.get("retained_source_ids"))
    if not retained:
        retained = _context_ids(child)
    parent_sources = _context_ids(parent)
    if not retained and parent_sources:
        removed_set = set(removed)
        retained = [source_id for source_id in parent_sources if source_id not in removed_set]

    parent_tokens = _prompt_tokens(parent)
    child_tokens = _prompt_tokens(child)
    saved_tokens = (
        parent_tokens - child_tokens
        if parent_tokens is not None and child_tokens is not None else None
    )
    parent_identity = _identity(parent)
    child_identity = _identity(child)
    answer_match = parent.get("response") == child.get("response")
    finish_reason_match = parent.get("finish_reason") == child.get("finish_reason")

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "parent_run_id": parent_id,
        "child_run_id": child_id,
        "context": {
            "parent_source_ids": parent_sources,
            "retained_source_ids": retained,
            "removed_source_ids": removed,
        },
        "rendered_prompt_tokens": {
            "parent": parent_tokens,
            "child": child_tokens,
            "saved": saved_tokens,
            "improved": saved_tokens is not None and saved_tokens > 0,
        },
        "preservation": {
            "answer_exact_match": answer_match,
            "finish_reason_match": finish_reason_match,
            "exact": answer_match and finish_reason_match,
        },
        "generation_identity": {
            "parent": parent_identity,
            "child": child_identity,
            "match": parent_identity == child_identity,
        },
    }


__all__ = ["SCHEMA_VERSION", "compare_reduction"]
