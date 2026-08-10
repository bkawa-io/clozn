"""Pure deterministic tests for stable ``sel1`` selection references."""
from __future__ import annotations

import copy
import re

import pytest

from clozn import schemas
from clozn.runs import selection_reference as references
from clozn.runs.text_span_addresses import build_persisted_text_span_addresses, project_influence_addresses

from tests.test_selection_inspection import _influence, _influence_ids, _run


def _source_id(run):
    document = build_persisted_text_span_addresses(run)
    return next(item["address_id"] for item in document["addresses"]
                if item.get("native_ref", {}).get("client_source_id") == "doc-1")


def test_response_token_reference_is_deterministic_url_safe_and_resolves():
    run = _run()
    one = references.encode_selection_reference(run, {"kind": "response_token", "position": 0})
    two = references.encode_selection_reference(run, {"kind": "response_token", "position": 0})

    assert one == two
    assert re.fullmatch(r"sel1\.[A-Za-z0-9_-]+\.[0-9a-f]{24}", one["reference"])
    assert "/" not in one["reference"] and "+" not in one["reference"] and "=" not in one["reference"]
    assert len(one["reference"]) < references.MAX_REFERENCE_CHARS
    assert "Paris" not in one["reference"]
    schemas.validate(one, "clozn.selection-reference.v1")
    decoded = references.decode_selection_reference(one["reference"])
    assert decoded["payload"]["selection"] == {"kind": "response_token", "position": 0}
    resolved = references.resolve_selection_reference(run, one["reference"])
    assert resolved["state"] == "resolved"
    assert resolved["resolved_selection"]["position"] == 0


def test_sampling_defaults_position_and_answer_span_binds_exact_response_hash():
    run = _run()
    sampling = references.encode_selection_reference(run, {"kind": "sampling"})
    assert sampling["selection"] == {"kind": "sampling", "position": 0}
    assert sampling["binding"]["kind"] == "parent_execution"

    answer = references.encode_selection_reference(run, {"kind": "answer_span", "start": 0, "end": 6})
    assert answer["binding"]["kind"] == "recorded_answer"
    changed = copy.deepcopy(run)
    changed["response"] = "London is the capital."
    stale = references.resolve_selection_reference(changed, answer["reference"])
    assert stale["state"] == "stale"
    assert stale["reason"]["code"] in {"recorded_answer_changed", "selection_no_longer_valid"}


def test_context_source_and_exact_relationship_bind_stable_evidence():
    run = _run(influence_map=_influence(_run()))
    source_id, answer_id = _influence_ids(run)
    source = references.encode_selection_reference(run, {
        "kind": "context_span", "source_span_id": source_id,
    })
    relationship = references.encode_selection_reference(run, {
        "kind": "context_span", "source_span_id": source_id, "answer_span_id": answer_id,
    })
    assert source["binding"]["kind"] == "text_span_address"
    assert relationship["binding"]["kind"] == "measured_relationship"
    assert relationship["binding"]["source_span_id"] == source_id
    assert relationship["binding"]["answer_span_id"] == answer_id
    assert "Paris" not in repr(relationship)
    assert references.resolve_selection_reference(run, source["reference"])["state"] == "resolved"
    assert references.resolve_selection_reference(run, relationship["reference"])["state"] == "resolved"

    changed = copy.deepcopy(run)
    changed["influence_map"] = copy.deepcopy(run["influence_map"])
    changed["influence_map"]["links"][0]["delta_nats"] = 9.0
    stale = references.resolve_selection_reference(changed, relationship["reference"])
    assert stale["state"] == "stale"
    assert stale["reason"]["code"] == "influence_artifact_changed"


def test_parent_execution_change_stales_token_reference_even_if_position_remains():
    run = _run()
    reference = references.encode_selection_reference(run, {"kind": "response_token", "position": 0})
    changed = copy.deepcopy(run)
    changed["trace"]["token_ids"][0] = 999
    stale = references.resolve_selection_reference(changed, reference["reference"])
    assert stale["state"] == "stale"
    assert stale["reason"]["code"] == "parent_execution_changed"


@pytest.mark.parametrize("value,code", [
    ("selection.bad.payload", "invalid_reference_prefix"),
    ("sel1.bad.000000000000000000000000", "noncanonical_reference"),
])
def test_malformed_reference_is_caller_error(value, code):
    with pytest.raises(references.SelectionReferenceInputError) as exc:
        references.decode_selection_reference(value)
    assert exc.value.code == code


def test_checksum_mismatch_and_noncanonical_payload_are_rejected():
    run = _run()
    reference = references.encode_selection_reference(run, {"kind": "response_token", "position": 0})["reference"]
    with pytest.raises(references.SelectionReferenceInputError) as exc:
        references.decode_selection_reference(reference[:-1] + ("0" if reference[-1] != "0" else "1"))
    assert exc.value.code == "reference_checksum_mismatch"


def test_invalid_selection_never_gets_encoded():
    with pytest.raises(references.SelectionReferenceInputError) as exc:
        references.encode_selection_reference(_run(), {"kind": "response_token", "position": 999})
    assert exc.value.code == "invalid_position"
