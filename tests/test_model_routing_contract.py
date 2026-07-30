"""Model-free contract checks for ADR 004 / clozn.model-routing.v1."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from clozn import schemas


SCHEMA = "clozn.model-routing.v1"
FIXTURES = Path(__file__).parent / "fixtures" / "schemas" / SCHEMA

ERROR_MATRIX = {
    "invalid_model_selection": (400, False, ("selection",)),
    "unknown_model": (404, False, ("resolution",)),
    "model_not_ready": (409, True, ("resolution",)),
    "adapter_unavailable": (409, False, ("resolution",)),
    "load_queue_full": (429, True, ("load_queue",)),
    "generation_queue_full": (429, True, ("generation_queue",)),
    "queue_timeout": (504, True, ("load_queue", "generation_queue")),
    "model_load_timeout": (504, True, ("load",)),
    "model_load_failed": (503, True, ("load",)),
    "no_evictable_worker": (503, True, ("eviction",)),
    "request_cancelled": (499, False, ("request",)),
    "worker_failed": (502, True, ("generation",)),
    "worker_identity_mismatch": (502, False, ("handshake",)),
    "capability_unavailable": (422, False, ("capability",)),
}


def _fixture(name: str) -> dict:
    with (FIXTURES / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_lifecycle_and_error_vocabularies_are_closed():
    schema = schemas.load(SCHEMA)
    assert schema["$defs"]["LifecycleState"]["enum"] == [
        "unloaded", "loading", "ready", "evicting", "failed",
    ]
    branches = schema["$defs"]["Error"]["oneOf"]
    declared = {
        branch["properties"]["code"]["const"]: (
            branch["properties"]["http_status"]["const"],
            branch["properties"]["retryable"]["const"],
        )
        for branch in branches
    }
    assert declared == {
        code: (status, retryable)
        for code, (status, retryable, _phases) in ERROR_MATRIX.items()
    }


@pytest.mark.parametrize("code", sorted(ERROR_MATRIX))
def test_every_error_code_has_one_deterministic_status_retryability_and_phase(code):
    status, retryable, phases = ERROR_MATRIX[code]
    document = _fixture("valid__unknown_model_error.json")
    document["result"]["error"] = {
        "code": code,
        "http_status": status,
        "retryable": retryable,
        "phase": phases[0],
        "message": f"{code} contract probe",
    }
    schemas.validate(document)

    wrong_status = deepcopy(document)
    wrong_status["result"]["error"]["http_status"] = status + 1
    with pytest.raises(schemas.ValidationError):
        schemas.validate(wrong_status)

    wrong_retryability = deepcopy(document)
    wrong_retryability["result"]["error"]["retryable"] = not retryable
    with pytest.raises(schemas.ValidationError):
        schemas.validate(wrong_retryability)

    wrong_phase = deepcopy(document)
    wrong_phase["result"]["error"]["phase"] = "not-a-routing-phase"
    with pytest.raises(schemas.ValidationError):
        schemas.validate(wrong_phase)

    for phase in phases:
        alternate = deepcopy(document)
        alternate["result"]["error"]["phase"] = phase
        schemas.validate(alternate)


def test_success_receipt_requires_every_immutable_resolution_facet():
    document = _fixture("valid__explicit_cold_load.json")
    receipt = document["result"]["receipt"]
    required = (
        "requested_model", "resolved_model_id", "resolved_artifact", "runtime_key",
        "worker_identity", "adapter", "load_event",
    )
    for field in required:
        missing = deepcopy(document)
        del missing["result"]["receipt"][field]
        with pytest.raises(schemas.ValidationError):
            schemas.validate(missing)

    # Runtime-key facets are closed: a future behavior-bearing launch option cannot be silently omitted
    # from canonicalization by riding as an unknown sibling.
    unkeyed = deepcopy(document)
    unkeyed["result"]["receipt"]["runtime_key"]["unkeyed_behavior_flag"] = True
    with pytest.raises(schemas.ValidationError):
        schemas.validate(unkeyed)

    assert receipt["requested_model"] == "qwen-work"
    assert receipt["resolved_model_id"] == "qwen-2.5-7b-q5"


def test_omitted_model_and_unknown_explicit_model_remain_distinct():
    omitted = _fixture("valid__omitted_model_uses_ready_default.json")
    schemas.validate(omitted)
    assert omitted["request"]["requested_model"] is None
    assert omitted["result"]["receipt"]["requested_model"] is None
    assert omitted["result"]["receipt"]["resolved_model_id"] == omitted["policy"]["default_model_id"]

    unknown = _fixture("valid__unknown_model_error.json")
    schemas.validate(unknown)
    assert unknown["request"]["requested_model"] == "not-configured"
    assert unknown["result"]["error"]["code"] == "unknown_model"
    assert "resolved_model_id" not in unknown["result"]["receipt"]


def test_protocol_surface_cannot_claim_another_surfaces_route():
    document = _fixture("valid__explicit_cold_load.json")
    document["protocol"] = {
        "surface": "native",
        "route": "/v1/chat/completions",
    }
    with pytest.raises(schemas.ValidationError):
        schemas.validate(document)
