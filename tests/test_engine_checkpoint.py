"""Model-free EngineClient coverage for restart-safe checkpoint creation."""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))

from clozn_engine import EngineClient, EngineError  # noqa: E402


class StubClient(EngineClient):
    def __init__(self, responses):
        super().__init__(port=1)
        self.calls = []
        self.responses = list(responses)

    def _post(self, path, body):
        self.calls.append((path, body))
        return self.responses.pop(0)


def checkpoint_response(**changes):
    response = {
        "checkpoint_id": "ckpt-generation-a-0",
        "worker_generation_id": "generation-a",
        "n_past": 4,
        "n_tokens": 4,
        "size_bytes": 8192,
    }
    response.update(changes)
    return response


def test_create_checkpoint_minimal_request_and_response_are_exact():
    response = checkpoint_response(additive_new_field={"kept": True})
    client = StubClient([response])
    result = client.create_checkpoint([10, 11, 12, 13])

    assert client.calls == [("/v1/checkpoint", {"tokens": [10, 11, 12, 13]})]
    assert result == response
    assert result["additive_new_field"] == {"kept": True}


def test_checkpoint_payload_hash_is_an_additive_worker_reply_field():
    """The gateway needs the worker-computed hash as a proof in a later closed continuation call."""
    digest = "a" * 64
    client = StubClient([checkpoint_response(payload_sha256=digest)])
    result = client.create_checkpoint([10, 11, 12, 13])
    assert result["payload_sha256"] == digest

    source = open(
        os.path.join(REPO_ROOT, "engine", "core", "serve", "server_main.cpp"), encoding="utf-8"
    ).read()
    checkpoint_block = source[source.index('svr.Post("/v1/checkpoint"'):source.index('svr.Post("/v1/restore"')]
    import_block = source[source.index('svr.Post("/v1/checkpoint/import"'):source.index('svr.Post("/v1/checkpoint/truncate"')]
    assert '{"payload_sha256", payload_sha256}' in checkpoint_block
    assert '{"payload_sha256", actual_hash}' in import_block


def test_create_checkpoint_sends_exact_execution_provenance_and_generation_precondition():
    client = StubClient([checkpoint_response()])
    sampler = {
        "seed": 7,
        "rng_draws": 2,
        "temperature": 0.8,
        "top_k": 40,
        "top_p": 0.9,
        "rep_penalty": 1.1,
    }
    client.create_checkpoint(
        [10, 11, 12, 13],
        n_past=4,
        prefill_to=2,
        steer_vec=[0.1, 0.2],
        steer_coef=1.5,
        steer_layer=3,
        sampler=sampler,
        worker_generation_id="generation-a",
    )
    path, body = client.calls[0]
    assert path == "/v1/checkpoint"
    assert body == {
        "tokens": [10, 11, 12, 13],
        "n_past": 4,
        "prefill_to": 2,
        "steer_vec": pytest.approx([0.1, 0.2]),
        "steer_coef": 1.5,
        "steer_layer": 3,
        "sampler": sampler,
        "worker_generation_id": "generation-a",
    }
    assert body["sampler"] is not sampler


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tokens": []}, "tokens"),
        ({"tokens": [1, True]}, "tokens"),
        ({"tokens": [1, -1]}, "tokens"),
        ({"tokens": [1, 2], "n_past": 3}, "n_past"),
        ({"tokens": [1, 2], "n_past": 1, "prefill_to": 2}, "prefill_to"),
        ({"tokens": [1], "worker_generation_id": ""}, "worker_generation_id"),
        ({"tokens": [1], "sampler": {}}, "sampler"),
        ({"tokens": [1], "sampler": {"mystery": 1}}, "unsupported"),
        ({"tokens": [1], "sampler": {"top_p": 2.0}}, "top_p"),
        ({"tokens": [1], "steer_vec": [float("nan")]}, "steer_vec"),
    ],
)
def test_create_checkpoint_rejects_invalid_request_before_transport(kwargs, message):
    client = StubClient([])
    with pytest.raises(ValueError, match=message):
        client.create_checkpoint(**kwargs)
    assert client.calls == []


@pytest.mark.parametrize(
    "field",
    ["checkpoint_id", "worker_generation_id", "n_past", "n_tokens", "size_bytes"],
)
def test_create_checkpoint_does_not_invent_missing_response_fields(field):
    response = checkpoint_response()
    response.pop(field)
    client = StubClient([response])
    with pytest.raises(EngineError, match=field):
        client.create_checkpoint([10, 11, 12, 13])


def test_create_checkpoint_rejects_worker_restart_between_request_and_response():
    client = StubClient([checkpoint_response(worker_generation_id="generation-b")])
    with pytest.raises(EngineError, match="different worker_generation_id"):
        client.create_checkpoint(
            [10, 11, 12, 13], worker_generation_id="generation-a")


def test_create_checkpoint_rejects_inconsistent_counts():
    client = StubClient([checkpoint_response(n_tokens=99)])
    with pytest.raises(EngineError, match="n_tokens inconsistent"):
        client.create_checkpoint([10, 11, 12, 13])


def test_execution_fork_can_send_compound_reference_without_breaking_legacy_shape():
    client = StubClient([{"worker_generation_id": "generation-a"}])
    client.execution_fork(
        checkpoint_id="ckpt-generation-a-0",
        worker_generation_id="generation-a",
        truncate_to=3,
        max_tokens=1,
    )
    assert client.calls[0][1]["worker_generation_id"] == "generation-a"

    legacy = StubClient([{}])
    legacy.execution_fork(checkpoint_id="ckpt-generation-a-0", truncate_to=3, max_tokens=1)
    assert "worker_generation_id" not in legacy.calls[0][1]


def test_worker_source_uses_one_store_for_every_checkpoint_issuance():
    source = open(
        os.path.join(REPO_ROOT, "engine", "core", "serve", "server_main.cpp"),
        encoding="utf-8",
    ).read()
    assert source.count("make_worker_generation_id()") == 1
    # Checkpoint-producing call sites: completion checkpointing, /v1/checkpoint, /v1/execution-fork's
    # checkpoint_on_finish, ADR-010 continuation's optional final checkpoint, and (FORK-PIN-01)
    # /v1/checkpoint/import + /v1/checkpoint/truncate.
    assert source.count("checkpoints.insert(std::move(") == 6
    assert 'make_id("ckpt-")' not in source
    assert "checkpoints.erase(checkpoints.begin())" not in source
    # Completion checkpointing plus checkpoint/restore/branch/execution-fork/export/truncate all accept
    # the optional process-generation precondition (FORK-PIN-01's /v1/checkpoint/import deliberately
    # does NOT -- see its own docstring: it is importing possibly-foreign state, not referencing an
    # existing checkpoint_id already living in THIS worker's store, so there is no prior generation to
    # precondition against).
    assert source.count("validate_checkpoint_generation(body, res)") == 7
