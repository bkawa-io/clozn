"""Model-free wire contract for EngineClient.time_machine_continue.

The private worker route is deliberately closed: this test makes accidental forwarding of an
unproved sampler/template/adapter override impossible to miss without a live model.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))

from clozn_engine import EngineClient, EngineError  # noqa: E402


SHA = "a" * 64


def _client(reply=None):
    client = EngineClient(port=1)
    client.calls = []

    def fake_post(path, body):
        client.calls.append((path, body))
        return {} if reply is None else reply

    client._post = fake_post
    return client


def _call(client, **changes):
    values = {
        "checkpoint_id": "ckpt-generation-a-0",
        "worker_generation_id": "generation-a",
        "expected_n_past": 3,
        "expected_token_history_sha256": SHA,
        "expected_checkpoint_payload_sha256": "b" * 64,
        "append_token_ids": [9, 10],
        "append_token_ids_sha256": "c" * 64,
        "max_tokens": 16,
        "request_id": "tmc-request-a",
    }
    values.update(changes)
    return client.time_machine_continue(**values)


def test_time_machine_continue_posts_exact_closed_request_and_keeps_reply_raw():
    reply = {"status": "completed", "tokens": [77], "token_pieces": [" answer"], "future": True}
    client = _client(reply)

    result = _call(client)

    assert result is reply
    assert client.calls == [("/v1/time-machine/continue", {
        "checkpoint_id": "ckpt-generation-a-0",
        "worker_generation_id": "generation-a",
        "expected_n_past": 3,
        "expected_token_history_sha256": SHA,
        "expected_checkpoint_payload_sha256": "b" * 64,
        "append_token_ids": [9, 10],
        "append_token_ids_sha256": "c" * 64,
        "max_tokens": 16,
        "request_id": "tmc-request-a",
    })]


def test_time_machine_continue_only_includes_optional_checkpoint_flag_when_explicit():
    client = _client()
    _call(client, checkpoint_on_finish=True)

    assert client.calls[0][1]["checkpoint_on_finish"] is True


def test_time_machine_continue_preserves_typed_non_2xx_terminal_reply():
    client = EngineClient(port=1)
    reply = {
        "status": "unavailable",
        "code": "worker_generation_stale",
        "request_id": "tmc-request-a",
    }

    def fail(_path, _body):
        raise EngineError("POST failed", response=reply)

    client._post = fail

    assert _call(client) == reply


def test_time_machine_continue_does_not_swallow_untyped_transport_error():
    client = EngineClient(port=1)

    def fail(_path, _body):
        raise EngineError("connection failed")

    client._post = fail

    with pytest.raises(EngineError, match="connection failed"):
        _call(client)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"checkpoint_id": ""}, "checkpoint_id"),
        ({"expected_n_past": 0}, "expected_n_past"),
        ({"expected_token_history_sha256": "A" * 64}, "expected_token_history_sha256"),
        ({"append_token_ids": []}, "append_token_ids"),
        ({"append_token_ids": [1, True]}, "append_token_ids"),
        ({"max_tokens": False}, "max_tokens"),
        ({"checkpoint_on_finish": 1}, "checkpoint_on_finish"),
    ],
)
def test_time_machine_continue_rejects_bad_closed_fields_before_transport(changes, message):
    client = _client()
    with pytest.raises(ValueError, match=message):
        _call(client, **changes)
    assert client.calls == []
