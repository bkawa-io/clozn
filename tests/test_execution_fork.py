"""EngineClient.execution_fork -- the client-side wrapper for POST /v1/execution-fork (exact execution
forks): replay a saved KV checkpoint truncated back to a prior token position, optionally applying ONE
intervention on the forked continuation.

Model-free throughout: EngineClient._post is monkeypatched with a canned/fake reply (no C++ server, no
GPU, no network) -- mirroring test_engine_score.py / test_engine_apply_template.py's own conventions. The
C++ /v1/execution-fork route itself (engine/core/serve/clozn_server.cpp, owned by a different agent as of
this writing) and the live token-for-token exactness proof are validated separately by
scripts/smoke/execution_fork_battery.py against a running clozn-server + real GGUF -- not exercised here.

What this file polices, per the wire contract:
  * the five intervention shapes (none / force_token / sampling / steer / residual_write) are passed
    through verbatim into the request body;
  * an omitted (None) `intervention` is ABSENT from the body -- never sent as an explicit null;
  * a response with absent optional keys (the "omit, never null-pad" wire rule) parses back exactly as
    received -- no invented defaults for missing keys;
  * a `reprefill` response is never mistaken for `live_kv` exactness, even by a naive reader that only
    looks at one of the two fields that jointly decide it (restore_mode, exactness.source);
  * a malformed/out-of-range `truncate_to` (or the other required fields) surfaces as a typed client-side
    error -- BEFORE any request is sent -- and a server-reported error propagates as EngineError rather
    than being retried or smoothed into a degraded return value.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))

from clozn_engine import EngineClient, EngineError   # noqa: E402


def _client(reply=None, *, on_call=None):
    """An EngineClient whose _post is faked: records (path, body) calls and returns `reply` (or invokes
    `on_call(path, body)` when given, for tests that need to raise/inspect)."""
    ec = EngineClient(port=1)
    calls = []

    def fake_post(path, body):
        calls.append((path, body))
        if on_call is not None:
            return on_call(path, body)
        return {} if reply is None else reply

    ec._post = fake_post
    ec._calls = calls
    return ec


# ==================================================================================== request body construction

def test_execution_fork_sends_required_fields_and_path():
    ec = _client()
    ec.execution_fork(checkpoint_id="ckpt-12", truncate_to=445, max_tokens=128)
    path, body = ec._calls[0]
    assert path == "/v1/execution-fork"
    assert body == {
        "checkpoint_id": "ckpt-12", "truncate_to": 445, "max_tokens": 128,
        "checkpoint_on_finish": False,
    }


def test_execution_fork_checkpoint_on_finish_true_is_sent():
    ec = _client()
    ec.execution_fork(checkpoint_id="ckpt-12", truncate_to=445, max_tokens=128, checkpoint_on_finish=True)
    assert ec._calls[0][1]["checkpoint_on_finish"] is True


def test_execution_fork_omits_intervention_when_not_supplied():
    ec = _client()
    ec.execution_fork(checkpoint_id="ckpt-12", truncate_to=445, max_tokens=128)
    body = ec._calls[0][1]
    assert "intervention" not in body   # absent, never a null-padded key


# -- the five wire intervention shapes, passed through verbatim -----------------------------------------

def test_execution_fork_intervention_none():
    ec = _client()
    ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, intervention={"type": "none"})
    assert ec._calls[0][1]["intervention"] == {"type": "none"}


def test_execution_fork_intervention_force_token():
    ec = _client()
    iv = {"type": "force_token", "token_id": 1234}
    ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, intervention=iv)
    assert ec._calls[0][1]["intervention"] == iv


def test_execution_fork_intervention_sampling_any_subset():
    ec = _client()
    # the contract allows any subset of temperature/top_k/top_p/seed/rep_penalty
    iv = {"type": "sampling", "temperature": 0.8, "seed": 7}
    ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, intervention=iv)
    assert ec._calls[0][1]["intervention"] == iv

    ec2 = _client()
    iv_full = {"type": "sampling", "temperature": 0.7, "top_k": 40, "top_p": 0.9,
               "seed": 42, "rep_penalty": 1.1}
    ec2.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, intervention=iv_full)
    assert ec2._calls[0][1]["intervention"] == iv_full


def test_execution_fork_intervention_steer_push():
    ec = _client()
    iv = {"type": "steer", "steer_vec": [0.1, 0.2, 0.3], "steer_layer": 14, "steer_coef": 2.0}
    ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, intervention=iv)
    assert ec._calls[0][1]["intervention"] == iv


def test_execution_fork_intervention_steer_clear():
    ec = _client()
    iv = {"type": "steer", "clear": True}
    ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, intervention=iv)
    assert ec._calls[0][1]["intervention"] == iv


def test_execution_fork_intervention_residual_write():
    ec = _client()
    iv = {"type": "residual_write", "layer": 10, "position": 5, "values": [0.1, 0.2, 0.3]}
    ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, intervention=iv)
    assert ec._calls[0][1]["intervention"] == iv


def test_execution_fork_intervention_is_copied_not_aliased():
    """The body must not hold a live reference to the caller's dict -- mutating the caller's dict after
    the call must not retroactively change what was (already) sent."""
    ec = _client()
    iv = {"type": "force_token", "token_id": 1}
    ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, intervention=iv)
    iv["token_id"] = 999
    assert ec._calls[0][1]["intervention"] == {"type": "force_token", "token_id": 1}


# ==================================================================================== response parsing

def test_execution_fork_returns_the_raw_engine_reply_unmodified():
    canned = {
        "text": "hello", "tokens": [1, 2, 3], "prompt_len": 412, "n_past_restored": 445,
        "restore_mode": "live_kv_truncated",
        "exactness": {"source": "live_kv", "truncation_regime": "generated_token", "boundary_shape_true": True},
        "sampler_source": "checkpoint", "steer_source": "none",
        "intervention_applied": {"type": "none"}, "checkpoint_id": "ckpt-13",
    }
    ec = _client(reply=canned)
    out = ec.execution_fork(checkpoint_id="ckpt-12", truncate_to=445, max_tokens=128)
    assert out == canned


def test_execution_fork_response_with_absent_optional_keys_parses_without_inventing_defaults():
    """A leaner/older worker reply that omits several optional keys must come back exactly as received --
    no KeyError, and critically no invented defaults for the missing keys (they must stay absent, not
    become None/0/""/etc.)."""
    sparse = {"text": "hi", "tokens": [1], "prompt_len": 5, "n_past_restored": 5,
              "restore_mode": "reprefill"}
    ec = _client(reply=sparse)
    out = ec.execution_fork(checkpoint_id="ckpt-1", truncate_to=5, max_tokens=1)
    assert out == sparse
    for missing in ("exactness", "sampler_source", "steer_source", "intervention_applied", "checkpoint_id"):
        assert missing not in out


# ==================================================================================== the regime rule
# (truncate_to > prompt_len -> live_kv_truncated, exact; truncate_to <= prompt_len -> reprefill, and that
# reply must never be read as live_kv exact.)

def _is_exactness_live_kv(response: dict) -> bool:
    """A reader that decides live-KV exactness the SAFE way: both `restore_mode` and `exactness.source`
    must agree, not just one of them. Mirrors the check scripts/smoke/execution_fork_battery.py performs
    on live replies before trusting a fork as an exact continuation."""
    exactness = response.get("exactness") or {}
    return response.get("restore_mode") == "live_kv_truncated" and exactness.get("source") == "live_kv"


def test_reprefill_response_with_no_exactness_block_is_not_live_kv():
    reprefill = {"text": "x", "tokens": [1], "prompt_len": 5, "n_past_restored": 5,
                 "restore_mode": "reprefill"}
    ec = _client(reply=reprefill)
    out = ec.execution_fork(checkpoint_id="c", truncate_to=5, max_tokens=1)
    assert not _is_exactness_live_kv(out)


def test_reprefill_response_with_explicit_reprefill_source_is_not_live_kv():
    reprefill = {"text": "x", "tokens": [1], "prompt_len": 5, "n_past_restored": 5,
                 "restore_mode": "reprefill", "exactness": {"source": "reprefill"}}
    ec = _client(reply=reprefill)
    out = ec.execution_fork(checkpoint_id="c", truncate_to=5, max_tokens=1)
    assert not _is_exactness_live_kv(out)


def test_a_reprefill_restore_mode_overrides_a_lying_exactness_source():
    """Even a malformed/inconsistent reply (restore_mode says reprefill but exactness.source claims
    live_kv) must not be reported as live_kv exact -- restore_mode is the ground truth for WHICH regime
    ran; exactness.source alone is never sufficient."""
    inconsistent = {"text": "x", "tokens": [1], "prompt_len": 5, "n_past_restored": 5,
                    "restore_mode": "reprefill", "exactness": {"source": "live_kv"}}
    ec = _client(reply=inconsistent)
    out = ec.execution_fork(checkpoint_id="c", truncate_to=5, max_tokens=1)
    assert not _is_exactness_live_kv(out)


def test_live_kv_truncated_response_with_matching_exactness_is_live_kv():
    exact = {"text": "x", "tokens": [1], "prompt_len": 5, "n_past_restored": 445,
             "restore_mode": "live_kv_truncated",
             "exactness": {"source": "live_kv", "truncation_regime": "generated_token",
                           "boundary_shape_true": True}}
    ec = _client(reply=exact)
    out = ec.execution_fork(checkpoint_id="c", truncate_to=445, max_tokens=1)
    assert _is_exactness_live_kv(out)


# ==================================================================================== malformed input -> typed errors

def test_negative_truncate_to_raises_before_any_request():
    ec = _client()
    with pytest.raises(ValueError, match="truncate_to"):
        ec.execution_fork(checkpoint_id="c", truncate_to=-1, max_tokens=1)
    assert ec._calls == []   # never attempted the call -- no silent retry, no degraded send


def test_non_integer_truncate_to_raises():
    ec = _client()
    with pytest.raises(ValueError, match="truncate_to"):
        ec.execution_fork(checkpoint_id="c", truncate_to=3.5, max_tokens=1)
    assert ec._calls == []


def test_boolean_truncate_to_raises():
    """bool is an int subclass in Python; True/False must not sneak past an int check."""
    ec = _client()
    with pytest.raises(ValueError, match="truncate_to"):
        ec.execution_fork(checkpoint_id="c", truncate_to=True, max_tokens=1)
    assert ec._calls == []


def test_zero_or_negative_max_tokens_raises():
    ec = _client()
    with pytest.raises(ValueError, match="max_tokens"):
        ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=0)
    assert ec._calls == []


def test_empty_checkpoint_id_raises():
    ec = _client()
    with pytest.raises(ValueError, match="checkpoint_id"):
        ec.execution_fork(checkpoint_id="", truncate_to=1, max_tokens=1)
    assert ec._calls == []


def test_non_object_intervention_raises():
    ec = _client()
    with pytest.raises(ValueError, match="intervention"):
        ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, intervention="none")
    assert ec._calls == []


def test_non_bool_checkpoint_on_finish_raises():
    ec = _client()
    with pytest.raises(ValueError, match="checkpoint_on_finish"):
        ec.execution_fork(checkpoint_id="c", truncate_to=1, max_tokens=1, checkpoint_on_finish="yes")
    assert ec._calls == []


def test_server_reported_out_of_range_truncate_to_propagates_as_engine_error_not_retried():
    """A truncate_to the client cannot locally know is out of range (beyond the checkpoint's own history)
    is only caught server-side -- must surface as EngineError, exactly once, never retried or swallowed
    into a degraded/default return."""
    def boom(path, body):
        raise EngineError(f"POST {path} -> 400: truncate_to 999999 exceeds checkpoint history")

    ec = _client(on_call=boom)
    with pytest.raises(EngineError, match="truncate_to"):
        ec.execution_fork(checkpoint_id="ckpt-1", truncate_to=999999, max_tokens=1)
    assert len(ec._calls) == 1   # exactly one attempt -- no retry loop
