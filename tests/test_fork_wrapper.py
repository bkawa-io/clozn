"""FORK-02: POST /runs/<id>/fork's compatibility wrapper (clozn.replay.fork.compat_fork), model-free.

The exact execution-fork primitives (checkpoint capture -> plan -> execute) already reach the gateway
via the dedicated /runs/<id>/execution-fork* routes and are proven bit-exact against a real GGUF
elsewhere (scripts/smoke/execution_fork_battery.py, referenced by tests/test_execution_fork.py's own
docstring). Nothing called them from the "fork" button: POST /runs/<id>/fork always ran the legacy
text-splice (clozn.replay.fork.fork), even when an exact snapshot could have been used. This file
proves the WRAPPER's own orchestration/labeling decisions -- capture-then-plan-then-execute, degrade
only when honestly ineligible, never silently prefer the splice -- using the SAME fake-engine
conventions tests/test_execution_fork_gateway.py and tests/test_checkpoint_capture.py already
established for those primitives. It does not re-prove worker-side exactness itself; that would require
a live clozn-server + real GGUF (see the skipped test at the bottom for that follow-up).

Covers:
  * an eligible run gets `exact_execution_fork` -- never the splice -- and the response carries the
    exact path's own restore-mode/regime/unchanged-control facts verbatim (no renamed vocabulary).
  * an ineligible run (no `meta.prompt_tokens` -- a realistic "historical run" shape, and a worker-side
    checkpoint-capture failure) degrades to a LABELED `reconstructed_replay`, never silently.
  * a captured, planned, exact-eligible fork whose EXECUTION genuinely diverges is reported
    `unavailable` -- not masked behind a plausible-looking splice, because the checkpoint really did
    exist and really did fail, which is not "no exact state".
  * a forced token with no recorded numeric id (free text) can never be exact by construction, and is
    routed straight to reconstruction without wasting a checkpoint-capture call.
  * neither path eligible (no final_prompt at all) -> a typed `unavailable`, never a silent empty
    result, and no worker call is wasted on either path.
  * the pre-FORK-02 request shape and success-response fields (`position`/`token`/`token_id` in,
    `prefix_kept`/`forked_from_piece`/`retokenized`/`note`/`changes_applied` out) are unchanged --
    `outcome`/`reasons` are additive, never a new envelope.
"""
from __future__ import annotations

import pytest

import clozn.runs.store as runlog
from clozn.replay import execution_fork_results
from clozn.server.routes import fork as fork_routes


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "test-build",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
    "white_box_flags": {},
}
WORKER = {
    "worker_id": "generation-a",
    "worker_generation_id": "generation-a",
    "protocol_version": "1.1",
}


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    return tmp_path


def _eligible_parent(**overrides):
    """A parent run that is eligible for BOTH regimes: exact (complete meta.prompt_tokens/stream/decode
    provenance, matching identity) and reconstruction (final_prompt + complete trace). Individual tests
    knock out one or the other via `overrides` to reach a specific outcome."""
    values = {
        "source": "engine_chat",
        "client": "studio",
        "model": "fixture-model",
        "substrate": "engine",
        "messages": [{"role": "user", "content": "count"}],
        "assembled_messages": [{"role": "user", "content": "count"}],
        "response": "one two three",
        "final_prompt": "<prompt>",
        "trace": {
            "tokens": ["one", " two", " three"],
            "token_ids": [11, 22, 33],
            "alternatives": [[], [{"piece": " four", "token_id": 44, "prob": 0.02}], []],
        },
        "behavior": {"active_dials": {}},
        "meta": {
            "n_ctx": RUNTIME["context_size"],
            "device": RUNTIME["backend"],
            "prompt_tokens": 2,
            "stream": False,
            "decode": {"mode": "greedy", "temperature": 0.0, "seed": 0},
        },
        "identity": {
            "model_sha256": RUNTIME["model_sha256"],
            "template_fingerprint": RUNTIME["template_fingerprint"],
            "engine_build": RUNTIME["engine_build"],
            "white_box_flags": dict(RUNTIME["white_box_flags"]),
        },
    }
    values.update(overrides)
    run_id = runlog.record(**values)
    assert run_id
    return runlog.get_run(run_id)


class FakeEngine:
    """One engine double serving every seam compat_fork can reach: score()/create_checkpoint() for
    checkpoint capture, execution_fork() for the exact control+intervention pair, and complete() for
    the legacy text splice -- so a single object can stand in for "the current worker" in any of the
    three outcome scenarios below."""

    def __init__(self, *, prompt_ids=(1, 2), checkpoint_error=None,
                execution_fork_replies=None, complete_text=" and onward", complete_finish="stop"):
        self.prompt_ids = list(prompt_ids)
        self.checkpoint_error = checkpoint_error
        self.execution_fork_replies = list(execution_fork_replies or [])
        self.complete_text = complete_text
        self.complete_finish = complete_finish
        self.calls = []

    def health(self):
        return {}

    def score(self, **kwargs):
        self.calls.append(("score", dict(kwargs)))
        continuation = list(kwargs["continuation_ids"])
        return {
            "n_prompt": len(self.prompt_ids),
            "n_cont": len(continuation),
            "prompt_ids": list(self.prompt_ids),
            "tokens": [{"id": tid, "piece": str(tid), "logprob": -0.1} for tid in continuation],
        }

    def create_checkpoint(self, tokens, **kwargs):
        self.calls.append(("checkpoint", {"tokens": list(tokens), **dict(kwargs)}))
        if self.checkpoint_error is not None:
            raise self.checkpoint_error
        return {
            "checkpoint_id": "ckpt-generation-a-9",
            "worker_generation_id": WORKER["worker_generation_id"],
            "n_past": len(tokens),
            "n_tokens": len(tokens),
            "size_bytes": 12345,
        }

    def execution_fork(self, **kwargs):
        self.calls.append(("execution_fork", dict(kwargs)))
        if not self.execution_fork_replies:
            raise AssertionError("FakeEngine.execution_fork called with no queued reply")
        reply = self.execution_fork_replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return dict(reply)

    def complete(self, prompt, **params):
        self.calls.append(("complete", dict(params, prompt=prompt)))
        return {"choices": [{"text": self.complete_text, "finish_reason": self.complete_finish}]}


def _exec_reply(*, tokens, text, truncate_to, prompt_tokens, applied):
    live = truncate_to > prompt_tokens
    return {
        "worker_generation_id": WORKER["worker_generation_id"],
        "text": text,
        "tokens": list(tokens),
        "prompt_len": prompt_tokens,
        "n_past_restored": truncate_to,
        "restore_mode": "live_kv_truncated" if live else "reprefill",
        "exactness": {
            "source": "live_kv" if live else "reprefill",
            "truncation_regime": "generated_token" if live else "prompt_boundary",
            "boundary_shape_true": True,
        },
        "sampler_source": "checkpoint",
        "steer_source": "none",
        "intervention_applied": applied,
        "finish_reason": "stop",
    }


class FakeSub:
    def __init__(self, engine, *, runtime=RUNTIME, worker=WORKER):
        self.engine = engine
        self.runtime_identity = lambda: dict(runtime)
        self.worker_identity = lambda: dict(worker)


class Handler:
    """Mirrors clozn.server.app's per-request handler surface just enough for the route: an injectable
    substrate (ctx.active_sub reads h._inj_sub) and the _json(status, body) sink."""

    def __init__(self, sub):
        self._inj_sub = sub
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _post(sub, run_id, body):
    h = Handler(sub)
    claimed = fork_routes.try_post(h, f"/runs/{run_id}/fork", body)
    return claimed, h


# =================================================================================== exact_execution_fork
def test_eligible_run_gets_exact_execution_fork_not_the_splice(stores):
    parent = _eligible_parent()
    engine = FakeEngine(execution_fork_replies=[
        # 1) checkpoint_capture's OWN internal unchanged-control proof (position 0, the whole reply).
        _exec_reply(tokens=[11, 22, 33], text="one two three", truncate_to=2, prompt_tokens=2,
                    applied={"type": "none"}),
        # 2) execute_exact_fork's mandatory unchanged control for THIS request (position 1).
        _exec_reply(tokens=[22, 33], text=" two three", truncate_to=3, prompt_tokens=2,
                    applied={"type": "none"}),
        # 3) the actual forced intervention.
        _exec_reply(tokens=[44, 55], text=" four five", truncate_to=3, prompt_tokens=2,
                    applied={"type": "force_token", "token_id": 44}),
    ])
    claimed, h = _post(FakeSub(engine), parent["id"], {"position": 1, "token_id": 44})

    assert claimed is True
    assert h.status == 200
    body = h.body
    assert body["outcome"] == "exact_execution_fork"
    assert body["reasons"][0]["code"] == "exact_preconditions_met"
    assert body["response"] == "one four five"
    assert body["parent_run_id"] == parent["id"]
    assert body["source"] == "fork"
    assert body["retokenized"] is False                    # nothing was retokenized -- KV was restored
    assert body["prefix_kept"] == "one"
    assert body["forked_from_piece"] == " two"
    assert body["exactness"]["regime"] == "generated_token_live_kv"
    assert body["unchanged_control"]["status"] == "matched"
    assert body["unchanged_control"]["result"]["exact_match"] is True
    assert body["execution_fork_execution_id"]
    # the full FORK-01 receipt is embedded on the child exactly as execute_exact_fork always does --
    # this wrapper does not strip or rename it.
    assert body["execution_fork"]["phase"] == "completed"
    assert body["execution_fork"]["execution_id"] == body["execution_fork_execution_id"]
    # never fell back to the splice
    assert all(name != "complete" for name, _call in engine.calls)
    assert [name for name, _call in engine.calls] == [
        "score", "checkpoint", "execution_fork", "execution_fork", "execution_fork"]


# =================================================================================== reconstructed_replay
def test_ineligible_run_degrades_to_labeled_reconstructed_replay(stores):
    """A historical-run shape: no meta.prompt_tokens (as an older run predating checkpoint capture would
    look), so an exact checkpoint can never be captured -- but the recorded final_prompt and complete
    token boundaries still support the honest text splice."""
    parent = _eligible_parent(meta={"n_ctx": RUNTIME["context_size"], "device": RUNTIME["backend"]})
    engine = FakeEngine(complete_text=" four five", complete_finish="stop")
    claimed, h = _post(FakeSub(engine), parent["id"], {"position": 1, "token_id": 44})

    assert claimed is True
    assert h.status == 200
    body = h.body
    assert body["outcome"] == "reconstructed_replay"
    assert body["reasons"][0]["code"] == "missing_prompt_boundary"
    assert body["exactness"]["regime"] == "reconstructed_text"
    assert "kv_state_not_restored" in body["unavoidable_differences"]
    assert "prompt_prefix_retokenized" in body["unavoidable_differences"]
    assert body["parent_run_id"] == parent["id"]
    assert body["response"] == "one" + " four" + " four five"
    # capture never got far enough to touch the worker's exact seams; only the splice ran
    assert [name for name, _call in engine.calls] == ["complete"]


def test_checkpoint_capture_worker_failure_degrades_to_reconstruction(stores):
    """A genuine WORKER-side capture failure (e.g. the checkpoint store was full / evicted) is still
    "no exact state available" -- degrading to the labeled splice, distinct from a failure AFTER an
    exact plan was already confirmed (see test_exact_execution_failure_is_unavailable_not_masked_as_
    splice, below, which must NOT degrade)."""
    parent = _eligible_parent()
    engine = FakeEngine(checkpoint_error=RuntimeError("worker checkpoint store full"),
                        complete_text=" four five")
    claimed, h = _post(FakeSub(engine), parent["id"], {"position": 1, "token_id": 44})

    assert claimed is True
    assert h.status == 200
    assert h.body["outcome"] == "reconstructed_replay"
    assert h.body["reasons"][0]["code"] == "checkpoint_capture_failed"
    assert [name for name, _call in engine.calls] == ["score", "checkpoint", "complete"]


def test_free_token_text_with_no_recorded_id_forces_reconstruction(stores):
    """A free-text forced token that matches no recorded alternative has no numeric id for the exact
    wire -- exact is never even attempted (no checkpoint-capture call wasted), and the request degrades
    straight to reconstruction, which -- unlike the exact path -- has always supported arbitrary text."""
    parent = _eligible_parent()
    engine = FakeEngine(complete_text=" and beyond")
    claimed, h = _post(FakeSub(engine), parent["id"], {"position": 1, "token": " banana"})

    assert claimed is True
    assert h.status == 200
    assert h.body["outcome"] == "reconstructed_replay"
    assert h.body["reasons"][0]["code"] == "exact_requires_token_id"
    assert h.body["changes_applied"]["fork"]["was_recorded_alternative"] is False
    assert all(name in ("complete",) for name, _call in engine.calls)


# =================================================================================== unavailable
def test_neither_path_eligible_returns_labeled_unavailable(stores):
    """No final_prompt at all: reconstruction cannot render a prompt, and checkpoint capture cannot
    identity-qualify a tokenization either. Neither path runs; the wrapper says so explicitly instead
    of guessing or fabricating a 500 with no explanation."""
    parent = _eligible_parent(final_prompt=None)
    engine = FakeEngine()
    claimed, h = _post(FakeSub(engine), parent["id"], {"position": 1, "token_id": 44})

    assert claimed is True
    assert h.status == 422
    assert h.body["outcome"] == "unavailable"
    assert set(h.body.keys()) == {"outcome", "reasons"}     # no child, no invented facts either
    codes = {reason["code"] for reason in h.body["reasons"]}
    assert "missing_final_prompt" in codes
    assert "reconstruction_prompt_unavailable" in codes
    assert engine.calls == []                              # no worker call wasted on either path


def test_exact_execution_failure_is_unavailable_not_masked_as_splice(stores):
    """The checkpoint WAS captured and the plan WAS exact -- but the mandatory unchanged control
    diverged from the parent's recorded suffix. This is not "no exact state was available"; it is a
    failed one, and reporting it as reconstructed_replay would hide a real divergence behind a
    plausible-looking response. The wrapper must never take that shortcut."""
    parent = _eligible_parent()
    engine = FakeEngine(execution_fork_replies=[
        # checkpoint_capture's own internal control succeeds -- the checkpoint IS captured and eligible.
        _exec_reply(tokens=[11, 22, 33], text="one two three", truncate_to=2, prompt_tokens=2,
                    applied={"type": "none"}),
        # execute_exact_fork's mandatory control for THIS request diverges from the recorded suffix.
        _exec_reply(tokens=[999, 33], text=" wrong three", truncate_to=3, prompt_tokens=2,
                    applied={"type": "none"}),
    ])
    claimed, h = _post(FakeSub(engine), parent["id"], {"position": 1, "token_id": 44})

    assert claimed is True
    assert h.status == 422
    assert h.body["outcome"] == "unavailable"
    assert h.body["reasons"][0]["code"] == "control_diverged"
    assert h.body["execution_fork_execution_id"]
    assert h.body["unchanged_control"]["status"] == "diverged"
    assert "response" not in h.body                        # never fabricated as a generation run
    # the diverged control was reached (the checkpoint DID exist) -- never fell through to complete()
    assert [name for name, _call in engine.calls] == [
        "score", "checkpoint", "execution_fork", "execution_fork"]
    assert all(name != "complete" for name, _call in engine.calls)


# =================================================================================== compatibility
def test_existing_request_shape_and_response_fields_still_work(stores):
    """FORK-02 is additive: the pre-existing request shape (position + token, no `outcome` awareness
    needed) and the pre-existing success-response fields are unchanged, regardless of which outcome
    path actually ran under the hood."""
    parent = _eligible_parent()
    engine = FakeEngine(complete_text=" and beyond")
    claimed, h = _post(FakeSub(engine), parent["id"], {"position": 1, "token": " free"})

    assert claimed is True
    assert h.status == 200
    body = h.body
    for legacy_field in ("prefix_kept", "forked_from_piece", "retokenized", "note",
                         "changes_applied", "parent_run_id", "source", "response", "id"):
        assert legacy_field in body, f"missing pre-FORK-02 field {legacy_field!r}"
    assert body["prefix_kept"] == "one"
    assert body["forked_from_piece"] == " two"
    assert body["source"] == "fork"
    # additive-only: outcome/reasons ride alongside, never replacing anything
    assert body["outcome"] in ("exact_execution_fork", "reconstructed_replay")
    assert isinstance(body["reasons"], list) and body["reasons"]


def test_unknown_run_404(stores):
    claimed, h = _post(FakeSub(FakeEngine()), "run_nope", {"position": 0, "token": "x"})
    assert claimed is True
    assert h.status == 404


def test_bad_position_400(stores):
    parent = _eligible_parent()
    claimed, h = _post(FakeSub(FakeEngine()), parent["id"], {"position": 99, "token": "x"})
    assert claimed is True
    assert h.status == 400


def test_no_engine_still_503_same_as_before_fork02(stores):
    """The "no worker at all" gate is unchanged by FORK-02 -- it is a hard operational failure, not one
    of the three fork outcomes (neither path could ever be attempted). Checked against a REAL run so
    the 404 (unknown run) check upstream of it doesn't mask what this test is actually about."""
    parent = _eligible_parent()

    class NoEngine:
        engine = None
    claimed, h = _post(NoEngine(), parent["id"], {"position": 0, "token": "x"})
    assert claimed is True
    assert h.status == 503


# =================================================================================== live follow-up
@pytest.mark.skip(reason=(
    "compat_fork's own orchestration (capture -> plan -> execute -> label, and each degrade decision) "
    "is fully exercised above with a fake engine, the same model-free convention "
    "tests/test_execution_fork_gateway.py and tests/test_checkpoint_capture.py already use for the "
    "primitives this wrapper composes. What none of that proves is that a REAL clozn-server + a real "
    "GGUF worker reproduces bit-exact token ids/text THROUGH THIS ROUTE end to end -- that needs a live "
    "worker. scripts/smoke/execution_fork_battery.py already proves the underlying primitive that way; "
    "a live follow-up should point an equivalent battery at POST /runs/<id>/fork directly against a "
    "running clozn-server and assert outcome == 'exact_execution_fork' with a bit-exact continuation."
))
def test_live_worker_end_to_end_exactness_needs_a_running_engine():
    pass
