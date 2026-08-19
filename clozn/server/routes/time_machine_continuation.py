"""Public exact appended-turn Time Machine route (ADR 010).

This is deliberately separate from structural ``/branch`` and same-prompt
``/time-machine/branch``.  It restores one identity-qualified checkpoint, proves a newly rendered
conversation is a strict token suffix, sends only that suffix to the private worker, and records one
immutable child plus a terminal receipt.  There is no replay/re-prefill fallback after capture.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import time
import uuid


CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/time-machine/continue"


def _sha_json(value) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _source_projection(requested_run: Mapping, source_run: Mapping, turn: int) -> dict:
    out = {
        "status": "resolved",
        "source_run_id": source_run["id"],
        "source_turn": turn,
        "resolution": (
            "exact_latest_run"
            if source_run.get("id") == requested_run.get("id")
            else "exact_organic_session_prefix"
        ),
    }
    session_key = source_run.get("session_key")
    if isinstance(session_key, str) and session_key:
        out["session_key_sha256"] = hashlib.sha256(session_key.encode("utf-8")).hexdigest()
    return out


def _sampler_projection(capture_material: Mapping) -> dict | None:
    from clozn.replay.time_machine_continuation import (
        sampler_config_sha256,
        sampler_state_sha256,
    )

    recorded = capture_material.get("sampler")
    if not isinstance(recorded, Mapping):
        return None
    mode = recorded.get("mode")
    if mode == "greedy":
        return {
            "status": "preserved",
            "mode": "greedy",
            "source": "checkpoint",
            "config_sha256": sampler_config_sha256(has_sampler=False),
            "state_sha256": None,
            "rng_draws_before_append": 0,
        }
    if mode != "sample":
        return None
    try:
        seed = recorded["seed"]
        rng_draws = recorded["rng_draws"]
        return {
            "status": "preserved",
            "mode": "sample",
            "source": "checkpoint",
            "config_sha256": sampler_config_sha256(
                has_sampler=True,
                temperature=recorded["temperature"],
                repeat_penalty=recorded["rep_penalty"],
                top_k=recorded["top_k"],
                top_p=recorded["top_p"],
            ),
            "state_sha256": sampler_state_sha256(seed=seed, rng_draws=rng_draws),
            "rng_draws_before_append": rng_draws,
        }
    except (KeyError, TypeError, ValueError):
        return None


def _identity_projection(
    source_run: Mapping,
    runtime_identity: Mapping,
    worker_identity: Mapping,
    health: Mapping,
    generation_settings_sha256: str,
) -> dict | None:
    # Identity projection is an immutable execution fact with one neutral kernel owner.  Time
    # Machine's continuation semantics are unchanged; only the source of these helpers moves.
    from clozn.experiments.execution_facts import (
        parent_runtime_projection,
        runtime_projection as _runtime_projection,
        worker_identity_projection as _worker_projection,
    )

    source_runtime = parent_runtime_projection(source_run)
    selected_runtime = _runtime_projection(runtime_identity)
    selected_worker = _worker_projection(worker_identity)
    tokenizer_sha256 = health.get("tokenizer_sha256")
    if (
        source_runtime is None
        or selected_runtime is None
        or source_runtime != selected_runtime
        or selected_worker is None
        or not isinstance(tokenizer_sha256, str)
        or len(tokenizer_sha256) != 64
        or any(char not in "0123456789abcdef" for char in tokenizer_sha256)
    ):
        return None
    # Older run journals did not separately promote the tokenizer digest.  The byte-identical GGUF
    # digest in the runtime key still binds those tokenizer bytes; when a source does carry the
    # narrower digest, require it to agree with the selected worker's measured value as well.
    source_identity = source_run.get("identity")
    source_tokenizer = (
        source_identity.get("tokenizer_sha256")
        if isinstance(source_identity, Mapping) else None
    )
    if source_tokenizer is not None and source_tokenizer != tokenizer_sha256:
        return None
    return {
        "status": "matched",
        "source_runtime_key_sha256": source_runtime["runtime_key_sha256"],
        "selected_runtime_key_sha256": selected_runtime["runtime_key_sha256"],
        "model_sha256": selected_runtime["model_sha256"],
        "tokenizer_sha256": tokenizer_sha256,
        "template_fingerprint": selected_runtime["template_fingerprint"],
        "generation_settings_sha256": generation_settings_sha256,
        "engine_build": selected_runtime["engine_build"],
        "context_size": selected_runtime["context_size"],
        "backend": selected_runtime["backend"],
        "adapter": deepcopy(selected_runtime["adapter"]),
        "worker": selected_worker,
        "checks": {
            "runtime": True,
            "model": True,
            "tokenizer": True,
            "template": True,
            "adapter": True,
            "settings": True,
            "checkpoint": True,
        },
    }


def _checkpoint_projection(
    source_run: Mapping,
    capture: Mapping,
    capture_material: Mapping,
    resolved_pin: Mapping | None,
) -> dict | None:
    from clozn.replay.time_machine_continuation import token_ids_sha256

    reference = capture.get("checkpoint_reference")
    response = capture_material.get("checkpoint_response")
    history = capture_material.get("historical_token_ids")
    if not (
        capture.get("status") == "available"
        and isinstance(reference, Mapping)
        and isinstance(response, Mapping)
        and isinstance(history, list)
        and history
    ):
        return None
    payload_sha256 = response.get("payload_sha256")
    envelope = resolved_pin.get("envelope") if isinstance(resolved_pin, Mapping) else None
    if not isinstance(payload_sha256, str) and isinstance(envelope, Mapping):
        payload_sha256 = envelope.get("payload_sha256")
    if not (
        isinstance(payload_sha256, str)
        and len(payload_sha256) == 64
        and all(char in "0123456789abcdef" for char in payload_sha256)
    ):
        return None
    manifest = resolved_pin.get("manifest") if isinstance(resolved_pin, Mapping) else None
    durable = isinstance(manifest, Mapping)
    source_generation = reference.get("worker_generation_id")
    if durable:
        manifest_source = manifest.get("source")
        if isinstance(manifest_source, Mapping):
            source_generation = manifest_source.get("worker_generation_id")
    out = {
        "status": "available",
        "provenance": "durable_pin_import" if durable else "live_worker_checkpoint",
        "capture_regime": capture_material.get("capture_regime"),
        "restart_safe": durable,
        "source_run_id": source_run["id"],
        "checkpoint_reference_id": capture["checkpoint_reference_id"],
        "checkpoint_id": reference.get("checkpoint_id"),
        "source_worker_generation_id": source_generation,
        "executing_worker_generation_id": reference.get("worker_generation_id"),
        "prompt_tokens": reference.get("prompt_tokens"),
        "n_past": reference.get("n_past"),
        "token_history_sha256": token_ids_sha256(history),
        "payload_sha256": payload_sha256,
    }
    if durable:
        blob = manifest.get("blob")
        out.update({
            "pin_id": manifest.get("pin_id"),
            "pin_blob_sha256": blob.get("sha256") if isinstance(blob, Mapping) else None,
        })
    return out


def _render_append_callback(
    source_run: Mapping,
    engine,
    historical_token_ids: list[int],
    *,
    template_fingerprint: str,
    tokenizer_sha256: str,
    state: dict,
):
    import clozn.replay.timetravel as timetravel

    def render(request):
        messages = timetravel._completed_messages(source_run)
        messages.append({"role": "user", "content": request.user_content})
        info = engine.apply_template_info(messages, add_assistant=True)
        without_cue = engine.apply_template_info(messages, add_assistant=False)
        if not isinstance(info, Mapping) or not isinstance(info.get("prompt"), str):
            raise ValueError("worker template response omitted the rendered prompt")
        full_prompt = info["prompt"]
        score = engine.score(
            prompt=full_prompt,
            continuation_ids=[historical_token_ids[-1]],
            topk=0,
        )
        full_ids = score.get("prompt_ids") if isinstance(score, Mapping) else None
        if not (
            isinstance(full_ids, list)
            and full_ids
            and score.get("n_prompt") == len(full_ids)
            and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0
                    for token in full_ids)
        ):
            raise ValueError("worker did not return exact full-render token IDs")
        trace = source_run.get("trace")
        pieces = trace.get("tokens") if isinstance(trace, Mapping) else None
        historical_text = source_run.get("final_prompt")
        if not (
            isinstance(historical_text, str)
            and isinstance(pieces, list)
            and all(isinstance(piece, str) for piece in pieces)
        ):
            raise ValueError("source run omitted exact rendered historical text")
        historical_text += "".join(pieces)
        if not full_prompt.startswith(historical_text):
            raise ValueError("new template render does not preserve historical rendered text")
        prompt_tokens = info.get("prompt_tokens")
        without_cue_tokens = without_cue.get("prompt_tokens") if isinstance(without_cue, Mapping) else None
        cue_count = (
            prompt_tokens - without_cue_tokens
            if isinstance(prompt_tokens, int) and isinstance(without_cue_tokens, int)
            and prompt_tokens >= without_cue_tokens
            else 0
        )
        state.update({"messages": messages, "final_prompt": full_prompt})
        return {
            "full_render_token_ids": full_ids,
            "rendered_append": full_prompt[len(historical_text):],
            "template_fingerprint": template_fingerprint,
            "tokenizer_sha256": tokenizer_sha256,
            "generation_prefix_token_count": cue_count,
        }

    return render


def _persist_child_callback(
    requested_run: Mapping,
    source_run: Mapping,
    render_state: Mapping,
):
    def persist(payload: Mapping):
        import clozn.runs.store as runlog

        worker_result = payload.get("worker_result")
        receipt = payload.get("receipt")
        messages = render_state.get("messages")
        final_prompt = render_state.get("final_prompt")
        if not (
            isinstance(worker_result, Mapping)
            and isinstance(receipt, Mapping)
            and isinstance(messages, list)
            and isinstance(final_prompt, str)
        ):
            return None
        tokens = worker_result.get("tokens")
        pieces = worker_result.get("token_pieces")
        if not (
            isinstance(tokens, list)
            and isinstance(pieces, list)
            and len(tokens) == len(pieces)
        ):
            return None
        meta = deepcopy(source_run.get("meta") or {})
        meta.update({
            "prompt_tokens": worker_result.get("n_past_after_append"),
            "max_tokens": payload.get("max_tokens"),
            "stream": False,
            "time_machine_continuation_id": receipt.get("continuation_id"),
        })
        changes = {
            "time_machine_continuation": {
                "continuation_id": receipt.get("continuation_id"),
                "source_checkpoint_run_id": payload.get("source_checkpoint_run_id"),
                "source_turn": payload.get("source_turn"),
                "append_token_ids_sha256": receipt.get("append", {}).get("append_token_ids_sha256"),
            },
        }
        started = receipt.get("created_ts")
        ended = receipt.get("finished_ts")
        run_id = runlog.record(
            source="fork",
            client="studio",
            model=str(source_run.get("model") or ""),
            substrate=str(source_run.get("substrate") or ""),
            messages=deepcopy(messages),
            assembled_messages=deepcopy(messages),
            response=worker_result.get("text", ""),
            trace={"tokens": list(pieces), "token_ids": list(tokens)},
            started=started,
            ended=ended,
            parent_run_id=requested_run["id"],
            changes_applied=changes,
            finish_reason=worker_result.get("finish_reason"),
            meta=meta,
            final_prompt=final_prompt,
            identity=deepcopy(source_run.get("identity") or {}),
            session_key=requested_run.get("session_key"),
            client_key=requested_run.get("client_key"),
            client_key_source=requested_run.get("client_key_source"),
            project_key=requested_run.get("project_key"),
            time_machine_continuation_receipt=deepcopy(receipt),
            _reserved_run_id=payload.get("child_run_id"),
        )
        return {"child_run_id": run_id} if run_id else None

    return persist


def _save_and_reply(h, receipt: dict) -> None:
    from clozn.replay import time_machine_continuation_results

    # Completed receipts are already atomically embedded in their child.  All other outcomes have no
    # run, so this result store is their only durable evidence and a write failure must be visible.
    try:
        receipt = time_machine_continuation_results.save(receipt)
    except Exception as exc:
        if receipt.get("status") != "completed":
            h._json(500, {
                "error": (
                    "Time Machine continuation receipt could not be persisted: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "code": "time_machine_continuation_receipt_persistence_error",
            })
            return
    status = receipt.get("status")
    h._json(201 if status == "completed" else 409 if status == "cancelled" else 422, receipt)


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False
    requested_run_id = p[len("/runs/"):-len(_SUFFIX)]
    import clozn.runs.store as runlog
    from clozn.replay import time_machine_continuation as continuation

    requested_run = runlog.get_run(requested_run_id)
    if requested_run is None:
        h._json(404, {"error": "run not found"})
        return True
    try:
        request = continuation.parse_continuation_request(body)
    except continuation.ClosedRequestError as exc:
        h._json(400, {"error": str(exc), "code": exc.code})
        return True

    request_id = "tmc_req_" + uuid.uuid4().hex[:20]
    continuation_id = "tmc_" + uuid.uuid4().hex[:20]
    import clozn.replay.timetravel as timetravel
    source_run = timetravel.resolve_exact_turn_source_run(requested_run, request.turn)
    # Exact continuation inherits the selected immutable source's generation contract.  A request
    # made against a later run may resolve an older organic prefix, so hashing the requested run's
    # settings here would falsely certify a different contract.
    settings_run = source_run if isinstance(source_run, Mapping) else requested_run
    meta = settings_run.get("meta")
    generation_source = {
        "decode": deepcopy(meta.get("decode")) if isinstance(meta, Mapping) else None,
        "sampler_mode": meta.get("sampler_mode") if isinstance(meta, Mapping) else None,
        "output_contract": bool(settings_run.get("output_contract")),
        "reasoning": bool(settings_run.get("reasoning")),
    }
    generation_settings_sha256 = _sha_json(generation_source)
    base = continuation.new_receipt_base(
        request,
        requested_run_id=requested_run_id,
        request_id=request_id,
        generation_config_sha256=generation_settings_sha256,
        continuation_id=continuation_id,
    )

    if not isinstance(source_run, Mapping):
        receipt = continuation.build_unavailable_receipt(
            base,
            stage="source_resolution",
            code="historical_source_unavailable",
            message="no unique immutable organic source run proves the requested completed turn",
            evidence={
                "source": {
                    "status": "unavailable",
                    "reasons": [_reason(
                        "historical_source_unavailable",
                        "no unique immutable organic source run proves the requested completed turn",
                    )],
                },
            },
        )
        _save_and_reply(h, receipt)
        return True
    source = _source_projection(requested_run, source_run, request.turn)

    from clozn.server.model_routing import select_run_model_facts
    facts = select_run_model_facts(
        h, source_run, route="/runs/<id>/time-machine/continue")
    if facts is None:
        return True
    runtime_identity, worker_identity, engine, _sub = facts
    if engine is None or not isinstance(runtime_identity, Mapping) or not isinstance(worker_identity, Mapping):
        receipt = continuation.build_unavailable_receipt(
            base,
            stage="identity",
            code="worker_capability_missing",
            message="exact continuation requires a ready identity-qualified worker",
            evidence={"source": source},
        )
        _save_and_reply(h, receipt)
        return True
    try:
        health = engine.health()
    except Exception:
        health = {}
    capabilities = health.get("capabilities") if isinstance(health, Mapping) else None
    if not (
        isinstance(capabilities, Mapping)
        and capabilities.get("time_machine_continuation") is True
        and callable(getattr(engine, "time_machine_continue", None))
    ):
        receipt = continuation.build_unavailable_receipt(
            base,
            stage="worker_restore",
            code="worker_capability_missing",
            message="selected worker does not advertise exact append-only continuation",
            evidence={"source": source},
        )
        _save_and_reply(h, receipt)
        return True

    resolved_pin = None
    checkpoint_envelope = None
    candidate = None
    try:
        from clozn.replay.checkpoint_pin_store import resolve_pin
        candidate = resolve_pin(source_run["id"])
        if isinstance(candidate, Mapping) and candidate.get("ok") is True:
            resolved_pin = candidate
            checkpoint_envelope = candidate.get("envelope")
    except Exception:
        pass
    if resolved_pin is None or not isinstance(checkpoint_envelope, Mapping):
        unavailable = candidate.get("unavailable") if isinstance(candidate, Mapping) else None
        corrupt = isinstance(unavailable, str) and any(
            marker in unavailable.lower()
            for marker in ("corrupt", "digest", "blob missing", "sidecar")
        )
        code = "checkpoint_corrupt" if corrupt else "checkpoint_unavailable"
        message = (
            "the exact source pin failed integrity verification"
            if corrupt else
            "exact appended-turn continuation requires a durable pin for this source run"
        )
        receipt = continuation.build_unavailable_receipt(
            base,
            stage="checkpoint",
            code=code,
            message=message,
            evidence={
                "source": source,
                "source_checkpoint": {
                    "status": "unavailable",
                    "reasons": [_reason(code, message)],
                },
            },
            retryable=not corrupt,
        )
        _save_and_reply(h, receipt)
        return True

    from clozn.replay.checkpoint_capture import capture_parent_checkpoint
    capture_material: dict = {}
    try:
        capture = capture_parent_checkpoint(
            source_run,
            engine,
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            checkpoint_envelope=checkpoint_envelope,
            material_out=capture_material,
        )
    except Exception as exc:
        capture = {
            "status": "failed",
            "reasons": [_reason(
                "checkpoint_capture_failed",
                f"checkpoint capture failed: {type(exc).__name__}: {exc}",
            )],
        }
    if not isinstance(capture, Mapping) or capture.get("status") != "available":
        reason = (capture.get("reasons") or [{}])[0] if isinstance(capture, Mapping) else {}
        code = str(reason.get("code") or "checkpoint_unavailable")
        if code not in {
            "checkpoint_unavailable", "checkpoint_expired", "checkpoint_corrupt",
            "checkpoint_identity_mismatch", "checkpoint_import_failed",
        }:
            code = "checkpoint_unavailable"
        receipt = continuation.build_unavailable_receipt(
            base,
            stage="checkpoint",
            code=code,
            message=str(reason.get("message") or "exact source checkpoint is unavailable"),
            evidence={
                "source": source,
                "source_checkpoint": {
                    "status": "unavailable",
                    "reasons": [_reason(code, str(
                        reason.get("message") or "exact source checkpoint is unavailable"))],
                },
            },
        )
        _save_and_reply(h, receipt)
        return True

    source_checkpoint = _checkpoint_projection(
        source_run, capture, capture_material, resolved_pin)
    sampler = _sampler_projection(capture_material)
    identity = _identity_projection(
        source_run,
        runtime_identity,
        worker_identity,
        health if isinstance(health, Mapping) else {},
        generation_settings_sha256,
    )
    historical_ids = capture_material.get("historical_token_ids")
    if not isinstance(source_checkpoint, Mapping) or not isinstance(historical_ids, list):
        receipt = continuation.build_unavailable_receipt(
            base,
            stage="checkpoint",
            code="checkpoint_identity_mismatch",
            message="checkpoint omitted the payload or token-history proof required for exact append",
            evidence={"source": source},
        )
        _save_and_reply(h, receipt)
        return True
    if not isinstance(identity, Mapping) or not isinstance(sampler, Mapping):
        receipt = continuation.build_unavailable_receipt(
            base,
            stage="identity",
            code="checkpoint_identity_mismatch",
            message="source runtime, tokenizer, or sampler identity could not be matched exactly",
            evidence={"source": source, "source_checkpoint": source_checkpoint},
        )
        _save_and_reply(h, receipt)
        return True

    render_state: dict = {}
    render_append = _render_append_callback(
        source_run,
        engine,
        historical_ids,
        template_fingerprint=identity["template_fingerprint"],
        tokenizer_sha256=identity["tokenizer_sha256"],
        state=render_state,
    )
    source_settings = {
        "native_grammar_constraints_required": bool(source_run.get("output_contract")),
        "additional_stop_constraints_required": bool(
            isinstance(meta, Mapping) and meta.get("additional_stop_constraints_applied") is True),
    }
    receipt = continuation.orchestrate_continuation(
        body,
        requested_run_id=requested_run_id,
        request_id=request_id,
        generation_config_sha256=generation_settings_sha256,
        source=source,
        source_checkpoint=source_checkpoint,
        identity=identity,
        sampler=sampler,
        historical_token_ids=historical_ids,
        render_append=render_append,
        worker_post=lambda _endpoint, worker_body: engine.time_machine_continue(**worker_body),
        persist_child=_persist_child_callback(requested_run, source_run, render_state),
        source_generation_settings=source_settings,
        continuation_id=continuation_id,
        child_run_id_factory=lambda: "run_tmc_" + uuid.uuid4().hex[:20],
        checkpoint_on_finish=True,
    )
    _save_and_reply(h, receipt)
    return True
