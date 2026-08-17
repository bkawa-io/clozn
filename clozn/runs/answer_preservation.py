"""Exact recorded-answer eligibility and classification primitives.

The experimental kernel owns probe orchestration and evidence persistence.
This module supplies the trusted token/contract/runtime checks and the shared
classifier used by the execution substrate.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from clozn.receipts.rederive import with_arm_conditions


class ExactAnswerPreservationError(ValueError):
    """Raised when an exact-reference probe cannot be constructed honestly."""


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_int(value: Any, minimum: int | None = None) -> bool:
    return (isinstance(value, int) and not isinstance(value, bool)
            and (minimum is None or value >= minimum))


def _finite(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    return ((minimum is None or float(value) >= minimum)
            and (maximum is None or float(value) <= maximum))


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _termination_from_run(run: Mapping[str, Any]) -> dict[str, Any] | None:
    meta = run.get("meta") if isinstance(run.get("meta"), Mapping) else {}
    for value in (
        run.get("termination"),
        (run.get("context_receipt") or {}).get("termination")
        if isinstance(run.get("context_receipt"), Mapping) else None,
        meta.get("termination"),
    ):
        if isinstance(value, Mapping) and value:
            reason = value.get("reason", value.get("kind"))
            raw = value.get("reason_raw", value.get("kind", reason))
            if isinstance(reason, str) and reason:
                return {"reason": reason, "reason_raw": raw if isinstance(raw, str) else reason}
    finish = run.get("finish_reason", meta.get("finish_reason"))
    if isinstance(finish, str) and finish:
        # This is only an explicit recorded finish field.  It is not a guessed
        # default; callers still need a complete generation contract below.
        return {"reason": finish, "reason_raw": finish}
    return None


def _generation_contract_from_run(run: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Extract a complete canonical contract without inventing sampler values."""
    meta = run.get("meta") if isinstance(run.get("meta"), Mapping) else {}
    explicit = run.get("generation_contract")
    if not isinstance(explicit, Mapping):
        explicit = meta.get("generation_contract")
    if isinstance(explicit, Mapping):
        raw = dict(explicit)
        mode = raw.get("decode_mode", raw.get("mode"))
        if mode is None and isinstance(raw.get("decode"), Mapping):
            mode = raw["decode"].get("mode")
        max_new = raw.get("max_new", raw.get("max_tokens"))
        stop = raw.get("stop")
        sampling = raw.get("sampling")
        expected = raw.get("expected_termination")
    else:
        decode = meta.get("decode") if isinstance(meta.get("decode"), Mapping) else {}
        mode = decode.get("mode", meta.get("sampler_mode"))
        max_new = meta.get("max_tokens")
        stop = meta.get("stop")
        sampling = None
        expected = None
        if mode == "sample":
            sampling = {
                "temperature": _first(decode, "temperature") if "temperature" in decode else meta.get("temperature"),
                "top_p": _first(decode, "top_p") if "top_p" in decode else meta.get("top_p"),
                "top_k": _first(decode, "top_k") if "top_k" in decode else meta.get("top_k"),
                "repeat_penalty": _first(decode, "repeat_penalty") if "repeat_penalty" in decode
                else _first(meta, "repeat_penalty", "repetition_penalty"),
                "seed": _first(decode, "seed") if "seed" in decode else meta.get("seed"),
            }
        if isinstance(meta.get("sampling"), Mapping):
            sampling = dict(meta["sampling"])
        expected = _termination_from_run(run)

    if mode not in {"greedy", "sample"}:
        return None, "generation_contract_incomplete"
    if not _is_int(max_new, 1):
        return None, "sampled_replay_not_proven" if mode == "sample" else "generation_contract_incomplete"
    if not isinstance(stop, list) or any(not isinstance(item, str) for item in stop):
        return None, "sampled_replay_not_proven" if mode == "sample" else "generation_contract_incomplete"
    if not isinstance(expected, Mapping) or not isinstance(expected.get("reason"), str):
        return None, "sampled_replay_not_proven" if mode == "sample" else "generation_contract_incomplete"

    contract: dict[str, Any] = {
        "decode_mode": mode,
        "sampling": None,
        "max_new": int(max_new),
        "stop": list(stop),
        "expected_termination": {
            "reason": expected["reason"],
            "reason_raw": expected.get("reason_raw", expected["reason"]),
        },
    }
    if mode == "sample":
        if not isinstance(sampling, Mapping):
            return None, "sampled_replay_not_proven"
        fields = ("temperature", "top_p", "top_k", "repeat_penalty", "seed")
        if any(field not in sampling for field in fields):
            return None, "sampled_replay_not_proven"
        if not _finite(sampling["temperature"], minimum=0.0) or not _finite(
                sampling["top_p"], minimum=0.0, maximum=1.0):
            return None, "sampled_replay_not_proven"
        if not _is_int(sampling["top_k"], 0) or not _finite(sampling["repeat_penalty"], minimum=0.0):
            return None, "sampled_replay_not_proven"
        if not _is_int(sampling["seed"], 0):
            return None, "sampled_replay_not_proven"
        contract["sampling"] = {
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
            "top_k": int(sampling["top_k"]),
            "repeat_penalty": float(sampling["repeat_penalty"]),
            "seed": int(sampling["seed"]),
        }
    return contract, None


def _trace_token_pieces(run: Mapping[str, Any]) -> tuple[list[int] | None, str | None]:
    conditions = with_arm_conditions(dict(run))
    ids = conditions.get("continuation_ids")
    if not isinstance(ids, list) or not ids or any(not _is_int(item) for item in ids):
        return None, "missing_exact_recorded_token_ids"
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    pieces: list[Any] = []
    steps = trace.get("steps")
    if isinstance(steps, list) and steps:
        pieces = [step.get("piece") for step in steps if isinstance(step, Mapping)]
    elif isinstance(trace.get("tokens"), list):
        pieces = list(trace["tokens"])
    response = run.get("response")
    if not isinstance(response, str) or not pieces or any(not isinstance(piece, str) for piece in pieces):
        return None, "token_pieces_do_not_reconstruct_response"
    if len(pieces) != len(ids) or "".join(pieces) != response:
        return None, "token_pieces_do_not_reconstruct_response"
    return list(ids), None


def _runtime_projection_for_sub(run: Mapping[str, Any], sub: Any) -> dict | None:
    if sub is None:
        return None
    identity_fn = getattr(sub, "identity_meta", None)
    meta_fn = getattr(sub, "run_meta", None)
    if not callable(identity_fn) or not callable(meta_fn):
        return None
    try:
        from clozn.replay.execution_fork import parent_runtime_projection
        current = {
            "id": run.get("id", "current"),
            "model": run.get("model"),
            "identity": dict(identity_fn() or {}),
            "meta": dict(meta_fn() or {}),
        }
        return parent_runtime_projection(current)
    except Exception:
        return None


def assess_exact_eligibility(run: Mapping[str, Any], sub: Any = None,
                             *, current_runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a pure eligibility projection; never calls the model."""
    reasons: list[str] = []
    if not isinstance(run, Mapping):
        reasons.append("invalid_run")
        return {"eligible": False, "reasons": reasons, "reason": reasons[0]}
    contract, contract_reason = _generation_contract_from_run(run)
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    ids, token_reason = _trace_token_pieces(run)
    if token_reason:
        reasons.append(token_reason)
    # RunStore emits an empty output_contract object for ordinary text runs.
    # That placeholder is not structured output; only a non-empty contract is
    # an exact-replay exclusion.
    if (("output_contract" in run and run.get("output_contract") not in (None, {}))
            or (isinstance(run.get("meta"), Mapping)
                and "output_contract" in run["meta"]
                and run["meta"].get("output_contract") not in (None, {}))
            or not isinstance(run.get("response"), str)):
        reasons.append("unsupported_structured_output")
    if trace.get("retokenized") is True or trace.get("boundary_approximate") is True or run.get("retokenized") is True:
        reasons.append("retokenized_continuation")
    if contract_reason:
        reasons.append(contract_reason)
    elif contract and contract["decode_mode"] == "sample":
        # The current ordinary worker replay is deterministic only when the
        # complete sampler contract is persisted.  This branch is explicit so
        # future samplers cannot silently fall back to greedy.
        if contract["sampling"].get("seed") is None:
            reasons.append("sampled_replay_not_proven")

    from clozn.replay import execution_fork
    from clozn.replay.execution_fork import parent_runtime_projection
    recorded_runtime = None
    try:
        recorded_runtime = parent_runtime_projection(run)
    except Exception:
        recorded_runtime = None
    if recorded_runtime is None:
        reasons.append("runtime_identity_unavailable")
    if current_runtime is not None:
        current_projection = execution_fork._runtime_projection(
            current_runtime, run_meta=current_runtime)
    else:
        current_projection = _runtime_projection_for_sub(run, sub)
    if current_projection is None:
        reasons.append("runtime_identity_unavailable")
    elif recorded_runtime is not None and current_projection != recorded_runtime:
        if current_projection.get("template_fingerprint") != recorded_runtime.get("template_fingerprint"):
            reasons.append("template_mismatch")
        else:
            reasons.append("runtime_identity_mismatch")

    reasons = list(dict.fromkeys(reasons))
    eligible = not reasons and ids is not None and contract is not None
    return {
        "eligible": eligible,
        "reasons": reasons,
        "reason": reasons[0] if reasons else None,
        "generation_contract": deepcopy(contract) if contract else None,
        "reference_token_count": len(ids) if ids is not None else None,
        "reference_token_ids_sha256": _sha256(ids) if ids is not None else None,
        "recorded_runtime": deepcopy(recorded_runtime),
        "current_runtime": deepcopy(current_projection),
    }


def _actual_termination(evidence: Mapping[str, Any]) -> tuple[str | None, str | None]:
    termination = evidence.get("termination")
    if not isinstance(termination, Mapping) and any(
            key in evidence for key in ("kind", "reason", "reason_raw")):
        termination = evidence
    if isinstance(termination, Mapping):
        raw = termination.get("kind", termination.get("reason_raw", termination.get("reason")))
        if isinstance(raw, str) and raw:
            return raw, raw
    raw = evidence.get("finish_reason_raw", evidence.get("finish_reason"))
    return (raw, raw) if isinstance(raw, str) and raw else (None, None)


def _termination_match(expected: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    actual_raw, _ = _actual_termination(evidence)
    expected_raw = expected.get("reason_raw", expected.get("reason"))
    if not isinstance(expected_raw, str) or not isinstance(actual_raw, str):
        return False
    if expected_raw == actual_raw:
        return True
    # The public finish reason is coarser than the worker termination kind.
    aliases = {
        "stop": {"eos", "template_stop_sequence", "user_stop_sequence"},
        "length": {"length", "steps_exhausted"},
        "max_tokens": {"length", "steps_exhausted"},
        "stop_sequence": {"template_stop_sequence", "user_stop_sequence", "stop"},
        "eos": {"stop"},
    }
    return actual_raw in aliases.get(expected_raw, set())


def classify_reference_match(reference_token_ids: Iterable[int], generated_token_ids: Iterable[int],
                             *, diverged: bool | None = None, diverged_at: int | None = None,
                             termination: Mapping[str, Any] | None = None,
                             finish_reason: str | None = None,
                             expected_termination: Mapping[str, Any] | None = None,
                             max_new: int | None = None) -> dict[str, Any]:
    """Classify a worker probe, including the full-answer boundary case."""
    expected = [int(value) for value in reference_token_ids]
    actual = [int(value) for value in generated_token_ids]
    termination_evidence = dict(termination or {})
    if finish_reason and "finish_reason" not in termination_evidence:
        termination_evidence["finish_reason"] = finish_reason
    term_match = (
        _termination_match(expected_termination, termination_evidence)
        if isinstance(expected_termination, Mapping) else False
    )
    mismatch = next((index for index, (want, got) in enumerate(zip(expected, actual)) if want != got), None)
    divergence_index = diverged_at if _is_int(diverged_at) else mismatch
    if mismatch is not None:
        return {
            "status": "diverged", "matched_token_count": mismatch,
            "first_divergence_index": mismatch, "expected_token_id": expected[mismatch],
            "actual_token_id": actual[mismatch], "termination_match": term_match,
            "divergence_kind": "token_mismatch",
        }
    if len(actual) < len(expected):
        return {
            "status": "diverged", "matched_token_count": len(actual),
            "first_divergence_index": divergence_index, "expected_token_id": expected[len(actual)],
            "actual_token_id": None, "termination_match": term_match,
            "divergence_kind": "early_termination",
        }
    if len(actual) > len(expected) or (diverged is True and divergence_index == len(expected)):
        actual_id = actual[len(expected)] if len(actual) > len(expected) else None
        return {
            "status": "diverged", "matched_token_count": len(expected),
            "first_divergence_index": len(expected), "expected_token_id": None,
            "actual_token_id": actual_id, "termination_match": term_match,
            "divergence_kind": "extra_token_after_reference",
        }
    if expected_termination is not None and not _termination_match(expected_termination, termination_evidence):
        return {
            "status": "diverged", "matched_token_count": len(expected),
            "first_divergence_index": None, "expected_token_id": None, "actual_token_id": None,
            "termination_match": False, "divergence_kind": "termination_mismatch",
        }
    if max_new is not None and len(actual) > max_new:
        return {
            "status": "diverged", "matched_token_count": len(expected),
            "first_divergence_index": len(expected), "expected_token_id": None,
            "actual_token_id": actual[len(expected)] if len(actual) > len(expected) else None,
            "termination_match": False, "divergence_kind": "extra_token_after_reference",
        }
    return {
        "status": "matched", "matched_token_count": len(expected),
        "first_divergence_index": None, "expected_token_id": None, "actual_token_id": None,
        "termination_match": True, "divergence_kind": None,
    }


def is_reference_match_preserving(observation: Mapping[str, Any] | Any) -> bool:
    """Return the canonical exact-reference preservation verdict.

    Exact probes are preserving only when the shared classifier emits
    ``status='matched'``.  Missing, unavailable, or malformed evidence never
    becomes a preserving result by omission.
    """
    return isinstance(observation, Mapping) and observation.get("status") == "matched"


def is_reference_match_failed(observation: Mapping[str, Any] | Any) -> bool:
    """Return true only for a directly classified exact-reference divergence."""
    return isinstance(observation, Mapping) and observation.get("status") == "diverged"
