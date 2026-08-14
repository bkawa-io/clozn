"""Run-level, direct Context Dependence measurements.

The costly operation in this module is deliberately target-agnostic.  It
teacher-forces the *whole recorded continuation* once in the full context and
once for every canonical source-deletion arm.  A UI selection is consequently
a cheap read-only projection over persisted token evidence, rather than a
reason to score the model again or mint another experiment identity.

Context source resolution and mutation are owned by
``clozn.replay.span_bridge.resolve_context_receipt_source_set``.  This module
does not reproduce its receipt validation or splice algorithm: every arm uses
the bridge's verified messages and exact removed-range evidence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import time
from typing import Any, Iterable, Mapping

from clozn import schemas
from clozn.replay.span_bridge import (
    ContextReceiptSourceResolutionError,
    NEUTRALIZATION_LENGTH_CONTRACT,
    NEUTRALIZATION_OPERATOR,
    NEUTRALIZATION_RECIPE,
    NEUTRALIZATION_STRATEGY,
    neutralize_context_receipt_sources,
    resolve_context_receipt_source_set,
)

from . import rederive


SCHEMA = "clozn.context-dependence-study.v2"
INTERVENTION_OPERATOR = "delete_source"
PROVENANCE = "measured"
NEUTRALIZATION_PROVENANCE = "measured_matched_length_neutralization_control"


class ContextDependenceError(ValueError):
    """A direct Context Dependence measurement could not be made faithfully."""


class UnknownSourceIdError(ContextDependenceError):
    """A requested source did not have a canonical identity in this study."""


class MeasurementUnavailableError(ContextDependenceError):
    """The recorded continuation could not be teacher-forced and scored."""


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_messages(messages: Any) -> list:
    return deepcopy(messages if isinstance(messages, list) else [])


def _prompt_identity(run: Mapping[str, Any]) -> dict:
    """Return recorded model/runtime/template facets without capture-time noise."""
    captured = run.get("identity")
    captured = deepcopy(captured) if isinstance(captured, dict) else {}
    captured.pop("captured_at", None)
    receipt = run.get("context_receipt")
    receipt = receipt if isinstance(receipt, dict) else {}
    rendered = receipt.get("rendered")
    rendered = rendered if isinstance(rendered, dict) else {}
    final_prompt = run.get("final_prompt")
    final_prompt_hash = (
        hashlib.sha256(final_prompt.encode("utf-8")).hexdigest()
        if isinstance(final_prompt, str) else None
    )
    return {
        "model": deepcopy(run.get("model")),
        "substrate": deepcopy(run.get("substrate")),
        "runtime_artifact_identity": captured,
        "template_fingerprint": (
            rendered.get("template_fingerprint")
            if isinstance(rendered.get("template_fingerprint"), str)
            else receipt.get("template_fingerprint")
            if isinstance(receipt.get("template_fingerprint"), str)
            else captured.get("template_fingerprint")
        ),
        "rendered_prompt_sha256": (
            rendered.get("sha256") if isinstance(rendered.get("sha256"), str) else final_prompt_hash
        ),
    }


def _removed_source_set(removed_source_ids: Iterable[str]) -> list[str]:
    if isinstance(removed_source_ids, (str, bytes)):
        raise ContextDependenceError("removed_source_ids must be an iterable of source ID strings")
    try:
        ids = list(removed_source_ids)
    except TypeError as exc:
        raise ContextDependenceError("removed_source_ids must be an iterable of source ID strings") from exc
    if not ids or any(not isinstance(item, str) or not item for item in ids):
        raise ContextDependenceError("removed_source_ids must contain at least one non-empty source ID")
    if len(set(ids)) != len(ids):
        raise ContextDependenceError("removed_source_ids must not contain duplicate source IDs")
    return sorted(ids)


def _candidate_source_ids(run: Mapping[str, Any]) -> list[str]:
    """Find IDs only to ask the strict resolver to prove them.

    This does *not* establish a source-to-message correspondence.  The bridge
    remains the sole authority for that proof and can return a richer source
    span set than these legacy receipt rows.  Keeping this small compatibility
    probe lets v2 measure current v1 Context Receipts while source-span receipts
    roll out.
    """
    receipt = run.get("context_receipt")
    receipt = receipt if isinstance(receipt, Mapping) else {}
    seen: set[str] = set()
    result: list[str] = []

    def append(value: Any) -> None:
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            result.append(value)

    # New span-aware bridges expose canonical sources explicitly.  Old
    # receipts expose segment rows.  Prefer explicit source records but retain
    # the segment rows as an intentionally read-compatible fallback.
    for key in ("sources", "source_spans", "canonical_sources"):
        rows = receipt.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    append(row.get("source_id"))
    for key in ("assembled", "delivered"):
        rows = receipt.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    append(row.get("source_id"))
                    append(row.get("segment_id"))
                    spans = row.get("sources")
                    if isinstance(spans, list):
                        for span in spans:
                            if isinstance(span, Mapping):
                                append(span.get("source_id"))
    return result


def _as_int_token_id(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _validate_scored_tokens(tokens: Any, *, expected_ids: list[int] | None) -> list[dict]:
    if not isinstance(tokens, list) or not tokens:
        raise MeasurementUnavailableError("the scorer returned no teacher-forced token log-probabilities")
    if expected_ids is not None and len(tokens) != len(expected_ids):
        raise MeasurementUnavailableError("the scorer did not return one score per recorded token ID")
    validated: list[dict] = []
    for index, token in enumerate(tokens):
        if not isinstance(token, Mapping):
            raise MeasurementUnavailableError("the scorer returned a malformed token record")
        logprob = token.get("logprob")
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
            raise MeasurementUnavailableError("the scorer returned a non-finite token log-probability")
        token_id = _as_int_token_id(token.get("id"))
        if expected_ids is not None and token_id != expected_ids[index]:
            raise MeasurementUnavailableError("the scorer did not preserve the recorded continuation IDs")
        item = {"index": index, "piece": str(token.get("piece", "")), "logprob": float(logprob)}
        if token_id is not None:
            item["token_id"] = token_id
        validated.append(item)
    return validated


def _baseline_token_evidence(tokens: list[dict]) -> tuple[list[dict], str]:
    """Attach exact Unicode ranges to the scorer's ordered token pieces."""
    cursor = 0
    evidence: list[dict] = []
    pieces: list[str] = []
    for token in tokens:
        piece = token["piece"]
        end = cursor + len(piece)
        item = {
            "index": token["index"],
            "piece": piece,
            "unicode_range": [cursor, end],
            "logprob": token["logprob"],
        }
        if "token_id" in token:
            item["token_id"] = token["token_id"]
        evidence.append(item)
        pieces.append(piece)
        cursor = end
    return evidence, "".join(pieces)


def _source_from_range(source_id: str, removed_range: Mapping[str, Any], *, messages: list) -> dict:
    """Compatibility metadata for an old resolver that only returns ranges.

    The bridge verified the range before returning it.  We merely retain that
    evidence and annotate the owning scoring-view message; no deletion or
    correspondence resolution happens here.
    """
    index = removed_range.get("message_index")
    unicode_range = removed_range.get("unicode_range")
    byte_range = removed_range.get("byte_range")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ContextDependenceError("source resolver returned a malformed message index")
    if not (
        isinstance(unicode_range, list) and len(unicode_range) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in unicode_range)
        and isinstance(byte_range, list) and len(byte_range) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in byte_range)
    ):
        raise ContextDependenceError("source resolver returned malformed exact removed ranges")
    item = {
        "source_id": source_id,
        "message_index": index,
        "unicode_range": list(unicode_range),
        "byte_range": list(byte_range),
    }
    if 0 <= index < len(messages) and isinstance(messages[index], Mapping):
        message = messages[index]
        role = message.get("role")
        content = message.get("content")
        if isinstance(role, str):
            item["role"] = role
        if isinstance(content, str):
            start, end = unicode_range
            if 0 <= start <= end <= len(content):
                selected = content[start:end]
                item["content_sha256"] = hashlib.sha256(selected.encode("utf-8")).hexdigest()
    if source_id.startswith("seg_"):
        item["segment_id"] = source_id
    return item


def _source_records(resolved: Mapping[str, Any], *, messages: list) -> list[dict]:
    """Consume current and span-aware bridge contracts defensively.

    Source-span work can provide ``sources``/``canonical_sources`` directly;
    the former whole-message bridge only provides one exact range per requested
    source.  Both cases preserve bridge evidence and never recreate mutations.
    """
    available = resolved.get("available_source_ids")
    if not isinstance(available, list) or any(not isinstance(item, str) or not item for item in available):
        raise ContextDependenceError("source resolver returned no canonical source IDs")
    available = list(dict.fromkeys(available))
    direct = resolved.get("sources")
    if not isinstance(direct, list):
        direct = resolved.get("canonical_sources")
    by_id: dict[str, dict] = {}
    if isinstance(direct, list):
        for item in direct:
            if not isinstance(item, Mapping):
                continue
            source_id = item.get("source_id")
            if isinstance(source_id, str) and source_id in available and source_id not in by_id:
                by_id[source_id] = deepcopy(dict(item))

    ranges = resolved.get("exact_removed_ranges")
    range_by_id = {
        item.get("source_id"): item for item in ranges
        if isinstance(item, Mapping) and isinstance(item.get("source_id"), str)
    } if isinstance(ranges, list) else {}
    records: list[dict] = []
    for source_id in available:
        record = by_id.get(source_id)
        if record is None:
            one = range_by_id.get(source_id)
            if one is None:
                # A bridge may only include selected ranges.  The caller
                # invokes it once with every candidate so this is evidence of
                # a contract mismatch, never a reason to infer offsets.
                raise ContextDependenceError("source resolver omitted exact range evidence for a source")
            record = _source_from_range(source_id, one, messages=messages)
        else:
            record["source_id"] = source_id
        records.append(record)
    return records


class ContextDependenceStudy:
    """One cached run-level baseline plus directly measured source deletions."""

    def __init__(self, run: dict, sub, *, target: dict | None = None, clock=time.perf_counter):
        if not isinstance(run, dict) or not run:
            raise ContextDependenceError("a non-empty recorded run is required")
        if not isinstance(run.get("id"), str) or not run["id"]:
            raise ContextDependenceError("a recorded run ID is required for a persistable study")
        if target is not None:
            raise ContextDependenceError(
                "target is not accepted by clozn.context-dependence-study.v2; "
                "measure the full continuation and project a Unicode selection afterwards"
            )
        self._run = deepcopy(run)
        self._sub = sub
        self._clock = clock
        self._conditions = deepcopy(rederive.with_arm_conditions(self._run))
        self._messages = _copy_messages(self._conditions.get("messages"))
        self._block = deepcopy(self._conditions.get("block"))
        self._baseline_tokens: list[dict] | None = None
        self._baseline_evidence: list[dict] | None = None
        self._continuation: dict | None = None
        self._experiments: list[dict] = []
        self._experiments_by_id: dict[str, dict] = {}
        self._robustness_controls: list[dict] = []
        self._robustness_controls_by_id: dict[str, dict] = {}
        self._context_identity = _prompt_identity(self._run)
        self._source_resolution_error: str | None = None
        self._sources: list[dict] = []
        self._source_view = "unknown"
        self._full_context_hash = _canonical_digest({"messages": self._messages, "block": self._block})

        candidates = _candidate_source_ids(self._run)
        if candidates:
            try:
                # This single call both verifies the source catalogue and
                # establishes the exact unmutated context digest.  It is
                # model-free; the score budget begins only at _ensure_baseline.
                try:
                    resolved = resolve_context_receipt_source_set(self._run, candidates)
                except ContextReceiptSourceResolutionError:
                    # During the receipt migration an old resolver knows only
                    # complete ``seg_`` sources while a new receipt may already
                    # carry ``src_`` descriptors.  Ask the resolver again for
                    # its legacy vocabulary; never synthesize an arm here.
                    legacy_candidates = [item for item in candidates if item.startswith("seg_")]
                    if not legacy_candidates or legacy_candidates == candidates:
                        raise
                    resolved = resolve_context_receipt_source_set(self._run, legacy_candidates)
                available = resolved.get("available_source_ids")
                if not isinstance(available, list):
                    raise ContextDependenceError("source resolver returned no canonical source IDs")
                basis_messages = resolved.get("basis_messages")
                if not isinstance(basis_messages, list) or not all(isinstance(item, Mapping) for item in basis_messages):
                    raise ContextDependenceError("source resolver returned no clean baseline message list")
                # The resolver owns prompt-visible source mutation.  Use its
                # clean baseline too, otherwise private journal keys would
                # differ between baseline and arm even though no source bytes
                # changed.
                self._messages = deepcopy(basis_messages)
                self._full_context_hash = _canonical_digest({"messages": self._messages, "block": self._block})
                # New resolvers may intentionally reject legacy segment IDs
                # in favour of sub-message source IDs.  Their returned list is
                # authoritative, and their range/source metadata is retained.
                self._sources = _source_records(resolved, messages=self._messages)
                self._source_view = str(resolved.get("basis") or "unknown")
            except (ContextReceiptSourceResolutionError, ContextDependenceError) as exc:
                # Preserve v1's fail-closed caller experience: a requested
                # source is reported as unknown before a score pass.  Keep the
                # resolver message for the error rather than deleting based on
                # stale receipt metadata.
                self._source_resolution_error = str(exc)

        self._source_by_id = {source["source_id"]: source for source in self._sources}

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source["source_id"] for source in self._sources)

    def _ensure_baseline(self) -> tuple[list[dict], list[dict], dict]:
        if self._baseline_tokens is not None and self._baseline_evidence is not None and self._continuation is not None:
            return self._baseline_tokens, self._baseline_evidence, self._continuation
        if self._conditions.get("continuation_ids") is None and not self._conditions.get("response"):
            raise MeasurementUnavailableError("the run has neither recorded continuation IDs nor response text")
        tokens, ok = rederive.score_arm(
            self._sub,
            self._conditions,
            messages=_copy_messages(self._messages),
            block=deepcopy(self._block),
            steer_strengths=deepcopy(self._conditions.get("steer_strengths") or {}),
        )
        if not ok:
            raise MeasurementUnavailableError("teacher-forced score_tokens is unavailable")
        baseline = _validate_scored_tokens(tokens, expected_ids=self._conditions.get("continuation_ids"))
        evidence, scored_text = _baseline_token_evidence(baseline)
        recorded_text = self._conditions.get("response")
        recorded_text = recorded_text if isinstance(recorded_text, str) else ""
        if scored_text != recorded_text:
            kind = "exact recorded continuation IDs" if self._conditions.get("continuation_ids") is not None else "retokenized recorded response"
            raise MeasurementUnavailableError(
                f"the {kind} did not decode to the recorded response text"
            )
        continuation_ids = self._conditions.get("continuation_ids")
        continuation = {
            "kind": "recorded_token_ids" if continuation_ids is not None else "recorded_response_text_retokenized",
            "fidelity": (
                "exact_recorded_token_ids" if continuation_ids is not None
                else "recomputed_from_recorded_response_text"
            ),
            "token_ids_exact": continuation_ids is not None,
            "retokenized": bool(self._conditions.get("retokenized", True)),
            "recorded_text": recorded_text,
            "scored_text": scored_text,
            "unicode_offset_basis": "recorded_response_unicode",
        }
        if isinstance(continuation_ids, list):
            continuation["token_ids"] = list(continuation_ids)
        self._baseline_tokens = baseline
        self._baseline_evidence = evidence
        self._continuation = continuation
        return baseline, evidence, continuation

    def _experiment_identity(self, *, removed_ids: list[str], continuation: Mapping[str, Any]) -> str:
        binding = {
            "schema_version": SCHEMA,
            "context_model_identity": self._context_identity,
            "full_context_hash": self._full_context_hash,
            "recorded_continuation": continuation,
            "intervention_operator": INTERVENTION_OPERATOR,
            "removed_source_ids": removed_ids,
        }
        return f"cdx_{_canonical_digest(binding)[:24]}"

    def _neutralization_control_identity(self, *, source_ids: list[str],
                                         continuation: Mapping[str, Any]) -> str:
        binding = {
            "schema_version": SCHEMA,
            "context_model_identity": self._context_identity,
            "full_context_hash": self._full_context_hash,
            "recorded_continuation": continuation,
            "intervention_operator": NEUTRALIZATION_OPERATOR,
            "source_ids": source_ids,
            "strategy": NEUTRALIZATION_STRATEGY,
            "recipe": NEUTRALIZATION_RECIPE,
            "length_contract": NEUTRALIZATION_LENGTH_CONTRACT,
        }
        return f"cdrc_{_canonical_digest(binding)[:24]}"

    def _resolve_arm(self, removed_ids: list[str]) -> Mapping[str, Any]:
        try:
            resolved = resolve_context_receipt_source_set(self._run, removed_ids)
        except ContextReceiptSourceResolutionError as exc:
            raise UnknownSourceIdError(str(exc)) from None
        canonical = resolved.get("canonical_source_ids")
        if canonical != removed_ids:
            raise ContextDependenceError("source resolver did not preserve the canonical requested source set")
        arm_messages = resolved.get("messages")
        if not isinstance(arm_messages, list):
            raise ContextDependenceError("source resolver returned no mutated message list")
        exact_ranges = resolved.get("exact_removed_ranges")
        if not isinstance(exact_ranges, list) or len(exact_ranges) != len(removed_ids):
            raise ContextDependenceError("source resolver did not return exact evidence for every removed source")
        return resolved

    def _resolve_neutralization_control(self, source_ids: list[str]) -> Mapping[str, Any]:
        try:
            resolved = neutralize_context_receipt_sources(self._run, source_ids)
        except ContextReceiptSourceResolutionError as exc:
            raise UnknownSourceIdError(str(exc)) from None
        if resolved.get("canonical_source_ids") != source_ids:
            raise ContextDependenceError(
                "neutralization resolver did not preserve the canonical requested source set")
        if not isinstance(resolved.get("messages"), list):
            raise ContextDependenceError("neutralization resolver returned no mutated message list")
        exact_ranges = resolved.get("exact_neutralized_ranges")
        if not isinstance(exact_ranges, list) or len(exact_ranges) != len(source_ids):
            raise ContextDependenceError(
                "neutralization resolver did not return exact evidence for every source")
        neutralization = resolved.get("neutralization")
        if not isinstance(neutralization, Mapping) or neutralization.get("operator") != NEUTRALIZATION_OPERATOR:
            raise ContextDependenceError("neutralization resolver did not identify its intervention")
        return resolved

    def _score_intervened_messages(self, *, messages: list, baseline: list[dict]) -> tuple[list[dict], float]:
        """Score one already-resolved arm and prove continuation alignment."""
        arm_started = self._clock()
        arm_tokens, arm_ok = rederive.score_arm(
            self._sub,
            self._conditions,
            messages=deepcopy(messages),
            block=deepcopy(self._block),
            steer_strengths=deepcopy(self._conditions.get("steer_strengths") or {}),
        )
        score_ms = max(0.0, (self._clock() - arm_started) * 1000.0)
        if not arm_ok:
            raise MeasurementUnavailableError("the intervened teacher-forced score call did not complete")
        arm = _validate_scored_tokens(arm_tokens, expected_ids=self._conditions.get("continuation_ids"))
        if len(arm) != len(baseline):
            raise MeasurementUnavailableError("the intervened score did not align with the baseline")
        for base, intervened in zip(baseline, arm):
            # Text-fallback scoring may retokenize at a prompt boundary. A
            # full vector is projectable only when the arm preserves the same
            # token pieces (and IDs whenever the scorer supplies them) as the
            # baseline. Equal lengths alone would let unrelated tokenizations
            # masquerade as position-wise evidence.
            if base["piece"] != intervened["piece"]:
                raise MeasurementUnavailableError(
                    "the intervened score did not preserve baseline continuation token pieces"
                )
            base_id, intervened_id = base.get("token_id"), intervened.get("token_id")
            if (base_id is None) != (intervened_id is None) or (
                base_id is not None and base_id != intervened_id
            ):
                raise MeasurementUnavailableError(
                    "the intervened score did not preserve baseline continuation token IDs"
                )
        return arm, score_ms

    def measure_removal_effect(self, removed_source_ids: Iterable[str]) -> dict:
        """Teacher-force one canonical deletion arm over the full continuation."""
        removed_ids = _removed_source_set(removed_source_ids)
        unknown = [source_id for source_id in removed_ids if source_id not in self._source_by_id]
        if unknown:
            available = ", ".join(self.source_ids) or "none"
            detail = f" ({self._source_resolution_error})" if self._source_resolution_error else ""
            raise UnknownSourceIdError(
                "unknown canonical Context Receipt source ID(s): "
                f"{', '.join(unknown)}; available: {available}{detail}"
            )
        baseline, _evidence, continuation = self._ensure_baseline()
        experiment_id = self._experiment_identity(removed_ids=removed_ids, continuation=continuation)
        cached = self._experiments_by_id.get(experiment_id)
        if cached is not None:
            return deepcopy(cached)

        resolved = self._resolve_arm(removed_ids)
        arm, score_ms = self._score_intervened_messages(messages=resolved["messages"], baseline=baseline)

        per_token = [
            base["logprob"] - intervened["logprob"]
            for base, intervened in zip(baseline, arm)
        ]
        baseline_logp = sum(token["logprob"] for token in baseline)
        intervened_logp = sum(token["logprob"] for token in arm)
        # Score identity includes the explicit block where the scoring view is
        # raw messages.  The bridge's intervened digest covers only its message
        # basis, so it is useful provenance but must not replace this hash.
        context_hash = _canonical_digest({"messages": resolved["messages"], "block": self._block})
        experiment = {
            "experiment_id": experiment_id,
            "intervention_operator": INTERVENTION_OPERATOR,
            "removed_source_ids": removed_ids,
            "exact_removed_ranges": deepcopy(resolved["exact_removed_ranges"]),
            "context_hash": context_hash,
            "intervened_logp": intervened_logp,
            "baseline_logp": baseline_logp,
            "delta_nats": sum(per_token),
            "per_token_delta_nats": per_token,
            "token_indices": list(range(len(per_token))),
            "provenance": PROVENANCE,
            "score_ms": score_ms,
        }
        stored = deepcopy(experiment)
        self._experiments.append(stored)
        self._experiments_by_id[experiment_id] = stored
        return deepcopy(stored)

    def measure_neutralization_control(self, source_ids: Iterable[str]) -> dict:
        """Score a separately-labelled matched-length neutralization control.

        This never augments or reinterprets ``experiments``: deletion remains
        the canonical Context Dependence quantity.  The returned evidence
        lives in ``robustness_controls`` and carries its own operator,
        strategy, length contract, exact ranges, vectors, and context hash.
        """
        canonical_ids = _removed_source_set(source_ids)
        unknown = [source_id for source_id in canonical_ids if source_id not in self._source_by_id]
        if unknown:
            available = ", ".join(self.source_ids) or "none"
            detail = f" ({self._source_resolution_error})" if self._source_resolution_error else ""
            raise UnknownSourceIdError(
                "unknown canonical Context Receipt source ID(s): "
                f"{', '.join(unknown)}; available: {available}{detail}"
            )
        baseline, _evidence, continuation = self._ensure_baseline()
        control_id = self._neutralization_control_identity(
            source_ids=canonical_ids, continuation=continuation,
        )
        cached = self._robustness_controls_by_id.get(control_id)
        if cached is not None:
            return deepcopy(cached)

        resolved = self._resolve_neutralization_control(canonical_ids)
        arm, score_ms = self._score_intervened_messages(messages=resolved["messages"], baseline=baseline)
        per_token = [
            base["logprob"] - intervened["logprob"]
            for base, intervened in zip(baseline, arm)
        ]
        control = {
            "control_id": control_id,
            "intervention_operator": NEUTRALIZATION_OPERATOR,
            "provenance": NEUTRALIZATION_PROVENANCE,
            "neutralized_source_ids": canonical_ids,
            "exact_neutralized_ranges": deepcopy(resolved["exact_neutralized_ranges"]),
            "effective_neutralized_ranges": deepcopy(resolved["effective_neutralized_ranges"]),
            "neutralization": deepcopy(resolved["neutralization"]),
            "context_hash": _canonical_digest({"messages": resolved["messages"], "block": self._block}),
            "intervened_logp": sum(token["logprob"] for token in arm),
            "baseline_logp": sum(token["logprob"] for token in baseline),
            "delta_nats": sum(per_token),
            "per_token_delta_nats": per_token,
            "token_indices": list(range(len(per_token))),
            "basis_digest": resolved["basis_digest"],
            "intervened_context_digest": resolved["intervened_context_digest"],
            "score_ms": score_ms,
        }
        stored = deepcopy(control)
        self._robustness_controls.append(stored)
        self._robustness_controls_by_id[control_id] = stored
        return deepcopy(stored)

    def document(self) -> dict:
        """Return the accumulated schema-valid target-agnostic v2 artifact."""
        baseline, evidence, continuation = self._ensure_baseline()
        study_binding = {
            "schema_version": SCHEMA,
            "run_id": self._run.get("id"),
            "context_model_identity": self._context_identity,
            "full_context_hash": self._full_context_hash,
            "continuation": continuation,
            "source_identity": self._sources,
        }
        def is_span(source: Mapping[str, Any]) -> bool:
            if source.get("source_kind") == "span" or str(source.get("source_id", "")).startswith("src_"):
                return True
            index, offsets = source.get("message_index"), source.get("unicode_range")
            if not isinstance(index, int) or not isinstance(offsets, list) or len(offsets) != 2:
                return False
            if not (0 <= index < len(self._messages)) or not isinstance(self._messages[index], Mapping):
                return False
            content = self._messages[index].get("content")
            return isinstance(content, str) and offsets != [0, len(content)]

        source_kind = "context_receipt_source_span" if any(
            is_span(source) for source in self._sources if isinstance(source, Mapping)
        ) else "context_receipt_segment_id"
        document = {
            "schema_version": SCHEMA,
            "study_id": f"cds_{_canonical_digest(study_binding)[:24]}",
            "run_id": str(self._run.get("id") or ""),
            "context_model_identity": deepcopy(self._context_identity),
            "source_identity": {
                "kind": source_kind,
                "view": self._source_view,
                "sources": deepcopy(self._sources),
            },
            "continuation": deepcopy(continuation),
            "baseline": {
                "teacher_forced_logp": sum(token["logprob"] for token in baseline),
                "context_hash": self._full_context_hash,
                "scored_once": True,
                "provenance": PROVENANCE,
                "tokens": deepcopy(evidence),
            },
            "experiments": deepcopy(self._experiments),
            "robustness_controls": deepcopy(self._robustness_controls),
            "budget": {
                "passes_requested": 1 + len(self._experiments) + len(self._robustness_controls),
                "passes_consumed": 1 + len(self._experiments) + len(self._robustness_controls),
            },
        }
        schemas.validate(document)
        return document


def measure_removal_effect(run: dict, sub, *, target: dict | None = None,
                           removed_source_ids: Iterable[str], clock=time.perf_counter) -> dict:
    """Measure one deletion set as a one-experiment run-level v2 study.

    ``target`` remains in this call signature only to give legacy callers a
    clear migration error.  It never affects scoring or identity.
    """
    study = ContextDependenceStudy(run, sub, target=target, clock=clock)
    study.measure_removal_effect(removed_source_ids)
    return study.document()


build_context_dependence_study = measure_removal_effect


def measure_neutralization_control(run: dict, sub, *, source_ids: Iterable[str],
                                   target: dict | None = None, clock=time.perf_counter) -> dict:
    """Build a one-control v2 study without changing delete-source semantics."""
    study = ContextDependenceStudy(run, sub, target=target, clock=clock)
    study.measure_neutralization_control(source_ids)
    return study.document()


__all__ = [
    "ContextDependenceError",
    "ContextDependenceStudy",
    "INTERVENTION_OPERATOR",
    "NEUTRALIZATION_OPERATOR",
    "NEUTRALIZATION_PROVENANCE",
    "MeasurementUnavailableError",
    "PROVENANCE",
    "SCHEMA",
    "UnknownSourceIdError",
    "build_context_dependence_study",
    "measure_removal_effect",
    "measure_neutralization_control",
]
