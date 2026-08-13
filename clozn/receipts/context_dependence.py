"""Direct, set-valued Context Dependence measurements.

This module deliberately sits beside (and does not reinterpret)
``context_answer_influence``.  Its atomic intervention is deletion of one or
more *canonical Context Receipt sources*, not matched-length replacement of a
span.  Every arm teacher-forces the recorded continuation through
``rederive.score_arm``; no generation API is used here.

``ContextDependenceStudy`` is the reusable unit: it scores the full-context
baseline at most once, then can score any number of source sets against that
same target.  The module-level :func:`measure_removal_effect` is the compact
one-experiment convenience wrapper and returns a schema-governed study with
one measured experiment.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import time
from typing import Iterable

from clozn import schemas
from clozn.runs.context_receipt import segment_id as _receipt_segment_id

from . import rederive


SCHEMA = "clozn.context-dependence-study.v1"
INTERVENTION_OPERATOR = "delete_source"
PROVENANCE = "measured"


class ContextDependenceError(ValueError):
    """A direct Context Dependence measurement could not be made faithfully."""


class UnknownSourceIdError(ContextDependenceError):
    """A requested source did not have a canonical identity in this study."""


class MeasurementUnavailableError(ContextDependenceError):
    """The exact recorded continuation could not be teacher-forced and scored."""


def _canonical_digest(value) -> str:
    """A stable digest independent of clocks, mapping insertion order, and host state."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_messages(messages) -> list:
    """Copy score inputs so neither this module nor a substrate can mutate the run record."""
    return deepcopy(messages if isinstance(messages, list) else [])


def _range(value, *, name: str, upper: int | None = None) -> list[int]:
    """Normalize a half-open range accepted as ``[start, end]`` or an object."""
    if isinstance(value, dict):
        value = [value.get("start"), value.get("end")]
    if not (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        raise ContextDependenceError(f"{name} must be a two-integer half-open range")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end <= start or (upper is not None and end > upper):
        suffix = f" within [0, {upper}]" if upper is not None else ""
        raise ContextDependenceError(f"{name} must be non-empty and ordered{suffix}")
    return [start, end]


def _prompt_identity(run: dict) -> dict:
    """Return the available model/runtime/template/rendering identity facets.

    ``identity`` is intentionally retained as recorded (apart from capture time),
    rather than guessing which nested field happens to name a model artifact for
    a particular runtime.  This binds future runtime identity additions without
    a schema migration.
    """
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


def _canonical_sources(run: dict, conditions: dict) -> tuple[list[dict], str]:
    """Resolve deletable messages exclusively through Context Receipt identities.

    ``assembled`` is the authoritative correspondence when teacher forcing uses
    ``assembled_messages``.  For old runs where assembled text is absent, a
    delivered mapping may be used only when the score view is exactly the raw
    message list; that is an existing receipt correspondence, not a text or
    position similarity guess.  Any untraceable message is simply not an
    eligible source, which makes an attempted deletion fail closed.
    """
    receipt = run.get("context_receipt")
    receipt = receipt if isinstance(receipt, dict) else {}
    messages = conditions.get("messages")
    messages = messages if isinstance(messages, list) else []
    block_source = conditions.get("block_source")

    segments = receipt.get("assembled") if block_source == "assembled_messages" else receipt.get("delivered")
    view = "assembled" if block_source == "assembled_messages" else "delivered"
    if not isinstance(segments, list) and block_source == "assembled_messages":
        raw_messages = conditions.get("raw_messages")
        if isinstance(raw_messages, list) and messages == raw_messages:
            segments, view = receipt.get("delivered"), "delivered"
    if not isinstance(segments, list):
        return [], view

    # A receipt may be retained independently from the run record.  Its
    # original_order field is useful only after proving that it still names the
    # exact scoring message at that index.  In particular, never let a stale
    # or tampered receipt turn a canonical source ID into deletion of whatever
    # message happens to occupy the old position today.
    expected_by_index: dict[int, tuple[str, str]] = {}
    occurrences: dict[tuple[str, str], int] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        key = (role, content)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        expected_by_index[index] = (
            _receipt_segment_id(role, content, occurrence=occurrence),
            hashlib.sha256(content.encode("utf-8")).hexdigest()[:16],
        )

    by_index: dict[int, dict] = {}
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        source_id = segment.get("segment_id")
        index = segment.get("original_order")
        if not isinstance(source_id, str) or not source_id:
            continue
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            continue
        expected = expected_by_index.get(index)
        if expected is None:
            # A receipt reference to a non-message (or an out-of-range index)
            # cannot be applied as a source deletion.
            continue
        expected_id, expected_content_hash = expected
        if (
            source_id != expected_id
            or segment.get("content_hash") != expected_content_hash
        ):
            # ``segment_id`` commits role+content+occurrence and
            # ``content_hash`` commits the exact message bytes.  Requiring both
            # makes the receipt-to-score-view correspondence explicit instead
            # of trusting an index alone.
            continue
        # Ambiguous receipt identity is not safe to delete.  Leave it absent
        # so callers receive UnknownSourceIdError instead of a guessed arm.
        if index in by_index or any(item.get("segment_id") == source_id for item in by_index.values()):
            return [], view
        by_index[index] = segment

    sources = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        segment = by_index.get(index)
        if segment is None:
            continue
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        source_id = segment["segment_id"]
        source = {
            "source_id": source_id,
            "segment_id": source_id,
            "message_index": index,
            "unicode_range": [0, len(content)],
            "byte_range": [0, len(content.encode("utf-8"))],
            "role": str(message.get("role") or ""),
        }
        if isinstance(segment.get("client_source_id"), str) and segment["client_source_id"]:
            source["client_source_id"] = segment["client_source_id"]
        if isinstance(segment.get("source_label"), str) and segment["source_label"]:
            source["source_label"] = segment["source_label"]
        sources.append(source)
    return sources, view


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


def _validate_scored_tokens(tokens: list, *, expected_ids: list[int] | None) -> list[dict]:
    if not isinstance(tokens, list) or not tokens:
        raise MeasurementUnavailableError("the scorer returned no teacher-forced token log-probabilities")
    if expected_ids is not None and len(tokens) != len(expected_ids):
        raise MeasurementUnavailableError("the scorer did not return one score per recorded token ID")
    validated = []
    for index, token in enumerate(tokens):
        if not isinstance(token, dict):
            raise MeasurementUnavailableError("the scorer returned a malformed token record")
        logprob = token.get("logprob")
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)) or not math.isfinite(logprob):
            raise MeasurementUnavailableError("the scorer returned a non-finite token log-probability")
        token_id = token.get("id")
        if expected_ids is not None and token_id != expected_ids[index]:
            raise MeasurementUnavailableError("the scorer did not preserve the recorded continuation IDs")
        validated.append({
            "index": index,
            "token_id": token_id,
            "piece": str(token.get("piece", "")),
            "logprob": float(logprob),
        })
    return validated


def _normalize_target(target, baseline: list[dict]) -> dict:
    """Bind a target to exact teacher-forced token and Unicode ranges."""
    source = target if isinstance(target, dict) else {}
    if target is not None and not isinstance(target, dict):
        raise ContextDependenceError("target must be an object when supplied")
    token_count = len(baseline)
    token_range = _range(
        source.get("recorded_token_range", [0, token_count]),
        name="target.recorded_token_range", upper=token_count,
    )
    starts = []
    cursor = 0
    for token in baseline:
        starts.append((cursor, cursor + len(token["piece"])))
        cursor += len(token["piece"])
    unicode_range = [starts[token_range[0]][0], starts[token_range[1] - 1][1]]
    if "unicode_range" in source:
        supplied = _range(source["unicode_range"], name="target.unicode_range", upper=cursor)
        if supplied != unicode_range:
            raise ContextDependenceError(
                "target.unicode_range must exactly cover target.recorded_token_range"
            )
    prefix_range = [0, token_range[0]]
    if "recorded_prefix_range" in source:
        supplied_prefix = source["recorded_prefix_range"]
        if isinstance(supplied_prefix, dict):
            supplied_prefix = [supplied_prefix.get("start"), supplied_prefix.get("end")]
        if not (
            isinstance(supplied_prefix, (list, tuple)) and len(supplied_prefix) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) for item in supplied_prefix)
            and list(supplied_prefix) == prefix_range
        ):
            raise ContextDependenceError(
                "target.recorded_prefix_range must be the exact [0, target start) conditioned prefix"
            )
    return {
        "unicode_range": unicode_range,
        "recorded_token_range": token_range,
        "recorded_prefix_range": prefix_range,
        "unicode_offset_basis": "teacher_forced_scored_continuation",
    }


def _delete_sources(messages: list, source_by_id: dict[str, dict], removed_ids: list[str]) -> tuple[list, list[dict]]:
    removed = [source_by_id[source_id] for source_id in removed_ids]
    removed_indices = {item["message_index"] for item in removed}
    # Filtering preserves the exact relative ordering of every remaining source.
    remaining = [deepcopy(message) for index, message in enumerate(messages) if index not in removed_indices]
    ranges = []
    for source in sorted(removed, key=lambda item: (item["message_index"], item["source_id"])):
        ranges.append({
            "source_id": source["source_id"],
            "message_index": source["message_index"],
            "unicode_range": list(source["unicode_range"]),
            "byte_range": list(source["byte_range"]),
        })
    return remaining, ranges


class ContextDependenceStudy:
    """One target and one cached full-context baseline for measured source sets."""

    def __init__(self, run: dict, sub, *, target: dict | None = None, clock=time.perf_counter):
        if not isinstance(run, dict) or not run:
            raise ContextDependenceError("a non-empty recorded run is required")
        if not isinstance(run.get("id"), str) or not run["id"]:
            raise ContextDependenceError("a recorded run ID is required for a persistable study")
        self._run = deepcopy(run)
        self._sub = sub
        self._clock = clock
        self._conditions = deepcopy(rederive.with_arm_conditions(self._run))
        self._messages = _copy_messages(self._conditions.get("messages"))
        self._block = deepcopy(self._conditions.get("block"))
        self._target_input = deepcopy(target)
        self._baseline_tokens: list[dict] | None = None
        self._target: dict | None = None
        self._experiments: list[dict] = []
        self._experiments_by_id: dict[str, dict] = {}
        self._sources, self._source_view = _canonical_sources(self._run, self._conditions)
        self._source_by_id = {source["source_id"]: source for source in self._sources}
        self._context_identity = _prompt_identity(self._run)
        self._full_context_hash = _canonical_digest({"messages": self._messages, "block": self._block})

    @property
    def source_ids(self) -> tuple[str, ...]:
        """The canonical receipt IDs eligible for direct deletion, in prompt order."""
        return tuple(source["source_id"] for source in self._sources)

    def _ensure_baseline(self) -> tuple[list[dict], dict]:
        if self._baseline_tokens is not None and self._target is not None:
            return self._baseline_tokens, self._target
        conditions = self._conditions
        if conditions.get("continuation_ids") is None and not conditions.get("response"):
            raise MeasurementUnavailableError("the run has neither recorded continuation IDs nor response text")
        tokens, ok = rederive.score_arm(
            self._sub,
            conditions,
            messages=_copy_messages(self._messages),
            block=deepcopy(self._block),
            steer_strengths=deepcopy(conditions.get("steer_strengths") or {}),
        )
        if not ok:
            raise MeasurementUnavailableError("teacher-forced score_tokens is unavailable")
        self._baseline_tokens = _validate_scored_tokens(
            tokens, expected_ids=conditions.get("continuation_ids"),
        )
        # Exact recorded IDs alone do not make character offsets auditable: the
        # scorer also has to decode those IDs to the recorded continuation we
        # name in the artifact.  A mismatch would leave the persisted target
        # unicode range addressing scorer-only text rather than the recorded
        # answer, so fail closed rather than silently relabeling it as exact.
        if conditions.get("continuation_ids") is not None:
            recorded_text = conditions.get("response")
            scored_text = "".join(token["piece"] for token in self._baseline_tokens)
            if isinstance(recorded_text, str) and recorded_text and scored_text != recorded_text:
                raise MeasurementUnavailableError(
                    "the exact recorded continuation IDs did not decode to the recorded response text"
                )
        self._target = _normalize_target(self._target_input, self._baseline_tokens)
        return self._baseline_tokens, self._target

    def _experiment_identity(self, *, removed_ids: list[str], target: dict) -> str:
        continuation_ids = self._conditions.get("continuation_ids")
        continuation = (
            {"kind": "recorded_token_ids", "token_ids": list(continuation_ids)}
            if isinstance(continuation_ids, list)
            else {"kind": "recorded_response_text_retokenized_approximate", "text": self._conditions.get("response", "")}
        )
        binding = {
            "schema_version": SCHEMA,
            "context_model_identity": self._context_identity,
            "full_context_hash": self._full_context_hash,
            "recorded_continuation": continuation,
            "target": target,
            "intervention_operator": INTERVENTION_OPERATOR,
            "removed_source_ids": removed_ids,
        }
        return f"cdx_{_canonical_digest(binding)[:24]}"

    def measure_removal_effect(self, removed_source_ids: Iterable[str]) -> dict:
        """Measure one exact set deletion and return the measured experiment record.

        A source set is canonicalized lexicographically for identity, while the
        actual deletion preserves the original receipt/message ordering of all
        remaining messages.  Unknown IDs raise before *any* score call.
        """
        removed_ids = _removed_source_set(removed_source_ids)
        unknown = [source_id for source_id in removed_ids if source_id not in self._source_by_id]
        if unknown:
            available = ", ".join(self.source_ids) or "none"
            raise UnknownSourceIdError(
                f"unknown canonical Context Receipt source ID(s): {', '.join(unknown)}; available: {available}"
            )
        baseline, target = self._ensure_baseline()
        experiment_id = self._experiment_identity(removed_ids=removed_ids, target=target)
        cached = self._experiments_by_id.get(experiment_id)
        if cached is not None:
            # Multiple search paths may request the same source set.  It is
            # one content-addressed direct experiment, not another score pass
            # or a duplicate audit record.
            return deepcopy(cached)
        arm_messages, exact_ranges = _delete_sources(self._messages, self._source_by_id, removed_ids)
        arm_started = self._clock()
        arm_tokens, arm_ok = rederive.score_arm(
            self._sub,
            self._conditions,
            messages=arm_messages,
            block=deepcopy(self._block),
            steer_strengths=deepcopy(self._conditions.get("steer_strengths") or {}),
        )
        score_ms = max(0.0, (self._clock() - arm_started) * 1000.0)
        if not arm_ok:
            raise MeasurementUnavailableError("the intervened teacher-forced score call did not complete")
        arm = _validate_scored_tokens(arm_tokens, expected_ids=self._conditions.get("continuation_ids"))
        if len(arm) != len(baseline):
            raise MeasurementUnavailableError("the intervened score did not align with the baseline")

        start, end = target["recorded_token_range"]
        per_token = [
            baseline[index]["logprob"] - arm[index]["logprob"]
            for index in range(start, end)
        ]
        # Deliberately aggregate from the same per-token values that are
        # persisted, so callers can exactly audit the stated aggregate.
        delta_nats = sum(per_token)
        baseline_logp = sum(token["logprob"] for token in baseline[start:end])
        intervened_logp = sum(token["logprob"] for token in arm[start:end])
        experiment = {
            "experiment_id": experiment_id,
            "intervention_operator": INTERVENTION_OPERATOR,
            "removed_source_ids": removed_ids,
            "exact_removed_ranges": exact_ranges,
            "context_hash": _canonical_digest({"messages": arm_messages, "block": self._block}),
            "target_logp": intervened_logp,
            "baseline_target_logp": baseline_logp,
            "delta_nats": delta_nats,
            "per_target_token_delta_nats": per_token,
            "target_token_indices": list(range(start, end)),
            "provenance": PROVENANCE,
            "score_ms": score_ms,
        }
        stored = deepcopy(experiment)
        self._experiments.append(stored)
        self._experiments_by_id[experiment_id] = stored
        return deepcopy(stored)

    def document(self) -> dict:
        """Return the schema-valid study artifact accumulated so far.

        The document intentionally omits unimplemented screen and verified-set
        estimators; this Task 1 primitive contains only direct measurements.
        """
        baseline, target = self._ensure_baseline()
        target_start, target_end = target["recorded_token_range"]
        continuation_ids = self._conditions.get("continuation_ids")
        continuation = {
            "kind": (
                "recorded_token_ids" if continuation_ids is not None
                else "recorded_response_text_retokenized_approximate"
            ),
            "token_ids_exact": continuation_ids is not None,
            "retokenized": bool(self._conditions.get("retokenized", True)),
            "recorded_text": str(self._conditions.get("response") or ""),
        }
        if continuation_ids is not None:
            continuation["token_ids"] = list(continuation_ids)
        study_binding = {
            "schema_version": SCHEMA,
            "run_id": self._run.get("id"),
            "context_model_identity": self._context_identity,
            "full_context_hash": self._full_context_hash,
            "continuation": continuation,
            "target": target,
        }
        document = {
            "schema_version": SCHEMA,
            "study_id": f"cds_{_canonical_digest(study_binding)[:24]}",
            "run_id": str(self._run.get("id") or ""),
            "context_model_identity": self._context_identity,
            "source_identity": {
                "kind": "context_receipt_segment_id",
                "view": self._source_view,
                "sources": deepcopy(self._sources),
            },
            "target": target,
            "continuation": continuation,
            "baseline": {
                "teacher_forced_logp": sum(token["logprob"] for token in baseline[target_start:target_end]),
                "context_hash": self._full_context_hash,
                "scored_once": True,
                "provenance": PROVENANCE,
            },
            "experiments": deepcopy(self._experiments),
            "budget": {
                "passes_requested": 1 + len(self._experiments),
                "passes_consumed": 1 + len(self._experiments),
            },
        }
        # This producer owns the schema; validate before an untrusted caller can
        # persist the artifact.  The pure addition does not touch legacy maps.
        schemas.validate(document)
        return document


def measure_removal_effect(run: dict, sub, *, target: dict | None = None,
                           removed_source_ids: Iterable[str], clock=time.perf_counter) -> dict:
    """Measure one canonical source set and return a one-experiment study artifact.

    For repeated set measurements, construct :class:`ContextDependenceStudy`
    directly, call ``study.measure_removal_effect(...)`` for each set, then
    call ``study.document()``.  That path reuses one baseline score.
    """
    study = ContextDependenceStudy(run, sub, target=target, clock=clock)
    study.measure_removal_effect(removed_source_ids)
    return study.document()


build_context_dependence_study = measure_removal_effect
