"""Stable typed portions of Clozn gateway and engine responses."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

from ._transport import CloznProtocolError

JsonObject = dict[str, Any]


def require_object(value: Any, label: str) -> JsonObject:
    if not isinstance(value, Mapping):
        raise CloznProtocolError(f"{label} must be a JSON object")
    return dict(value)


def _objects(value: Any, label: str) -> tuple[JsonObject, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise CloznProtocolError(f"{label} must be a JSON array")
    return tuple(require_object(item, f"{label}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True)
class Run:
    id: str
    source: str | None
    model: str | None
    response: str | None
    created_at: str | None
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "Run":
        obj = require_object(value, "run")
        run_id = obj.get("id")
        if not isinstance(run_id, str) or not run_id:
            raise CloznProtocolError("run.id must be a non-empty string")
        return cls(
            id=run_id,
            source=obj.get("source") if isinstance(obj.get("source"), str) else None,
            model=obj.get("model") if isinstance(obj.get("model"), str) else None,
            response=obj.get("response") if isinstance(obj.get("response"), str) else None,
            created_at=obj.get("created_at") if isinstance(obj.get("created_at"), str) else None,
            raw=obj,
        )


@dataclass(frozen=True)
class RunPage:
    runs: tuple[Run, ...]
    cursor: str | None
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "RunPage":
        obj = require_object(value, "run page")
        rows = obj.get("runs", [])
        if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
            raise CloznProtocolError("run page.runs must be an array")
        cursor = obj.get("cursor") or obj.get("next_cursor")
        return cls(
            runs=tuple(Run.from_json(row) for row in rows),
            cursor=cursor if isinstance(cursor, str) else None,
            raw=obj,
        )


@dataclass(frozen=True)
class LatestRun:
    available: bool
    run: Run | None
    association: JsonObject
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "LatestRun":
        obj = require_object(value, "latest run response")
        available = bool(obj.get("available"))
        run_value = obj.get("run")
        run = None if run_value is None else Run.from_json(run_value)
        if available != (run is not None):
            raise CloznProtocolError("latest run availability disagrees with the run payload")
        association = obj.get("association", {})
        return cls(available, run, require_object(association, "association"), obj)


@dataclass(frozen=True)
class Timeline:
    run_id: str
    events: tuple[JsonObject, ...]
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "Timeline":
        obj = require_object(value, "timeline")
        run_id = obj.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise CloznProtocolError("timeline.run_id must be a non-empty string")
        return cls(run_id, _objects(obj.get("events", []), "timeline.events"), obj)


@dataclass(frozen=True)
class ReceiptBundle:
    raw: JsonObject

    @classmethod
    def from_json(cls, value: Any) -> "ReceiptBundle":
        return cls(require_object(value, "receipt bundle"))


@dataclass(frozen=True)
class AttentionKnockout:
    """One native attention-edge cut applied during teacher-forced scoring."""

    layer: int
    queries: tuple[int, ...]
    keys: tuple[int, ...]
    renormalize: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.layer, int) or isinstance(self.layer, bool) or self.layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if not self.queries:
            raise ValueError("queries must not be empty")
        if not self.keys:
            raise ValueError("keys must not be empty")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in (*self.queries, *self.keys)):
            raise ValueError("query/key positions must be non-negative integers")
        if not isinstance(self.renormalize, bool):
            raise ValueError("renormalize must be a bool")

    def to_wire(self) -> JsonObject:
        return {
            "layer": int(self.layer),
            "queries": sorted({int(value) for value in self.queries}),
            "keys": sorted({int(value) for value in self.keys}),
            "renormalize": self.renormalize,
        }


@dataclass(frozen=True)
class ScoreToken:
    id: int
    piece: str
    logprob: float
    topk: tuple[JsonObject, ...]
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "ScoreToken":
        obj = require_object(value, "score token")
        token_id = obj.get("id")
        piece = obj.get("piece")
        logprob = obj.get("logprob")
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise CloznProtocolError("score token.id must be an integer")
        if not isinstance(piece, str):
            raise CloznProtocolError("score token.piece must be a string")
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise CloznProtocolError("score token.logprob must be a number")
        return cls(token_id, piece, float(logprob), _objects(obj.get("topk", []), "score token.topk"), obj)


@dataclass(frozen=True)
class ScoreResult:
    n_prompt: int
    n_cont: int
    tokens: tuple[ScoreToken, ...]
    sum_logprob: float
    boundary_approximate: bool
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(cls, value: Any) -> "ScoreResult":
        obj = require_object(value, "score response")
        n_prompt = obj.get("n_prompt")
        n_cont = obj.get("n_cont")
        sum_logprob = obj.get("sum_logprob")
        if not isinstance(n_prompt, int) or isinstance(n_prompt, bool) or n_prompt < 0:
            raise CloznProtocolError("score.n_prompt must be a non-negative integer")
        if not isinstance(n_cont, int) or isinstance(n_cont, bool) or n_cont < 0:
            raise CloznProtocolError("score.n_cont must be a non-negative integer")
        if not isinstance(sum_logprob, (int, float)) or isinstance(sum_logprob, bool):
            raise CloznProtocolError("score.sum_logprob must be a number")
        rows = obj.get("tokens", [])
        if isinstance(rows, (str, bytes, bytearray)) or not isinstance(rows, Sequence):
            raise CloznProtocolError("score.tokens must be an array")
        tokens = tuple(ScoreToken.from_json(row) for row in rows)
        if len(tokens) != n_cont:
            raise CloznProtocolError("score.n_cont disagrees with score.tokens")
        return cls(n_prompt, n_cont, tokens, float(sum_logprob),
                   bool(obj.get("boundary_approximate", False)), obj)


_SCORE_VARIANTS_SCHEMA = "clozn.score-variants-worker.v1"
_SCORE_VARIANTS_STATUSES = frozenset({"completed", "partial", "failed", "cancelled"})
_SCORE_VARIANT_ROW_STATUSES = frozenset({
    "completed", "failed", "cancelled", "not_started", "unsupported",
})


def _lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class ScoreVariantResult:
    """One ordered row from an unqualified native variant-scoring screen."""

    id: str
    status: str
    prompt_token_count: int
    target_logprobs: tuple[float, ...] | None
    error: str | None
    raw: JsonObject = field(repr=False)


@dataclass(frozen=True)
class ScoreVariantsExecutionIdentity:
    algorithm_id: str
    worker_generation_id: str
    model_sha256: str | None
    tokenizer_sha256: str | None
    raw: JsonObject = field(repr=False)


@dataclass(frozen=True)
class ScoreVariantsWave:
    wave_index: int
    variant_ids: tuple[str, ...]
    unique_prompt_count: int
    logical_prompt_rows: int
    physical_prompt_rows: int
    continuation_kv_rows: int
    required_kv_rows: int
    raw: JsonObject = field(repr=False)


@dataclass(frozen=True)
class ScoreVariantsExecution:
    """Execution facts that keep native results explicitly screening-only."""

    strategy: str
    numerical_regime: str
    evidence_use: str
    qualified: bool
    execution_identity: ScoreVariantsExecutionIdentity
    waves: tuple[ScoreVariantsWave, ...]
    full_vocab_host_reduction: bool | None
    target_logprob_reduction: str | None
    generation_state_reuse: JsonObject | None
    raw: JsonObject = field(repr=False)


@dataclass(frozen=True)
class ScoreVariantsResult:
    """Validated screening result from ``POST /v1/score-variants``.

    This type can never represent trusted evidence: v1 requires an unqualified native numerical
    regime, ``evidence_use == "screening"``, and ``qualified is False``. Claim-critical or borderline
    effects must be remeasured through scalar :meth:`EngineClient.score`.
    """

    schema_version: str
    status: str
    continuation_ids: tuple[int, ...]
    results: tuple[ScoreVariantResult, ...]
    execution: ScoreVariantsExecution
    raw: JsonObject = field(repr=False)

    @classmethod
    def from_json(
        cls,
        value: Any,
        *,
        variant_ids: Sequence[str],
        prompt_lengths: Sequence[int],
        continuation_ids: Sequence[int],
    ) -> "ScoreVariantsResult":
        obj = require_object(value, "score variants response")
        if obj.get("schema_version") != _SCORE_VARIANTS_SCHEMA:
            raise CloznProtocolError("score variants response has an unsupported schema_version")
        status = obj.get("status")
        if status not in _SCORE_VARIANTS_STATUSES:
            raise CloznProtocolError("score variants response has an invalid top-level status")

        expected_continuation = tuple(continuation_ids)
        echoed = obj.get("continuation_ids")
        if (
            not isinstance(echoed, list)
            or any(not isinstance(token, int) or isinstance(token, bool) for token in echoed)
            or tuple(echoed) != expected_continuation
        ):
            raise CloznProtocolError("score variants response continuation_ids do not match the request")

        raw_rows = obj.get("results")
        if not isinstance(raw_rows, list) or len(raw_rows) != len(variant_ids):
            raise CloznProtocolError("score variants response has the wrong number of result rows")
        if len(prompt_lengths) != len(variant_ids):
            raise CloznProtocolError("score variants request identity is internally inconsistent")

        rows: list[ScoreVariantResult] = []
        completed_count = 0
        for index, (raw, expected_id, prompt_length) in enumerate(
            zip(raw_rows, variant_ids, prompt_lengths)
        ):
            row = require_object(raw, f"score variants results[{index}]")
            if row.get("id") != expected_id:
                raise CloznProtocolError(f"score variants results[{index}] is out of caller order")
            row_status = row.get("status")
            if row_status not in _SCORE_VARIANT_ROW_STATUSES:
                raise CloznProtocolError(f"score variants results[{index}].status is invalid")
            prompt_count = row.get("prompt_token_count")
            if (
                not isinstance(prompt_count, int) or isinstance(prompt_count, bool)
                or prompt_count != prompt_length
            ):
                raise CloznProtocolError(
                    f"score variants results[{index}].prompt_token_count is invalid")

            scores: tuple[float, ...] | None = None
            error = row.get("error") if isinstance(row.get("error"), str) else None
            if row_status == "completed":
                completed_count += 1
                raw_scores = row.get("target_logprobs")
                if (
                    not isinstance(raw_scores, list)
                    or len(raw_scores) != len(expected_continuation)
                    or any(
                        isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                        for score in raw_scores
                    )
                ):
                    raise CloznProtocolError(
                        f"score variants results[{index}].target_logprobs is invalid")
                scores = tuple(float(score) for score in raw_scores)
            elif "target_logprobs" in row:
                raise CloznProtocolError(
                    f"score variants results[{index}] carries scores despite not completing")
            if row_status in {"failed", "unsupported"} and not error:
                raise CloznProtocolError(f"score variants results[{index}].error is required")
            rows.append(ScoreVariantResult(
                id=expected_id,
                status=row_status,
                prompt_token_count=prompt_count,
                target_logprobs=scores,
                error=error,
                raw=row,
            ))

        if status == "completed" and completed_count != len(rows):
            raise CloznProtocolError("score variants completed status disagrees with its rows")
        if status == "partial" and not (0 < completed_count < len(rows)):
            raise CloznProtocolError("score variants partial status disagrees with its rows")
        if status in {"failed", "cancelled"} and completed_count:
            raise CloznProtocolError(f"score variants {status} status disagrees with its rows")

        execution_raw = require_object(obj.get("execution"), "score variants execution")
        strategy = execution_raw.get("strategy")
        if not isinstance(strategy, str) or not strategy.startswith("native_"):
            raise CloznProtocolError("score variants execution.strategy must name a native strategy")
        if (
            execution_raw.get("numerical_regime") != "screening_unqualified"
            or execution_raw.get("evidence_use") != "screening"
            or execution_raw.get("qualified") is not False
        ):
            raise CloznProtocolError("score variants response attempted to promote screening evidence")

        identity_raw = require_object(
            execution_raw.get("execution_identity"), "score variants execution_identity")
        for name in ("algorithm_id", "worker_generation_id"):
            if not isinstance(identity_raw.get(name), str) or not identity_raw[name]:
                raise CloznProtocolError(f"score variants execution_identity.{name} is invalid")
        for name in ("model_sha256", "tokenizer_sha256"):
            if name in identity_raw and not _lower_sha256(identity_raw.get(name)):
                raise CloznProtocolError(f"score variants execution_identity.{name} is invalid")
        identity = ScoreVariantsExecutionIdentity(
            algorithm_id=identity_raw["algorithm_id"],
            worker_generation_id=identity_raw["worker_generation_id"],
            model_sha256=identity_raw.get("model_sha256"),
            tokenizer_sha256=identity_raw.get("tokenizer_sha256"),
            raw=identity_raw,
        )

        raw_waves = execution_raw.get("waves")
        if not isinstance(raw_waves, list) or not raw_waves:
            raise CloznProtocolError("score variants execution.waves must be a non-empty array")
        waves: list[ScoreVariantsWave] = []
        wave_members: list[str] = []
        count_names = (
            "unique_prompt_count", "logical_prompt_rows", "physical_prompt_rows",
            "continuation_kv_rows", "required_kv_rows",
        )
        for wave_index, raw in enumerate(raw_waves):
            wave = require_object(raw, f"score variants execution.waves[{wave_index}]")
            if wave.get("wave_index") != wave_index:
                raise CloznProtocolError(
                    f"score variants execution.waves[{wave_index}].wave_index is invalid")
            members = wave.get("variant_ids")
            if (
                not isinstance(members, list) or not members
                or any(not isinstance(member, str) for member in members)
            ):
                raise CloznProtocolError(
                    f"score variants execution.waves[{wave_index}].variant_ids is invalid")
            counts: dict[str, int] = {}
            for name in count_names:
                count = wave.get(name)
                if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                    raise CloznProtocolError(
                        f"score variants execution.waves[{wave_index}].{name} is invalid")
                counts[name] = count
            wave_members.extend(members)
            waves.append(ScoreVariantsWave(
                wave_index=wave_index,
                variant_ids=tuple(members),
                unique_prompt_count=counts["unique_prompt_count"],
                logical_prompt_rows=counts["logical_prompt_rows"],
                physical_prompt_rows=counts["physical_prompt_rows"],
                continuation_kv_rows=counts["continuation_kv_rows"],
                required_kv_rows=counts["required_kv_rows"],
                raw=wave,
            ))
        if sorted(wave_members) != sorted(variant_ids) or len(wave_members) != len(variant_ids):
            raise CloznProtocolError(
                "score variants execution has incomplete or duplicate wave membership")

        full_vocab_host_reduction = execution_raw.get("full_vocab_host_reduction")
        if full_vocab_host_reduction is not None and not isinstance(
            full_vocab_host_reduction, bool
        ):
            raise CloznProtocolError(
                "score variants execution.full_vocab_host_reduction is invalid")
        target_logprob_reduction = execution_raw.get("target_logprob_reduction")
        if target_logprob_reduction is not None and (
            not isinstance(target_logprob_reduction, str) or not target_logprob_reduction
        ):
            raise CloznProtocolError(
                "score variants execution.target_logprob_reduction is invalid")
        generation_state_raw = execution_raw.get("generation_state_reuse")
        if generation_state_raw is not None and not isinstance(generation_state_raw, Mapping):
            raise CloznProtocolError(
                "score variants execution.generation_state_reuse is invalid")

        execution = ScoreVariantsExecution(
            strategy=strategy,
            numerical_regime="screening_unqualified",
            evidence_use="screening",
            qualified=False,
            execution_identity=identity,
            waves=tuple(waves),
            full_vocab_host_reduction=full_vocab_host_reduction,
            target_logprob_reduction=target_logprob_reduction,
            generation_state_reuse=(
                dict(generation_state_raw) if generation_state_raw is not None else None
            ),
            raw=execution_raw,
        )
        return cls(
            schema_version=_SCORE_VARIANTS_SCHEMA,
            status=status,
            continuation_ids=expected_continuation,
            results=tuple(rows),
            execution=execution,
            raw=obj,
        )


@dataclass(frozen=True)
class HarvestResult:
    """Residual activations returned by ``POST /harvest``."""

    tokens: tuple[str, ...]
    layer: int
    activations: Any = field(repr=False, compare=False)
    raw: JsonObject = field(repr=False, compare=False)

    @property
    def n_tokens(self) -> int:
        return int(self.activations.shape[0])

    @property
    def n_embd(self) -> int:
        return int(self.activations.shape[1])

    @classmethod
    def from_json(cls, value: Any) -> "HarvestResult":
        from .tensors import decode_float32_tensor

        obj = require_object(value, "harvest response")
        layer = obj.get("layer")
        tokens_value = obj.get("tokens")
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise CloznProtocolError("harvest.layer must be a non-negative integer")
        if (isinstance(tokens_value, (str, bytes, bytearray))
                or not isinstance(tokens_value, Sequence)
                or any(not isinstance(token, str) for token in tokens_value)):
            raise CloznProtocolError("harvest.tokens must be a string array")
        activations = decode_float32_tensor(obj.get("activations"), label="harvest.activations")
        if activations.ndim != 2:
            raise CloznProtocolError("harvest.activations must be a rank-2 tensor")
        tokens = tuple(tokens_value)
        if activations.shape[0] != len(tokens):
            raise CloznProtocolError("harvest token count disagrees with activation rows")
        return cls(tokens=tokens, layer=layer, activations=activations, raw=obj)


@dataclass(frozen=True)
class Observation:
    """Prediction movement reported by one native residual write."""

    applied: bool
    layer: int
    moved_l2: float
    baseline_top: tuple[JsonObject, ...]
    edited_top: tuple[JsonObject, ...]
    error: str | None
    raw: JsonObject = field(repr=False)

    @property
    def shifted(self) -> bool:
        return bool(
            self.applied and self.baseline_top and self.edited_top
            and self.baseline_top[0].get("token") != self.edited_top[0].get("token")
        )

    @classmethod
    def from_json(cls, value: Any) -> "Observation":
        obj = require_object(value, "state observation")
        applied = obj.get("applied")
        layer = obj.get("layer")
        moved_l2 = obj.get("moved_l2")
        error = obj.get("error")
        if not isinstance(applied, bool):
            raise CloznProtocolError("state.applied must be a bool")
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise CloznProtocolError("state.layer must be a non-negative integer")
        if not isinstance(moved_l2, (int, float)) or isinstance(moved_l2, bool):
            raise CloznProtocolError("state.moved_l2 must be a number")
        if error is not None and not isinstance(error, str):
            raise CloznProtocolError("state.error must be a string or null")
        return cls(
            applied=applied,
            layer=layer,
            moved_l2=float(moved_l2),
            baseline_top=_objects(obj.get("baseline_top", []), "state.baseline_top"),
            edited_top=_objects(obj.get("edited_top", []), "state.edited_top"),
            error=error,
            raw=obj,
        )


@dataclass(frozen=True)
class PatchArm:
    """One named residual write in a patch sweep."""

    name: str
    positions: tuple[int, ...]
    values: Any = field(repr=False, compare=False)
    metadata: JsonObject = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("patch arm name must be a non-empty string")
        if not self.positions:
            raise ValueError("patch arm positions must not be empty")
        if any(not isinstance(pos, int) or isinstance(pos, bool) or pos < 0 for pos in self.positions):
            raise ValueError("patch arm positions must be non-negative integers")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("patch arm metadata must be an object")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "positions", tuple(self.positions))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PatchArmResult:
    name: str
    positions: tuple[int, ...]
    observation: Observation
    metadata: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class PatchSweepResult:
    harvest: HarvestResult
    arms: tuple[PatchArmResult, ...]
