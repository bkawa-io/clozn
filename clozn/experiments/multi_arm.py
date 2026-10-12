"""Generic substrate multi-arm dispatch.

This layer only schedules already-prepared semantic requests.  It does not
know recipes, source IDs, or how evidence is classified.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import math
from typing import Any


class MultiArmError(RuntimeError):
    def __init__(self, message: str, *, arm_index: int | None = None,
                 completed: Iterable[Any] = (),
                 completed_indices: Iterable[int] = (),
                 dispatched_indices: Iterable[int] = ()):
        super().__init__(message)
        self.arm_index = arm_index
        self.completed = list(completed)
        self.completed_indices = tuple(int(index) for index in completed_indices)
        self.dispatched_indices = tuple(int(index) for index in dispatched_indices)


class BatchCancelled(RuntimeError):
    def __init__(self, message: str = "multi-arm dispatch cancelled", *,
                 completed: Iterable[Any] = (), next_index: int | None = None,
                 completed_indices: Iterable[int] = (),
                 dispatched_indices: Iterable[int] = ()):
        super().__init__(message)
        self.completed = list(completed)
        self.next_index = next_index
        self.completed_indices = tuple(int(index) for index in completed_indices)
        self.dispatched_indices = tuple(int(index) for index in dispatched_indices)


@dataclass(frozen=True)
class ScoreTokensManyResult:
    """Score-many results with their execution and evidence contract.

    ``metadata`` is the substrate's returned envelope with ``results``
    removed.  Scalar reference execution has an in-process contract, while a
    native implementation must return its numerical/evidence contract in the
    envelope before it can be qualified.
    """

    results: tuple[Any, ...]
    execution_strategy: str
    numerical_regime: str
    evidence_use: str
    qualified: bool
    execution_identity: Mapping[str, Any] | None
    metadata: Mapping[str, Any]

    @property
    def trusted(self) -> bool:
        return bool(self.qualified and self.evidence_use == "trusted")

    def provenance(self) -> dict[str, Any]:
        return {
            "execution_strategy": self.execution_strategy,
            "numerical_regime": self.numerical_regime,
            "evidence_use": self.evidence_use,
            "qualified": self.qualified,
            "execution_identity": deepcopy(dict(self.execution_identity or {})),
            "execution_metadata": deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class ScoreTokensScreeningRow:
    """Immutable, deliberately small input to a refinement selector."""

    arm_id: str
    arm_index: int
    token_logprobs: tuple[float, ...]
    total_logprob: float


@dataclass(frozen=True)
class ScoreTokensRefinementPolicy:
    """Explicit policy for promoting selected screening arms to scalar evidence.

    Proof-critical arm IDs are always remeasured.  The optional selector is
    invoked exactly once with screening summaries only and may add borderline
    arm IDs.  It never receives semantic requests or scalar results.
    """

    proof_critical_arm_ids: tuple[str, ...] = ()
    borderline_selector: Callable[
        [tuple[ScoreTokensScreeningRow, ...]], Iterable[str]
    ] | None = None

    def __post_init__(self) -> None:
        ids = tuple(self.proof_critical_arm_ids)
        if any(not isinstance(arm_id, str) or not arm_id for arm_id in ids):
            raise ValueError("proof-critical arm IDs must be non-empty strings")
        if len(ids) != len(set(ids)):
            raise ValueError("proof-critical arm IDs must not contain duplicates")
        if self.borderline_selector is not None and not callable(self.borderline_selector):
            raise TypeError("borderline_selector must be callable")
        object.__setattr__(self, "proof_critical_arm_ids", ids)


@dataclass(frozen=True)
class ScoreTokensRefinedRow:
    """One scalar-reference row, kept separate from its native screen."""

    arm_id: str
    arm_index: int
    result: Any


@dataclass(frozen=True)
class ScoreTokensScreenThenRefineResult:
    """Two-phase result whose numerical regimes are never merged."""

    screening: ScoreTokensManyResult
    scalar_refinement: ScoreTokensManyResult | None
    refined_rows: tuple[ScoreTokensRefinedRow, ...]
    proof_critical_arm_ids: tuple[str, ...]
    borderline_arm_ids: tuple[str, ...]
    selected_arm_ids: tuple[str, ...]

    @property
    def refined_by_arm_id(self) -> dict[str, Any]:
        return {row.arm_id: deepcopy(row.result) for row in self.refined_rows}

    def provenance(self) -> dict[str, Any]:
        return {
            "execution_strategy": "native_screen_then_scalar_refine",
            "qualified": False,
            "screening_execution": self.screening.provenance(),
            "scalar_refinement_execution": (
                self.scalar_refinement.provenance()
                if self.scalar_refinement is not None else None
            ),
            "proof_critical_arm_ids": list(self.proof_critical_arm_ids),
            "borderline_arm_ids": list(self.borderline_arm_ids),
            "selected_arm_ids": list(self.selected_arm_ids),
        }


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    method = getattr(cancel, "is_set", None)
    if callable(method):
        return bool(method())
    return bool(cancel)


def _freeze(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


_PER_ARM_FIELDS = frozenset({
    "messages", "prompt", "removed_source_ids", "retained_source_ids",
    "context_digest", "intervened_context_digest",
})


def _validate(kind: str, index: int, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MultiArmError(f"{kind} arm {index} must be a mapping", arm_index=index)
    arm = deepcopy(dict(raw))
    if not isinstance(arm.get("messages"), (list, tuple)):
        raise MultiArmError(f"{kind} arm {index} must carry a messages list", arm_index=index)
    if kind == "probe_reference_match":
        ids = arm.get("reference_token_ids")
        if not isinstance(ids, (list, tuple)) or not ids or any(
                isinstance(item, bool) or not isinstance(item, int) for item in ids):
            raise MultiArmError(f"{kind} arm {index} has invalid reference_token_ids", arm_index=index)
        if not isinstance(arm.get("generation_contract"), Mapping):
            raise MultiArmError(f"{kind} arm {index} has invalid generation_contract", arm_index=index)
    else:
        ids = arm.get("continuation_ids")
        if ids is not None and (not isinstance(ids, (list, tuple)) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in ids)):
            raise MultiArmError(f"{kind} arm {index} has invalid continuation_ids", arm_index=index)
    return arm


def _groups(kind: str, arms: list[dict[str, Any]]) -> list[list[tuple[int, dict[str, Any]]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    order: list[str] = []
    for index, arm in enumerate(arms):
        immutable = {key: value for key, value in arm.items() if key not in _PER_ARM_FIELDS}
        key = _freeze(immutable)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((index, arm))
    return [groups[key] for key in order]


def _invoke_native_detailed(
    method: Callable[..., Any], arms: list[dict[str, Any]], *, cancel: Any,
    require_native_screening: bool = False,
) -> tuple[list[Any], dict[str, Any]]:
    try:
        signature = inspect.signature(method)
        accepts_cancel = "cancel" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values())
        accepts_requirement = (
            "require_native_screening" in signature.parameters
            or any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                   for parameter in signature.parameters.values())
        )
    except (TypeError, ValueError):
        accepts_cancel = True
        accepts_requirement = False
    kwargs = {"cancel": cancel} if accepts_cancel else {}
    if require_native_screening and accepts_requirement:
        kwargs["require_native_screening"] = True
    raw = method(arms, **kwargs)
    metadata: dict[str, Any] = {}
    if isinstance(raw, Mapping) and isinstance(raw.get("results"), list):
        metadata = deepcopy({key: value for key, value in raw.items() if key != "results"})
        raw = raw["results"]
    if not isinstance(raw, (list, tuple)) or len(raw) != len(arms):
        raise MultiArmError("a multi-arm method must return one result per input arm")
    if raw and all(isinstance(row, Mapping) and isinstance(row.get("arm_index"), int)
                   and "result" in row for row in raw):
        restored = [None] * len(raw)
        seen: set[int] = set()
        for row in raw:
            index = row["arm_index"]
            if index < 0 or index >= len(raw) or index in seen:
                raise MultiArmError("native multi-arm results contain invalid arm indexes")
            seen.add(index)
            restored[index] = row["result"]
        raw = restored
    return [deepcopy(row) for row in raw], metadata


def _scalar(method: Callable[..., Any], arms: list[dict[str, Any]], *, cancel: Any,
            base_index: int = 0) -> list[Any]:
    result: list[Any] = []
    completed_indices: list[int] = []
    for offset, arm in enumerate(arms):
        if _cancelled(cancel):
            raise BatchCancelled(
                completed=result, next_index=base_index + offset,
                completed_indices=completed_indices,
                dispatched_indices=completed_indices,
            )
        try:
            result.append(deepcopy(method(**arm)))
            completed_indices.append(base_index + offset)
        except Exception as exc:
            raise MultiArmError(
                str(exc), arm_index=base_index + offset, completed=result,
                completed_indices=completed_indices,
                dispatched_indices=[*completed_indices, base_index + offset],
            ) from exc
    return result


def _native_many(
    kind: str, native: Callable[..., Any], normalised: list[dict[str, Any]], *, cancel: Any,
    require_native_screening: bool = False,
) -> tuple[list[Any], list[dict[str, Any]]]:
    ordered: list[Any] = [None] * len(normalised)
    envelopes: list[dict[str, Any]] = []
    completed: list[Any] = []
    completed_indices: list[int] = []
    for group in _groups(kind, normalised):
        indexes = [index for index, _arm in group]
        group_arms = [arm for _index, arm in group]
        if _cancelled(cancel):
            raise BatchCancelled(
                completed=completed, next_index=indexes[0],
                completed_indices=completed_indices,
                dispatched_indices=completed_indices,
            )
        try:
            rows, metadata = _invoke_native_detailed(
                native, group_arms, cancel=cancel,
                require_native_screening=require_native_screening,
            )
        except BatchCancelled as exc:
            local_completed = tuple(exc.completed_indices)
            if not local_completed:
                local_completed = tuple(range(min(len(exc.completed), len(indexes))))
            local_dispatched = tuple(exc.dispatched_indices)
            if not local_dispatched:
                # The native call was accepted.  Unless its protocol names
                # the unstarted children, conservatively count the whole
                # submitted group as dispatched.
                local_dispatched = tuple(range(len(indexes)))
            group_completed_indices = [indexes[i] for i in local_completed if 0 <= i < len(indexes)]
            group_dispatched_indices = [indexes[i] for i in local_dispatched if 0 <= i < len(indexes)]
            group_completed_rows = list(exc.completed)
            raise BatchCancelled(
                completed=completed + group_completed_rows,
                next_index=exc.next_index if exc.next_index is not None else indexes[0],
                completed_indices=[*completed_indices, *group_completed_indices],
                dispatched_indices=[*completed_indices, *group_dispatched_indices],
            ) from exc
        except MultiArmError as exc:
            local_completed = tuple(exc.completed_indices)
            if not local_completed:
                local_completed = tuple(range(min(len(exc.completed), len(indexes))))
            local_dispatched = tuple(exc.dispatched_indices)
            if not local_dispatched:
                local_dispatched = tuple(range(len(indexes)))
            group_completed_indices = [indexes[i] for i in local_completed if 0 <= i < len(indexes)]
            group_dispatched_indices = [indexes[i] for i in local_dispatched if 0 <= i < len(indexes)]
            raise MultiArmError(
                str(exc),
                arm_index=(indexes[exc.arm_index] if isinstance(exc.arm_index, int)
                           and 0 <= exc.arm_index < len(indexes) else indexes[0]),
                completed=completed + list(exc.completed),
                completed_indices=[*completed_indices, *group_completed_indices],
                dispatched_indices=[*completed_indices, *group_dispatched_indices],
            ) from exc
        except Exception as exc:
            raise MultiArmError(
                str(exc), arm_index=indexes[0], completed=completed,
                completed_indices=completed_indices,
                dispatched_indices=[*completed_indices, *indexes],
            ) from exc
        for index, row in zip(indexes, rows):
            ordered[index] = row
        envelopes.append(metadata)
        completed.extend(rows)
        completed_indices.extend(indexes)
    return ordered, envelopes


def _many(kind: str, substrate: Any, arms: Iterable[Mapping[str, Any]], *, cancel: Any = None,
          proof_grade: bool = True) -> list[Any]:
    normalised = [_validate(kind, index, raw) for index, raw in enumerate(arms)]
    if _cancelled(cancel):
        raise BatchCancelled(completed=[], next_index=0)
    scalar = getattr(substrate, kind, None)
    if not callable(scalar):
        raise MultiArmError(f"substrate has no scalar {kind} method")
    native = getattr(substrate, f"{kind}_many", None)
    native_ok = bool(proof_grade and getattr(substrate, f"{kind}_many_proof_grade", False))
    if not native_ok:
        return _scalar(scalar, normalised, cancel=cancel)
    rows, _envelopes = _native_many(kind, native, normalised, cancel=cancel)
    return rows


_SCALAR_SCORE_EXECUTION_IDENTITY = {
    "kind": "clozn.score-tokens-scalar-reference.v1",
}


def _native_score_strategy(value: Any) -> bool:
    return isinstance(value, str) and (value == "native" or value.startswith("native_"))


def _scalar_score_result(rows: list[Any], *, strategy: str,
                         metadata: Mapping[str, Any] | None = None) -> ScoreTokensManyResult:
    return ScoreTokensManyResult(
        results=tuple(deepcopy(rows)),
        execution_strategy=strategy,
        numerical_regime="scalar_reference",
        evidence_use="trusted",
        qualified=True,
        execution_identity=deepcopy(_SCALAR_SCORE_EXECUTION_IDENTITY),
        metadata=deepcopy(dict(metadata or {})),
    )


def _native_score_result(
    rows: list[Any], envelopes: list[dict[str, Any]], *, declared_strategy: str,
) -> ScoreTokensManyResult:
    contracts: list[Mapping[str, Any]] = []
    for envelope in envelopes:
        nested = envelope.get("execution_metadata")
        contracts.append(nested if isinstance(nested, Mapping) else envelope)

    metadata: dict[str, Any]
    if len(envelopes) == 1:
        metadata = deepcopy(envelopes[0])
    else:
        metadata = {"groups": deepcopy(envelopes)}

    # An adaptive native-capable substrate may honestly decline a workload and
    # execute its scalar reference path instead.  Preserve that actual regime
    # rather than relabeling it as native merely because dispatch entered the
    # substrate's many-arm seam.
    scalar_strategies = {"scalar_serial", "scalar_http_concurrent"}
    if contracts and all(
        item.get("execution_strategy") in scalar_strategies
        and item.get("numerical_regime") == "scalar_reference"
        and item.get("evidence_use") == "trusted"
        and item.get("qualified") is True
        and isinstance(item.get("execution_identity"), Mapping)
        and dict(item.get("execution_identity")) == _SCALAR_SCORE_EXECUTION_IDENTITY
        for item in contracts
    ):
        strategies = [str(item["execution_strategy"]) for item in contracts]
        identities = [deepcopy(dict(item["execution_identity"])) for item in contracts]
        if all(strategy == strategies[0] for strategy in strategies):
            identity: Mapping[str, Any] = (
                identities[0] if all(item == identities[0] for item in identities)
                else {"kind": "clozn.score-tokens-scalar-reference-groups.v1",
                      "groups": identities}
            )
            return ScoreTokensManyResult(
                results=tuple(deepcopy(rows)),
                execution_strategy=strategies[0],
                numerical_regime="scalar_reference",
                evidence_use="trusted",
                qualified=True,
                execution_identity=identity,
                metadata=metadata,
            )

    strategies = [item.get("execution_strategy") for item in contracts]
    regimes = [item.get("numerical_regime") for item in contracts]
    evidence_uses = [item.get("evidence_use") for item in contracts]
    identities = [item.get("execution_identity") for item in contracts]
    qualified = bool(contracts) and all(
        item.get("qualified") is True
        and item.get("execution_strategy") == declared_strategy
        and isinstance(item.get("numerical_regime"), str) and item.get("numerical_regime")
        and item.get("evidence_use") in {"screening", "trusted"}
        and isinstance(item.get("execution_identity"), Mapping) and item.get("execution_identity")
        for item in contracts
    )
    strategy = (
        strategies[0]
        if strategies and isinstance(strategies[0], str) and strategies[0]
        and all(item == strategies[0] for item in strategies)
        else declared_strategy
    )
    regime = (
        regimes[0]
        if regimes and isinstance(regimes[0], str) and regimes[0]
        and all(item == regimes[0] for item in regimes)
        else "unqualified_native"
    )
    evidence_use = (
        evidence_uses[0]
        if evidence_uses and all(item == evidence_uses[0] for item in evidence_uses)
        and evidence_uses[0] in {"screening", "trusted"}
        else "unqualified"
    )
    identity: Mapping[str, Any] | None = None
    if identities and all(isinstance(item, Mapping) and item for item in identities):
        identity = (
            deepcopy(dict(identities[0])) if len(identities) == 1
            else {"kind": "clozn.score-tokens-native-groups.v1",
                  "groups": [deepcopy(dict(item)) for item in identities]}
        )
    return ScoreTokensManyResult(
        results=tuple(deepcopy(rows)),
        execution_strategy=str(strategy),
        numerical_regime=str(regime),
        evidence_use=str(evidence_use),
        qualified=qualified,
        execution_identity=identity,
        metadata=metadata,
    )


def score_tokens_many(
    substrate: Any, arms: Iterable[Mapping[str, Any]], *, cancel: Any = None,
    proof_grade: bool = True, require_native_screening: bool = False,
) -> ScoreTokensManyResult:
    """Score arms and retain execution/qualification metadata.

    This path never infers native execution from the
    ``score_tokens_many_proof_grade`` flag alone. A substrate
    must declare ``score_tokens_many_execution_strategy`` as either the known
    scalar HTTP wrapper or an explicit ``native*`` strategy.  Undeclared
    implementations take the scalar reference path.
    """

    if not isinstance(require_native_screening, bool):
        raise TypeError("require_native_screening must be a boolean")

    normalised = [_validate("score_tokens", index, raw) for index, raw in enumerate(arms)]
    if _cancelled(cancel):
        raise BatchCancelled(completed=[], next_index=0)
    scalar = getattr(substrate, "score_tokens", None)
    if not callable(scalar):
        raise MultiArmError("substrate has no scalar score_tokens method")
    declared = getattr(substrate, "score_tokens_many_execution_strategy", None)
    many = getattr(substrate, "score_tokens_many", None)
    many_enabled = bool(
        proof_grade and getattr(substrate, "score_tokens_many_proof_grade", False)
        and callable(many)
    )
    explicit_many = bool(require_native_screening and many_enabled)
    if (
        declared not in {"scalar_http_concurrent"}
        and not _native_score_strategy(declared)
        and not explicit_many
    ):
        return _scalar_score_result(
            _scalar(scalar, normalised, cancel=cancel), strategy="scalar_serial",
        )
    if not many_enabled or (declared == "scalar_serial" and not explicit_many):
        return _scalar_score_result(
            _scalar(scalar, normalised, cancel=cancel), strategy="scalar_serial",
        )
    rows, envelopes = _native_many(
        "score_tokens", many, normalised, cancel=cancel,
        require_native_screening=require_native_screening,
    )
    if declared == "scalar_http_concurrent" and not require_native_screening:
        metadata = envelopes[0] if len(envelopes) == 1 else {"groups": envelopes}
        return _scalar_score_result(rows, strategy=declared, metadata=metadata)
    effective_strategy = declared
    if require_native_screening and not _native_score_strategy(effective_strategy):
        contracts = []
        for envelope in envelopes:
            nested = envelope.get("execution_metadata")
            contracts.append(nested if isinstance(nested, Mapping) else envelope)
        strategies = [item.get("execution_strategy") for item in contracts]
        if (
            strategies
            and isinstance(strategies[0], str)
            and _native_score_strategy(strategies[0])
            and all(item == strategies[0] for item in strategies)
        ):
            effective_strategy = strategies[0]
        else:
            # Preserve an unqualified result rather than blessing a substrate
            # that accepted the explicit requirement without declaring one
            # coherent native execution strategy.
            effective_strategy = "native_required_screening"
    return _native_score_result(
        rows, envelopes, declared_strategy=str(effective_strategy),
    )


def _screening_rows(
    rows: tuple[Any, ...], arm_ids: tuple[str, ...],
) -> tuple[ScoreTokensScreeningRow, ...]:
    summaries: list[ScoreTokensScreeningRow] = []
    for arm_index, (arm_id, row) in enumerate(zip(arm_ids, rows)):
        if not isinstance(row, (list, tuple)) or not row:
            raise MultiArmError(
                "native screening returned an empty or malformed token row",
                arm_index=arm_index,
            )
        logprobs: list[float] = []
        for token in row:
            value = token.get("logprob") if isinstance(token, Mapping) else None
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                raise MultiArmError(
                    "native screening returned a malformed token logprob",
                    arm_index=arm_index,
                )
            logprobs.append(float(value))
        summaries.append(ScoreTokensScreeningRow(
            arm_id=arm_id,
            arm_index=arm_index,
            token_logprobs=tuple(logprobs),
            total_logprob=sum(logprobs),
        ))
    return tuple(summaries)


def _selected_arm_ids(
    policy: ScoreTokensRefinementPolicy,
    summaries: tuple[ScoreTokensScreeningRow, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    known = {row.arm_id for row in summaries}
    missing = [arm_id for arm_id in policy.proof_critical_arm_ids if arm_id not in known]
    if missing:
        raise MultiArmError(
            "proof-critical refinement IDs are not present in the score batch: "
            + ", ".join(missing)
        )
    borderline: tuple[str, ...] = ()
    if policy.borderline_selector is not None:
        try:
            selected = policy.borderline_selector(summaries)
            if isinstance(selected, (str, bytes)):
                raise TypeError("the selector must return an iterable of arm IDs")
            borderline = tuple(selected)
        except Exception as exc:
            raise MultiArmError(f"borderline refinement selection failed: {exc}") from exc
        if any(not isinstance(arm_id, str) or not arm_id for arm_id in borderline):
            raise MultiArmError("borderline refinement IDs must be non-empty strings")
        if len(borderline) != len(set(borderline)):
            raise MultiArmError("borderline refinement IDs must not contain duplicates")
        unknown = [arm_id for arm_id in borderline if arm_id not in known]
        if unknown:
            raise MultiArmError(
                "borderline refinement IDs are not present in the score batch: "
                + ", ".join(unknown)
            )
    selected_set = set(policy.proof_critical_arm_ids) | set(borderline)
    ordered = tuple(row.arm_id for row in summaries if row.arm_id in selected_set)
    ordered_borderline = tuple(row.arm_id for row in summaries if row.arm_id in set(borderline))
    return ordered, ordered_borderline


def score_tokens_many_screen_then_refine(
    substrate: Any,
    arms: Iterable[Mapping[str, Any]],
    *,
    arm_ids: Iterable[str],
    policy: ScoreTokensRefinementPolicy,
    cancel: Any = None,
) -> ScoreTokensScreenThenRefineResult:
    """Run one native screen, then scalar-remeasure only explicitly selected arms.

    This operation is intentionally separate from :func:`score_tokens_many`;
    merely enabling the native endpoint never changes the default scalar path.
    Screening rows remain in ``screening`` and scalar rows remain in
    ``scalar_refinement``/``refined_rows``.  There is no combined row vector
    that could accidentally be treated as one numerical regime.
    """

    if not isinstance(policy, ScoreTokensRefinementPolicy):
        raise TypeError("policy must be a ScoreTokensRefinementPolicy")
    normalised = [_validate("score_tokens", index, raw) for index, raw in enumerate(arms)]
    ids = tuple(arm_ids)
    if len(ids) != len(normalised):
        raise MultiArmError("arm_ids must contain one ID per score arm")
    if any(not isinstance(arm_id, str) or not arm_id for arm_id in ids):
        raise MultiArmError("score refinement arm IDs must be non-empty strings")
    if len(ids) != len(set(ids)):
        raise MultiArmError("score refinement arm IDs must be unique")
    missing = [arm_id for arm_id in policy.proof_critical_arm_ids if arm_id not in set(ids)]
    if missing:
        raise MultiArmError(
            "proof-critical refinement IDs are not present in the score batch: "
            + ", ".join(missing)
        )

    screening = score_tokens_many(
        substrate, normalised, cancel=cancel, proof_grade=True,
        require_native_screening=True,
    )
    if screening.evidence_use != "screening" or screening.trusted:
        raise MultiArmError(
            "screen-then-refine requires an explicit screening-only native execution"
        )
    try:
        summaries = _screening_rows(screening.results, ids)
        selected_ids, borderline_ids = _selected_arm_ids(policy, summaries)
    except MultiArmError as exc:
        raise MultiArmError(
            str(exc), arm_index=exc.arm_index,
            completed=(), completed_indices=(),
            dispatched_indices=tuple(range(len(normalised))),
        ) from exc
    selected = set(selected_ids)
    if not selected_ids:
        return ScoreTokensScreenThenRefineResult(
            screening=screening,
            scalar_refinement=None,
            refined_rows=(),
            proof_critical_arm_ids=tuple(policy.proof_critical_arm_ids),
            borderline_arm_ids=borderline_ids,
            selected_arm_ids=(),
        )
    if _cancelled(cancel):
        raise BatchCancelled(
            "score refinement cancelled after native screening",
            completed=(),
            next_index=min(index for index, arm_id in enumerate(ids) if arm_id in selected),
            completed_indices=(),
            dispatched_indices=tuple(range(len(ids))),
        )
    scalar = getattr(substrate, "score_tokens", None)
    if not callable(scalar):
        raise MultiArmError("substrate has no scalar score_tokens method")
    selected_indexes = [index for index, arm_id in enumerate(ids) if arm_id in selected]
    selected_arms = [normalised[index] for index in selected_indexes]
    try:
        exact_rows = _scalar(scalar, selected_arms, cancel=cancel)
    except (BatchCancelled, MultiArmError) as exc:
        # Screening has already dispatched every arm.  Do not expose partial
        # scalar rows without a phase-aware caller; fail the orchestration
        # closed instead of allowing a native row to stand in for evidence.
        error_type = BatchCancelled if isinstance(exc, BatchCancelled) else MultiArmError
        kwargs = {
            "completed": (),
            "completed_indices": (),
            "dispatched_indices": tuple(range(len(normalised))),
        }
        if error_type is BatchCancelled:
            kwargs["next_index"] = (
                selected_indexes[exc.next_index]
                if isinstance(exc.next_index, int) and 0 <= exc.next_index < len(selected_indexes)
                else selected_indexes[0]
            )
        else:
            kwargs["arm_index"] = (
                selected_indexes[exc.arm_index]
                if isinstance(exc.arm_index, int) and 0 <= exc.arm_index < len(selected_indexes)
                else selected_indexes[0]
            )
        raise error_type(str(exc), **kwargs) from exc
    scalar_refinement = _scalar_score_result(
        exact_rows,
        strategy="scalar_serial",
        metadata={
            "kind": "clozn.score-tokens-scalar-refinement.v1",
            "selected_arm_ids": list(selected_ids),
        },
    )
    refined_rows = tuple(
        ScoreTokensRefinedRow(
            arm_id=ids[index], arm_index=index, result=deepcopy(result),
        )
        for index, result in zip(selected_indexes, exact_rows)
    )
    return ScoreTokensScreenThenRefineResult(
        screening=screening,
        scalar_refinement=scalar_refinement,
        refined_rows=refined_rows,
        proof_critical_arm_ids=tuple(policy.proof_critical_arm_ids),
        borderline_arm_ids=borderline_ids,
        selected_arm_ids=selected_ids,
    )


def probe_reference_match_many(substrate: Any, arms: Iterable[Mapping[str, Any]], *, cancel: Any = None,
                               proof_grade: bool = True) -> list[Any]:
    return _many("probe_reference_match", substrate, arms, cancel=cancel, proof_grade=proof_grade)


def serial_many(method: Callable[..., Any], arms: Iterable[Mapping[str, Any]], *, cancel: Any = None) -> list[Any]:
    return _scalar(method, [deepcopy(dict(arm)) for arm in arms], cancel=cancel)


def concurrent_many(method: Callable[..., Any], arms: Iterable[Mapping[str, Any]], *, cancel: Any = None,
                    max_workers: int = 2) -> list[Any]:
    normalised = [deepcopy(dict(arm)) for arm in arms]
    if max_workers <= 1:
        return _scalar(method, normalised, cancel=cancel)
    if _cancelled(cancel):
        raise BatchCancelled(completed=[], next_index=0)
    executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="clozn-batch")
    futures = {executor.submit(method, **arm): index for index, arm in enumerate(normalised)}
    results: list[Any] = [None] * len(normalised)
    try:
        while futures:
            done, _pending = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                index = futures.pop(future)
                try:
                    results[index] = deepcopy(future.result())
                except Exception as exc:
                    completed = [row for row in results if row is not None]
                    completed_indices = [item for item, row in enumerate(results) if row is not None]
                    dispatched_indices = list(futures.values()) + [index]
                    raise MultiArmError(
                        str(exc), arm_index=index, completed=completed,
                        completed_indices=completed_indices,
                        dispatched_indices=dispatched_indices,
                    ) from exc
            if _cancelled(cancel):
                for future in futures:
                    future.cancel()
                raise BatchCancelled(
                    completed=[row for row in results if row is not None],
                    next_index=min(futures.values(), default=None),
                    completed_indices=[item for item, row in enumerate(results) if row is not None],
                    dispatched_indices=list(futures.values()),
                )
        return results
    finally:
        executor.shutdown(wait=not _cancelled(cancel), cancel_futures=True)


__all__ = [
    "BatchCancelled", "MultiArmError", "ScoreTokensManyResult", "concurrent_many",
    "probe_reference_match_many", "score_tokens_many", "serial_many",
]
