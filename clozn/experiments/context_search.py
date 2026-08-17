"""Generic DeleteSource + ExactReferenceMatch search dispatch."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from .effective_prompt import EffectivePrompt, resolve_effective_prompt
from .execution import DeleteSourceExactReferenceAdapter
from .evaluators import ExactReferenceMatch
from .interventions import DeleteSource
from .kernel import Experiment
from .persistence import ObservationStore
from .runner import ExperimentResult, run_experiment
from .search import PreparedCandidate
from .selections import ContextSelection
from .state import ExecutionState


class ContextSearchUnavailable(ValueError):
    """The context search cannot obtain faithful objective or execution evidence."""

    def __init__(self, message: str, *, reason: str = "context_search_unavailable"):
        super().__init__(message)
        self.reason = reason


def _as_messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContextSearchUnavailable("context resolver returned no message list", reason="messages_unavailable")
    messages = [dict(item) for item in value if isinstance(item, Mapping)]
    if len(messages) != len(value):
        raise ContextSearchUnavailable("context resolver returned malformed messages", reason="messages_malformed")
    return messages


class ContextSearchDispatcher:
    """Translate retained-source candidates into ordinary kernel experiments."""

    def __init__(self, run: Mapping[str, Any], universe: Mapping[str, Any], *, substrate: Any = None,
                 engine: Any = None, observation_store: ObservationStore | None = None,
                 execution_adapter: Any = None, evaluator: ExactReferenceMatch | None = None,
                 execution_strategy: str = "auto",
                 prompt_token_counter: Callable[[Sequence[Mapping[str, Any]]], int] | None = None,
                 render_messages: Callable[[tuple[str, ...]], Sequence[Mapping[str, Any]]] | None = None,
                 cancel: Callable[[], bool] | None = None):
        if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
            raise ValueError("ContextSearchDispatcher requires a recorded run")
        if not isinstance(universe, Mapping):
            raise ValueError("ContextSearchDispatcher requires a planned universe")
        source_ids = universe.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids or any(not isinstance(item, str) for item in source_ids):
            raise ContextSearchUnavailable("planned universe has no canonical source IDs", reason="universe_unavailable")
        state = ExecutionState.from_run(run)
        self.run = deepcopy(dict(run))
        self.universe = dict(universe)
        self.source_ids = tuple(source_ids)
        self.state = state
        self.evaluator = evaluator or ExactReferenceMatch()
        if not isinstance(self.evaluator, ExactReferenceMatch):
            raise TypeError("context search supports ExactReferenceMatch only")
        self.observation_store = observation_store
        self.substrate = substrate
        self.engine = engine
        self.prompt_token_counter = prompt_token_counter
        self.cancel = cancel
        self._uses_canonical_prompt = render_messages is None
        self._local_observations: dict[str, Any] = {}
        self._last_probe_context: dict[str, Any] = {}
        self.execution_adapter = execution_adapter or DeleteSourceExactReferenceAdapter(
            substrate, run=self.run, engine=engine, execution_strategy=execution_strategy,
        )
        if render_messages is None:
            self._render_messages = self._default_render_messages
        else:
            self._render_messages = render_messages

    def _default_render_messages(self, retained_ids: tuple[str, ...]) -> Sequence[Mapping[str, Any]]:
        return resolve_effective_prompt(self.run, self._intervention(tuple(retained_ids))).rendered_messages()

    def _effective_prompt(self, retained_ids: tuple[str, ...]) -> EffectivePrompt:
        if self._uses_canonical_prompt:
            return resolve_effective_prompt(self.run, self._intervention(retained_ids))
        return EffectivePrompt(
            messages=tuple(_as_messages(self._render_messages(retained_ids))),
            block=None,
            basis="custom_renderer",
        )

    def _prompt_cost(self, prompt: EffectivePrompt) -> int:
        messages = prompt.rendered_messages()
        try:
            if self.prompt_token_counter is not None:
                value = self.prompt_token_counter(messages)
            elif self.engine is not None and callable(getattr(self.engine, "apply_template_info", None)):
                rendered = self.engine.apply_template_info([dict(item) for item in messages])
                value = rendered.get("prompt_tokens") if isinstance(rendered, Mapping) else None
            else:
                value = None
        except Exception as exc:
            raise ContextSearchUnavailable(
                f"faithful rendered prompt token counting failed: {exc}",
                reason="rendered_prompt_token_count_unavailable",
            ) from exc
        if value is None:
            raise ContextSearchUnavailable(
                "faithful rendered prompt token counting is unavailable",
                reason="rendered_prompt_token_count_unavailable",
            )
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContextSearchUnavailable(
                "prompt token seam did not return an exact non-negative count",
                reason="rendered_prompt_token_count_malformed",
            )
        return value

    def _intervention(self, retained_ids: tuple[str, ...]) -> DeleteSource | None:
        removed = tuple(source_id for source_id in self.source_ids if source_id not in set(retained_ids))
        if not removed:
            return None
        return DeleteSource(ContextSelection(removed))

    def prepare_candidate(self, retained_ids: tuple[str, ...]) -> PreparedCandidate:
        retained = tuple(retained_ids)
        prompt = self._effective_prompt(retained)
        messages = prompt.rendered_messages()
        intervention = self._intervention(retained)
        payload = {
            "retained_ids": retained,
            "messages": messages,
            "intervention": intervention,
        }
        return PreparedCandidate(retained, self._prompt_cost(prompt), payload)

    def _known_observation(self, intervention: DeleteSource | None):
        from .observations import execution_observation_identity
        identity = execution_observation_identity(self.state, self.evaluator, intervention)
        key = identity["observation_key_sha256"]
        if self.observation_store is not None:
            found = self.observation_store.find_observation(key)
            if found is not None:
                return found
        return self._local_observations.get(key)

    def candidate_is_reusable(self, retained_ids: tuple[str, ...]) -> bool:
        return self._known_observation(self._intervention(tuple(retained_ids))) is not None

    def set_probe_context(self, *, stage: str, parent_retained_ids: tuple[str, ...]) -> None:
        self._last_probe_context = {
            "stage": stage,
            "parent_retained_ids": tuple(parent_retained_ids),
        }

    def on_control_accepted(self, candidate: Any, prepared: Any, evidence: Any) -> None:
        hook = getattr(self.execution_adapter, "on_control_accepted", None)
        if callable(hook):
            hook(candidate=candidate, prepared=prepared, evidence=evidence, dispatcher=self)

    def on_candidate_accepted(self, candidate: Any, prepared: Any, evidence: Any) -> None:
        hook = getattr(self.execution_adapter, "on_candidate_accepted", None)
        if callable(hook):
            hook(candidate=candidate, prepared=prepared, evidence=evidence, dispatcher=self)

    def close(self) -> None:
        hook = getattr(self.execution_adapter, "close", None)
        if callable(hook):
            hook()

    def _one_ref(self, result: ExperimentResult, arm_id: str, intervention: DeleteSource | None,
                 *, reused: bool = False) -> dict[str, Any]:
        if arm_id == "control":
            observation = result.control
            status = observation.status if observation is not None else result.diagnostics.get("control_error", "unavailable")
            disposition = result.control_disposition
        else:
            row = result.arm_for(arm_id)
            observation = row.observation
            status = observation.status if observation is not None else row.status
            disposition = str(row.diagnostics.get("execution_disposition", "not_executed"))
        if reused:
            disposition = "reused"
        if observation is not None:
            key = observation.observation_id if observation.completed else None
            if observation.completed:
                self._local_observations[observation.observation_key_sha256] = observation
        else:
            key = None
        classification = "preserves" if status in {"exact_preserved", "matched"} else (
            "diverged" if status == "diverged" else "unknown"
        )
        return {
            "experiment_id": result.experiment_id,
            "arm_id": arm_id,
            "observation_id": key,
            "observation_status": status,
            "classification": classification,
            "disposition": disposition,
            "intervention": intervention.to_dict() if intervention is not None else None,
        }

    def probe_many(self, prepared_candidates: Sequence[PreparedCandidate]) -> list[dict[str, Any]]:
        candidates = list(prepared_candidates)
        if not candidates:
            return []
        if len(candidates) == 1 and candidates[0].retained_ids == self.source_ids:
            experiment = Experiment(base=self.state, evaluator=self.evaluator, arms=[])
            result = run_experiment(
                experiment, self.execution_adapter,
                observation_store=self.observation_store,
                cancel=self.cancel,
                diagnostics={"context_search": True, "control": True},
            )
            reused = bool(result.diagnostics.get("control_reused"))
            return [self._one_ref(result, "control", None, reused=reused)]

        interventions: list[DeleteSource] = []
        for candidate in candidates:
            intervention = candidate.probe_payload.get("intervention")
            if not isinstance(intervention, DeleteSource):
                raise ContextSearchUnavailable(
                    "a counterfactual candidate did not resolve to DeleteSource",
                    reason="intervention_unavailable",
                )
            interventions.append(intervention)
        experiment = Experiment(base=self.state, evaluator=self.evaluator, arms=interventions)
        known = [self._known_observation(item) is not None for item in interventions]
        result = run_experiment(
            experiment, self.execution_adapter,
            observation_store=self.observation_store,
            cancel=self.cancel,
            diagnostics={"context_search": True, "candidate_count": len(interventions),
                         "probe_context": deepcopy(self._last_probe_context)},
        )
        return [
            self._one_ref(result, arm.arm_id, intervention, reused=known[index])
            for index, (arm, intervention) in enumerate(zip(experiment.arms, interventions))
        ]


__all__ = ["ContextSearchDispatcher", "ContextSearchUnavailable"]
