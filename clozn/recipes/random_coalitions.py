"""Opt-in budgeted coalition experiment; not a production Studio contract.

Uses the same recorded-answer scorer, deletion semantics, observation identities,
and Lean Context tolerances as the shipped recipes. No embeddings, text matching,
surrogate model, regenerated answers, or additional native runtime are involved.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
import statistics
import time
from typing import Any

from clozn.experiments.coalitions import shapley_pairs
from clozn.experiments.effective_prompt import resolve_effective_prompt
from clozn.experiments.evaluators import ScoreRecordedContinuation
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.observations import TokenScoreObservation, execution_observation_identity
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import run_experiment
from clozn.experiments.score_context_search import continuation_support_loss
from clozn.experiments.scoring import DeleteSourceRecordedContinuationScoreAdapter
from clozn.experiments.selections import ContextSelection
from clozn.experiments.state import ExecutionState
from clozn.recipes.lean_context import (
    DEFAULT_MEAN_LOSS_TOLERANCE_NATS, DEFAULT_MAX_TOKEN_LOSS_TOLERANCE_NATS,
)
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.runs.context_units import max_context_units


def _trusted(observation: Any) -> bool:
    return (
        isinstance(observation, TokenScoreObservation)
        and observation.completed and observation.trusted
        and bool(observation.recorded_token_ids)
    )


class _Stopped(Exception):
    pass


class _Scorer:
    """Budget unique score attempts, including the control and failed attempts.

    Each arm runs without an implicit control; we score that exactly once above
    the kernel. This also guarantees in-memory reuse when no durable store exists.
    """

    def __init__(self, run, ids, *, substrate, engine, store, adapter, counter, budget, cancel, progress):
        self.run, self.ids = run, ids
        self.state = ExecutionState.from_run(run)
        self.evaluator = ScoreRecordedContinuation()
        self.adapter = adapter or DeleteSourceRecordedContinuationScoreAdapter(substrate, run=run)
        self.engine, self.store, self.counter = engine, store, counter
        self.budget, self.used, self.reused = budget, 0, 0
        self.cancel, self.progress = cancel, progress
        self.cache: dict[tuple[str, ...], TokenScoreObservation] = {}
        self.rows: dict[tuple[str, ...], dict[str, Any]] = {}

    def intervention(self, retained):
        kept = set(retained)
        removed = tuple(sid for sid in self.ids if sid not in kept)
        return DeleteSource(ContextSelection(removed)) if removed else None

    def lookup(self, retained):
        if retained in self.cache:
            return self.cache[retained]
        if self.store is not None:
            identity = execution_observation_identity(self.state, self.evaluator, self.intervention(retained))
            observation = self.store.find_observation(identity["observation_key_sha256"])
            if _trusted(observation):
                self.cache[retained] = observation
                return observation
        return None

    def check(self, candidates):
        if self.cancel is not None and self.cancel():
            raise _Stopped("cancelled")
        needed = sum(self.lookup(ids) is None for ids in set(candidates))
        if self.used + needed > self.budget:
            raise _Stopped("execution_budget")

    def score(self, retained):
        self.check([retained])
        if retained in self.rows:
            return self.cache[retained]
        started = time.perf_counter()
        intervention = self.intervention(retained)
        prompt = resolve_effective_prompt(self.run, intervention)
        messages = prompt.rendered_messages()
        cost = (self.counter(messages) if self.counter is not None else
                self.engine.apply_template_info(messages).get("prompt_tokens"))
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
            raise ValueError("exact rendered prompt token count unavailable")
        prepared_seconds = time.perf_counter() - started
        observation = self.lookup(retained)
        disposition = "reused"
        diagnostics = {}
        if observation is None:
            self.used += 1  # reserve before execution; failures must not be free
            experiment = Experiment(
                base=self.state, evaluator=self.evaluator,
                arms=[intervention] if intervention is not None else [],
            )
            result = run_experiment(
                experiment, self.adapter, include_control=intervention is None,
                observation_store=self.store, cancel=self.cancel,
                diagnostics={"random_coalitions": True},
            )
            if intervention is None:
                observation, disposition = result.control, result.control_disposition
                diagnostics = dict(result.diagnostics)
            else:
                arm = result.arm_for(experiment.arms[0].arm_id)
                observation = arm.observation
                disposition = str(arm.diagnostics.get("execution_disposition", "not_executed"))
                diagnostics = {"error": dict(arm.error), **dict(arm.diagnostics)}
            if disposition != "executed":
                self.used -= 1
        if disposition == "reused":
            self.reused += 1
        row = {
            "retained_source_ids": list(retained),
            "rendered_prompt_tokens": cost,
            "disposition": disposition,
            "prepare_seconds": prepared_seconds,
            "elapsed_seconds": time.perf_counter() - started,
            "observation_id": observation.observation_id if observation else None,
            "status": observation.status if observation else "unavailable",
            "diagnostics": diagnostics,
        }
        self.rows[retained] = row
        if self.progress is not None:
            self.progress(dict(row, used_new_executions=self.used))
        if not _trusted(observation):
            raise _Stopped("cancelled" if self.cancel and self.cancel() else "score_unavailable")
        control = self.cache.get(self.ids)
        if control is not None and (
            control.recorded_token_ids != observation.recorded_token_ids
            or control.token_pieces != observation.token_pieces
        ):
            self.cache.pop(retained, None)
            row["status"] = "token_alignment_mismatch"
            raise _Stopped("token_alignment_mismatch")
        self.cache[retained] = observation
        row["total_logprob_nats"] = sum(observation.token_logprobs)
        return observation


def run_random_coalitions(
    run: Mapping[str, Any], *, max_new_executions: int = 16, max_pairs: int = 32,
    seed: int = 0, max_units: int | None = None,
    mean_loss_tolerance_nats: float = DEFAULT_MEAN_LOSS_TOLERANCE_NATS,
    max_token_loss_tolerance_nats: float = DEFAULT_MAX_TOKEN_LOSS_TOLERANCE_NATS,
    substrate: Any = None, engine: Any = None,
    observation_store: ObservationStore | None = None, execution_adapter: Any = None,
    prompt_token_counter: Callable[[Sequence[Mapping[str, Any]]], int] | None = None,
    cancel: Callable[[], bool] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
    sample_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Sample paired marginals, and retain the cheapest tested tolerant context.

    Score(S) is the sum of teacher-forced log probabilities for the SAME answer.
    Signed marginals retain both support and suppression. Lean Context uses its
    separate positive-only loss test: improvements cannot offset local damage.
    A small sampling budget is neither an optimality nor a redundancy proof.
    """
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise ValueError("run_random_coalitions requires a recorded run")
    for name, value in (("max_new_executions", max_new_executions), ("max_pairs", max_pairs)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    for name, value in (("mean_loss_tolerance_nats", mean_loss_tolerance_nats),
                        ("max_token_loss_tolerance_nats", max_token_loss_tolerance_nats)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if observation_store is not None and not isinstance(observation_store, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")
    if prompt_token_counter is None and not callable(getattr(engine, "apply_template_info", None)):
        raise ValueError("an exact prompt token counter or engine is required")
    universe = plan_context_search_universe(
        run, run.get("context_units"), max_units=max_context_units() if max_units is None else max_units,
    )
    if universe["status"] != "planned":
        raise ValueError(universe["condition"]["message"])
    ids = tuple(universe["source_ids"])
    scorer = _Scorer(
        run, ids, substrate=substrate, engine=engine, store=observation_store,
        adapter=execution_adapter, counter=prompt_token_counter, budget=max_new_executions,
        cancel=cancel, progress=progress,
    )
    started = time.perf_counter()
    samples = []
    control = None
    stop_reason = "sample_limit"
    try:
        control = scorer.score(ids)
        # Existing leave-one-out evidence is useful for comparison AND the best
        # tested context. Never spend extra executions on this comparison sweep.
        for sid in ids:
            retained = tuple(other for other in ids if other != sid)
            if scorer.lookup(retained) is not None:
                scorer.score(retained)
        pairs = shapley_pairs(ids, seed=seed)
        for _ in range(max_pairs):
            sid, without, with_source = next(pairs)
            # Stop before spending on an unfinishable pair. Do not cherry-pick a
            # cheaper draw, which would distort the intended sampling law.
            scorer.check([without, with_source])
            a, b = scorer.score(without), scorer.score(with_source)
            deltas = [y - x for x, y in zip(a.token_logprobs, b.token_logprobs)]
            sample = {
                "source_id": sid, "without_source_ids": list(without),
                "with_source_ids": list(with_source), "coalition_size": len(without),
                "without_observation_id": a.observation_id, "with_observation_id": b.observation_id,
                "delta_nats": sum(deltas), "token_deltas_nats": deltas,
            }
            samples.append(sample)
            if sample_progress is not None:
                sample_progress({
                    "sample": dict(sample),
                    "completed_pairs": len(samples),
                    "used_new_executions": scorer.used,
                    "reused_observation_count": scorer.reused,
                })
    except _Stopped as exc:
        stop_reason = str(exc)
    finally:
        close = getattr(scorer.adapter, "close", None)
        if callable(close):
            close()

    estimates = []
    for sid in ids:
        draws = [sample for sample in samples if sample["source_id"] == sid]
        values = [sample["delta_nats"] for sample in draws]
        unique = {tuple(sample["without_source_ids"]) for sample in draws}
        marginal = statistics.fmean(values) if values else None
        without = tuple(other for other in ids if other != sid)
        single = scorer.cache.get(without) if without in scorer.rows else None
        loo = (sum(x - y for x, y in zip(control.token_logprobs, single.token_logprobs))
               if control is not None and single is not None else None)
        estimates.append({
            "source_id": sid, "sample_count": len(draws), "unique_pair_count": len(unique),
            "coverage": "unsampled" if not draws else "one_coalition" if len(unique) == 1 else "multiple_coalitions",
            "mean_marginal_nats": marginal,
            "sample_stddev_nats": statistics.stdev(values) if len(values) > 1 else None,
            "mean_token_marginals_nats": [
                statistics.fmean(sample["token_deltas_nats"][i] for sample in draws)
                for i in range(len(draws[0]["token_deltas_nats"]))
            ] if draws else None,
            "leave_one_out_nats": loo,
            "mean_minus_leave_one_out_nats": marginal - loo if marginal is not None and loo is not None else None,
        })
    best = None
    for retained, row in scorer.rows.items():
        observation = scorer.cache.get(retained)
        loss = continuation_support_loss(control, observation) if control is not None and observation else None
        row["loss"] = loss
        row["within_tolerance"] = bool(
            loss is not None
            and loss["mean_positive_support_loss_nats_per_token"] <= mean_loss_tolerance_nats
            and loss["max_token_support_loss_nats"] <= max_token_loss_tolerance_nats
        )
        if row["within_tolerance"] and (best is None or row["rendered_prompt_tokens"] < best["rendered_prompt_tokens"]):
            best = row
    return {
        "format": "clozn.random-coalitions.experimental.v0", "run_id": run["id"],
        "status": "unavailable" if control is None else (
            "completed" if stop_reason in {"sample_limit", "execution_budget"} else "partial"
        ),
        "stop_reason": stop_reason, "seed": seed,
        "base_execution_fingerprint": scorer.state.execution_fingerprint,
        "universe": universe,
        "sampling": {"kind": "shapley_size_uniform_paired_marginals", "max_pairs": max_pairs,
                     "source_schedule": "shuffled_rounds", "text_matching": False},
        "claim": {"kind": "exploratory_sampled_marginals", "minimum_proven": False,
                  "redundancy_proven": False, "confidence_intervals": None,
                  "best_context": "cheapest_directly_tested_within_tolerance"},
        "budget": {"max_new_executions": max_new_executions, "used_new_executions": scorer.used,
                   "reused_observation_count": scorer.reused, "control_included": True,
                   "completed_pairs": len(samples)},
        "tolerance": {"mean_positive_support_loss_nats_per_token": mean_loss_tolerance_nats,
                      "max_token_support_loss_nats": max_token_loss_tolerance_nats},
        "elapsed_seconds": time.perf_counter() - started,
        "answer_tokens": [{"id": token_id, "piece": piece, "span": list(span)}
                          for token_id, piece, span in zip(control.recorded_token_ids, control.token_pieces, control.token_spans)]
                         if control is not None else [],
        "source_estimates": estimates, "samples": samples, "candidates": list(scorer.rows.values()),
        "best": best,
    }
