"""Generate execution for resolved answer-token states.

This adapter intentionally stops at standalone evidence.  It uses the existing
exact planner/control proof and the existing reconstructed text seams, but it
never calls the legacy executor that materializes a child Run.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from clozn.replay.execution_fork import (
    parent_runtime_projection, plan_execution_fork,
)
from clozn.replay.execution_fork_execute import (
    ExecutionForkExecutionError, _worker_generation_steps, _worker_receipt,
    prove_unchanged_control,
)
from clozn.replay import fork as reconstructed_fork

from .evaluators import Generate
from .interventions import ForceToken
from .observations import GeneratedObservation, execution_observation_identity
from .state import ExecutionState, digest
from .state_ref import ResolvedState, StateRefError


class GenerateExecutionError(ValueError):
    """A Generate arm cannot produce honest standalone evidence."""


def _runtime_from_substrate(run: Mapping[str, Any], substrate: Any) -> Mapping[str, Any] | None:
    value = getattr(substrate, "runtime_identity", None)
    if isinstance(value, Mapping):
        return dict(value)
    identity_fn = getattr(substrate, "identity_meta", None)
    meta_fn = getattr(substrate, "run_meta", None)
    if callable(identity_fn) and callable(meta_fn):
        try:
            raw = dict(identity_fn() or {})
            meta = dict(meta_fn() or {})
            # The planner's runtime projection accepts this same compact shape.
            raw.setdefault("model_sha256", raw.get("model_sha256") or run.get("identity", {}).get("model_sha256"))
            raw.setdefault("template_fingerprint", raw.get("template_fingerprint") or run.get("identity", {}).get("template_fingerprint"))
            raw.setdefault("engine_build", raw.get("engine_build") or run.get("identity", {}).get("engine_build"))
            raw.setdefault("context_size", raw.get("context_size", meta.get("n_ctx")))
            raw.setdefault("backend", raw.get("backend", meta.get("device")))
            raw.setdefault("adapter", raw.get("adapter", {"present": False, "identity_sha256": None,
                                                            "artifact_sha256": None, "scale": None}))
            raw.setdefault("white_box_flags", raw.get("white_box_flags", meta.get("white_box_flags", {})))
            return raw
        except Exception:
            return None
    try:
        return parent_runtime_projection(run)
    except Exception:
        return None


def _worker_from_substrate(substrate: Any) -> Mapping[str, Any] | None:
    value = getattr(substrate, "worker_identity", None)
    if isinstance(value, Mapping):
        return dict(value)
    health = getattr(getattr(substrate, "engine", substrate), "health", None)
    if callable(health):
        try:
            raw = dict(health() or {})
            generation = raw.get("worker_generation_id")
            protocol = raw.get("protocol_version")
            worker_id = raw.get("worker_id", raw.get("id"))
            if all(isinstance(item, str) and item for item in (worker_id, generation, protocol)):
                return {"worker_id": worker_id, "worker_generation_id": generation,
                        "protocol_version": protocol}
        except Exception:
            pass
    return None


def _identity_kwargs(base: ResolvedState, evaluator: Generate, intervention: ForceToken | None) -> dict[str, Any]:
    identity = execution_observation_identity(base, evaluator, intervention)
    key = identity["observation_key"]
    return {
        "observation_id": identity["observation_id"],
        "observation_key_sha256": identity["observation_key_sha256"],
        "observation_key": key,
        "run_id": base.run_id,
        "base_execution_fingerprint": base.execution_fingerprint,
        "evaluator": key["evaluator"], "condition": key["condition"], "contract": key["contract"],
    }


def _unavailable(base: ResolvedState, evaluator: Generate, intervention: ForceToken | None,
                 code: str, message: str, *, diagnostics: Mapping[str, Any] | None = None) -> GeneratedObservation:
    value = {"reason_code": code, "message": message}
    value.update(dict(diagnostics or {}))
    return GeneratedObservation(
        **_identity_kwargs(base, evaluator, intervention), status="unavailable",
        state_ref=base.state_ref, realization=base.realization,
        fidelity={"classification": base.classification, "proof_status": "not_confirmed"},
        intervention=intervention, generated_suffix_text="", generated_token_ids=(),
        execution_provenance={"adapter": "generate", "state_resolution": base.to_dict()},
        runtime_provenance={}, exact_control_proof={}, generation_contract=evaluator.to_dict(),
        proof_grade="unavailable", trusted=False, diagnostics=value,
    )


def _failed(base: ResolvedState, evaluator: Generate, intervention: ForceToken | None,
            code: str, message: str, *, diagnostics: Mapping[str, Any] | None = None) -> GeneratedObservation:
    observation = _unavailable(base, evaluator, intervention, code, message, diagnostics=diagnostics)
    # Reconstruct with the explicit failure state; failed evidence is equally non-reusable.
    return GeneratedObservation(
        **{name: getattr(observation, name) for name in (
            "observation_id", "observation_key_sha256", "observation_key", "run_id",
            "base_execution_fingerprint", "evaluator", "condition", "contract")},
        status="failed", state_ref=observation.state_ref, realization=observation.realization,
        fidelity=observation.fidelity, intervention=intervention, generated_suffix_text="",
        generated_token_ids=(), execution_provenance=observation.execution_provenance,
        runtime_provenance={}, exact_control_proof={}, generation_contract=evaluator.to_dict(),
        proof_grade="unavailable", trusted=False,
        diagnostics={"reason_code": code, "message": message, **dict(diagnostics or {})},
    )


def _validate_force_against_recorded(run: Mapping[str, Any], position: int, force: ForceToken) -> str | None:
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    pieces = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else []
    if position < 0 or position >= len(pieces):
        return "force token position is outside recorded token history"
    if force.token_id is not None and position < len(ids) and ids[position] == force.token_id:
        if force.token_piece is not None and pieces[position] != force.token_piece:
            return "token_id and token_piece disagree with the recorded token boundary"
    alternatives = trace.get("alternatives")
    values = alternatives[position] if isinstance(alternatives, list) and position < len(alternatives) else []
    if isinstance(values, list):
        for item in values:
            if not isinstance(item, Mapping):
                continue
            item_id = item.get("token_id", item.get("id"))
            item_piece = item.get("piece", item.get("text"))
            if force.token_id is not None and item_id == force.token_id and force.token_piece is not None \
                    and item_piece != force.token_piece:
                return "token_id and token_piece disagree in recorded alternatives"
            if force.token_piece is not None and item_piece == force.token_piece and force.token_id is not None \
                    and item_id is not None and item_id != force.token_id:
                return "token_id and token_piece identify different recorded alternatives"
    return None


def _worker_evidence(reply: Mapping[str, Any]) -> tuple[str, tuple[int, ...], list[dict[str, Any]] | None, str | None]:
    if not isinstance(reply, Mapping) or not isinstance(reply.get("text"), str):
        return "", (), None, "worker returned malformed generated text"
    text = reply["text"]
    ids = reply.get("tokens")
    if not isinstance(ids, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids):
        return "", (), None, "worker returned malformed generated token IDs"
    pieces = reply.get("token_pieces")
    if pieces is not None:
        if not isinstance(pieces, list) or len(pieces) != len(ids) or any(not isinstance(item, str) for item in pieces):
            return "", (), None, "worker returned malformed generated token pieces"
        if "".join(pieces) != text:
            return "", (), None, "worker token pieces do not decode to worker text"
    steps = _worker_generation_steps(reply)
    if steps is not None:
        if any(not isinstance(step, Mapping) or not isinstance(step.get("piece"), str)
               for step in steps):
            return "", (), None, "worker returned malformed generation steps"
    return text, tuple(ids), steps, None


def _sampler_preserved(reply: Mapping[str, Any]) -> bool:
    if reply.get("sampler_state_preserved") is True:
        return True
    sampler = reply.get("sampler")
    return isinstance(sampler, Mapping) and (
        sampler.get("sampler_state_preserved") is True or sampler.get("state_preserved") is True
    )


class GenerateExecutionAdapter:
    """Execute Generate arms into GeneratedObservation objects only."""

    def __init__(self, substrate: Any, *, run: Mapping[str, Any] | None = None,
                 run_loader: Callable[[str], Mapping[str, Any] | None] | None = None,
                 runtime_identity: Mapping[str, Any] | None = None,
                 worker_identity: Mapping[str, Any] | None = None):
        if substrate is None:
            raise ValueError("GenerateExecutionAdapter requires a substrate")
        self.substrate = substrate
        self.engine = getattr(substrate, "engine", substrate)
        self._run = deepcopy(dict(run)) if isinstance(run, Mapping) else None
        self._run_loader = run_loader
        self.runtime_identity = dict(runtime_identity) if isinstance(runtime_identity, Mapping) else None
        self.worker_identity = dict(worker_identity) if isinstance(worker_identity, Mapping) else None
        self._control_proofs: dict[str, Mapping[str, Any]] = {}

    def load_run(self, state: ExecutionState | ResolvedState) -> Mapping[str, Any] | None:
        run_id = state.run_id
        if self._run_loader is not None:
            return self._run_loader(run_id)
        if self._run is not None and self._run.get("id") == run_id:
            return deepcopy(self._run)
        return None

    def _validated_run(self, resolved: ResolvedState) -> dict[str, Any]:
        run = self.load_run(resolved)
        if not isinstance(run, Mapping):
            raise GenerateExecutionError("the base run could not be loaded")
        try:
            current = resolved.state_ref.assert_current(run)
        except StateRefError as exc:
            raise GenerateExecutionError(str(exc)) from exc
        if current.execution_fingerprint != resolved.execution_fingerprint:
            raise GenerateExecutionError("the base execution fingerprint changed")
        return dict(run)

    def _identities(self, run: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
        runtime = self.runtime_identity or _runtime_from_substrate(run, self.substrate)
        worker = self.worker_identity or _worker_from_substrate(self.substrate)
        return runtime, worker

    def _exact_plan(self, resolved: ResolvedState, run: Mapping[str, Any], change: Mapping[str, Any]) -> dict[str, Any]:
        checkpoint = resolved.realization.get("checkpoint_reference")
        runtime, worker = self._identities(run)
        return plan_execution_fork(
            run, {"position": resolved.position.index, "change": dict(change)},
            checkpoint=checkpoint, runtime_identity=runtime, worker_identity=worker,
        )

    def _control_exact(self, resolved: ResolvedState, evaluator: Generate,
                       run: Mapping[str, Any]) -> GeneratedObservation:
        try:
            plan = resolved.plan or self._exact_plan(resolved, run, {"type": "none"})
            proof = prove_unchanged_control(run, plan, self.engine)
        except Exception as exc:
            return _unavailable(resolved, evaluator, None, "exact_control_unavailable", str(exc))
        self._control_proofs[resolved.state_fingerprint] = proof
        proof_receipt = proof.get("worker_receipt") if isinstance(proof, Mapping) else {}
        if proof.get("status") != "matched":
            return _unavailable(
                resolved, evaluator, None, "exact_control_mismatch",
                "the unchanged exact control diverged; exact intervention was not run",
                diagnostics={"control_proof": deepcopy(proof)},
            )
        if proof_receipt.get("sampler_state_preserved") is not True:
            return _unavailable(
                resolved, evaluator, None, "sampler_state_unavailable",
                "exact Generate requires a captured and verified sampler state",
                diagnostics={"control_proof": deepcopy(proof)},
            )
        trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
        pieces = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
        ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else []
        position = resolved.position.index
        steps = trace.get("steps") if isinstance(trace.get("steps"), list) else None
        return GeneratedObservation(
            **_identity_kwargs(resolved, evaluator, None), status="completed", state_ref=resolved.state_ref,
            realization=resolved.realization,
            fidelity={"classification": "exact_execution_fork", "proof_status": "confirmed",
                      "exact_match": True, "unchanged_control": "matched"},
            intervention=None, generated_suffix_text="".join(pieces[position:]),
            generated_token_ids=ids[position:], generated_steps=deepcopy(steps[position:]) if steps else None,
            finish_reason=run.get("finish_reason"), generation_contract=evaluator.to_dict(),
            runtime_provenance={"worker_receipt": deepcopy(proof_receipt)},
            exact_control_proof=deepcopy(proof),
            execution_provenance={"adapter": "generate", "execution": "exact_control_proof"},
            proof_grade="trusted", trusted=True,
            diagnostics={"reason_code": "exact_control_confirmed"},
        )

    def _reconstructed_completion(self, run: Mapping[str, Any], prompt: str, budget: int):
        if budget <= 0:
            return "", "branch_horizon_exhausted", None
        extra = reconstructed_fork._steer_kwargs(self.substrate, dict(run))
        traced = reconstructed_fork._complete_traced(self.engine, prompt, budget, extra)
        if traced is not None:
            return traced
        text, finish = reconstructed_fork._complete_greedy(self.engine, prompt, budget, extra)
        return text, finish, None

    def _reconstructed(self, resolved: ResolvedState, evaluator: Generate,
                       intervention: ForceToken | None, run: Mapping[str, Any]) -> GeneratedObservation:
        if evaluator.decode_mode != "greedy":
            return _unavailable(resolved, evaluator, intervention, "stochastic_execution_unbound",
                                "reconstructed Generate is reusable only for a greedy continuation")
        if intervention is not None and intervention.token_piece is None:
            return _unavailable(resolved, evaluator, intervention, "reconstruction_token_piece_unavailable",
                                "reconstructed ForceToken requires token_piece")
        trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
        pieces = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
        position = resolved.position.index
        prompt, prompt_source = reconstructed_fork._prompt_base(dict(run), self.substrate)
        if not isinstance(prompt, str) or not prompt:
            return _unavailable(resolved, evaluator, intervention, "reconstruction_prompt_unavailable",
                                "the parent exact rendered prompt is unavailable")
        prefix = "".join(str(piece) for piece in pieces[:position])
        forced = intervention.token_piece if intervention is not None else ""
        budget = evaluator.max_new if intervention is None else max(0, evaluator.max_new - 1)
        try:
            continuation, finish, steps = self._reconstructed_completion(run, prompt + prefix + forced, budget)
        except Exception as exc:
            return _failed(resolved, evaluator, intervention, "generation_failed", str(exc))
        if not isinstance(continuation, str):
            return _failed(resolved, evaluator, intervention, "generation_malformed", "generator returned no text")
        suffix = forced + continuation
        generated_ids: tuple[int, ...] = ()
        if steps is not None:
            candidate_ids = [step.get("token_id") for step in steps]
            if all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in candidate_ids):
                generated_ids = tuple(([intervention.token_id] if intervention and intervention.token_id is not None else []) + candidate_ids)
        elif intervention is not None and intervention.token_id is not None:
            generated_ids = (intervention.token_id,)
        retokenized = reconstructed_fork._detect_retokenization(
            self.substrate, dict(run), [*pieces[:position], forced] if intervention else list(pieces[:position]),
        )
        realization = deepcopy(resolved.realization)
        fidelity = {
            "classification": "reconstructed_replay", "proof_status": "not_applicable",
            "unavoidable_differences": list(realization.get("unavoidable_differences") or []),
            "retokenized_prefix": True if retokenized is None else bool(retokenized),
        }
        return GeneratedObservation(
            **_identity_kwargs(resolved, evaluator, intervention), status="completed",
            state_ref=resolved.state_ref, realization=realization, fidelity=fidelity,
            intervention=intervention, generated_suffix_text=suffix,
            generated_token_ids=generated_ids, generated_steps=steps,
            finish_reason=finish, generation_contract=evaluator.to_dict(),
            runtime_provenance={"runtime_identity": deepcopy(self._identities(run)[0]),
                                "prompt_source": prompt_source},
            exact_control_proof={}, execution_provenance={"adapter": "generate", "execution": "reconstructed"},
            proof_grade="reconstructed", trusted=True,
            diagnostics={"reason_code": "reconstructed_replay"},
        )

    def execute_control(self, state: ResolvedState, *, evaluator: Generate | None = None) -> GeneratedObservation:
        evaluator = evaluator or Generate(max_new=1)
        if not isinstance(state, ResolvedState) or not isinstance(evaluator, Generate):
            raise TypeError("Generate control requires ResolvedState and Generate")
        try:
            run = self._validated_run(state)
        except Exception as exc:
            return _unavailable(state, evaluator, None, "base_run_unavailable", str(exc))
        if state.classification == "unavailable":
            return _unavailable(state, evaluator, None, "state_unavailable", "the resolved state is unavailable")
        if state.classification == "exact_execution_fork":
            return self._control_exact(state, evaluator, run)
        return self._reconstructed(state, evaluator, None, run)

    def execute(self, state: ResolvedState, intervention: ForceToken | None = None, *,
                evaluator: Generate | None = None, arm_id: str | None = None) -> GeneratedObservation:
        evaluator = evaluator or Generate(max_new=1)
        if not isinstance(state, ResolvedState) or not isinstance(evaluator, Generate):
            raise TypeError("Generate execution requires ResolvedState and Generate")
        if not isinstance(intervention, ForceToken):
            raise TypeError("Generate arms require ForceToken")
        try:
            run = self._validated_run(state)
        except Exception as exc:
            return _unavailable(state, evaluator, intervention, "base_run_unavailable", str(exc))
        disagreement = _validate_force_against_recorded(run, state.position.index, intervention)
        if disagreement:
            return _failed(state, evaluator, intervention, "force_token_mismatch", disagreement)
        if state.classification == "unavailable":
            return _unavailable(state, evaluator, intervention, "state_unavailable", "the resolved state is unavailable")
        if state.classification == "reconstructed_replay":
            return self._reconstructed(state, evaluator, intervention, run)
        if evaluator.decode_mode != "greedy":
            return _unavailable(state, evaluator, intervention, "stochastic_execution_unbound",
                                "exact Generate requires a verified sampler/RNG state")
        proof = self._control_proofs.get(state.state_fingerprint)
        if not isinstance(proof, Mapping) or proof.get("status") != "matched":
            control = self._control_exact(state, evaluator, run)
            if not control.completed:
                return _unavailable(state, evaluator, intervention, "exact_control_mismatch",
                                    "the unchanged exact control did not establish fidelity")
            proof = self._control_proofs.get(state.state_fingerprint)
        change = {"type": "force_token", "token_id": intervention.token_id}
        if intervention.token_piece is not None:
            change["token_piece"] = intervention.token_piece
        plan = self._exact_plan(state, run, change)
        if plan.get("classification") != "exact_execution_fork":
            reason = (plan.get("reasons") or [{}])[0]
            return _unavailable(state, evaluator, intervention, str(reason.get("code") or "exact_state_unavailable"),
                                str(reason.get("message") or "exact execution is unavailable"))
        try:
            reply = self.engine.execution_fork(
                checkpoint_id=plan["checkpoint_reference"]["checkpoint_id"],
                worker_generation_id=plan["checkpoint_reference"]["worker_generation_id"],
                truncate_to=plan["exactness"]["truncate_to"], max_tokens=evaluator.max_new,
                intervention={"type": "force_token", "token_id": intervention.token_id},
            )
            receipt = _worker_receipt(reply, plan, {"type": "force_token", "token_id": intervention.token_id})
            if not _sampler_preserved(reply):
                return _unavailable(state, evaluator, intervention, "sampler_state_unavailable",
                                    "worker did not verify sampler/RNG preservation")
            text, ids, steps, evidence_error = _worker_evidence(reply)
            if evidence_error:
                return _failed(state, evaluator, intervention, "malformed_worker_token_evidence", evidence_error)
            if steps is not None and "".join(step.get("piece", "") for step in steps) != text:
                return _failed(state, evaluator, intervention, "malformed_worker_token_evidence",
                               "worker generation steps do not decode to worker text")
            forced_piece = intervention.token_piece
            if forced_piece is None and steps and steps[0].get("token_id") == intervention.token_id:
                forced_piece = steps[0].get("piece")
            if forced_piece is None:
                alternatives = (run.get("trace") or {}).get("alternatives") if isinstance(run.get("trace"), Mapping) else None
                values = alternatives[state.position.index] if isinstance(alternatives, list) and state.position.index < len(alternatives) else []
                if isinstance(values, list):
                    for item in values:
                        if isinstance(item, Mapping) and item.get("token_id", item.get("id")) == intervention.token_id:
                            candidate = item.get("piece", item.get("text"))
                            if isinstance(candidate, str):
                                forced_piece = candidate
                            break
            if forced_piece is not None:
                if text.startswith(forced_piece):
                    if not ids or ids[0] != intervention.token_id:
                        return _failed(state, evaluator, intervention, "malformed_worker_token_evidence",
                                       "worker output did not report the forced token as its first token")
                else:
                    # The trusted worker protocol may return only the post-force
                    # continuation.  The forced token itself is known from the
                    # typed intervention, so prepend that evidence explicitly;
                    # never tokenize the returned text to recover it.
                    text = forced_piece + text
                    ids = (intervention.token_id, *ids)
                    if steps is not None:
                        steps = [{"token_id": intervention.token_id, "piece": forced_piece}, *steps]
        except Exception as exc:
            return _unavailable(state, evaluator, intervention, "exact_generation_unavailable", str(exc))
        return GeneratedObservation(
            **_identity_kwargs(state, evaluator, intervention), status="completed", state_ref=state.state_ref,
            realization=state.realization,
            fidelity={"classification": "exact_execution_fork", "proof_status": "confirmed",
                      "exact_match": True, "unchanged_control": "matched"},
            intervention=intervention, generated_suffix_text=text, generated_token_ids=ids,
            generated_steps=steps, finish_reason=reply.get("finish_reason"),
            generation_contract=evaluator.to_dict(), runtime_provenance={"worker_receipt": receipt},
            exact_control_proof=deepcopy(proof),
            execution_provenance={"adapter": "generate", "execution": "exact_intervention"},
            proof_grade="trusted", trusted=True, diagnostics={"reason_code": "exact_confirmed"},
        )


__all__ = ["GenerateExecutionAdapter", "GenerateExecutionError"]
