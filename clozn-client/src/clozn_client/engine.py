"""Explicit client for native/private engine research operations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._transport import CloznProtocolError, JsonTransport
from .manifests import (
    InterventionArm,
    InterventionArmResult,
    InterventionManifest,
    InterventionRunResult,
    ScoreRequest,
)
from .patch_manifests import PatchSweepArtifact, PatchSweepManifest
from .models import (
    AttentionKnockout,
    HarvestResult,
    Observation,
    PatchArm,
    PatchArmResult,
    PatchSweepResult,
    ScoreResult,
    ScoreVariantsResult,
    require_object,
)

_USER_AGENT = "clozn-client-engine/0.11.0"
_SCORE_VARIANTS_MAX = 64
_SCORE_VARIANT_ID_MAX = 128


class EngineClient:
    """Direct native engine client, separate from the public Clozn gateway.

    Pass the native worker URL intentionally. This class never discovers or guesses a supervised
    worker port from a gateway URL.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8091", *, timeout: float = 900.0):
        self._transport = JsonTransport(base_url, timeout=timeout, headers={"User-Agent": _USER_AGENT})

    @property
    def base_url(self) -> str:
        return self._transport.base_url

    def health(self) -> dict[str, Any]:
        return require_object(self._transport.request_json("GET", "/health"), "engine health")

    def contract_report(self, *operations: object):
        """Check this engine against stable receipt-backed intervention operations.

        Pass operation identifiers explicitly, or pass one InterventionManifest/PatchSweepManifest
        and the client will derive its stable operation requirements.
        """
        from .contracts import check_contract, required_operations

        if len(operations) == 1:
            try:
                selected = required_operations(operations[0])
            except ValueError:
                selected = operations
        else:
            selected = operations
        return check_contract(self.health(), selected)

    def supports_attention_knockout(self) -> bool:
        health = self.health()
        capabilities = health.get("capabilities")
        return bool(isinstance(capabilities, Mapping) and capabilities.get("attn_knockout"))

    def harvest(self, text: str, layer: int | None = None, *, budget: object | None = None) -> HarvestResult:
        """Read every token residual at one native engine tap layer."""
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        body: dict[str, Any] = {"text": text}
        if layer is not None:
            if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
                raise ValueError("layer must be a non-negative integer")
            body["layer"] = layer
        result = HarvestResult.from_json(
            self._transport.request_json("POST", "/harvest", body=body)
        )
        from .contracts import DEFAULT_CAPTURE_BUDGET, CaptureBudget
        selected_budget = DEFAULT_CAPTURE_BUDGET if budget is None else budget
        if not isinstance(selected_budget, CaptureBudget):
            raise ValueError("budget must be a CaptureBudget or None")
        selected_budget.check(n_tokens=result.n_tokens, n_embd=result.n_embd)
        return result

    def write_state(self, text: str, layer: int, positions: Sequence[int], values: Any) -> Observation:
        """Write residual rows and observe next-token movement.

        Values must be shaped ``[len(positions), n_embd]`` (or a vector for one position).
        The client validates row count locally and transmits position-major float32 values.
        """
        from .tensors import flatten_float32

        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise ValueError("layer must be a non-negative integer")
        normalized_positions = self._ids(positions, "positions")
        flat, shape = flatten_float32(values)
        rows = 1 if len(shape) == 1 else shape[0]
        if rows != len(normalized_positions):
            raise ValueError(
                f"values row count {rows} does not match {len(normalized_positions)} positions"
            )
        response = self._transport.request_json("POST", "/state", body={
            "text": text,
            "layer": layer,
            "positions": normalized_positions,
            "values": flat,
        })
        return Observation.from_json(response)

    def patch_sweep(self, text: str, arms: Sequence[PatchArm], *, layer: int | None = None, budget: object | None = None) -> PatchSweepResult:
        """Harvest once, then replay named residual patches at that exact layer."""
        arm_list = tuple(arms)
        if not arm_list:
            raise ValueError("patch sweep must contain at least one arm")
        if any(not isinstance(arm, PatchArm) for arm in arm_list):
            raise ValueError("patch sweep arms must be PatchArm objects")
        names = [arm.name for arm in arm_list]
        if len(set(names)) != len(names):
            raise ValueError("patch sweep arm names must be unique")
        harvest = self.harvest(text, layer, budget=budget)
        results: list[PatchArmResult] = []
        for arm in arm_list:
            if any(pos >= harvest.n_tokens for pos in arm.positions):
                raise ValueError(
                    f"patch arm {arm.name!r} contains a position outside 0..{harvest.n_tokens - 1}"
                )
            observation = self.write_state(text, harvest.layer, arm.positions, arm.values)
            results.append(PatchArmResult(
                name=arm.name, positions=arm.positions, observation=observation, metadata=arm.metadata
            ))
        return PatchSweepResult(harvest=harvest, arms=tuple(results))

    def run_patch_manifest(self, manifest: PatchSweepManifest) -> PatchSweepArtifact:
        """Replay a portable patch-sweep manifest and return a JSON-ready artifact."""
        if not isinstance(manifest, PatchSweepManifest):
            raise ValueError("manifest must be a PatchSweepManifest")
        health: dict[str, Any] = {}
        if manifest.expected_health:
            health = self.health()
            self._require_expected(health, manifest.expected_health)
        result = self.patch_sweep(
            manifest.text,
            tuple(arm.to_patch_arm() for arm in manifest.arms),
            layer=manifest.layer,
        )
        return PatchSweepArtifact.from_result(manifest, result, engine_health=health)

    def replay_patch_manifest(self, manifest: PatchSweepManifest) -> PatchSweepArtifact:
        """Alias emphasizing deterministic patch-manifest replay."""
        return self.run_patch_manifest(manifest)

    def complete(self, prompt: str, *, max_tokens: int = 32, temperature: float = 0.0,
                 **options: Any) -> dict[str, Any]:
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if isinstance(max_tokens, bool) or int(max_tokens) < 1:
            raise ValueError("max_tokens must be a positive integer")
        reserved = {"prompt", "max_tokens", "temperature", "attn_knockout"}.intersection(options)
        if reserved:
            raise ValueError(f"reserved completion options: {', '.join(sorted(reserved))}")
        body = {"prompt": prompt, "max_tokens": int(max_tokens),
                "temperature": float(temperature), **options}
        return require_object(self._transport.request_json("POST", "/v1/completions", body=body),
                              "engine completion")

    def score(self, *, prompt: str | None = None, prompt_ids: Sequence[int] | None = None,
              continuation: str | None = None, continuation_ids: Sequence[int] | None = None,
              topk: int = 0, steer: Mapping[str, Any] | None = None,
              steer_vec: Sequence[float] | None = None,
              attn_knockout: Sequence[AttentionKnockout] | None = None) -> ScoreResult:
        self._exclusive(prompt, prompt_ids, "prompt", "prompt_ids")
        self._exclusive(continuation, continuation_ids, "continuation", "continuation_ids")
        if isinstance(topk, bool) or int(topk) < 0:
            raise ValueError("topk must be a non-negative integer")
        body: dict[str, Any] = {"topk": int(topk)}
        if prompt_ids is not None:
            body["prompt_ids"] = self._ids(prompt_ids, "prompt_ids")
        else:
            body["prompt"] = prompt
        if continuation_ids is not None:
            body["continuation_ids"] = self._ids(continuation_ids, "continuation_ids")
        else:
            body["continuation"] = continuation
        if steer is not None:
            if not isinstance(steer, Mapping):
                raise ValueError("steer must be an object")
            body["steer"] = dict(steer)
        if steer_vec is not None:
            body["steer_vec"] = [float(value) for value in steer_vec]
        if attn_knockout is not None:
            specs = list(attn_knockout)
            if not specs:
                raise ValueError("attn_knockout must not be empty")
            if any(not isinstance(spec, AttentionKnockout) for spec in specs):
                raise ValueError("attn_knockout entries must be AttentionKnockout objects")
            body["attn_knockout"] = [spec.to_wire() for spec in specs]
        response = self._transport.request_json("POST", "/score", body=body)
        return ScoreResult.from_json(response)

    def score_variants(
        self,
        variants: Sequence[Mapping[str, Any]],
        continuation_ids: Sequence[int],
        *,
        generation_state: Mapping[str, Any] | None = None,
    ) -> ScoreVariantsResult:
        """Score one exact continuation under related exact-token prompts.

        This native endpoint is strictly an unqualified screening primitive. Prefix reuse and
        multi-sequence execution may produce small numerical differences from scalar :meth:`score`.
        The returned type therefore rejects any response that claims qualification or trusted-evidence
        use. Remeasure claim-critical and borderline effects through scalar ``/score``. An optional
        ``generation_state`` can reference the original generation's worker-local checkpoint; the
        worker validates its identity and reports whether reuse was applied or safely skipped.
        """
        if isinstance(variants, (str, bytes, bytearray)) or not isinstance(variants, Sequence):
            raise ValueError("variants must be a non-empty array")
        items = list(variants)
        if not items or len(items) > _SCORE_VARIANTS_MAX:
            raise ValueError(
                f"variants must contain between 1 and {_SCORE_VARIANTS_MAX} entries")

        normalized: list[dict[str, Any]] = []
        variant_ids: list[str] = []
        prompt_lengths: list[int] = []
        seen: set[str] = set()
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"variants[{index}] must be an object")
            if set(item) != {"id", "prompt_ids"}:
                raise ValueError(
                    f"variants[{index}] must contain exactly 'id' and 'prompt_ids'")
            variant_id = item.get("id")
            id_is_safe = (
                isinstance(variant_id, str)
                and 0 < len(variant_id) <= _SCORE_VARIANT_ID_MAX
                and all(
                    character.isascii()
                    and (character.isalnum() or character in "_-.:")
                    for character in variant_id
                )
            )
            if not id_is_safe:
                raise ValueError(
                    f"variants[{index}].id must be 1..{_SCORE_VARIANT_ID_MAX} "
                    "ASCII-safe characters")
            if variant_id in seen:
                raise ValueError(f"duplicate variant id {variant_id!r}")
            seen.add(variant_id)
            prompt_value = item.get("prompt_ids")
            if (
                isinstance(prompt_value, (str, bytes, bytearray))
                or not isinstance(prompt_value, Sequence)
            ):
                raise ValueError(f"variants[{index}].prompt_ids must be an integer sequence")
            prompt_ids = self._ids(prompt_value, f"variants[{index}].prompt_ids")
            normalized.append({"id": variant_id, "prompt_ids": prompt_ids})
            variant_ids.append(variant_id)
            prompt_lengths.append(len(prompt_ids))

        if (
            isinstance(continuation_ids, (str, bytes, bytearray))
            or not isinstance(continuation_ids, Sequence)
        ):
            raise ValueError("continuation_ids must be an integer sequence")
        continuation = self._ids(continuation_ids, "continuation_ids")
        body: dict[str, Any] = {
            "variants": normalized,
            "continuation_ids": continuation,
        }
        if generation_state is not None:
            if not isinstance(generation_state, Mapping):
                raise ValueError("generation_state must be an object")
            required = {"checkpoint_id", "worker_generation_id", "baseline_variant_id"}
            if set(generation_state) != required:
                raise ValueError(
                    "generation_state must contain exactly checkpoint_id, "
                    "worker_generation_id, and baseline_variant_id")
            checkpoint_id = generation_state.get("checkpoint_id")
            worker_generation_id = generation_state.get("worker_generation_id")
            baseline_variant_id = generation_state.get("baseline_variant_id")
            if (not isinstance(checkpoint_id, str) or not checkpoint_id
                    or len(checkpoint_id) > 512):
                raise ValueError(
                    "generation_state.checkpoint_id must be a bounded non-empty string")
            if (not isinstance(worker_generation_id, str) or not worker_generation_id
                    or len(worker_generation_id) > 128):
                raise ValueError(
                    "generation_state.worker_generation_id must be a bounded non-empty string")
            if baseline_variant_id not in seen:
                raise ValueError(
                    "generation_state.baseline_variant_id must name a request variant")
            body["generation_state"] = {
                "checkpoint_id": checkpoint_id,
                "worker_generation_id": worker_generation_id,
                "baseline_variant_id": baseline_variant_id,
            }
        response = self._transport.request_json(
            "POST",
            "/v1/score-variants",
            body=body,
        )
        return ScoreVariantsResult.from_json(
            response,
            variant_ids=variant_ids,
            prompt_lengths=prompt_lengths,
            continuation_ids=continuation,
        )

    def knockout_score(self, *, knockouts: Sequence[AttentionKnockout],
                        prompt: str | None = None, prompt_ids: Sequence[int] | None = None,
                        continuation: str | None = None,
                        continuation_ids: Sequence[int] | None = None,
                        topk: int = 0) -> ScoreResult:
        """Teacher-force a continuation while cutting named attention query->key edges."""
        return self.score(
            prompt=prompt,
            prompt_ids=prompt_ids,
            continuation=continuation,
            continuation_ids=continuation_ids,
            topk=topk,
            attn_knockout=knockouts,
        )

    def score_request(self, request: ScoreRequest, *, arm: InterventionArm | None = None) -> ScoreResult:
        """Execute a typed request, optionally under one named intervention arm."""
        if not isinstance(request, ScoreRequest):
            raise ValueError("request must be a ScoreRequest")
        if arm is not None and not isinstance(arm, InterventionArm):
            raise ValueError("arm must be an InterventionArm")
        return self.score(
            prompt=request.prompt,
            prompt_ids=request.prompt_ids,
            continuation=request.continuation,
            continuation_ids=request.continuation_ids,
            topk=request.topk,
            steer=None if arm is None else arm.steer,
            steer_vec=None if arm is None else arm.steer_vec,
            attn_knockout=None if arm is None or not arm.attention_knockout else arm.attention_knockout,
        )

    def run_manifest(self, manifest: InterventionManifest) -> InterventionRunResult:
        """Replay one versioned scoring manifest against this explicitly selected engine.

        The baseline is always scored first with no intervention. Arms then run sequentially so the
        returned order is stable. ``support_drop`` is baseline sum-logprob minus arm sum-logprob;
        positive values mean the intervention made the recorded continuation less supported.
        """
        if not isinstance(manifest, InterventionManifest):
            raise ValueError("manifest must be an InterventionManifest")
        health: dict[str, Any] = {}
        if manifest.expected_health:
            health = self.health()
            self._require_expected(health, manifest.expected_health)
        baseline = self.score_request(manifest.request)
        arms = tuple(
            InterventionArmResult(
                name=arm.name,
                score=(score := self.score_request(manifest.request, arm=arm)),
                support_drop=baseline.sum_logprob - score.sum_logprob,
                metadata=arm.metadata,
            )
            for arm in manifest.arms
        )
        return InterventionRunResult(
            manifest_sha256=manifest.sha256,
            baseline=baseline,
            arms=arms,
            engine_health=health,
        )

    def replay_manifest(self, manifest: InterventionManifest) -> InterventionRunResult:
        """Alias for :meth:`run_manifest`, emphasizing deterministic manifest replay."""
        return self.run_manifest(manifest)

    @classmethod
    def _require_expected(cls, actual: Mapping[str, Any], expected: Mapping[str, Any],
                          prefix: str = "health") -> None:
        for key, expected_value in expected.items():
            if key not in actual:
                raise CloznProtocolError(f"{prefix}.{key} is missing from engine health")
            actual_value = actual[key]
            if isinstance(expected_value, Mapping):
                if not isinstance(actual_value, Mapping):
                    raise CloznProtocolError(f"{prefix}.{key} is not an object")
                cls._require_expected(actual_value, expected_value, f"{prefix}.{key}")
            elif actual_value != expected_value:
                raise CloznProtocolError(
                    f"{prefix}.{key} mismatch: expected {expected_value!r}, got {actual_value!r}"
                )

    @staticmethod
    def _exclusive(left: Any, right: Any, left_name: str, right_name: str) -> None:
        if (left is None) == (right is None):
            raise ValueError(f"provide exactly one of {left_name} or {right_name}")
        if left is not None and not isinstance(left, str):
            raise ValueError(f"{left_name} must be a string")

    @staticmethod
    def _ids(values: Sequence[int], label: str) -> list[int]:
        if isinstance(values, (str, bytes, bytearray)):
            raise ValueError(f"{label} must be an integer sequence")
        result = []
        for value in values:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{label} must contain non-negative integers")
            result.append(value)
        if not result:
            raise ValueError(f"{label} must not be empty")
        return result
