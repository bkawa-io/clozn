from __future__ import annotations

from threading import Event

import pytest

from clozn.experiments.multi_arm import (
    BatchCancelled,
    MultiArmError,
    ScoreTokensRefinementPolicy,
    concurrent_many,
    probe_reference_match_many,
    score_tokens_many,
    score_tokens_many_screen_then_refine,
)


CONTRACT = {
    "decode_mode": "greedy",
    "max_new": 4,
    "stop": [],
    "expected_termination": {"reason": "stop", "reason_raw": "eos"},
}


class ScalarSubstrate:
    def __init__(self):
        self.score_calls = []
        self.probe_calls = []
        self.chat_calls = 0

    def score_tokens(self, messages, continuation_ids, *, block=None,
                     steer_vec=None, topk=0, continuation=None):
        self.score_calls.append({
            "messages": messages,
            "continuation_ids": continuation_ids,
            "block": block,
            "steer_vec": steer_vec,
            "topk": topk,
            "continuation": continuation,
        })
        value = float(len(str(block or "")))
        return [{"id": token, "piece": str(token), "logprob": -(value + index)}
                for index, token in enumerate(continuation_ids or [1, 2])]

    def probe_reference_match(self, messages, reference_token_ids, *, generation_contract,
                              explicit_conditions=None):
        conditions = dict(explicit_conditions or {})
        self.probe_calls.append({
            "messages": messages,
            "reference_token_ids": reference_token_ids,
            "generation_contract": generation_contract,
            "explicit_conditions": conditions,
        })
        block = str(conditions.get("block") or "")
        if block == "termination-mismatch":
            return {
                "status": "diverged",
                "matched_token_count": 2,
                "first_divergence_index": None,
                "divergence_kind": "termination_mismatch",
                "termination_match": False,
            }
        if block == "diverge-late":
            return {
                "status": "diverged",
                "matched_token_count": 3,
                "first_divergence_index": 3,
                "divergence_kind": "token_mismatch",
                "termination_match": True,
            }
        return {
            "status": "matched",
            "matched_token_count": len(reference_token_ids),
            "first_divergence_index": None,
            "divergence_kind": None,
            "termination_match": True,
        }


class NativeBatchSubstrate(ScalarSubstrate):
    """A deliberately completion-reordered native fake."""

    score_tokens_many_proof_grade = True
    probe_reference_match_many_proof_grade = True

    def score_tokens_many(self, arms, *, cancel=None):
        results = [{"arm_index": index, "result": self.score_tokens(**arm)}
                   for index, arm in enumerate(arms)]
        return list(reversed(results))

    def probe_reference_match_many(self, arms, *, cancel=None):
        results = [{"arm_index": index, "result": self.probe_reference_match(**arm)}
                   for index, arm in enumerate(arms)]
        return list(reversed(results))


def _score_arms():
    return [
        {"messages": [{"role": "user", "content": "q"}], "continuation_ids": [11, 12],
         "block": "a"},
        {"messages": [{"role": "user", "content": "q"}], "continuation_ids": [11, 12],
         "block": "b"},
        {"messages": [{"role": "user", "content": "q"}], "continuation_ids": [11, 12],
         "block": "c"},
    ]


def _related_score_arms():
    return [
        {"messages": [{"role": "user", "content": f"q{index}"}],
         "continuation_ids": [11, 12], "block": None}
        for index in range(3)
    ]


def _probe_arms():
    return [
        {"messages": [{"role": "user", "content": "q"}], "reference_token_ids": [1, 2, 3, 4],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": "matched"}},
        {"messages": [{"role": "user", "content": "q"}], "reference_token_ids": [1, 2, 3, 4],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": "diverge-late"}},
        {"messages": [{"role": "user", "content": "q"}], "reference_token_ids": [1, 2, 3, 4],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": "termination-mismatch"}},
    ]


def test_score_many_keeps_undeclared_many_method_on_scalar_reference_path():
    substrate = NativeBatchSubstrate()
    result = score_tokens_many(substrate, _score_arms())

    assert result.execution_strategy == "scalar_serial"
    assert result.numerical_regime == "scalar_reference"
    assert result.evidence_use == "trusted"
    assert result.qualified is True
    assert result.trusted is True
    assert len(substrate.score_calls) == 3


def test_score_many_preserves_qualified_native_envelope_metadata():
    class QualifiedNative(ScalarSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_prefix_reuse"

        def score_tokens_many(self, arms, *, cancel=None):
            return {
                "results": [self.score_tokens(**arm) for arm in arms],
                "execution_metadata": {
                    "execution_strategy": "native_prefix_reuse",
                    "numerical_regime": "prefix_reuse_fp32_v1",
                    "evidence_use": "trusted",
                    "qualified": True,
                    "execution_identity": {"implementation": "fixture", "version": 1},
                },
                "metrics": {"reused_prompt_tokens": 17},
            }

    substrate = QualifiedNative()
    result = score_tokens_many(substrate, _related_score_arms())

    assert result.execution_strategy == "native_prefix_reuse"
    assert result.numerical_regime == "prefix_reuse_fp32_v1"
    assert result.trusted is True
    assert result.execution_identity == {"implementation": "fixture", "version": 1}
    assert result.metadata["metrics"] == {"reused_prompt_tokens": 17}


def test_native_score_many_is_unqualified_without_execution_identity():
    class MissingIdentity(ScalarSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_multisequence"

        def score_tokens_many(self, arms, *, cancel=None):
            return {
                "results": [self.score_tokens(**arm) for arm in arms],
                "execution_strategy": "native_multisequence",
                "numerical_regime": "batched_fp32_v1",
                "evidence_use": "trusted",
                "qualified": True,
                "metrics": {"waves": 1},
            }

    result = score_tokens_many(MissingIdentity(), _related_score_arms())

    assert result.execution_strategy == "native_multisequence"
    assert result.qualified is False
    assert result.trusted is False
    assert result.metadata["metrics"] == {"waves": 1}


def test_native_score_many_reports_but_does_not_qualify_strategy_mismatch():
    class MismatchedNative(ScalarSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_prefix_reuse"

        def score_tokens_many(self, arms, *, cancel=None):
            return {
                "results": [self.score_tokens(**arm) for arm in arms],
                "execution_strategy": "native_multisequence",
                "numerical_regime": "batched_fp32_v1",
                "evidence_use": "trusted",
                "qualified": True,
                "execution_identity": {"implementation": "fixture", "version": 1},
            }

    result = score_tokens_many(MismatchedNative(), _related_score_arms())

    assert result.execution_strategy == "native_multisequence"
    assert result.qualified is False
    assert result.trusted is False


def test_native_screen_refines_only_explicit_proof_critical_and_borderline_arms():
    class ScreeningSubstrate(ScalarSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_prefix_reuse"

        def __init__(self):
            super().__init__()
            self.native_calls = 0

        def score_tokens_many(self, arms, *, cancel=None):
            self.native_calls += 1
            return {
                "results": [
                    [{"id": token_id, "logprob": -(arm_index + token_index + 1.0)}
                     for token_index, token_id in enumerate(arm["continuation_ids"])]
                    for arm_index, arm in enumerate(arms)
                ],
                "execution_metadata": {
                    "execution_strategy": "native_prefix_reuse",
                    "numerical_regime": "screening_unqualified",
                    "evidence_use": "screening",
                    "qualified": False,
                    "execution_identity": {"implementation": "fixture", "version": 1},
                },
            }

    substrate = ScreeningSubstrate()
    seen = []

    def borderline(rows):
        seen.append(rows)
        return [row.arm_id for row in rows if row.total_logprob < -6.0]

    result = score_tokens_many_screen_then_refine(
        substrate,
        _related_score_arms(),
        arm_ids=("arm-a", "arm-b", "arm-c"),
        policy=ScoreTokensRefinementPolicy(
            proof_critical_arm_ids=("arm-b",),
            borderline_selector=borderline,
        ),
    )

    assert substrate.native_calls == 1
    assert len(seen) == 1
    assert [row.arm_id for row in seen[0]] == ["arm-a", "arm-b", "arm-c"]
    assert result.screening.evidence_use == "screening"
    assert result.screening.trusted is False
    assert result.scalar_refinement is not None
    assert result.scalar_refinement.numerical_regime == "scalar_reference"
    assert result.scalar_refinement.trusted is True
    assert result.proof_critical_arm_ids == ("arm-b",)
    assert result.borderline_arm_ids == ("arm-c",)
    assert result.selected_arm_ids == ("arm-b", "arm-c")
    assert [row.arm_id for row in result.refined_rows] == ["arm-b", "arm-c"]
    assert [call["messages"][0]["content"] for call in substrate.score_calls] == ["q1", "q2"]
    provenance = result.provenance()
    assert provenance["screening_execution"]["numerical_regime"] == "screening_unqualified"
    assert provenance["scalar_refinement_execution"]["numerical_regime"] == "scalar_reference"
    assert "numerical_regime" not in provenance


def test_native_screen_refinement_fails_closed_on_unknown_selector_arm():
    class ScreeningSubstrate(ScalarSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_prefix_reuse"

        def score_tokens_many(self, arms, *, cancel=None):
            return {
                "results": [
                    [{"id": token_id, "logprob": -1.0} for token_id in arm["continuation_ids"]]
                    for arm in arms
                ],
                "execution_metadata": {
                    "execution_strategy": "native_prefix_reuse",
                    "numerical_regime": "screening_unqualified",
                    "evidence_use": "screening",
                    "qualified": False,
                    "execution_identity": {"implementation": "fixture", "version": 1},
                },
            }

    substrate = ScreeningSubstrate()
    with pytest.raises(MultiArmError, match="not present"):
        score_tokens_many_screen_then_refine(
            substrate,
            _related_score_arms()[:2],
            arm_ids=("arm-a", "arm-b"),
            policy=ScoreTokensRefinementPolicy(
                borderline_selector=lambda _rows: ("unknown-arm",),
            ),
        )

    assert substrate.score_calls == []


def test_engine_scalar_wrapper_declares_its_actual_worker_strategy(monkeypatch):
    from clozn.server import app as cs

    substrate = object.__new__(cs.EngineSubstrate)
    monkeypatch.setenv("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "1")
    assert substrate.score_tokens_many_execution_strategy == "scalar_serial"
    monkeypatch.setenv("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "3")
    assert substrate.score_tokens_many_execution_strategy == "scalar_http_concurrent"


def test_engine_substrate_initializes_optional_steering_resolver():
    from clozn.server import app as cs

    engine = _NativeScoreVariantsEngine()
    substrate = cs.EngineSubstrate(engine=engine)

    assert substrate.steer is None


class _NativeScoreVariantsEngine:
    def __init__(self, *, capability=True):
        self.capability = capability
        self.template_calls = []
        self.scalar_calls = []
        self.variant_calls = []

    def health(self):
        return {
            "model": "",
            "capabilities": {"score_variants": self.capability},
        }

    def apply_template(self, messages, add_assistant=True):
        return "|".join(str(message.get("content") or "") for message in messages)

    def apply_template_info(self, messages, add_assistant=True, include_token_ids=False):
        copied = [dict(message) for message in messages]
        self.template_calls.append({
            "messages": copied,
            "add_assistant": add_assistant,
            "include_token_ids": include_token_ids,
        })
        content = "|".join(str(message.get("content") or "") for message in messages)
        return {
            "prompt": content,
            "prompt_token_ids": [700, len(content)],
        }

    def score(self, *, prompt, topk, continuation_ids):
        self.scalar_calls.append({
            "prompt": prompt, "topk": topk,
            "continuation_ids": list(continuation_ids),
        })
        return {"tokens": [
            {"id": token_id, "piece": str(token_id), "logprob": -float(index + 1)}
            for index, token_id in enumerate(continuation_ids)
        ]}

    def score_variants(self, variants, continuation_ids):
        copied_variants = [
            {"id": variant["id"], "prompt_ids": list(variant["prompt_ids"])}
            for variant in variants
        ]
        self.variant_calls.append({
            "variants": copied_variants,
            "continuation_ids": list(continuation_ids),
        })
        ids = [variant["id"] for variant in copied_variants]
        return {
            "schema_version": "clozn.score-variants-worker.v1",
            "status": "completed",
            "continuation_ids": list(continuation_ids),
            "results": [
                {
                    "id": variant["id"],
                    "status": "completed",
                    "prompt_token_count": len(variant["prompt_ids"]),
                    "target_logprobs": [
                        -float(arm_index + token_index + 1)
                        for token_index, _token_id in enumerate(continuation_ids)
                    ],
                }
                for arm_index, variant in enumerate(copied_variants)
            ],
            "execution": {
                "strategy": "native_prefix_reuse_step_batch",
                "numerical_regime": "screening_unqualified",
                "evidence_use": "screening",
                "qualified": False,
                "execution_identity": {
                    "algorithm_id": "clozn.score-variants.fixture.v1",
                    "worker_generation_id": "worker-fixture",
                },
                "waves": [{
                    "wave_index": 0,
                    "variant_ids": ids,
                    "unique_prompt_count": len(ids),
                    "logical_prompt_rows": 4,
                    "physical_prompt_rows": 3,
                    "continuation_kv_rows": len(ids),
                    "required_kv_rows": 5,
                }],
            },
            "metrics": {"decoded_rows": 5},
            "future_additive_field": {"retained": True},
        }


def test_engine_score_many_defaults_to_scalar_even_when_worker_has_native_capability(monkeypatch):
    from clozn.server import app as cs

    monkeypatch.delenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", raising=False)
    monkeypatch.setenv("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "1")
    engine = _NativeScoreVariantsEngine()
    substrate = cs.EngineSubstrate(engine=engine)

    rows = substrate.score_tokens_many(_related_score_arms())

    assert substrate.score_tokens_many_execution_strategy == "scalar_serial"
    assert len(rows) == 3
    assert len(engine.scalar_calls) == 3
    assert engine.variant_calls == []
    assert substrate.native_score_variants_available is True

    monkeypatch.setenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", "1")
    incapable = cs.EngineSubstrate(engine=_NativeScoreVariantsEngine(capability=False))
    assert incapable.score_tokens_many_execution_strategy == "scalar_serial"
    assert incapable.native_score_variants_available is False


def test_explicit_engine_native_screen_bypasses_only_automatic_rollout_gate(monkeypatch):
    from clozn.server import app as cs

    monkeypatch.delenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", raising=False)
    monkeypatch.setenv("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "1")
    engine = _NativeScoreVariantsEngine()
    substrate = cs.EngineSubstrate(engine=engine)

    result = substrate.score_tokens_many(
        _related_score_arms(), require_native_screening=True,
    )

    assert substrate.score_tokens_many_execution_strategy == "scalar_serial"
    assert result["execution_strategy"] == "native_prefix_reuse_step_batch"
    assert result["evidence_use"] == "screening"
    assert result["rollout"]["reason_code"] == "required_by_screen_then_refine"
    assert len(engine.variant_calls) == 1
    assert engine.scalar_calls == []


def test_engine_native_score_many_uses_exact_prompt_ids_and_stays_screening(monkeypatch):
    from clozn.server import app as cs

    monkeypatch.setenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", "1")
    monkeypatch.setenv("CLOZN_NATIVE_SCORE_VARIANTS_MIN_ARMS", "1")
    engine = _NativeScoreVariantsEngine()
    substrate = cs.EngineSubstrate(engine=engine)

    result = score_tokens_many(substrate, _related_score_arms())

    assert substrate.score_tokens_many_execution_strategy == "native_prefix_reuse_step_batch"
    assert len(engine.variant_calls) == 1
    assert engine.variant_calls[0] == {
        "variants": [
            {"id": "arm-0", "prompt_ids": [700, 2]},
            {"id": "arm-1", "prompt_ids": [700, 2]},
            {"id": "arm-2", "prompt_ids": [700, 2]},
        ],
        "continuation_ids": [11, 12],
    }
    assert all(call["include_token_ids"] is True for call in engine.template_calls)
    assert engine.scalar_calls == []
    assert result.execution_strategy == "native_prefix_reuse_step_batch"
    assert result.numerical_regime == "screening_unqualified"
    assert result.evidence_use == "screening"
    assert result.qualified is False
    assert result.trusted is False
    assert result.metadata["metrics"] == {"decoded_rows": 5}
    assert result.metadata["worker_response"]["future_additive_field"] == {"retained": True}


@pytest.mark.parametrize("incompatible", [
    {"topk": 1},
    {"continuation": "answer"},
    {"steer_vec": [0.25]},
    {"continuation_ids": [11, 13]},
])
def test_engine_native_score_many_rejects_incompatible_arms_before_dispatch(
        monkeypatch, incompatible):
    from clozn.server import app as cs

    monkeypatch.setenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", "true")
    engine = _NativeScoreVariantsEngine()
    substrate = cs.EngineSubstrate(engine=engine)
    arms = _related_score_arms()[:2]
    arms[1] = {**arms[1], **incompatible}

    with pytest.raises(MultiArmError) as raised:
        substrate.score_tokens_many(arms)

    assert raised.value.arm_index == 1
    assert engine.variant_calls == []


def test_engine_native_score_many_partial_failure_never_fabricates_completion(monkeypatch):
    from clozn.server import app as cs

    monkeypatch.setenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", "1")
    monkeypatch.setenv("CLOZN_NATIVE_SCORE_VARIANTS_MIN_ARMS", "1")
    engine = _NativeScoreVariantsEngine()
    complete_response = engine.score_variants

    def partial_response(variants, continuation_ids):
        response = complete_response(variants, continuation_ids)
        response["status"] = "partial"
        response["results"][1] = {
            "id": variants[1]["id"],
            "status": "failed",
            "prompt_token_count": len(variants[1]["prompt_ids"]),
            "error": "fixture decode failure",
        }
        return response

    engine.score_variants = partial_response
    substrate = cs.EngineSubstrate(engine=engine)

    with pytest.raises(MultiArmError) as raised:
        substrate.score_tokens_many(_related_score_arms()[:2])

    assert raised.value.arm_index == 1
    assert raised.value.completed_indices == (0,)
    assert raised.value.dispatched_indices == (0, 1)
    assert raised.value.completed == [[
        {"id": 11, "logprob": -1.0},
        {"id": 12, "logprob": -2.0},
    ]]


def test_engine_native_rollout_uses_honest_scalar_fallback_below_crossover(monkeypatch):
    from clozn.server import app as cs

    monkeypatch.setenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", "1")
    monkeypatch.setenv("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "1")
    monkeypatch.delenv("CLOZN_NATIVE_SCORE_VARIANTS_MIN_ARMS", raising=False)
    engine = _NativeScoreVariantsEngine()
    substrate = cs.EngineSubstrate(engine=engine)

    result = score_tokens_many(substrate, _related_score_arms())

    assert result.execution_strategy == "scalar_serial"
    assert result.numerical_regime == "scalar_reference"
    assert result.evidence_use == "trusted"
    assert result.qualified is True
    assert result.trusted is True
    assert len(engine.scalar_calls) == 3
    assert engine.variant_calls == []
    assert engine.template_calls == []
    assert result.metadata["rollout"] == {
        "decision": "scalar_fallback",
        "reason_code": "variant_count_below_native_threshold",
        "variant_count": 3,
        "minimum_variant_count": 4,
        "minimum_prefix_reuse_ratio": 0.25,
        "template_preflight_performed": False,
        "template_preflight_seconds": 0.0,
    }


def test_engine_native_rollout_falls_back_when_exact_prefix_reuse_is_too_low(monkeypatch):
    from clozn.server import app as cs

    class LowReuseEngine(_NativeScoreVariantsEngine):
        def apply_template_info(self, messages, add_assistant=True, include_token_ids=False):
            result = super().apply_template_info(
                messages, add_assistant=add_assistant,
                include_token_ids=include_token_ids,
            )
            ordinal = len(self.template_calls)
            result["prompt_token_ids"] = [700 + ordinal, 900 + ordinal]
            return result

    monkeypatch.setenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", "1")
    monkeypatch.setenv("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "1")
    monkeypatch.delenv("CLOZN_NATIVE_SCORE_VARIANTS_MIN_ARMS", raising=False)
    monkeypatch.delenv(
        "CLOZN_NATIVE_SCORE_VARIANTS_MIN_PREFIX_REUSE_RATIO", raising=False)
    engine = LowReuseEngine()
    substrate = cs.EngineSubstrate(engine=engine)
    arms = _related_score_arms() + [{
        "messages": [{"role": "user", "content": "q3"}],
        "continuation_ids": [11, 12], "block": None,
    }]

    result = score_tokens_many(substrate, arms)

    assert result.execution_strategy == "scalar_serial"
    assert result.trusted is True
    assert len(engine.scalar_calls) == 4
    assert engine.variant_calls == []
    rollout = result.metadata["rollout"]
    assert rollout["reason_code"] == "prefix_reuse_below_native_threshold"
    assert rollout["variant_count"] == 4
    assert rollout["unique_prompt_count"] == 4
    assert rollout["estimated_prefix_reuse_ratio"] == 0.0
    assert rollout["minimum_prefix_reuse_ratio"] == 0.25


def test_engine_native_rollout_admits_four_exactly_related_variants(monkeypatch):
    from clozn.server import app as cs

    monkeypatch.setenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", "1")
    monkeypatch.delenv("CLOZN_NATIVE_SCORE_VARIANTS_MIN_ARMS", raising=False)
    monkeypatch.delenv(
        "CLOZN_NATIVE_SCORE_VARIANTS_MIN_PREFIX_REUSE_RATIO", raising=False)
    engine = _NativeScoreVariantsEngine()
    substrate = cs.EngineSubstrate(engine=engine)
    arms = _related_score_arms() + [{
        "messages": [{"role": "user", "content": "q3"}],
        "continuation_ids": [11, 12], "block": None,
    }]

    result = score_tokens_many(substrate, arms)

    assert result.execution_strategy == "native_prefix_reuse_step_batch"
    assert result.evidence_use == "screening"
    assert result.trusted is False
    assert len(engine.variant_calls) == 1
    assert engine.scalar_calls == []
    rollout = result.metadata["rollout"]
    assert rollout["decision"] == "native_admitted"
    assert rollout["reason_code"] == "workload_thresholds_met"
    assert rollout["variant_count"] == 4
    assert rollout["estimated_prefix_reuse_ratio"] >= 0.25


def test_adaptive_scalar_fallback_requires_the_exact_scalar_execution_identity():
    class ForgedScalarFallback(ScalarSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_prefix_reuse_step_batch"

        def score_tokens_many(self, arms, *, cancel=None):
            return {
                "results": [self.score_tokens(**arm) for arm in arms],
                "execution_strategy": "scalar_serial",
                "numerical_regime": "scalar_reference",
                "evidence_use": "trusted",
                "qualified": True,
                "execution_identity": {"kind": "forged-scalar-identity"},
            }

    result = score_tokens_many(
        ForgedScalarFallback(), _related_score_arms(),
    )

    assert result.execution_strategy == "scalar_serial"
    assert result.numerical_regime == "scalar_reference"
    assert result.qualified is False
    assert result.trusted is False


def test_screen_then_refine_can_require_native_below_rollout_crossover(monkeypatch):
    from clozn.server import app as cs

    monkeypatch.delenv("CLOZN_ENABLE_NATIVE_SCORE_VARIANTS", raising=False)
    monkeypatch.setenv("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "1")
    monkeypatch.delenv("CLOZN_NATIVE_SCORE_VARIANTS_MIN_ARMS", raising=False)
    engine = _NativeScoreVariantsEngine()
    substrate = cs.EngineSubstrate(engine=engine)

    result = score_tokens_many_screen_then_refine(
        substrate,
        _related_score_arms(),
        arm_ids=("arm-a", "arm-b", "arm-c"),
        policy=ScoreTokensRefinementPolicy(proof_critical_arm_ids=("arm-b",)),
    )

    assert len(engine.variant_calls) == 1
    assert len(engine.scalar_calls) == 1
    assert substrate.score_tokens_many_execution_strategy == "scalar_serial"
    assert result.selected_arm_ids == ("arm-b",)
    assert result.screening.evidence_use == "screening"
    assert result.screening.metadata["rollout"]["reason_code"] == (
        "required_by_screen_then_refine"
    )


def test_scalar_and_native_batch_exact_results_preserve_mixed_divergence_and_termination():
    scalar = ScalarSubstrate()
    expected = [scalar.probe_reference_match(**arm) for arm in _probe_arms()]
    native = NativeBatchSubstrate()
    actual = probe_reference_match_many(native, _probe_arms())
    assert actual == expected
    assert [item["status"] for item in actual] == ["matched", "diverged", "diverged"]
    assert actual[1]["first_divergence_index"] == 3
    assert actual[2]["divergence_kind"] == "termination_mismatch"


def test_batch_cancellation_returns_completed_results_and_does_not_dispatch_queued_groups():
    substrate = ScalarSubstrate()
    cancel = Event()
    original = substrate.score_tokens

    def score_and_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        cancel.set()
        return result

    substrate.score_tokens = score_and_cancel
    with pytest.raises(BatchCancelled) as raised:
        score_tokens_many(substrate, _score_arms(), cancel=cancel)
    assert len(raised.value.completed) == 1
    assert len(substrate.score_calls) == 1


def test_different_messages_share_one_native_score_batch_when_contract_matches():
    class GroupingSubstrate(ScalarSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_prefix_reuse"

        def __init__(self):
            super().__init__()
            self.batches = []

        def score_tokens_many(self, arms, *, cancel=None):
            self.batches.append(list(arms))
            return {
                "results": [self.score_tokens(**arm) for arm in arms],
                "execution_metadata": {
                    "execution_strategy": "native_prefix_reuse",
                    "numerical_regime": "fixture_fp32_v1",
                    "evidence_use": "trusted",
                    "qualified": True,
                    "execution_identity": {"implementation": "fixture", "version": 1},
                },
            }

    substrate = GroupingSubstrate()
    arms = [
        {"messages": [{"role": "user", "content": "minus A"}], "continuation_ids": [1, 2], "block": None},
        {"messages": [{"role": "user", "content": "minus B"}], "continuation_ids": [1, 2], "block": None},
        {"messages": [{"role": "user", "content": "minus C"}], "continuation_ids": [1, 2], "block": None},
    ]
    score_tokens_many(substrate, arms)
    assert len(substrate.batches) == 1
    assert [arm["messages"][0]["content"] for arm in substrate.batches[0]] == [
        "minus A", "minus B", "minus C"
    ]


def test_different_messages_share_exact_batch_but_contract_changes_split_groups():
    class GroupingSubstrate(ScalarSubstrate):
        probe_reference_match_many_proof_grade = True

        def __init__(self):
            super().__init__()
            self.batches = []

        def probe_reference_match_many(self, arms, *, cancel=None):
            self.batches.append(list(arms))
            return [self.probe_reference_match(**arm) for arm in arms]

    substrate = GroupingSubstrate()
    arms = [
        {"messages": [{"role": "user", "content": "minus A"}], "reference_token_ids": [1, 2],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": None}},
        {"messages": [{"role": "user", "content": "minus B"}], "reference_token_ids": [1, 2],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": None}},
        {"messages": [{"role": "user", "content": "different contract"}], "reference_token_ids": [1, 2],
         "generation_contract": {**CONTRACT, "max_new": 5}, "explicit_conditions": {"block": None}},
    ]
    probe_reference_match_many(substrate, arms)
    assert [len(batch) for batch in substrate.batches] == [2, 1]


def test_different_messages_cancel_inside_one_scalar_group_without_unavailable_evidence():
    substrate = ScalarSubstrate()
    cancel = Event()
    original = substrate.score_tokens

    def score_and_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(substrate.score_calls) == 3:
            cancel.set()
        return result

    substrate.score_tokens = score_and_cancel
    arms = [
        {"messages": [{"role": "user", "content": f"minus {index}"}], "continuation_ids": [1, 2], "block": None}
        for index in range(5)
    ]
    with pytest.raises(BatchCancelled) as raised:
        score_tokens_many(substrate, arms, cancel=cancel)
    assert len(substrate.score_calls) == 3
    assert len(raised.value.completed) == 3


def test_bounded_concurrent_many_preserves_scalar_result_order():
    calls = []

    def scalar(value):
        calls.append(value)
        return {"value": value}

    result = concurrent_many(scalar, [{"value": 0}, {"value": 1}, {"value": 2}], max_workers=2)
    assert result == [{"value": 0}, {"value": 1}, {"value": 2}]
    assert sorted(calls) == [0, 1, 2]


def test_malformed_arm_is_rejected_before_any_dispatch():
    substrate = ScalarSubstrate()
    with pytest.raises(MultiArmError) as raised:
        score_tokens_many(substrate, [_score_arms()[0], {"messages": "not-a-list"}])
    assert raised.value.arm_index == 1
    assert substrate.score_calls == []
