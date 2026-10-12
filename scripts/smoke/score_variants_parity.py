"""Live parity battery for POST /v1/score-variants.

The native endpoint deliberately uses a different execution shape from scalar
``POST /score``, so bit-for-bit equality is not expected.  This battery checks
the protocol invariants and measures the numerical drift against independent
scalar scoring on a real worker.

Usage:
    python scripts/smoke/score_variants_parity.py --base-url http://127.0.0.1:18123
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Case:
    name: str
    variants: list[dict[str, Any]]
    continuation_ids: list[int]
    expect_multiple_waves: bool = False
    expect_duplicate_pair: tuple[str, str] | None = None
    expect_prefix_reuse: bool = False
    expect_zero_continuation_kv: bool = False
    generation_state: dict[str, str] | None = None
    expect_generation_reuse: bool = False
    expect_generation_fallback: bool = False
    parity_gate: bool = True


def _request(base_url: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"content-type": "application/json"},
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path} returned a non-object JSON response")
    return payload


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _assert_completed(response: dict[str, Any], case: Case) -> list[dict[str, Any]]:
    if response.get("status") != "completed":
        raise AssertionError(f"{case.name}: native status is {response.get('status')!r}")
    if response.get("continuation_ids") != case.continuation_ids:
        raise AssertionError(f"{case.name}: continuation_ids were not echoed exactly")
    results = response.get("results")
    expected_ids = [variant["id"] for variant in case.variants]
    if not isinstance(results, list) or [row.get("id") for row in results] != expected_ids:
        raise AssertionError(f"{case.name}: result ordering differs from caller ordering")
    for row in results:
        scores = row.get("target_logprobs")
        if row.get("status") != "completed":
            raise AssertionError(f"{case.name}/{row.get('id')}: row did not complete")
        if not isinstance(scores, list) or len(scores) != len(case.continuation_ids):
            raise AssertionError(f"{case.name}/{row.get('id')}: wrong target_logprobs shape")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in scores):
            raise AssertionError(f"{case.name}/{row.get('id')}: non-finite target logprob")
    execution = response.get("execution") or {}
    strategy = execution.get("strategy")
    allowed_strategies = (
        {"native_generation_prefix_reuse_step_batch",
         "native_generation_prefix_reuse_step_batch_mixed"}
        if case.expect_generation_reuse else {"native_prefix_reuse_step_batch"}
    )
    if strategy not in allowed_strategies:
        raise AssertionError(f"{case.name}: unexpected execution strategy")
    if execution.get("qualified") is not False or execution.get("evidence_use") != "screening":
        raise AssertionError(f"{case.name}: native screen was incorrectly promoted to trusted evidence")
    waves = execution.get("waves")
    if not isinstance(waves, list) or not waves:
        raise AssertionError(f"{case.name}: missing execution waves")
    wave_ids = [variant_id for wave in waves for variant_id in wave.get("variant_ids", [])]
    if wave_ids != expected_ids:
        raise AssertionError(f"{case.name}: wave membership does not cover variants in order")
    if case.expect_multiple_waves and len(waves) < 2:
        raise AssertionError(f"{case.name}: expected at least two KV-aware waves")
    if case.expect_prefix_reuse and not any(
        wave.get("physical_prompt_rows", 0) < wave.get("deduplicated_prompt_rows", 0)
        for wave in waves
    ):
        raise AssertionError(f"{case.name}: no physical prompt-row saving was reported")
    if case.expect_zero_continuation_kv and any(wave.get("continuation_kv_rows") != 0 for wave in waves):
        raise AssertionError(f"{case.name}: one-token continuation should retain no continuation KV rows")
    if case.expect_generation_reuse:
        reuse = execution.get("generation_state_reuse") or {}
        if reuse.get("status") != "applied" or reuse.get("applied_wave_count") != len(waves):
            raise AssertionError(f"{case.name}: original-generation state was not applied to every wave")
        if any(not (wave.get("generation_state_reuse") or {}).get("applied") for wave in waves):
            raise AssertionError(f"{case.name}: a wave omitted generation-state reuse")
    if case.expect_generation_fallback:
        reuse = execution.get("generation_state_reuse") or {}
        if reuse.get("applied_wave_count") != 0 or reuse.get("status") != (
            "checkpoint_shared_prefix_too_short"
        ):
            raise AssertionError(f"{case.name}: shallow checkpoint reuse did not fail safe")
        if any((wave.get("generation_state_reuse") or {}).get("applied") for wave in waves):
            raise AssertionError(f"{case.name}: a shallow-prefix wave reused generation KV")
    if case.expect_duplicate_pair:
        by_id = {row["id"]: row for row in results}
        left, right = case.expect_duplicate_pair
        if by_id[left]["target_logprobs"] != by_id[right]["target_logprobs"]:
            raise AssertionError(f"{case.name}: duplicate prompts produced different native scores")
    return results


def _scalar_scores(
    base_url: str,
    prompt_ids: list[int],
    continuation_ids: list[int],
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], list[float]],
) -> list[float]:
    key = (tuple(prompt_ids), tuple(continuation_ids))
    if key in cache:
        return cache[key]
    response = _request(base_url, "/score", {
        "prompt_ids": prompt_ids,
        "continuation_ids": continuation_ids,
        "topk": 0,
    })
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != len(continuation_ids):
        raise AssertionError("scalar /score returned the wrong token count")
    if [token.get("id") for token in tokens] != continuation_ids:
        raise AssertionError("scalar /score returned different continuation token ids")
    scores = [float(token["logprob"]) for token in tokens]
    if not all(math.isfinite(value) for value in scores):
        raise AssertionError("scalar /score returned a non-finite logprob")
    cache[key] = scores
    return scores


def _build_cases(prompt_ids: list[int], continuation_ids: list[int]) -> list[Case]:
    if len(prompt_ids) < 24 or len(continuation_ids) < 3:
        raise RuntimeError("the selected template/continuation did not produce enough tokens")

    common = prompt_ids[:-3]
    shared = [
        {"id": "shared-0", "prompt_ids": prompt_ids},
        {"id": "shared-1", "prompt_ids": common + [prompt_ids[-3], 1000, prompt_ids[-1]]},
        {"id": "shared-2", "prompt_ids": common + [prompt_ids[-3], 1001, prompt_ids[-1]]},
        {"id": "shared-3", "prompt_ids": common + [1002, prompt_ids[-1]]},
    ]
    prefix_and_duplicate = [
        {"id": "prefix-8", "prompt_ids": prompt_ids[:8]},
        {"id": "prefix-16", "prompt_ids": prompt_ids[:16]},
        {"id": "prefix-16-copy", "prompt_ids": prompt_ids[:16]},
        {"id": "prefix-24", "prompt_ids": prompt_ids[:24]},
    ]
    multi_wave = [
        {"id": f"wave-{index:02d}", "prompt_ids": prompt_ids[:8] + [2000 + index]}
        for index in range(17)
    ]
    return [
        Case("shared-prefix", shared, continuation_ids[:3], expect_prefix_reuse=True),
        Case(
            "prefix-and-duplicate",
            prefix_and_duplicate,
            continuation_ids[:3],
            expect_duplicate_pair=("prefix-16", "prefix-16-copy"),
            expect_prefix_reuse=True,
        ),
        Case(
            "multi-wave",
            multi_wave,
            continuation_ids[:3],
            expect_multiple_waves=True,
            expect_prefix_reuse=True,
        ),
        Case(
            "one-token-continuation",
            shared,
            continuation_ids[:1],
            expect_prefix_reuse=True,
            expect_zero_continuation_kv=True,
        ),
    ]


def run(base_url: str, max_abs_error: float) -> dict[str, Any]:
    health = _request(base_url, "/health")
    if not (health.get("capabilities") or {}).get("score_variants"):
        raise RuntimeError("worker does not advertise capabilities.score_variants")

    rendered = _request(base_url, "/apply_template", {
        "messages": [{
            "role": "user",
            "content": "Explain why shared prefixes matter in one sentence.",
        }],
        "add_assistant": True,
        "include_token_ids": True,
    })
    prompt_ids = rendered.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or not all(type(token) is int for token in prompt_ids):
        raise RuntimeError("worker did not return exact template token ids")

    seed_score = _request(base_url, "/score", {
        "prompt_ids": prompt_ids,
        "continuation": "Shared prefixes avoid repeated computation.",
        "topk": 0,
    })
    continuation_ids = [token["id"] for token in seed_score.get("tokens", [])]
    cases = _build_cases(prompt_ids, continuation_ids)
    generated = _request(base_url, "/v1/completions", {
        "prompt": rendered["prompt"],
        "max_tokens": 4,
        "temperature": 0,
        "checkpoint_on_finish": True,
    })
    board = generated.get("board")
    checkpoint_id = generated.get("checkpoint_id")
    worker_generation_id = health.get("worker_generation_id")
    if (
        not isinstance(board, list)
        or len(board) <= len(prompt_ids)
        or not isinstance(checkpoint_id, str)
        or not checkpoint_id
        or not isinstance(worker_generation_id, str)
        or not worker_generation_id
    ):
        raise RuntimeError("worker did not return a reusable generation checkpoint")
    generated_ids = board[len(prompt_ids):]
    generation_variants = [
        {"id": "generated-baseline", "prompt_ids": prompt_ids},
        {"id": "generated-changed", "prompt_ids": prompt_ids[:-2] + [1000, prompt_ids[-1]]},
        {"id": "generated-changed-2", "prompt_ids": prompt_ids[:-3] + [1001, prompt_ids[-2], prompt_ids[-1]]},
    ]
    cases.append(Case(
        "original-generation-state",
        generation_variants,
        generated_ids,
        expect_prefix_reuse=True,
        generation_state={
            "checkpoint_id": checkpoint_id,
            "worker_generation_id": worker_generation_id,
            "baseline_variant_id": "generated-baseline",
        },
        expect_generation_reuse=True,
    ))
    shallow_variants = [
        {"id": "shallow-baseline", "prompt_ids": prompt_ids},
        {"id": "shallow-changed", "prompt_ids": prompt_ids[:-2] + [1000, prompt_ids[-1]]},
        {"id": "shallow-prefix", "prompt_ids": prompt_ids[:12]},
    ]
    cases.append(Case(
        "original-generation-state-shallow-fallback",
        shallow_variants,
        generated_ids,
        expect_prefix_reuse=True,
        generation_state={
            "checkpoint_id": checkpoint_id,
            "worker_generation_id": worker_generation_id,
            "baseline_variant_id": "shallow-baseline",
        },
        expect_generation_fallback=True,
        # This case validates admission/fallback mechanics with an adversarial synthetic token
        # mutation. Its numerical result remains visible but does not widen the predeclared
        # qualification fixture used by the ordinary and late-generation cases.
        parity_gate=False,
    ))
    scalar_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], list[float]] = {}
    reports = []
    all_errors: list[float] = []
    gated_errors: list[float] = []

    for case in cases:
        native_body = {
            "variants": case.variants,
            "continuation_ids": case.continuation_ids,
        }
        if case.generation_state is not None:
            native_body["generation_state"] = case.generation_state
        native = _request(base_url, "/v1/score-variants", native_body)
        rows = _assert_completed(native, case)
        errors = []
        sum_errors = []
        for variant, row in zip(case.variants, rows):
            scalar = _scalar_scores(
                base_url, variant["prompt_ids"], case.continuation_ids, scalar_cache)
            native_scores = [float(value) for value in row["target_logprobs"]]
            errors.extend(abs(left - right) for left, right in zip(native_scores, scalar))
            sum_errors.append(abs(sum(native_scores) - sum(scalar)))
        all_errors.extend(errors)
        if case.parity_gate:
            gated_errors.extend(errors)
        metrics = native.get("metrics") or {}
        reports.append({
            "name": case.name,
            "variant_count": len(case.variants),
            "continuation_token_count": len(case.continuation_ids),
            "wave_count": metrics.get("wave_count"),
            "planned_logical_prompt_rows": metrics.get("planned_logical_prompt_rows"),
            "planned_physical_prompt_rows": metrics.get("planned_physical_prompt_rows"),
            "max_abs_token_logprob_error": max(errors),
            "mean_abs_token_logprob_error": statistics.fmean(errors),
            "p95_abs_token_logprob_error": _percentile(errors, 0.95),
            "max_abs_sequence_logprob_error": max(sum_errors),
            "full_vocab_host_reduction": (native.get("execution") or {}).get(
                "full_vocab_host_reduction"),
            "target_logprob_reduction": (native.get("execution") or {}).get(
                "target_logprob_reduction"),
            "logprob_host_output_floats": metrics.get("logprob_host_output_floats"),
            "parity_gate": case.parity_gate,
            "within_parity_threshold": max(errors) <= max_abs_error,
        })

    overall_max = max(gated_errors)
    report = {
        "schema_version": "clozn.score-variants-parity.v1",
        "status": "PASS" if overall_max <= max_abs_error else "FAIL",
        "threshold": {"max_abs_token_logprob_error": max_abs_error},
        "worker": {
            "model": health.get("model"),
            "model_sha256": health.get("model_sha256"),
            "tokenizer_sha256": health.get("tokenizer_sha256"),
            "backend": health.get("backend"),
            "device": health.get("device"),
            "gpu_layers": health.get("gpu_layers"),
            "n_ctx": health.get("n_ctx"),
            "n_batch": health.get("n_batch"),
            "n_ubatch": health.get("n_ubatch"),
            "vocab_size": health.get("vocab_size"),
            "llama_cpp_commit": health.get("llama_cpp_commit"),
        },
        "overall": {
            "comparison_count": len(gated_errors),
            "max_abs_token_logprob_error": overall_max,
            "mean_abs_token_logprob_error": statistics.fmean(gated_errors),
            "p95_abs_token_logprob_error": _percentile(gated_errors, 0.95),
            "reported_comparison_count_including_non_gate_cases": len(all_errors),
            "non_gate_case_count": sum(not case.parity_gate for case in cases),
            "full_vocab_host_reduction_case_count": sum(
                item["full_vocab_host_reduction"] is True for item in reports
            ),
            "target_logprob_reductions": sorted({
                str(item["target_logprob_reduction"])
                for item in reports if item["target_logprob_reduction"] is not None
            }),
            "reported_logprob_host_output_floats": sum(
                int(item["logprob_host_output_floats"])
                for item in reports
                if isinstance(item["logprob_host_output_floats"], int)
                and not isinstance(item["logprob_host_output_floats"], bool)
            ),
        },
        "cases": reports,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18123")
    parser.add_argument(
        "--max-abs-error",
        type=float,
        default=0.25,
        help="maximum tolerated per-token native-vs-scalar logprob drift (default: 0.25 nats)",
    )
    args = parser.parse_args()
    try:
        report = run(args.base_url, args.max_abs_error)
    except Exception as exc:  # battery output should remain machine-readable on hard failure
        print(json.dumps({
            "schema_version": "clozn.score-variants-parity.v1",
            "status": "FAIL",
            "error": str(exc),
        }, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
