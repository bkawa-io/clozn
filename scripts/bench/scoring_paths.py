#!/usr/bin/env python3
"""Compare the live worker's production teacher-forced scoring paths.

This opt-in harness measures the same recorded continuation under scalar cold,
bounded host-concurrent scalar, scalar shared-prefix, and (when advertised) the
experimental native exact-prefix-trie path. It also records (but does not infer)
the worker's actual compute backend, such as Metal. It never starts or stops a
worker.

The concurrent path is a host-side batch of independent scalar requests, not a
native multi-arm transformer forward pass. The shared-prefix path is the
worker's exact scalar prefix cache. Those distinctions are part of the report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
CLIENT_ROOT = ROOT / "engine" / "client"
if str(CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIENT_ROOT))

from clozn_engine import EngineClient  # noqa: E402


def _messages(common_paragraphs: int, arm_count: int) -> list[list[dict[str, str]]]:
    common = "\n\n".join(
        f"Section {index}. The project rule is stable, explicit, and recorded for later review. "
        "Keep this sentence in its original order when comparing scoring paths."
        for index in range(1, common_paragraphs + 1)
    )
    return [[
        {"role": "system", "content": f"# Shared benchmark context\n\n{common}\n\nArm marker: {index}."},
        {"role": "user", "content": "Summarize the governing project rule in one sentence."},
    ] for index in range(arm_count)]


def _cache_totals(responses: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "request_count": len(responses),
        "cache_hit_count": 0,
        "matched_tokens": 0,
        "cached_tokens": 0,
        "evaluated_tokens": 0,
    }
    for response in responses:
        cache = response.get("cache")
        if not isinstance(cache, dict):
            raise RuntimeError("worker score response omitted cache metrics")
        for field in ("matched_tokens", "cached_tokens", "evaluated_tokens"):
            value = cache.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RuntimeError(f"worker score response has invalid cache.{field}")
            totals[field] += value
        totals["cache_hit_count"] += cache["cached_tokens"] > 0
    return totals


def _token_signature(response: dict[str, Any]) -> tuple[tuple[int, float], ...]:
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise RuntimeError("worker score response omitted token scores")
    signature = []
    for token in tokens:
        if not isinstance(token, dict):
            raise RuntimeError("worker score response contains an invalid token row")
        token_id = token.get("id")
        logprob = token.get("logprob")
        if (not isinstance(token_id, int) or isinstance(token_id, bool)
                or not isinstance(logprob, (int, float)) or isinstance(logprob, bool)):
            raise RuntimeError("worker score response contains an invalid token score")
        signature.append((token_id, float(logprob)))
    return tuple(signature)


def _signature_delta(
    expected: list[tuple[tuple[int, float], ...]],
    actual: list[tuple[tuple[int, float], ...]],
) -> dict[str, Any]:
    ids_equal = (
        len(expected) == len(actual)
        and all(
            [token_id for token_id, _ in left] == [token_id for token_id, _ in right]
            for left, right in zip(expected, actual)
        )
    )
    if not ids_equal:
        return {"token_ids_equal": False, "logprobs_exact": False, "max_abs_logprob_delta": None}
    deltas = [
        abs(left_logprob - right_logprob)
        for left, right in zip(expected, actual)
        for (_, left_logprob), (_, right_logprob) in zip(left, right)
    ]
    relative_deltas: list[float] = []
    if expected and actual:
        expected_control = expected[0]
        actual_control = actual[0]
        for expected_arm, actual_arm in zip(expected[1:], actual[1:]):
            for ((_, expected_value), (_, actual_value),
                 (_, expected_control_value), (_, actual_control_value)) in zip(
                    expected_arm, actual_arm, expected_control, actual_control,
                 ):
                relative_deltas.append(abs(
                    (actual_value - actual_control_value)
                    - (expected_value - expected_control_value)
                ))
    return {
        "token_ids_equal": True,
        "logprobs_exact": all(delta == 0.0 for delta in deltas),
        "max_abs_logprob_delta": max(deltas, default=0.0),
        "max_abs_arm_effect_delta": max(relative_deltas, default=0.0),
    }


def _measure(
    prompts: list[str],
    continuation_ids: list[int],
    score: Callable[[str, list[int]], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    responses = [score(prompt, continuation_ids) for prompt in prompts]
    elapsed = time.perf_counter() - started
    cache = _cache_totals(responses)
    return ({
        "elapsed_seconds": elapsed,
        "requests_per_second": len(responses) / elapsed if elapsed else None,
        "implementation": "scalar_http",
        "cache": cache,
        "numerical_regimes": sorted({
            str(response.get("numerical_regime") or "unknown") for response in responses
        }),
    }, responses)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.arms < 2:
        raise ValueError("--arms must be at least 2")
    if args.workers < 2:
        raise ValueError("--workers must be at least 2")
    if args.common_paragraphs < 1:
        raise ValueError("--common-paragraphs must be positive")

    engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
    health = engine.health()
    capabilities = health.get("capabilities")
    if not isinstance(capabilities, dict) or capabilities.get("score_prefix_cache") is not True:
        raise RuntimeError("worker does not advertise score_prefix_cache")

    rendered = [engine.apply_template_info(messages) for messages in _messages(
        args.common_paragraphs, args.arms,
    )]
    prompts = [str(item["prompt"]) for item in rendered]
    prompt_tokens = [int(item["prompt_tokens"]) for item in rendered]

    # Tokenize the continuation once outside all timed paths. Later requests use
    # these exact IDs, avoiding text-boundary drift between benchmark regimes.
    bootstrap = engine.score(
        prompt=prompts[0], continuation=args.continuation, use_prefix_cache=False,
    )
    continuation_ids = [token_id for token_id, _ in _token_signature(bootstrap)]

    cold_score = lambda prompt, ids: engine.score(  # noqa: E731
        prompt=prompt, continuation_ids=ids, use_prefix_cache=False,
    )
    serial, serial_responses = _measure(prompts, continuation_ids, cold_score)
    serial["name"] = "scalar_serial_cold"
    serial["dispatch"] = "one request after the previous request completes"

    concurrent_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        concurrent_responses = list(pool.map(
            lambda prompt: cold_score(prompt, continuation_ids), prompts,
        ))
    concurrent_elapsed = time.perf_counter() - concurrent_started
    concurrent = {
        "name": "host_concurrent_scalar_cold",
        "elapsed_seconds": concurrent_elapsed,
        "requests_per_second": len(prompts) / concurrent_elapsed if concurrent_elapsed else None,
        "implementation": "bounded_host_concurrency_over_scalar_http",
        "dispatch": f"up to {args.workers} independent scalar requests in flight",
        "workers": args.workers,
        "cache": _cache_totals(concurrent_responses),
        "numerical_regimes": sorted({
            str(response.get("numerical_regime") or "unknown")
            for response in concurrent_responses
        }),
    }

    cached_score = lambda prompt, ids: engine.score(  # noqa: E731
        prompt=prompt, continuation_ids=ids, use_prefix_cache=True,
    )
    shared, shared_responses = _measure(prompts, continuation_ids, cached_score)
    shared["name"] = "scalar_serial_shared_prefix"
    shared["implementation"] = "scalar_http_with_exact_worker_prefix_cache"
    shared["dispatch"] = "serial requests ordered to share a long leading token prefix"

    serial_signatures = [_token_signature(item) for item in serial_responses]
    concurrent_signatures = [_token_signature(item) for item in concurrent_responses]
    shared_signatures = [_token_signature(item) for item in shared_responses]
    parity = {
        "concurrent_equals_cold_scalar": concurrent_signatures == serial_signatures,
        "shared_prefix_equals_cold_scalar": shared_signatures == serial_signatures,
    }

    native = None
    native_host = None
    native_parity = None
    native_host_parity = None
    if isinstance(capabilities, dict) and capabilities.get("teacher_forced_score_many") is True:
        native_started = time.perf_counter()
        native_response = engine.teacher_forced_score_many(
            [{"prompt": prompt} for prompt in prompts],
            continuation_ids=continuation_ids,
        )
        native_elapsed = time.perf_counter() - native_started
        native_results = native_response["results"]
        native_signatures = [_token_signature(item) for item in native_results]
        native_parity = _signature_delta(serial_signatures, native_signatures)
        native_parity["within_abs_tolerance_1e_5"] = bool(
            native_parity["token_ids_equal"]
            and native_parity["max_abs_logprob_delta"] is not None
            and native_parity["max_abs_logprob_delta"] <= 1e-5
        )
        parity["native_within_abs_tolerance_1e_5"] = native_parity[
            "within_abs_tolerance_1e_5"
        ]
        native = {
            "name": "native_prefix_trie_teacher_forced",
            "elapsed_seconds": native_elapsed,
            "requests_per_second": len(prompts) / native_elapsed if native_elapsed else None,
            "implementation": "one_native_multi_sequence_http_request",
            "numerical_regime": native_response.get("numerical_regime"),
            "worker_qualified": native_response.get("qualified"),
            "execution": native_response.get("execution"),
        }

        native_host_started = time.perf_counter()
        native_host_response = engine.teacher_forced_score_many(
            [{"prompt": prompt} for prompt in prompts],
            continuation_ids=continuation_ids,
            reduction="host_full_vocab",
        )
        native_host_elapsed = time.perf_counter() - native_host_started
        native_host_results = native_host_response["results"]
        native_host_signatures = [_token_signature(item) for item in native_host_results]
        native_host_parity = _signature_delta(serial_signatures, native_host_signatures)
        native_host_parity["within_abs_tolerance_1e_5"] = bool(
            native_host_parity["token_ids_equal"]
            and native_host_parity["max_abs_logprob_delta"] is not None
            and native_host_parity["max_abs_logprob_delta"] <= 1e-5
        )
        parity["native_host_within_abs_tolerance_1e_5"] = native_host_parity[
            "within_abs_tolerance_1e_5"
        ]
        native_host = {
            "name": "native_prefix_trie_host_float32",
            "elapsed_seconds": native_host_elapsed,
            "requests_per_second": len(prompts) / native_host_elapsed if native_host_elapsed else None,
            "implementation": "native_multi_sequence_with_full_vocab_host_reduction",
            "numerical_regime": native_host_response.get("numerical_regime"),
            "worker_qualified": native_host_response.get("qualified"),
            "execution": native_host_response.get("execution"),
        }

    return {
        "status": "ok" if all(parity.values()) else "parity_failed",
        "worker": {
            key: health.get(key) for key in (
                "model", "model_sha256", "backend", "device", "gpu_layers",
                "engine_version", "build_id", "llama_cpp_commit", "n_ctx", "n_batch", "n_ubatch",
            )
        },
        "workload": {
            "arm_count": len(prompts),
            "prompt_tokens_min": min(prompt_tokens),
            "prompt_tokens_max": max(prompt_tokens),
            "continuation_tokens": len(continuation_ids),
            "fixture": "shared_prefix_suffix_variants",
        },
        "paths": {
            "scalar": serial,
            "batched_dispatch": concurrent,
            "shared_prefix": shared,
            "native_prefix_trie": native,
            "native_prefix_trie_host_reference": native_host,
            "metal": {
                "observed": str(health.get("backend") or health.get("device") or "").lower() == "metal",
                "backend": health.get("backend"),
                "device": health.get("device"),
                "gpu_layers": health.get("gpu_layers"),
                "scope": "all three measured score paths above use this worker backend",
                "cpu_comparison": "not_run",
            },
        },
        "parity": parity,
        "native_parity": native_parity,
        "native_host_parity": native_host_parity,
        "interpretation": {
            "batched_dispatch_is_native_multi_arm": False,
            "native_path_is_production_enabled": False,
            "shared_prefix_fixture_is_best_case": True,
            "timing_includes": ["worker queueing", "model execution", "HTTP", "JSON serialization"],
            "timing_excludes": ["chat generation", "Studio rendering", "analysis persistence"],
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--host", default="127.0.0.1")
    value.add_argument("--port", type=int, default=8765)
    value.add_argument("--timeout", type=float, default=600.0)
    value.add_argument("--arms", type=int, default=4)
    value.add_argument("--workers", type=int, default=2)
    value.add_argument("--common-paragraphs", type=int, default=24)
    value.add_argument(
        "--continuation",
        default="The governing rule is stable, explicit, and recorded for later review.",
    )
    value.add_argument("--output", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        result = run(args)
        payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists():
                raise ValueError(f"refusing to overwrite existing output: {args.output}")
            args.output.write_text(payload, encoding="utf-8")
        else:
            print(payload, end="")
        return 0 if result["status"] == "ok" else 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
