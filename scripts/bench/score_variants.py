#!/usr/bin/env python3
"""Benchmark native related-prompt scoring against independent scalar scoring.

This requires an already-running autoregressive Clozn worker.  It deliberately
keeps numerical qualification separate from throughput: native rows are timed
as screening results and compared with scalar ``POST /score`` rows, never
promoted to trusted evidence by this script.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "engine" / "client"
for path in (ROOT, CLIENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def _timing(values: list[float]) -> dict[str, float]:
    return {
        "samples": len(values),
        "min_seconds": min(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": _percentile(values, 0.95),
        "max_seconds": max(values),
        "mean_seconds": statistics.fmean(values),
    }


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = call()
    return result, time.perf_counter() - started


def _replace(prompt: list[int], position: int, value: int) -> list[int]:
    out = list(prompt)
    out[position] = value
    return out


def _variants(prompt: list[int], count: int, pattern: str) -> list[dict[str, Any]]:
    if count < 1 or count > 8:
        raise ValueError("variant count must be in [1, 8]")
    if len(prompt) < 8:
        raise ValueError("benchmark prompt is unexpectedly short")
    position = {
        "early": 1,
        "middle": len(prompt) // 2,
        "late": len(prompt) - 2,
    }.get(pattern)
    rows = []
    for index in range(count):
        if pattern == "duplicate":
            ids = list(prompt)
        elif pattern == "no-overlap":
            ids = _replace(prompt, 0, 3000 + index)
        elif position is not None:
            ids = list(prompt) if index == 0 else _replace(prompt, position, 3000 + index)
        else:
            raise ValueError(f"unsupported pattern: {pattern}")
        rows.append({"id": f"{pattern}-{count}-{index}", "prompt_ids": ids})
    return rows


def _cases(prompt: list[int]) -> list[tuple[str, list[dict[str, Any]]]]:
    cases = [
        (f"late-{count}", _variants(prompt, count, "late"))
        for count in (1, 2, 4, 8)
    ]
    cases.extend((pattern, _variants(prompt, 8, pattern)) for pattern in (
        "early", "middle", "no-overlap", "duplicate",
    ))
    return cases


def _native_scores(response: dict[str, Any], expected_ids: list[str], n_cont: int) -> list[list[float]]:
    if response.get("status") != "completed":
        raise RuntimeError(f"native scoring did not complete: {response.get('status')!r}")
    rows = response.get("results")
    if not isinstance(rows, list) or [row.get("id") for row in rows] != expected_ids:
        raise RuntimeError("native scoring did not preserve variant order")
    scores = [row.get("target_logprobs") for row in rows]
    if any(not isinstance(row, list) or len(row) != n_cont for row in scores):
        raise RuntimeError("native scoring returned a malformed score matrix")
    return [[float(value) for value in row] for row in scores]


def _scalar_scores(engine: Any, variants: list[dict[str, Any]], continuation_ids: list[int]) -> list[list[float]]:
    rows = []
    for variant in variants:
        response = engine.score(
            prompt_ids=variant["prompt_ids"], continuation_ids=continuation_ids, topk=0)
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or [token.get("id") for token in tokens] != continuation_ids:
            raise RuntimeError("scalar scoring returned a malformed or misaligned row")
        rows.append([float(token["logprob"]) for token in tokens])
    return rows


def _max_abs_error(left: list[list[float]], right: list[list[float]]) -> float:
    if len(left) != len(right):
        raise RuntimeError("native/scalar result row count mismatch")
    errors = [
        abs(native - scalar)
        for native_row, scalar_row in zip(left, right)
        for native, scalar in zip(native_row, scalar_row)
    ]
    return max(errors, default=0.0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups must be non-negative")
    try:
        from clozn_engine import EngineClient
    except ImportError as exc:
        raise RuntimeError(
            f"could not import the repository EngineClient ({exc}); use the repository .venv"
        ) from exc

    engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
    health = engine.health()
    if not (health.get("capabilities") or {}).get("score_variants"):
        raise RuntimeError("worker does not advertise capabilities.score_variants")
    rendered = engine.apply_template_info([{
        "role": "user",
        "content": "Explain why exact shared prefixes reduce repeated inference work.",
    }], include_token_ids=True)
    prompt_ids = rendered.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise RuntimeError("worker did not return exact rendered prompt token ids")
    seed = engine.score(
        prompt_ids=prompt_ids,
        continuation="Shared prefixes let the engine reuse prior computation.",
        topk=0,
    )
    continuation_ids = [token["id"] for token in seed.get("tokens", [])]
    if not continuation_ids:
        raise RuntimeError("continuation fixture tokenized to an empty sequence")

    reports = []
    for case_name, variants in _cases(prompt_ids):
        expected_ids = [variant["id"] for variant in variants]
        native_call = lambda: engine.score_variants(variants, continuation_ids)
        scalar_call = lambda: _scalar_scores(engine, variants, continuation_ids)
        for _ in range(args.warmups):
            native_call()
            scalar_call()

        native_times: list[float] = []
        scalar_times: list[float] = []
        worker_times: list[float] = []
        reference_native: list[list[float]] | None = None
        reference_scalar: list[list[float]] | None = None
        last_native: dict[str, Any] | None = None
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic cache/thermal bias.
            if repeat % 2 == 0:
                native_response, native_elapsed = _timed(native_call)
                scalar_response, scalar_elapsed = _timed(scalar_call)
            else:
                scalar_response, scalar_elapsed = _timed(scalar_call)
                native_response, native_elapsed = _timed(native_call)
            native_scores = _native_scores(native_response, expected_ids, len(continuation_ids))
            if reference_native is None:
                reference_native = native_scores
                reference_scalar = scalar_response
            native_times.append(native_elapsed)
            scalar_times.append(scalar_elapsed)
            worker_wall = (native_response.get("metrics") or {}).get("wall_seconds")
            if isinstance(worker_wall, (int, float)) and not isinstance(worker_wall, bool):
                worker_times.append(float(worker_wall))
            last_native = native_response

        assert reference_native is not None and reference_scalar is not None and last_native is not None
        native_median = statistics.median(native_times)
        scalar_median = statistics.median(scalar_times)
        metrics = last_native.get("metrics") or {}
        execution = last_native.get("execution") or {}
        reports.append({
            "case": case_name,
            "variant_count": len(variants),
            "prompt_tokens_per_variant": [len(row["prompt_ids"]) for row in variants],
            "continuation_tokens": len(continuation_ids),
            "native_client_wall": _timing(native_times),
            "native_worker_wall": _timing(worker_times) if worker_times else {"status": "unavailable"},
            "scalar_client_wall": _timing(scalar_times),
            "median_speedup_scalar_over_native": scalar_median / native_median,
            "max_abs_token_logprob_error": _max_abs_error(reference_native, reference_scalar),
            "planned_logical_prompt_rows": metrics.get("planned_logical_prompt_rows"),
            "planned_physical_prompt_rows": metrics.get("planned_physical_prompt_rows"),
            "planned_continuation_kv_rows": metrics.get("planned_continuation_kv_rows"),
            "successful_decode_rows": metrics.get("successful_decode_rows"),
            "successful_decode_calls": metrics.get("successful_decode_calls"),
            "wave_count": metrics.get("wave_count"),
            "full_vocab_host_reduction": execution.get("full_vocab_host_reduction"),
            "target_logprob_reduction": execution.get("target_logprob_reduction"),
            "logprob_host_output_floats": metrics.get("logprob_host_output_floats"),
        })

    return {
        "schema_version": "clozn.score-variants-benchmark.v1",
        "status": "ok",
        "measurement": {
            "phase": "warm_live_worker",
            "cold_model_load_included": False,
            "warmups_per_case": args.warmups,
            "measured_repeats_per_case": args.repeats,
            "timing_clock": "time.perf_counter",
            "client_wall_includes_http_and_json": True,
        },
        "worker": {
            key: health.get(key) for key in (
                "model", "model_sha256", "tokenizer_sha256", "backend", "device", "gpu_layers",
                "n_ctx", "n_batch", "n_ubatch", "vocab_size", "llama_cpp_commit",
            )
        },
        "cases": reports,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--host", default="127.0.0.1")
    result.add_argument("--port", type=int, default=8765)
    result.add_argument("--timeout", type=float, default=120.0)
    result.add_argument("--warmups", type=int, default=1)
    result.add_argument("--repeats", type=int, default=5)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({
            "schema_version": "clozn.score-variants-benchmark.v1",
            "status": "error",
            "error": str(exc),
        }, indent=2, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
