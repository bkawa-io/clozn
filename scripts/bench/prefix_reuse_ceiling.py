#!/usr/bin/env python3
"""Measure exact-token prefix overlap without running model inference.

The worker is used only as the source of truth for chat-template rendering and tokenization.  The
script never acquires a context lease, calls llama_decode, generates a completion, or invokes any
reference-match execution strategy.  JSON is written to stdout; progress is written to stderr.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "engine" / "client", BENCH_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from native_reference_match import (  # noqa: E402
    _arms,
    _require_supported_python,
    build_fixture,
    certification_sets,
)


def progress(message: str, started: float, enabled: bool) -> None:
    if enabled:
        elapsed = time.perf_counter() - started
        print(f"[prefix-reuse-ceiling +{elapsed:7.1f}s] {message}",
              file=sys.stderr, flush=True)


def exact_trie_rows(prompts: list[tuple[int, ...]]) -> int:
    """Count distinct exact-token prefix edges in a sorted-index trie."""
    ordered = sorted(prompts)
    if not ordered:
        return 0

    rows = len(ordered[0])
    previous = ordered[0]
    for current in ordered[1:]:
        lcp = 0
        for left, right in zip(previous, current):
            if left != right:
                break
            lcp += 1
        rows += len(current) - lcp
        previous = current
    return rows


def _percent(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _pack_resident(prompts: list[tuple[int, ...]], n_ctx: int) -> list[list[tuple[int, ...]]]:
    """Reproduce the route's input-order <=16-arm and sum-length<=n_ctx packing."""
    batches: list[list[tuple[int, ...]]] = []
    current: list[tuple[int, ...]] = []
    current_rows = 0
    for prompt in prompts:
        if len(prompt) > n_ctx:
            raise ValueError("prompt exceeds n_ctx during resident packing")
        if current and (len(current) >= 16 or current_rows + len(prompt) > n_ctx):
            batches.append(current)
            current = []
            current_rows = 0
        current.append(prompt)
        current_rows += len(prompt)
    if current:
        batches.append(current)
    return batches


def _resident_measure(prompts: list[tuple[int, ...]], n_ctx: int) -> dict:
    batches = _pack_resident(prompts, n_ctx)
    logical = sum(len(prompt) for prompt in prompts)
    physical = sum(exact_trie_rows(batch) for batch in batches)
    reused = logical - physical
    return {
        "batch_count": len(batches),
        "logical_prompt_rows": logical,
        "physical_prompt_rows": physical,
        "reused_rows": reused,
        "reuse_percent": _percent(reused, logical),
    }


def _tokenize_pairs(engine, fixture, pair_sets, started: float, progress_enabled: bool,
                    progress_every: int) -> list[tuple[int, ...]]:
    prompts: list[tuple[int, ...]] = []
    for index, retained in enumerate(pair_sets, start=1):
        messages = _arms(fixture, [retained])[0]["messages"]
        info = engine.apply_template_info(messages, include_token_ids=True)
        prompt_tokens = info.get("prompt_tokens")
        token_ids = info.get("prompt_token_ids")
        if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool):
            raise RuntimeError(f"arm {index} omitted exact prompt_tokens")
        if not isinstance(token_ids, list) or any(
                not isinstance(token, int) or isinstance(token, bool) for token in token_ids):
            raise RuntimeError(f"arm {index} omitted exact prompt_token_ids")
        if len(token_ids) != prompt_tokens:
            raise RuntimeError(
                f"arm {index} token-count mismatch: ids={len(token_ids)} prompt_tokens={prompt_tokens}")
        prompts.append(tuple(token_ids))
        if index == 1 or index == len(pair_sets) or index % progress_every == 0:
            progress(f"tokenized {index}/{len(pair_sets)} pair arms", started, progress_enabled)
    return prompts


def run(args: argparse.Namespace) -> dict:
    _require_supported_python()
    started = time.perf_counter()
    progress_enabled = not args.quiet
    if args.n_ctx < 1:
        raise ValueError("--n-ctx must be positive")

    fixture = build_fixture()
    workload = certification_sets()[51:]
    if len(workload) != 1225:
        raise RuntimeError(f"expected 1,225 pair arms, got {len(workload)}")
    progress(f"prepared exact certification slice: {len(workload)} pair arms",
             started, progress_enabled)

    try:
        from clozn_engine import EngineClient
    except ImportError as exc:
        raise RuntimeError("could not import EngineClient; use the repository .venv") from exc

    progress(f"connecting to worker {args.host}:{args.port}", started, progress_enabled)
    engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
    health = engine.health()
    if not health.get("capabilities", {}).get("reference_match_arms"):
        raise RuntimeError("worker does not advertise capabilities.reference_match_arms")
    worker_n_ctx = health.get("n_ctx")
    if worker_n_ctx != args.n_ctx:
        progress(f"warning: CLI n_ctx={args.n_ctx} differs from worker n_ctx={worker_n_ctx}",
                 started, progress_enabled)

    prompts = _tokenize_pairs(
        engine, fixture, workload, started, progress_enabled, max(1, args.progress_every))
    logical = sum(len(prompt) for prompt in prompts)
    global_unique = exact_trie_rows(prompts)
    global_reusable = logical - global_unique
    current = _resident_measure(prompts, args.n_ctx)
    sorted_order = _resident_measure(sorted(prompts), args.n_ctx)

    validation = {}
    for count, expected_logical, expected_physical in (
        (16, 9341, 5817),
        (45, 26286, 16420),
    ):
        observed = _resident_measure(prompts[:count], args.n_ctx)
        result = {
            "expected": {
                "logical_prompt_rows": expected_logical,
                "physical_prompt_rows": expected_physical,
            },
            "observed_from_planner": {
                "logical_prompt_rows": observed["logical_prompt_rows"],
                "physical_prompt_rows": observed["physical_prompt_rows"],
            },
            "match": (observed["logical_prompt_rows"] == expected_logical and
                      observed["physical_prompt_rows"] == expected_physical),
        }
        validation[str(count)] = result
        if not result["match"]:
            progress(
                f"WARNING: {count}-arm sanity mismatch; do not interpret global result",
                started, progress_enabled)

    validation_passed = all(item["match"] for item in validation.values())
    cross_batch_gap = current["physical_prompt_rows"] - global_unique
    if cross_batch_gap < 0:
        raise RuntimeError("current resident rows fell below the global exact-prefix lower bound")
    rows_saved = current["physical_prompt_rows"] - sorted_order["physical_prompt_rows"]
    progress(
        f"analysis complete: current={current['physical_prompt_rows']} rows, "
        f"global={global_unique} rows, token-sorted={sorted_order['physical_prompt_rows']} rows",
        started, progress_enabled)

    return {
        "status": "ok",
        "model": health.get("model"),
        "n_ctx": args.n_ctx,
        "worker_n_ctx": worker_n_ctx,
        "pair_arms": len(prompts),
        "validation": validation,
        "validation_passed": validation_passed,
        "global_exact_prefix_ceiling": {
            "logical_prompt_rows": logical,
            "global_unique_prefix_rows": global_unique,
            "global_reusable_rows": global_reusable,
            "global_reuse_percent": _percent(global_reusable, logical),
            "physical_prompt_rows_lower_bound": global_unique,
            "reusable_rows_ceiling": global_reusable,
            "reuse_percent_ceiling": _percent(global_reusable, logical),
        },
        "current_resident_order": current,
        "cross_batch_gap": {
            "duplicate_prefix_rows_above_global_lower_bound": cross_batch_gap,
            "maximum_additional_rows_available": cross_batch_gap,
            "percent_of_current_physical_rows": _percent(
                cross_batch_gap, current["physical_prompt_rows"]),
        },
        "token_lexicographic_resident_order": {
            **sorted_order,
            "rows_saved_vs_current": rows_saved,
            "percent_saved_vs_current": _percent(
                rows_saved, current["physical_prompt_rows"]),
        },
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--n-ctx", type=int, default=12288)
    parser.add_argument("--progress-every", type=int, default=50,
                        help="log every N tokenized arms")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress progress logs (JSON output remains on stdout)")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        print(json.dumps(run(parser().parse_args(argv)), ensure_ascii=False,
                          sort_keys=True, indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
