#!/usr/bin/env python3
"""Reproducible serial-vs-batch Minimal Context benchmark harness.

This first benchmark intentionally uses a deterministic in-process substrate.
It measures the scheduling, cache, and certificate work without pretending
that synthetic timings are GGUF performance.  A real substrate can be
plugged into the same ``run_case`` adapter when its native batch endpoint is
available.

Examples::

    python scripts/bench/minimal_context_multi_arm.py --json-out /tmp/mc.json
    python scripts/bench/minimal_context_multi_arm.py --source-counts 10 --prompt-sizes short --answer-sizes short --json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from collections.abc import Iterable, Mapping
from typing import Any


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clozn.runs.minimal_context import (  # noqa: E402
    EXACT_PRESERVATION_KIND,
    PRESERVATION_KIND,
    PRESERVATION_TARGET,
    run_minimal_context_search,
)
from clozn.runs.multi_arm import probe_reference_match_many, score_tokens_many  # noqa: E402


PROMPT_SIZES = {"short": 128, "medium": 512, "long": 2048}
ANSWER_SIZES = {"short": 16, "medium": 64}
TARGET_RETAINED = 3


def _source_id(index: int) -> str:
    return f"src_{index:04d}"


def _removed_key(removed: Iterable[str]) -> str:
    return ",".join(removed) or "none"


class DeterministicBenchmarkSubstrate:
    """A small substrate with exact scalar and serial-native batch methods."""

    def __init__(self, source_count: int, prompt_tokens: int, answer_tokens: int):
        self.source_count = source_count
        self.prompt_tokens = prompt_tokens
        self.answer_tokens = answer_tokens
        self.score_calls = 0
        self.probe_calls = 0

    def _meta(self, block: Any) -> dict[str, Any]:
        try:
            value = json.loads(str(block or "{}"))
        except (TypeError, ValueError):
            value = {}
        return value if isinstance(value, dict) else {}

    def score_tokens(self, messages, continuation_ids, *, block=None, steer_strengths=None,
                     steer_vec=None, topk=0, continuation=None):
        self.score_calls += 1
        meta = self._meta(block)
        retained = int(meta.get("retained", 0))
        penalty = 0.0 if retained >= TARGET_RETAINED else 1.0
        count = len(continuation_ids or []) or self.answer_tokens
        return [{"id": index, "piece": "x", "logprob": -penalty / count}
                for index in range(count)]

    def score_tokens_many(self, arms, *, cancel=None):
        return [self.score_tokens(**arm) for arm in arms]

    def probe_reference_match(self, messages, reference_token_ids, *, generation_contract,
                              explicit_conditions=None):
        self.probe_calls += 1
        conditions = dict(explicit_conditions or {})
        meta = self._meta(conditions.get("block"))
        retained = int(meta.get("retained", 0))
        if retained >= TARGET_RETAINED:
            return {
                "status": "matched",
                "matched_token_count": len(reference_token_ids),
                "first_divergence_index": None,
                "divergence_kind": None,
                "termination_match": True,
            }
        divergence = min(len(reference_token_ids), max(1, retained + 1))
        return {
            "status": "diverged",
            "matched_token_count": divergence,
            "first_divergence_index": divergence,
            "divergence_kind": "token_mismatch",
            "termination_match": True,
        }

    def probe_reference_match_many(self, arms, *, cancel=None):
        return [self.probe_reference_match(**arm) for arm in arms]


def _arm_messages(prompt_tokens: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": "p" * max(1, prompt_tokens)}]


def _arm_block(removed: tuple[str, ...], source_count: int) -> str:
    return json.dumps({"removed": list(removed), "retained": source_count - len(removed)}, sort_keys=True)


def _probe_document(
    removed: tuple[str, ...], raw: Mapping[str, Any], *, run_id: str, mode: str, elapsed_ms: float,
) -> dict[str, Any]:
    digest = hashlib.sha256(f"{run_id}\0{mode}\0{_removed_key(removed)}".encode()).hexdigest()[:24]
    if mode == "exact":
        return {
            "schema_version": "clozn.reference-match-probe.v1",
            "probe_id": f"rmp_{digest}",
            "run_id": run_id,
            "removed_source_ids": list(removed),
            "result": dict(raw),
            "provenance": "direct_generation_probe",
            "elapsed_ms": elapsed_ms,
        }
    return {
        "experiment_id": f"exp_{digest}",
        "removed_source_ids": list(removed),
        "delta_nats": float(raw["delta_nats"]),
        "provenance": "measured",
        "elapsed_ms": elapsed_ms,
    }


def run_case(
    *, source_count: int, prompt_size: str, answer_size: str, mode: str, execution: str,
    search_seed: int = 0,
) -> dict[str, Any]:
    prompt_tokens = PROMPT_SIZES[prompt_size]
    answer_tokens = ANSWER_SIZES[answer_size]
    source_ids = tuple(_source_id(index) for index in range(source_count))
    messages = _arm_messages(prompt_tokens)
    continuation_ids = list(range(answer_tokens))
    substrate = DeterministicBenchmarkSubstrate(source_count, prompt_tokens, answer_tokens)
    # Keep the recorded-run identity independent of the scheduling mode so a
    # serial-vs-batch comparison can prove that it produced the same evidence
    # identity, not merely the same headline certificate.
    run_id = f"bench_{source_count}_{prompt_size}_{answer_size}_{mode}"
    started = time.perf_counter()
    phase_times: dict[str, float] = {}
    first_preserving: float | None = None
    divergence_tokens: list[int] = []
    prefill_ms: list[float] = []
    probe_count = 0

    def phase_callback(name: str, completed: int, total: int) -> None:
        phase_times.setdefault(name, time.perf_counter() - started)

    def make_arm(removed: tuple[str, ...]) -> dict[str, Any]:
        block = _arm_block(removed, source_count)
        if mode == "exact":
            return {
                "messages": messages,
                "reference_token_ids": continuation_ids,
                "generation_contract": {
                    "decode_mode": "greedy",
                    "max_new": answer_tokens,
                    "stop": [],
                    "expected_termination": {"reason": "stop", "reason_raw": "eos"},
                },
                "explicit_conditions": {"block": block, "steer_strengths": {}},
            }
        return {
            "messages": messages,
            "continuation_ids": continuation_ids,
            "block": block,
            "steer_strengths": {},
        }

    def one_measure(removed: tuple[str, ...]) -> dict[str, Any]:
        nonlocal first_preserving, probe_count
        checkpoint = time.perf_counter()
        if mode == "exact":
            raw = substrate.probe_reference_match(**make_arm(removed))
            if raw.get("status") == "matched" and first_preserving is None:
                first_preserving = time.perf_counter() - started
            if raw.get("status") == "diverged" and isinstance(raw.get("first_divergence_index"), int):
                divergence_tokens.append(raw["first_divergence_index"])
        else:
            tokens = substrate.score_tokens(**make_arm(removed))
            intervened = sum(float(item["logprob"]) for item in tokens)
            raw = {"delta_nats": -intervened, "status": "measured"}
            if abs(raw["delta_nats"]) <= 0.01 and first_preserving is None:
                first_preserving = time.perf_counter() - started
        probe_count += 1
        prefill_ms.append((time.perf_counter() - checkpoint) * 1000.0)
        return _probe_document(removed, raw, run_id=run_id, mode=mode, elapsed_ms=prefill_ms[-1])

    def many_measure(removed_sets: tuple[tuple[str, ...], ...]) -> list[dict[str, Any]]:
        nonlocal first_preserving, probe_count
        if mode == "exact":
            arms = [make_arm(removed) for removed in removed_sets]
            checkpoint = time.perf_counter()
            raw_results = probe_reference_match_many(substrate, arms)
        else:
            arms = [make_arm(removed) for removed in removed_sets]
            checkpoint = time.perf_counter()
            raw_results = score_tokens_many(substrate, arms)
        per_arm_ms = (time.perf_counter() - checkpoint) * 1000.0 / max(1, len(removed_sets))
        rows = []
        for removed, raw in zip(removed_sets, raw_results):
            if mode == "exact":
                if raw.get("status") == "matched" and first_preserving is None:
                    first_preserving = time.perf_counter() - started
                if raw.get("status") == "diverged" and isinstance(raw.get("first_divergence_index"), int):
                    divergence_tokens.append(raw["first_divergence_index"])
            else:
                intervened = sum(float(item["logprob"]) for item in raw)
                raw = {"delta_nats": -intervened, "status": "measured"}
                if abs(raw["delta_nats"]) <= 0.01 and first_preserving is None:
                    first_preserving = time.perf_counter() - started
            probe_count += 1
            prefill_ms.append(per_arm_ms)
            rows.append(_probe_document(removed, raw, run_id=run_id, mode=mode, elapsed_ms=per_arm_ms))
        return rows

    preservation = (
        {"kind": EXACT_PRESERVATION_KIND, "target": PRESERVATION_TARGET}
        if mode == "exact" else
        {"kind": PRESERVATION_KIND, "target": PRESERVATION_TARGET, "tolerance_nats": 0.01}
    )
    search_budget = max(1, source_count * 3)
    lower_candidates = sum(math.comb(source_count, cardinality) for cardinality in range(TARGET_RETAINED))
    certification_budget = lower_candidates + TARGET_RETAINED + 8
    kwargs: dict[str, Any] = {}
    if execution == "batch":
        kwargs["measure_removed_many"] = many_measure
        scalar = None
    else:
        scalar = one_measure
    result = run_minimal_context_search(
        source_ids, scalar,
        tolerance_nats=0.01,
        search_probe_budget=search_budget,
        certification_probe_budget=certification_budget,
        search_seed=search_seed,
        run_id=run_id,
        preservation=preservation,
        phase_callback=phase_callback,
        **kwargs,
    )
    wall_s = time.perf_counter() - started
    probes = int(result["budget"]["total_new_probes"])
    reused = int(result["budget"]["reused_experiments"])
    total_evidence_accesses = probes + reused
    return {
        "model_identity": "deterministic.minimal-context-benchmark.v1",
        "engine_build": os.environ.get("CLOZN_ENGINE_BUILD_ID", "serial-fallback"),
        "hardware_backend": f"{platform.system()}:{platform.machine()}:python",
        "execution": execution,
        "preservation": mode,
        "n_ctx": source_count,
        "prompt_size": prompt_size,
        "prompt_tokens": prompt_tokens,
        "answer_size": answer_size,
        "answer_tokens": answer_tokens,
        "source_count": source_count,
        "probe_count": probes,
        "search_probe_count": int(result["budget"].get("search_new_probes", 0)),
        "certification_probe_count": int(result["budget"].get("certification_new_probes", 0)),
        "lower_cardinality_candidate_count": int(
            result.get("search", {}).get("certification_lower_cardinality_candidate_count", 0)
        ),
        "wall_time_s": wall_s,
        "arms_per_sec": probes / wall_s if wall_s else None,
        "prompt_tokens_per_sec": (probes * prompt_tokens) / wall_s if wall_s else None,
        "median_prompt_prefill_ms": statistics.median(prefill_ms) if prefill_ms else None,
        "median_divergence_token": statistics.median(divergence_tokens) if divergence_tokens else None,
        "reference_tokens_before_divergence": (
            statistics.median(divergence_tokens) if divergence_tokens else answer_tokens
        ),
        "time_to_first_preserving_candidate_s": first_preserving,
        "time_to_inclusion_minimal_s": phase_times.get("inclusion_minimal"),
        "time_to_exact_certificate_s": phase_times.get("exact_certificate"),
        "cache_reuse_count": reused,
        "cache_reuse_rate": reused / total_evidence_accesses if total_evidence_accesses else 0.0,
        "status": result["status"],
        "certificate_kind": (result.get("certificate") or {}).get("kind"),
        "candidate_retained_source_count": (result.get("candidate") or {}).get("retained_source_count"),
        "search_seed": search_seed,
        "result_id": result["result_id"],
        "substrate_calls": {"score": substrate.score_calls, "probe": substrate.probe_calls},
    }


def run_benchmark(*, source_counts: Iterable[int] = (10, 25, 50, 100),
                  prompt_sizes: Iterable[str] = tuple(PROMPT_SIZES),
                  answer_sizes: Iterable[str] = tuple(ANSWER_SIZES),
                  modes: Iterable[str] = ("exact", "likelihood"),
                  executions: Iterable[str] = ("serial", "batch"),
                  search_seed: int = 0) -> dict[str, Any]:
    source_counts = list(source_counts)
    prompt_sizes = list(prompt_sizes)
    answer_sizes = list(answer_sizes)
    modes = list(modes)
    executions = list(executions)
    rows = []
    for source_count in source_counts:
        for prompt_size in prompt_sizes:
            for answer_size in answer_sizes:
                for mode in modes:
                    for execution in executions:
                        rows.append(run_case(
                            source_count=int(source_count), prompt_size=prompt_size,
                            answer_size=answer_size, mode=mode, execution=execution,
                            search_seed=search_seed,
                        ))
    return {
        "schema_version": "clozn.minimal-context-benchmark.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "matrix": {
            "source_counts": list(source_counts),
            "prompt_sizes": list(prompt_sizes),
            "answer_sizes": list(answer_sizes),
            "preservation_modes": list(modes),
            "execution_modes": list(executions),
        },
        "rows": rows,
    }


def _summary(report: Mapping[str, Any]) -> str:
    rows = report.get("rows") or []
    lines = [f"Minimal Context multi-arm benchmark: {len(rows)} cases"]
    by_case: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["source_count"], row["prompt_size"], row["answer_size"], row["preservation"])
        by_case.setdefault(key, {})[row["execution"]] = row
    for key, modes in sorted(by_case.items(), key=lambda item: item[0]):
        serial = modes.get("serial")
        batch = modes.get("batch")
        ratio = None
        if serial and batch and batch.get("wall_time_s"):
            ratio = serial["wall_time_s"] / batch["wall_time_s"]
        suffix = f", serial/batch wall ratio {ratio:.3f}" if ratio is not None else ""
        lines.append(
            f"  n={key[0]:>3} {key[1]:>6}/{key[2]:>6} {key[3]:>9}: "
            f"serial arms/s={float(serial['arms_per_sec']):.1f}" + suffix
            if serial else f"  n={key[0]} {key[1]}/{key[2]} {key[3]}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-counts", nargs="+", type=int, default=[10, 25, 50, 100])
    parser.add_argument("--prompt-sizes", nargs="+", choices=tuple(PROMPT_SIZES), default=list(PROMPT_SIZES))
    parser.add_argument("--answer-sizes", nargs="+", choices=tuple(ANSWER_SIZES), default=list(ANSWER_SIZES))
    parser.add_argument("--modes", nargs="+", choices=("exact", "likelihood"), default=["exact", "likelihood"])
    parser.add_argument("--executions", nargs="+", choices=("serial", "batch"), default=["serial", "batch"])
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--json", action="store_true", help="print only machine-readable JSON")
    args = parser.parse_args(argv)
    report = run_benchmark(
        source_counts=args.source_counts, prompt_sizes=args.prompt_sizes,
        answer_sizes=args.answer_sizes, modes=args.modes, executions=args.executions,
        search_seed=args.search_seed,
    )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        print(_summary(report))
        if args.json_out:
            print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
