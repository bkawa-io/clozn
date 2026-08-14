#!/usr/bin/env python3
"""Benchmark scalar versus bounded concurrent Minimal Context arms on a live engine.

Unlike ``minimal_context_multi_arm.py``, this harness loads a recorded run from
the local run store and reconstructs strict source-deletion prompts through the
production Context Receipt resolver.  It is opt-in because it requires a live
GGUF worker and never downloads a model.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time


def _arms(run: dict, count: int) -> list[list[str]]:
    from clozn.receipts.context_dependence import ContextDependenceStudy

    # The study constructor is model-free and owns strict source resolution.
    # A placeholder substrate is sufficient for discovering canonical IDs.
    study = ContextDependenceStudy(run, object())
    source_ids = list(study.source_ids)
    if len(source_ids) < count:
        raise ValueError(f"run has only {len(source_ids)} canonical sources; need {count}")
    return [[source_id] for source_id in source_ids[:count]]


def _identity(sub) -> dict:
    identity = getattr(sub, "identity_meta", None)
    return dict(identity() or {}) if callable(identity) else {}


def run(args: argparse.Namespace) -> dict:
    import clozn.runs.store as runlog
    from clozn.receipts.context_dependence import ContextDependenceStudy
    from clozn.server.substrates import EngineSubstrate

    run_record = runlog.get_run(args.run_id)
    if not isinstance(run_record, dict):
        raise ValueError(f"run {args.run_id!r} was not found")
    try:
        from clozn_engine import EngineClient
    except ImportError as exc:
        client_root = Path(__file__).resolve().parents[2] / "engine" / "client"
        if str(client_root) not in sys.path:
            sys.path.insert(0, str(client_root))
        try:
            from clozn_engine import EngineClient
        except ImportError:
            raise RuntimeError(
                "the local engine client is unavailable; start the production runtime first"
            ) from exc
    engine = EngineClient(host=args.host, port=args.port)
    sub = EngineSubstrate(engine=engine)
    source_sets = _arms(run_record, args.arms)

    old_workers = os.environ.get("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS")
    try:
        os.environ["CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS"] = "1"
        scalar_study = ContextDependenceStudy(deepcopy(run_record), sub)
        scalar_study._ensure_baseline()
        started = time.perf_counter()
        scalar = [scalar_study.measure_removal_effect(removed) for removed in source_sets]
        scalar_seconds = max(0.0, time.perf_counter() - started)

        os.environ["CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS"] = str(args.workers)
        batch_study = ContextDependenceStudy(deepcopy(run_record), sub)
        batch_study._ensure_baseline()
        started = time.perf_counter()
        batched = batch_study.measure_removal_effect_many(source_sets)
        batch_seconds = max(0.0, time.perf_counter() - started)
    finally:
        if old_workers is None:
            os.environ.pop("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", None)
        else:
            os.environ["CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS"] = old_workers

    scalar_evidence = [
        {key: value for key, value in row.items() if key not in {"score_ms"}}
        for row in scalar
    ]
    batch_evidence = [
        {key: value for key, value in row.items() if key not in {"score_ms"}}
        for row in batched
    ]
    if scalar_evidence != batch_evidence:
        raise RuntimeError("scalar and concurrent Minimal Context evidence differ")
    return {
        "run_id": args.run_id,
        "runtime_identity": _identity(sub),
        "arms": len(source_sets),
        "workers": args.workers,
        "scalar_wall_time_s": scalar_seconds,
        "batch_wall_time_s": batch_seconds,
        "scalar_arms_per_second": len(source_sets) / scalar_seconds if scalar_seconds else None,
        "batch_arms_per_second": len(source_sets) / batch_seconds if batch_seconds else None,
        "speedup": scalar_seconds / batch_seconds if batch_seconds else None,
        "median_divergence_position": None,
        "memory_vram_increase": "not measured by harness",
        "context_slot_occupancy": "engine-dependent",
        "evidence_equal": True,
        "baseline_passes_excluded_from_timing": True,
        "protocol": "teacher_forced_likelihood",
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--arms", type=int, default=8)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True, indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
