#!/usr/bin/env python3
"""Profile the opt-in coalition recipe against an already running local worker.

Set CLOZN_DATA_ROOT to the gateway's data root. --worker-port is the private
native worker port, not the public OpenAI gateway port. JSON goes to stdout;
progress goes to stderr. Compatible observations are cached in the normal store.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from clozn_engine import EngineClient
from clozn.artifacts.contracts import sha256_file
from clozn.experiments.persistence import ObservationStore
from clozn.recipes.context_effects import measure_context_effects
from clozn.recipes.random_coalitions import run_random_coalitions
from clozn.experiments.scoring import _runtime_binding
from clozn.runs.store import get_run
# app initializes the existing substrate/app import seam in the supported order.
from clozn.server.app import EngineSubstrate


class TimedEngine(EngineClient):
    def __init__(self, **kwargs):
        self.requests = []
        super().__init__(**kwargs)

    def _request(self, method, path, body=None):
        started = time.perf_counter()
        result = None
        try:
            result = super()._request(method, path, body)
            return result
        finally:
            row = {"path": path, "elapsed_seconds": time.perf_counter() - started,
                   "ok": result is not None}
            if result is not None and path == "/score":
                for key in ("n_prompt", "n_cont", "cache", "numerical_regime"):
                    row[key] = result.get(key)
                print(f"worker score: {row.get('n_prompt')} prompt tokens, "
                      f"{row['elapsed_seconds']:.3f}s", file=sys.stderr, flush=True)
            self.requests.append(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--worker-executable", type=Path, required=True,
                        help="executable used to start this worker; verify against its process command")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-units", type=int, default=32)
    parser.add_argument("--compare-leave-one-out", action="store_true",
                        help="run a separate baseline sweep first (up to n+1 scores, outside --budget)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.budget < 0 or args.pairs < 0 or args.max_units <= 0:
        parser.error("budget/pairs must be non-negative and max-units must be positive")
    if args.output is not None and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("output must be a new file in an existing directory")
    run = get_run(args.run_id)
    if run is None:
        parser.error("recorded run not found; check CLOZN_DATA_ROOT")
    engine = TimedEngine(port=args.worker_port, timeout=180)
    substrate = EngineSubstrate(engine)
    substrate._engine_artifact_context = {"artifact_sha256": sha256_file(str(args.worker_executable.resolve()))}
    _, mismatch = _runtime_binding(run, substrate)
    if mismatch:
        parser.error(f"{mismatch}: use a run captured by this worker build; do not relabel old evidence")
    store = ObservationStore()
    baseline = None
    if args.compare_leave_one_out:
        print("Running separate leave-one-out comparison sweep", file=sys.stderr, flush=True)
        started = time.perf_counter()
        first_request = len(engine.requests)
        measured = measure_context_effects(run, substrate=substrate, observation_store=store)
        baseline = {"elapsed_seconds": time.perf_counter() - started,
                    "requests": engine.requests[first_request:],
                    "experiment_id": measured.experiment_id,
                    "control_status": measured.control.status if measured.control else None,
                    "arm_statuses": [row.status for row in measured.arms]}
    first_sampling_request = len(engine.requests)

    def progress(row):
        print(f"{row['disposition']:8s} kept={len(row['retained_source_ids']):2d} "
              f"tokens={row['rendered_prompt_tokens']:5d} seconds={row['elapsed_seconds']:.3f} "
              f"new={row['used_new_executions']}/{args.budget} {row['status']}",
              file=sys.stderr, flush=True)

    result = run_random_coalitions(
        run, substrate=substrate, engine=engine, observation_store=store,
        max_new_executions=args.budget, max_pairs=args.pairs, seed=args.seed,
        max_units=args.max_units, progress=progress,
    )
    requests = engine.requests[first_sampling_request:]
    scores = [row for row in requests if row["path"] == "/score"]
    result["worker_profile"] = {
        "note": "HTTP wall times include worker compute/queue/serialization; not GPU kernel timings",
        "score_request_count": len(scores),
        "score_http_seconds": sum(row["elapsed_seconds"] for row in scores),
        "other_http_seconds": sum(row["elapsed_seconds"] for row in requests if row["path"] != "/score"),
        "requests": requests,
    }
    result["leave_one_out_benchmark"] = baseline
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        # Exclusive create: experiment reports must not overwrite earlier results.
        with args.output.open("x") as output:
            output.write(encoded)
        print(f"Report: {args.output}", file=sys.stderr)
    else:
        print(encoded, end="")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
