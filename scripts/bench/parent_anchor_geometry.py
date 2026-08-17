#!/usr/bin/env python3
"""Measure the Phase-A parent-to-child KV reuse ceiling on saved scaled runs.

The saved reducer ledger supplies preservation outcomes and candidate order.
The live worker is used only for ordinary exact template rendering with token
IDs, so this diagnostic performs no model inference and does not change the
search behavior or certificate authority.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ENGINE_CLIENT_ROOT = ROOT / "engine" / "client"
if str(ENGINE_CLIENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_CLIENT_ROOT))

from clozn.experiments.effective_prompt import render_effective_prompt_for_retained
from clozn.experiments.search import PreparedCandidate, accepted_trial, run_budgeted_reduction
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.runs.parent_anchor_geometry import (
    SCHEMA,
    aggregate_batches,
    aggregate_case,
    build_probe_row,
)
from clozn.runs.store import get_run


DEFAULT_REPORT = "/tmp/minimal_context_scaled_live.json"
DEFAULT_OUTPUT = "/tmp/parent_anchor_geometry_v0.json"


def _progress(message: str, started: float, enabled: bool) -> None:
    if enabled:
        print(
            f"[parent-anchor-geometry +{time.perf_counter() - started:7.1f}s] {message}",
            file=sys.stderr,
            flush=True,
        )


class _RecordedLedgerAdapter:
    def __init__(self, *, run: dict[str, Any], universe_ids: tuple[str, ...], ledger: list[dict[str, Any]], engine: Any):
        self.run = run
        self.universe_ids = universe_ids
        self.ledger = ledger
        self.engine = engine
        self.by_ids = {tuple(row["retained_source_ids"]): row for row in ledger}
        if len(self.by_ids) != len(ledger):
            raise ValueError("saved trial ledger contains duplicate retained-source candidates")
        self.token_ids_by_candidate: dict[tuple[str, ...], tuple[int, ...]] = {}
        self.worker_cost_by_candidate: dict[tuple[str, ...], int] = {}
        self.cost_mismatches: list[dict[str, Any]] = []

    def prepare_candidate(self, retained_ids: tuple[str, ...]) -> PreparedCandidate:
        row = self.by_ids.get(tuple(retained_ids))
        messages = render_effective_prompt_for_retained(self.run, self.universe_ids, tuple(retained_ids))
        rendered = self.engine.apply_template_info(messages, include_token_ids=True)
        token_ids = tuple(rendered["prompt_token_ids"])
        worker_cost = len(token_ids)
        recorded_cost = int(row["cost"]) if row is not None else worker_cost
        if row is not None and worker_cost != recorded_cost:
            self.cost_mismatches.append({
                "retained_source_ids": list(retained_ids),
                "recorded_prompt_tokens": recorded_cost,
                "worker_prompt_tokens": worker_cost,
            })
        self.token_ids_by_candidate[tuple(retained_ids)] = token_ids
        self.worker_cost_by_candidate[tuple(retained_ids)] = worker_cost
        # Recorded costs freeze the original path for tested candidates.  The
        # reducer also prepares untested candidates before sorting a batch, so
        # those use the current exact worker count only; they must never be
        # dispatched because their behavioral outcome is intentionally absent
        # from the saved ledger.
        return PreparedCandidate(
            retained_ids=tuple(retained_ids),
            cost=recorded_cost,
            probe_payload={"retained_source_ids": list(retained_ids)},
        )

    def probe_many(self, prepared_candidates: list[PreparedCandidate]) -> list[dict[str, Any]]:
        output = []
        for prepared in prepared_candidates:
            row = self.by_ids.get(tuple(prepared.retained_ids))
            if row is None:
                raise ValueError(
                    "replay attempted to dispatch a candidate absent from the saved ledger: "
                    f"{prepared.retained_ids!r}"
                )
            output.append({
                "status": "matched" if bool(row["preserves"]) else "diverged",
                "replay": True,
            })
        return output


def _compare_replay(result: Any, ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mismatches = []
    if len(result.trials) != len(ledger):
        mismatches.append({"kind": "trial_count", "expected": len(ledger), "actual": len(result.trials)})
    for index, (trial, expected) in enumerate(zip(result.trials, ledger)):
        actual = {
            "ordinal": trial.ordinal,
            "stage": trial.stage,
            "retained_source_ids": list(trial.retained_ids),
            "cost": trial.cost,
            "preserves": trial.preserves,
        }
        wanted = {
            "ordinal": expected["ordinal"],
            "stage": expected["stage"],
            "retained_source_ids": list(expected["retained_source_ids"]),
            "cost": expected["cost"],
            "preserves": expected["preserves"],
        }
        if actual != wanted:
            mismatches.append({"kind": "trial", "index": index, "expected": wanted, "actual": actual})
    return mismatches


def _case_geometry(case_report: dict[str, Any], *, engine: Any, max_units: int, progress_started: float, quiet: bool) -> dict[str, Any]:
    case_id = str(case_report["case_id"])
    run_id = str(case_report["run_id"])
    run = get_run(run_id)
    if not isinstance(run, dict):
        raise ValueError(f"recorded run {run_id!r} is not available in the local run store")
    universe = plan_context_search_universe(run, run.get("context_units"), max_units=max_units)
    if universe.get("status") != "planned":
        raise ValueError(f"case {case_id} has no planned search universe")
    universe_ids = tuple(universe["source_ids"])
    ledger = list(case_report.get("trial_ledger") or [])
    if not ledger:
        raise ValueError(f"case {case_id} has no saved trial ledger")
    adapter = _RecordedLedgerAdapter(
        run=run,
        universe_ids=universe_ids,
        ledger=ledger,
        engine=engine,
    )
    _progress(f"{case_id}: replaying {len(ledger) - 1} counterfactual probes", progress_started, quiet)
    result = run_budgeted_reduction(
        universe_ids,
        int(case_report["budget"]["max_counterfactual_probes"]),
        adapter.prepare_candidate,
        adapter.probe_many,
        attempt_inclusion_check=True,
    )
    replay_mismatches = _compare_replay(result, ledger)
    if replay_mismatches:
        raise ValueError(f"case {case_id} replay did not match saved ledger: {replay_mismatches[0]}")

    accepted = {trial.ordinal: accepted_trial(result, trial) for trial in result.trials}
    rows = []
    for trial in result.trials:
        if trial.stage == "control":
            continue
        parent_ids = tuple(trial.parent_retained_ids)
        child_ids = tuple(trial.retained_ids)
        if not parent_ids:
            raise ValueError(f"case {case_id} trial {trial.ordinal} has no semantic parent")
        rows.append(build_probe_row(
            ordinal=trial.ordinal,
            stage=trial.stage,
            batch_id=int(trial.batch_id),
            parent_source_ids=parent_ids,
            child_source_ids=child_ids,
            parent_token_ids=adapter.token_ids_by_candidate[parent_ids],
            child_token_ids=adapter.token_ids_by_candidate[child_ids],
            preserved=trial.preserves,
            accepted_as_best=accepted[trial.ordinal],
        ))

    aggregate = aggregate_case(rows)
    return {
        "case_id": case_id,
        "run_id": run_id,
        "search_universe_count": len(universe_ids),
        "replay": {
            "behavioral_outcomes": "saved_trial_ledger",
            "ledger_match": True,
            "cost_mismatch_count": len(adapter.cost_mismatches),
            "cost_mismatches": adapter.cost_mismatches,
        },
        "aggregate": aggregate,
        "batches": aggregate_batches(rows),
        "probes": rows,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    report_path = Path(args.report)
    source_report = json.loads(report_path.read_text())
    if not isinstance(source_report.get("cases"), list) or len(source_report["cases"]) != 6:
        raise ValueError("Phase A requires the six-case scaled report")
    from clozn_engine import EngineClient

    engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
    health = engine.health()
    _progress(
        f"worker healthy: model={health.get('model', '?')} n_batch={health.get('n_batch', '?')} "
        f"n_ubatch={health.get('n_ubatch', '?')}", started, not args.quiet,
    )
    cases = [
        _case_geometry(
            dict(case), engine=engine, max_units=args.max_units,
            progress_started=started, quiet=not args.quiet,
        )
        for case in source_report["cases"]
    ]
    request_local_values = [
        case["aggregate"]["request_local_reduction_percent"]
        for case in cases
        if case["aggregate"]["request_local_reduction_percent"] is not None
    ]
    gate_cases = [
        case["case_id"] for case in cases
        if (case["aggregate"]["request_local_reduction_percent"] or 0) >= 25.0
    ]
    one_case_at_40 = any(value >= 40.0 for value in request_local_values)
    gate_passed = len(gate_cases) >= 2 or one_case_at_40
    return {
        "schema_version": SCHEMA,
        "phase": "A",
        "source_report": str(report_path),
        "workload": {
            "case_count": len(cases),
            "max_units": args.max_units,
            "p200_included": False,
            "native_inference_performed": False,
            "worker": {
                "host": args.host,
                "port": args.port,
                "model": health.get("model"),
                "n_batch": health.get("n_batch"),
                "n_ubatch": health.get("n_ubatch"),
                "n_ctx": health.get("n_ctx"),
            },
        },
        "definitions": {
            "semantic_parent": "reducer current directly-preserving best candidate at child dispatch time",
            "child": "directly tested retained-source subset of that semantic parent",
            "logical_rows": "sum of child prompt token counts in a search batch",
            "request_local_parent_ideal_rows": "one parent prompt plus each child's non-LCP suffix",
            "persistent_parent_ideal_rows": "each child's non-LCP suffix; parent residency is free",
            "interpretation": "structural theoretical rows only; no wall-speedup claim",
        },
        "gate": {
            "request_local_threshold_percent": 25.0,
            "required_cases_at_threshold": 2,
            "single_case_threshold_percent": 40.0,
            "cases_at_threshold": gate_cases,
            "single_case_at_threshold": one_case_at_40,
            "passed": gate_passed,
            "phase_b_implemented": False,
        },
        "cases": cases,
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-units", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
        output = Path(args.output)
        output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        print(json.dumps({
            "status": "ok",
            "output": str(output),
            "phase": result["phase"],
            "gate_passed": result["gate"]["passed"],
            "case_count": len(result["cases"]),
        }, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
