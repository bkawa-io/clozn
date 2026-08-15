#!/usr/bin/env python3
"""Evaluate five ordinary recorded Clozn runs with the budgeted reducer.

Without ``--live`` or a saved run, this command only validates the deterministic
fixture corpus.  ``--live`` records each baseline once, then runs exact
counterfactual probes against that saved ordinary run.
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

from clozn.runs.realistic_minimal_context import (  # noqa: E402
    CHECKPOINT_TARGETS,
    GEOMETRIC_CHECKPOINT_TARGETS,
    SCHEMA,
    bind_engine_recorded_run,
    evaluate_recorded_run,
    serialize_outcome,
    suite_summary,
)
from clozn import schemas  # noqa: E402
from scripts.bench.fixtures.minimal_context_realistic import (  # noqa: E402
    SCENARIO_IDS,
    RealisticScenario,
    build_fixture_run,
    built_in_scenarios,
    get_scenario,
    validate_registry,
)
from scripts.bench.fixtures.minimal_context_scaled import (  # noqa: E402
    SCALED_SCENARIO_IDS,
    built_in_scaled_scenarios,
    get_scaled_scenario,
    validate_scaled_registry,
)


ALL_SCENARIO_IDS = SCENARIO_IDS + SCALED_SCENARIO_IDS


def _progress(message: str, started: float, enabled: bool) -> None:
    if enabled:
        print(
            f"[realistic-minimal-context +{time.perf_counter() - started:7.1f}s] {message}",
            file=sys.stderr,
            flush=True,
        )


def _load_run(path: str | None, run_id: str | None) -> dict[str, Any]:
    if path:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("run"), dict):
            payload = payload["run"]
        if not isinstance(payload, dict):
            raise ValueError("--run-file must contain one recorded run object")
        return payload
    if run_id:
        from clozn.runs.store import get_run
        run = get_run(run_id)
        if run is None:
            raise ValueError(f"recorded run not found: {run_id}")
        return run
    raise ValueError("a saved run path or ID is required")


def _suite_scenarios(suite: str) -> list[RealisticScenario]:
    if suite == "small_semantic":
        return list(built_in_scenarios())
    if suite == "scaled_realistic":
        return list(built_in_scaled_scenarios())
    return list(built_in_scenarios()) + list(built_in_scaled_scenarios())


def _suite_registry(suite: str, *, max_units: int) -> list[dict[str, Any]]:
    if suite == "small_semantic":
        rows = validate_registry(max_units=max_units)
        return [{**row, "suite": suite} for row in rows]
    if suite == "scaled_realistic":
        rows = validate_scaled_registry(max_units=max_units)
        return [{**row, "suite": suite} for row in rows]
    small_rows = validate_registry(max_units=max_units)
    scaled_rows = validate_scaled_registry(max_units=max_units)
    rows = small_rows + scaled_rows
    split = len(small_rows)
    return [
        {**row, "suite": "small_semantic" if index < split else "scaled_realistic"}
        for index, row in enumerate(rows)
    ]


def _scenario_for_case(case_id: str) -> RealisticScenario:
    if case_id in SCENARIO_IDS:
        return get_scenario(case_id)
    if case_id in SCALED_SCENARIO_IDS:
        return get_scaled_scenario(case_id)
    raise KeyError(f"unknown realistic scenario: {case_id}")


def _suite_for_case(case_id: str) -> str:
    return "small_semantic" if case_id in SCENARIO_IDS else "scaled_realistic"


def _engine_types():
    try:
        from clozn.server import app as cs
        EngineSubstrate = cs.EngineSubstrate
    except ImportError:
        EngineSubstrate = None
    try:
        from clozn_engine import EngineClient
    except ImportError:
        client_root = ROOT / "engine" / "client"
        if str(client_root) not in sys.path:
            sys.path.insert(0, str(client_root))
        from clozn_engine import EngineClient
    if EngineSubstrate is None:
        from clozn.server import app as cs
        EngineSubstrate = cs.EngineSubstrate
    return EngineClient, EngineSubstrate


def _capture_live_run(
    scenario: RealisticScenario, *, engine: Any, substrate: Any, health: dict[str, Any], max_new: int,
) -> tuple[dict[str, Any], float]:
    """Capture one normal greedy EngineSubstrate inference in the run store."""
    from clozn.runs import store

    messages = [dict(message) for message in scenario.messages]
    trace_steps: list[dict[str, Any]] = []
    memory: dict[str, Any] = {}
    started = time.time()
    wall_started = time.perf_counter()
    reply = substrate.chat(
        deepcopy(messages), max_new=max_new, sample=False, trace_out=trace_steps,
        mem_out=memory, stop=None,
    )
    ended = time.time()
    raw_response = "".join(
        str(step.get("piece", "")) for step in trace_steps if isinstance(step, dict)
    ) or str(reply)
    finish = substrate.last_finish_reason()
    raw_finish = substrate.last_finish_reason_raw()
    prompt_tokens = substrate.last_prompt_tokens()
    meta = dict(substrate.run_meta() or {})
    contract = None
    if isinstance(finish, str) and finish:
        contract = {
            "decode_mode": "greedy",
            "sampling": None,
            "max_new": int(max_new),
            "stop": [],
            "expected_termination": {
                "reason": finish,
                "reason_raw": raw_finish if isinstance(raw_finish, str) and raw_finish else finish,
            },
        }
        meta["generation_contract"] = contract
    if isinstance(prompt_tokens, int):
        meta["prompt_tokens"] = prompt_tokens
    model = str((health or {}).get("model") or "")
    rid = store.record(
        source="realistic_minimal_context",
        client="realistic_minimal_context",
        model=model,
        substrate="engine",
        messages=messages,
        response=raw_response,
        trace=trace_steps,
        started=started,
        ended=ended,
        finish_reason=finish if isinstance(finish, str) else None,
        meta=meta,
        assembled_messages=memory.get("assembled_messages", messages),
        final_prompt=memory.get("final_prompt"),
        identity=substrate.identity_meta(),
    )
    if not isinstance(rid, str) or not rid:
        raise RuntimeError("ordinary run recording failed")
    run = store.get_run(rid)
    if not isinstance(run, dict):
        raise RuntimeError("ordinary run could not be reloaded after recording")
    return run, max(0.0, time.perf_counter() - wall_started)


def _case_report(
    scenario: RealisticScenario, run: dict[str, Any], *, engine: Any, substrate: Any,
    max_units: int, max_probes: int, checkpoints: list[int], started: float,
    suite: str, baseline_capture_wall_seconds: float | None = None,
) -> dict[str, Any]:
    binding = bind_engine_recorded_run(
        run, engine=engine, substrate=substrate, max_units=max_units,
    )
    outcome = evaluate_recorded_run(
        run, adapter=binding.adapter, max_counterfactual_probes=max_probes,
        max_units=max_units, eligibility=binding.eligibility,
    )
    return serialize_outcome(
        case_id=scenario.case_id,
        description=scenario.description,
        tags=scenario.tags,
        run=run,
        outcome=outcome,
        timing_seconds=time.perf_counter() - started,
        max_counterfactual_probes=max_probes,
        checkpoints=checkpoints,
        suite=suite,
        baseline_capture_wall_seconds=baseline_capture_wall_seconds,
    )


def _failed_case_report(
    scenario: RealisticScenario, *, status: str, reason: str, max_probes: int,
    max_units: int, timing_seconds: float = 0.0, suite: str,
) -> dict[str, Any]:
    run = build_fixture_run(scenario)
    manifest = run["context_units"]
    return {
        "case_id": scenario.case_id,
        "description": scenario.description,
        "case_tags": list(scenario.tags),
        "suite": suite,
        "run_id": run["id"],
        "status": status,
        "reason": reason,
        "exact_replay_eligibility": {"eligible": False, "reason": reason, "reasons": []},
        "universe": {
            "source_count": len(manifest.get("default_source_ids") or []),
            "universe_id": None,
            "removable_unit_count": len(manifest.get("default_source_ids") or []),
            "protected_message_indices": list(manifest.get("protected_message_indices") or []),
        },
        "budget": {
            "max_counterfactual_probes": max_probes,
            "used_counterfactual_probes": 0,
            "total_counterfactual_probes": 0,
        },
        "checkpoints": [],
        "final": None,
        "timing": {"wall_seconds": round(float(timing_seconds), 6)},
        "termination": {"reason": "fixture_invalid", "probe": None},
    }


def _print_table(reports: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    print("case                   suite             raw  search  tokens probes  p1    p2    p4    p8   p16   final  cert", file=sys.stderr)
    print("---------------------  ----------------  ---  ------  ------ ------ ----- ----- ----- ----- ----- ------ ----------------", file=sys.stderr)
    for report in reports:
        original = report.get("original") or {}
        by_probe = {item.get("probe_count"): item.get("reduction_percent") for item in report.get("geometric_checkpoints", [])}
        cells = ["-" if by_probe.get(target) is None else f"{by_probe[target]:4.1f}%" for target in (1, 2, 4, 8, 16)]
        final_reduction = (report.get("final") or {}).get("reduction_percent")
        print(
            f"{str(report.get('case_id', '')):21}  {str(report.get('suite', '-')):16}  "
            f"{str(report.get('raw_context_unit_count', '-')):>3}  {str(report.get('bounded_search_universe_count', '-')):>6}  "
            f"{str(original.get('rendered_prompt_tokens', '-')):>6} {str((report.get('budget') or {}).get('used_counterfactual_probes', '-')):>6}  "
            f"{' '.join(f'{cell:>5}' for cell in cells)} {('-' if final_reduction is None else f'{final_reduction:5.1f}%'):>6}  "
            f"{str((report.get('final') or {}).get('certificate_level') or '-'):16}",
            file=sys.stderr,
        )
    print(
        f"eligible={summary.get('eligible_case_count', 0)} "
        f"counterfactual_probes={summary.get('total_counterfactual_probes', 0)} "
        f"terminated<16={((summary.get('descriptive_questions') or {}).get('terminated_before_probe_16') or {}).get('count', 0)} "
        f"terminated<32={((summary.get('descriptive_questions') or {}).get('terminated_before_probe_32') or {}).get('count', 0)} "
        f"hit200={((summary.get('descriptive_questions') or {}).get('hit_probe_200') or {}).get('count', 0)}",
        file=sys.stderr,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.max_probes < 0 or args.max_units <= 0 or args.max_new <= 0:
        raise ValueError("max-probes must be non-negative; max-units and max-new must be positive")
    if args.run_file and args.run_id:
        raise ValueError("use only one of --run-file or --run-id")
    if args.live and (args.run_file or args.run_id):
        raise ValueError("--live captures new runs; use --run-file/--run-id for saved runs")

    registry = _suite_registry(args.suite, max_units=args.max_units)
    if args.list_cases:
        return {
            "schema_version": SCHEMA,
            "cases": registry,
            "suite_summary": {"case_count": len(registry), "suite": args.suite},
        }
    checkpoints = [value for value in CHECKPOINT_TARGETS if value <= args.max_probes]
    if not checkpoints:
        checkpoints = [0]

    if args.run_file or args.run_id:
        run = _load_run(args.run_file, args.run_id)
        case_id = args.case or run.get("case_id") or "saved_run"
        scenario = _scenario_for_case(case_id) if case_id in ALL_SCENARIO_IDS else RealisticScenario(
            case_id, "A saved ordinary recorded run.", ("saved",),
            tuple(dict(message) for message in run.get("messages") or ()),
        )
        case_suite = args.suite if args.suite != "all" else _suite_for_case(case_id)
        EngineClient, EngineSubstrate = _engine_types()
        _progress(f"connecting to worker {args.host}:{args.port}", started, not args.quiet)
        engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
        health = engine.health()
        substrate = EngineSubstrate(engine=engine)
        report = _case_report(
            scenario, run, engine=engine, substrate=substrate, max_units=args.max_units,
            max_probes=args.max_probes, checkpoints=checkpoints, started=started,
            suite=case_suite,
        )
        reports = [report]
    elif args.live:
        EngineClient, EngineSubstrate = _engine_types()
        _progress(f"connecting to worker {args.host}:{args.port}", started, not args.quiet)
        engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
        health = engine.health()
        substrate = EngineSubstrate(engine=engine)
        reports = []
        scenarios = [_scenario_for_case(args.case)] if args.case else _suite_scenarios(args.suite)
        for scenario in scenarios:
            case_started = time.perf_counter()
            try:
                _progress(f"capturing {scenario.case_id}", started, not args.quiet)
                run, capture_wall_seconds = _capture_live_run(
                    scenario, engine=engine, substrate=substrate, health=health, max_new=args.max_new,
                )
                _progress(f"reducing {scenario.case_id} with {args.max_probes} counterfactual probes", started, not args.quiet)
                reports.append(_case_report(
                    scenario, run, engine=engine, substrate=substrate, max_units=args.max_units,
                    max_probes=args.max_probes, checkpoints=checkpoints, started=case_started,
                    suite=_suite_for_case(scenario.case_id),
                    baseline_capture_wall_seconds=capture_wall_seconds,
                ))
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                _progress(f"{scenario.case_id} unavailable: {exc}", started, not args.quiet)
                reports.append(_failed_case_report(
                    scenario, status="case_unavailable", reason=str(exc),
                    max_probes=args.max_probes, max_units=args.max_units,
                    timing_seconds=time.perf_counter() - case_started,
                    suite=_suite_for_case(scenario.case_id),
                ))
    else:
        selected_ids = [args.case] if args.case else [scenario.case_id for scenario in _suite_scenarios(args.suite)]
        fixture_runs = [build_fixture_run(_scenario_for_case(case_id)) for case_id in selected_ids]
        reports = []
        registry_by_id = {item["case_id"]: item for item in registry}
        for case_id, run in zip(selected_ids, fixture_runs):
            item = registry_by_id[case_id]
            reports.append({
                **item,
                "run_id": run["id"],
                "status": "fixture_only",
                "universe": {
                    "source_count": item.get("source_count", item.get("bounded_search_universe_count")),
                    "universe_id": item["universe_id"],
                    "removable_unit_count": item.get("removable_unit_count", item.get("bounded_search_universe_count")),
                    "protected_message_indices": item["protected_message_indices"],
                },
                "raw_context_unit_count": item.get("raw_context_unit_count", len(run["context_units"].get("units") or [])),
                "bounded_search_universe_count": item.get("bounded_search_universe_count", item.get("source_count")),
            })

    summary = suite_summary(reports, max_counterfactual_probes=args.max_probes)
    return {
        "schema_version": SCHEMA,
        "cases": reports,
        "suite_summary": summary,
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="capture and evaluate the selected live suite")
    parser.add_argument("--suite", choices=("small_semantic", "scaled_realistic", "all"), default="all")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--max-probes", type=int, default=200)
    parser.add_argument("--max-units", type=int, default=50)
    parser.add_argument("--case", choices=ALL_SCENARIO_IDS)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--run-file")
    parser.add_argument("--run-id")
    parser.add_argument("--output", help="also write the full JSON report to this path")
    parser.add_argument("--quiet", action="store_true", help="suppress progress logs")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
        report = run(args)
        schemas.validate(report)
        encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
        if args.output:
            Path(args.output).write_text(encoded + "\n", encoding="utf-8")
        print(encoded)
        if not args.list_cases and not (not args.live and not args.run_file and not args.run_id):
            _print_table(report["cases"], report["suite_summary"])
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"schema_version": SCHEMA, "status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
