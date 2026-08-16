#!/usr/bin/env python3
"""Run the opt-in Budgeted Minimal Context v0 Aurora regression harness.

The reducer reports the lowest-cost preserving candidate observed within the
direct probe budget.  It makes no global-minimum claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))


def _progress(message: str, started: float, enabled: bool) -> None:
    if enabled:
        print(
            f"[budgeted-minimal-context +{time.perf_counter() - started:7.1f}s] {message}",
            file=sys.stderr,
            flush=True,
        )


def _compact_report(result: Any, fixture: Any, max_probes: int) -> dict[str, Any]:
    original = result.original_candidate
    best = result.best_candidate
    removed_cost = original.cost - best.cost
    removed_percent = 100.0 * removed_cost / original.cost if original.cost else None
    return {
        "status": result.status,
        "objective": "rendered_prompt_tokens",
        "control": {
            "passed": result.trials[0].preserves if result.trials else False,
        },
        "original": {
            "unit_count": len(original.retained_ids),
            "cost": original.cost,
        },
        "best_verified": {
            "retained_unit_ids": list(best.retained_ids),
            "retained_unit_count": len(best.retained_ids),
            "cost": best.cost,
            "removed_cost": removed_cost,
            "removed_percent": removed_percent,
        },
        "certificate_level": result.certificate_level,
        "budget": {
            "max_counterfactual_probes": max_probes,
            "used_counterfactual_probes": result.budget.used_counterfactual_probes,
            "exhausted": result.budget.exhausted,
        },
        "direct_experiments": result.direct_experiments,
        "trajectory": [
            {
                "after_probe": item.counterfactual_probe_count,
                "stage": item.stage,
                "retained_unit_count": item.retained_unit_count,
                "cost": item.cost,
                "retained_unit_ids": list(item.retained_ids),
            }
            for item in result.trajectory
        ],
        "inclusion_check": {
            "attempted": result.inclusion_check.attempted,
            "complete": result.inclusion_check.complete,
            "tested_child_count": result.inclusion_check.tested_child_count,
            "total_child_count": result.inclusion_check.total_child_count,
            "all_children_failed": result.inclusion_check.all_children_failed,
        },
        "trial_ledger": [
            {
                "ordinal": trial.ordinal,
                "stage": trial.stage,
                "retained_unit_ids": list(trial.retained_ids),
                "cost": trial.cost,
                "preserves": trial.preserves,
            }
            for trial in result.trials
        ],
        "fixture": {
            "unit_count": len(fixture.units),
            # Regression metadata only; this value is never passed to the
            # reducer as a candidate or search hint.
            "known_preserving_candidate": list(fixture.known_preserving_candidate),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    if args.max_probes < 0:
        raise ValueError("--max-probes must be non-negative")

    from native_reference_match import QUESTION, _contract, build_fixture, render

    fixture = build_fixture()
    if not args.live:
        return {
            "status": "fixture_only",
            "objective": "rendered_prompt_tokens",
            "max_counterfactual_probes": args.max_probes,
            "fixture": {
                "unit_count": len(fixture.units),
                "known_preserving_candidate": list(fixture.known_preserving_candidate),
            },
        }

    try:
        from clozn.server import app as cs
        from clozn.runs.budgeted_reduce_reference import (
            EngineReferenceMatchAdapter,
            PersistentEngineReferenceMatchAdapter,
            run_engine_reference_match_reduction,
            run_engine_reference_match_persistent_reduction,
        )
        EngineSubstrate = cs.EngineSubstrate
    except ImportError:
        client_root = ROOT / "engine" / "client"
        if str(client_root) not in sys.path:
            sys.path.insert(0, str(client_root))
        from clozn.server import app as cs
        from clozn.runs.budgeted_reduce_reference import (
            EngineReferenceMatchAdapter,
            PersistentEngineReferenceMatchAdapter,
            run_engine_reference_match_reduction,
            run_engine_reference_match_persistent_reduction,
        )
        EngineSubstrate = cs.EngineSubstrate
    try:
        from clozn_engine import EngineClient
    except ImportError:
        client_root = ROOT / "engine" / "client"
        if str(client_root) not in sys.path:
            sys.path.insert(0, str(client_root))
        from clozn_engine import EngineClient

    _progress(f"connecting to worker {args.host}:{args.port}", started, not args.quiet)
    engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
    health = engine.health()
    _progress(
        "worker healthy: "
        f"model={health.get('model', '<unknown>')} "
        f"ctx={health.get('n_ctx', '?')} batch={health.get('n_batch', '?')} "
        f"ubatch={health.get('n_ubatch', '?')}",
        started,
        not args.quiet,
    )

    sub = EngineSubstrate(engine=engine)
    full_retained = tuple(range(len(fixture.units)))
    full_messages = [
        {"role": "system", "content": render(fixture.units, full_retained)},
        {"role": "user", "content": QUESTION},
    ]
    _progress("rendering full context and collecting the recorded reference", started, not args.quiet)
    rendered = engine.apply_template_info(full_messages)
    baseline = engine.complete(
        rendered["prompt"],
        max_tokens=args.max_new,
        temperature=0.0,
        rep_penalty=1.0,
        top_k=0,
        top_p=1.0,
        seed=0,
    )
    prompt_tokens = rendered.get("prompt_tokens")
    board = baseline.get("board")
    if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int) or not isinstance(board, list):
        raise RuntimeError("worker baseline omitted exact prompt_tokens or board token IDs")
    reference = [int(token) for token in board[prompt_tokens:]]
    if not reference:
        raise RuntimeError("worker baseline generated no reference tokens")
    termination = dict(baseline.get("termination") or {})
    contract = _contract(len(reference) + 1, termination)

    def render_messages(retained: tuple[int, ...]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": render(fixture.units, retained)},
            {"role": "user", "content": QUESTION},
        ]

    adapter_type = (PersistentEngineReferenceMatchAdapter
                    if args.experimental_persistent_parent else EngineReferenceMatchAdapter)
    adapter = adapter_type(
        engine=engine,
        substrate=sub,
        render_messages=render_messages,
        reference_token_ids=tuple(reference),
        generation_contract=contract,
    )
    _progress(
        f"running reducer with max_counterfactual_probes={args.max_probes}",
        started,
        not args.quiet,
    )
    if args.experimental_persistent_parent:
        result = run_engine_reference_match_persistent_reduction(
            adapter, full_retained, args.max_probes,
            attempt_inclusion_check=not args.no_inclusion_check,
        )
    else:
        result = run_engine_reference_match_reduction(
            adapter, full_retained, args.max_probes,
            attempt_inclusion_check=not args.no_inclusion_check,
        )
    report = _compact_report(result, fixture, args.max_probes)
    report["reference"] = {
        "prompt_tokens": prompt_tokens,
        "token_count": len(reference),
        "generation_contract": contract,
    }
    report["execution_mode"] = (
        "experimental_persistent_parent" if args.experimental_persistent_parent else "trusted_scalar"
    )
    if args.experimental_persistent_parent:
        report["persistent_parent"] = {
            "session": adapter.persistent_parent_final_report,
            "round_metrics": adapter.persistent_parent_metrics,
            "promotion_metrics": adapter.persistent_parent_promotion_metrics,
            "parity_mismatches": adapter.persistent_parent_parity_mismatches,
            "proof_grade": False,
        }
    return report


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="run against a live clozn worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--max-probes", type=int, default=100)
    parser.add_argument("--no-inclusion-check", action="store_true")
    parser.add_argument(
        "--experimental-persistent-parent", action="store_true",
        help="use the opt-in persistent accepted-parent native session with scalar confirmation",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress logs on stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        print(json.dumps(run(parser().parse_args(argv)), ensure_ascii=False, sort_keys=True, indent=2))
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
