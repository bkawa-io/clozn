#!/usr/bin/env python3
"""Run the opt-in parent-anchor native arm on the frozen Phase-A batches.

Every native batch is paired with the existing scalar direct-reference probe on
the same messages, reference token IDs, and generation contract. Native rows
are diagnostic only; scalar classifications remain the evidence authority.
"""
from __future__ import annotations

import argparse
import json
import os
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

from clozn.runs.answer_preservation import _generation_contract_from_run, classify_reference_match
from clozn.experiments.multi_arm import probe_reference_match_many
from clozn.experiments.effective_prompt import render_effective_prompt_for_retained
from clozn.runs.store import get_run


SCHEMA = "clozn.parent-anchor-native-eval.v0"
DEFAULT_GEOMETRY = "/tmp/parent_anchor_geometry_v0.json"
DEFAULT_OUTPUT = "/tmp/parent_anchor_native_v0.json"


def _progress(message: str, started: float, enabled: bool) -> None:
    if enabled:
        print(
            f"[parent-anchor-native +{time.perf_counter() - started:7.1f}s] {message}",
            file=sys.stderr,
            flush=True,
        )


def _evidence_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key) for key in (
            "status", "matched_token_count", "first_divergence_index",
            "expected_token_id", "actual_token_id", "divergence_kind",
            "termination_match",
        )
    }


def _classify_native(row: dict[str, Any], reference: list[int], contract: dict[str, Any]) -> dict[str, Any]:
    raw = dict(row.get("result") or {})
    generated = raw.get("generated_token_ids")
    if not isinstance(generated, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in generated):
        raise ValueError("native parent-anchor result omitted integer generated_token_ids")
    classified = classify_reference_match(
        reference,
        generated,
        diverged=raw.get("diverged"),
        diverged_at=raw.get("diverged_at"),
        termination=raw.get("termination"),
        finish_reason=raw.get("finish_reason"),
        expected_termination=contract.get("expected_termination"),
        max_new=contract.get("max_new"),
    )
    classified.update({
        "generated_token_ids": list(generated),
        "finish_reason": raw.get("finish_reason"),
        "termination": dict(raw.get("termination") or {}),
    })
    return classified


def _case(
    case: dict[str, Any], geometry_case: dict[str, Any], *, engine: Any,
    substrate: Any, quiet: bool, started: float, max_batches: int | None = None,
) -> dict[str, Any]:
    run = get_run(str(case["run_id"]))
    if not isinstance(run, dict):
        raise ValueError(f"run {case['run_id']!r} is unavailable")
    conditions = __import__("clozn.receipts.rederive", fromlist=["with_arm_conditions"]).with_arm_conditions(run)
    contract, reason = _generation_contract_from_run(run)
    if not isinstance(contract, dict) or reason:
        raise ValueError(f"case {case['case_id']} has no exact generation contract: {reason}")
    reference = list(conditions["continuation_ids"])
    universe_ids = tuple(case["trial_ledger"][0]["retained_source_ids"])
    by_ordinal = {int(row["probe_ordinal"]): row for row in geometry_case["probes"]}
    batches = []
    parity_mismatches = []
    native_rows = 0
    scalar_rows = 0
    selected_batches = geometry_case["batches"]
    if max_batches is not None:
        selected_batches = selected_batches[:max_batches]
    for batch in selected_batches:
        probe_rows = [by_ordinal[int(ordinal)] for ordinal in batch["probe_ordinals"]]
        parent_ids = tuple(batch["parent_source_ids"])
        parent_messages = render_effective_prompt_for_retained(run, universe_ids, parent_ids)
        parent_prompt = engine.apply_template(parent_messages)
        native_arms = []
        scalar_arms = []
        for index, probe in enumerate(probe_rows):
            child_ids = tuple(probe["child_source_ids"])
            messages = render_effective_prompt_for_retained(run, universe_ids, child_ids)
            prompt = engine.apply_template(messages)
            native_arms.append({"arm_id": index, "prompt": prompt})
            scalar_arms.append({
                "messages": messages,
                "reference_token_ids": reference,
                "generation_contract": contract,
                "explicit_conditions": {},
            })

        _progress(
            f"{case['case_id']} batch {batch['batch_id']}: {len(native_arms)} children",
            started, not quiet,
        )
        scalar_started = time.perf_counter()
        scalar_evidence = probe_reference_match_many(substrate, scalar_arms, proof_grade=True)
        scalar_wall = max(0.0, time.perf_counter() - scalar_started)
        native_started = time.perf_counter()
        native_response = engine.reference_match_arms(
            native_arms,
            reference_token_ids=reference,
            generation_contract=contract,
            parent_anchor_prompt=parent_prompt,
        )
        native_wall = max(0.0, time.perf_counter() - native_started)
        native_evidence = [
            _classify_native(row, reference, contract)
            for row in native_response["results"]
        ]
        current_error = None
        current_started = time.perf_counter()
        try:
            current_response = engine.reference_match_arms(
                native_arms,
                reference_token_ids=reference,
                generation_contract=contract,
            )
            current_evidence = [
                _classify_native(row, reference, contract)
                for row in current_response["results"]
            ]
        except Exception as exc:
            current_response = {"metrics": {}}
            current_evidence = []
            current_error = str(exc)
        current_wall = max(0.0, time.perf_counter() - current_started)
        native_rows += len(native_evidence)
        scalar_rows += len(scalar_evidence)
        mismatches = [
            index for index, (native, scalar) in enumerate(zip(native_evidence, scalar_evidence))
            if _evidence_projection(native) != _evidence_projection(scalar)
        ]
        current_mismatches = [
            index for index, (current, scalar) in enumerate(zip(current_evidence, scalar_evidence))
            if _evidence_projection(current) != _evidence_projection(scalar)
        ]
        parity_mismatches.extend({
            "case_id": case["case_id"],
            "batch_id": batch["batch_id"],
            "arm_index": index,
            "strategy": "parent_anchor",
            "native": _evidence_projection(native_evidence[index]),
            "scalar": _evidence_projection(scalar_evidence[index]),
        } for index in mismatches)
        parity_mismatches.extend({
            "case_id": case["case_id"],
            "batch_id": batch["batch_id"],
            "arm_index": index,
            "strategy": "current_native",
            "native": _evidence_projection(current_evidence[index]),
            "scalar": _evidence_projection(scalar_evidence[index]),
        } for index in current_mismatches)
        batches.append({
            "batch_id": batch["batch_id"],
            "stage": batch["stage"],
            "parent_source_ids": list(parent_ids),
            "probe_ordinals": list(batch["probe_ordinals"]),
            "native_wall_seconds": round(native_wall, 6),
            "current_native_wall_seconds": round(current_wall, 6) if current_error is None else None,
            "scalar_wall_seconds": round(scalar_wall, 6),
            "speedup_vs_scalar": round(scalar_wall / native_wall, 6) if native_wall else None,
            "speedup_vs_current_native": (
                round(current_wall / native_wall, 6)
                if native_wall and current_error is None else None
            ),
            "parity": {
                "passed": not mismatches and not current_mismatches,
                "current_native_status": "ok" if current_error is None else "unavailable",
                "parent_anchor_mismatch_arm_indexes": mismatches,
                "current_native_mismatch_arm_indexes": current_mismatches,
            },
            "current_native_error": current_error,
            "current_native_metrics": dict(current_response.get("metrics") or {}),
            "native_metrics": dict(native_response.get("metrics") or {}),
        })
        if current_error is not None:
            break
    return {
        "case_id": case["case_id"],
        "run_id": case["run_id"],
        "batch_count": len(batches),
        "batch_limit": max_batches,
        "complete_case": max_batches is None or len(batches) == len(geometry_case["batches"]),
        "native_probe_count": native_rows,
        "scalar_probe_count": scalar_rows,
        "parity_passed": not any(item["case_id"] == case["case_id"] for item in parity_mismatches),
        "batches": batches,
        "parity_mismatches": [item for item in parity_mismatches if item["case_id"] == case["case_id"]],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    geometry = json.loads(Path(args.geometry).read_text())
    source = json.loads(Path(args.source_report).read_text())
    source_by_case = {str(case["case_id"]): case for case in source["cases"]}
    geometry_by_case = {str(case["case_id"]): case for case in geometry["cases"]}
    requested = args.cases or sorted(source_by_case)
    if any(case_id not in source_by_case or case_id not in geometry_by_case for case_id in requested):
        raise ValueError("requested native cases are missing from the Phase-A artifacts")

    from clozn.server import app as cs
    from clozn.server.substrates import EngineSubstrate
    from clozn_engine import EngineClient

    engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
    health = engine.health()
    substrate = EngineSubstrate(engine=engine)
    # The paired scalar arm must not accidentally select an experimental path
    # inherited from the shell environment.
    os.environ["CLOZN_ENABLE_NATIVE_REFERENCE_MATCH_ARMS"] = "0"
    os.environ["CLOZN_ENABLE_NATIVE_PARENT_ANCHOR"] = "0"
    cases = []
    for case_id in requested:
        cases.append(_case(
            source_by_case[case_id], geometry_by_case[case_id],
            engine=engine, substrate=substrate, quiet=args.quiet, started=started,
            max_batches=args.max_batches,
        ))
    parity_passed = all(bool(case["parity_passed"]) for case in cases)
    return {
        "schema_version": SCHEMA,
        "phase": "B",
        "native_execution": "experimental_parent_anchor_non_proof_grade",
        "scalar_certificate_authority": True,
        "worker": {
            "host": args.host,
            "port": args.port,
            "model": health.get("model"),
            "n_batch": health.get("n_batch"),
            "n_ubatch": health.get("n_ubatch"),
            "n_ctx": health.get("n_ctx"),
        },
        "same_frozen_batches": True,
        "coverage": "complete" if all(case["complete_case"] for case in cases) else "bounded_batch_subset",
        "parity_passed": parity_passed,
        "cases": cases,
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default=DEFAULT_GEOMETRY)
    parser.add_argument("--source-report", default="/tmp/minimal_context_scaled_live.json")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="limit each case to the first N frozen batches for a bounded parity run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        print(json.dumps({
            "status": "ok",
            "output": args.output,
            "parity_passed": result["parity_passed"],
            "case_count": len(result["cases"]),
        }, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
