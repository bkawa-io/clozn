#!/usr/bin/env python3
"""Replay captured <=32-probe reducer traces through the persistent-parent worker session.

The trace is frozen before execution: parent candidates, ordered children, and the ordinary scalar
accepted trajectory come from the saved Phase-A ledger/geometry.  The persistent executor therefore
cannot change search order.  Native rows are paired with trusted scalar rows for every child; any
classification mismatch stops the case.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import struct
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
from clozn.runs.multi_arm import probe_reference_match_many
from clozn.runs.persistent_parent import PersistentParentSessionClient, assert_scalar_parity, candidate_id
from clozn.runs.realistic_minimal_context import _render_messages_for_retained
from clozn.runs.store import get_run


SCHEMA = "clozn.persistent-parent-session-eval.v0"
DEFAULT_GEOMETRY = "/tmp/parent_anchor_geometry_v0.json"
DEFAULT_SOURCE_REPORT = "/tmp/minimal_context_scaled_live.json"
DEFAULT_OUTPUT = "/tmp/persistent_parent_session_v0.json"
DEFAULT_CASES = ("long_rag_redundant", "long_multi_turn", "code_context", "long_rag_distributed")


def _digest_token_ids(values: list[int] | tuple[int, ...]) -> str:
    # RunningSha256 in checkpoint_codec.hpp updates int32 token IDs in native byte order. The worker
    # platforms used for this experiment are little-endian; keeping this explicit makes the replay
    # comparison deterministic instead of comparing only token counts.
    return hashlib.sha256(b"".join(struct.pack("<i", int(value)) for value in values)).hexdigest()


def _projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "status", "matched_token_count", "first_divergence_index", "expected_token_id",
        "actual_token_id", "divergence_kind", "termination_match",
    )}


def _classify_native(row: Mapping[str, Any], reference: list[int], contract: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(row.get("result") or {})
    generated = raw.get("generated_token_ids")
    if not isinstance(generated, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in generated):
        raise ValueError("persistent native row omitted integer generated_token_ids")
    result = classify_reference_match(
        reference,
        generated,
        diverged=raw.get("diverged"),
        diverged_at=raw.get("diverged_at"),
        termination=raw.get("termination"),
        finish_reason=raw.get("finish_reason"),
        expected_termination=contract.get("expected_termination"),
        max_new=contract["max_new"],
    )
    result.update({
        "generated_token_ids": list(generated),
        "finish_reason": raw.get("finish_reason"),
        "termination": dict(raw.get("termination") or {}),
    })
    return result


def _case(case: Mapping[str, Any], geometry_case: Mapping[str, Any], *, engine: Any,
          substrate: Any, timeout: float, quiet: bool, max_batches: int | None) -> dict[str, Any]:
    run = get_run(str(case["run_id"]))
    if not isinstance(run, dict):
        raise ValueError(f"run {case['run_id']!r} is unavailable")
    conditions = __import__("clozn.receipts.rederive", fromlist=["with_arm_conditions"]).with_arm_conditions(run)
    contract, reason = _generation_contract_from_run(run)
    if not isinstance(contract, dict) or reason:
        raise ValueError(f"case {case['case_id']} has no exact generation contract: {reason}")
    reference = list(conditions["continuation_ids"])
    universe_ids = tuple(case["trial_ledger"][0]["retained_source_ids"])

    def messages(retained_ids: tuple[str, ...]) -> list[dict[str, str]]:
        return _render_messages_for_retained(run, universe_ids, retained_ids)

    initial_parent_ids = tuple(case["trial_ledger"][0]["retained_source_ids"])
    session = PersistentParentSessionClient(engine, tuple(reference), contract)
    session.create(engine.apply_template(messages(initial_parent_ids)))
    current_parent_ids = initial_parent_ids
    current_version = session.parent_version
    batches = list(geometry_case.get("batches") or [])
    if max_batches is not None:
        batches = batches[:max_batches]
    output_batches = []
    parity_mismatches = []
    scalar_wall = 0.0
    stateless_native_wall = 0.0
    persistent_native_wall = 0.0
    scalar_confirmation_wall = 0.0
    logical_rows = 0
    theoretical_rows = 0
    actual_rows = 0
    reused_rows = 0
    suffix_rows = 0
    promotions = 0
    started = time.perf_counter()

    try:
        for batch in batches:
            parent_ids = tuple(batch["parent_source_ids"])
            if parent_ids != current_parent_ids:
                raise ValueError(
                    f"frozen parent trajectory mismatch before batch {batch['batch_id']}: "
                    f"expected {current_parent_ids!r}, got {parent_ids!r}"
                )
            probe_rows = [
                next(row for row in geometry_case["probes"] if int(row["probe_ordinal"]) == int(ordinal))
                for ordinal in batch["probe_ordinals"]
            ]
            scalar_arms = []
            native_children = []
            for rank, probe in enumerate(probe_rows):
                child_ids = tuple(probe["child_source_ids"])
                child_messages = messages(child_ids)
                scalar_arms.append({
                    "messages": child_messages,
                    "reference_token_ids": reference,
                    "generation_contract": contract,
                    "explicit_conditions": {},
                })
                native_children.append({
                    "candidate_id": candidate_id(child_ids),
                    "candidate_rank": rank,
                    "prompt": engine.apply_template(child_messages),
                })
            scalar_started = time.perf_counter()
            scalar = probe_reference_match_many(substrate, scalar_arms, proof_grade=True)
            scalar_elapsed = max(0.0, time.perf_counter() - scalar_started)
            scalar_wall += scalar_elapsed

            native_started = time.perf_counter()
            try:
                stateless_arms = [{"arm_id": i, "prompt": child["prompt"]}
                                  for i, child in enumerate(native_children)]
                stateless_response = engine.reference_match_arms(
                    stateless_arms, reference_token_ids=reference, generation_contract=contract,
                )
                stateless_rows = [_classify_native(row, reference, contract)
                                  for row in stateless_response["results"]]
                stateless_error = None
            except Exception as exc:
                stateless_rows = []
                stateless_response = {"metrics": {}}
                stateless_error = str(exc)
            stateless_elapsed = max(0.0, time.perf_counter() - native_started)
            stateless_native_wall += stateless_elapsed

            persistent_started = time.perf_counter()
            persistent_response = session.probe_round(native_children, expected_parent_version=current_version)
            persistent_elapsed = max(0.0, time.perf_counter() - persistent_started)
            persistent_native_wall += persistent_elapsed
            persistent_rows = [_classify_native(row, reference, contract)
                               for row in persistent_response["results"]]
            try:
                assert_scalar_parity(persistent_rows, scalar)
            except Exception as exc:
                mismatches = getattr(exc, "mismatches", [{"error": str(exc)}])
                parity_mismatches.extend({
                    "case_id": case["case_id"], "batch_id": batch["batch_id"], **dict(item)
                } for item in mismatches)
                raise

            round_metrics = dict(persistent_response.get("round_metrics") or {})
            logical_rows += int(round_metrics.get("logical_child_prompt_rows", 0))
            theoretical_rows += int(round_metrics.get("theoretical_suffix_rows", 0))
            actual_rows += int(round_metrics.get("actual_child_prompt_rows_evaluated", 0))
            reused_rows += int(round_metrics.get("reused_parent_prefix_rows", 0))
            suffix_rows += int(round_metrics.get("evaluated_child_suffix_rows", 0))

            accepted_indexes = [index for index, probe in enumerate(probe_rows)
                                if bool(probe.get("accepted_as_best"))]
            promoted_candidate = None
            confirmation = None
            confirmation_elapsed = 0.0
            if accepted_indexes:
                if len(accepted_indexes) != 1:
                    raise ValueError(f"batch {batch['batch_id']} has multiple accepted candidates")
                accepted_index = accepted_indexes[0]
                accepted_child = native_children[accepted_index]
                confirmation_started = time.perf_counter()
                confirmation = probe_reference_match_many(
                    substrate, [scalar_arms[accepted_index]], proof_grade=True,
                )[0]
                confirmation_elapsed = max(0.0, time.perf_counter() - confirmation_started)
                scalar_confirmation_wall += confirmation_elapsed
                if _projection(confirmation) != _projection(scalar[accepted_index]):
                    raise ValueError(f"scalar confirmation changed for accepted candidate in batch {batch['batch_id']}")
                if confirmation.get("status") != "matched":
                    raise ValueError(f"accepted frozen candidate was not scalar-preserving in batch {batch['batch_id']}")
                promotion_response = session.promote(
                    accepted_child["candidate_id"], scalar_preserves=True,
                    native_preserves=bool(persistent_response["results"][accepted_index].get("native_preserves")),
                )
                promotions += 1
                current_version = session.parent_version
                current_parent_ids = tuple(probe_rows[accepted_index]["child_source_ids"])
                promoted_prompt_info = engine.apply_template_info(
                    messages(current_parent_ids), include_token_ids=True,
                )
                expected_digest = _digest_token_ids(tuple(promoted_prompt_info["prompt_token_ids"]))
                if promotion_response.get("parent_prompt_digest") != expected_digest:
                    raise ValueError(
                        f"promotion prompt digest mismatch in batch {batch['batch_id']}: "
                        f"worker={promotion_response.get('parent_prompt_digest')} expected={expected_digest}"
                    )
                promoted_candidate = accepted_child["candidate_id"]
            else:
                if session.parent_version != current_version:
                    raise ValueError(f"parent version changed without a frozen accepted promotion in batch {batch['batch_id']}")

            output_batches.append({
                "batch_id": batch["batch_id"],
                "stage": batch["stage"],
                "parent_source_ids": list(parent_ids),
                "probe_ordinals": list(batch["probe_ordinals"]),
                "scalar_wall_seconds": round(scalar_elapsed, 6),
                "stateless_native_wall_seconds": round(stateless_elapsed, 6),
                "persistent_native_wall_seconds": round(persistent_elapsed, 6),
                "persistent_native_plus_scalar_confirmation_wall_seconds": round(
                    persistent_elapsed + confirmation_elapsed, 6,
                ),
                "stateless_native_error": stateless_error,
                "round_metrics": round_metrics,
                "promoted_candidate_id": promoted_candidate,
                "parity_passed": True,
            })
    finally:
        session.close()

    report = session.report()
    telemetry = dict(report.get("telemetry") or {})
    return {
        "case_id": case["case_id"],
        "run_id": case["run_id"],
        "batch_count": len(output_batches),
        "complete_case": max_batches is None or len(output_batches) == len(geometry_case.get("batches") or []),
        "elapsed_seconds": round(max(0.0, time.perf_counter() - started), 6),
        "probes": sum(len(batch["probe_ordinals"]) for batch in output_batches),
        "accepted_promotions": promotions,
        "rows": {
            "logical_child_prompt_rows": logical_rows,
            "theoretical_suffix_rows": theoretical_rows,
            "actual_persistent_child_prompt_rows": actual_rows,
            "parent_prefix_rows_reused": reused_rows,
            "evaluated_child_suffix_rows": suffix_rows,
            "parent_refill_rows_after_initial_create": telemetry.get("parent_refill_rows_after_initial_create", 0),
        },
        "wall": {
            "scalar": round(scalar_wall, 6),
            "current_stateless_native": round(stateless_native_wall, 6),
            "persistent_native": round(persistent_native_wall, 6),
            "persistent_native_plus_scalar_confirm": round(persistent_native_wall + scalar_confirmation_wall, 6),
        },
        "speedup": {
            "persistent_native_vs_scalar": round(scalar_wall / persistent_native_wall, 6) if persistent_native_wall else None,
            "persistent_plus_confirm_vs_scalar": round(
                scalar_wall / (persistent_native_wall + scalar_confirmation_wall), 6
            ) if persistent_native_wall + scalar_confirmation_wall else None,
            "persistent_vs_stateless_native": round(
                stateless_native_wall / persistent_native_wall, 6
            ) if persistent_native_wall and stateless_native_wall else None,
        },
        "parity": {"passed": not parity_mismatches, "mismatches": parity_mismatches},
        "promotion_state_parity": {"passed": True, "checked": promotions},
        "session": report,
        "batches": output_batches,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    geometry = json.loads(Path(args.geometry).read_text())
    source = json.loads(Path(args.source_report).read_text())
    source_by_case = {str(case["case_id"]): case for case in source["cases"]}
    geometry_by_case = {str(case["case_id"]): case for case in geometry["cases"]}
    requested = args.cases or list(DEFAULT_CASES)
    missing = [case_id for case_id in requested if case_id not in source_by_case or case_id not in geometry_by_case]
    if missing:
        raise ValueError(f"requested cases are missing from frozen artifacts: {missing}")
    # Import the canonical app seam first.  substrates.py intentionally imports that seam for
    # late-bound server state, while app.py re-exports EngineSubstrate; importing substrates first
    # exposes the app<->substrates cycle before app.py has finished initialization.
    from clozn.server import app as _server_app
    from clozn.server.substrates import EngineSubstrate
    from clozn_engine import EngineClient

    engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
    health = engine.health()
    substrate = EngineSubstrate(engine=engine)
    cases = []
    for case_id in requested:
        if not args.quiet:
            print(f"[persistent-parent] {case_id}: replaying", file=sys.stderr, flush=True)
        cases.append(_case(
            source_by_case[case_id], geometry_by_case[case_id], engine=engine,
            substrate=substrate, timeout=args.timeout, quiet=args.quiet, max_batches=args.max_batches,
        ))
    return {
        "schema_version": SCHEMA,
        "trace": {"source_report": args.source_report, "geometry": args.geometry, "frozen": True,
                   "max_counterfactual_probes": 32, "p200_included": False},
        "worker": {key: health.get(key) for key in ("model", "n_ctx", "n_batch", "n_ubatch", "worker_generation_id")},
        "cases": cases,
        "parity_passed": all(case["parity"]["passed"] for case in cases),
        "promotion_state_parity_passed": all(case["promotion_state_parity"]["passed"] for case in cases),
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default=DEFAULT_GEOMETRY)
    parser.add_argument("--source-report", default=DEFAULT_SOURCE_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _table(result: Mapping[str, Any]) -> str:
    lines = ["case                         probes  promotions  scalar(s)  persistent(s)  p+confirm(s)  parity"]
    for case in result["cases"]:
        wall = case["wall"]
        lines.append(
            f"{case['case_id']:<28} {case['probes']:>6}  {case['accepted_promotions']:>10}  "
            f"{wall['scalar']:>9.2f}  {wall['persistent_native']:>13.2f}  "
            f"{wall['persistent_native_plus_scalar_confirm']:>12.2f}  "
            f"{'PASS' if case['parity']['passed'] else 'FAIL'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = run(args)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        print(_table(result), file=sys.stderr)
        print(json.dumps({"status": "ok", "output": args.output,
                          "parity_passed": result["parity_passed"],
                          "promotion_state_parity_passed": result["promotion_state_parity_passed"]}, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
