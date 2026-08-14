#!/usr/bin/env python3
"""Benchmark the native exact-reference arm wire against scalar probes.

The fixture mirrors ``ollama_minimal_context_baseline_v3.py`` without tags in
the source text: natural Markdown sections are unitized automatically, a
50-unit universe is declared, and the 0/1/2-retained certification workload
contains exactly 1 + 50 + 1,225 = 1,276 arms.

This harness is intentionally opt-in and requires a live Clozn GGUF worker.
It never treats the native regime as proof-grade; the parity gate is reported
separately and scalar evidence remains the certificate authority.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class Unit:
    index: int
    title: str
    text: str


@dataclass(frozen=True)
class Fixture:
    raw_document: str
    units: tuple[Unit, ...]
    question: str
    known_preserving_candidate: tuple[int, ...]


QUESTION = """Using only the operations manual in the system message, answer the Aurora-7 question.
Do not infer missing values and do not substitute values from another program.

Reply with EXACTLY one line in this format:
CAP=<digits>|WAIT=<digits>|CODE=<code>

Formatting rules:
- CAP must contain ASCII digits only: no dollar sign, comma, decimal point, or spaces.
- WAIT must contain ASCII digits only: no units or spaces.
- CODE must be copied exactly from the document.
- Output nothing before or after the line.

Example FORMAT ONLY (the values below are not answers):
CAP=123|WAIT=4|CODE=AB-12

What are Aurora-7's reimbursement cap, mandatory waiting period, and approval code?"""


def _title_for(chunk: str) -> str:
    for line in chunk.splitlines():
        line = line.strip()
        if line.startswith("## "):
            return line[3:][:100]
    return "(empty)"


def derive_units(text: str, max_units: int = 50) -> tuple[Unit, ...]:
    """Derive units from natural Markdown headings, with deterministic bounds."""
    import re

    heads = list(re.finditer(r"(?m)^#{2,6}\s+\S.*$", text))
    if len(heads) < 2:
        raise ValueError("fixture did not produce structured sections")
    chunks = []
    for index, match in enumerate(heads):
        start = 0 if index == 0 else match.start()
        end = heads[index + 1].start() if index + 1 < len(heads) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
    while len(chunks) > max_units:
        pair = min(range(len(chunks) - 1), key=lambda i: (len(chunks[i]) + len(chunks[i + 1]), i))
        chunks[pair:pair + 2] = [chunks[pair] + "\n\n" + chunks[pair + 1]]
    return tuple(Unit(i, _title_for(chunk), chunk) for i, chunk in enumerate(chunks))


def build_fixture(section_count: int = 50) -> Fixture:
    if section_count != 50:
        raise ValueError("the parity fixture is intentionally fixed at 50 units")
    cap_index = round(section_count * 0.22)
    wait_index = max(cap_index + 2, round(section_count * 0.58))
    code_index = min(section_count, max(wait_index + 2, round(section_count * 0.86)))
    programs = [
        "Borealis-2", "Cinder-4", "Delta-9", "Ember-6", "Falcon-3",
        "Granite-5", "Harbor-8", "Ion-4", "Juniper-2", "Kestrel-6",
    ]
    filler = [
        "Staff must use the program identifier from the submitted claim rather than a nearby account name.",
        "Records received through the standard portal should be attached to the active case before review begins.",
        "A reviewer may request supplemental documentation when dates, identities, or covered events are ambiguous.",
        "Administrative notes do not change monetary limits stated in a program-specific rule.",
        "Examples illustrate workflow only and are not substitutes for program-specific values.",
        "Amounts belonging to another program must not be copied merely because claim categories look similar.",
        "The audit trail should preserve the received value, reviewed value, and reason for any correction.",
        "If a required program fact is absent, the correct action is escalation rather than inference.",
    ]
    sections = []
    for index in range(1, section_count + 1):
        program = programs[(index - 1) % len(programs)]
        title = f"## {index:02d}. {program} — Operations Rule"
        body = (
            f"For {program}, the reimbursement ceiling is ${1200 + ((index * 317) % 7600):,}; "
            f"the standard review delay is {3 + ((index * 7) % 31)} calendar days; "
            f"the routing reference is {chr(65 + (index % 20))}{chr(65 + ((index * 3) % 20))}-"
            f"{100 + index}. These values apply only to {program}."
        )
        if index == cap_index:
            title = f"## {index:02d}. Aurora-7 — Reimbursement Limit"
            body = (
                "For the Aurora-7 program, the reimbursement cap is exactly $4,700. "
                "This cap is program-specific and must not be borrowed from another program."
            )
        elif index == wait_index:
            title = f"## {index:02d}. Aurora-7 — Waiting Period"
            body = (
                "For an Aurora-7 claim, the mandatory waiting period is exactly 19 calendar days. "
                "This value is independent of intake priority and reimbursement amount."
            )
        elif index == code_index:
            title = f"## {index:02d}. Aurora-7 — Approval Routing"
            body = (
                "The approval code for Aurora-7 is VX-31. VX-31 is the program approval code, "
                "not a queue name, vendor identifier, or example code."
            )
        sections.append(f"{title}\n\n{body}\n\n{' '.join(filler)}")
    raw = (
        "# Meridian Claims Operations Manual\n\n"
        "This manual contains independent program rules. Use only the rule for the program named "
        "by the question. Values from one program never override another program.\n\n"
        + "\n\n".join(sections)
    )
    units = derive_units(raw)
    if len(units) != 50:
        raise AssertionError(f"fixture unitizer produced {len(units)} units")
    return Fixture(raw, units, QUESTION, (cap_index - 1, wait_index - 1, code_index - 1))


def render(units: tuple[Unit, ...], retained: tuple[int, ...]) -> str:
    keep = set(retained)
    return "\n\n".join(unit.text for unit in units if unit.index in keep)


def certification_sets(unit_count: int = 50) -> list[tuple[int, ...]]:
    return [
        tuple(),
        *itertools.combinations(range(unit_count), 1),
        *itertools.combinations(range(unit_count), 2),
    ]


def _contract(max_new: int, termination: dict) -> dict:
    return {
        "decode_mode": "greedy",
        "sampling": None,
        "max_new": max_new,
        "stop": [],
        "expected_termination": {
            "reason": termination.get("reason", termination.get("kind", "stop")),
            "reason_raw": termination.get("reason_raw", termination.get("kind", "stop")),
        },
    }


def _arms(fixture: Fixture, retained_sets: list[tuple[int, ...]]) -> list[dict]:
    return [{
        "messages": [
            {"role": "system", "content": render(fixture.units, retained)},
            {"role": "user", "content": fixture.question},
        ],
    } for retained in retained_sets]


def _evidence_projection(row: dict) -> dict:
    return {key: row.get(key) for key in (
        "status", "matched_token_count", "first_divergence_index", "expected_token_id",
        "actual_token_id", "divergence_kind", "termination_match", "divergence_kind",
        "termination", "finish_reason", "generated_token_ids",
    )}


def run(args: argparse.Namespace) -> dict:
    fixture = build_fixture()
    sets = certification_sets()
    requested = sets[:args.max_arms]
    if args.max_arms > len(sets):
        raise ValueError("max-arms exceeds the fixed 1,276-arm certification workload")
    if not args.live:
        return {
            "fixture": {
                "raw_document_characters": len(fixture.raw_document),
                "unit_count": len(fixture.units),
                "known_preserving_candidate": list(fixture.known_preserving_candidate),
            },
            "certification_workload": {
                "retained_cardinality_0": 1,
                "retained_cardinality_1": 50,
                "retained_cardinality_2": 1225,
                "total": len(sets),
            },
            "status": "fixture_only",
        }

    from clozn.server.substrates import EngineSubstrate
    from clozn.runs.multi_arm import probe_reference_match_many
    try:
        from clozn_engine import EngineClient
    except ImportError:
        client_root = ROOT / "engine" / "client"
        sys.path.insert(0, str(client_root))
        from clozn_engine import EngineClient

    engine = EngineClient(host=args.host, port=args.port, timeout=args.timeout)
    health = engine.health()
    if not health.get("capabilities", {}).get("reference_match_arms"):
        raise RuntimeError("worker does not advertise capabilities.reference_match_arms")
    sub = EngineSubstrate(engine=engine)
    full_messages = _arms(fixture, [tuple(range(len(fixture.units)))])[0]["messages"]
    rendered = engine.apply_template_info(full_messages)
    baseline = engine.complete(rendered["prompt"], max_tokens=args.max_new,
                               temperature=0.0, rep_penalty=1.0, top_k=0, top_p=1.0, seed=0)
    prompt_tokens = rendered.get("prompt_tokens")
    board = baseline.get("board")
    if not isinstance(prompt_tokens, int) or not isinstance(board, list):
        raise RuntimeError("worker baseline omitted prompt token count or board token IDs")
    reference = [int(token) for token in board[prompt_tokens:]]
    if not reference:
        raise RuntimeError("worker baseline generated no reference tokens")
    termination = dict(baseline.get("termination") or {})
    contract = _contract(len(reference) + 1, termination)
    arms = _arms(fixture, requested)
    arms = [{**arm, "reference_token_ids": reference, "generation_contract": contract,
             "explicit_conditions": {}} for arm in arms]

    parity_requested = arms[:min(args.parity_arms, len(arms))]
    scalar_started = time.perf_counter_ns()
    # Use the existing public batch seam so this measurement includes the
    # production bounded concurrent scalar scheduler, not a special sequential
    # benchmark-only loop.
    scalar = probe_reference_match_many(sub, arms)
    scalar_wall = time.perf_counter_ns() - scalar_started

    import os
    old_flag = os.environ.get("CLOZN_ENABLE_NATIVE_REFERENCE_MATCH_ARMS")
    os.environ["CLOZN_ENABLE_NATIVE_REFERENCE_MATCH_ARMS"] = "1"
    try:
        native_started = time.perf_counter_ns()
        native = probe_reference_match_many(sub, parity_requested, proof_grade=False)
        native_wall = time.perf_counter_ns() - native_started
        parity_equal = all(
            _evidence_projection(scalar[index]) == _evidence_projection(native[index])
            for index in range(len(parity_requested))
        )
        native_metrics = dict(sub.last_native_reference_match_metrics or {})
        if not parity_equal:
            return {
                "status": "parity_failed",
                "proof_grade": False,
                "parity_gate": False,
                "parity_arms": len(parity_requested),
                "first_mismatch": next((i for i in range(len(parity_requested))
                                         if _evidence_projection(scalar[i]) !=
                                         _evidence_projection(native[i])), None),
                "execution_regime": "native_batched_non_proof_grade",
            }

        full_native = None
        full_native_wall = None
        if args.max_arms > len(parity_requested):
            full_started = time.perf_counter_ns()
            full_native = probe_reference_match_many(sub, arms, proof_grade=False)
            full_native_wall = time.perf_counter_ns() - full_started
            native_metrics = dict(sub.last_native_reference_match_metrics or native_metrics)
    finally:
        if old_flag is None:
            os.environ.pop("CLOZN_ENABLE_NATIVE_REFERENCE_MATCH_ARMS", None)
        else:
            os.environ["CLOZN_ENABLE_NATIVE_REFERENCE_MATCH_ARMS"] = old_flag

    return {
        "status": "ok",
        "proof_grade": False,
        "parity_gate": True,
        "fixture": {
            "raw_document_characters": len(fixture.raw_document),
            "unit_count": len(fixture.units),
            "known_preserving_candidate": list(fixture.known_preserving_candidate),
        },
        "certification_workload": {
            "retained_cardinality_0": 1,
            "retained_cardinality_1": 50,
            "retained_cardinality_2": 1225,
            "total": len(sets),
            "measured": len(full_native) if full_native is not None else len(parity_requested),
        },
        "scalar": {
            "wall_time_ns": scalar_wall,
            "arms_per_second": len(arms) / (scalar_wall / 1e9) if scalar_wall else None,
            "execution_regime": "scalar_bounded_concurrent",
        },
        "native_many": {
            "wall_time_ns": full_native_wall if full_native_wall is not None else native_wall,
            "arms_per_second": len(full_native or parity_requested) /
                               ((full_native_wall or native_wall) / 1e9)
                               if (full_native_wall or native_wall) else None,
            "metrics": native_metrics,
            "execution_regime": "native_batched_non_proof_grade",
        },
        "native_prefix_reuse": {"status": "not_implemented_in_this_wave", "proof_grade": False},
        "evidence_equal_on_parity_suite": True,
        "certificate_authority": "scalar_direct_reference_match",
    }


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="run against a live clozn-server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-new", type=int, default=64)
    parser.add_argument("--max-arms", type=int, default=1276)
    parser.add_argument("--parity-arms", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        print(json.dumps(run(parser().parse_args(argv)), ensure_ascii=False, sort_keys=True, indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
