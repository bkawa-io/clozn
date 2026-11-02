#!/usr/bin/env python3
"""Live Context Effects + Lean Context acceptance path for Clozn v1.

The harness attaches to an already running ``clozn serve`` gateway, records a
fresh run, measures Context Effects, and proves that Lean Context returns a
smaller tested prompt plus fresh full/lean generations.

Example::

    python scripts/smoke/v1_acceptance.py \
        --base-url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import time

import context_investigation_gateway_smoke as effects_smoke


Client = effects_smoke.Client
SmokeFailure = effects_smoke.SmokeFailure
_expect = effects_smoke._expect
_quoted = effects_smoke._quoted


def _poll_lean_context(
    client: Client, *, root: str, job: dict, timeout: float,
) -> dict:
    job_id = job.get("job_id")
    _expect(isinstance(job_id, str) and job_id, "Lean Context start omitted job_id")
    deadline = time.monotonic() + timeout
    current = job
    while current.get("state") not in {"completed", "failed", "cancelled"}:
        if time.monotonic() >= deadline:
            raise SmokeFailure(f"Lean Context job {job_id} did not finish within {timeout:g}s")
        time.sleep(0.05)
        status, current = client.get(root + f"/{_quoted(job_id)}")
        _expect(status == 200, f"Lean Context poll returned HTTP {status}: {current}")
    return current


def _lean_context(
    client: Client, *, run_id: str, expected_source_ids: list[str], job_timeout: float,
) -> dict:
    root = f"/runs/{_quoted(run_id)}/context-investigation/lean-context/jobs"
    status, started = client.post(root, {"max_new_executions": 16})
    _expect(status == 202, f"Lean Context start returned HTTP {status}: {started}")
    terminal = _poll_lean_context(client, root=root, job=started, timeout=job_timeout)
    _expect(terminal.get("state") == "completed", f"Lean Context failed: {terminal}")
    result = terminal.get("result") if isinstance(terminal.get("result"), dict) else {}
    _expect(result.get("schema_version") == "clozn.lean-context-result.v1", (
        f"Lean Context returned no typed result: {terminal}"
    ))
    _expect(result.get("status") == "completed", f"Lean Context did not complete: {result}")
    universe = result.get("universe") if isinstance(result.get("universe"), dict) else {}
    _expect(universe.get("source_ids") == expected_source_ids, (
        "Lean Context did not use the same canonical source identities as Context Effects: "
        f"{universe.get('source_ids')} != {expected_source_ids}"
    ))
    best = result.get("best") if isinstance(result.get("best"), dict) else {}
    removed = best.get("removed_source_ids") if isinstance(best.get("removed_source_ids"), list) else []
    retained = best.get("retained_source_ids") if isinstance(best.get("retained_source_ids"), list) else []
    _expect(bool(removed), f"qualification fixture produced no tested reduction: {result}")
    _expect(set(removed).isdisjoint(retained), f"winner source sets overlap: {best}")
    _expect(set(removed) | set(retained) == set(expected_source_ids), (
        f"winner source sets do not cover the declared universe: {best}"
    ))
    reduction = result.get("reduction") if isinstance(result.get("reduction"), dict) else {}
    original = result.get("original") if isinstance(result.get("original"), dict) else {}
    original_tokens = original.get("rendered_prompt_token_cost")
    retained_tokens = best.get("rendered_prompt_token_cost")
    _expect(
        isinstance(original_tokens, int)
        and isinstance(retained_tokens, int)
        and retained_tokens < original_tokens,
        f"winner did not reduce rendered prompt tokens: {reduction}",
    )
    comparison = (
        result.get("generation_comparison")
        if isinstance(result.get("generation_comparison"), dict)
        else {}
    )
    full = comparison.get("full_context") if isinstance(comparison.get("full_context"), dict) else {}
    lean = comparison.get("lean_context") if isinstance(comparison.get("lean_context"), dict) else {}
    _expect(comparison.get("status") == "completed", f"fresh generations failed: {comparison}")
    _expect(isinstance(full.get("text"), str) and isinstance(lean.get("text"), str), (
        f"fresh generation texts are missing: {comparison}"
    ))
    budget = result.get("budget") if isinstance(result.get("budget"), dict) else {}
    return {
        "search_id": result.get("search_id"),
        "source_count": len(expected_source_ids),
        "retained_source_count": len(retained),
        "removed_source_count": len(removed),
        "original_prompt_tokens": original_tokens,
        "retained_prompt_tokens": retained_tokens,
        "reduction_percent": (
            (100.0 * reduction.get("removed_prompt_tokens", 0) / original_tokens)
            if original_tokens else 0.0
        ),
        "used_new_executions": budget.get("used_new_executions"),
        "fresh_answers_exact_match": comparison.get("exact_match"),
    }


def run(client: Client, *, model: str | None, job_timeout: float) -> dict:
    started = time.monotonic()
    effects = effects_smoke.run(client, model=model, job_timeout=job_timeout)
    lean = _lean_context(
        client, run_id=effects["run_id"],
        expected_source_ids=list(effects["source_ids"]), job_timeout=job_timeout,
    )
    return {
        "schema": "clozn.v1-live-acceptance.v1",
        "ok": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "model": effects["model"],
        "parent_run_id": effects["run_id"],
        "answer": effects["answer"],
        "context_effects": {
            "source_id": effects["source_id"],
            "source_ids": effects["source_ids"],
        },
        "lean_context": lean,
        "inspect_url": effects["inspect_url"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout")
    parser.add_argument("--job-timeout", type=float, default=180.0)
    args = parser.parse_args()
    try:
        report = run(
            Client(args.base_url, args.timeout), model=args.model,
            job_timeout=args.job_timeout,
        )
    except (SmokeFailure, OSError) as exc:
        print(json.dumps({
            "schema": "clozn.v1-live-acceptance.v1",
            "ok": False,
            "error": str(exc),
        }, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
