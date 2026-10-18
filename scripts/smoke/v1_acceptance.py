#!/usr/bin/env python3
"""Live two-feature acceptance path for the Clozn v1 product.

The harness attaches to an already running ``clozn serve`` gateway and proves
the retained journey over one fresh recorded run:

1. Context Effects records and publishes trusted source evidence.
2. Minimal Context verifies a smaller prompt under the recorded contract.
3. Saving the immutable result creates one exact child, and retries reuse it.
4. Parent, result, and child are all readable from durable storage.

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


def _poll_minimal_context(
    client: Client, *, root: str, job: dict, timeout: float,
) -> dict:
    job_id = job.get("job_id")
    _expect(isinstance(job_id, str) and job_id, "Minimal Context start omitted job_id")
    deadline = time.monotonic() + timeout
    current = job
    while current.get("state") not in {"completed", "failed", "cancelled"}:
        if time.monotonic() >= deadline:
            raise SmokeFailure(f"Minimal Context job {job_id} did not finish within {timeout:g}s")
        time.sleep(0.05)
        status, current = client.get(root + f"/jobs/{_quoted(job_id)}")
        _expect(status == 200, f"Minimal Context poll returned HTTP {status}: {current}")
    return current


def _minimal_context(
    client: Client, *, run_id: str, expected_source_ids: list[str], job_timeout: float,
) -> dict:
    root = f"/runs/{_quoted(run_id)}/minimal-context"
    status, started = client.post(root + "/jobs", {})
    _expect(status == 202, f"Minimal Context start returned HTTP {status}: {started}")
    terminal = _poll_minimal_context(client, root=root, job=started, timeout=job_timeout)
    _expect(terminal.get("state") == "completed", f"Minimal Context failed: {terminal}")
    result = terminal.get("result") if isinstance(terminal.get("result"), dict) else {}
    _expect(result.get("schema_version") == "clozn.minimal-context-search-result.v2", (
        f"Minimal Context returned no typed result: {terminal}"
    ))
    _expect(result.get("status") == "completed", f"Minimal Context did not complete: {result}")
    universe = result.get("universe") if isinstance(result.get("universe"), dict) else {}
    _expect(universe.get("source_ids") == expected_source_ids, (
        "Minimal Context did not use the same canonical source identities as Context Effects: "
        f"{universe.get('source_ids')} != {expected_source_ids}"
    ))
    best = result.get("best") if isinstance(result.get("best"), dict) else {}
    removed = best.get("removed_source_ids") if isinstance(best.get("removed_source_ids"), list) else []
    retained = best.get("retained_source_ids") if isinstance(best.get("retained_source_ids"), list) else []
    _expect(best.get("observation_status") == "exact_preserved", (
        f"Minimal Context winner is not exact-preserved: {best}"
    ))
    _expect(bool(removed), f"qualification fixture produced no verified reduction: {result}")
    _expect(set(removed).isdisjoint(retained), f"winner source sets overlap: {best}")
    _expect(set(removed) | set(retained) == set(expected_source_ids), (
        f"winner source sets do not cover the declared universe: {best}"
    ))
    reduction = result.get("reduction") if isinstance(result.get("reduction"), dict) else {}
    _expect(
        isinstance(reduction.get("retained_prompt_token_cost"), int)
        and isinstance(reduction.get("original_prompt_token_cost"), int)
        and reduction["retained_prompt_token_cost"] < reduction["original_prompt_token_cost"],
        f"winner did not reduce rendered prompt tokens: {reduction}",
    )

    result_id = result.get("result_id")
    _expect(isinstance(result_id, str) and result_id, "Minimal Context result omitted result_id")
    status, listing = client.get(root + "/results")
    rows = listing.get("results") if isinstance(listing.get("results"), list) else []
    _expect(status == 200 and any(row.get("result_id") == result_id for row in rows), (
        f"durable Minimal Context result was not listed: {listing}"
    ))
    status, reopened_result = client.get(root + f"/results/{_quoted(result_id)}")
    _expect(status == 200 and reopened_result.get("result_id") == result_id, (
        f"durable Minimal Context result could not be reopened: {reopened_result}"
    ))
    _expect(reopened_result.get("current_binding", {}).get("status") == "current", (
        f"new Minimal Context result is already stale: {reopened_result}"
    ))

    materialize_path = root + f"/results/{_quoted(result_id)}/materialize"
    first_status, first = client.post(materialize_path, {})
    second_status, second = client.post(materialize_path, {})
    _expect(first_status == 201 and first.get("state") == "completed", (
        f"Minimal Context materialization failed: HTTP {first_status} {first}"
    ))
    _expect(second_status == 201 and second.get("state") == "completed", (
        f"Minimal Context materialization retry failed: HTTP {second_status} {second}"
    ))
    child_run_id = first.get("child_run_id")
    _expect(isinstance(child_run_id, str) and child_run_id, (
        f"Minimal Context materialization omitted child_run_id: {first}"
    ))
    _expect(second.get("child_run_id") == child_run_id, (
        "Minimal Context materialization was not idempotent: "
        f"{child_run_id} != {second.get('child_run_id')}"
    ))
    parent_status, parent = client.get(f"/runs/{_quoted(run_id)}")
    child_status, child = client.get(f"/runs/{_quoted(child_run_id)}")
    _expect(parent_status == 200 and parent.get("id") == run_id, "parent Run could not be reopened")
    _expect(child_status == 200 and child.get("id") == child_run_id, "reduced child Run could not be reopened")
    _expect(child.get("parent_run_id") == run_id, f"child lost its parent binding: {child}")
    _expect(child.get("response") == parent.get("response"), (
        f"reduced child did not preserve the recorded answer exactly: {child.get('response')!r}"
    ))

    return {
        "result_id": result_id,
        "certificate": result.get("certificate"),
        "stopping_reason": result.get("stopping_reason"),
        "source_count": len(expected_source_ids),
        "retained_source_count": len(retained),
        "removed_source_count": len(removed),
        "original_prompt_tokens": reduction.get("original_prompt_token_cost"),
        "retained_prompt_tokens": reduction.get("retained_prompt_token_cost"),
        "reduction_percent": reduction.get("percent"),
        "child_run_id": child_run_id,
    }


def run(client: Client, *, model: str | None, job_timeout: float) -> dict:
    started = time.monotonic()
    effects = effects_smoke.run(client, model=model, job_timeout=job_timeout)
    minimal = _minimal_context(
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
        "minimal_context": minimal,
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
