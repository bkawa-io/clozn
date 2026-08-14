#!/usr/bin/env python3
"""Run a small bounded Minimal Context job through a live Clozn server.

This is intentionally separate from the synthetic combinatorial benchmark:
the server must supply the recorded Context Receipt, model routing, direct
substrate, persistence, and result reload.  It never claims a theorem that
the live result did not produce.
"""
from __future__ import annotations

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request(base_url: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    payload = None
    headers = {"Accept": "application/json"}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(base_url.rstrip("/") + path, data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            detail = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            detail = {"error": raw}
        return exc.code, detail


def build_request(args: argparse.Namespace) -> dict:
    preservation = {
        "kind": args.preservation,
        "target": "whole_recorded_continuation",
    }
    if args.preservation == "teacher_forced_likelihood":
        preservation["tolerance_nats"] = args.tolerance_nats
    return {
        "preservation": preservation,
        "universe": {"max_units": args.max_units},
        "search_probe_budget": args.search_probe_budget,
        "certification_probe_budget": args.certification_probe_budget,
        "search_seed": args.search_seed,
    }


def summarize(run: dict, result: dict, support_ids: list[str]) -> dict:
    source = result.get("source_universe") if isinstance(result.get("source_universe"), dict) else {}
    candidate = result.get("candidate") if isinstance(result.get("candidate"), dict) else {}
    certificate = result.get("certificate") if isinstance(result.get("certificate"), dict) else {}
    budget = result.get("budget") if isinstance(result.get("budget"), dict) else {}
    return {
        "run_id": run.get("id"),
        "model": run.get("model"),
        "runtime_identity": run.get("identity") if isinstance(run.get("identity"), dict) else {},
        "preservation_kind": (result.get("preservation") or {}).get("kind"),
        "source_universe_count": source.get("source_count"),
        "new_probe_count": budget.get("total_new_probes"),
        "reused_probe_count": budget.get("reused_experiments"),
        "candidate_retained_source_count": candidate.get("retained_source_count"),
        "certificate_kind": certificate.get("kind"),
        "support_study_ids": support_ids,
        "result_id": result.get("result_id"),
    }


def run(args: argparse.Namespace) -> dict:
    status, run = _request(args.base_url, "GET", f"/runs/{args.run_id}")
    if status != 200:
        raise RuntimeError(f"run lookup failed ({status}): {run}")
    body = build_request(args)
    status, job = _request(args.base_url, "POST", f"/runs/{args.run_id}/minimal-context/jobs", body)
    if status not in {200, 202}:
        raise RuntimeError(f"Minimal Context job creation failed ({status}): {job}")
    job_id = job.get("job_id")
    if not isinstance(job_id, str):
        raise RuntimeError("server did not return a job_id")
    deadline = time.monotonic() + args.timeout
    while True:
        status, snapshot = _request(
            args.base_url,
            "GET",
            f"/runs/{args.run_id}/minimal-context/jobs/{job_id}",
        )
        if status != 200:
            raise RuntimeError(f"job lookup failed ({status}): {snapshot}")
        state = snapshot.get("state")
        if state in {"completed", "failed", "cancelled"}:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Minimal Context job did not finish: {snapshot}")
        time.sleep(args.poll_interval)
    if state != "completed":
        raise RuntimeError(f"Minimal Context job ended in {state}: {snapshot}")
    result = snapshot.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("completed Minimal Context job returned no result")
    result_id = result.get("result_id")
    if not isinstance(result_id, str):
        raise RuntimeError("completed Minimal Context result has no result_id")
    status, loaded = _request(
        args.base_url,
        "GET",
        f"/runs/{args.run_id}/minimal-context/{result_id}",
    )
    if status != 200 or loaded.get("result_id") != result_id:
        raise RuntimeError(f"persisted Minimal Context result reload failed ({status}): {loaded}")
    status, reloaded_run = _request(args.base_url, "GET", f"/runs/{args.run_id}")
    if status != 200:
        raise RuntimeError(f"run reload failed ({status}): {reloaded_run}")
    support = reloaded_run.get("minimal_context_support")
    support_ids = sorted(support) if isinstance(support, dict) and "unavailable" not in support else []
    return summarize(reloaded_run, loaded, support_ids)


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--preservation",
        choices=("teacher_forced_likelihood", "exact_recorded_output"),
        default="teacher_forced_likelihood",
    )
    parser.add_argument("--tolerance-nats", type=float, default=0.3)
    parser.add_argument("--max-units", type=int, default=10)
    parser.add_argument("--search-probe-budget", type=int, default=4)
    parser.add_argument("--certification-probe-budget", type=int, default=8)
    parser.add_argument("--search-seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        print(json.dumps(run(args), ensure_ascii=False, sort_keys=True, indent=2))
    except (OSError, URLError, RuntimeError, TimeoutError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
