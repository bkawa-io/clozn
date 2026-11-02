#!/usr/bin/env python3
"""Generate and measure the compact Context Effects and Lean Context demos.

The two scenarios intentionally tell different product stories:

* ``flagship`` is a compact landing-site synthesis with a longer, fact-linked answer.
* ``lean`` is a one-relevant-record lookup where bounded Lean Context should be
  able to discard the unrelated records.

The script talks only to the public Clozn gateway. JSON progress is never mixed
with the report: status goes to stderr and the complete report goes to stdout or
an exclusively-created output file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


FLAGSHIP_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a Mars mission-planning analyst. Use only the supplied survey. "
            "Make one clear recommendation and do not invent measurements."
        ),
    },
    {
        "role": "user",
        "content": """## Mission objective
The rover must reach exposed hydrated-sulfate deposits within 2.0 km of landing. The prime mission is to collect samples that can preserve evidence of ancient groundwater.

## Operating limits
The rover can safely cross slopes up to 12 degrees. The landing site must provide at least 7.0 hours of winter sunlight and at least 90% direct-to-orbiter visibility during the first 30 sols.

## Zephyr Basin
Hydrated-sulfate outcrops begin 1.8 km from the ellipse. The traverse is mostly 4–8 degrees. Winter sunlight averages 8.4 hours and orbiter visibility is 94%. A shallow dust field covers the final 300 m; wheel slip is the main remaining risk.

## Aurora Ridge
The sulfate outcrops are only 0.5 km away, but the direct route crosses repeated 15–18 degree slopes. Winter sunlight averages 6.1 hours and orbiter visibility is 88%.

## Cinder Plain
Winter sunlight averages 10.6 hours and orbiter visibility is 97%. The terrain is nearly flat, but orbital spectroscopy found no hydrated-sulfate exposure within 6 km.

## Available mitigation
Mission control can reserve two sols for a low-speed wheel-slip survey and route the rover around the southern edge of Zephyr's dust field, adding 0.4 km to the traverse.

## Request
Write a 140–180 word landing recommendation. Open with the selected site. Explain how it meets the science, mobility, power, and communications requirements; identify the main remaining risk and its mitigation; then close by comparing the two rejected sites. Use one compact paragraph with no heading.""",
    },
]


LEAN_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a mission archive assistant. Find the exact mission ID and reproduce "
            "its briefing capsule verbatim. Do not combine or rewrite records."
        ),
    },
    {
        "role": "user",
        "content": """## RAVEN-12 · coastal drone
Briefing capsule: The drone mapped a 4.1 km methane plume above the western shelf. Winds shifted northeast at 06:20 UTC.

## ORBIT-31 · telescope calibration
Briefing capsule: The ultraviolet channel completed a dark-frame sequence. Detector temperature held at 41 K.

## MOSS-08 · greenhouse trial
Briefing capsule: Tomato bed three reached 78% humidity. The irrigation controller skipped one scheduled cycle.

## TIDAL-54 · buoy network
Briefing capsule: Buoy seven reported a 0.8 m swell and a 12-second period. Its backup battery remains at 64%.

## PINE-63 · forest sensor
Briefing capsule: Soil temperature fell 1.7 degrees overnight. No smoke particles were detected in the northern sector.

## QUILL-19 · archive scan
Briefing capsule: The scanner completed 620 pages of the polar survey ledger. Twelve pages require a second pass.

## KESTREL-4 · under-ice mission
Briefing capsule: At 02:14 UTC, KESTREL-4 detected a stable 37 Hz pulse beneath 6.2 km of ice; the signal persisted for 11 minutes. Surface accelerometers stayed below their motion threshold and the hydrophone self-test passed at 01:50 UTC, making surface vibration and instrument failure unlikely primary causes. Repeat the measurement from a stationary platform at the 03:00 UTC communications window and preserve the raw hydrophone stream before any drilling decision.

## Request
Return the three sentences in KESTREL-4's briefing capsule exactly as written, without the label or heading.""",
    },
]


class DemoFailure(RuntimeError):
    pass


def _request(
    base_url: str,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if data is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        raw = exc.read()
    try:
        result = json.loads(raw.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise DemoFailure(f"{method} {path} returned non-JSON HTTP {status}") from exc
    if not isinstance(result, dict):
        raise DemoFailure(f"{method} {path} returned a non-object")
    return status, result


def _require_status(status: int, expected: int, path: str, body: dict[str, Any]) -> None:
    if status != expected:
        raise DemoFailure(f"{path} returned HTTP {status}: {body}")


def _poll_job(
    base_url: str,
    root: str,
    job: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    job_id = job.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise DemoFailure(f"job start omitted job_id: {job}")
    deadline = time.monotonic() + timeout
    last_progress: object = None
    while job.get("state") not in {"completed", "failed", "cancelled"}:
        progress = job.get("progress")
        if progress != last_progress:
            print(f"{job_id}: {progress}", file=sys.stderr, flush=True)
            last_progress = progress
        if time.monotonic() >= deadline:
            raise DemoFailure(f"job {job_id} exceeded {timeout:g}s")
        time.sleep(0.2)
        status, job = _request(
            base_url, "GET", f"{root}/{urllib.parse.quote(job_id, safe='')}", timeout=timeout,
        )
        _require_status(status, 200, root, job)
    if job.get("state") != "completed":
        raise DemoFailure(f"job {job_id} did not complete: {job}")
    return job


def _resolve_model(base_url: str, requested: str | None, *, timeout: float) -> str:
    if requested:
        return requested
    status, document = _request(base_url, "GET", "/v1/models", timeout=timeout)
    rows = document.get("data") if isinstance(document.get("data"), list) else []
    if status != 200 or not rows or not isinstance(rows[0], dict):
        raise DemoFailure(f"could not resolve the served model: {document}")
    model = rows[0].get("id")
    if not isinstance(model, str) or not model:
        raise DemoFailure(f"served model omitted its id: {document}")
    return model


def _semantic_text_ranges(
    text: str, *, minimum: int = 24, target: int = 72, maximum: int = 120,
) -> list[tuple[int, int]]:
    """Match Studio's readable answer partition for saved demo queries."""
    if not text:
        return []
    sentence_end = set(".!?。！？")
    sentence_without_space = set("。！？")
    clause_end = set(",;:—–，；：")
    clause_without_space = set("—–，；：")
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        remaining = len(text) - start
        if remaining <= target:
            ranges.append((start, len(text)))
            break
        finish = min(len(text), start + maximum)
        strong: list[int] = []
        clauses: list[int] = []
        words: list[int] = []
        for boundary in range(start + minimum, finish + 1):
            previous = text[boundary - 1]
            following = text[boundary] if boundary < len(text) else None
            after_whitespace = previous.isspace() and (
                following is None or not following.isspace()
            )
            after_punctuation = following is None or not following.isspace()
            previous_non_whitespace = boundary - 1
            while (
                previous_non_whitespace >= start
                and text[previous_non_whitespace].isspace()
            ):
                previous_non_whitespace -= 1
            mark = text[previous_non_whitespace] if previous_non_whitespace >= start else ""
            if previous == "\n" or (
                mark in sentence_end
                and (
                    after_whitespace
                    or (after_punctuation and mark in sentence_without_space)
                )
            ):
                strong.append(boundary)
            elif mark in clause_end and (
                after_whitespace or (after_punctuation and mark in clause_without_space)
            ):
                clauses.append(boundary)
            elif after_whitespace:
                words.append(boundary)

        def nearest(values: list[int]) -> int | None:
            if not values:
                return None
            return min(values, key=lambda value: (abs(value - start - target), value))

        readable = [
            boundary for boundary in strong
            if len(text) - boundary == 0 or len(text) - boundary >= minimum
        ]
        end = nearest(readable)
        if end is None and remaining <= maximum:
            ranges.append((start, len(text)))
            break
        end = end or nearest(clauses) or nearest(words) or finish
        ranges.append((start, end))
        start = end
    if len(ranges) > 1:
        previous, tail = ranges[-2], ranges[-1]
        if tail[1] - tail[0] < minimum and tail[1] - previous[0] <= maximum:
            ranges[-2:] = [(previous[0], tail[1])]
    return ranges


def _reader_path(configuration: dict[str, Any], *, floor_nats: float) -> str:
    query: list[tuple[str, str]] = [("floor_nats", str(floor_nats))]
    for key in ("granularity", "max_arms", "scoring_mode"):
        if key in configuration:
            query.append((key, str(configuration[key])))
    return "/context-investigation/reader?" + urllib.parse.urlencode(query)


def _source_text(reader: dict[str, Any], source: dict[str, Any]) -> str:
    context = reader.get("context") if isinstance(reader.get("context"), dict) else {}
    blocks = context.get("blocks") if isinstance(context.get("blocks"), list) else []
    block = next((
        row for row in blocks
        if isinstance(row, dict) and row.get("message_index") == source.get("message_index")
    ), None)
    span = source.get("unicode_range")
    if (
        not isinstance(block, dict) or not isinstance(block.get("text"), str)
        or not isinstance(span, list) or len(span) != 2
        or not all(isinstance(value, int) for value in span)
    ):
        return ""
    return block["text"][span[0]:span[1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("flagship", "lean"), default="flagship")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--seed", type=int, default=6201)
    parser.add_argument("--scoring-mode", choices=("fast", "exact"))
    parser.add_argument("--max-arms", type=int, default=10)
    parser.add_argument("--lean-budget", type=int, default=64)
    parser.add_argument("--floor-nats", type=float, default=0.1)
    parser.add_argument("--output", type=Path, help="Exclusively create a complete JSON report.")
    parser.add_argument(
        "--fixture",
        type=Path,
        help="Exclusively create a Studio demo fixture with the reader and saved passage queries.",
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_arms <= 0 or not 1 <= args.lean_budget <= 64:
        parser.error("timeout and max-arms must be positive; lean-budget must be 1–64")
    if not math.isfinite(args.floor_nats) or args.floor_nats < 0:
        parser.error("floor-nats must be finite and non-negative")
    for path in (args.output, args.fixture):
        if path is not None and (path.exists() or not path.parent.is_dir()):
            parser.error(f"output path must be new and its parent must exist: {path}")

    model = _resolve_model(args.base_url, args.model, timeout=args.timeout)
    messages = FLAGSHIP_MESSAGES if args.scenario == "flagship" else LEAN_MESSAGES
    scoring_mode = args.scoring_mode or ("exact" if args.scenario == "flagship" else "fast")
    completion_request = {
        "model": model,
        "messages": messages,
        "max_tokens": 260 if args.scenario == "flagship" else 140,
        "temperature": 0.0,
        "seed": args.seed,
    }
    session = f"demo-{args.scenario}-{time.time_ns()}"
    status, completion = _request(
        args.base_url,
        "POST",
        "/v1/chat/completions",
        completion_request,
        timeout=args.timeout,
        headers={"X-Clozn-Client": "flagship-context-examples/1", "X-Clozn-Session": session},
    )
    _require_status(status, 200, "/v1/chat/completions", completion)
    run_id = completion.get("clozn_run_id")
    choices = completion.get("choices") if isinstance(completion.get("choices"), list) else []
    answer = (
        choices[0].get("message", {}).get("content")
        if choices and isinstance(choices[0], dict) else None
    )
    if not isinstance(run_id, str) or not run_id or not isinstance(answer, str) or not answer:
        raise DemoFailure(f"completion omitted its recorded run or answer: {completion}")
    print(f"{args.scenario}: {run_id}\n{answer}", file=sys.stderr, flush=True)

    configuration: dict[str, Any] = {"scoring_mode": scoring_mode}
    if args.scenario == "flagship":
        configuration.update({"granularity": "adaptive", "max_arms": args.max_arms})
    quoted_run = urllib.parse.quote(run_id, safe="")
    effects_root = f"/runs/{quoted_run}/context-investigation/effects/jobs"
    started = time.perf_counter()
    status, job = _request(
        args.base_url, "POST", effects_root, configuration, timeout=args.timeout,
    )
    _require_status(status, 202, effects_root, job)
    effects_job = _poll_job(args.base_url, effects_root, job, timeout=args.timeout)
    effects_seconds = time.perf_counter() - started

    reader_suffix = _reader_path(configuration, floor_nats=args.floor_nats)
    reader_path = f"/runs/{quoted_run}{reader_suffix}"
    status, reader = _request(args.base_url, "GET", reader_path, timeout=args.timeout)
    _require_status(status, 200, reader_path, reader)
    measured_ids = set((reader.get("measurement") or {}).get("source_universe") or [])
    source_rows = []
    for source in reader.get("sources") or []:
        if not isinstance(source, dict) or source.get("source_id") not in measured_ids:
            continue
        summary = source.get("summary") if isinstance(source.get("summary"), dict) else {}
        source_rows.append({
            "source_id": source.get("source_id"),
            "message_index": source.get("message_index"),
            "text": _source_text(reader, source),
            "total_continuation_delta_nats": summary.get("total_continuation_delta_nats"),
            "max_abs_token_delta_nats": summary.get("max_abs_token_delta_nats"),
        })

    queries: dict[str, dict[str, Any]] = {}
    for start, end in _semantic_text_ranges(answer):
        query_body = {
            "answer_start": start,
            "answer_end": end,
            "floor_nats": args.floor_nats,
            **configuration,
        }
        query_path = f"/runs/{quoted_run}/context-investigation/query"
        status, query = _request(
            args.base_url, "POST", query_path, query_body, timeout=args.timeout,
        )
        _require_status(status, 200, query_path, query)
        queries[f"{start}:{end}"] = query

    lean_job = None
    if args.scenario == "lean":
        lean_root = f"/runs/{quoted_run}/context-investigation/lean-context/jobs"
        status, job = _request(
            args.base_url,
            "POST",
            lean_root,
            {"max_new_executions": args.lean_budget},
            timeout=args.timeout,
        )
        _require_status(status, 202, lean_root, job)
        lean_job = _poll_job(args.base_url, lean_root, job, timeout=args.timeout)

    report = {
        "schema": "clozn.flagship-context-example-report.v1",
        "scenario": args.scenario,
        "model": model,
        "run_id": run_id,
        "messages": messages,
        "answer": answer,
        "usage": completion.get("usage"),
        "effects": {
            "configuration": configuration,
            "elapsed_seconds": effects_seconds,
            "job_result": effects_job.get("result"),
            "source_count": len(source_rows),
            "sources": source_rows,
            "answer_passages": [
                {"unicode_range": [start, end], "text": answer[start:end]}
                for start, end in _semantic_text_ranges(answer)
            ],
        },
        "lean_context": lean_job.get("result") if isinstance(lean_job, dict) else None,
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(encoded)
        print(f"report: {args.output}", file=sys.stderr)
    else:
        print(encoded, end="")
    if args.fixture is not None:
        fixture = json.dumps(
            {"reader": reader, "queries": queries},
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
        with args.fixture.open("x", encoding="utf-8") as output:
            output.write(fixture)
        print(f"fixture: {args.fixture}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
