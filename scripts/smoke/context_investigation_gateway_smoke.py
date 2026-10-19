#!/usr/bin/env python3
"""Live gateway smoke for the v1 Context Effects workflow.

The script creates a real recorded run, executes the single direct Context
Effects job, refreshes the canonical reader, and checks an answer selection.
An optional second run exercises acknowledged cancellation.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request


class SmokeFailure(RuntimeError):
    pass


class Client:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                status = int(response.status)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            status = int(exc.code)
        try:
            document = json.loads(raw.decode("utf-8", "replace") or "{}")
        except json.JSONDecodeError as exc:
            raise SmokeFailure(f"{method} {path} returned non-JSON HTTP {status}") from exc
        if not isinstance(document, dict):
            raise SmokeFailure(f"{method} {path} returned a non-object JSON document")
        return status, document

    def get(self, path: str) -> tuple[int, dict]:
        return self.request("GET", path)

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        return self.request("POST", path, body)


def _expect(condition: object, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)


def _quoted(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def _job_path(run_id: str, job_id: str = "") -> str:
    root = f"/runs/{_quoted(run_id)}/context-investigation/effects/jobs"
    return root if not job_id else f"{root}/{_quoted(job_id)}"


def _poll_job(client: Client, *, run_id: str, job: dict, timeout: float) -> dict:
    job_id = job.get("job_id")
    _expect(isinstance(job_id, str) and job_id, "Context Effects start omitted job_id")
    deadline = time.monotonic() + timeout
    current = job
    while current.get("state") not in {"completed", "failed", "cancelled"}:
        if time.monotonic() >= deadline:
            raise SmokeFailure(f"Context Effects job {job_id} did not finish within {timeout:g}s")
        time.sleep(0.05)
        status, current = client.get(_job_path(run_id, job_id))
        _expect(status == 200, f"Context Effects poll returned HTTP {status}: {current}")
    return current


def _source_fixture(nonce: str) -> str:
    paragraphs = [
        ("launch-code", "Launch code", "The launch code is ORCHID-731."),
        ("launch-site", "Launch site", "The launch site is North Annex."),
        ("launch-window", "Launch window", "The launch window begins at dawn."),
        (
            "launch-badge",
            "Launch badge",
            f"The authorized badge color is amber. Qualification nonce: {nonce}.",
        ),
    ]
    return "\n\n".join(f"## {label}\n{paragraph}" for _source_id, label, paragraph in paragraphs)


def _cancellation_source_fixture(nonce: str, source_count: int) -> str:
    _expect(source_count >= 8, "cancellation fixture needs at least eight sources")
    paragraphs = [
        (
            f"cancel-record-{index:02d}",
            f"Cancellation record {index:02d}",
            (
                f"Record {index:02d}: call sign CEDAR-{index:02d}; status verified; "
                f"qualification nonce {nonce}-{index:02d}."
            ),
        )
        for index in range(1, source_count + 1)
    ]
    return "\n\n".join(f"## {label}\n{paragraph}" for _source_id, label, paragraph in paragraphs)


def _validate_cancelled_effects(job: dict, *, reader_after: dict, source_count: int) -> dict:
    _expect(job.get("state") == "cancelled", f"Context Effects was not cancelled: {job}")
    _expect(job.get("cancel_requested") is True, "terminal job omitted acknowledged cancellation")
    _expect("result" not in job, "cancelled job exposed a completed result")
    measurement = reader_after.get("measurement")
    measurement = measurement if isinstance(measurement, dict) else {}
    completed = measurement.get("completed_source_count")
    failed = measurement.get("failed_source_count")
    unavailable = measurement.get("unavailable_source_count")
    not_measured = measurement.get("not_measured_source_count")
    counts = (completed, failed, unavailable, not_measured)
    _expect(all(isinstance(value, int) and not isinstance(value, bool) for value in counts), (
        f"post-cancel reader omitted typed source counts: {measurement}"
    ))
    _expect(sum(counts) == source_count, f"post-cancel reader source counts are inconsistent: {measurement}")
    _expect(completed < source_count, "cancelled job nevertheless measured every source")
    return {
        "measured_source_count": completed,
        "not_measured_source_count": not_measured,
        "source_count": source_count,
    }


def _create_run(client: Client, *, model: str, context: str, prompt: str,
                max_tokens: int) -> tuple[str, dict]:
    status, generated = client.post("/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer from the supplied context."},
            {"role": "user", "content": context},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    })
    run_id = generated.get("clozn_run_id")
    _expect(status == 200 and isinstance(run_id, str) and run_id, f"generation failed: {generated}")
    return run_id, generated


def run_cancellation(
    client: Client,
    *,
    model: str,
    job_timeout: float,
    source_count: int,
    cancel_delay: float,
) -> dict:
    started = time.monotonic()
    context = _cancellation_source_fixture(str(time.time_ns()), source_count)
    run_id, generated = _create_run(
        client,
        model=model,
        context=context,
        prompt=(
            "Produce a verification ledger with one numbered line for every supplied record, "
            "in order. Copy each record number, call sign, and status."
        ),
        max_tokens=384,
    )
    completion_tokens = generated.get("usage", {}).get("completion_tokens")
    _expect(isinstance(completion_tokens, int) and completion_tokens >= source_count, (
        f"cancellation fixture continuation was unexpectedly short: {completion_tokens}"
    ))

    reader_path = f"/runs/{_quoted(run_id)}/context-investigation/reader?floor_nats=0.1"
    status, reader_before = client.get(reader_path)
    _expect(status == 200, f"cancellation reader returned HTTP {status}: {reader_before}")
    universe = reader_before.get("measurement", {}).get("source_universe")
    _expect(isinstance(universe, list) and len(universe) == source_count, (
        f"cancellation reader lost exact sources: {universe}"
    ))

    status, started_job = client.post(_job_path(run_id), {"source_ids": universe})
    _expect(status == 202, f"Context Effects start failed: {started_job}")
    job_id = started_job.get("job_id")
    _expect(isinstance(job_id, str) and job_id, "Context Effects start omitted job_id")
    deadline = time.monotonic() + job_timeout
    current = started_job
    phase_seen_at = None
    while True:
        if current.get("state") in {"completed", "failed", "cancelled"}:
            raise SmokeFailure(f"cancellation fixture finished before cancellation boundary: {current}")
        phase = current.get("progress", {}).get("phase")
        if phase == "scoring_recorded_continuation":
            phase_seen_at = phase_seen_at or time.monotonic()
            if time.monotonic() - phase_seen_at >= cancel_delay:
                break
        if time.monotonic() >= deadline:
            raise SmokeFailure("cancellation fixture never entered direct scoring")
        time.sleep(0.01)
        status, current = client.get(_job_path(run_id, job_id))
        _expect(status == 200, f"cancellation poll returned HTTP {status}: {current}")

    status, cancellation = client.post(_job_path(run_id, job_id) + "/cancel", {})
    _expect(status == 200, f"Context Effects cancel returned HTTP {status}: {cancellation}")
    _expect(cancellation.get("cancel_accepted") is True, (
        f"Context Effects cancellation was not acknowledged: {cancellation}"
    ))
    terminal = _poll_job(client, run_id=run_id, job=cancellation, timeout=job_timeout)
    status, reader_after = client.get(reader_path)
    _expect(status == 200, f"post-cancel reader returned HTTP {status}: {reader_after}")
    return {
        "ok": True,
        "run_id": run_id,
        "job_id": job_id,
        "continuation_tokens": completion_tokens,
        "cancel_delay_seconds": cancel_delay,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        **_validate_cancelled_effects(
            terminal, reader_after=reader_after, source_count=source_count,
        ),
    }


def run(client: Client, *, model: str | None, job_timeout: float) -> dict:
    started = time.monotonic()
    status, health = client.get("/readyz")
    _expect(status == 200 and health.get("status") == "ok", f"gateway is not ready: {health}")
    if not model:
        status, models = client.get("/v1/models")
        rows = models.get("data") if isinstance(models.get("data"), list) else []
        model = str(rows[0].get("id")) if status == 200 and rows and isinstance(rows[0], dict) else ""
    _expect(bool(model), "gateway did not expose a model identity")

    context = _source_fixture(str(time.time_ns()))
    run_id, _generated = _create_run(
        client,
        model=model,
        context=context,
        prompt="According to the supplied facts, what is the launch code? Answer with the code only.",
        max_tokens=24,
    )
    reader_path = f"/runs/{_quoted(run_id)}/context-investigation/reader?floor_nats=0.1"
    status, reader_before = client.get(reader_path)
    _expect(status == 200, f"reader returned HTTP {status}: {reader_before}")
    universe = reader_before.get("measurement", {}).get("source_universe")
    _expect(isinstance(universe, list) and len(universe) == 4, (
        f"reader source universe did not expose all derived Markdown chunks: {universe}"
    ))

    status, job_start = client.post(_job_path(run_id), {"source_ids": universe})
    _expect(status == 202, f"Context Effects start returned HTTP {status}: {job_start}")
    job = _poll_job(client, run_id=run_id, job=job_start, timeout=job_timeout)
    _expect(job.get("state") == "completed", f"Context Effects failed: {job}")
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    _expect(
        result.get("schema_version") == "clozn.context-investigation-effects-job-result.v1",
        f"Context Effects omitted its typed result: {job}",
    )
    _expect(result.get("source_count") == len(universe), f"Context Effects lost source coverage: {result}")

    status, reader_after = client.get(reader_path)
    _expect(status == 200, f"refreshed reader returned HTTP {status}: {reader_after}")
    measured = [
        row for row in reader_after.get("sources", [])
        if isinstance(row, dict) and row.get("measurement", {}).get("status") == "measured"
    ]
    _expect(measured, f"direct measurements were not published to the reader: {reader_after}")
    source_id = measured[0].get("source_id")
    _expect(isinstance(source_id, str) and source_id, "measured source has no canonical identity")

    answer = reader_after.get("answer") if isinstance(reader_after.get("answer"), dict) else {}
    answer_text = answer.get("text")
    _expect(isinstance(answer_text, str) and answer_text, "reader omitted the recorded answer")
    status, query = client.post(
        f"/runs/{_quoted(run_id)}/context-investigation/query",
        {"answer_start": 0, "answer_end": len(answer_text)},
    )
    _expect(status == 200 and query.get("status") in {"completed", "partial"}, (
        f"locked query failed: {query}"
    ))
    _expect(query.get("selection", {}).get("text") == answer_text, "locked query changed answer text")
    return {
        "schema": "clozn.context-effects-gateway-smoke.v1",
        "ok": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "model": model,
        "run_id": run_id,
        "source_id": source_id,
        "source_ids": list(universe),
        "answer": answer_text,
        "inspect_url": (
            f"{client.base_url}/app/index.html#/inspect/{_quoted(run_id)}"
            f"?answerStart=0&answerEnd={len(answer_text)}&sourceId={_quoted(source_id)}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout")
    parser.add_argument("--job-timeout", type=float, default=120.0)
    parser.add_argument(
        "--exercise-cancellation", action="store_true",
        help="also cancel a deliberately expensive fresh direct-measurement job",
    )
    parser.add_argument("--cancellation-source-count", type=int, default=16)
    parser.add_argument("--cancellation-delay", type=float, default=0.1)
    args = parser.parse_args()
    try:
        _expect(args.cancellation_source_count >= 8, "cancellation source count must be at least 8")
        _expect(args.cancellation_delay >= 0, "cancellation delay must be non-negative")
        report = run(
            Client(args.base_url, args.timeout),
            model=args.model,
            job_timeout=args.job_timeout,
        )
        if args.exercise_cancellation:
            report["cancellation"] = run_cancellation(
                Client(args.base_url, args.timeout),
                model=str(report["model"]),
                job_timeout=args.job_timeout,
                source_count=args.cancellation_source_count,
                cancel_delay=args.cancellation_delay,
            )
    except (SmokeFailure, OSError) as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
