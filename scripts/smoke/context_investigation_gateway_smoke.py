#!/usr/bin/env python3
"""Live public-gateway smoke for the Studio Context Investigation workflow.

This script attaches to an already running ``clozn serve`` gateway.  It uses a
real model to create a source-annotated Run, measures context effects, refreshes
the canonical reader/query projection, exercises both source actions, and
proves counterfactual materialization is idempotent.

Example::

    python scripts/smoke/context_investigation_gateway_smoke.py \
        --base-url http://127.0.0.1:8080 --native --exercise-cancellation

The browser half of the release gate should open the reported ``inspect_url``
and verify the explicit action/confirmation controls.  This harness owns the
repeatable backend half and prints a machine-readable qualification report.

``--exercise-cancellation`` creates a second, deliberately expensive Run.  It
waits until coarse scoring is active, requests cancellation, and accepts the
gate only when the terminal result proves that real model work happened, some
planned work was skipped, and the canonical reader remained unchanged.
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


def _job_path(run_id: str, family: str, job_id: str = "") -> str:
    root = f"/runs/{_quoted(run_id)}/context-investigation/{family}/jobs"
    return root if not job_id else f"{root}/{_quoted(job_id)}"


def _poll_job(
    client: Client,
    *,
    run_id: str,
    family: str,
    job: dict,
    timeout: float,
) -> dict:
    job_id = job.get("job_id")
    _expect(isinstance(job_id, str) and job_id, f"{family} start omitted job_id")
    deadline = time.monotonic() + timeout
    current = job
    while current.get("state") not in {"completed", "failed", "cancelled"}:
        if time.monotonic() >= deadline:
            raise SmokeFailure(f"{family} job {job_id} did not finish within {timeout:g}s")
        time.sleep(0.05)
        status, current = client.get(_job_path(run_id, family, job_id))
        _expect(status == 200, f"{family} poll returned HTTP {status}: {current}")
    return current


def _source_fixture(nonce: str) -> tuple[str, list[dict]]:
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
    text = "\n".join(paragraph for _source_id, _label, paragraph in paragraphs)
    sources = []
    cursor = 0
    for source_id, label, paragraph in paragraphs:
        start = text.index(paragraph, cursor)
        end = start + len(paragraph)
        sources.append({
            "message_index": 0,
            "source_id": source_id,
            "label": label,
            "provenance_kind": "retrieved_document",
            "unicode_range": [start, end],
        })
        cursor = end
    return text, sources


def _cancellation_source_fixture(nonce: str, source_count: int) -> tuple[str, list[dict]]:
    """Build enough exact, related spans to keep a real scoring job in flight."""
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
    text = "\n".join(paragraph for _source_id, _label, paragraph in paragraphs)
    sources = []
    cursor = 0
    for source_id, label, paragraph in paragraphs:
        start = text.index(paragraph, cursor)
        end = start + len(paragraph)
        sources.append({
            "message_index": 0,
            "source_id": source_id,
            "label": label,
            "provenance_kind": "retrieved_document",
            "unicode_range": [start, end],
        })
        cursor = end
    return text, sources


def _reader_publication_snapshot(reader: dict) -> dict:
    """Retain only canonical evidence-publication fields from a reader."""
    measurement = reader.get("measurement")
    measurement = measurement if isinstance(measurement, dict) else {}
    sources = reader.get("sources") if isinstance(reader.get("sources"), list) else []
    source_measurements = {}
    for row in sources:
        if not isinstance(row, dict) or not isinstance(row.get("source_id"), str):
            continue
        item = row.get("measurement") if isinstance(row.get("measurement"), dict) else {}
        source_measurements[row["source_id"]] = {
            key: item.get(key)
            for key in ("status", "experiment_id", "arm_id", "observation_id")
        }
    return {
        "status": reader.get("status"),
        "measurement": {
            key: measurement.get(key)
            for key in (
                "status", "experiment_id", "baseline_observation_id",
                "completed_source_count", "unavailable_source_count",
                "failed_source_count", "not_measured_source_count",
            )
        },
        "sources": source_measurements,
    }


def _validate_cancelled_search(
    job: dict, *, reader_before: dict, reader_after: dict, planned_measurements: int,
) -> dict:
    """Validate cancellation as a measured boundary, not merely an HTTP acknowledgement."""
    _expect(job.get("state") == "cancelled", f"effects-search was not cancelled: {job}")
    _expect(job.get("cancel_requested") is True, "terminal job omitted acknowledged cancellation")
    result = job.get("result") if isinstance(job.get("result"), dict) else {}
    _expect(
        result.get("schema_version")
        == "clozn.context-investigation-effects-search-job-result.v1",
        f"cancelled effects-search omitted its typed result: {job}",
    )
    projection = result.get("reader_projection")
    projection = projection if isinstance(projection, dict) else {}
    _expect(projection.get("status") == "not_published_cancelled", (
        f"cancelled effects-search claimed reader publication: {projection}"
    ))
    _expect(projection.get("control_observation_id") is None, (
        f"cancelled effects-search exposed a reader control: {projection}"
    ))
    _expect(projection.get("trusted_singleton_source_ids") == [], (
        f"cancelled effects-search exposed trusted singleton evidence: {projection}"
    ))

    budget_usage = result.get("budget_usage")
    budget_usage = budget_usage if isinstance(budget_usage, dict) else {}
    actual = budget_usage.get("actual_work")
    actual = actual if isinstance(actual, dict) else {}
    executed = actual.get("model_measurements_executed_total")
    skipped = actual.get("model_measurements_skipped_total")
    _expect(isinstance(executed, int) and not isinstance(executed, bool) and executed > 0, (
        f"cancellation happened before any real model measurement: {actual}"
    ))
    _expect(isinstance(skipped, int) and not isinstance(skipped, bool) and skipped > 0, (
        f"cancellation did not retain skipped-work accounting: {actual}"
    ))
    _expect(executed < planned_measurements, (
        f"cancellation arrived after all {planned_measurements} planned measurements: {actual}"
    ))
    saved = actual.get("saved_work_breakdown")
    saved = saved if isinstance(saved, dict) else {}
    _expect(saved.get("scheduler_or_cancellation_skip") == skipped, (
        f"cancellation skip accounting is inconsistent: {actual}"
    ))
    _expect(_reader_publication_snapshot(reader_after) == _reader_publication_snapshot(reader_before), (
        "canonical reader changed after acknowledged cancellation"
    ))
    return {
        "model_measurements_executed": executed,
        "model_measurements_skipped": skipped,
        "planned_measurements": planned_measurements,
        "reader_projection_status": projection.get("status"),
    }


def run_cancellation(
    client: Client,
    *,
    model: str,
    job_timeout: float,
    source_count: int,
    cancel_delay: float,
) -> dict:
    """Exercise acknowledged cancellation after real model work has started."""
    started = time.monotonic()
    context, source_metadata = _cancellation_source_fixture(
        str(time.time_ns()), source_count,
    )
    status, generated = client.post("/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content": context},
            {
                "role": "user",
                "content": (
                    "Produce a verification ledger with one numbered line for every supplied "
                    "record, in order. Copy each record number, call sign, and status. Do not "
                    "summarize or omit records."
                ),
            },
        ],
        "clozn_sources": source_metadata,
        "max_tokens": 384,
        "temperature": 0,
        "clozn_receipt": False,
    })
    run_id = generated.get("clozn_run_id")
    _expect(status == 200 and isinstance(run_id, str) and run_id, (
        f"cancellation fixture generation failed: {generated}"
    ))
    completion_tokens = generated.get("usage", {}).get("completion_tokens")
    _expect(
        isinstance(completion_tokens, int) and completion_tokens >= source_count,
        f"cancellation fixture continuation was unexpectedly short: {completion_tokens}",
    )

    reader_path = f"/runs/{_quoted(run_id)}/context-investigation/reader?floor_nats=0.1"
    status, reader_before = client.get(reader_path)
    _expect(status == 200, f"cancellation reader returned HTTP {status}: {reader_before}")
    universe = reader_before.get("measurement", {}).get("source_universe")
    _expect(isinstance(universe, list) and len(universe) == source_count, (
        f"cancellation reader lost exact sources: {universe}"
    ))
    _expect(reader_before.get("measurement", {}).get("status") == "not_measured", (
        f"fresh cancellation fixture already carried reader evidence: {reader_before}"
    ))

    drilldowns = universe[: min(8, len(universe))]
    # Keep the coarse experiment structurally distinct from the canonical
    # all-singleton reader plan.  Its durable partial accounting can then be
    # inspected without itself becoming reader-visible evidence.
    coarse_group_size = 2
    coarse_arm_count = (len(universe) + coarse_group_size - 1) // coarse_group_size
    status, search_start = client.post(_job_path(run_id, "effects-search"), {
        "source_ids": universe,
        "coarse_group_size": coarse_group_size,
        "refine_threshold_nats": 0,
        "drilldown_source_ids": drilldowns,
        "budget": {
            "max_coarse_arms": coarse_arm_count,
            "max_fine_arms": len(drilldowns),
            "max_joint_arms": 0,
        },
        "execution_policy": "trusted_scalar_only",
    })
    _expect(status == 202, f"cancellation effects-search start failed: {search_start}")
    job_id = search_start.get("job_id")
    _expect(isinstance(job_id, str) and job_id, "cancellation effects-search omitted job_id")

    scoring_phases = {
        "coarse_scalar_scoring", "coarse_control_and_native_screening",
    }
    deadline = time.monotonic() + job_timeout
    current = search_start
    phase_seen_at = None
    while True:
        state = current.get("state")
        if state in {"completed", "failed", "cancelled"}:
            raise SmokeFailure(
                f"cancellation fixture finished before cancellation boundary: {current}"
            )
        phase = current.get("progress", {}).get("phase")
        if phase in scoring_phases:
            phase_seen_at = phase_seen_at or time.monotonic()
            if time.monotonic() - phase_seen_at >= cancel_delay:
                break
        if time.monotonic() >= deadline:
            raise SmokeFailure("cancellation fixture never entered coarse scoring")
        time.sleep(0.01)
        status, current = client.get(_job_path(run_id, "effects-search", job_id))
        _expect(status == 200, f"cancellation poll returned HTTP {status}: {current}")

    cancel_path = _job_path(run_id, "effects-search", job_id) + "/cancel"
    status, cancellation = client.post(cancel_path, {})
    _expect(status == 200, f"effects-search cancel returned HTTP {status}: {cancellation}")
    _expect(cancellation.get("cancel_accepted") is True, (
        f"effects-search cancellation was not acknowledged: {cancellation}"
    ))
    terminal = _poll_job(
        client, run_id=run_id, family="effects-search", job=cancellation,
        timeout=job_timeout,
    )
    status, reader_after = client.get(reader_path)
    _expect(status == 200, f"post-cancel reader returned HTTP {status}: {reader_after}")
    planned_measurements = 1 + coarse_arm_count + len(drilldowns)
    accounting = _validate_cancelled_search(
        terminal,
        reader_before=reader_before,
        reader_after=reader_after,
        planned_measurements=planned_measurements,
    )
    return {
        "ok": True,
        "run_id": run_id,
        "job_id": job_id,
        "source_count": len(universe),
        "continuation_tokens": completion_tokens,
        "cancel_delay_seconds": cancel_delay,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        **accounting,
    }


def run(client: Client, *, model: str | None, job_timeout: float, native: bool) -> dict:
    started = time.monotonic()
    status, health = client.get("/readyz")
    _expect(status == 200 and health.get("status") == "ok", f"gateway is not ready: {health}")
    capabilities = health.get("capabilities") if isinstance(health.get("capabilities"), dict) else {}
    if native:
        _expect(capabilities.get("score_variants") is True, "worker does not advertise score_variants")

    if not model:
        status, models = client.get("/v1/models")
        rows = models.get("data") if isinstance(models.get("data"), list) else []
        model = str(rows[0].get("id")) if status == 200 and rows and isinstance(rows[0], dict) else ""
    _expect(bool(model), "gateway did not expose a model identity")

    # A unique irrelevant fact prevents a previous qualification run's trusted
    # observations from satisfying this run entirely from the content-addressed
    # cache.  The smoke must prove that the selected worker executes now.
    context, source_metadata = _source_fixture(str(time.time_ns()))
    status, generated = client.post("/v1/chat/completions", {
        "model": model,
        "messages": [
            {"role": "system", "content": context},
            {
                "role": "user",
                "content": (
                    "According to the supplied facts, what is the launch code? "
                    "Answer with the code only."
                ),
            },
        ],
        "clozn_sources": source_metadata,
        "max_tokens": 24,
        "temperature": 0,
        "clozn_receipt": False,
    })
    run_id = generated.get("clozn_run_id")
    _expect(status == 200 and isinstance(run_id, str) and run_id, f"generation failed: {generated}")

    reader_path = f"/runs/{_quoted(run_id)}/context-investigation/reader?floor_nats=0.1"
    status, reader_before = client.get(reader_path)
    _expect(status == 200, f"reader returned HTTP {status}: {reader_before}")
    universe = reader_before.get("measurement", {}).get("source_universe")
    _expect(isinstance(universe, list) and len(universe) == len(source_metadata), (
        f"reader source universe did not preserve all exact sources: {universe}"
    ))
    source_rows = reader_before.get("sources") if isinstance(reader_before.get("sources"), list) else []
    launch = next(
        (row for row in source_rows if isinstance(row, dict) and row.get("label") == "Launch code"),
        None,
    )
    source_id = launch.get("source_id") if isinstance(launch, dict) else None
    _expect(isinstance(source_id, str) and source_id in universe, "launch-code source was not canonicalized")

    execution_policy = "native_screen_then_scalar_refine" if native else "trusted_scalar_only"
    status, search_start = client.post(_job_path(run_id, "effects-search"), {
        "source_ids": universe,
        "coarse_group_size": 1,
        "refine_threshold_nats": 0,
        "drilldown_source_ids": [source_id],
        "budget": {
            "max_coarse_arms": len(universe),
            "max_fine_arms": 1,
            "max_joint_arms": 0,
        },
        "execution_policy": execution_policy,
    })
    _expect(status == 202, f"effects-search start returned HTTP {status}: {search_start}")
    search_job = _poll_job(
        client, run_id=run_id, family="effects-search", job=search_start, timeout=job_timeout,
    )
    _expect(search_job.get("state") == "completed", f"effects-search failed: {search_job}")
    search_result = search_job.get("result") if isinstance(search_job.get("result"), dict) else {}
    arms = search_result.get("arms") if isinstance(search_result.get("arms"), list) else []
    trusted = [row for row in arms if isinstance(row, dict) and row.get("evidence_status") == "trusted"]
    _expect(trusted, f"effects-search produced no trusted refinement: {search_result}")
    if native:
        screened = [
            row for row in arms
            if isinstance(row, dict) and row.get("numerical_regime") == "screening_unqualified"
        ]
        _expect(screened, f"native screening was requested but no native screening row survived: {arms}")

    status, reader_after = client.get(reader_path)
    _expect(status == 200, f"refreshed reader returned HTTP {status}: {reader_after}")
    measured = next(
        (row for row in reader_after.get("sources", []) if row.get("source_id") == source_id),
        {},
    )
    _expect(measured.get("measurement", {}).get("status") == "measured", (
        f"trusted singleton was not published to the reader: {measured}"
    ))

    answer = reader_after.get("answer") if isinstance(reader_after.get("answer"), dict) else {}
    answer_text = answer.get("text")
    _expect(isinstance(answer_text, str) and answer_text, "reader omitted the recorded answer")
    status, query = client.post(
        f"/runs/{_quoted(run_id)}/context-investigation/query",
        {"answer_start": 0, "answer_end": len(answer_text)},
    )
    _expect(
        status == 200 and query.get("status") in {"completed", "partial"},
        f"locked query failed: {query}",
    )
    _expect(query.get("selection", {}).get("text") == answer_text, "locked query changed answer text")

    status, remove_start = client.post(_job_path(run_id, "remove-test"), {"source_id": source_id})
    _expect(status == 202, f"remove-test start returned HTTP {status}: {remove_start}")
    remove_job = _poll_job(
        client, run_id=run_id, family="remove-test", job=remove_start, timeout=job_timeout,
    )
    _expect(remove_job.get("state") == "completed", f"remove-test worker failed: {remove_job}")
    remove_result = remove_job.get("result") if isinstance(remove_job.get("result"), dict) else {}
    _expect(
        remove_result.get("schema_version") == "clozn.context-investigation-remove-test-result.v1",
        f"remove-test omitted its typed result: {remove_job}",
    )

    status, counterfactual_start = client.post(
        _job_path(run_id, "counterfactuals"), {"source_id": source_id},
    )
    _expect(status == 202, f"counterfactual start returned HTTP {status}: {counterfactual_start}")
    counterfactual_job = _poll_job(
        client, run_id=run_id, family="counterfactuals", job=counterfactual_start,
        timeout=job_timeout,
    )
    _expect(counterfactual_job.get("state") == "completed", (
        f"counterfactual worker failed: {counterfactual_job}"
    ))
    counterfactual = (
        counterfactual_job.get("result")
        if isinstance(counterfactual_job.get("result"), dict) else {}
    )
    _expect(counterfactual.get("status") == "completed", (
        f"counterfactual produced no generated observation: {counterfactual}"
    ))
    materialization_request = {
        "experiment_id": counterfactual.get("experiment_id"),
        "arm_id": counterfactual.get("arm_id"),
        "observation_id": counterfactual.get("observation_id"),
    }
    _expect(all(isinstance(value, str) and value for value in materialization_request.values()), (
        f"counterfactual omitted materialization identity: {counterfactual}"
    ))
    materialize_path = (
        f"/runs/{_quoted(run_id)}/context-investigation/counterfactuals/materialize"
    )
    status, first_child = client.post(materialize_path, materialization_request)
    _expect(status == 201 and first_child.get("state") == "completed", (
        f"materialization failed: HTTP {status} {first_child}"
    ))
    status, second_child = client.post(materialize_path, materialization_request)
    _expect(status == 201 and second_child.get("state") == "completed", (
        f"repeated materialization failed: HTTP {status} {second_child}"
    ))
    child_run_id = first_child.get("child_run_id")
    _expect(
        isinstance(child_run_id, str) and child_run_id and child_run_id != run_id,
        f"materialization did not create an immutable child: {first_child}",
    )
    _expect(second_child.get("child_run_id") == child_run_id, (
        f"materialization was not idempotent: {child_run_id} != {second_child.get('child_run_id')}"
    ))
    status, child_run = client.get(f"/runs/{_quoted(child_run_id)}")
    _expect(status == 200 and child_run.get("id") == child_run_id, "materialized child is unreadable")

    inspect_url = (
        f"{client.base_url}/app/index.html#/inspect/{_quoted(run_id)}"
        f"?answerStart=0&answerEnd={len(answer_text)}&sourceId={_quoted(source_id)}"
    )
    return {
        "schema": "clozn.context-investigation-gateway-smoke.v1",
        "ok": True,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "model": model,
        "run_id": run_id,
        "source_id": source_id,
        "answer": answer_text,
        "execution_policy": execution_policy,
        "screening_regimes": sorted({
            row.get("numerical_regime") for row in arms
            if isinstance(row, dict) and isinstance(row.get("numerical_regime"), str)
        }),
        "remove_test_status": remove_result.get("status"),
        "counterfactual_text": counterfactual.get("generated_text"),
        "child_run_id": child_run_id,
        "compare_path": first_child.get("compare_path"),
        "inspect_url": inspect_url,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=float, default=120.0, help="per-request timeout")
    parser.add_argument("--job-timeout", type=float, default=120.0)
    parser.add_argument(
        "--native", action="store_true",
        help="require native screening followed by exact scalar refinement",
    )
    parser.add_argument(
        "--exercise-cancellation", action="store_true",
        help="also cancel a deliberately expensive fresh effects-search after scoring begins",
    )
    parser.add_argument(
        "--cancellation-source-count", type=int, default=24,
        help="number of exact sources in the cancellation fixture (minimum 8)",
    )
    parser.add_argument(
        "--cancellation-delay", type=float, default=0.1,
        help="seconds to leave coarse scoring active before requesting cancellation",
    )
    args = parser.parse_args()
    try:
        _expect(args.cancellation_source_count >= 8, "cancellation source count must be at least 8")
        _expect(args.cancellation_delay >= 0, "cancellation delay must be non-negative")
        report = run(
            Client(args.base_url, args.timeout),
            model=args.model,
            job_timeout=args.job_timeout,
            native=args.native,
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
        print(json.dumps({
            "schema": "clozn.context-investigation-gateway-smoke.v1",
            "ok": False,
            "error": str(exc),
        }, indent=2, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
