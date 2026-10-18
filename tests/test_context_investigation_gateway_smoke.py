from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "smoke"
    / "context_investigation_gateway_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "context_investigation_gateway_smoke", SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _reader(*, measured=1, failed=0, unavailable=0, not_measured=3):
    return {
        "measurement": {
            "completed_source_count": measured,
            "failed_source_count": failed,
            "unavailable_source_count": unavailable,
            "not_measured_source_count": not_measured,
        },
    }


def _cancelled_job(**overrides):
    return {
        "state": "cancelled",
        "cancel_requested": True,
        **overrides,
    }


def test_cancellation_fixture_has_non_overlapping_exact_source_ranges():
    text, sources = smoke._cancellation_source_fixture("nonce", 8)

    assert len(sources) == 8
    for index, source in enumerate(sources, start=1):
        start, end = source["unicode_range"]
        assert text[start:end].startswith(f"Record {index:02d}:")
        if index > 1:
            assert sources[index - 2]["unicode_range"][1] < start


def test_live_fixture_protects_instructions_and_question_but_marks_context_removable():
    class Client:
        body = None

        def post(self, path, body):
            assert path == "/v1/chat/completions"
            self.body = body
            return 200, {"clozn_run_id": "run_one"}

    client = Client()
    run_id, _generated = smoke._create_run(
        client,
        model="fixture-model",
        context="Northstar launches May 22.",
        sources=[{
            "message_index": 0,
            "source_id": "launch-date",
            "unicode_range": [0, 26],
        }],
        prompt="When does Northstar launch?",
        max_tokens=32,
    )

    assert run_id == "run_one"
    assert [message["role"] for message in client.body["messages"]] == ["system", "user", "user"]
    assert client.body["clozn_sources"][0]["message_index"] == 1
    assert "clozn_receipt" not in client.body


def test_cancelled_direct_measurement_gate_accepts_partial_honest_evidence():
    summary = smoke._validate_cancelled_effects(
        _cancelled_job(), reader_after=_reader(), source_count=4,
    )

    assert summary == {
        "measured_source_count": 1,
        "not_measured_source_count": 3,
        "source_count": 4,
    }


@pytest.mark.parametrize(
    ("job", "reader", "message"),
    [
        ({"state": "completed", "cancel_requested": True}, _reader(), "was not cancelled"),
        (_cancelled_job(result={}), _reader(), "completed result"),
        (_cancelled_job(), _reader(measured=4, not_measured=0), "measured every source"),
        (_cancelled_job(), _reader(measured=1, not_measured=1), "inconsistent"),
    ],
)
def test_cancelled_direct_measurement_gate_rejects_false_positives(job, reader, message):
    with pytest.raises(smoke.SmokeFailure, match=message):
        smoke._validate_cancelled_effects(job, reader_after=reader, source_count=4)
