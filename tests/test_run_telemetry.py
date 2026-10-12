"""Model-free checks for privacy-safe OTLP/OpenInference run export."""
from __future__ import annotations

import copy
import json

import pytest

from clozn.runs import telemetry


def _run(**updates):
    run = {
        "id": "run_0000000000abc_123456",
        "created_ts": 1_700_000_000.125,
        "source": "openai_api",
        "client": "openai-python",
        "model": "qwen-local",
        "substrate": "EngineSubstrate",
        "messages": [{"role": "user", "content": "private user prompt"}],
        "assembled_messages": [
            {"role": "system", "content": "private injected memory"},
            {"role": "user", "content": "private user prompt"},
        ],
        "final_prompt": "private rendered template",
        "prompt_summary": "private summary",
        "response": "private model response",
        "response_summary": "private response summary",
        "reasoning": {"text": "private hidden reasoning"},
        "memory": {"cards_applied": ["private card"]},
        "timing": {"started_at": 1_700_000_000.125, "ended_at": 1_700_000_001.625,
                   "duration_ms": 1500},
        "finish_reason": "stop",
        "error": None,
        "meta": {
            "prompt_tokens": 12, "max_tokens": 64, "temperature": 0.0,
            "generation_duration_ms": 900.5, "generation_tokens_per_second": 5.5,
        },
        "trace": {"tokens": ["one", "two", "three"]},
        "context_receipt": {
            "rendered": {"tokens": 12},
            "termination": {"reason": "eos", "generated_tokens": 3},
        },
        "identity": {"model_sha256": "a" * 64, "template_fingerprint": "0123456789abcdef"},
    }
    run.update(updates)
    return run


def _span(record):
    return record["resourceSpans"][0]["scopeSpans"][0]["spans"][0]


def _attrs(span):
    out = {}
    for attribute in span["attributes"]:
        value = attribute["value"]
        out[attribute["key"]] = next(iter(value.values()))
    return out


def test_default_export_is_otlp_openinference_and_omits_all_content():
    run = _run(error="private transport failure")
    record = telemetry.export_runs([run])[0]
    span = _span(record)
    attrs = _attrs(span)
    serialized = json.dumps(record, ensure_ascii=False)

    assert set(record) == {"resourceSpans"}
    assert span["name"] == "clozn.chat"
    assert span["kind"] == 1
    assert len(span["traceId"]) == 32 and int(span["traceId"], 16)
    assert len(span["spanId"]) == 16 and int(span["spanId"], 16)
    assert span["startTimeUnixNano"] == "1700000000125000000"
    assert span["endTimeUnixNano"] == "1700000001625000000"
    assert span["status"] == {"code": 2, "message": "run recorded an error"}
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["llm.model_name"] == "qwen-local"
    assert attrs["llm.token_count.prompt"] == "12"
    assert attrs["llm.token_count.completion"] == "3"
    assert attrs["llm.token_count.total"] == "15"
    assert attrs["clozn.timing.duration_ms"] == "1500"
    assert attrs["clozn.content.policy"] == "omitted"
    assert attrs["llm.input_messages.0.message.role"] == "system"
    assert attrs["llm.output_messages.0.message.role"] == "assistant"
    assert not any(key.endswith(".message.content") for key in attrs)
    for secret in (
        "private user prompt", "private injected memory", "private rendered template",
        "private summary", "private model response", "private response summary",
        "private hidden reasoning", "private card", "private transport failure",
    ):
        assert secret not in serialized


def test_ids_are_stable_and_depend_on_run_id_not_private_content():
    first = telemetry.run_to_span(_run())
    second = telemetry.run_to_span(_run(messages=[{"role": "user", "content": "changed"}],
                                         response="changed"))
    other = telemetry.run_to_span(_run(id="run_other"))

    assert first["traceId"] == second["traceId"]
    assert first["spanId"] == second["spanId"]
    assert (first["traceId"], first["spanId"]) != (other["traceId"], other["spanId"])


def test_explicit_content_uses_model_input_messages_and_response_without_mutating_run():
    run = _run()
    before = copy.deepcopy(run)
    span = telemetry.run_to_span(run, include_content=True)
    attrs = _attrs(span)

    assert attrs["clozn.content.policy"] == "included"
    assert attrs["llm.input_messages.0.message.content"] == "private injected memory"
    assert attrs["llm.input_messages.1.message.content"] == "private user prompt"
    assert attrs["llm.output_messages.0.message.content"] == "private model response"
    assert "input.value" not in attrs
    assert run == before


def test_explicit_literal_redaction_covers_input_and_output_deterministically():
    run = _run(
        assembled_messages=[{"role": "user", "content": "Alice owns abc and ab"}],
        response="Alice replied with abc and ab",
    )
    redactions_a = {"ab": "[SHORT]", "abc": "[LONG]", "Alice": "[NAME]"}
    redactions_b = {"Alice": "[NAME]", "abc": "[LONG]", "ab": "[SHORT]"}
    first = telemetry.run_to_span(run, include_content=True, redactions=redactions_a)
    second = telemetry.run_to_span(run, include_content=True, redactions=redactions_b)
    attrs = _attrs(first)

    assert first == second
    assert attrs["clozn.content.policy"] == "redacted"
    assert attrs["llm.input_messages.0.message.content"] == "[NAME] owns [LONG] and [SHORT]"
    assert attrs["llm.output_messages.0.message.content"] == "[NAME] replied with [LONG] and [SHORT]"
    assert "Alice" not in json.dumps(first)


def test_redaction_requires_content_opt_in_and_inputs_are_validated():
    with pytest.raises(telemetry.TelemetryExportError, match="require include_content"):
        telemetry.export_runs([_run()], redactions={"secret": "[REDACTED]"})
    with pytest.raises(telemetry.TelemetryExportError, match="missing a non-empty id"):
        telemetry.export_runs([_run(id="")])
    with pytest.raises(telemetry.TelemetryExportError, match="iterable"):
        telemetry.export_runs(_run())
    with pytest.raises(telemetry.TelemetryExportError, match="non-empty strings"):
        telemetry.export_runs([_run()], include_content=True, redactions={"": "x"})
    with pytest.raises(telemetry.TelemetryExportError, match="require include_content"):
        telemetry.export_runs([], redactions={"secret": "[REDACTED]"})


def test_model_filename_fallback_never_exports_local_path():
    span = telemetry.run_to_span(_run(
        model="", meta={"model_file": r"C:\\Users\\person\\private\\model.gguf", "prompt_tokens": 2},
        context_receipt={}, trace={"tokens": ["x"]},
    ))
    attrs = _attrs(span)
    assert attrs["llm.model_name"] == "model.gguf"
    assert "person" not in json.dumps(span)

    direct = telemetry.run_to_span(_run(model=r"C:\\Users\\person\\private\\other.gguf"))
    assert _attrs(direct)["llm.model_name"] == "other.gguf"


def test_unknown_message_role_is_not_a_default_privacy_escape_hatch():
    span = telemetry.run_to_span(_run(
        assembled_messages=[{"role": "private-role-secret", "content": "private content"}],
    ))
    assert "private-role-secret" not in json.dumps(span)
    assert "private content" not in json.dumps(span)


def test_jsonl_is_one_otlp_record_per_line_and_write_is_atomic(tmp_path):
    records = telemetry.export_runs([_run(), _run(id="run_2")])
    text = telemetry.format_jsonl(records)
    lines = text.splitlines()
    assert len(lines) == 2
    assert [json.loads(line) for line in lines] == records
    assert text.endswith("\n")

    target = tmp_path / "nested" / "runs.jsonl"
    returned = telemetry.write_jsonl(str(target), records)
    assert returned == str(target.resolve())
    assert target.read_text(encoding="utf-8") == text

    target.write_text("prior-good-data", encoding="utf-8")
    with pytest.raises(telemetry.TelemetryExportError, match="canonical JSON"):
        telemetry.write_jsonl(str(target), [{"bad": float("nan")}])
    assert target.read_text(encoding="utf-8") == "prior-good-data"


def test_timestamp_and_meta_token_fallbacks_are_deterministic():
    span = telemetry.run_to_span(_run(
        created_ts=None, created_at="2024-01-01T00:00:00Z",
        timing={"duration_ms": 250}, context_receipt={},
        meta={"prompt_tokens": 4, "completion_tokens": 2}, trace={},
    ))
    attrs = _attrs(span)
    assert span["startTimeUnixNano"] == "1704067200000000000"
    assert span["endTimeUnixNano"] == "1704067200250000000"
    assert attrs["llm.token_count.prompt"] == "4"
    assert attrs["llm.token_count.completion"] == "2"
