"""Feature 06: context receipts (clozn.context-receipt.v1), segment identity, termination
normalization, receipt privacy tiers, dual-shape reading (legacy pre-2026-07-27 + new), and the
`clozn context` CLI. See clozn/runs/context_receipt.py's module docstring for the shape history."""
from __future__ import annotations

import json
import os

import pytest

import clozn.runs.store as runlog
from clozn.cli import main as cli
from clozn.runs.context_receipt import build_context_receipt, normalize_termination, read_receipt, \
    segment_id

FIXTURE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "schemas",
                            "clozn.context-receipt.v1")


def _legacy_fixture() -> dict:
    """The real pre-2026-07-27 shape, loaded from the same fixture the schema-contract suite uses to
    prove the NEW schema correctly rejects it -- one source of truth for "what legacy looks like"."""
    with open(os.path.join(FIXTURE_ROOT, "invalid__legacy_pre_2026_07_27_shape.json"), encoding="utf-8") \
            as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------------------------------
# build_context_receipt: legacy-shaped fields stay byte-identical (diagnosis.py / other tests depend on this)
# ---------------------------------------------------------------------------------------------------

def test_legacy_shaped_fields_are_additive_and_unchanged():
    delivered = [{"role": "system", "content": "caller rule"},
                 {"role": "user", "content": "question"}]
    assembled = [{"role": "system", "content": "caller rule\n\nmemory card"},
                 {"role": "user", "content": "question"}]
    receipt = build_context_receipt(
        messages=delivered, assembled_messages=assembled, final_prompt="<rendered exact>",
        finish_reason="length", meta={"max_tokens": 8, "n_ctx": 128, "prompt_tokens": 31},
        trace={"tokens": ["a", "b"]}, run_id="run_test1",
    )

    assert receipt["survived"]["assembled_messages"] == assembled
    assert receipt["survived"]["final_prompt"] == "<rendered exact>"
    assert receipt["input_truncated"] is False
    assert receipt["output_cut_off"] is True
    assert receipt["warnings"][0]["code"] == "output_truncated"
    assert receipt["limits"] == {
        "prompt_tokens": 31, "context_window_tokens": 128,
        "requested_max_tokens": 8, "generated_tokens": 2,
    }
    # no schema_validation_error -- a well-formed call must validate cleanly
    assert "schema_validation_error" not in receipt


def test_run_store_persists_context_receipt_and_summary_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    rid = runlog.record(
        source="openai_api", messages=[{"role": "user", "content": "hello"}],
        assembled_messages=[{"role": "system", "content": "memory"},
                            {"role": "user", "content": "hello"}],
        final_prompt="EXACT PROMPT", response="partial", finish_reason="length",
        meta={"max_tokens": 2, "n_ctx": 64, "prompt_tokens": 9},
        trace=[{"piece": "par"}, {"piece": "tial"}],
    )
    run = runlog.get_run(rid)
    assert run["context_receipt"]["survived"]["final_prompt"] == "EXACT PROMPT"
    assert run["warnings"][0]["code"] == "output_truncated"
    assert runlog.list_runs(1)[0]["warnings"] == run["warnings"]
    # the new schema fields ride alongside, unbroken
    assert run["context_receipt"]["schema_version"] == "clozn.context-receipt.v1"
    assert run["context_receipt"]["run_id"] == rid


# ---------------------------------------------------------------------------------------------------
# segment identity
# ---------------------------------------------------------------------------------------------------

def test_segment_id_is_stable_content_derived_and_run_independent():
    a = segment_id("user", "hello there")
    b = segment_id("user", "hello there")
    assert a == b
    assert a.startswith("seg_")
    assert segment_id("user", "different content") != a
    assert segment_id("assistant", "hello there") != a          # role matters too


def test_delivered_segments_carry_order_label_and_hash():
    receipt = build_context_receipt(
        messages=[{"role": "system", "content": "rule"}, {"role": "user", "content": "hi"}],
        run_id="run_test2",
    )
    delivered = receipt["delivered"]
    assert [seg["original_order"] for seg in delivered] == [0, 1]
    assert [seg["source_label"] for seg in delivered] == ["system", "user"]
    assert all(seg["source_type"] == "message" for seg in delivered)
    assert all(len(seg["content_hash"]) == 16 for seg in delivered)
    assert delivered[0]["segment_id"] != delivered[1]["segment_id"]


def test_repeated_identical_message_gets_disambiguated_id():
    receipt = build_context_receipt(
        messages=[{"role": "user", "content": "yes"}, {"role": "user", "content": "yes"}],
        run_id="run_test3",
    )
    ids = [seg["segment_id"] for seg in receipt["delivered"]]
    assert ids[0] != ids[1]
    assert ids[0] == segment_id("user", "yes", occurrence=0)
    assert ids[1] == segment_id("user", "yes", occurrence=1)


def test_same_turn_gets_same_segment_id_across_two_different_runs():
    """The property feature 10 (run change explainer) needs: an unchanged system prompt recurring in two
    different runs must be the same segment_id so the two runs' delivered lists can be diffed."""
    msgs_a = [{"role": "system", "content": "you are helpful"}, {"role": "user", "content": "q1"}]
    msgs_b = [{"role": "system", "content": "you are helpful"}, {"role": "user", "content": "q2"}]
    receipt_a = build_context_receipt(messages=msgs_a, run_id="run_a")
    receipt_b = build_context_receipt(messages=msgs_b, run_id="run_b")
    assert receipt_a["delivered"][0]["segment_id"] == receipt_b["delivered"][0]["segment_id"]
    assert receipt_a["delivered"][1]["segment_id"] != receipt_b["delivered"][1]["segment_id"]


def test_assembled_segment_matches_delivered_by_content_not_position():
    delivered = [{"role": "system", "content": "rule"}, {"role": "user", "content": "hi"}]
    reordered = [{"role": "user", "content": "hi"}, {"role": "system", "content": "rule"}]
    receipt = build_context_receipt(messages=delivered, assembled_messages=reordered, run_id="run_test4")
    system_seg = next(s for s in receipt["delivered"] if s["source_label"] == "system")
    user_seg = next(s for s in receipt["delivered"] if s["source_label"] == "user")
    assert system_seg["included"] is True
    assert user_seg["included"] is True
    assembled_ids = {s["segment_id"] for s in receipt["assembled"]}
    assert system_seg["segment_id"] in assembled_ids
    assert user_seg["segment_id"] in assembled_ids


def test_delivered_segment_marked_not_included_when_content_does_not_survive():
    delivered = [{"role": "system", "content": "caller rule"}, {"role": "user", "content": "question"}]
    assembled = [{"role": "system", "content": "caller rule\n\nmemory card"},   # content changed
                 {"role": "user", "content": "question"}]
    receipt = build_context_receipt(messages=delivered, assembled_messages=assembled, run_id="run_test5")
    system_seg = next(s for s in receipt["delivered"] if s["source_label"] == "system")
    user_seg = next(s for s in receipt["delivered"] if s["source_label"] == "user")
    assert system_seg["included"] is False       # exact bytes did not survive -- honest, not a guess
    assert user_seg["included"] is True
    # no transformation claims the system segment was "modified into" a specific assembled one -- clozn
    # has no content-similarity signal to back that; only the always-true template_transformed fires here.
    reasons = {t["reason"] for t in receipt["transformations"]}
    assert reasons == {"template_transformed"} or reasons == set()


def test_no_assembled_key_when_assembled_messages_was_never_captured():
    """omit, never null-pad: assembled_messages=None means 'not captured', not 'captured, empty'."""
    receipt = build_context_receipt(messages=[{"role": "user", "content": "hi"}], run_id="run_test6")
    assert "assembled" not in receipt


def test_unicode_and_non_string_content_do_not_crash_segment_computation():
    messages = [
        {"role": "user", "content": "héllo wörld unicode-test"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},   # tool-call msg, content None
        {"role": "user"},                                                     # missing content key entirely
    ]
    receipt = build_context_receipt(messages=messages, run_id="run_test7")
    assert len(receipt["delivered"]) == 3
    for seg in receipt["delivered"]:
        assert seg["segment_id"].startswith("seg_")
        assert isinstance(seg["delivered_bytes"], int)


# ---------------------------------------------------------------------------------------------------
# termination normalization
# ---------------------------------------------------------------------------------------------------

def test_termination_absent_when_nothing_to_classify():
    assert normalize_termination(None, None, {}, {}) is None


def test_termination_eos():
    out = normalize_termination("stop", None, {}, {"generated_tokens": 5})
    assert out == {"reason": "eos", "reason_raw": "stop", "generated_tokens": 5}


def test_termination_tool_call():
    out = normalize_termination("tool_calls", None, {}, {})
    assert out["reason"] == "tool_call"
    assert out["reason_raw"] == "tool_calls"


def test_termination_max_tokens_when_only_the_output_cap_is_provably_hit():
    limits = {"prompt_tokens": 10, "context_window_tokens": 4096, "requested_max_tokens": 8,
             "generated_tokens": 8}
    out = normalize_termination("length", None, {}, limits)
    assert out["reason"] == "max_tokens"


def test_termination_context_limit_when_the_window_is_provably_hit():
    limits = {"prompt_tokens": 4090, "context_window_tokens": 4096, "requested_max_tokens": 512,
             "generated_tokens": 6}
    out = normalize_termination("length", None, {}, limits)
    assert out["reason"] == "context_limit"


def test_termination_unknown_length_when_limits_cannot_separate_the_cause():
    out = normalize_termination("length", None, {}, {"generated_tokens": 3})
    assert out["reason"] == "unknown"
    assert out["reason_raw"] == "length"


def test_termination_client_cancelled_reads_stream_failure_not_the_free_text_error():
    """Regression guard: clozn.runs.diagnosis._cutoff_finding compares run['error'] to the bare literal
    "client_disconnected", which the actual stored text ("client disconnected mid-stream: ...") never
    equals. This must read meta['stream_failure'] instead, not repeat that comparison."""
    meta = {"stream_failure": "client_disconnected"}
    out = normalize_termination(None, "client disconnected mid-stream: timed out", meta, {})
    assert out["reason"] == "client_cancelled"
    assert out["reason_raw"] == "client_disconnected"


def test_termination_worker_error_from_stream_failure():
    out = normalize_termination(None, None, {"stream_failure": "worker_disconnected"}, {})
    assert out["reason"] == "worker_error"


def test_termination_worker_error_from_generic_error_string():
    out = normalize_termination(None, "engine returned a malformed frame", {}, {})
    assert out["reason"] == "worker_error"
    assert out["reason_raw"] == "engine returned a malformed frame"


def test_termination_unknown_for_an_unrecognized_finish_reason():
    out = normalize_termination("something_new", None, {}, {})
    assert out["reason"] == "unknown"
    assert out["reason_raw"] == "something_new"


def test_termination_reasons_with_no_live_detection_path_are_still_valid_schema_values():
    """stop_sequence / content_filter / timeout: clozn has no engine value, no moderation layer, and
    queue-level timeouts never reach a recorded run -- so normalize_termination never emits them, but
    they must still be legal per the schema for forward compatibility."""
    from clozn import schemas
    enum = schemas.load("clozn.context-receipt.v1")["properties"]["termination"]["properties"]["reason"][
        "enum"]
    for reason in ("stop_sequence", "content_filter", "timeout"):
        assert reason in enum


def test_build_context_receipt_uses_normalize_termination():
    receipt = build_context_receipt(
        messages=[{"role": "user", "content": "hi"}], finish_reason="length",
        meta={"max_tokens": 4, "n_ctx": 4096, "prompt_tokens": 10},
        trace={"tokens": ["a", "b", "c", "d"]}, run_id="run_test8",
    )
    assert receipt["termination"]["reason"] == "max_tokens"
    assert receipt["termination"]["generated_tokens"] == 4


# ---------------------------------------------------------------------------------------------------
# rendered / identity fingerprint
# ---------------------------------------------------------------------------------------------------

def test_rendered_sha256_and_exact_token_count():
    receipt = build_context_receipt(
        messages=[{"role": "user", "content": "hi"}], final_prompt="THE EXACT PROMPT",
        meta={"prompt_tokens": 42}, run_id="run_test9",
    )
    import hashlib
    assert receipt["rendered"]["sha256"] == hashlib.sha256(b"THE EXACT PROMPT").hexdigest()
    assert receipt["rendered"]["token_count"] == 42
    assert receipt["rendered"]["estimated"] is False


def test_rendered_omitted_when_nothing_measurable():
    receipt = build_context_receipt(messages=[{"role": "user", "content": "hi"}], run_id="run_test10")
    assert "rendered" not in receipt


def test_template_fingerprint_reused_with_explicit_conflation_caveat():
    receipt = build_context_receipt(
        messages=[{"role": "user", "content": "hi"}],
        identity={"template_fingerprint": "9a4c1f0b7e2d6835"}, run_id="run_test11",
    )
    assert receipt["template_fingerprint"] == "9a4c1f0b7e2d6835"
    assert receipt["tokenizer_conflated_with_template"] is True
    assert "tokenizer_fingerprint" not in receipt          # deliberately not added -- see adjudication (b)


# ---------------------------------------------------------------------------------------------------
# receipt privacy tiers
# ---------------------------------------------------------------------------------------------------

_ARGS = dict(
    messages=[{"role": "system", "content": "rule"}, {"role": "user", "content": "question"}],
    assembled_messages=[{"role": "system", "content": "rule"}, {"role": "user", "content": "question"}],
    final_prompt="EXACT RENDERED PROMPT", run_id="run_priv",
)


def test_privacy_full_keeps_everything():
    receipt = build_context_receipt(**_ARGS, privacy="full")
    assert receipt["survived"]["final_prompt"] == "EXACT RENDERED PROMPT"
    assert receipt["survived"]["assembled_messages"] is not None
    assert receipt["delivered"][0]["source_label"] == "system"
    assert receipt["rendered"]["sha256"]
    assert receipt["privacy"] == "full"


def test_privacy_metadata_only_drops_full_text_keeps_segment_metadata():
    receipt = build_context_receipt(**_ARGS, privacy="metadata_only")
    assert "final_prompt" not in receipt["survived"]
    assert "assembled_messages" not in receipt["survived"]
    assert receipt["survived"]["content_withheld_by_privacy_tier"] == "metadata_only"
    assert receipt["delivered"][0]["source_label"] == "system"      # segment metadata still present
    assert receipt["rendered"]["sha256"]                            # hashes always retained
    assert receipt["delivered"][0]["redaction_state"] == "redacted"


def test_privacy_hashes_only_drops_segment_metadata_too():
    receipt = build_context_receipt(**_ARGS, privacy="hashes_only")
    assert "final_prompt" not in receipt["survived"]
    seg = receipt["delivered"][0]
    assert "source_label" not in seg
    assert "delivered_bytes" not in seg
    assert seg["content_hash"]
    assert seg["redaction_state"] == "hash_only"
    assert receipt["rendered"]["sha256"]                            # hashes always retained


def test_privacy_off_builds_nothing_beyond_the_required_marker():
    receipt = build_context_receipt(**_ARGS, privacy="off")
    assert receipt["privacy"] == "off"
    assert receipt["schema_version"] == "clozn.context-receipt.v1"
    assert receipt["run_id"] == "run_priv"
    assert "delivered" not in receipt
    assert "survived" not in receipt


def test_receipt_privacy_tier_settable_and_persisted(tmp_path, monkeypatch):
    from clozn.runs import receipt_privacy
    import clozn.settings as settings
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "studio_settings.json"))
    assert receipt_privacy.tier() == "full"
    assert receipt_privacy.set_tier("hashes_only") is True
    assert receipt_privacy.tier() == "hashes_only"
    assert receipt_privacy.set_tier("not_a_real_tier") is False
    assert receipt_privacy.tier() == "hashes_only"


# ---------------------------------------------------------------------------------------------------
# resilience: a broken/incomplete build never raises (a receipt bug must cost its own field, not the run)
# ---------------------------------------------------------------------------------------------------

def test_missing_run_id_degrades_visibly_instead_of_raising(capsys):
    receipt = build_context_receipt(messages=[{"role": "user", "content": "hi"}])
    assert "schema_validation_error" in receipt
    assert receipt["delivered"]                                     # best-effort content still present
    err = capsys.readouterr().err
    assert "failed schema validation" in err


# ---------------------------------------------------------------------------------------------------
# dual-shape reading: legacy (pre-2026-07-27) vs. new, proven against the REAL legacy shape
# ---------------------------------------------------------------------------------------------------

def test_read_receipt_detects_legacy_shape():
    run = {"id": "run_old", "context_receipt": _legacy_fixture()}
    view = read_receipt(run)
    assert view["shape"] == "legacy"
    assert view["receipt"]["schema"] == "clozn.context_receipt.v1"


def test_read_receipt_detects_new_shape():
    receipt = build_context_receipt(messages=[{"role": "user", "content": "hi"}], run_id="run_new")
    run = {"id": "run_new", "context_receipt": receipt}
    view = read_receipt(run)
    assert view["shape"] == "new"


def test_read_receipt_absent_when_no_context_receipt_field():
    assert read_receipt({"id": "run_x"})["shape"] == "absent"


def test_legacy_fixture_is_rejected_by_the_new_schema():
    """The flip side of test_read_receipt_detects_legacy_shape: the same real legacy document must NOT
    validate against clozn.context-receipt.v1 -- it is read via shape detection, never migrated in
    place, and never claimed to satisfy a schema it predates."""
    from clozn import schemas
    with pytest.raises(schemas.ValidationError):
        schemas.validate(_legacy_fixture(), "clozn.context-receipt.v1")


# ---------------------------------------------------------------------------------------------------
# CLI: clozn context last / show, dual-shape rendering, --detailed
# ---------------------------------------------------------------------------------------------------

def test_context_last_uses_latest_organic_run_and_prints_both_sections(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    organic = runlog.record(
        source="cli", client="cli", messages=[{"role": "user", "content": "raw question"}],
        assembled_messages=[{"role": "user", "content": "raw question"}],
        final_prompt="RENDERED QUESTION", response="partial", finish_reason="length",
        meta={"max_tokens": 4}, started=1.0,
    )
    runlog.record(
        source="replay", parent_run_id=organic,
        messages=[{"role": "user", "content": "internal replay"}], response="child", started=2.0,
    )

    assert cli.main(["context", "last"]) == 0
    output = capsys.readouterr().out
    assert organic in output
    assert "internal replay" not in output
    assert "DELIVERED" in output and "raw question" in output
    assert "SURVIVED" in output and "RENDERED QUESTION" in output
    assert "WARNING" in output and "reply may be incomplete" in output

    assert cli.main(["context", "last", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == organic
    assert payload["context_receipt"]["schema_version"] == "clozn.context-receipt.v1"
    assert isinstance(payload["context_receipt"]["delivered"], list)


def test_context_show_looks_up_one_run_by_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    rid = runlog.record(source="cli", client="cli",
                        messages=[{"role": "user", "content": "specific question"}],
                        response="answer", started=1.0)
    assert cli.main(["context", "show", rid]) == 0
    assert "specific question" in capsys.readouterr().out

    assert cli.main(["context", "show", "run_does_not_exist"]) != 0


def test_context_detailed_shows_segments_and_transformations(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    rid = runlog.record(
        source="cli", client="cli", messages=[{"role": "user", "content": "hi"}],
        assembled_messages=[{"role": "user", "content": "hi"}],
        final_prompt="RENDERED", response="ok", started=1.0,
    )
    assert cli.main(["context", "show", rid, "--detailed"]) == 0
    output = capsys.readouterr().out
    assert "SEGMENTS" in output
    assert "TRANSFORMATIONS" in output
    assert "template_transformed" in output


def test_cli_renders_a_legacy_shaped_run_without_crashing(tmp_path, monkeypatch, capsys):
    """The dual-shape requirement end to end: a run recorded before this feature's rewrite must still
    render sensibly through the CLI, not just through read_receipt() in isolation."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    rid = runlog.record(source="cli", client="cli", messages=[{"role": "user", "content": "hi"}],
                        response="ok", started=1.0)
    run = runlog.get_run(rid)
    run["context_receipt"] = _legacy_fixture()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    assert cli.main(["context", "show", rid]) == 0
    output = capsys.readouterr().out
    assert "DELIVERED" in output
    assert "caller rule" in output          # from the legacy fixture's delivered.messages
    assert "SURVIVED" in output


def test_reconstructs_a_best_effort_receipt_for_a_run_with_no_context_receipt_at_all(tmp_path, monkeypatch,
                                                                                     capsys):
    """Older-than-Phase-2.4 runs (or privacy="off") have no context_receipt field at all."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    rid = runlog.record(source="cli", client="cli", messages=[{"role": "user", "content": "hi"}],
                        response="ok", started=1.0)
    run = runlog.get_run(rid)
    del run["context_receipt"]
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    assert cli.main(["context", "show", rid]) == 0
    assert "DELIVERED" in capsys.readouterr().out
