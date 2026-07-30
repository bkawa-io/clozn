"""test_diagnosis_rules -- clozn/runs/diagnosis_rules.py (`clozn.diagnosis-findings.v1`, D1): the
deterministic diagnostic rule engine over one run's already-persisted evidence.

Model-free throughout: every run/comparison_run/influence_map fixture is a hand-built dict shaped like a
real stored run record (clozn.runs.store) / a real clozn.context_answer_influence.v1 artifact -- no
engine, no GPU, no network, no filesystem I/O (the module itself never touches disk; this suite therefore
needs no tmp_path isolation from the conftest tripwire, but stays well clear of `~/.clozn` regardless).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import json  # noqa: E402

import pytest  # noqa: E402

from clozn import schemas  # noqa: E402
from clozn.runs import diagnosis_rules as dr  # noqa: E402


# ==================================================================================== fixture builders

def _msg(role: str, content: str, source_id: "str | None" = None) -> dict:
    out = {"role": role, "content": content}
    if source_id:
        out["source_id"] = source_id
    return out


def _run(*, id="run_1", messages=None, response="ok", finish_reason=None, context_receipt=None,
        influence_map=None, identity=None, meta=None, redaction=None, flags=None) -> dict:
    out: dict = {"id": id, "messages": messages if messages is not None else [_msg("user", "hi")],
                "response": response}
    if finish_reason is not None:
        out["finish_reason"] = finish_reason
    if context_receipt is not None:
        out["context_receipt"] = context_receipt
    if influence_map is not None:
        out["influence_map"] = influence_map
    if identity is not None:
        out["identity"] = identity
    if meta is not None:
        out["meta"] = meta
    if redaction is not None:
        out["redaction"] = redaction
    if flags is not None:
        out["flags"] = flags
    return out


def _finding_by_id(doc: dict, rule_id: str) -> dict:
    return next(f for f in doc["findings"] if f["rule_id"] == rule_id)


_INFLUENCE_METHOD = {"name": "x", "mode": "forced_score_intervention", "claim_limit": "x", "caveat": "x"}


def _influence_map(*, thresholds=None, prompt_spans=(), links=(), status="ok") -> dict:
    out = {"schema": "clozn.context_answer_influence.v1", "status": status, "available": status == "ok",
           "method": dict(_INFLUENCE_METHOD), "identity": {}}
    if thresholds is not None:
        out["thresholds"] = thresholds
    if prompt_spans:
        out["prompt_spans"] = list(prompt_spans)
    if links:
        out["links"] = list(links)
    return out


def _link(context_span_id, *, clears_floor, abs_delta_nats, effect="supports") -> dict:
    return {"context_span_id": context_span_id, "answer_span_id": "as1", "context_index": 0,
           "answer_index": 0, "delta_nats": abs_delta_nats, "abs_delta_nats": abs_delta_nats,
           "effect": effect, "clears_floor": clears_floor,
           "evidence_state": "causally_supported" if clears_floor else "observed"}


# ==================================================================================== helper unit tests

def test_split_sentences_splits_on_period_and_newline():
    spans = dr._split_sentences("First one. Second one.\nThird line.")
    text = "First one. Second one.\nThird line."
    pieces = [text[a:b] for a, b in spans]
    assert pieces == ["First one.", "Second one.", "Third line."]


def test_split_sentences_empty_string():
    assert dr._split_sentences("") == [(0, 0)]


def test_split_sentences_no_boundary_is_one_span():
    text = "just one clause with no terminator"
    assert dr._split_sentences(text) == [(0, len(text))]


def test_split_sentences_does_not_merge_short_sentences():
    """REQUIRED CONTRAST: unlike clozn.runs.sections.drill_split (DRILL_MIN_CHARS=40), two short
    sentences stay separate -- the whole reason this module does not reuse that splitter."""
    text = "Always be brief. Never be rude."
    spans = dr._split_sentences(text)
    assert len(spans) == 2
    assert text[spans[0][0]:spans[0][1]] == "Always be brief."
    assert text[spans[1][0]:spans[1][1]] == "Never be rude."


@pytest.mark.parametrize("sentence,expected", [
    ("Always answer in English.", "positive"),
    ("Never answer in English.", "negative"),
    ("Please always be concise.", "positive"),
    ("Please never use jargon.", "negative"),
    ("You must always cite sources.", None),   # leading marker is "you must", not matched by _DIRECTIVE_RE
    ("Must respond quickly.", "positive"),
    ("Do not use emojis.", "negative"),
    ("Don't use emojis.", "negative"),
    ("This must be a mistake.", None),          # "must" mid-sentence, not a leading directive marker
    ("What is the capital of France?", None),
])
def test_directive_polarity(sentence, expected):
    assert dr._directive_polarity(sentence) == expected


def test_directive_subject_strips_leading_marker_and_normalizes():
    assert dr._directive_subject("Always Answer In English.") == "answer in english"
    assert dr._directive_subject("Never   Answer In English.") == "answer in english"


def test_normalize_directive_text_collapses_whitespace_case_and_trailing_punct():
    assert dr._normalize_directive_text("  Always   Be Concise!!  ") == "always be concise"


@pytest.mark.parametrize("name,reply,expected", [
    ("json", '{"a": 1}', True),
    ("json", "not json at all", False),
    ("json", "Here:\n```json\n{\"a\": 1}\n```\n", True),
    ("bulleted_list", "- one\n- two\n", True),
    ("bulleted_list", "one, two", False),
    ("numbered_list", "1. one\n2. two\n", True),
    ("numbered_list", "one, two", False),
    ("markdown_table", "| a | b |\n| - | - |\n| 1 | 2 |\n", True),
    ("markdown_table", "a, b", False),
    ("single_word", "Paris", True),
    ("single_word", "Paris France", False),
    ("yes_or_no", "Yes.", True),
    ("yes_or_no", "no", True),
    ("yes_or_no", "Maybe", False),
])
def test_format_satisfied(name, reply, expected):
    assert dr._format_satisfied(name, reply) is expected


# =========================================================================== rule registry / evaluate plumbing

def test_rule_registry_has_twelve_rules_in_order():
    assert dr.RULE_IDS == tuple(f"R{n:02d}" for n in range(1, 13))
    assert len(dr.RULE_REGISTRY) == 12


def test_evaluate_returns_one_finding_per_registry_rule_in_order():
    doc = dr.evaluate(_run(), generated_at="2026-07-29T00:00:00Z")
    assert [f["rule_id"] for f in doc["findings"]] == list(dr.RULE_IDS)
    assert [r["rule_id"] for r in doc["rule_registry"]] == list(dr.RULE_IDS)


def test_evaluate_result_validates_against_schema():
    doc = dr.evaluate(_run(), generated_at="2026-07-29T00:00:00Z")
    schemas.validate(doc)   # also exercised internally by evaluate(); re-checked here explicitly
    assert doc["schema_version"] == "clozn.diagnosis-findings.v1"


def test_evaluate_never_raises_on_empty_run():
    doc = dr.evaluate({}, generated_at="2026-07-29T00:00:00Z")
    assert doc["run_id"] == "?"
    assert len(doc["findings"]) == 12
    schemas.validate(doc)


def test_evaluate_never_raises_on_non_mapping_run():
    for bogus in (None, "not a dict", 42, [1, 2, 3]):
        doc = dr.evaluate(bogus, generated_at="2026-07-29T00:00:00Z")
        assert doc["run_id"] == "?"
        schemas.validate(doc)


def test_evaluate_never_raises_on_non_mapping_comparison_run():
    doc = dr.evaluate(_run(), comparison_run="not a dict", generated_at="2026-07-29T00:00:00Z")
    assert _finding_by_id(doc, "R12")["status"] == "pending"


def test_suppressed_rule_ids_flip_status_without_evaluating():
    doc = dr.evaluate(_run(messages=[_msg("system", "Always X."), _msg("user", "Never X.")]),
                      suppressed_rule_ids=["R03"], generated_at="2026-07-29T00:00:00Z")
    entry = _finding_by_id(doc, "R03")
    assert entry["status"] == "suppressed"
    assert entry["evidence"] == []
    assert "severity" not in entry
    assert doc["suppressed_rule_ids"] == ["R03"]


def test_unknown_suppressed_rule_id_is_ignored():
    doc = dr.evaluate(_run(), suppressed_rule_ids=["R03", "not-a-rule", 42],
                      generated_at="2026-07-29T00:00:00Z")
    assert doc["suppressed_rule_ids"] == ["R03"]
    assert len(doc["findings"]) == 12


def test_summary_status_counts_match_findings():
    doc = dr.evaluate(_run(), suppressed_rule_ids=["R05"], generated_at="2026-07-29T00:00:00Z")
    counts = doc["summary"]["status_counts"]
    assert sum(counts.values()) == 12
    for status in dr.STATUS_VALUES:
        assert counts[status] == sum(1 for f in doc["findings"] if f["status"] == status)


def test_non_finding_entries_never_carry_severity_confidence_or_suggested_actions():
    doc = dr.evaluate(_run(), suppressed_rule_ids=["R11"], generated_at="2026-07-29T00:00:00Z")
    for entry in doc["findings"]:
        if entry["status"] != "finding":
            assert "severity" not in entry
            assert "confidence" not in entry
            assert "suggested_actions" not in entry


def test_finding_entries_always_carry_severity_confidence_and_suggested_actions():
    run = _run(finish_reason="length")
    doc = dr.evaluate(run, generated_at="2026-07-29T00:00:00Z")
    findings = [f for f in doc["findings"] if f["status"] == "finding"]
    assert findings
    for entry in findings:
        assert entry["severity"] in dr.SEVERITY_VALUES
        assert entry["confidence"] in dr.CONFIDENCE_VALUES
        assert isinstance(entry["suggested_actions"], list)


def test_evaluate_without_validate_flag_skips_schema_check(monkeypatch):
    """`validate=False` must skip THIS module's own clozn.diagnosis-findings.v1 check -- the shared span
    document `_Context` builds along the way still validates ITSELF internally
    (clozn.runs.text_span_addresses.build_persisted_text_span_addresses does that unconditionally, a
    different module's own contract), so the fair check is "no call validated a diagnosis-findings
    document", not "schemas.validate was never called at all"."""
    real_validate = schemas.validate
    seen_schema_versions = []

    def _spy(document, name=None):
        if isinstance(document, dict):
            seen_schema_versions.append(document.get("schema_version"))
        return real_validate(document, name)

    monkeypatch.setattr(schemas, "validate", _spy)
    dr.evaluate(_run(), validate=False, generated_at="2026-07-29T00:00:00Z")
    assert "clozn.diagnosis-findings.v1" not in seen_schema_versions


# ==================================================================================== determinism (REQUIRED)

def test_evaluate_twice_is_byte_identical():
    """REQUIRED PROOF: the same run in => the same findings out, byte-deterministic."""
    run = _run(
        messages=[_msg("system", "Always answer in English."), _msg("user", "Never answer in English."),
                 _msg("user", "Give me one word answer.")],
        response="Paris, the capital city.", finish_reason="length",
        context_receipt={"schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
                         "termination": {"reason": "max_tokens", "generated_tokens": 64},
                         "limits": {"prompt_tokens": 10, "context_window_tokens": 100}},
    )
    first = dr.evaluate(run, generated_at="2026-07-29T00:00:00Z")
    second = dr.evaluate(run, generated_at="2026-07-29T00:00:00Z")
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_evaluate_twice_with_comparison_run_is_byte_identical():
    run = _run(id="run_b", meta={"temperature": 0.2})
    comparison = _run(id="run_a", meta={"temperature": 0.8})
    first = dr.evaluate(run, comparison_run=comparison, generated_at="2026-07-29T00:00:00Z")
    second = dr.evaluate(run, comparison_run=comparison, generated_at="2026-07-29T00:00:00Z")
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# =========================================================================================== R01

def test_r01_finds_omitted_segments_with_span_evidence():
    run = _run(
        messages=[_msg("user", "some text", "docA"), _msg("user", "final ask")],
        context_receipt={
            "schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
            "omissions": [{"segment_id": "seg_aaaaaaaaaaaaaaaa", "reason": "context_budget"}],
            "delivered": [
                {"segment_id": "seg_aaaaaaaaaaaaaaaa", "source_type": "message", "original_order": 0,
                 "included": False, "client_source_id": "docA"},
                {"segment_id": "seg_bbbbbbbbbbbbbbbb", "source_type": "message", "original_order": 1,
                 "included": True},
            ],
        })
    doc = dr.evaluate(run, generated_at="2026-07-29T00:00:00Z")
    entry = _finding_by_id(doc, "R01")
    assert entry["status"] == "finding"
    assert entry["severity"] == "medium"
    assert entry["confidence"] == "exact"
    assert len(entry["evidence"]) == 1
    assert entry["evidence"][0]["kind"] == "text_span"
    assert entry["evidence"][0]["address_id"].startswith("span_")


def test_r01_not_observed_when_nothing_omitted():
    run = _run(context_receipt={
        "schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
        "delivered": [{"segment_id": "seg_aaaaaaaaaaaaaaaa", "source_type": "message", "original_order": 0,
                      "included": True}],
    })
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R01")
    assert entry["status"] == "not_observed"


def test_r01_unavailable_when_no_context_receipt():
    entry = _finding_by_id(dr.evaluate(_run(), generated_at="2026-07-29T00:00:00Z"), "R01")
    assert entry["status"] == "unavailable"


def test_r01_unavailable_when_redacted():
    run = _run(redaction={"status": "redacted"}, flags=["redacted"])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R01")
    assert entry["status"] == "unavailable"
    assert "redacted" in entry["summary"]


# =========================================================================================== R02

def test_r02_finding_on_context_limit_termination():
    run = _run(context_receipt={"schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
                                "termination": {"reason": "context_limit"}})
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R02")
    assert entry["status"] == "finding"
    assert entry["severity"] == "high"
    assert entry["confidence"] == "exact"


def test_r02_finding_on_high_ratio_heuristic():
    run = _run(context_receipt={"schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
                                "limits": {"prompt_tokens": 3700, "context_window_tokens": 4000}})
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R02")
    assert entry["status"] == "finding"
    assert entry["confidence"] == "derived"


def test_r02_not_observed_below_ratio():
    run = _run(context_receipt={"schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
                                "limits": {"prompt_tokens": 100, "context_window_tokens": 4000}})
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R02")
    assert entry["status"] == "not_observed"


def test_r02_unavailable_when_missing_fields():
    entry = _finding_by_id(dr.evaluate(_run(), generated_at="2026-07-29T00:00:00Z"), "R02")
    assert entry["status"] == "unavailable"


# =========================================================================================== R03 / R10

def test_r03_finds_conflicting_directives():
    run = _run(messages=[_msg("system", "Always answer in English."), _msg("user", "Never answer in English.")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R03")
    assert entry["status"] == "finding"
    assert entry["severity"] == "medium"
    assert entry["confidence"] == "pattern_match"
    assert len(entry["evidence"]) == 2
    assert all(e["kind"] == "text_span" for e in entry["evidence"])


def test_r03_not_observed_when_no_conflict():
    run = _run(messages=[_msg("system", "Always be concise."), _msg("user", "What time is it?")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R03")
    assert entry["status"] == "not_observed"


def test_r03_unavailable_when_redacted():
    run = _run(messages=[_msg("system", "Always X."), _msg("user", "Never X.")],
              redaction={"status": "redacted"})
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R03")
    assert entry["status"] == "unavailable"


def test_r03_falls_back_to_field_evidence_when_no_span_document_is_buildable():
    """A run with no usable 'id' cannot get a clozn.text-span-addresses.v1 document built for it (that
    module requires a non-empty run_id) -- evidence must fall back to a `field` entry citing the message
    index/offsets directly, never a fabricated address_id."""
    run = {"messages": [{"role": "system", "content": "Always X."}, {"role": "user", "content": "Never X."}],
          "response": "ok"}
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R03")
    assert entry["status"] == "finding"
    assert entry["evidence"]
    assert all(e["kind"] == "field" for e in entry["evidence"])
    assert all(e["path"].startswith("messages[") for e in entry["evidence"])


def test_r10_finds_conflict_between_final_request_and_earlier_instruction():
    run = _run(messages=[_msg("system", "Always answer in English."), _msg("user", "hello"),
                         _msg("user", "Never answer in English.")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R10")
    assert entry["status"] == "finding"
    assert entry["confidence"] == "pattern_match"


def test_r10_not_observed_when_final_message_has_no_directive():
    run = _run(messages=[_msg("system", "Always answer in English."), _msg("user", "What time is it?")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R10")
    assert entry["status"] == "not_observed"


# =========================================================================================== R04

def test_r04_finds_exact_duplicate():
    run = _run(messages=[_msg("system", "Always be concise."), _msg("user", "Always be concise.")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R04")
    assert entry["status"] == "finding"
    assert "1 exact" in entry["summary"]


def test_r04_finds_near_duplicate():
    run = _run(messages=[_msg("system", "Always be concise in your replies."),
                         _msg("user", "Always be concise in your reply.")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R04")
    assert entry["status"] == "finding"
    assert "near-duplicate" in entry["summary"]


def test_r04_not_observed_when_directives_are_distinct():
    run = _run(messages=[_msg("system", "Always be concise."), _msg("user", "Never use emojis.")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R04")
    assert entry["status"] == "not_observed"


# =========================================================================================== R05

def test_r05_finds_repeated_source_content():
    run = _run(context_receipt={
        "schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
        "delivered": [
            {"segment_id": "seg_aaaaaaaaaaaaaaaa", "source_type": "message", "original_order": 0,
             "included": True, "client_source_id": "docA", "content_hash": "deadbeefdeadbeef"},
            {"segment_id": "seg_bbbbbbbbbbbbbbbb", "source_type": "message", "original_order": 1,
             "included": True, "client_source_id": "docB", "content_hash": "deadbeefdeadbeef"},
        ],
    })
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R05")
    assert entry["status"] == "finding"
    assert entry["confidence"] == "exact"
    assert len(entry["evidence"]) == 2


def test_r05_not_observed_when_hashes_differ():
    run = _run(context_receipt={
        "schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
        "delivered": [
            {"segment_id": "seg_aaaaaaaaaaaaaaaa", "source_type": "message", "original_order": 0,
             "included": True, "client_source_id": "docA", "content_hash": "aaaaaaaaaaaaaaaa"},
            {"segment_id": "seg_bbbbbbbbbbbbbbbb", "source_type": "message", "original_order": 1,
             "included": True, "client_source_id": "docB", "content_hash": "bbbbbbbbbbbbbbbb"},
        ],
    })
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R05")
    assert entry["status"] == "not_observed"


def test_r05_unavailable_when_no_delivered_list():
    entry = _finding_by_id(dr.evaluate(_run(), generated_at="2026-07-29T00:00:00Z"), "R05")
    assert entry["status"] == "unavailable"


# =========================================================================================== R06

def test_r06_finding_when_json_requested_but_reply_is_not_json():
    run = _run(messages=[_msg("user", "Answer in JSON please.")], response="Sure, the answer is 42.")
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R06")
    assert entry["status"] == "finding"
    assert "json" in entry["summary"]


def test_r06_not_observed_when_reply_satisfies_requested_format():
    run = _run(messages=[_msg("user", "Answer in JSON please.")], response='{"answer": 42}')
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R06")
    assert entry["status"] == "not_observed"
    assert "satisfies" in entry["summary"]


def test_r06_not_observed_when_no_format_requested():
    run = _run(messages=[_msg("user", "What time is it?")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R06")
    assert entry["status"] == "not_observed"
    assert "no recognized" in entry["summary"]


def test_r06_unavailable_when_no_reply_recorded():
    run = _run(messages=[_msg("user", "Answer in JSON please.")], response="")
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R06")
    assert entry["status"] == "unavailable"


# =========================================================================================== R07

def test_r07_finds_far_instruction():
    run = _run(messages=[_msg("system", "Always answer in English."), _msg("user", "hi"),
                         _msg("assistant", "hello"), _msg("user", "how are you"),
                         _msg("assistant", "fine"), _msg("user", "what time is it")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R07")
    assert entry["status"] == "finding"
    assert entry["confidence"] == "derived"


def test_r07_not_observed_when_instruction_is_close():
    run = _run(messages=[_msg("system", "Always answer in English."), _msg("user", "what time is it")])
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R07")
    assert entry["status"] == "not_observed"


# =========================================================================================== R08 / R09

def test_r08_pending_when_no_influence_map_key():
    entry = _finding_by_id(dr.evaluate(_run(), generated_at="2026-07-29T00:00:00Z"), "R08")
    assert entry["status"] == "pending"


def test_r09_pending_when_no_influence_map_key():
    entry = _finding_by_id(dr.evaluate(_run(), generated_at="2026-07-29T00:00:00Z"), "R09")
    assert entry["status"] == "pending"


def test_r08_unavailable_when_influence_blob_marker():
    run = _run(influence_map={"unavailable": "blob expired", "sha256": "a" * 64})
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R08")
    assert entry["status"] == "unavailable"
    assert "blob expired" in entry["summary"]


def test_r08_unavailable_when_status_not_ok():
    run = _run(influence_map=_influence_map(status="error"))
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R08")
    assert entry["status"] == "unavailable"


def test_r08_finding_when_a_source_never_clears_floor():
    influence = _influence_map(
        thresholds={"cell_abs_delta_nats": 0.05},
        prompt_spans=[{"id": "ps1", "start": 0, "end": 4, "text": "text"}],
        links=[_link("ps1", clears_floor=False, abs_delta_nats=0.01)])
    run = _run(influence_map=influence)
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R08")
    assert entry["status"] == "finding"
    assert entry["confidence"] == "exact"


def test_r08_not_observed_when_every_source_clears_floor():
    influence = _influence_map(
        thresholds={"cell_abs_delta_nats": 0.05},
        prompt_spans=[{"id": "ps1", "start": 0, "end": 4, "text": "text"}],
        links=[_link("ps1", clears_floor=True, abs_delta_nats=1.0)])
    run = _run(influence_map=influence)
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R08")
    assert entry["status"] == "not_observed"


def test_r09_finding_when_source_clears_floor_narrowly():
    influence = _influence_map(
        thresholds={"cell_abs_delta_nats": 0.05},
        prompt_spans=[{"id": "ps1", "start": 0, "end": 4, "text": "text"}],
        links=[_link("ps1", clears_floor=True, abs_delta_nats=0.06)])   # < 0.05 * 1.5 = 0.075
    run = _run(influence_map=influence)
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R09")
    assert entry["status"] == "finding"
    assert entry["confidence"] == "derived"


def test_r09_not_observed_when_source_clears_floor_comfortably():
    influence = _influence_map(
        thresholds={"cell_abs_delta_nats": 0.05},
        prompt_spans=[{"id": "ps1", "start": 0, "end": 4, "text": "text"}],
        links=[_link("ps1", clears_floor=True, abs_delta_nats=1.0)])    # well above 0.05 * 1.5
    run = _run(influence_map=influence)
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R09")
    assert entry["status"] == "not_observed"


# =========================================================================================== R11

def test_r11_finding_max_tokens_termination():
    run = _run(context_receipt={"schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
                                "termination": {"reason": "max_tokens", "generated_tokens": 200}})
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R11")
    assert entry["status"] == "finding"
    assert entry["severity"] == "high"
    assert any(a["kind"] == "increase_max_tokens" for a in entry["suggested_actions"])


def test_r11_finding_context_limit_termination():
    run = _run(context_receipt={"schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
                                "termination": {"reason": "context_limit"}})
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R11")
    assert entry["status"] == "finding"
    assert any(a["kind"] == "increase_context_budget" for a in entry["suggested_actions"])


def test_r11_finding_legacy_finish_reason_length():
    run = _run(finish_reason="length")
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R11")
    assert entry["status"] == "finding"
    assert "termination" in entry["limitations"][0]


def test_r11_not_observed_on_normal_stop():
    run = _run(finish_reason="stop")
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R11")
    assert entry["status"] == "not_observed"


def test_r11_not_observed_on_non_length_termination():
    run = _run(context_receipt={"schema_version": "clozn.context-receipt.v1", "run_id": "run_1",
                                "termination": {"reason": "eos"}})
    entry = _finding_by_id(dr.evaluate(run, generated_at="2026-07-29T00:00:00Z"), "R11")
    assert entry["status"] == "not_observed"


def test_r11_unavailable_when_nothing_recorded():
    entry = _finding_by_id(dr.evaluate(_run(), generated_at="2026-07-29T00:00:00Z"), "R11")
    assert entry["status"] == "unavailable"


# =========================================================================================== R12

def test_r12_pending_when_no_comparison_run():
    entry = _finding_by_id(dr.evaluate(_run(), generated_at="2026-07-29T00:00:00Z"), "R12")
    assert entry["status"] == "pending"


def test_r12_finding_on_identity_and_generation_drift():
    run = _run(id="run_b", identity={"model_sha256": "b" * 64}, meta={"temperature": 0.2})
    comparison = _run(id="run_a", identity={"model_sha256": "a" * 64}, meta={"temperature": 0.9})
    doc = dr.evaluate(run, comparison_run=comparison, generated_at="2026-07-29T00:00:00Z")
    entry = _finding_by_id(doc, "R12")
    assert entry["status"] == "finding"
    assert entry["confidence"] == "exact"
    assert doc["comparison_run_id"] == "run_a"
    paths = {e["path"] for e in entry["evidence"]}
    assert "identity.model_sha256" in paths
    assert "generation.temperature" in paths


def test_r12_not_observed_when_no_drift():
    run = _run(id="run_b", identity={"model_sha256": "a" * 64}, meta={"temperature": 0.5})
    comparison = _run(id="run_a", identity={"model_sha256": "a" * 64}, meta={"temperature": 0.5})
    entry = _finding_by_id(
        dr.evaluate(run, comparison_run=comparison, generated_at="2026-07-29T00:00:00Z"), "R12")
    assert entry["status"] == "not_observed"


def test_r12_unavailable_when_comparison_run_cannot_be_diffed():
    entry = _finding_by_id(
        dr.evaluate(_run(id="run_b"), comparison_run={}, generated_at="2026-07-29T00:00:00Z"), "R12")
    assert entry["status"] == "unavailable"


# ==================================================================================== schema fixture cross-check

def test_fixture_documents_still_validate():
    """The two hand-generated fixtures under tests/fixtures/schemas/clozn.diagnosis-findings.v1/ are
    exercised by tests/test_schema_contracts.py already; re-checked here so a change to evaluate()'s
    output shape that breaks them is caught by this suite too, not only the schema-contracts one."""
    fixture_dir = os.path.join(REPO_ROOT, "tests", "fixtures", "schemas", "clozn.diagnosis-findings.v1")
    found = 0
    for name in os.listdir(fixture_dir):
        if name.startswith("valid__"):
            with open(os.path.join(fixture_dir, name), encoding="utf-8") as handle:
                document = json.load(handle)
            schemas.validate(document)
            found += 1
    assert found >= 2
