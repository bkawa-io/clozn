"""Tests for clozn.runs.claims -- deterministic answer segmentation and claim extraction (E1).

No model, no network, no filesystem outside `tmp_path`. `build_answer_claims` is a pure function of one
run dict; every test either asserts on its return value directly or, where a test wants a stronger
byte-level determinism check than Python dict equality gives for free, serializes to `tmp_path` and
compares the written bytes.
"""
from __future__ import annotations

import copy
import json

import pytest

from clozn import schemas
from clozn.runs import claims


def _canonical(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _texts(document: dict, answer: str) -> list[str]:
    """The exact answer substring each claim's span covers, in order."""
    out = []
    for claim in document["claims"]:
        canonical = claim["text_span"]["resolution"]["canonical"]
        out.append(answer[canonical["start"]:canonical["end"]])
    return out


def _categories(document: dict) -> list[tuple[str, str]]:
    return [(claim["category"], claim["category_reason"]) for claim in document["claims"]]


# ======================================================================================================
# Determinism -- the load-bearing property
# ======================================================================================================

def test_deterministic_byte_identical_output(tmp_path):
    answer = ("The Eiffel Tower was completed in 1889. You should visit at sunset. "
              "It might be crowded in summer.\n1. Buy a ticket online.\n2. Arrive early.")
    run = {"id": "run-det", "response": answer}

    first = claims.build_answer_claims(copy.deepcopy(run))
    second = claims.build_answer_claims(copy.deepcopy(run))

    path_a = tmp_path / "first.json"
    path_b = tmp_path / "second.json"
    path_a.write_bytes(_canonical(first))
    path_b.write_bytes(_canonical(second))

    assert path_a.read_bytes() == path_b.read_bytes()
    assert first == second


def test_deterministic_across_privacy_modes_offsets_and_categories_match(tmp_path):
    """privacy only changes whether `text` is embedded -- offsets, hashes, and categories must not move."""
    answer = "Water boils at 100 degrees. You should let it cool first."
    run = {"id": "run-priv", "response": answer}
    metadata_only = claims.build_answer_claims(copy.deepcopy(run), privacy="metadata_only")
    full = claims.build_answer_claims(copy.deepcopy(run), privacy="full")

    assert len(metadata_only["claims"]) == len(full["claims"])
    for a, b in zip(metadata_only["claims"], full["claims"]):
        ca = a["text_span"]["resolution"]["canonical"]
        cb = b["text_span"]["resolution"]["canonical"]
        assert (ca["start"], ca["end"]) == (cb["start"], cb["end"])
        assert ca["basis_sha256"] == cb["basis_sha256"]
        assert ca["span_sha256"] == cb["span_sha256"]
        assert a["category"] == b["category"]
        assert a["category_reason"] == b["category_reason"]
        assert "text" not in ca
        assert cb["text"] == answer[cb["start"]:cb["end"]]


# ======================================================================================================
# Required case 1: a plain factual paragraph
# ======================================================================================================

def test_plain_factual_paragraph():
    answer = ("The Eiffel Tower was completed in 1889. It is located in Paris, France and stands "
              "330 meters tall. It is a famous landmark.")
    doc = claims.build_answer_claims({"id": "run-1", "response": answer})
    schemas.validate(doc)

    assert doc["segmentation"] == {"state": "ok", "claim_count": 3}
    assert _texts(doc, answer) == [
        "The Eiffel Tower was completed in 1889.",
        "It is located in Paris, France and stands 330 meters tall.",
        "It is a famous landmark.",
    ]
    categories = _categories(doc)
    # Sentences carrying concrete evidence (a year, a place name + measurement) are the conservative
    # positive case for factual_claim.
    assert categories[0] == ("factual_claim", "factual_declarative")
    assert categories[1] == ("factual_claim", "factual_declarative")
    # A bare "It is a famous landmark" has a copula but no digit/quote/internal proper noun -- the
    # conservative default applies, not a fabricated factual_claim tag.
    assert categories[2] == ("non_verifiable_prose", "no_deterministic_category_match")


# ======================================================================================================
# Required case 2: a numbered instruction list
# ======================================================================================================

def test_numbered_instruction_list():
    answer = "To set up the project:\n1. Install Python.\n2. Run the installer.\n3. Restart your terminal."
    doc = claims.build_answer_claims({"id": "run-2", "response": answer})
    schemas.validate(doc)

    assert doc["segmentation"]["state"] == "ok"
    texts = _texts(doc, answer)
    assert texts[1:] == ["1. Install Python.", "2. Run the installer.", "3. Restart your terminal."]
    categories = _categories(doc)
    for category, reason in categories[1:]:
        assert (category, reason) == ("instruction_procedure", "list_item_imperative")


def test_list_items_without_a_blank_line_between_them_do_not_merge():
    answer = "1. Install Python.\n2. Run the installer.\n3. Restart your terminal."
    doc = claims.build_answer_claims({"id": "run-list-tight", "response": answer})
    assert doc["segmentation"]["claim_count"] == 3
    assert _texts(doc, answer) == [
        "1. Install Python.", "2. Run the installer.", "3. Restart your terminal.",
    ]


def test_list_item_wrapped_continuation_line_stays_one_claim():
    answer = "1. Install Python\n   from the official site.\n2. Run the installer."
    doc = claims.build_answer_claims({"id": "run-wrap", "response": answer})
    assert doc["segmentation"]["claim_count"] == 2
    texts = _texts(doc, answer)
    assert texts[0] == "1. Install Python\n   from the official site."
    assert texts[1] == "2. Run the installer."


def test_unrecognized_verb_list_item_is_not_forced_into_instruction_procedure():
    """A closed imperative-verb list is a real limitation, not a bug: a list item whose first word isn't
    in it falls through to the ordinary rules like any other sentence -- it is not blindly assumed to be
    a procedure step just because it is formatted as a list."""
    answer = "1. Buy milk.\n2. Feed the cat."
    doc = claims.build_answer_claims({"id": "run-unrecognized-verb", "response": answer})
    for category, reason in _categories(doc):
        assert reason != "list_item_imperative"


# ======================================================================================================
# Required case 3: hedged / uncertain statements
# ======================================================================================================

def test_hedged_uncertain_statements():
    answer = "This might work, but I am not entirely sure. It could also fail depending on your setup."
    doc = claims.build_answer_claims({"id": "run-3", "response": answer})
    schemas.validate(doc)

    assert doc["segmentation"]["state"] == "ok"
    for category, reason in _categories(doc):
        assert (category, reason) == ("uncertainty_statement", "hedge_marker")


def test_hedge_takes_precedence_over_recommendation_and_instruction():
    """A hedged suggestion/instruction reads as uncertain first -- see RULES' documented rule order."""
    answer = "You might want to try restarting the server, but I'm not sure it will help."
    doc = claims.build_answer_claims({"id": "run-precedence", "response": answer})
    assert _categories(doc) == [("uncertainty_statement", "hedge_marker")]


def test_recommendation_vs_plain_imperative_are_distinguished():
    answer = "You should back up your data first. Restart the server to apply changes."
    doc = claims.build_answer_claims({"id": "run-rec-vs-imp", "response": answer})
    assert _categories(doc) == [
        ("recommendation", "recommendation_marker"),
        ("instruction_procedure", "imperative_lead"),
    ]


# ======================================================================================================
# Required case 4: a code block
# ======================================================================================================

def test_code_block_is_never_factual_claim():
    answer = "Here's an example:\n```python\nprint(\"hello\")\n```\nThat should work for most setups."
    doc = claims.build_answer_claims({"id": "run-4", "response": answer})
    schemas.validate(doc)

    texts = _texts(doc, answer)
    categories = _categories(doc)
    fence_index = next(i for i, t in enumerate(texts) if t.startswith("```"))
    assert texts[fence_index] == "```python\nprint(\"hello\")\n```"
    assert categories[fence_index] == ("non_verifiable_prose", "code_fence_block")
    assert all(category != "factual_claim" for category, _reason in categories), (
        "no claim in this answer may be factual_claim: the fence must never be, and neither surrounding "
        "sentence carries concrete evidence"
    )


def test_multiple_code_blocks_each_become_their_own_claim():
    answer = "First:\n```py\na = 1\n```\nSecond:\n```py\nb = 2\n```\n"
    doc = claims.build_answer_claims({"id": "run-two-fences", "response": answer})
    fences = [
        (t, c) for t, (c, r) in zip(_texts(doc, answer), _categories(doc)) if r == "code_fence_block"
    ]
    assert len(fences) == 2
    assert fences[0][0] == "```py\na = 1\n```"
    assert fences[1][0] == "```py\nb = 2\n```"


# ======================================================================================================
# Required case 5: a non-English answer
# ======================================================================================================

def test_non_english_latin_script_answer_segments_normally():
    """French sentences split correctly (same ASCII punctuation conventions as English) but the
    category rules are English-phrase lists, so the honest, non-fabricated result is that they default
    to non_verifiable_prose -- proving the conservative default holds across languages, not just when
    English markers happen to be absent from an English sentence."""
    answer = "Le chat est noir. Il dort sur le canape. C'est mignon."
    doc = claims.build_answer_claims({"id": "run-5", "response": answer})
    schemas.validate(doc)

    assert doc["segmentation"]["state"] == "ok"
    assert doc["segmentation"]["claim_count"] == 3
    assert _texts(doc, answer) == [
        "Le chat est noir.", "Il dort sur le canape.", "C'est mignon.",
    ]
    for category, reason in _categories(doc):
        assert (category, reason) == ("non_verifiable_prose", "no_deterministic_category_match")


def test_unsupported_script_is_segmentation_limited_not_broken_spans():
    """A script these ASCII-punctuation heuristics cannot reliably tokenize (dense CJK here) must
    produce a visible typed limited state and zero claims -- never a guessed, wrong span."""
    answer = "日本語のテキストです。これは漢字とひらがなを含みます。テストのための文章です。"
    doc = claims.build_answer_claims({"id": "run-cjk", "response": answer})
    schemas.validate(doc)

    assert doc["segmentation"] == {
        "state": "segmentation_limited", "reason": "unsupported_script_density",
    }
    assert doc["claims"] == []
    # The answer's own hash/length are still honestly reported even though nothing was segmented.
    assert doc["answer_source"]["basis_sha256"]
    assert doc["answer_source"]["basis_code_points"] == len(answer)


# ======================================================================================================
# Required case 6: an empty answer
# ======================================================================================================

def test_empty_answer():
    doc = claims.build_answer_claims({"id": "run-6", "response": ""})
    schemas.validate(doc)

    assert doc["segmentation"] == {"state": "empty", "reason": "answer_text_empty"}
    assert doc["claims"] == []
    assert doc["answer_source"]["basis_code_points"] == 0
    assert doc["answer_source"]["basis_utf8_bytes"] == 0


def test_missing_response_is_unavailable_not_empty():
    """No 'response' key at all (e.g. an errored run) is a distinct, honestly different state from an
    explicit empty string: the model produced nothing to even attempt segmenting."""
    doc = claims.build_answer_claims({"id": "run-no-response"})
    schemas.validate(doc)
    assert doc["segmentation"] == {"state": "unavailable", "reason": "no_answer_text"}
    assert doc["claims"] == []
    assert "basis_sha256" not in doc["answer_source"]


# ======================================================================================================
# Required case 7: a pathological no-punctuation wall of text
# ======================================================================================================

def test_pathological_no_punctuation_wall_of_text():
    answer = "x" * 500
    doc = claims.build_answer_claims({"id": "run-7", "response": answer})
    schemas.validate(doc)

    assert doc["segmentation"] == {"state": "ok", "claim_count": 1}
    assert _texts(doc, answer) == [answer]
    # No terminal punctuation and no recognizable marker: the honest conservative default.
    assert _categories(doc) == [("non_verifiable_prose", "no_deterministic_category_match")]


def test_wall_of_text_with_only_commas_is_still_one_claim():
    answer = ", ".join(["word"] * 80)
    doc = claims.build_answer_claims({"id": "run-commas", "response": answer})
    assert doc["segmentation"]["claim_count"] == 1
    assert _texts(doc, answer) == [answer]


# ======================================================================================================
# Redaction / privacy
# ======================================================================================================

def test_redacted_run_is_unavailable_and_never_leaks_the_literal_text():
    secret = "the launch codes are 04-19-1998"
    run = {"id": "run-redacted", "response": secret, "redaction": {"status": "redacted"}}
    doc = claims.build_answer_claims(run)
    schemas.validate(doc)

    assert doc["segmentation"] == {"state": "unavailable", "reason": "answer_text_redacted"}
    assert doc["claims"] == []
    assert secret not in _canonical(doc).decode("utf-8")
    assert "basis_sha256" not in doc["answer_source"], "omit, never null-pad, for redacted text"


def test_redacted_flag_list_is_also_honored():
    """`_run_is_redacted`'s sibling convention in text_span_addresses.py also checks `flags` for the
    literal string 'redacted', not just `redaction.status` -- this module honors the same signal."""
    run = {"id": "run-flag-redacted", "response": "secret text", "flags": ["redacted"]}
    doc = claims.build_answer_claims(run)
    assert doc["segmentation"]["state"] == "unavailable"
    assert doc["segmentation"]["reason"] == "answer_text_redacted"


def test_literal_redacted_status_does_not_wipe_the_answer():
    """Only the full 'redacted' status removes text (matching text_span_addresses.py's own
    _run_is_fully_redacted distinction) -- 'literal_redacted' leaves the recorded answer available to
    segment, same as text_span_addresses.py's own _run_is_redacted/_run_is_fully_redacted split."""
    run = {"id": "run-literal", "response": "Paris is the capital of France.",
           "redaction": {"status": "literal_redacted"}}
    doc = claims.build_answer_claims(run)
    assert doc["segmentation"]["state"] == "ok"
    assert doc["segmentation"]["claim_count"] == 1


# ======================================================================================================
# The run document is never mutated
# ======================================================================================================

def test_run_document_is_never_mutated():
    run = {"id": "run-immutable", "response": "The sky is blue. You should look up sometime.",
           "meta": {"unrelated": True}}
    before = copy.deepcopy(run)
    claims.build_answer_claims(run)
    assert run == before


# ======================================================================================================
# Structural invariants across everything above
# ======================================================================================================

@pytest.mark.parametrize("answer", [
    "The Eiffel Tower was completed in 1889. You should visit at sunset. It might be crowded.",
    "1. Install Python.\n2. Run the installer.\n3. Restart your terminal.",
    "Here's code:\n```py\nx = 1\n```\nDone.",
    "Le chat est noir. Il dort. C'est mignon.",
])
def test_claim_spans_are_ordered_non_overlapping_and_in_bounds(answer):
    doc = claims.build_answer_claims({"id": "run-invariant", "response": answer})
    cursor = 0
    for index, claim in enumerate(doc["claims"]):
        assert claim["index"] == index
        canonical = claim["text_span"]["resolution"]["canonical"]
        start, end = canonical["start"], canonical["end"]
        assert 0 <= start < end <= len(answer)
        assert start >= cursor, "claims must not overlap or go backwards"
        cursor = end


@pytest.mark.parametrize("answer", [
    "Water boils at 100 degrees. Consider letting it cool.",
    "1. Open the terminal.\n2. Type the command.",
    "Le chat est noir.",
])
def test_offsets_exactly_match_the_source_text_privacy_full(answer):
    doc = claims.build_answer_claims({"id": "run-offsets", "response": answer}, privacy="full")
    for claim in doc["claims"]:
        canonical = claim["text_span"]["resolution"]["canonical"]
        assert canonical["text"] == answer[canonical["start"]:canonical["end"]]


def test_every_claim_validates_as_a_standalone_text_span_address_kind():
    """Each embedded text_span independently carries the exact clozn.text-span-addresses.v1 address
    shape -- kind 'claim', native_ref.collection 'derived.claims' -- proving no second scheme exists."""
    doc = claims.build_answer_claims({
        "id": "run-kind-check", "response": "Rome is the capital of Italy. You should visit in spring.",
    })
    for claim in doc["claims"]:
        span = claim["text_span"]
        assert span["kind"] == "claim"
        assert span["native_ref"]["collection"] == "derived.claims"
        assert span["native_ref"]["artifact_schema"] == claims.SCHEMA_VERSION
        assert span["address_id"].startswith("span_")
        assert span["relation_key"].startswith("rel_")


# ======================================================================================================
# The rules table itself
# ======================================================================================================

def test_rules_table_matches_the_schemas_category_reason_enum():
    schema = schemas.load(claims.SCHEMA_VERSION)
    schema_reasons = set(schema["$defs"]["claim"]["properties"]["category_reason"]["enum"])
    table_reasons = {rule_id for rule_id, _category, _description in claims.RULES}
    assert table_reasons == schema_reasons


def test_rules_table_rule_ids_are_unique():
    rule_ids = [rule_id for rule_id, _category, _description in claims.RULES]
    assert len(rule_ids) == len(set(rule_ids))


def test_rules_table_categories_are_all_known():
    for _rule_id, category, _description in claims.RULES:
        assert category in claims.CATEGORIES


def test_categorize_claim_only_ever_returns_a_rule_from_the_table():
    table = {(category, rule_id) for rule_id, category, _description in claims.RULES}
    samples = [
        ("A short fenced block.", "fence"),
        ("Is this correct?", "sentence"),
        ("It might be true.", "sentence"),
        ("You should try this.", "sentence"),
        ("Install the package.", "sentence"),
        ("1. Install the package.", "list_item"),
        ("Rome is the capital of Italy.", "sentence"),
        ("Something ambiguous happened.", "sentence"),
    ]
    for text, structural_kind in samples:
        result = claims.categorize_claim(text, structural_kind)
        assert result in table


def test_categorize_claim_rejects_unknown_structural_kind():
    with pytest.raises(ValueError):
        claims.categorize_claim("text", "paragraph")


# ======================================================================================================
# Input validation
# ======================================================================================================

def test_build_answer_claims_requires_a_run_id():
    with pytest.raises(ValueError):
        claims.build_answer_claims({"response": "text"})


def test_build_answer_claims_rejects_bad_privacy_value():
    with pytest.raises(ValueError):
        claims.build_answer_claims({"id": "run-x", "response": "text"}, privacy="private")


def test_build_answer_claims_result_always_validates():
    for run in (
        {"id": "run-a", "response": "Plain text here."},
        {"id": "run-b", "response": ""},
        {"id": "run-c"},
        {"id": "run-d", "response": "secret", "redaction": {"status": "redacted"}},
        {"id": "run-e", "response": "日本語のテキストです。これはテストです。"},
    ):
        document = claims.build_answer_claims(run)
        schemas.validate(document)  # schema_version-inferred, exercising that path too
