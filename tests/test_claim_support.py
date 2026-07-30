"""Tests for clozn.runs.claim_support -- deterministic claim-to-supplied-source support mapping (E2).

No model, no network, no filesystem outside `tmp_path`. `build_claim_support` is a pure function of
`(run, claims_document)`; every test either asserts on the return value directly or, for the
determinism requirement, serializes to `tmp_path` and compares written bytes -- the same standard E1's
own test suite holds itself to.
"""
from __future__ import annotations

import copy
import json

import pytest

from clozn import schemas
from clozn.runs import claim_support, claims


def _method_stub() -> dict:
    return {
        "name": "context_answer_influence", "mode": "forced_score_intervention",
        "claim_limit": "no percentage claim", "caveat": "measured effect only, not correctness",
    }


def _influence(
    *, status="ok", available=True, prompt_spans=(), answer_text="", answer_spans=(), links=(),
) -> dict:
    doc = {
        "schema": "clozn.context_answer_influence.v1",
        "status": status,
        "available": available,
        "method": _method_stub(),
        "identity": {"model_sha256": "a" * 64},
    }
    if status == "ok":
        doc.update({
            "prompt_spans": list(prompt_spans),
            "answer": {"scored_text": answer_text},
            "answer_spans": list(answer_spans),
            "links": list(links),
        })
    else:
        doc["error"] = {"code": "no_text_context", "message": "no context available"}
    return doc


def _canonical(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _by_index(document: dict) -> dict[int, dict]:
    return {result["claim_index"]: result for result in document["results"]}


# ======================================================================================================
# Required case: claim supported by a causally_supported link
# ======================================================================================================

def test_claim_supported_by_causally_supported_link():
    answer = "The Eiffel Tower was completed in 1889."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 45,
                       "text": "The Eiffel Tower was completed in 1889 in Paris."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[{"context_span_id": "ps-1", "answer_span_id": "as-1", "context_index": 0,
                "answer_index": 0, "delta_nats": -3.2, "abs_delta_nats": 3.2, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    run = {"id": "run-supported", "response": answer, "influence_map": influence}
    cdoc = claims.build_answer_claims(run)
    sdoc = claim_support.build_claim_support(run, cdoc)
    schemas.validate(sdoc)

    result = sdoc["results"][0]
    assert result["status"] == "supported"
    assert result["method"] == {"name": "forced_score_intervention", "max_abs_delta_nats": 3.2}
    assert len(result["source_span_ids"]) == 1
    assert result["source_span_ids"][0].startswith("span_")


def test_supported_requires_causally_supported_not_merely_clears_floor_false():
    """clears_floor alone is not the gate -- evidence_state is (honesty rule 1)."""
    answer = "The bridge was built in 1920."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 40, "text": "The bridge dates to 1920."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[{"context_span_id": "ps-1", "answer_span_id": "as-1", "context_index": 0,
                "answer_index": 0, "delta_nats": -5.0, "abs_delta_nats": 5.0, "effect": "supports",
                "clears_floor": True, "evidence_state": "observed"}],
    )
    run = {"id": "run-observed-only", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    assert sdoc["results"][0]["status"] != "supported"


# ======================================================================================================
# Required case: claim with textual overlap only (caps at weakly_supported)
# ======================================================================================================

def test_textual_overlap_only_caps_at_weakly_supported():
    answer = "The bridge was built in 1920."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 60,
                       "text": "The old bridge downtown was apparently built sometime in the 1920s."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[],  # no link at all -- pure textual overlap, no measurement backing it
    )
    run = {"id": "run-weak", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))

    result = sdoc["results"][0]
    assert result["status"] == "weakly_supported"
    assert result["method"]["name"] == "textual_overlap"
    assert 0.0 < result["method"]["overlap_fraction"] <= 1.0
    assert len(result["source_span_ids"]) == 1


def test_mere_presence_in_prompt_sources_never_counts_even_with_no_prompt_spans():
    """prompt_sources licenses only 'was in the assembled prompt' -- never consulted for any status."""
    answer = "The bridge was built in 1920."
    influence = _influence(
        prompt_spans=[],  # nothing in the addressable/textual list
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[],
    )
    influence["prompt_sources"] = [
        {"id": "src-1", "text": "The bridge was built in 1920.", "selected": True},
    ]
    run = {"id": "run-presence-only", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    assert sdoc["results"][0]["status"] == "unsupported_by_supplied_materials"


# ======================================================================================================
# Required case: claim with no matching source (unsupported, WITH an influence map present)
# ======================================================================================================

def test_no_matching_source_is_unsupported_by_supplied_materials_with_influence_map_present():
    answer = "The bridge was built in 1920."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 40,
                       "text": "Completely unrelated text about cooking recipes and spices."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[],
    )
    run = {"id": "run-unsupported", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    schemas.validate(sdoc)

    result = sdoc["results"][0]
    assert result["status"] == "unsupported_by_supplied_materials"
    assert result["method"] == {"name": "measured_comparison_no_match"}
    assert "source_span_ids" not in result


def test_unsupported_by_supplied_materials_never_reads_as_false_in_the_schema():
    """The house honesty rule, checked directly against the shipped schema text, not just this
    module's own docstring: the status means exactly 'the supplied materials do not support this,'
    and the schema must explicitly disclaim the true/false reading rather than merely avoid the word
    'false' by accident."""
    schema = schemas.load(claim_support.SCHEMA_VERSION)
    description = schema["description"].lower()
    assert "does not support this" in description or "do not support" in description
    assert "not a claim about whether" in description or "never be read that way" in description
    # It must never POSITIVELY assert falsity -- "means false" / "is false" would be exactly that.
    assert "means false" not in description
    assert "is false" not in description


# ======================================================================================================
# Required case: run with no influence map -> measurement_unavailable, never unsupported
# ======================================================================================================

def test_no_influence_map_yields_measurement_unavailable_never_unsupported():
    answer = "The bridge was built in 1920."
    run = {"id": "run-no-influence", "response": answer}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    schemas.validate(sdoc)

    result = sdoc["results"][0]
    assert result["status"] == "measurement_unavailable"
    assert result["status"] != "unsupported_by_supplied_materials"
    assert result["method"] == {"name": "no_influence_map"}
    assert sdoc["source"]["influence_map"]["gate"] == "no_influence_map"


@pytest.mark.parametrize("status,expected_method", [
    ("unavailable", "influence_measurement_unavailable"),
    ("error", "influence_measurement_error"),
])
def test_influence_map_present_but_not_ok_is_also_measurement_unavailable(status, expected_method):
    answer = "The bridge was built in 1920."
    influence = _influence(status=status, available=False)
    run = {"id": f"run-{status}", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    result = sdoc["results"][0]
    assert result["status"] == "measurement_unavailable"
    assert result["method"]["name"] == expected_method


def test_answer_text_mismatch_between_claim_and_influence_map_is_measurement_unavailable():
    """The influence map's own scored answer text differs from run.response -- offsets cannot be
    honestly reconciled, so this is a measurement problem, not evidence of anything."""
    answer = "The bridge was built in 1920."
    stale_scored_text = "A completely different answer was recorded here."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 30, "text": "The bridge dates to 1920."}],
        answer_text=stale_scored_text,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(stale_scored_text), "text": stale_scored_text}],
        links=[],
    )
    run = {"id": "run-mismatch", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    result = sdoc["results"][0]
    assert result["status"] == "measurement_unavailable"
    assert result["method"]["name"] == "answer_text_mismatch"


def test_influence_map_blob_unavailable_marker_is_measurement_unavailable():
    """clozn.runs.store.get_run's own shape for an unresolved blob-backed influence_map_ref."""
    answer = "The bridge was built in 1920."
    run = {"id": "run-blob", "response": answer,
           "influence_map": {"unavailable": "blob missing", "sha256": "a" * 64}}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    assert sdoc["results"][0]["status"] == "measurement_unavailable"
    assert sdoc["results"][0]["method"]["name"] == "no_influence_map"


# ======================================================================================================
# Required case: a direct numeric contradiction
# ======================================================================================================

def test_direct_numeric_contradiction():
    answer = "The bridge was built in 1920."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 60,
                       "text": "The historic bridge was actually built in 1953, records show."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[],
    )
    run = {"id": "run-numeric-contra", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    schemas.validate(sdoc)

    result = sdoc["results"][0]
    assert result["status"] == "contradicted"
    assert result["method"] == {"name": "numeric_or_date_mismatch"}
    assert len(result["source_span_ids"]) == 1


def test_direct_negation_contradiction():
    answer = "The museum is open on Mondays."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 60,
                       "text": "Please note the museum is not open on Mondays this season."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[],
    )
    run = {"id": "run-negation-contra", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    result = sdoc["results"][0]
    assert result["status"] == "contradicted"
    assert result["method"] == {"name": "direct_negation"}


def test_low_overlap_number_mismatch_is_not_a_false_contradiction():
    """A source with an unrelated number and low content overlap must never trip 'contradicted' --
    the strong-overlap gate is the whole point of honesty rule 3."""
    answer = "The bridge was built in 1920."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 40,
                       "text": "In 1953 the city opened a new library downtown."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[],
    )
    run = {"id": "run-guard", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    assert sdoc["results"][0]["status"] != "contradicted"


def test_both_claim_and_source_negated_is_agreement_not_contradiction():
    """A claim that itself says 'is not open' matching a source that also says 'is not open' is
    agreement, not contradiction -- the negation check only fires when the claim is affirmative."""
    answer = "The museum is not open on Mondays."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 60,
                       "text": "Please note the museum is not open on Mondays this season."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[],
    )
    run = {"id": "run-double-negative", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    assert sdoc["results"][0]["status"] != "contradicted"


def test_shared_number_is_not_a_mismatch():
    """The claim and source agreeing on the SAME number must never be flagged as contradicting."""
    answer = "The bridge was built in 1920."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 40,
                       "text": "Historical records confirm the bridge was built in 1920."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[],
    )
    run = {"id": "run-agree", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    assert sdoc["results"][0]["status"] != "contradicted"


# ======================================================================================================
# Required case: a hedged non-factual claim (category rule)
# ======================================================================================================

def test_hedged_non_factual_claim_is_unverifiable_by_category_rule():
    answer = "This might work, but I am not entirely sure."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 30, "text": "This might work in most cases."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[{"context_span_id": "ps-1", "answer_span_id": "as-1", "context_index": 0,
                "answer_index": 0, "delta_nats": -5.0, "abs_delta_nats": 5.0, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    run = {"id": "run-hedged", "response": answer, "influence_map": influence}
    cdoc = claims.build_answer_claims(run)
    assert cdoc["claims"][0]["category"] == "uncertainty_statement"

    sdoc = claim_support.build_claim_support(run, cdoc)
    result = sdoc["results"][0]
    # Even with a real causally_supported link sitting right there, the category rule wins --
    # verification never second-guesses a non-factual category on a per-claim basis.
    assert result["status"] == "unverifiable_from_available_evidence"
    assert result["method"] == {"name": "category_rule"}
    assert "source_span_ids" not in result


@pytest.mark.parametrize("answer", [
    "You should back up your data first.",
    "Restart the server to apply changes.",
    "This might work, but I am not sure.",
    "```python\nprint(1)\n```",
])
def test_every_non_factual_category_is_always_unverifiable_never_measured(answer):
    run = {"id": "run-category-sweep", "response": answer}
    cdoc = claims.build_answer_claims(run)
    sdoc = claim_support.build_claim_support(run, cdoc)
    for claim, result in zip(cdoc["claims"], sdoc["results"]):
        if claim["category"] != "factual_claim":
            assert result["status"] == "unverifiable_from_available_evidence"
            assert result["method"]["name"] == "category_rule"


def test_factual_claim_can_never_be_unverifiable_from_available_evidence():
    """The converse invariant: unverifiable_from_available_evidence is exclusively a category rule for
    non-factual claims -- a factual claim always lands on one of the other five statuses."""
    answer = "The Eiffel Tower was completed in 1889."
    for run in (
        {"id": "run-a", "response": answer},
        {"id": "run-b", "response": answer, "influence_map": _influence(status="error", available=False)},
    ):
        cdoc = claims.build_answer_claims(run)
        assert cdoc["claims"][0]["category"] == "factual_claim"
        sdoc = claim_support.build_claim_support(run, cdoc)
        assert sdoc["results"][0]["status"] != "unverifiable_from_available_evidence"


# ======================================================================================================
# Determinism
# ======================================================================================================

def test_deterministic_byte_identical_output(tmp_path):
    answer = ("The Eiffel Tower was completed in 1889. You should visit at sunset. "
              "The museum is not open on Mondays.")
    influence = _influence(
        prompt_spans=[
            {"id": "ps-1", "start": 0, "end": 45, "text": "The Eiffel Tower was completed in 1889."},
            {"id": "ps-2", "start": 46, "end": 90, "text": "Note the museum is not open on Mondays."},
        ],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": 40, "text": answer[0:40]}],
        links=[{"context_span_id": "ps-1", "answer_span_id": "as-1", "context_index": 0,
                "answer_index": 0, "delta_nats": -2.0, "abs_delta_nats": 2.0, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    run = {"id": "run-det", "response": answer, "influence_map": influence}

    first = claim_support.build_claim_support(copy.deepcopy(run), claims.build_answer_claims(copy.deepcopy(run)))
    second = claim_support.build_claim_support(copy.deepcopy(run), claims.build_answer_claims(copy.deepcopy(run)))

    path_a = tmp_path / "first.json"
    path_b = tmp_path / "second.json"
    path_a.write_bytes(_canonical(first))
    path_b.write_bytes(_canonical(second))

    assert path_a.read_bytes() == path_b.read_bytes()
    assert first == second


# ======================================================================================================
# Structural invariants
# ======================================================================================================

def test_run_and_claims_document_are_never_mutated():
    answer = "The bridge was built in 1920."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 30, "text": "The bridge dates to 1920."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[{"context_span_id": "ps-1", "answer_span_id": "as-1", "context_index": 0,
                "answer_index": 0, "delta_nats": -1.0, "abs_delta_nats": 1.0, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    run = {"id": "run-immutable", "response": answer, "influence_map": influence}
    cdoc = claims.build_answer_claims(run)
    run_before = copy.deepcopy(run)
    cdoc_before = copy.deepcopy(cdoc)
    claim_support.build_claim_support(run, cdoc)
    assert run == run_before
    assert cdoc == cdoc_before


def test_evidence_bearing_statuses_always_carry_non_empty_source_span_ids():
    answer = ("The Eiffel Tower was completed in 1889. The bridge was built in 1920. "
              "The museum is open on Mondays.")
    influence = _influence(
        prompt_spans=[
            {"id": "ps-1", "start": 0, "end": 45, "text": "The Eiffel Tower was completed in 1889."},
            {"id": "ps-2", "start": 46, "end": 90, "text": "The old bridge was built sometime in the 1920s."},
            {"id": "ps-3", "start": 91, "end": 130, "text": "The museum is not open on Mondays, we regret."},
        ],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": 40, "text": answer[:40]}],
        links=[{"context_span_id": "ps-1", "answer_span_id": "as-1", "context_index": 0,
                "answer_index": 0, "delta_nats": -2.0, "abs_delta_nats": 2.0, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    run = {"id": "run-evidence-check", "response": answer, "influence_map": influence}
    sdoc = claim_support.build_claim_support(run, claims.build_answer_claims(run))
    seen_statuses = set()
    for result in sdoc["results"]:
        seen_statuses.add(result["status"])
        if result["status"] in ("supported", "weakly_supported", "contradicted"):
            assert "source_span_ids" in result
            assert len(result["source_span_ids"]) >= 1
            assert all(sid.startswith("span_") for sid in result["source_span_ids"])
        else:
            assert "source_span_ids" not in result
    assert {"supported", "weakly_supported", "contradicted"} & seen_statuses, (
        "fixture should exercise at least one evidence-bearing status"
    )


def test_claim_index_and_address_id_round_trip_to_the_claims_document():
    answer = "The bridge was built in 1920. You should visit soon."
    run = {"id": "run-roundtrip", "response": answer}
    cdoc = claims.build_answer_claims(run)
    sdoc = claim_support.build_claim_support(run, cdoc)
    assert len(sdoc["results"]) == len(cdoc["claims"])
    for claim, result in zip(cdoc["claims"], sdoc["results"]):
        assert result["claim_index"] == claim["index"]
        assert result["claim_address_id"] == claim["text_span"]["address_id"]


def test_privacy_full_and_metadata_only_agree_on_status_and_citations():
    answer = "The Eiffel Tower was completed in 1889."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 45,
                       "text": "The Eiffel Tower was completed in 1889 in Paris."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[{"context_span_id": "ps-1", "answer_span_id": "as-1", "context_index": 0,
                "answer_index": 0, "delta_nats": -3.0, "abs_delta_nats": 3.0, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    run = {"id": "run-privacy", "response": answer, "influence_map": influence}
    cdoc = claims.build_answer_claims(run)
    metadata_only = claim_support.build_claim_support(run, cdoc, privacy="metadata_only")
    full = claim_support.build_claim_support(run, cdoc, privacy="full")
    assert metadata_only["results"][0]["status"] == full["results"][0]["status"]
    assert metadata_only["results"][0]["source_span_ids"] == full["results"][0]["source_span_ids"]


# ======================================================================================================
# Input validation
# ======================================================================================================

def test_build_claim_support_rejects_a_claims_document_for_a_different_run():
    run_a = {"id": "run-a", "response": "Text A."}
    run_b = {"id": "run-b", "response": "Text B."}
    cdoc_a = claims.build_answer_claims(run_a)
    with pytest.raises(ValueError):
        claim_support.build_claim_support(run_b, cdoc_a)


def test_build_claim_support_rejects_a_non_claims_document():
    run = {"id": "run-x", "response": "Text."}
    with pytest.raises(ValueError):
        claim_support.build_claim_support(run, {"schema_version": "clozn.something-else.v1"})


def test_build_claim_support_rejects_bad_privacy_value():
    run = {"id": "run-x", "response": "Text."}
    cdoc = claims.build_answer_claims(run)
    with pytest.raises(ValueError):
        claim_support.build_claim_support(run, cdoc, privacy="private")


def test_build_claim_support_requires_a_run_id():
    with pytest.raises(ValueError):
        claim_support.build_claim_support({"response": "Text."}, {"schema_version": "clozn.answer-claims.v1",
                                                                   "run_id": "", "claims": []})


@pytest.mark.parametrize("run,influence_kwargs", [
    ({"id": "run-e", "response": "Plain text with no markers at all here."}, None),
    ({"id": "run-f", "response": ""}, None),
    ({"id": "run-g"}, None),
    ({"id": "run-h", "response": "secret", "redaction": {"status": "redacted"}}, None),
])
def test_build_claim_support_result_always_validates(run, influence_kwargs):
    cdoc = claims.build_answer_claims(run)
    document = claim_support.build_claim_support(run, cdoc)
    schemas.validate(document)
