"""test_diagnosis_narratives -- clozn/runs/diagnosis_narratives.py (`clozn.diagnosis-narrative.v1`, D2):
plain-language "Why?" narratives over D1's `clozn.diagnosis-findings.v1` plus a structural run comparison.

Model-free throughout: every run/comparison_run/findings fixture is a hand-built dict shaped like a real
stored run record -- no engine, no GPU, no network, no filesystem I/O (the module itself never touches
disk).
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
from clozn.runs import diagnosis_narratives as dn  # noqa: E402
from clozn.runs import diagnosis_rules as dr  # noqa: E402


# ==================================================================================== fixture builders

def _msg(role: str, content: str, source_id: "str | None" = None) -> dict:
    out = {"role": role, "content": content}
    if source_id:
        out["source_id"] = source_id
    return out


def _run(*, id="run_1", messages=None, response="ok", finish_reason=None, identity=None, meta=None,
        influence_map=None) -> dict:
    out: dict = {"id": id, "messages": messages if messages is not None else [_msg("user", "hi")],
                "response": response}
    if finish_reason is not None:
        out["finish_reason"] = finish_reason
    if identity is not None:
        out["identity"] = identity
    if meta is not None:
        out["meta"] = meta
    if influence_map is not None:
        out["influence_map"] = influence_map
    return out


_INFLUENCE_METHOD = {"name": "x", "mode": "forced_score_intervention", "claim_limit": "x", "caveat": "x"}


def _influence_map(*, thresholds, links, prompt_spans) -> dict:
    return {"schema": "clozn.context_answer_influence.v1", "status": "ok", "available": True,
           "method": dict(_INFLUENCE_METHOD), "identity": {}, "thresholds": thresholds,
           "prompt_spans": prompt_spans, "links": links}


def _link(context_span_id, *, clears_floor, abs_delta_nats) -> dict:
    return {"context_span_id": context_span_id, "answer_span_id": "as1", "context_index": 0,
           "answer_index": 0, "delta_nats": abs_delta_nats, "abs_delta_nats": abs_delta_nats,
           "effect": "supports", "clears_floor": clears_floor,
           "evidence_state": "causally_supported" if clears_floor else "observed"}


def _below_floor_run(id="run_influ") -> dict:
    return _run(
        id=id,
        messages=[_msg("user", "Doc A text here.", "docA"), _msg("user", "what does doc A say")],
        influence_map=_influence_map(
            thresholds={"cell_abs_delta_nats": 0.05},
            prompt_spans=[{"id": "ps1", "start": 0, "end": 10, "text": "Doc A text", "client_source_id": "docA"}],
            links=[_link("ps1", clears_floor=False, abs_delta_nats=0.01)]))


# ==================================================================================== schema plumbing

def test_narrate_with_no_comparison_run_validates():
    doc = dn.narrate(_run(), generated_at="2026-07-29T00:00:00Z")
    schemas.validate(doc)
    assert doc["schema_version"] == "clozn.diagnosis-narrative.v1"
    assert doc["comparison_available"] is False
    assert "comparison_run_id" not in doc
    assert doc["registers"] == {"observed_changes": [], "measured_effects": [], "plausible_but_unproven": []}


def test_narrate_findings_schema_version_matches_diagnosis_rules():
    doc = dn.narrate(_run(), generated_at="2026-07-29T00:00:00Z")
    assert doc["findings_schema_version"] == dr.SCHEMA_VERSION


def test_narrate_never_raises_on_malformed_inputs():
    for run in (None, "not a dict", 42, {}):
        for comparison in (None, "nope", 7):
            for findings in (None, "nope", {"schema_version": "wrong"}):
                doc = dn.narrate(run, comparison_run=comparison, findings=findings,
                                 generated_at="2026-07-29T00:00:00Z")
                schemas.validate(doc)


def test_narrate_reuses_precomputed_findings_without_recomputing(monkeypatch):
    """The route computes findings ONCE and hands them to narrate() -- narrate() must use them AS GIVEN,
    never silently recompute (which could disagree, or double the work)."""
    run = _run(messages=[_msg("system", "Always X."), _msg("user", "Never X.")])
    findings = dr.evaluate(run, generated_at="2026-07-29T00:00:00Z")

    def _boom(*_a, **_kw):
        raise AssertionError("evaluate() must not be called when findings= was supplied")

    monkeypatch.setattr(dr, "evaluate", _boom)
    doc = dn.narrate(run, findings=findings, generated_at="2026-07-29T00:00:00Z")
    schemas.validate(doc)
    assert any(e["rule_id"] == "R03" for e in doc["registers"]["measured_effects"])


def test_narrate_computes_findings_when_not_supplied():
    run = _run(finish_reason="length")
    doc = dn.narrate(run, generated_at="2026-07-29T00:00:00Z")
    assert any(e["rule_id"] == "R11" for e in doc["registers"]["measured_effects"])


def test_narrate_ignores_findings_with_a_different_schema_version():
    run = _run(finish_reason="length")
    doc = dn.narrate(run, findings={"schema_version": "clozn.run_diagnosis.v1", "run_id": "run_1"},
                     generated_at="2026-07-29T00:00:00Z")
    # falls back to computing its own -- R11 (length cutoff) is present either way
    assert any(e["rule_id"] == "R11" for e in doc["registers"]["measured_effects"])


# ==================================================================================== determinism (REQUIRED)

def test_narrate_twice_is_byte_identical_no_comparison():
    run = _run(finish_reason="length",
              messages=[_msg("system", "Always answer in English."), _msg("user", "Never answer in English.")])
    first = dn.narrate(run, generated_at="2026-07-29T00:00:00Z")
    second = dn.narrate(run, generated_at="2026-07-29T00:00:00Z")
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_narrate_twice_is_byte_identical_with_comparison():
    run_a = _run(id="run_a", identity={"model_sha256": "a" * 64}, meta={"temperature": 0.2})
    run_b = _run(id="run_b", identity={"model_sha256": "b" * 64}, meta={"temperature": 0.9})
    first = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    second = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_narrate_twice_is_byte_identical_with_influence_map():
    run = _below_floor_run()
    first = dn.narrate(run, generated_at="2026-07-29T00:00:00Z")
    second = dn.narrate(run, generated_at="2026-07-29T00:00:00Z")
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# ==================================================================================== the canonical trap

def test_temperature_and_output_change_stay_two_separate_observed_facts_never_a_causal_claim():
    """REQUIRED PROOF, the spec's own example verbatim: 'Temperature changed from 0 to 0.8 AND the output
    diverged' must land as TWO separate registers.observed_changes entries. No entry anywhere may claim
    the output diverged BECAUSE temperature changed."""
    run_a = _run(id="run_a", meta={"temperature": 0.0}, response="Paris.")
    run_b = _run(id="run_b", meta={"temperature": 0.8}, response="Paris, the capital of France.")
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")

    observed_dimensions = {e["dimension"] for e in doc["registers"]["observed_changes"]}
    assert "generation.temperature" in observed_dimensions
    assert "output.text" in observed_dimensions
    # the temperature-change sentence names ONLY temperature -- it never mentions the output at all.
    temp_entry = next(e for e in doc["registers"]["observed_changes"] if e["dimension"] == "generation.temperature")
    assert "output" not in temp_entry["text"]
    assert "response" not in temp_entry["text"]
    # no plausible-but-unproven bridge is manufactured for this case either -- a real setting difference
    # was found, so the narrower "identical settings, different output" trigger must not fire.
    assert doc["registers"]["plausible_but_unproven"] == []


def test_quantization_style_identity_change_and_output_change_stay_separate():
    run_a = _run(id="run_a", identity={"model_sha256": "a" * 64}, response="Paris.")
    run_b = _run(id="run_b", identity={"model_sha256": "b" * 64}, response="The capital of France is Paris.")
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    dims = {e["dimension"] for e in doc["registers"]["observed_changes"]}
    assert "identity.model_sha256" in dims
    assert "output.text" in dims
    identity_entry = next(e for e in doc["registers"]["observed_changes"] if e["dimension"] == "identity.model_sha256")
    assert "output" not in identity_entry["text"]


_BANNED_CAUSAL_WORDS = ("because", "caused", "causes", "causing", "due to", "responsible for",
                        "leads to", "results in", "the reason")


def test_no_causal_vocabulary_in_narratives_built_from_structural_evidence_alone():
    """REQUIRED PROOF: scans actual generated narrative text (not just the module source) across a range
    of scenarios -- comparison-only, findings-only, both, and the randomness-sensitive case -- for the
    banned causal vocabulary."""
    scenarios = [
        dn.narrate(_run(id="run_b", meta={"temperature": 0.8}),
                  comparison_run=_run(id="run_a", meta={"temperature": 0.0}),
                  generated_at="2026-07-29T00:00:00Z"),
        dn.narrate(_run(finish_reason="length",
                        messages=[_msg("system", "Always X."), _msg("user", "Never X.")]),
                  generated_at="2026-07-29T00:00:00Z"),
        dn.narrate(_run(id="run_b", identity={"model_sha256": "a" * 64}, response="different reply"),
                  comparison_run=_run(id="run_a", identity={"model_sha256": "a" * 64}, response="reply"),
                  generated_at="2026-07-29T00:00:00Z"),
        dn.narrate(_below_floor_run(), generated_at="2026-07-29T00:00:00Z"),
    ]
    for document in scenarios:
        text = json.dumps(document).lower()
        for word in _BANNED_CAUSAL_WORDS:
            assert word not in text, f"causal vocabulary {word!r} leaked into a narrative"


def test_module_source_never_contains_causal_vocabulary():
    path = os.path.join(REPO_ROOT, "clozn", "runs", "diagnosis_narratives.py")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    quoted_mentions = ('"because"', '"caused"', '"causes"', '"causing"', '"due to"', '"responsible for"',
                       '"leads to"', '"results in"', '"the reason"')
    remainder = text
    for mention in quoted_mentions:
        assert mention in remainder, f"expected the module docstring to name {mention} as banned vocabulary"
        remainder = remainder.replace(mention, "", 1)
    lowered = remainder.lower()
    for word in _BANNED_CAUSAL_WORDS:
        assert word not in lowered, f"causal vocabulary {word!r} used outside its one quoted mention"


# ==================================================================================== randomness sensitivity

def test_identical_identity_and_settings_with_output_divergence_yields_plausible_not_a_factor():
    """REQUIRED PROOF (spec, verbatim): identical run identities with residual divergence => the
    narrative states the difference MAY be sampling-sensitive, never a fabricated ranked factor."""
    run_a = _run(id="run_a", identity={"model_sha256": "a" * 64}, meta={"temperature": 0.0}, response="Paris.")
    run_b = _run(id="run_b", identity={"model_sha256": "a" * 64}, meta={"temperature": 0.0},
                 response="The capital is Paris.")
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    assert doc["registers"]["measured_effects"] == []
    plausible = doc["registers"]["plausible_but_unproven"]
    assert len(plausible) == 1
    assert "sampling-sensitive" in plausible[0]["text"]
    assert "not backed by a" in plausible[0]["note"]
    assert plausible[0]["evidence"]
    assert "rank" not in plausible[0]
    assert "severity" not in plausible[0]
    assert "confidence" not in plausible[0]
    assert "rule_id" not in plausible[0]


def test_no_plausible_entry_when_output_is_identical():
    run_a = _run(id="run_a", identity={"model_sha256": "a" * 64}, response="Paris.")
    run_b = _run(id="run_b", identity={"model_sha256": "a" * 64}, response="Paris.")
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    assert doc["registers"]["plausible_but_unproven"] == []


def test_no_plausible_entry_without_a_comparison_run():
    doc = dn.narrate(_run(), generated_at="2026-07-29T00:00:00Z")
    assert doc["registers"]["plausible_but_unproven"] == []


# ==================================================================================== measured_effects: ranking

def test_no_finding_no_factor_a_bare_structural_difference_is_never_ranked():
    """REQUIRED PROOF: a structural difference with NO applicable D1 finding backing it never appears in
    registers.measured_effects, no matter how suggestive it looks."""
    run_a = _run(id="run_a", meta={"top_p": 0.9})
    run_b = _run(id="run_b", meta={"top_p": 0.5})
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    # top_p differs (an observed change)...
    assert any(e["dimension"] == "generation.top_p" for e in doc["registers"]["observed_changes"])
    # ...and IS backed by R12 (run_to_run_drift), so it DOES legitimately rank -- but ONLY via that named
    # finding, never as a bare, un-backed structural difference. Confirm every ranked entry names a real
    # rule_id.
    for entry in doc["registers"]["measured_effects"]:
        assert entry["rule_id"] in dr.RULE_IDS


def test_every_measured_effect_rule_id_is_a_real_finding_status_finding_entry():
    run = _run(finish_reason="length",
              messages=[_msg("system", "Always X."), _msg("user", "Never X.")])
    findings = dr.evaluate(run, generated_at="2026-07-29T00:00:00Z")
    doc = dn.narrate(run, findings=findings, generated_at="2026-07-29T00:00:00Z")
    findings_by_id = {f["rule_id"]: f for f in findings["findings"]}
    assert doc["registers"]["measured_effects"]
    for entry in doc["registers"]["measured_effects"]:
        assert findings_by_id[entry["rule_id"]]["status"] == "finding"


def test_measured_effects_are_ranked_by_severity_then_confidence():
    run = _run(finish_reason="length",   # R11: severity high, confidence exact
              messages=[_msg("system", "Always X."), _msg("user", "Never X.")])  # R03: severity medium, pattern_match
    doc = dn.narrate(run, generated_at="2026-07-29T00:00:00Z")
    ranks = [e["rank"] for e in doc["registers"]["measured_effects"]]
    assert ranks == list(range(1, len(ranks) + 1))
    severities = [e["severity"] for e in doc["registers"]["measured_effects"]]
    severity_order = {name: i for i, name in enumerate(dr.SEVERITY_VALUES)}
    assert severities == sorted(severities, key=lambda s: -severity_order[s])
    rule_ids = [e["rule_id"] for e in doc["registers"]["measured_effects"]]
    assert rule_ids[0] == "R11"   # high severity ranks first


def test_measured_effects_empty_when_no_findings_fire():
    doc = dn.narrate(_run(), generated_at="2026-07-29T00:00:00Z")
    assert doc["registers"]["measured_effects"] == []


# ==================================================================================== basis: intervention vs rule_finding

def test_intervention_backed_rules_get_intervention_basis():
    doc = dn.narrate(_below_floor_run(), generated_at="2026-07-29T00:00:00Z")
    entry = next(e for e in doc["registers"]["measured_effects"] if e["rule_id"] == "R08")
    assert entry["basis"] == "intervention"


def test_ordinary_rule_findings_get_rule_finding_basis():
    run = _run(finish_reason="length")
    doc = dn.narrate(run, generated_at="2026-07-29T00:00:00Z")
    entry = next(e for e in doc["registers"]["measured_effects"] if e["rule_id"] == "R11")
    assert entry["basis"] == "rule_finding"


# ==================================================================================== evidence linking

def test_every_observed_change_has_evidence():
    run_a = _run(id="run_a", meta={"temperature": 0.0})
    run_b = _run(id="run_b", meta={"temperature": 0.8})
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    for entry in doc["registers"]["observed_changes"]:
        assert entry["evidence"]
        assert entry["evidence"][0] == {"kind": "diff_field", "dimension": entry["dimension"]}


def test_every_measured_effect_has_finding_evidence():
    run = _run(finish_reason="length")
    doc = dn.narrate(run, generated_at="2026-07-29T00:00:00Z")
    for entry in doc["registers"]["measured_effects"]:
        assert {"kind": "finding", "rule_id": entry["rule_id"]} in entry["evidence"]


def test_measured_effect_propagates_text_span_evidence_from_the_finding():
    doc = dn.narrate(_below_floor_run(), generated_at="2026-07-29T00:00:00Z")
    entry = next(e for e in doc["registers"]["measured_effects"] if e["rule_id"] == "R08")
    span_refs = [e for e in entry["evidence"] if e["kind"] == "text_span"]
    assert span_refs
    assert span_refs[0]["address_id"].startswith("span_")


def test_every_plausible_entry_has_evidence():
    run_a = _run(id="run_a", identity={"model_sha256": "a" * 64}, response="Paris.")
    run_b = _run(id="run_b", identity={"model_sha256": "a" * 64}, response="The capital is Paris.")
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    for entry in doc["registers"]["plausible_but_unproven"]:
        assert entry["evidence"]


# ==================================================================================== observed-change rendering

@pytest.mark.parametrize("run_a_kwargs,run_b_kwargs,expected_dimension,expected_substring", [
    (dict(identity={"model_sha256": "a" * 64}), dict(identity={"model_sha256": "b" * 64}),
     "identity.model_sha256", "model file changed"),
    (dict(meta={"seed": 1}), dict(meta={"seed": 2}), "generation.seed", "seed changed"),
    (dict(finish_reason="stop"), dict(finish_reason="length"), "output.finish_reason", "finish_reason changed"),
])
def test_observed_change_dimension_templates(run_a_kwargs, run_b_kwargs, expected_dimension, expected_substring):
    run_a = _run(id="run_a", **run_a_kwargs)
    run_b = _run(id="run_b", **run_b_kwargs)
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    entry = next(e for e in doc["registers"]["observed_changes"] if e["dimension"] == expected_dimension)
    assert expected_substring in entry["text"]


def test_observed_change_unknown_dimension_falls_back_to_generic_template(monkeypatch):
    """A forward-compatible identity.ext.* facet (or any dimension this module has never named) still
    gets a plain, non-crashing sentence -- never a KeyError, never a blank."""
    text = dn._observed_text({"dimension": "identity.ext.some_future_facet", "kind": "changed",
                              "value_a": "x", "value_b": "y"})
    assert "some_future_facet changed" in text


def test_observed_change_unavailable_kind_states_it_plainly():
    text = dn._observed_text({"dimension": "context.limits.prompt_tokens", "kind": "unavailable"})
    assert "could not be compared" in text


# ==================================================================================== headline

def test_headline_no_comparison_no_findings():
    doc = dn.narrate(_run(), generated_at="2026-07-29T00:00:00Z")
    assert doc["headline"] == "no ranked findings."


def test_headline_mentions_comparison_and_ranked_counts():
    run_a = _run(id="run_a", meta={"temperature": 0.0})
    run_b = _run(id="run_b", meta={"temperature": 0.8})
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    assert "structural difference" in doc["headline"]
    assert "ranked finding" in doc["headline"]


def test_headline_mentions_plausible_notes():
    run_a = _run(id="run_a", identity={"model_sha256": "a" * 64}, response="Paris.")
    run_b = _run(id="run_b", identity={"model_sha256": "a" * 64}, response="The capital is Paris.")
    doc = dn.narrate(run_b, comparison_run=run_a, generated_at="2026-07-29T00:00:00Z")
    assert "plausible-but-unproven" in doc["headline"]


# ==================================================================================== comparison_run_id

def test_comparison_run_id_present_only_when_comparison_supplied():
    doc_without = dn.narrate(_run(), generated_at="2026-07-29T00:00:00Z")
    assert "comparison_run_id" not in doc_without
    doc_with = dn.narrate(_run(id="run_b"), comparison_run=_run(id="run_a"),
                          generated_at="2026-07-29T00:00:00Z")
    assert doc_with["comparison_run_id"] == "run_a"


def test_comparison_run_without_an_id_omits_comparison_run_id():
    doc = dn.narrate(_run(id="run_b"), comparison_run={"messages": []}, generated_at="2026-07-29T00:00:00Z")
    assert doc["comparison_available"] is True
    assert "comparison_run_id" not in doc


# ==================================================================================== fixture cross-check

def test_fixture_documents_still_validate():
    fixture_dir = os.path.join(REPO_ROOT, "tests", "fixtures", "schemas", "clozn.diagnosis-narrative.v1")
    found = 0
    for name in os.listdir(fixture_dir):
        if name.startswith("valid__"):
            with open(os.path.join(fixture_dir, name), encoding="utf-8") as handle:
                document = json.load(handle)
            schemas.validate(document)
            found += 1
    assert found >= 3
