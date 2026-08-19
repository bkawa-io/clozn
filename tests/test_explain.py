"""test_explain -- model-free tests for research/explain.py (EXPLAIN_THIS_ANSWER_SPEC.md Milestone 1).

Drives explain.explain() directly against fixture run dicts. Most are built through the REAL
runlog.record() + get_run() round trip so the trace/memory/behavior shapes are byte-for-byte what the
real logging paths persist, not a hand-rolled guess at the schema (mirrors test_runlog.py's own `store`
fixture); a couple are hand-built on purpose, to exercise shapes runlog itself can't currently produce (a
is isolated to a tmp file (mirrors test_profiles_server.py's `iso` fixture) so provenance lookups are real
card-store reads, not mocks.

The invariants under test are the spec's honesty invariants, not just "does it return something":
  * no aggregate confidence number ever appears anywhere in the returned object -- a recursive scan of
    every dict (not just the top-level "confidence" key), since the whole point is that this can never
    sneak back in under a different name;
  * every active-influence entry (card or dial) carries causal_verified: null;
  * a missing signal is an explicit {"available": false, "note": ...} field, never a silently-absent key.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.receipts.explain as explain          # noqa: E402

import clozn.runs.store as runlog            # noqa: E402


# --- isolation: point both flat-file stores this module touches at tmp paths for the duration of a test --

@pytest.fixture
def store(tmp_path, monkeypatch):
    return runlog


# --- a recursive scan for the banned "aggregate confidence" shape -----------------------------------------

_BANNED_KEYS = {"confidence_pct", "confidence_score", "avg_confidence", "average_confidence",
                "overall_confidence", "aggregate_confidence", "mean_confidence", "confidence_percent",
                "confidence_percentage"}


def _assert_no_aggregate_confidence(obj, path="explanation"):
    """Walk the WHOLE returned object -- not just the top-level "confidence" key -- and assert no
    aggregate-confidence-shaped key exists anywhere. The dead scalar self-report probe (EXPLAIN_THIS_
    ANSWER_SPEC.md's principle section) must never sneak back in under a
    different key name, nested anywhere in the tree."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert str(k).lower() not in _BANNED_KEYS, f"aggregate-confidence-shaped key at {path}.{k}"
            _assert_no_aggregate_confidence(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_aggregate_confidence(v, f"{path}[{i}]")


# ------------------------------------------------------------------------------------- fixture: with-trace

def test_confidence_with_trace_finds_uncertain_moments_and_their_alternatives(store):
    tokens = ["The", " sky", " is", " blue", "."]
    confidence = [0.95, 0.30, 0.92, 0.41, 0.99]
    alternatives = [[], [{"piece": " sea", "prob": 0.22}], [],
                    [{"piece": " grey", "prob": 0.31}, {"piece": " green", "prob": 0.10}], []]
    rid = store.record(source="engine_chat", model="clozn-qwen",
                       messages=[{"role": "user", "content": "what color is the sky?"}],
                       response="The sky is blue.",
                       trace={"tokens": tokens, "confidence": confidence, "alternatives": alternatives})
    run = store.get_run(rid)

    out = explain.explain(run)
    conf = out["confidence"]
    assert conf["available"] is True
    assert conf["threshold"] == explain.LOW_CONF == 0.5
    assert conf["n_tokens"] == 5
    # exactly the two tokens below 0.5, in order, each carrying its recorded alternatives
    assert [u["index"] for u in conf["uncertain_moments"]] == [1, 3]
    assert conf["uncertain_moments"][0]["token"] == " sky"
    assert conf["uncertain_moments"][0]["confidence"] == 0.30
    assert conf["uncertain_moments"][0]["alternatives"] == [{"piece": " sea", "prob": 0.22}]
    assert conf["uncertain_moments"][1]["alternatives"][1]["piece"] == " green"
    assert conf["summary"] == "2 hesitations"   # the one-line "N hesitations" count


def test_confidence_summary_pluralizes_correctly_at_zero_and_one(store):
    rid0 = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                        trace={"tokens": ["hey"], "confidence": [0.99]})
    assert explain.explain(store.get_run(rid0))["confidence"]["summary"] == "0 hesitations"

    rid1 = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="uh, hey",
                        trace={"tokens": ["uh", ", hey"], "confidence": [0.2, 0.9]})
    assert explain.explain(store.get_run(rid1))["confidence"]["summary"] == "1 hesitation"


def test_confidence_tolerates_a_token_with_no_alternatives_recorded(store):
    """Only SOME tokens get alternatives in practice (runlog.steps_to_trace only stores the `alternatives`
    key at all when at least one step had some); a token with none must default to [], not KeyError."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
                       trace={"tokens": ["a", "b"], "confidence": [0.1, 0.9]})   # no alternatives key at all
    conf = explain.explain(store.get_run(rid))["confidence"]
    assert conf["available"] is True
    assert conf["uncertain_moments"][0]["alternatives"] == []


# ---------------------------------------------------------------------------------- fixture: without-trace



# ------------------------------------------------------------------------------- fixture: with-cards+provenance















# ---------------------------------------------------------------------------------------- fixture: with-dials



# --------------------------------------------------------------------------------------- fixture: sections

def test_sections_active_lists_the_manifest_with_causal_verified_null(store):
    """Forward-compatible contract (mirrors the concepts fixture below): no current path threads `sections`
    through runlog.record() itself (that producer -- clozn.runs.sections/store -- lands separately), so this
    constructs the field by hand on a fetched run, exactly like test_concepts_available_when_the_run_carries_
    sae_readouts does for trace["concepts"]. Proves the ASSEMBLY side of the contract."""
    rid = store.record(source="engine_chat", messages=[{"role": "user", "content": "q"}], response="a")
    run = store.get_run(rid)
    run["sections"] = [
        {"id": "sec_rag", "name": "rag_context", "source": "auto",
         "parts": [{"message_index": 0, "start": 0, "end": 5}],
         "char_count": 5, "preview": "hello"},
        {"id": "sec_sys", "name": "system_prompt", "source": "api",
         "parts": [{"message_index": None, "start": 0, "end": 10}],
         "char_count": 10, "preview": "You are.."},
    ]
    out = explain.explain(run)
    sections = out["influences_active"]["sections"]
    assert sections["available"] is True
    names = {s["name"] for s in sections["sections"]}
    assert names == {"rag_context", "system_prompt"}
    assert all(s["causal_verified"] is None for s in sections["sections"])
    rag = next(s for s in sections["sections"] if s["name"] == "rag_context")
    assert rag["source"] == "auto"
    assert rag["char_count"] == 5
    assert rag["preview"] == "hello"
    assert rag["id"] == "sec_rag"


def test_sections_active_absent_manifest_degrades_explicitly(store):
    """A run predating section capture carries no `sections` key at all -- an explicit
    {"available": false, "note": ...}, never a silently-missing key or a misleadingly empty list."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    run = store.get_run(rid)
    assert "sections" not in run             # confirms runlog really stored nothing (no producer wired yet)
    sections = explain.explain(run)["influences_active"]["sections"]
    assert sections == {"available": False, "note": explain._NO_SECTIONS_NOTE}


def test_sections_active_tolerates_a_malformed_manifest(store):
    """Guard field-by-field like every other panel: a `sections` value that isn't a list degrades to the
    honest unavailable shape; a list with junk entries mixed in keeps only the well-formed ones."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    run = store.get_run(rid)

    run["sections"] = "not-a-list"
    assert explain.explain(run)["influences_active"]["sections"] == {
        "available": False, "note": explain._NO_SECTIONS_NOTE}

    run["sections"] = ["not-a-dict", 42, {"id": "sec_ok", "name": "ok", "source": "auto"}]
    secs = explain.explain(run)["influences_active"]["sections"]
    assert secs["available"] is True
    assert len(secs["sections"]) == 1
    assert secs["sections"][0]["name"] == "ok"
    assert secs["sections"][0]["causal_verified"] is None


def test_no_aggregate_confidence_field_still_holds_with_a_sections_manifest(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    run = store.get_run(rid)
    run["sections"] = [{"id": "sec_a", "name": "a", "source": "auto", "char_count": 3, "preview": "abc"}]
    _assert_no_aggregate_confidence(explain.explain(run))


# ----------------------------------------------------------------------------------------- fixture: concepts

def test_concepts_unavailable_on_an_ordinary_run(store):
    """Honest as of today: NO logging path threads sae:<id> readouts onto the stored run (runlog.TRACE_KEYS
    doesn't carry a concepts slot), so a completely ordinary run -- even with a full trace -- must report
    the explicit unavailable note, never a silently-missing key."""
    rid = store.record(source="engine_chat", messages=[{"role": "user", "content": "q"}], response="a",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    out = explain.explain(store.get_run(rid))
    assert out["concepts"] == {"available": False,
                               "note": "no named concept readout was captured for this run."}


def test_concepts_available_when_the_run_carries_sae_readouts(store):
    """Forward-compatible contract: IF a run's trace ever carries a `concepts` list (the engine's sae:<id>
    StepFeatures readouts, per-span), assembly surfaces the top features per span, sorted by score, capped
    at 5. No current producer builds this shape (see explain._concepts' docstring) -- constructed by hand
    here, mutating a fetched run, to prove the assembler's side of the forward-compatible contract."""
    rid = store.record(source="engine_chat", messages=[{"role": "user", "content": "tell me about dragons"}],
                       response="Dragons are mythical.",
                       trace={"tokens": ["Dragons"], "confidence": [0.9]})
    run = store.get_run(rid)
    run["trace"]["concepts"] = [
        {"position": 0, "piece": "Dragons", "features": [
            {"id": "sae:9001", "label": "dragon", "score": 0.83},
            {"id": "sae:42", "label": "mythical-creature", "score": 0.91},
            {"id": "sae:7", "label": "low-relevance", "score": 0.01},
            {"id": "sae:8", "label": "low-relevance-2", "score": 0.02},
            {"id": "sae:9", "label": "low-relevance-3", "score": 0.03},
            {"id": "sae:10", "label": "low-relevance-4", "score": 0.04},
        ]},
    ]
    out = explain.explain(run)
    assert out["concepts"]["available"] is True
    span = out["concepts"]["spans"][0]
    assert span["position"] == 0 and span["piece"] == "Dragons"
    assert len(span["features"]) == 5                             # capped to top 5
    assert span["features"][0]["id"] == "sae:42"                  # sorted by score descending (0.91 first)
    assert span["features"][1]["id"] == "sae:9001"


# --------------------------------------------------------------------------------------- fixture: empty run







# ------------------------------------------------------------------------- the honesty invariants, globally

@pytest.mark.parametrize("run_kwargs", [
    dict(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
         trace={"tokens": ["a", "b"], "confidence": [0.1, 0.9]}),
    dict(source="studio_chat", messages=[{"role": "user", "content": "q"}], response="a",
),
    dict(source="cli", messages=[{"role": "user", "content": "q"}], response="a"),
])
def test_no_aggregate_confidence_field_ever_appears(store, run_kwargs):
    rid = store.record(**run_kwargs)
    out = explain.explain(store.get_run(rid))
    assert isinstance(out["confidence"], dict)     # never a bare scalar masquerading as "the" confidence
    _assert_no_aggregate_confidence(out)


def test_no_aggregate_confidence_field_on_empty_and_garbage_input():
    _assert_no_aggregate_confidence(explain.explain({}))
    _assert_no_aggregate_confidence(explain.explain(None))




def test_explanation_top_level_shape(store):
    """The named panels (spec's M1 bullet list: confidence, influences, concepts) plus `forks` (the
    close-call locator) and a run_id for traceability -- nothing more, nothing silently missing."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    out = explain.explain(store.get_run(rid))
    assert set(out.keys()) == {"run_id", "confidence", "influences_active", "concepts", "forks"}
    assert out["run_id"] == rid
