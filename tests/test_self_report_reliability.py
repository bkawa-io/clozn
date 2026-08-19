"""test_self_report_reliability -- model-free tests for clozn/receipts/self_report_reliability.py (X1: grade
the self-report against the CAUSAL receipt, not against presence).

Drives causal_explanation()/classify_run() directly against FIXTURE manifest/prove dicts (the exact shapes
explain.explain() and core.prove_all() return) plus a FAKE, deterministic support_matcher -- no model, no
GPU, no NLI checkpoint -- so every classification is driven purely by which keyword the fake matcher
recognizes in a claim, never by anything probabilistic. Mirrors test_receipts.py's/test_explain.py's own
model-free fixture-driven style; the REAL NLI judge's separating power is already proven in
test_semantic_matcher_gated.py (gated behind -m model) -- this file is strictly about classify_run's own
taxonomy logic (does it wire faithful/confabulated/missed/silent up correctly), not the judge underneath it.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import clozn.receipts.self_report_reliability as srr   # noqa: E402


# --- a deterministic, keyword-driven FAKE matcher (no NLI, no torch) -----------------------------------------
# Recognizes "vegetarian" against a card whose text mentions it, "teal" against a card whose text mentions
# it -- so a test can hand-write a self-report and know EXACTLY which taxonomy bucket each claim must land
# in, with no probabilistic judge in the loop at all.

def _fake_matcher(claim: str, explanation: dict) -> dict:
    claim_l = (claim or "").lower()
    infl = (explanation or {}).get("influences_active") or {}
    for c in infl.get("cards") or []:
        text = (c.get("text") or "").lower()
        for kw in ("vegetarian", "teal"):
            if kw in claim_l and kw in text:
                return {"supported": True, "score": 0.91, "matched_id": c.get("id"), "method": "fake-nli"}
    return {"supported": False, "score": 0.05, "matched_id": None, "method": "fake-nli"}


# --- fixtures: an explain.explain()-shaped manifest + a core.prove_all()-shaped receipt set -------------------
# card_a (vegetarian) is LOAD-BEARING (has_effect True); card_b (teal) is a PASSENGER (present, has_effect
# False); card_c rode along in the manifest but was never even attempted (no receipt at all) -- it must land
# in NEITHER set, since nothing was actually proven about it.

RUN = {"id": "run_x1_demo", "messages": [{"role": "user", "content": "suggest a dinner recipe for me"}],
      "response": "Here's a lentil stew recipe."}

MANIFEST = {
    "run_id": "run_x1_demo",
    "confidence": {"available": False},
    "influences_active": {
        "cards": [
            {"id": "card_a", "text": "The user is vegetarian.", "quoted_span": "I don't eat meat",
             "causal_verified": None, "has_provenance": True},
            {"id": "card_b", "text": "The user's favorite color is teal.", "quoted_span": "I love teal",
             "causal_verified": None, "has_provenance": True},
            {"id": "card_c", "text": "The user lives in a rainy city.", "quoted_span": "always raining here",
             "causal_verified": None, "has_provenance": True},
        ],
        "gate": 0.9, "mode": "prompt",
    },
    "concepts": {"available": False},
}

PROVE = {
    "run_id": "run_x1_demo",
    "receipts": [
        {"influence": {"card_id": "card_a", "text": "The user is vegetarian."}, "has_effect": True,
         "causal_verified": True, "baseline_reply": "Here's a lentil stew recipe.",
         "ablated_reply": "Here's a beef stew recipe."},
        {"influence": {"card_id": "card_b", "text": "The user's favorite color is teal."}, "has_effect": False,
         "causal_verified": True, "baseline_reply": "Here's a lentil stew recipe.",
         "ablated_reply": "Here's a lentil stew recipe."},
    ],
    "skipped": [], "redundant_pairs": [],
}


# ================================================================================== causal_explanation (1a)


def test_causal_explanation_excludes_a_card_with_no_receipt_at_all():
    """card_c is in the presence manifest but never appears in prove_all's receipts (e.g. skipped) --
    nothing was proven about it, so it must not show up as load-bearing OR be silently dropped-as-passenger."""
    ce = srr.causal_explanation(RUN, None, manifest=MANIFEST, prove=PROVE)
    ids = {c["id"] for c in ce["influences_active"]["cards"]}
    assert "card_c" not in ids


def test_causal_explanation_notes_the_degenerate_no_load_bearing_case():
    prove_no_effect = {"run_id": "run_x1_demo", "receipts": [
        {"influence": {"card_id": "card_a", "text": "x"}, "has_effect": False, "causal_verified": True},
    ], "skipped": [], "redundant_pairs": []}
    manifest = {"run_id": "run_x1_demo", "confidence": {"available": False},
               "influences_active": {"cards": [{"id": "card_a", "text": "x"}]},
               "concepts": {"available": False}}
    ce = srr.causal_explanation(RUN, None, manifest=manifest, prove=prove_no_effect)
    assert ce["influences_active"]["cards"] == []
    assert "no load-bearing influence" in ce["influences_active"]["note"]


def test_causal_explanation_never_raises_on_garbage_input():
    ce = srr.causal_explanation(None, None, manifest=None, prove=None)
    assert ce["influences_active"]["cards"] == []
    ce2 = srr.causal_explanation("not a dict", None, manifest={}, prove={})
    assert ce2["influences_active"]["cards"] == []


# ========================================================================================== classify_run (1b)






def test_classify_run_excludes_the_never_ablated_card_from_both_sets():
    out = srr.classify_run(RUN, None, manifest=MANIFEST, prove=PROVE,
                           self_report="It's rainy where you live, so soup felt right.",
                           support_matcher=_fake_matcher)
    ids_lb = {c.get("id") or c.get("name") for c in out["load_bearing"]}
    ids_pass = {c.get("id") or c.get("name") for c in out["passenger"]}
    assert "card_c" not in ids_lb and "card_c" not in ids_pass
    # clause_split breaks this into two clauses on ", so "; the fake matcher recognizes neither as a
    # keyword match, so both are unattributed regardless -- the point of this test is card_c's exclusion.
    assert out["counts"]["unattributed_claim"] == 2


def test_classify_run_degrades_honestly_when_nli_unavailable(monkeypatch):
    monkeypatch.setattr(srr, "_default_matcher", lambda: None)
    out = srr.classify_run(RUN, None, manifest=MANIFEST, prove=PROVE, self_report="anything")
    assert out["method"] == "nli-unavailable"
    assert out["counts"] == {k: 0 for k in srr.TAXONOMY}
    assert "nli-unavailable" in out["note"] or "unavailable" in out["note"].lower()


def test_classify_run_never_raises_on_garbage_input():
    out = srr.classify_run(None, None, manifest=None, prove=None, self_report=None,
                           support_matcher=_fake_matcher)
    assert out["run_id"] is None
    assert out["counts"] == {k: 0 for k in srr.TAXONOMY}
    assert out["claims"] == []

    out2 = srr.classify_run("not a dict", object(), support_matcher=_fake_matcher)
    assert out2["counts"] == {k: 0 for k in srr.TAXONOMY}


def test_classify_run_never_uses_introspection_vocabulary_in_its_own_notes():
    """The banned-word guard the brief calls out explicitly: this module's own strings must never claim
    the model 'knows itself' or 'is aware' -- the framing is strictly self-report reliability."""
    out = srr.classify_run(RUN, None, manifest=MANIFEST, prove=PROVE, self_report="x",
                           support_matcher=_fake_matcher)
    banned = ("introspect", "self-aware", "is aware", "knows itself")
    note_l = out["note"].lower()
    assert not any(b in note_l for b in banned)


# ================================================================ real-NLI wiring receipt (gated, -m model)
# Proves the brief's central reuse claim for real, not just by shape inspection: causal_explanation()'s
# output drops into semantic_matcher.nli_support_matcher completely unchanged, and classify_run's own
# taxonomy wiring holds with the REAL cross-encoder judge in the loop, not just the fixture's fake one.

@pytest.mark.model
def test_real_nli_matcher_accepts_causal_explanation_shape_unchanged():
    """MEASURED, not assumed: a claim phrased to match the card's own grammatical subject ("you are
    vegetarian", mirroring "The user is vegetarian") entails cleanly (~0.98); the SAME claim phrased purely
    from the assistant's side ("I kept the recipe vegetarian", no "you") scores ~0.001 against the identical
    premise -- a real, reportable ceiling of this matcher for memory-card claims (subject mismatch),
    not papered over here."""
    import clozn.receipts.semantic_matcher as sm
    if not sm.available():
        pytest.skip("cross-encoder NLI unavailable (sentence-transformers / checkpoint missing)")
    ce = srr.causal_explanation(RUN, None, manifest=MANIFEST, prove=PROVE)
    r = sm.nli_support_matcher("You are vegetarian, so I avoided meat.", ce)   # causal_explanation() IS an explanation
    assert r["method"] == "nli-deberta-v3"
    assert r["supported"] is True
    assert r["matched_id"] == "card_a"                        # the one load-bearing card -- card_b is gone


@pytest.mark.model
def test_real_nli_matcher_end_to_end_seeded_divergence():
    import clozn.receipts.semantic_matcher as sm
    if not sm.available():
        pytest.skip("cross-encoder NLI unavailable")
    self_report = ("You are vegetarian, so I avoided meat. You love the color teal, so I mentioned a "
                  "teal-colored garnish.")
    out = srr.classify_run(RUN, None, manifest=MANIFEST, prove=PROVE, self_report=self_report,
                           support_matcher=sm.nli_support_matcher)
    assert out["method"] == "nli-deberta-v3"
    classes = {c["classification"] for c in out["claims"]}
    assert "faithful_credit" in classes                       # the vegetarian clause credits the real driver
    assert "confabulated_credit" in classes                   # the teal clause credits the inert passenger
    print(f"\n[x1-real-nli] claims={out['claims']} counts={out['counts']}", flush=True)
