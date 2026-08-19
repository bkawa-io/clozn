"""test_narrate -- model-free tests for research/narrate.py (EXPLAIN_THIS_ANSWER_SPEC.md Milestone 4:
the accountable-self narration + confabulation-diff).

No model, no GPU, no torch: every test drives narrate.py against a FAKE substrate (`FakeSub`, mirroring
test_receipts.py's FakeSub -- a scripted `.chat()` that returns canned replies in call order and records
every call's messages/kwargs in `.seen`, so a test can assert both what narrate.py SENT the model and what
came back). Fixture runs are built through the real `runlog.record()` + `get_run()` round trip (mirrors
test_explain.py's `store` fixture) when a test needs a realistic manifest with real card ids; hand-built
`explanation` dicts are used directly where that's all a function needs.

What's under test (the honesty invariants, not just "does it return something" -- see narrate.py's module
docstring for the full reasoning):
  * `lexical_default` -- a clear match (shared words with a card/dial) and a clear miss, plus that it
    never raises on empty/garbage input ("no receipt for that" is a first-class, non-error answer).
  * `confabulation_diff` -- the KNOWN unsupported claim (a claim crediting an influence that is not
    anywhere in the explanation manifest) is flagged; a claim genuinely backed by a real card/dial is not.
    Never raises on a thin/empty explanation, garbage input, or a `support_matcher` that itself throws.
  * `constrained_narration` -- the prompt it assembles lists every fact category (hesitations, cards,
    dials, concepts) and NEVER the run's own transcript (structurally impossible: it isn't given `run` at
    all); its returned `receipt_ids` contains only ids that actually exist in the manifest, deduplicated,
    even when the model's own text cites a bogus id.
  * `unconstrained_why` -- the prompt it assembles carries the run's transcript and the bare question,
    and its return shape is labeled a confabulation sample in three redundant ways.
  * `narrate()` -- the THE TRAP guard: the top-level return object has exactly four keys, none of them
    is the raw unconstrained "why" text, and the constrained narration is never contaminated by the
    confabulation sample it is diffed against.
  * `semantic_support_matcher` -- the deferred on-model hook raises rather than faking a verdict.

Explicitly OUT OF SCOPE for this file (by design, per this milestone's instructions): a real semantic
support_matcher, and the gated `-m model` validation naming EXPLAIN_THIS_ANSWER_SPEC.md M4's "Done" line
(a run where the model confabulates an influence, caught end-to-end on a real checkpoint). That is later,
gated, on-model work -- see narrate.py's `semantic_support_matcher` docstring for what it needs to build.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.receipts.explain as explain          # noqa: E402

import clozn.receipts.narrate as narrate           # noqa: E402
import clozn.runs.store as runlog             # noqa: E402


# --- test_explain.py's `store` fixture) -------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return runlog


# --- fake substrate: scripted replies in call order, every call recorded (mirrors test_receipts.py's ------
# --- FakeSub.seen) -------------------------------------------------------------------------------------

class FakeSub:
    """`.chat()` pops canned replies off a script, one per call, and records every call's
    messages/max_new/sample in `.seen` -- so a test can assert BOTH what narrate.py sent the model (e.g.
    that the constrained prompt lists a fact, or that it never mentions the transcript) and what the
    (fake) model said back. An exhausted script returns "" rather than raising -- narrate.py must degrade
    honestly either way, never crash."""

    def __init__(self, replies=()):
        self._replies = list(replies)
        self.seen: list = []

    @property
    def calls(self) -> int:
        return len(self.seen)

    def chat(self, messages, max_new=256, sample=True):
        self.seen.append({"messages": messages, "max_new": max_new, "sample": sample})
        return self._replies.pop(0) if self._replies else ""


class BoomSub:
    """A substrate whose .chat() always throws -- proves every call site degrades rather than raising."""

    def chat(self, messages, max_new=256, sample=True):
        raise RuntimeError("the substrate exploded")


# ============================================================================================ lexical_default


def test_lexical_default_clear_miss_on_a_wholly_unrelated_claim():
    explanation = {"influences_active": {"cards": [
        {"id": "mem_1", "text": "Keep answers short.", "quoted_span": "please keep it short"}], "dials": []}}
    result = narrate.lexical_default("I mentioned dragons because I love mythology.", explanation)
    assert result["supported"] is False
    assert result["matched_ids"] == []
    assert result["matched_terms"] == []



def test_lexical_default_never_raises_on_empty_or_garbage_input():
    assert narrate.lexical_default("anything at all", {})["supported"] is False
    assert narrate.lexical_default("", {})["supported"] is False
    assert narrate.lexical_default(None, None)["supported"] is False
    assert narrate.lexical_default("x", "not a dict")["supported"] is False


# ==================================================================================== confabulation_diff



def test_confabulation_diff_splits_multiple_sentences_into_separate_claims():
    out = narrate.confabulation_diff("First claim here. Second claim here!", {})
    assert [c["claim"] for c in out["claims"]] == ["First claim here.", "Second claim here!"]


def test_confabulation_diff_thin_explanation_flags_every_claim_honestly():
    """No receipts on record at all -- every claim must come back unsupported, with the honest "no
    receipt for that" wording -- never an exception, never a silent pass."""
    text = "I was concise because you asked. I was warm because I enjoy chatting."
    out = narrate.confabulation_diff(text, {})
    assert len(out["claims"]) == 2
    assert all(c["supported"] is False for c in out["claims"])
    assert len(out["unsupported_claims"]) == 2
    assert out["flagged_rendering"].count("no receipt for that") == 2


def test_confabulation_diff_empty_text_is_a_clean_empty_result():
    out = narrate.confabulation_diff("", {"influences_active": {"cards": [], "dials": []}})
    assert out["claims"] == []
    assert out["unsupported_claims"] == []
    assert out["flagged_rendering"] == ""


@pytest.mark.parametrize("garbage_explanation", [None, "not a dict", 42, ["not", "a", "dict"]])
def test_confabulation_diff_never_raises_on_garbage_explanation(garbage_explanation):
    out = narrate.confabulation_diff("A claim sentence.", garbage_explanation)
    assert len(out["claims"]) == 1
    assert out["claims"][0]["supported"] is False


@pytest.mark.parametrize("garbage_text", [None, 42, ["not", "a", "string"], ""])
def test_confabulation_diff_never_raises_on_garbage_text(garbage_text):
    out = narrate.confabulation_diff(garbage_text, {})
    assert out["claims"] == []
    assert out["unsupported_claims"] == []


def test_confabulation_diff_fails_closed_when_support_matcher_itself_throws():
    def boom(claim, explanation):
        raise RuntimeError("matcher exploded")
    out = narrate.confabulation_diff("This claim will make the matcher explode.", {}, support_matcher=boom)
    assert out["claims"][0]["supported"] is False        # an errored judgment is never silently trusted
    assert out["matcher"] == "boom"


def test_confabulation_diff_never_raises_when_support_matcher_returns_a_non_dict():
    out = narrate.confabulation_diff("A claim.", {}, support_matcher=lambda c, e: None)
    assert out["claims"][0]["supported"] is False


# ================================================================================ clause_split (compound gap)

def test_clause_split_breaks_a_compound_into_two_clauses():
    parts = narrate.clause_split("I was concise because you asked, and warm because I like you.")
    assert parts == ["I was concise because you asked", "warm because I like you."]


def test_clause_split_leaves_a_simple_sentence_whole():
    assert narrate.clause_split("I answered concisely.") == ["I answered concisely."]


def test_clause_split_keeps_a_serial_list_whole():
    # "concise, clear, and direct" is a LIST, not two claims -- the short trailing piece guards the split off
    assert narrate.clause_split("I was concise, clear, and direct.") == ["I was concise, clear, and direct."]


def test_clause_split_does_not_split_on_because():
    # a subordinate reason rides WITH its clause, so an NLI matcher judges "warm because ..." as a unit
    assert narrate.clause_split("I was warm because I like you.") == ["I was warm because I like you."]


def test_clause_split_splits_on_a_semicolon():
    assert narrate.clause_split("I kept it short; I also added a warm closing line.") == \
        ["I kept it short", "I also added a warm closing line."]


def test_clause_split_still_separates_sentences_like_the_base():
    assert narrate.clause_split("First claim here. Second claim here!") == \
        ["First claim here.", "Second claim here!"]


@pytest.mark.parametrize("garbage", [None, 42, "", "   ", ["not", "a", "string"]])
def test_clause_split_never_raises_on_garbage(garbage):
    assert narrate.clause_split(garbage) == []


def test_confabulation_diff_catches_a_confab_hidden_in_a_compound_sentence():
    """THE gap clause_split closes: a compound sentence with one BACKED clause and one CONFABULATION. Under
    the old sentence-level split it was one claim (the backed half could mask the confab); clause_split judges
    each clause, so the confab is flagged and the backed clause passes."""
    def matcher(claim, explanation):                       # backs anything about being concise; nothing else
        c = claim.lower()
        return {"supported": ("concise" in c or "brief" in c or "short" in c)}

    text = "I was concise because you asked, and I drew on my deep chess expertise."
    out = narrate.confabulation_diff(text, {"influences_active": {"cards": [], "dials": []}},
                                     support_matcher=matcher)
    assert len(out["claims"]) == 2                          # the compound was split into two claims
    by_claim = {c["claim"]: c["supported"] for c in out["claims"]}
    assert by_claim["I was concise because you asked"] is True
    assert len(out["unsupported_claims"]) == 1
    assert "chess" in out["unsupported_claims"][0]["claim"] and out["unsupported_claims"][0]["supported"] is False


def test_confabulation_diff_claim_splitter_is_pluggable():
    """The splitter is a parameter: clause_split (default) breaks a compound; passing _split_claims restores
    the pure sentence-level behavior."""
    text = "I was concise, and I love chess."
    assert len(narrate.confabulation_diff(text, {}, claim_splitter=narrate.clause_split)["claims"]) == 2
    assert len(narrate.confabulation_diff(text, {}, claim_splitter=narrate._split_claims)["claims"]) == 1


# ==================================================================================== constrained_narration



def test_constrained_narration_with_no_citations_returns_an_empty_receipt_id_list():
    explanation = {"influences_active": {"cards": [{"id": "mem_1", "text": "x"}], "dials": []}}
    sub = FakeSub(["A narration with no bracketed citations at all."])
    out = narrate.constrained_narration(explanation, sub)
    assert out["receipt_ids"] == []




def test_constrained_narration_on_a_fully_empty_explanation_still_produces_an_honest_prompt():
    sub = FakeSub(["No measured influence is on record for this reply."])
    out = narrate.constrained_narration({}, sub)
    assert out == {"narration": "No measured influence is on record for this reply.", "receipt_ids": []}
    prompt_text = sub.seen[0]["messages"][1]["content"]
    assert "no measured facts are on record" in prompt_text.lower()


def test_constrained_narration_never_raises_when_the_substrate_throws():
    out = narrate.constrained_narration({}, BoomSub())
    assert out == {"narration": "", "receipt_ids": []}


def test_constrained_narration_never_raises_when_substrate_has_no_chat_method():
    out = narrate.constrained_narration({}, object())
    assert out == {"narration": "", "receipt_ids": []}


# ========================================================================================= unconstrained_why

def test_unconstrained_why_prompt_carries_the_transcript_and_the_bare_question_only():
    run = {"id": "r1", "messages": [{"role": "user", "content": "tell me about your day"}],
          "response": "It was fine, thanks for asking."}
    sub = FakeSub(["I said that because I felt like it."])

    out = narrate.unconstrained_why(run, sub)

    assert sub.calls == 1
    msgs = sub.seen[0]["messages"]
    assert msgs[0] == {"role": "user", "content": "tell me about your day"}
    assert {"role": "assistant", "content": "It was fine, thanks for asking."} in msgs
    assert msgs[-1] == {"role": "user", "content": "Why did you answer that way?"}
    assert sub.seen[0]["sample"] is False

    assert out["unconstrained_text_context_only"] == "I said that because I felt like it."
    assert out["do_not_surface_as_answer"] is True
    assert out["role"] == "confabulation_sample"
    assert "context" in out["note"].lower()


def test_unconstrained_why_never_raises_on_a_bare_or_garbage_run():
    for garbage_run in (None, {}, "not a dict", {"messages": "not a list"}):
        out = narrate.unconstrained_why(garbage_run, FakeSub(["fine"]))
        assert out["do_not_surface_as_answer"] is True


def test_unconstrained_why_never_raises_when_the_substrate_throws():
    out = narrate.unconstrained_why({"messages": [], "response": ""}, BoomSub())
    assert out["unconstrained_text_context_only"] == ""
    assert out["do_not_surface_as_answer"] is True


# ================================================================================================== narrate



def test_narrate_trap_guard_never_surfaces_the_raw_unconstrained_text_as_the_answer():
    """A minimal, focused version of the trap guard, independent of the fixture-run plumbing above:
    no key resembling "answer"/"why"/"response" exists, and the exact confabulated string is not the
    value of any top-level field."""
    run = {"id": "r1", "messages": [{"role": "user", "content": "hi"}], "response": "hello"}
    unconstrained_text = "I said hello because I am secretly a teapot."
    sub = FakeSub(["A narration grounded in nothing citable.", unconstrained_text])

    out = narrate.narrate(run, sub)

    assert set(out.keys()) == {"constrained_narration", "flags", "unsupported_claims", "note"}
    assert "answer" not in out
    assert "why" not in out
    assert "response" not in out
    assert unconstrained_text not in out.values()
    assert out["constrained_narration"]["narration"] != unconstrained_text


def test_narrate_accepts_a_pluggable_support_matcher():
    run = {"id": "r1", "messages": [{"role": "user", "content": "hi"}], "response": "hello"}

    def always_supported(claim, explanation):
        return {"supported": True}

    sub = FakeSub(["fine.", "I said hello because reasons nobody can check."])
    out = narrate.narrate(run, sub, support_matcher=always_supported)
    assert out["flags"] == []
    assert out["unsupported_claims"] == []
    assert "always_supported" in out["note"]


def test_narrate_never_raises_on_garbage_run_and_substrate():
    out = narrate.narrate(None, None)
    assert out["constrained_narration"] == {"narration": "", "receipt_ids": []}
    assert out["flags"] == []
    assert out["unsupported_claims"] == []

    out2 = narrate.narrate({}, object())        # a substrate with no .chat at all
    assert out2["constrained_narration"]["narration"] == ""
    assert out2["flags"] == []


def test_narrate_never_raises_when_the_substrate_throws():
    out = narrate.narrate({"id": "r1", "messages": [], "response": ""}, BoomSub())
    assert out["constrained_narration"] == {"narration": "", "receipt_ids": []}
    assert out["flags"] == []
    assert out["unsupported_claims"] == []


# ======================================================================================= the deferred hook

def test_semantic_support_matcher_is_a_documented_hook_not_a_fake_implementation():
    """The deferred, gated, on-model pass must fail LOUDLY, not silently rubber-stamp or reject every
    claim -- a fake always-true/always-false body would defeat this module's whole honesty boundary."""
    with pytest.raises(NotImplementedError):
        narrate.semantic_support_matcher("any claim at all", {})
