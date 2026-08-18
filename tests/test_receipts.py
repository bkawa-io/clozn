"""test_receipts -- model-free tests for clozn/receipts/core.py (EXPLAIN_THIS_ANSWER_SPEC.md Milestone 2).

No model, no GPU, no torch: drives receipts.receipt() / receipts.prove_all() against a FAKE substrate
whose .chat() is a DETERMINISTIC function of the messages it actually saw, so a receipt's
baseline-vs-ablated delta is driven ONLY by whatever replay.py actually changed, never by randomness.

Memory-card ablation and tone-dial ablation were both cut from the product (memory cards on 2026-07-27,
dials in the personalization cut that followed) -- `section` is the only influence kind
receipts.core._ablation_changes/deltas.py still recognize; a legacy `dial`/`memory_off`/`behavior_off`/
`card_id` influence spec is simply not something this codebase can ablate any more and degrades to None,
exactly like any other unrecognized spec. See test_section_influence.py for the full section-ablation
surface (regen AND forced mode); this file covers the metric math, the mode-dispatch plumbing, and the
degrade-on-bad-input paths.

What's under test:
  * the BOTH-ARMS-GREEDY seam: receipt() calls sub.chat() exactly twice, both greedy (sample=False), both
    over the run's own stored messages -- never touching the run's stored sampled `response`.
  * receipt_metrics() is the canonical metric assembly used by Replay, including the rounding
    ties (Math.round rounds a trailing .5 UP; Python's builtin round() would bankers'-round it down).
  * the sampled-reply-is-not-baseline `note` is present on every receipt.
  * prove_all()'s leave-one-out over the M1 manifest's `sections`.
  * mode dispatch (regen/forced/both) is byte-identical to calling the underlying helpers directly.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.settings as clozn_settings          # noqa: E402

from clozn import receipts          # noqa: E402
import clozn.runs.store as runlog             # noqa: E402


# --- a minimal fake substrate -----------------------------------------------------------------------------
# Card/dial ablation was cut from the product, so nothing here drives chat() by live substrate state any
# more -- this just proves prove_all()/receipt() degrade cleanly (bad input, no fired influences) without
# ever needing a real generation.

class FakeSub:
    def __init__(self):
        self.seen: list = []      # one entry per chat() call, in call order

    @property
    def calls(self):
        return len(self.seen)

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.seen.append({"messages": messages, "sample": sample})
        return "A plain reply."


RUN = {"id": "run_parent0", "model": "clozn-qwen", "substrate": "QwenSubstrate",
       "messages": [{"role": "user", "content": "tell me about your day"}],
       "response": "THE STORED SAMPLED REPLY -- must never be used as anyone's baseline"}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    return tmp_path


# ============================================================================================ metric math

def test_receipt_metrics_identical_replies_show_zero_change():
    m = receipts.receipt_metrics("one two three", "one two three")
    assert m == {"words": [3, 3], "wps": [3.0, 3.0], "changed": 0}


def test_receipt_metrics_fully_disjoint_word_types_are_100pct_changed():
    m = receipts.receipt_metrics("a a a", "b b b")
    assert m["words"] == [3, 3]
    assert m["changed"] == 100


def test_receipt_metrics_word_count_is_total_tokens_not_unique_types():
    # "changed" is a word-TYPE (unique) Jaccard distance, but "words" counts every token -- 5 tokens, 2
    # unique types, all shared -> changed stays 0 even though the raw token count differs from repl's.
    m = receipts.receipt_metrics("a a a b b", "a b")
    assert m["words"] == [5, 2]
    assert m["changed"] == 0


def test_receipt_metrics_wps_rounds_ties_up_like_js_not_bankers_rounding():
    # 1 word, 4 "sentences": three punctuation-only fillers ("-") survive the trim() filter (non-empty)
    # but contribute no [a-z0-9'] word -- so ow.length=1, sentCount=4 -> 1/4*10 = 2.5 EXACTLY. JS's
    # Math.round rounds a trailing .5 UP (-> 3 -> 0.3); Python's builtin round(2.5) would bankers'-round
    # to 2 (-> 0.2). This is precisely the discrepancy _js_round exists to prevent.
    text = "cat! -! -! -!"
    m = receipts.receipt_metrics(text, text)
    assert m["wps"] == [0.3, 0.3]


def test_receipt_metrics_changed_pct_rounds_ties_up_too():
    # oset={a,b,c,d,e} (5), rset={c,d,e,f,g,h} (6): intersection=3, union=8 -> (1 - 3/8) * 100 = 62.5
    # EXACTLY. JS rounds up to 63; Python's round(62.5) would bankers'-round to 62 (62 is even).
    m = receipts.receipt_metrics("a b c d e", "c d e f g h")
    assert m["words"] == [5, 6]
    assert m["changed"] == 63


def test_receipt_metrics_empty_replies_never_divide_by_zero():
    assert receipts.receipt_metrics("", "") == {"words": [0, 0], "wps": [0.0, 0.0], "changed": 0}


# ------------------------------------------------------------------------------- never raises / bad input

def test_receipt_returns_none_on_bad_influence_spec(iso):
    sub = FakeSub()
    assert receipts.receipt(RUN, {}, sub) is None
    assert receipts.receipt(RUN, {"nonsense": True}, sub) is None
    assert receipts.receipt(RUN, None, sub) is None


def test_receipt_returns_none_on_a_legacy_influence_spec(iso):
    """card/dial ablation was cut from the product -- a stray `card_id`/`memory_off`/`behavior_off` spec
    (from before that cut) degrades to None exactly like any other unrecognized influence, never raises."""
    sub = FakeSub()
    assert receipts.receipt(RUN, {"card_id": "mem_card_a"}, sub) is None
    assert receipts.receipt(RUN, {"memory_off": True}, sub) is None
    assert receipts.receipt(RUN, {"behavior_off": True}, sub) is None
    assert receipts.receipt(RUN, {"dial": "warm"}, sub) is None


def test_receipt_returns_none_on_empty_run(iso):
    assert receipts.receipt(None, {"section": "rag_context"}, FakeSub()) is None
    assert receipts.receipt({}, {"section": "rag_context"}, FakeSub()) is None


def test_receipt_returns_none_when_substrate_is_none(iso):
    assert receipts.receipt(RUN, {"section": "rag_context"}, None) is None


# ==================================================================================== prove_all: leave-one-out

def test_prove_all_no_fired_influences_is_a_clean_empty_result(iso):
    run = {"id": "run_bare", "messages": [{"role": "user", "content": "hi"}], "response": "sampled"}
    out = receipts.prove_all(run, FakeSub())
    assert out == {"run_id": "run_bare", "receipts": [], "skipped": [], "redundant_pairs": [],
                   "approximation_note": out["approximation_note"], "perf_note": out["perf_note"]}


def test_prove_all_never_raises_on_garbage_input(iso):
    assert receipts.prove_all(None, FakeSub())["receipts"] == []
    assert receipts.prove_all({}, FakeSub())["receipts"] == []
    assert receipts.prove_all("not a dict", FakeSub())["run_id"] is None


def test_prove_all_degrades_when_substrate_is_none(iso):
    run = {"id": "run_x", "messages": [{"role": "user", "content": "hi"}], "response": "sampled",
          "sections": [{"id": "sec_a", "name": "rag_context", "source": "auto",
                       "parts": [{"message_index": 0, "start": 0, "end": 2}],
                       "char_count": 2, "preview": "hi"}]}
    out = receipts.prove_all(run, None)
    assert out["receipts"] == []
    assert any("baseline" in s["reason"] for s in out["skipped"])


def test_prove_all_states_the_pairwise_approximation_and_the_perf_follow_up(iso):
    """approximation_note/perf_note are static plumbing text, present on every prove_all() result
    regardless of what fired -- no real run is needed to exercise them."""
    out = receipts.prove_all({"id": "run_bare", "messages": []}, FakeSub())
    assert "power set" in out["approximation_note"] or "power-set" in out["approximation_note"]
    assert "pair" in out["approximation_note"].lower()
    assert "batch" in out["perf_note"].lower()


# ==================================================================================== sections (prompt-section receipts)
#
# The section generalization: a run's `sections` manifest (id/name/source/parts/char_count/preview -- the
# fixed schema clozn.runs.sections/store produce) yields its own leave-one-out ablation arm, exactly like a
# card or a dial used to, EXCEPT for sections whose source is "memory_card" -- those duplicate a card
# influence that no longer exists (see deltas._section_influences' docstring for the full reasoning).

SECTION_PREFIX = "Question. "
SECTION_TEXT = "RAG-MARKER extra info here."

RAG_SECTION = {"id": "sec_rag", "name": "rag_context", "source": "auto",
              "parts": [{"message_index": 0, "start": len(SECTION_PREFIX),
                        "end": len(SECTION_PREFIX) + len(SECTION_TEXT)}],
              "char_count": len(SECTION_TEXT), "preview": SECTION_TEXT}

SECTION_RUN = {"id": "run_section0", "model": "clozn-qwen", "substrate": "QwenSubstrate",
              "messages": [{"role": "user", "content": SECTION_PREFIX + SECTION_TEXT}],
              "response": "SAMPLED reply, never a baseline",
              "sections": [RAG_SECTION]}


class SectionFakeSub:
    """chat() is a pure function of whether the RAG marker text is present anywhere in `messages` -- the
    deterministic signal is read off the messages actually passed in, since a section ablation changes
    prompt CONTENT rather than substrate state."""

    def __init__(self, marker="RAG-MARKER"):
        self.marker = marker
        self.seen: list = []

    @property
    def calls(self):
        return len(self.seen)

    def chat(self, messages, max_new=256, sample=True):
        self.seen.append({"messages": messages, "sample": sample})
        present = any(self.marker in str(m.get("content", "")) for m in (messages or []))
        return "Answer WITH context." if present else "Answer with no context at all."


def test_section_receipt_ablation_shows_effect_and_carries_kind_metadata(iso):
    sub = SectionFakeSub()
    rec = receipts.receipt(SECTION_RUN, {"section": "rag_context", "source": "auto"}, sub)
    assert rec is not None
    assert rec["kind"] == "section"
    assert rec["section_name"] == "rag_context"
    assert rec["section_source"] == "auto"
    assert rec["causal_verified"] is True
    assert rec["has_effect"] is True
    assert rec["baseline_reply"] == "Answer WITH context."
    assert rec["ablated_reply"] == "Answer with no context at all."
    assert rec["changes_applied"] == {"exclude_sections": ["rag_context"]}
    assert "prompt content" in rec["cost_note"] or "re-prefill" in rec["cost_note"]


def test_receipt_returns_none_when_the_named_section_cannot_be_resolved_but_arm_is_well_formed(iso):
    """An unknown section name still builds a receipt (replay's graceful skip kicks in, not receipt()'s own
    validation) -- but it must be honestly flagged unverified, not silently reported as no-effect-proven."""
    sub = SectionFakeSub()
    rec = receipts.receipt(SECTION_RUN, {"section": "not_a_real_section", "source": "auto"}, sub)
    assert rec is not None
    assert rec["causal_verified"] is False
    assert "no section named" in rec["ablation_note"]
    assert rec["has_effect"] is False                # nothing actually changed -> both arms identical


def test_prove_all_emits_a_section_arm_for_an_auto_section(iso):
    """No explicit manifest: prove_all's default explain.explain(run) reads the run's OWN `sections` field
    end to end -- the real wiring, not a hand-fed influence spec."""
    sub = SectionFakeSub()
    out = receipts.prove_all(SECTION_RUN, sub)
    assert out["skipped"] == []
    assert len(out["receipts"]) == 1
    rec = out["receipts"][0]
    assert rec["influence"] == {"section": "rag_context", "source": "auto"}
    assert rec["kind"] == "section"
    assert rec["has_effect"] is True
    assert rec["causal_verified"] is True


def test_section_influences_helper_skips_memory_card_sourced_sections(iso):
    """Direct unit test of the dedup rule: a manifest with one api-sourced and one memory_card-sourced
    section yields an ablation arm for the api one only."""
    manifest = {"influences_active": {"sections": {"available": True, "sections": [
        {"id": "sec_a", "name": "system_prompt", "source": "api"},
        {"id": "sec_b", "name": "user_pref_card", "source": "memory_card"},
    ]}}}
    influences, skipped = receipts._section_influences(manifest)
    assert influences == [{"section": "system_prompt", "source": "api"}]
    assert skipped == []


def test_section_influences_helper_accepts_the_auto_source_too(iso):
    manifest = {"influences_active": {"sections": {"available": True, "sections": [
        {"id": "sec_a", "name": "rag_context", "source": "auto"},
    ]}}}
    influences, _ = receipts._section_influences(manifest)
    assert influences == [{"section": "rag_context", "source": "auto"}]


def test_section_influences_helper_skips_a_nameless_section_honestly(iso):
    manifest = {"influences_active": {"sections": {"available": True, "sections": [
        {"id": "sec_a", "source": "auto"},        # no "name"
    ]}}}
    influences, skipped = receipts._section_influences(manifest)
    assert influences == []
    assert len(skipped) == 1
    assert "no name" in skipped[0]["reason"]


def test_section_influences_helper_degrades_when_no_sections_manifest_at_all(iso):
    assert receipts._section_influences({"influences_active": {}}) == ([], [])
    assert receipts._section_influences(None) == ([], [])


# NOTE: a test named test_prove_all_does_not_double_ablate_a_memory_card_backed_section lived here on the
# feat/prompt-section-influence branch. Dropped during the 342-commit-stale merge (2026-07-29): it asserted
# prove_all resolves a "memory_card"-sourced section back to its OWN card's richer `card_id` ablation path --
# but memory cards were cut from the product on 2026-07-27 (_fired_influences no longer walks
# `influences_active["cards"]` at all), and the test itself referenced an undefined `memory_mode` name (a
# leftover from the deleted `clozn.memory.mode` module), so it could never have passed against current HEAD.
# The surviving dedup test (`test_section_influences_helper_skips_memory_card_sourced_sections`, above) still
# covers the live half of this: a stray "memory_card"-sourced section in an old run's manifest is skipped,
# not ablated.


# --------------------------------------------------------------------------------- helper-level unit tests
# (used to back the forced-mode dial null-floor control, now retired along with dial ablation -- see
# test_section_influence.py for forced-mode section-ablation coverage, which has no null floor of its own.)

def test_matched_length_filler_is_deterministic_and_matches_length():
    a = receipts._matched_length_filler(37)
    b = receipts._matched_length_filler(37)
    assert a == b
    assert len(a) == 37


def test_random_vector_of_norm_is_deterministic_per_seed_and_matches_norm():
    v1 = receipts._random_vector_of_norm(5, 2.0, "seed-a")
    v2 = receipts._random_vector_of_norm(5, 2.0, "seed-a")
    assert v1 == v2
    assert abs(receipts._vector_norm(v1) - 2.0) < 1e-9
    v3 = receipts._random_vector_of_norm(5, 2.0, "seed-b")
    assert v3 != v1                                          # a different seed -> a different direction


# ============================================================================= mode dispatch (receipt/prove_all)

def test_receipt_default_mode_is_byte_identical_to_the_pre_s3_regen_receipt(iso):
    """The load-bearing regression: omitting `mode` (every pre-S3 caller) must produce EXACTLY
    receipts._receipt_regen's dict -- no added "mode"/"forced" keys, nothing changed."""
    influence = {"section": "rag_context", "source": "auto"}
    default = receipts.receipt(SECTION_RUN, influence, SectionFakeSub())
    explicit_regen = receipts.receipt(SECTION_RUN, influence, SectionFakeSub())
    direct = receipts._receipt_regen(SECTION_RUN, influence, SectionFakeSub())
    assert default == direct
    assert explicit_regen == direct
    assert "mode" not in default
    assert "forced" not in default
    assert "silent_influence" not in default


def test_prove_all_default_mode_is_byte_identical_to_the_pre_s3_function(iso):
    default = receipts.prove_all(SECTION_RUN, SectionFakeSub())
    direct = receipts._prove_all_regen(SECTION_RUN, SectionFakeSub())
    assert default == direct
    assert "mode" not in default
    assert "forced_receipts" not in default
