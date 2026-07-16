"""test_receipts -- model-free tests for research/receipts.py (EXPLAIN_THIS_ANSWER_SPEC.md Milestone 2).

No model, no GPU, no torch: drives receipts.receipt() / receipts.prove_all() against a FAKE substrate
(mirrors test_replay.py's FakeSub/FakeMem/FakeSteer) whose .chat() is a DETERMINISTIC function of exactly
which influences are live at call time -- which card ids are excluded, whether memory is off, and the
"concise"/"warm" dial values -- so a receipt's baseline-vs-ablated delta is driven ONLY by whatever
replay.py actually changed, never by randomness.

What's under test:
  * the BOTH-ARMS-GREEDY seam: receipt() calls sub.chat() exactly twice, both greedy (sample=False), both
    over the run's own stored messages -- never touching the run's stored sampled `response`.
  * receipt_metrics() mirrors run.js's receiptMetrics() EXACTLY, including the two JS-vs-Python rounding
    ties (Math.round rounds a trailing .5 UP; Python's builtin round() would bankers'-round it down).
  * causal_verified is True on a real ablation, and correctly FALSE (with an `ablation_note`) when the
    ablation could not actually apply (a per-card ablation attempted in "internalized" memory mode) --
    relaying replay.py's own honesty note rather than silently claiming "no effect" proved something.
  * the sampled-reply-is-not-baseline `note` is present on every receipt.
  * prove_all()'s leave-one-out over the M1 manifest, and the REDUNDANCY GUARD: two cards that are each
    individually load-bearing-free but jointly necessary are reported as a redundant pair, not "neither
    mattered".
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.memory.cards as memory_cards      # noqa: E402
import clozn.memory.mode as memory_mode       # noqa: E402
from clozn import receipts          # noqa: E402
import clozn.runs.store as runlog             # noqa: E402


# --- fakes (mirror test_replay.py's FakeSteer/FakeMem/FakeSub, extended with a deterministic, --------------
# --- influence-keyed chat() so baseline vs ablated replies differ EXACTLY when the ablated state differs --

class FakeSteer:
    def __init__(self, strength=None):
        self.strength = dict(strength or {})

    def set(self, name, value):
        self.strength[str(name)] = float(value)

    def clear(self):
        self.strength = {}

    def active(self):
        return {k: v for k, v in self.strength.items() if v}


class FakeMem:
    def __init__(self, strength=1.0, rules=None, prefix="PFX"):
        self.memory_strength = float(strength)
        self.rules = list(rules or [])
        self.prefix = prefix


class FakeSub:
    """chat() is a pure function of (memory_strength, excluded card ids, concise/warm dial values) -- no
    randomness -- so exact reply-string equality is a trustworthy "no effect" signal, exactly like a real
    greedy decode."""

    def __init__(self, mem=None, steer=None, concise_card_ids=()):
        self.memory = mem if mem is not None else FakeMem()
        self.steer = steer if steer is not None else FakeSteer()
        self.concise_card_ids = {str(i) for i in concise_card_ids}
        self.seen: list = []      # one entry per chat() call, in call order

    @property
    def calls(self):
        return len(self.seen)

    def chat(self, messages, max_new=256, sample=True):
        excluded = {str(i) for i in (getattr(self.memory, "_exclude_card_ids", None) or [])}
        self.seen.append({"messages": messages, "sample": sample,
                          "memory_strength": self.memory.memory_strength,
                          "exclude": sorted(excluded), "dials": dict(self.steer.strength)})
        if self.memory.memory_strength <= 0:
            return "Generic reply with memory off, noticeably longer and less tailored than usual."
        concise_active = self.concise_card_ids - excluded
        concise_dial = float(self.steer.strength.get("concise", 0.0) or 0.0)
        if concise_active or concise_dial > 0:
            base = "Short answer."
        else:
            base = ("A much longer rambling reply with plenty of extra words, since nothing left standing "
                    "kept this concise once every source of brevity was ablated away.")
        if float(self.steer.strength.get("warm", 0.0) or 0.0) > 0:
            base += " Hope that helps and warms your day a little!"
        return base


RUN = {"id": "run_parent0", "model": "clozn-qwen", "substrate": "QwenSubstrate",
       "messages": [{"role": "user", "content": "tell me about your day"}],
       "response": "THE STORED SAMPLED REPLY -- must never be used as anyone's baseline"}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every flat-file store replay.py / memory_mode.py / memory_cards.py touch (mirrors
    test_replay.py's `store` + test_memory_mode.py's `iso`). Mode starts UNSET (fresh-install default is
    "prompt"); tests that care pin it explicitly."""
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
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


# ================================================================================ both-arms-greedy receipt

def test_receipt_is_exactly_two_greedy_calls_over_the_runs_own_messages(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    rec = receipts.receipt(RUN, {"dial": "warm"}, sub)
    assert rec is not None
    assert sub.calls == 2                                        # baseline + ablated, nothing more
    assert all(call["sample"] is False for call in sub.seen)      # BOTH arms greedy
    assert all(call["messages"] == RUN["messages"] for call in sub.seen)   # the run's own stored messages


def test_receipt_dial_ablation_shows_effect_and_is_causally_verified(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    rec = receipts.receipt(RUN, {"dial": "warm"}, sub)
    assert rec["causal_verified"] is True
    assert rec["has_effect"] is True                              # the warm suffix disappears when ablated
    assert rec["baseline_reply"].endswith("a little!")
    assert not rec["ablated_reply"].endswith("a little!")
    assert rec["changes_applied"] == {"behavior_overrides": {"warm": 0.0}}
    assert rec["delta"] == receipts.receipt_metrics(rec["baseline_reply"], rec["ablated_reply"])


def test_receipt_never_uses_the_stored_sampled_reply_as_either_arm(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    rec = receipts.receipt(RUN, {"dial": "warm"}, sub)
    assert RUN["response"] not in (rec["baseline_reply"], rec["ablated_reply"])
    assert "sampled" in rec["note"].lower() and "baseline" in rec["note"].lower()


def test_receipt_memory_off_ablation(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    rec = receipts.receipt(RUN, {"memory_off": True}, sub)
    assert rec["causal_verified"] is True
    assert rec["has_effect"] is True
    assert rec["baseline_reply"] == "A much longer rambling reply with plenty of extra words, since nothing " \
                                    "left standing kept this concise once every source of brevity was ablated away."
    assert rec["ablated_reply"] == "Generic reply with memory off, noticeably longer and less tailored than usual."
    assert "front-of-context" in rec["cost_note"] or "re-prefill" in rec["cost_note"]


def test_receipt_behavior_off_ablation(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    rec = receipts.receipt(RUN, {"behavior_off": True}, sub)
    assert rec["causal_verified"] is True
    assert rec["has_effect"] is True
    assert "decode" in rec["cost_note"] or "cheap" in rec["cost_note"]


def test_receipt_card_ablation_in_prompt_mode_is_real_and_can_show_no_effect(iso):
    """A single card ablated alone, while a SECOND concise-inducing card is still active -> no effect (the
    other card alone is enough) -- the exact setup the redundancy guard exists to catch, exercised here as
    a plain single receipt."""
    memory_mode.set_mode("prompt")
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}), concise_card_ids=["card_a", "card_b"])
    rec = receipts.receipt(RUN, {"card_id": "card_a"}, sub)
    assert rec["causal_verified"] is True           # the ablation DID apply (prompt mode, real per-card)
    assert rec["has_effect"] is False                # but card_b alone still kept it concise
    assert rec["baseline_reply"] == rec["ablated_reply"] == "Short answer."


def test_receipt_card_ablation_alone_removing_the_only_concise_source_has_effect(iso):
    memory_mode.set_mode("prompt")
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}), concise_card_ids=["card_a"])
    rec = receipts.receipt(RUN, {"card_id": "card_a"}, sub)
    assert rec["causal_verified"] is True
    assert rec["has_effect"] is True
    assert rec["baseline_reply"] == "Short answer."
    assert rec["ablated_reply"].startswith("A much longer rambling reply")


# ------------------------------------------------------------- the honesty guard: an ablation that can't apply

def test_receipt_flags_unapplied_card_ablation_in_internalized_mode_as_not_verified(iso, monkeypatch):
    """replay.py can't remove ONE card from a fused internalized prefix -- it records an honest "not
    applied" note and leaves the state untouched. A receipt built on top of that MUST NOT claim
    causal_verified: true (that would silently launder "we never tried" into "proven no effect")."""
    monkeypatch.setenv("CLOZN_RUNTIME_KIND", "lab")
    memory_mode.set_mode("internalized")
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}), concise_card_ids=["card_a"])
    rec = receipts.receipt(RUN, {"card_id": "card_a"}, sub)
    assert rec is not None
    assert rec["causal_verified"] is False
    assert "ablation_note" in rec
    assert "internalized" in rec["ablation_note"] and "fused" in rec["ablation_note"]
    # nothing was actually tried -> the two arms are identical, but for the RIGHT (disclosed) reason
    assert rec["has_effect"] is False
    assert rec["baseline_reply"] == rec["ablated_reply"]


# ------------------------------------------------------------------------------- never raises / bad input

def test_receipt_returns_none_on_bad_influence_spec(iso):
    sub = FakeSub()
    assert receipts.receipt(RUN, {}, sub) is None
    assert receipts.receipt(RUN, {"nonsense": True}, sub) is None
    assert receipts.receipt(RUN, None, sub) is None


def test_receipt_returns_none_on_empty_run(iso):
    assert receipts.receipt(None, {"memory_off": True}, FakeSub()) is None
    assert receipts.receipt({}, {"memory_off": True}, FakeSub()) is None


def test_receipt_returns_none_when_substrate_is_none(iso):
    assert receipts.receipt(RUN, {"memory_off": True}, None) is None


# ==================================================================================== prove_all: leave-one-out

CARD_A, CARD_B = "mem_card_a", "mem_card_b"

REDUNDANT_RUN = {
    "id": "run_redundant", "model": "clozn-qwen", "substrate": "QwenSubstrate",
    "messages": [{"role": "user", "content": "how was your day"}],
    "response": "SAMPLED reply, never a baseline",
    "memory": {"cards_applied": ["Be concise.", "Keep it short."], "applied_ids": [CARD_A, CARD_B],
              "gate": 0.9, "mode": "prompt", "strength": 1.0},
    "behavior": {"active_dials": {"warm": 0.4}},
}


def test_prove_all_runs_leave_one_out_over_every_fired_card_and_dial(iso):
    memory_mode.set_mode("prompt")
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.4}), concise_card_ids=[CARD_A, CARD_B])
    out = receipts.prove_all(REDUNDANT_RUN, sub)
    assert out["run_id"] == "run_redundant"
    assert len(out["receipts"]) == 3                      # card_a, card_b, warm dial
    assert out["skipped"] == []
    labels = {r["influence"].get("card_id") or r["influence"].get("dial") for r in out["receipts"]}
    assert labels == {CARD_A, CARD_B, "warm"}
    assert all(r["causal_verified"] is True for r in out["receipts"])
    assert all("sampled" in r["note"].lower() for r in out["receipts"])


def test_prove_all_redundancy_guard_catches_the_ab_pair(iso):
    """card_a alone: no effect (card_b covers it). card_b alone: no effect (card_a covers it). BOTH
    dropped together: the reply changes. Leave-one-out alone would call this "neither mattered" -- the
    redundancy guard must catch it instead."""
    memory_mode.set_mode("prompt")
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.4}), concise_card_ids=[CARD_A, CARD_B])
    out = receipts.prove_all(REDUNDANT_RUN, sub)

    by_label = {(r["influence"].get("card_id") or r["influence"].get("dial")): r for r in out["receipts"]}
    assert by_label[CARD_A]["has_effect"] is False
    assert by_label[CARD_B]["has_effect"] is False
    assert by_label["warm"]["has_effect"] is True             # NOT redundant -- a genuine standalone effect

    assert len(out["redundant_pairs"]) == 1
    pair = out["redundant_pairs"][0]
    assert set(pair["redundant"]) == {f"card:{CARD_A}", f"card:{CARD_B}"}
    assert pair["note"] == "together they drive this; individually neither is load-bearing"


def test_prove_all_reuses_one_baseline_not_a_fresh_one_per_check(iso):
    """1 shared baseline + 3 leave-one-out ablations (card_a, card_b, warm) + 1 joint pair check
    (card_a+card_b, the only pair where BOTH sides showed no individual effect) = 5 calls. A naive
    per-influence receipt() loop (each regenerating its own baseline) would cost 8."""
    memory_mode.set_mode("prompt")
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.4}), concise_card_ids=[CARD_A, CARD_B])
    receipts.prove_all(REDUNDANT_RUN, sub)
    assert sub.calls == 5
    assert all(call["sample"] is False for call in sub.seen)     # every generation greedy


def test_prove_all_states_the_pairwise_approximation_and_the_perf_follow_up(iso):
    memory_mode.set_mode("prompt")
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.4}), concise_card_ids=[CARD_A, CARD_B])
    out = receipts.prove_all(REDUNDANT_RUN, sub)
    assert "power set" in out["approximation_note"] or "power-set" in out["approximation_note"]
    assert "pair" in out["approximation_note"].lower()
    assert "batch" in out["perf_note"].lower()


def test_prove_all_skips_a_card_with_no_resolvable_id_honestly(iso):
    run = {"id": "run_noid", "messages": [{"role": "user", "content": "hi"}], "response": "sampled",
          "memory": {"cards_applied": ["some fused rule"], "mode": "internalized"},   # no applied_ids at all
          "behavior": {"active_dials": {}}}
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    out = receipts.prove_all(run, sub)
    assert out["receipts"] == []
    assert sub.calls == 0                             # never even generated a baseline -- nothing to check
    assert len(out["skipped"]) == 1
    assert "no card id" in out["skipped"][0]["reason"]


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
    memory_mode.set_mode("prompt")
    out = receipts.prove_all(REDUNDANT_RUN, None)
    assert out["receipts"] == []
    assert any("baseline" in s["reason"] for s in out["skipped"])


# ==================================================================================== sections (prompt-section receipts)
#
# The section generalization: a run's `sections` manifest (id/name/source/parts/char_count/preview -- the
# fixed schema clozn.runs.sections/store produce) yields its own leave-one-out ablation arm, exactly like a
# card or a dial, EXCEPT for sections whose source is "memory_card" -- those duplicate a card influence that
# already has its own (richer, provenance-resolved) ablation path, so ablating both would double-count one
# real cause (see deltas._section_influences' docstring for the full reasoning).

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
    """chat() is a pure function of whether the RAG marker text is present anywhere in `messages` -- unlike
    this file's other FakeSub (a function of memory/dial STATE), a section ablation changes prompt CONTENT,
    so the deterministic signal has to be read off the messages actually passed in."""

    def __init__(self, marker="RAG-MARKER"):
        self.marker = marker
        self.memory = FakeMem()
        self.steer = FakeSteer()
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


def test_prove_all_does_not_double_ablate_a_memory_card_backed_section(iso):
    """A card influence AND a section describing that SAME fired card (source memory_card) both appear in
    the manifest -- prove_all must produce exactly ONE receipt (the card's real, provenance-resolved path),
    never a second one for the section that just redescribes it."""
    memory_mode.set_mode("prompt")
    manifest = {
        "influences_active": {
            "cards": [{"id": CARD_A, "text": "Be concise."}],
            "dials": [],
            "sections": {"available": True, "sections": [
                {"id": "sec_card_a", "name": "memory_card_a", "source": "memory_card"},
            ]},
        }
    }
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}), concise_card_ids=[CARD_A])
    out = receipts.prove_all(REDUNDANT_RUN, sub, manifest=manifest)
    assert len(out["receipts"]) == 1
    assert out["receipts"][0]["influence"].get("card_id") == CARD_A
    assert not any(r.get("kind") == "section" for r in out["receipts"])


# ============================================================================================================
# ============================================ forced-mode receipts ==
# ============================================================================================================
# A GRADED, per-token DEPENDENCE measurement via teacher-forced scoring (rederive.score_arm) -- no
# generation, no qwen substrate needed. Model-free: a ForcedFakeSub's .score_tokens is a deterministic
# function of the (block, steer_strengths, steer_vec) it was actually called with, so a test can craft
# EXACT with/without/control logprobs and assert on the resulting deltas/thresholds precisely -- the
# live-hardware acceptance numbers (real engine, real GPU) are reported separately, not here.

class ScoreFakeSteer:
    """Just enough of EngineSteer's surface for _forced_ablation's dial null-floor construction:
    .layer and a DETERMINISTIC .steer_vector() (a fixed per-axis direction scaled by the given
    strength) -- distinct from this file's other FakeSteer (which has no .steer_vector at all)."""

    def __init__(self, vecs, layer=14):
        self.vecs = dict(vecs)
        self.layer = layer
        self.calls = []

    def steer_vector(self, strengths):
        self.calls.append(dict(strengths or {}))
        active = {k: v for k, v in (strengths or {}).items() if v}
        if not active:
            return None
        dim = len(next(iter(self.vecs.values())))
        out = [0.0] * dim
        for name, val in active.items():
            vec = self.vecs.get(name)
            if not vec:
                continue
            for i, x in enumerate(vec):
                out[i] += x * val
        return out


class ForcedFakeSub:
    """.score_tokens() is a deterministic function of (block, steer_strengths, steer_vec) -- the with/
    without/control arms are distinguished by what's actually IN those three call args, so a test can
    hand-craft precise logprob scenarios (a real "10x above the null floor" case, a "no active dial"
    case, ...) without needing a live model."""

    def __init__(self, pieces, logprob_fn, steer=None):
        self.pieces = list(pieces)
        self.logprob_fn = logprob_fn
        self.steer = steer
        self.calls: list = []

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        self.calls.append({"messages": messages, "continuation_ids": continuation_ids, "block": block,
                          "steer_strengths": steer_strengths, "steer_vec": steer_vec})
        lps = self.logprob_fn(block, steer_strengths, steer_vec)
        return [{"id": i, "piece": p, "logprob": lp} for i, (p, lp) in enumerate(zip(self.pieces, lps))]


CARD_RUN = {
    "id": "run_card_x", "messages": [{"role": "user", "content": "weekend plans?"}],
    "response": "sure thing",
    "memory": {"cards_applied": ["The user loves rock climbing.", "The user has two cats."],
              "applied_ids": ["card_x", "card_y"],
              "prompt_block": "You are a helpful assistant talking with a returning user. Here is what "
                              "you know about them; use it naturally to tailor how you respond:\n"
                              "- The user loves rock climbing.\n- The user has two cats.",
              "mode": "prompt"},
    "behavior": {"active_dials": {}},
    "trace": {"token_ids": [1, 2, 3]},
}


def test_forced_receipt_card_ablation_shows_large_effect_clearing_the_null_floor(iso):
    """A hand-crafted scenario where removing the on-topic card crashes the answer tokens' logprob
    (real ablation), but swapping it for register-matched filler barely moves them (the null floor) --
    proving the MECHANISM correctly reports has_effect + a >5x floor-clearing ratio when the underlying
    numbers actually support it (REPRODUCE_AND_PROVE_PLAN.md's "sub-threshold receipt" headline)."""
    memory_mode.set_mode("prompt")

    def lp(block, steer_strengths, steer_vec):
        block = block or ""
        if "rock climbing" in block:
            return [-0.1, -0.1, -0.1]                       # WITH: the real card present
        if receipts._FILLER_TEXT[:15] in block:
            return [-0.15, -0.15, -0.15]                    # CONTROL: swapped for irrelevant filler
        return [-3.0, -3.0, -3.0]                           # WITHOUT: the card genuinely gone

    sub = ForcedFakeSub(["Sure", ",", " thing"], lp)
    rec = receipts.forced_receipt(CARD_RUN, {"card_id": "card_x"}, sub)
    assert rec["causal_verified"] is True
    assert rec["mode"] == "forced"
    assert rec["answer_tokens"] == ["Sure", ",", " thing"]
    assert rec["deltas"] == [round(-0.1 - -3.0, 6)] * 3
    assert rec["sum_nats"] == round(3 * 2.9, 6)
    assert rec["mean_nats_per_token"] == round(2.9, 6)
    assert rec["has_effect"] is True                       # 2.9 >> 0.05 mean AND >> 2.0 sum
    assert rec["top_dependent"][0]["delta"] == round(2.9, 6)
    nf = rec["null_floor"]
    assert nf["kind"] == "card_filler"
    assert nf["mean_nats_per_token"] == round(-0.1 - -0.15, 6)   # lp_with - lp_control == 0.05
    assert nf["ratio_real_over_floor"] == round(2.9 / nf["mean_nats_per_token"], 3)
    assert nf["ratio_real_over_floor"] > receipts._NULL_FLOOR_RATIO_MIN
    assert nf["exceeds_floor_by_order_of_magnitude"] is True
    assert rec["caveat"] == receipts._FORCED_CAVEAT
    assert "note" in rec


def test_forced_receipt_card_not_applied_is_an_honest_note_not_a_fabricated_zero(iso):
    memory_mode.set_mode("prompt")
    sub = ForcedFakeSub(["a"], lambda b, s, v: [-0.1])
    rec = receipts.forced_receipt(CARD_RUN, {"card_id": "no_such_card"}, sub)
    assert rec["causal_verified"] is False
    assert "not recorded as applied" in rec["note"]
    assert sub.calls == []                                  # never even scored -- nothing to ablate


DIAL_RUN = {
    "id": "run_dial_warm", "messages": [{"role": "user", "content": "hi"}], "response": "hello",
    "behavior": {"active_dials": {"warm": 1.0}},
    "trace": {"token_ids": [10, 11, 12]},
}


def _dial_lp(block, steer_strengths, steer_vec):
    if steer_strengths and steer_strengths.get("warm"):
        return [-0.1, -0.1, -0.1]                           # WITH: the real dial applied
    if steer_vec is not None:
        return [-0.2, -0.2, -0.2]                           # CONTROL: an equal-norm random direction
    return [-3.0, -3.0, -3.0]                               # WITHOUT: dial fully zeroed, nothing standing in


def test_forced_receipt_dial_ablation_shows_effect_and_a_small_null_floor(iso):
    steer = ScoreFakeSteer({"warm": [1.0, 0.0, 0.0]})
    sub = ForcedFakeSub(["a", "b", "c"], _dial_lp, steer=steer)
    rec = receipts.forced_receipt(DIAL_RUN, {"dial": "warm"}, sub)
    assert rec["causal_verified"] is True
    assert rec["mean_nats_per_token"] == round(2.9, 6)
    assert rec["has_effect"] is True
    nf = rec["null_floor"]
    assert nf["kind"] == "dial_random_vector"
    assert nf["mean_nats_per_token"] == round(0.1, 6)       # -0.1 - -0.2
    assert nf["ratio_real_over_floor"] > receipts._NULL_FLOOR_RATIO_MIN
    assert nf["exceeds_floor_by_order_of_magnitude"] is True
    # the control vector is a REAL raw steer_vec on the SAME (dial-zeroed) call, not a replacement of
    # steer_strengths -- so score_tokens saw steer_strengths={} (dial popped) AND a nonzero steer_vec
    control_calls = [c for c in sub.calls if c["steer_vec"] is not None]
    assert len(control_calls) == 1
    assert control_calls[0]["steer_strengths"] == {}
    assert len(control_calls[0]["steer_vec"]) == 3


def test_forced_receipt_dial_not_active_is_an_honest_note(iso):
    steer = ScoreFakeSteer({"warm": [1.0, 0.0, 0.0]})
    sub = ForcedFakeSub(["a"], lambda b, s, v: [-0.1], steer=steer)
    run = {"id": "run_x", "messages": [], "response": "x", "behavior": {"active_dials": {}},
          "trace": {"token_ids": [1]}}
    rec = receipts.forced_receipt(run, {"dial": "warm"}, sub)
    assert rec["causal_verified"] is True                   # the (no-op) ablation DID "apply" -- nothing changed
    assert rec["ablation_note"] == "dial 'warm' was not active on this run -- nothing to ablate"
    assert rec["mean_nats_per_token"] == 0.0                 # with == without == the same fixed logprob
    assert "null_floor" not in rec                           # steer_strengths.get(dial) is falsy -> no control


def test_forced_receipt_memory_off(iso):
    def lp(block, steer_strengths, steer_vec):
        return [-0.1, -0.1, -0.1] if block else [-3.0, -3.0, -3.0]
    sub = ForcedFakeSub(["a", "b", "c"], lp)
    rec = receipts.forced_receipt(CARD_RUN, {"memory_off": True}, sub)
    assert rec["causal_verified"] is True
    assert rec["has_effect"] is True
    assert rec["null_floor"]["kind"] == "block_filler"


def test_forced_receipt_memory_off_with_no_active_block_is_an_honest_note(iso):
    run = {"id": "run_bare", "messages": [], "response": "x", "memory": {}, "trace": {"token_ids": [1]}}
    sub = ForcedFakeSub(["a"], lambda b, s, v: [-0.1])
    rec = receipts.forced_receipt(run, {"memory_off": True}, sub)
    assert rec["causal_verified"] is True
    assert rec["ablation_note"] == "no active memory block on this run -- nothing to ablate"
    assert "null_floor" not in rec


def test_forced_receipt_behavior_off(iso):
    steer = ScoreFakeSteer({"warm": [1.0, 0.0, 0.0]})
    sub = ForcedFakeSub(["a", "b", "c"], _dial_lp, steer=steer)
    rec = receipts.forced_receipt(DIAL_RUN, {"behavior_off": True}, sub)
    assert rec["causal_verified"] is True
    assert rec["has_effect"] is True
    assert rec["null_floor"]["kind"] == "behavior_off_random_vector"


def test_forced_receipt_returns_none_on_bad_input(iso):
    sub = ForcedFakeSub(["a"], lambda b, s, v: [-0.1])
    assert receipts.forced_receipt(None, {"memory_off": True}, sub) is None
    assert receipts.forced_receipt({}, {"memory_off": True}, sub) is None
    assert receipts.forced_receipt(CARD_RUN, {}, sub) is None
    assert receipts.forced_receipt(CARD_RUN, None, sub) is None
    assert receipts.forced_receipt(CARD_RUN, {"nonsense": True}, sub) is None


def test_forced_receipt_degrades_honestly_when_substrate_cannot_score(iso):
    class NoScore:
        pass
    rec = receipts.forced_receipt(CARD_RUN, {"memory_off": True}, NoScore())
    assert rec["causal_verified"] is False
    assert "score_tokens" in rec["note"]


def test_forced_receipt_is_deterministic_across_repeated_calls(iso):
    steer = ScoreFakeSteer({"warm": [1.0, 0.0, 0.0]})
    sub = ForcedFakeSub(["a", "b", "c"], _dial_lp, steer=steer)
    a = receipts.forced_receipt(DIAL_RUN, {"dial": "warm"}, sub)
    b = receipts.forced_receipt(DIAL_RUN, {"dial": "warm"}, sub)
    assert a == b                                            # same seeded random control vector each time


# --------------------------------------------------------------------------------- helper-level unit tests

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
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.5}))
    default = receipts.receipt(RUN, {"dial": "warm"}, sub)
    explicit_regen = receipts.receipt(RUN, {"dial": "warm"}, FakeSub(mem=FakeMem(1.0),
                                                                     steer=FakeSteer({"warm": 0.5})))
    direct = receipts._receipt_regen(RUN, {"dial": "warm"}, FakeSub(mem=FakeMem(1.0),
                                                                    steer=FakeSteer({"warm": 0.5})))
    assert default == direct
    assert explicit_regen == direct
    assert "mode" not in default
    assert "forced" not in default
    assert "silent_influence" not in default


def test_prove_all_default_mode_is_byte_identical_to_the_pre_s3_function(iso):
    memory_mode.set_mode("prompt")
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.4}), concise_card_ids=[CARD_A, CARD_B])
    default = receipts.prove_all(REDUNDANT_RUN, sub)
    direct = receipts._prove_all_regen(REDUNDANT_RUN,
                                       FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.4}),
                                              concise_card_ids=[CARD_A, CARD_B]))
    assert default == direct
    assert "mode" not in default
    assert "forced_receipts" not in default


def test_receipt_mode_forced_returns_just_the_forced_receipt(iso):
    steer = ScoreFakeSteer({"warm": [1.0, 0.0, 0.0]})
    sub = ForcedFakeSub(["a", "b", "c"], _dial_lp, steer=steer)
    out = receipts.receipt(DIAL_RUN, {"dial": "warm"}, sub, mode="forced")
    assert out == receipts.forced_receipt(DIAL_RUN, {"dial": "warm"},
                                          ForcedFakeSub(["a", "b", "c"], _dial_lp,
                                                       steer=ScoreFakeSteer({"warm": [1.0, 0.0, 0.0]})))
    assert "baseline_reply" not in out                      # no regen fields at all in forced-only mode


def test_receipt_mode_both_combines_regen_and_forced_and_flags_silent_influence(iso):
    """The sub-threshold badge: regen shows NO text change (has_effect False) while forced clears the
    null floor by >5x -- exactly REPRODUCE_AND_PROVE_PLAN.md's headline scenario."""
    memory_mode.set_mode("prompt")
    regen_sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}), concise_card_ids=[])   # no text-diff signal
    forced_sub = ForcedFakeSub(["Sure", ",", " thing"],
                               lambda block, s, v: (
                                   [-0.1, -0.1, -0.1] if block and "rock climbing" in block else
                                   [-0.15, -0.15, -0.15] if block and receipts._FILLER_TEXT[:15] in block else
                                   [-3.0, -3.0, -3.0]))

    class BothSub:
        """One object that answers BOTH .chat() (regen) and .score_tokens() (forced) -- receipt(mode="both")
        drives each arm through the SAME `sub`."""
        def __init__(self):
            self.memory = regen_sub.memory
            self.steer = regen_sub.steer
        def chat(self, *a, **k):
            return regen_sub.chat(*a, **k)
        def score_tokens(self, *a, **k):
            return forced_sub.score_tokens(*a, **k)

    out = receipts.receipt(CARD_RUN, {"card_id": "card_x"}, BothSub(), mode="both")
    assert out["mode"] == "both"
    assert out["has_effect"] is False                        # regen: identical baseline/ablated text
    assert out["forced"]["has_effect"] is True
    assert out["forced"]["null_floor"]["exceeds_floor_by_order_of_magnitude"] is True
    assert out["silent_influence"] is True


def test_prove_all_mode_forced_never_touches_chat(iso):
    """mode="forced" needs no qwen substrate at all -- prove_all(mode="forced") must never call .chat()."""
    class ChatBoom:
        def chat(self, *a, **k):
            raise AssertionError("forced-only prove_all must never regenerate")
    manifest = {"influences_active": {"cards": [], "dials": [{"name": "warm", "value": 1.0}]}}
    steer = ScoreFakeSteer({"warm": [1.0, 0.0, 0.0]})

    class ForcedOnlySub(ChatBoom):
        def __init__(self):
            self.steer = steer
        def score_tokens(self, *a, **k):
            return ForcedFakeSub(["a", "b", "c"], _dial_lp, steer=steer).score_tokens(*a, **k)

    out = receipts.prove_all(DIAL_RUN, ForcedOnlySub(), manifest=manifest, mode="forced")
    assert out["mode"] == "forced"
    assert len(out["forced_receipts"]) == 1
    assert out["forced_receipts"][0]["influence"] == {"dial": "warm", "value": 1.0}


def test_prove_all_mode_both_includes_forced_receipts_alongside_regen(iso):
    memory_mode.set_mode("prompt")
    card_a, card_b = CARD_A, CARD_B
    steer = ScoreFakeSteer({"warm": [1.0, 0.0, 0.0]})

    class BothSub:
        def __init__(self):
            self.memory = FakeMem(1.0)
            self.steer = FakeSteer({"warm": 0.4})
            self.concise_card_ids = {card_a, card_b}
            self.seen = []
        @property
        def calls(self):
            return len(self.seen)
        def chat(self, messages, max_new=256, sample=True):
            return FakeSub(mem=self.memory, steer=self.steer,
                           concise_card_ids=self.concise_card_ids).chat(messages, max_new, sample)
        def score_tokens(self, *a, **k):
            return ForcedFakeSub(["a", "b", "c"], _dial_lp, steer=steer).score_tokens(*a, **k)

    out = receipts.prove_all(REDUNDANT_RUN, BothSub(), mode="both")
    assert out["mode"] == "both"
    assert len(out["receipts"]) == 3                         # unchanged regen leave-one-out
    assert len(out["forced_receipts"]) == 3                  # card_a, card_b, warm -- same fired influences
