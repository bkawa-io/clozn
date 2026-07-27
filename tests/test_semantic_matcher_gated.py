"""test_semantic_matcher_gated -- the M4 "Done" receipt (EXPLAIN_THIS_ANSWER_SPEC.md): the confabulation-diff
catches a SEEDED KNOWN divergence when driven by the real cross-encoder matcher
(semantic_matcher.nli_support_matcher), and narrate()'s trap-guard still holds with a real matcher wired in.

What this proves, and why it is the honesty-critical test the whole feature was building toward:
  A) THE SEEDED DIVERGENCE (the spec's "Done" line). A narrow manifest -- one "be concise" card (with a real
     quoted provenance) and one "warm" dial, and NOTHING about chess -- plus an unconstrained "why" that mixes
     two PARAPHRASED-TRUE claims ("I answered concisely" / "I used a warm tone") with one fluent CONFABULATION
     ("I drew on my knowledge of medieval chess strategy"). The real matcher must PASS the true ones and FLAG
     the confabulation, with per-influence attribution (concise -> the concise card, not the warm dial). And
     it must do so where the shipped lexical default FAILS -- lexical_default wrongly flags the paraphrase (no
     shared surface token with the card), so this is the receipt for the semantic matcher's added value.
  B) THE TRAP GUARD, with the REAL matcher (not just the scaffold's FakeSub-with-lexical tests): narrate()
     returns the constrained narration + flags and NEVER the raw confabulation, even end-to-end through the
     NLI judge.
  C) A REAL-MODEL DEMO (soft): the actual Qwen generates an unaided "why", and we show what the NLI matcher
     flags. Soft by design -- a real model's self-narration is not perfectly predictable, so the HARD proof is
     (A)'s seeded divergence; (C) is the end-to-end realism check + a real confabulation sample for the record.

Gated behind -m model: (A)/(B) load cross-encoder/nli-deberta-v3-base (~440MB, first run downloads it; CPU is
fine). (C) additionally loads Qwen2.5-1.5B on the GPU. Each skips CLEANLY if its resource is missing (mirrors
test_timetravel_determinism.py). Run:
    C:/Users/brigi/src/clozn/.venv/Scripts/python.exe -m pytest \
        research/tests/test_semantic_matcher_gated.py -m model -q -s
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.receipts.narrate as narrate                       # noqa: E402  (stdlib-only; no torch at import)
import clozn.receipts.semantic_matcher as sm        # noqa: E402  (stdlib-only top level; cross-encoder lazy)


# --- the seeded manifest: a "be concise" card with a real quote + a "warm" dial. NOTHING about chess. -------
def _explanation() -> dict:
    return {
        "confidence": {"available": False},
        "influences_active": {
            "cards": [{"id": "mem_concise", "text": "Keep replies short and to the point.",
                       "quoted_span": "please be brief", "gate": 0.71}],
            "dials": [{"name": "warm", "value": 0.6}],
        },
        "concepts": {"available": False},
    }


_TRUE_CONCISE = "I answered concisely."
_TRUE_WARM = "I used a friendly, warm tone."
_CONFAB = "I drew on my extensive knowledge of medieval chess strategy."
_UNCONSTRAINED = f"{_TRUE_CONCISE} {_TRUE_WARM} {_CONFAB}"


@pytest.mark.model
def test_nli_matcher_separates_seeded_true_from_confabulation():
    if not sm.available():
        pytest.skip("cross-encoder NLI unavailable (sentence-transformers / checkpoint missing)")
    expl = _explanation()

    r_concise = sm.nli_support_matcher(_TRUE_CONCISE, expl)
    r_warm = sm.nli_support_matcher(_TRUE_WARM, expl)
    r_confab = sm.nli_support_matcher(_CONFAB, expl)

    # 1) the honesty core: true paraphrases supported, confabulation flagged -- by a WIDE margin (not a
    #    knife-edge threshold, the fragility the memory-disorders classifier got burned by).
    assert r_concise["supported"] is True, r_concise
    assert r_warm["supported"] is True, r_warm
    assert r_confab["supported"] is False, r_confab
    margin = min(r_concise["score"], r_warm["score"]) - r_confab["score"]
    assert margin > 0.3, f"separation too thin to trust: {margin:.3f} ({r_concise} {r_warm} {r_confab})"

    # 2) per-influence ATTRIBUTION: the 'concise' claim credits the concise card, the 'warm' claim the dial.
    assert r_concise["matched_id"] == "mem_concise", r_concise
    assert r_warm["matched_id"] == "dial:warm", r_warm

    # 3) the semantic matcher's ADDED VALUE over the shipped default: lexical_default WRONGLY flags the true
    #    paraphrase (no shared surface token with the card text), the NLI matcher rescues it. If this
    #    precondition ever stops holding, the test below is no longer demonstrating the added value.
    lx = narrate.lexical_default(_TRUE_CONCISE, expl)
    assert lx["supported"] is False, f"precondition: lexical is supposed to MISS the paraphrase here, got {lx}"

    # 4) end-to-end through the honesty core: exactly the confab is unsupported; both true claims are not.
    diff = narrate.confabulation_diff(_UNCONSTRAINED, expl, support_matcher=sm.nli_support_matcher)
    unsupported = [e["claim"] for e in diff["unsupported_claims"]]
    assert any("chess" in c for c in unsupported), f"the confabulation should be flagged: {diff}"
    assert not any("concisely" in c for c in unsupported), f"the true concise claim must not be flagged: {diff}"
    assert not any("warm" in c for c in unsupported), f"the true warm claim must not be flagged: {diff}"

    print(f"\n[nli-matcher] concise={r_concise['score']:.3f} (->{r_concise['matched_id']}) "
          f"warm={r_warm['score']:.3f} (->{r_warm['matched_id']}) confab={r_confab['score']:.3f} "
          f"margin={margin:.3f} | lexical missed the paraphrase={not lx['supported']} | "
          f"diff flagged: {unsupported}", flush=True)


@pytest.mark.model
def test_narrate_trap_guard_holds_with_the_real_matcher():
    """narrate() returns the constrained narration + flags, and NEVER the raw confabulation -- verified with
    the REAL NLI matcher wired in, not only the scaffold's FakeSub-with-lexical tests. The FakeSub returns a
    confabulation on the unconstrained call and a receipt-cited narration on the constrained call."""
    if not sm.available():
        pytest.skip("cross-encoder NLI unavailable")

    class _FakeSub:
        def chat(self, messages, max_new=256, sample=True):
            joined = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
            if "Why did you answer that way" in joined:
                return _UNCONSTRAINED                                   # the confabulation sample
            return "I was brief [mem_concise] and warm [dial:warm]."    # the constrained, cited narration

    run = {"messages": [{"role": "user", "content": "Explain TCP vs UDP."}],
           "response": "TCP is reliable and ordered; UDP is fast and connectionless.",
           "memory": {"applied_ids": ["mem_concise"], "mode": "prompt", "cards_applied": [], "gate": 0.71},
           "behavior": {"active_dials": {"warm": 0.6}}}

    out = narrate.narrate(run, _FakeSub(), support_matcher=sm.nli_support_matcher)
    # exactly the four keys; no answer/why/response field anywhere (THE TRAP guard, structurally).
    assert set(out.keys()) == {"constrained_narration", "flags", "unsupported_claims", "note"}
    # the raw confabulation text never escapes as any value, and the chess claim never rides into the answer.
    assert _UNCONSTRAINED not in out.values()
    assert "chess" not in str(out["constrained_narration"]), out["constrained_narration"]
    print(f"\n[trap-guard+nli] flags={out['flags']}", flush=True)


@pytest.mark.model
def test_end_to_end_on_real_qwen_confabulation_demo():
    """SOFT, end-to-end: the actual Qwen2.5-1.5B generates an unaided "why", and we run it through the NLI
    matcher against the seeded manifest. Soft asserts (shape/liveness) + prints the real self-narration and
    what got flagged -- a real model's why is not perfectly predictable, so (A) is the hard proof and this is
    the realism check + a real confabulation sample for the record."""
    if not sm.available():
        pytest.skip("cross-encoder NLI unavailable")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import kv_timetravel as kvt
    except Exception as e:
        pytest.skip(f"torch/transformers unavailable: {e}")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA: the real-model demo needs the GPU")

    path = kvt.resolve_model_path("Qwen/Qwen2.5-1.5B-Instruct")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)

    class _QwenSub:
        def chat(self, messages, max_new=256, sample=False):
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new, do_sample=sample,
                                     pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    run = {"messages": [{"role": "user", "content": "In one line, what is a variable in programming?"}],
           "response": "A variable is a named box that holds a value you can read and change."}
    why = narrate.unconstrained_why(run, _QwenSub())["unconstrained_text_context_only"]
    diff = narrate.confabulation_diff(why, _explanation(), support_matcher=sm.nli_support_matcher)

    assert isinstance(diff.get("unsupported_claims"), list)
    assert diff.get("matcher") == "nli_support_matcher"
    flagged = [e["claim"] for e in diff["unsupported_claims"]]
    print(f"\n[real-why] {why!r}\n[flagged {len(flagged)}/{len(diff['claims'])}] {flagged}", flush=True)


@pytest.mark.model
def test_clause_split_catches_a_confab_hidden_in_a_compound_with_the_real_matcher():
    """The compound-sentence gap, closed on the REAL judge: a single sentence crediting one BACKED influence
    and one CONFABULATION. `clause_split` (the default claim_splitter) breaks it into two clauses; the NLI
    matcher passes the concise clause and flags the chess clause. Sentence-level splitting would have judged
    the whole sentence at once, letting the backed half mask the confabulation."""
    if not sm.available():
        pytest.skip("cross-encoder NLI unavailable")
    expl = _explanation()                                  # concise card + warm dial; NOTHING about chess
    text = "I answered concisely, and I drew on my deep knowledge of medieval chess."
    diff = narrate.confabulation_diff(text, expl, support_matcher=sm.nli_support_matcher)  # default clause_split
    claims = [c["claim"] for c in diff["claims"]]
    assert len(claims) == 2, f"clause_split should split the compound into two clauses: {claims}"
    unsupported = [e["claim"] for e in diff["unsupported_claims"]]
    assert any("chess" in c for c in unsupported), f"the confabulated clause must be flagged: {diff}"
    assert not any("concisely" in c for c in unsupported), f"the backed clause must NOT be flagged: {diff}"
    print(f"\n[clause-split+nli] claims={claims} | flagged={unsupported}", flush=True)
