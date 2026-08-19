"""test_section_influence -- model-free tests for the prompt-section influence FAST PATH:

  1. clozn.receipts.forced.forced_receipt()'s new "section" influence kind (mirrors test_receipts.py's own
     forced-mode section: a ForcedFakeSub-style fake, no model, no GPU, no network) -- INCLUDING the
     raw-prompt sibling (message_index: null sections, scored against `run["final_prompt"]` via
     EngineSubstrate.score_prompt_tokens instead of score_tokens; this used to be an honest "can't score
     it" skip and is now a real computation -- see forced.py's module docstring).
  2. clozn.server.routes.section_influence -- the pure `_shares`/`_summary` helpers, and the THIN
     POST /runs/<id>/section-influence endpoint wiring (mirrors test_receipts_server.py's no-socket
     object.__new__(H) handler-driving trick, isolated flat-file stores, a FakeSub standing in for the
     engine).
  3. clozn.cli.commands.run._format_influence_line -- the `--show-influence` / REPL `/influence` render,
     a pure function (dict in, text out) tested against canned payloads only.
  4. clozn.server.substrates.EngineSubstrate.score_prompt_tokens itself -- the raw-prompt-string scoring
     seam the raw-prompt section path (1) is built on, unit-tested directly against a fake engine (mirrors
     tests/test_engine_score.py's own score_tokens conventions -- object.__new__(cs.EngineSubstrate), a
     fake with just a .score() method).

No model, no GPU, no engine launch anywhere in this file -- every substrate here is a small deterministic
fake keyed on exactly the (messages / block / steer / raw prompt string) a real EngineSubstrate.
score_tokens / score_prompt_tokens would vary on.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn import receipts                                  # noqa: E402
from clozn.cli.commands.run import _format_influence_line    # noqa: E402
import clozn.runs.store as runlog                            # noqa: E402
import clozn.settings as clozn_settings                       # noqa: E402
from clozn.server import app as cs                            # noqa: E402
from clozn.server.routes import section_influence as si       # noqa: E402


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every flat-file store these tests touch (mirrors test_receipts.py / test_receipts_server.py's
    own `iso`). SUB starts at None -- the route's 503 path is the default; tests that want a working
    substrate set one explicitly."""
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    return tmp_path


# ============================================================================================================
# ================================= 1. forced_receipt()'s "section" influence kind ============================
# ============================================================================================================
# Card/dial ablations swap the block/steer args (test_receipts.py's own ForcedFakeSub keys off those); a
# SECTION ablation swaps message CONTENT instead (it splices `run["messages"]`, never a block or a
# strength) -- so this fake's logprob is a deterministic function of whether a marker string is still
# present anywhere in the `messages` it was actually called with.

class SectionForcedFakeSub:
    def __init__(self, pieces, marker, present_lp, absent_lp):
        self.pieces = list(pieces)
        self.marker = marker
        self.present_lp = present_lp
        self.absent_lp = absent_lp
        self.calls: list = []

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        self.calls.append({"messages": messages, "block": block, "steer_strengths": steer_strengths})
        text = " ".join(str((m or {}).get("content", "")) for m in (messages or []) if isinstance(m, dict))
        lps = self.present_lp if self.marker in text else self.absent_lp
        return [{"id": i, "piece": p, "logprob": lp} for i, (p, lp) in enumerate(zip(self.pieces, lps))]


RAG_TEXT = "RAG: the user loves kayaking."

SECTION_RUN = {
    "id": "run_sec_x",
    "messages": [
        {"role": "system", "content": "base system prompt"},
        {"role": "user", "content": RAG_TEXT},
        {"role": "user", "content": "What hobby do I have?"},
    ],
    "response": "kayaking",
    "behavior": {"active_dials": {}},
    "trace": {"token_ids": [101, 102]},
    "sections": [
        {"id": "sec_rag", "name": "rag_context", "source": "auto",
         "parts": [{"message_index": 1, "start": 0, "end": len(RAG_TEXT)}],
         "char_count": len(RAG_TEXT), "preview": RAG_TEXT[:20]},
    ],
}


def test_forced_receipt_section_ablation_shows_effect_no_null_floor(iso):
    """A real ablation (the marker message is fully spliced out): with-tokens see it, without-tokens
    don't -- deltas/sum/mean come out exactly as hand-crafted, has_effect fires, and -- unlike a card or
    dial -- there is NO null_floor key at all (no register-matched control exists for an arbitrary prompt
    span; see forced.py's own module docstring)."""
    sub = SectionForcedFakeSub(["kay", "aking"], "kayaking", present_lp=[-0.1, -0.1], absent_lp=[-3.0, -3.0])
    rec = receipts.forced_receipt(SECTION_RUN, {"section": "rag_context", "source": "auto"}, sub)
    assert rec["causal_verified"] is True
    assert rec["mode"] == "forced"
    assert rec["answer_tokens"] == ["kay", "aking"]
    assert rec["deltas"] == [round(2.9, 6)] * 2
    assert rec["sum_nats"] == round(5.8, 6)
    assert rec["mean_nats_per_token"] == round(2.9, 6)
    assert rec["has_effect"] is True
    assert "null_floor" not in rec
    assert rec["caveat"] == receipts._FORCED_CAVEAT
    assert len(sub.calls) == 2                              # with + without; no control call at all


def test_forced_receipt_section_unknown_name_is_an_honest_note(iso):
    sub = SectionForcedFakeSub(["a"], "never-matched", present_lp=[-0.1], absent_lp=[-0.1])
    rec = receipts.forced_receipt(SECTION_RUN, {"section": "not_a_real_section", "source": "auto"}, sub)
    assert rec["causal_verified"] is False
    assert "no section named" in rec["note"]
    assert sub.calls == []                                   # never even scored -- nothing to ablate


def test_forced_receipt_section_no_manifest_is_an_honest_note(iso):
    run = {"id": "run_bare", "messages": [{"role": "user", "content": "hi"}], "response": "hey",
          "trace": {"token_ids": [1]}}
    sub = SectionForcedFakeSub(["a"], "x", present_lp=[-0.1], absent_lp=[-0.1])
    rec = receipts.forced_receipt(run, {"section": "rag_context", "source": "auto"}, sub)
    assert rec["causal_verified"] is False
    assert "no section manifest" in rec["note"]


RAG_SEGMENT = "RAG: whales are mammals."
RAWPROMPT_FINAL_PROMPT = "SYSTEM\n" + RAG_SEGMENT + "\nQ: what are whales?"
_RAG_START = RAWPROMPT_FINAL_PROMPT.index(RAG_SEGMENT)
_RAG_END = _RAG_START + len(RAG_SEGMENT)

RAWPROMPT_SECTION_RUN = {
    "id": "run_sec_raw",
    "messages": [{"role": "user", "content": "irrelevant -- offsets are into final_prompt, not this list"}],
    "final_prompt": RAWPROMPT_FINAL_PROMPT,
    "response": "mammals",
    "trace": {"token_ids": [9]},
    "sections": [{"id": "sec_rag", "name": "rag_context", "source": "auto",
                 "parts": [{"message_index": None, "start": _RAG_START, "end": _RAG_END}],
                 "char_count": len(RAG_SEGMENT), "preview": RAG_SEGMENT}],
}


class RawPromptForcedFakeSub:
    """The raw-prompt sibling of SectionForcedFakeSub: this fake has NO `score_tokens` at all (a
    misdirected call would fail loudly, AttributeError, rather than silently succeeding on the wrong
    seam) -- only `score_prompt_tokens`, whose logprob is a deterministic function of whether `marker` is
    still present in the PROMPT STRING it was actually called with."""

    def __init__(self, pieces, marker, present_lp, absent_lp):
        self.pieces = list(pieces)
        self.marker = marker
        self.present_lp = present_lp
        self.absent_lp = absent_lp
        self.calls: list = []       # one entry per score_prompt_tokens() call, in call order

    def score_prompt_tokens(self, prompt, continuation_ids=None, *, continuation=None, topk=0):
        self.calls.append({"prompt": prompt, "continuation_ids": continuation_ids})
        lps = self.present_lp if self.marker in prompt else self.absent_lp
        return [{"id": i, "piece": p, "logprob": lp} for i, (p, lp) in enumerate(zip(self.pieces, lps))]


def test_forced_receipt_section_raw_prompt_now_scores_via_final_prompt_splice(iso):
    """THE FIX: message_index: null no longer degrades to an honest skip -- it splices final_prompt and
    teacher-forces both arms via score_prompt_tokens. Deltas/sum/mean come out exactly as hand-crafted
    (same shape and math as the message-anchored case), and -- like that case -- there is still no
    null_floor (no register-matched control for an arbitrary prompt span)."""
    sub = RawPromptForcedFakeSub(["m"], RAG_SEGMENT, present_lp=[-0.1], absent_lp=[-3.0])
    rec = receipts.forced_receipt(RAWPROMPT_SECTION_RUN, {"section": "rag_context", "source": "auto"}, sub)
    assert rec["causal_verified"] is True
    assert rec["mode"] == "forced"
    assert rec["answer_tokens"] == ["m"]
    assert rec["deltas"] == [round(2.9, 6)]
    assert rec["sum_nats"] == round(2.9, 6)
    assert rec["mean_nats_per_token"] == round(2.9, 6)
    assert rec["has_effect"] is True
    assert "null_floor" not in rec
    assert rec["caveat"] == receipts._FORCED_CAVEAT
    assert len(sub.calls) == 2                              # with (baseline) + without (ablated)


def test_forced_receipt_section_raw_prompt_ablated_arm_actually_lacks_the_section_text(iso):
    """Not just "a delta came back" -- prove the WITH call saw the unmodified final_prompt and the
    WITHOUT call's prompt genuinely has the section spliced out (not just a different score, in case the
    fake's marker-matching logic were the only thing exercised)."""
    sub = RawPromptForcedFakeSub(["m"], RAG_SEGMENT, present_lp=[-0.1], absent_lp=[-3.0])
    receipts.forced_receipt(RAWPROMPT_SECTION_RUN, {"section": "rag_context", "source": "auto"}, sub)
    with_prompt = sub.calls[0]["prompt"]
    without_prompt = sub.calls[1]["prompt"]
    assert with_prompt == RAWPROMPT_FINAL_PROMPT                      # baseline: final_prompt UNCHANGED
    assert RAG_SEGMENT in with_prompt
    assert RAG_SEGMENT not in without_prompt                          # ablated: section spliced OUT
    assert without_prompt == "SYSTEM\n\nQ: what are whales?"          # exact spliced result
    assert sub.calls[0]["continuation_ids"] == [9]                    # the run's own stored trace ids
    assert sub.calls[1]["continuation_ids"] == [9]


def test_forced_receipt_section_raw_prompt_degrades_when_substrate_has_no_score_prompt_tokens(iso):
    """The ceiling now is the SUBSTRATE's capability, not the section's anchor: a substrate exposing only
    score_tokens (the old message-anchored seam) still can't score a raw-prompt section, and says so --
    it never silently falls back to score_tokens with a fabricated messages list."""
    sub = SectionForcedFakeSub(["m"], RAG_SEGMENT, present_lp=[-0.1], absent_lp=[-3.0])
    rec = receipts.forced_receipt(RAWPROMPT_SECTION_RUN, {"section": "rag_context", "source": "auto"}, sub)
    assert rec["causal_verified"] is False
    assert "score_prompt_tokens" in rec["note"]


def test_forced_receipt_section_deterministic_across_repeated_calls(iso):
    sub = SectionForcedFakeSub(["kay", "aking"], "kayaking", present_lp=[-0.1, -0.1], absent_lp=[-3.0, -3.0])
    a = receipts.forced_receipt(SECTION_RUN, {"section": "rag_context", "source": "auto"}, sub)
    b = receipts.forced_receipt(SECTION_RUN, {"section": "rag_context", "source": "auto"}, sub)
    assert a == b


# ============================================================================================================
# ============================================= 2a. pure helper math (route-level) ============================
# ============================================================================================================

def test_shares_normal_case_sums_to_one():
    shares = si._shares([2.9, 0.05])
    assert shares == [round(2.9 / 2.95, 6), round(0.05 / 2.95, 6)]
    assert abs(sum(shares) - 1.0) < 1e-9


def test_shares_all_zero_guard_never_nan():
    shares = si._shares([0.0, 0.0, 0.0])
    assert shares == [0.0, 0.0, 0.0]
    assert all(s == s for s in shares)                       # s == s is False only for NaN


def test_shares_empty_list():
    assert si._shares([]) == []


def test_shares_only_magnitude_matters_not_sign():
    assert si._shares([-2.0, 2.0]) == [0.5, 0.5]


def test_summary_buckets_negligible():
    s = si._summary(0.01)
    assert "negligible" in s
    assert "caus" not in s.lower()


def test_summary_buckets_small_worse_and_better():
    assert si._summary(-0.2) == "removing this section makes the stored answer fit slightly worse"
    assert si._summary(0.3) == "removing this section makes the stored answer fit slightly better"


def test_summary_buckets_substantial_worse_and_better():
    assert si._summary(-0.8) == "removing this section makes the stored answer fit much worse"
    assert si._summary(1.2) == "removing this section makes the stored answer fit much better"
    assert "caus" not in si._summary(-0.8).lower()


# ============================================================================================================
# ==================================== 2b. POST /runs/<id>/section-influence ==================================
# ============================================================================================================
# Drives the REAL clozn_server do_POST handler with no socket (the object.__new__(H) trick used throughout
# this repo's *_server tests) -- so this proves the actual route registration in clozn.server.app, not just
# the module's functions in isolation.

def _dispatch(method, path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _post(path, body_obj=None):
    return _dispatch("POST", path, body_obj)


RAG_TEXT2 = "RAG: whales are mammals."
FEWSHOT_TEXT = "Example: Q: 2+2? A: 4"


class RouteFakeSub:
    """.score_tokens()'s logprob depends on which of the two markers ('whales are mammals' / '2+2') are
    still present in `messages` -- lets the happy-path test hand-craft one section with a LARGE effect and
    one with a SMALL one, so the ranking/share math is exercised end to end, not just with a single value."""
    name = "engine"

    def __init__(self):
        self.calls = 0

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        self.calls += 1
        text = " ".join(str((m or {}).get("content", "")) for m in (messages or []) if isinstance(m, dict))
        has_rag = "whales are mammals" in text
        has_fewshot = "2+2" in text
        if has_rag and has_fewshot:
            lp = -0.1                                        # baseline: both sections present
        elif has_fewshot and not has_rag:
            lp = -3.0                                        # rag_context removed -- big effect
        elif has_rag and not has_fewshot:
            lp = -0.15                                       # few_shot removed -- small effect
        else:
            lp = -5.0
        return [{"id": 1, "piece": "mammals", "logprob": lp}]


def _seed_two_section_run():
    messages = [
        {"role": "system", "content": "base"},
        {"role": "user", "content": RAG_TEXT2},
        {"role": "user", "content": FEWSHOT_TEXT},
        {"role": "user", "content": "What are whales?"},
    ]
    sections = [
        {"id": "sec_rag_context", "name": "rag_context", "source": "api",
         "parts": [{"message_index": 1, "start": 0, "end": len(RAG_TEXT2)}],
         "char_count": len(RAG_TEXT2), "preview": RAG_TEXT2[:20]},
        {"id": "sec_few_shot", "name": "few_shot", "source": "auto",
         "parts": [{"message_index": 2, "start": 0, "end": len(FEWSHOT_TEXT)}],
         "char_count": len(FEWSHOT_TEXT), "preview": FEWSHOT_TEXT[:20]},
    ]
    return runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="engine",
                         messages=messages, response="mammals", 
                         trace={"token_ids": [1]}, sections=sections)


def test_route_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/section-influence")
    assert out == {"error": "run not found"}


def test_route_no_manifest_is_a_clean_empty_200(iso):
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey")
    out = _post(f"/runs/{rid}/section-influence")
    assert out == {"run_id": rid, "sections": [], "any_meaningful": False,
                   "note": "no section manifest on this run"}


def test_route_needs_the_substrate_503(iso):
    rid = _seed_two_section_run()                            # iso's default SUB is None
    out = _post(f"/runs/{rid}/section-influence")
    assert out == {"error": "section-influence requires worker token scoring"}


def test_route_happy_path_ranks_sections_by_measured_effect(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", RouteFakeSub())
    rid = _seed_two_section_run()
    out = _post(f"/runs/{rid}/section-influence")
    assert out["run_id"] == rid
    assert out["method"] == "teacher_forced"
    assert "NOT causal proof" in out["note"]
    assert "receipts" in out["note"]
    assert out["baseline_logprob"] == round(-0.1, 6)
    assert out["any_meaningful"] is True                     # rag_context's -2.9 delta clears the floor
    names = [s["name"] for s in out["sections"]]
    assert names == ["rag_context", "few_shot"]              # biggest measured effect ranked first
    shares = [s["influence_share"] for s in out["sections"]]
    assert shares[0] > shares[1]
    assert abs(sum(shares) - 1.0) < 1e-6
    for s in out["sections"]:
        assert set(s.keys()) == {"id", "name", "source", "log_prob_delta", "influence_share",
                                 "per_token_delta", "summary"}
        assert s["log_prob_delta"] < 0                        # both removals make the answer fit worse here
        assert "worse" in s["summary"]


class NegligibleFakeSub:
    """Every arm scores ~the same -- removing EITHER section barely moves the fit (a parametric-knowledge
    answer: the reply came from the model's weights, not the prompt). The share math still normalizes to
    sum=1, so this is the case the any_meaningful guard exists to catch."""
    name = "engine"

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        text = " ".join(str((m or {}).get("content", "")) for m in (messages or []) if isinstance(m, dict))
        both = "whales are mammals" in text and "2+2" in text
        return [{"id": 1, "piece": "x", "logprob": -0.10 if both else -0.12}]   # |delta| = 0.02 < 0.05 floor


def test_route_flags_when_no_section_meaningfully_mattered(iso, monkeypatch):
    # Both removals move the fit by 0.02 nats/token -- under _NEGLIGIBLE. Shares still sum to 1 (that's the
    # trap), but any_meaningful must be False and the note must warn the shares aren't a ranking.
    monkeypatch.setattr(cs, "SUB", NegligibleFakeSub())
    rid = _seed_two_section_run()
    out = _post(f"/runs/{rid}/section-influence")
    assert out["any_meaningful"] is False
    assert len(out["sections"]) == 2                          # sections still scored, not dropped
    assert abs(sum(s["influence_share"] for s in out["sections"]) - 1.0) < 1e-6   # the misleading sum=1
    assert "not a meaningful ranking" in out["note"]          # ...but the note says don't trust it
    assert all("negligible" in s["summary"] for s in out["sections"])


def test_any_meaningful_true_when_one_row_clears_the_floor():
    assert si.any_meaningful([{"per_token_delta": -0.01}, {"per_token_delta": -2.4}]) is True


def test_any_meaningful_false_when_all_rows_below_floor():
    assert si.any_meaningful([{"per_token_delta": 0.01}, {"per_token_delta": -0.049}]) is False


def test_any_meaningful_empty_is_false():
    assert si.any_meaningful([]) is False


def test_any_meaningful_tolerates_missing_or_none_delta():
    assert si.any_meaningful([{"per_token_delta": None}, {}]) is False


# ============================================================================================================
# ================================== 3. CLI --show-influence / REPL /influence rendering =======================
# ============================================================================================================
# _format_influence_line is a pure function (dict in, text out) -- no server, no model, no I/O -- mirrors
# commands.explain's format_explain/format_narrate testing style exactly.

def test_format_influence_line_ranks_and_labels_only_the_first_entry():
    payload = {
        "run_id": "run_x", "method": "teacher_forced", "note": "...", "baseline_logprob": -10.0,
        "sections": [
            {"id": "sec_a", "name": "rag_context", "source": "api", "log_prob_delta": -3.8,
             "influence_share": 0.42, "per_token_delta": -0.031, "summary": "..."},
            {"id": "sec_b", "name": "few_shot", "source": "auto", "log_prob_delta": -0.7,
             "influence_share": 0.08, "per_token_delta": -0.01, "summary": "..."},
            {"id": "sec_c", "name": "auto_3", "source": "auto", "log_prob_delta": -0.3,
             "influence_share": 0.03, "per_token_delta": -0.005, "summary": "..."},
        ],
    }
    assert _format_influence_line(payload) == (
        "- sections: rag_context (42% influence), few_shot (8%), auto_3 (3%)  [approximate, teacher-forced]"
    )


def test_format_influence_line_reorders_by_descending_share():
    payload = {"sections": [
        {"id": "sec_a", "name": "small_one", "source": "api", "log_prob_delta": -0.1,
         "influence_share": 0.1, "per_token_delta": -0.01, "summary": "..."},
        {"id": "sec_b", "name": "big_one", "source": "auto", "log_prob_delta": -0.9,
         "influence_share": 0.9, "per_token_delta": -0.09, "summary": "..."},
    ]}
    line = _format_influence_line(payload)
    assert line.startswith("- sections: big_one (90% influence), small_one (10%)")


def test_format_influence_line_error_is_one_honest_line():
    payload = {"error": "couldn't reach the Clozn gateway on port 8080 (Connection refused)"}
    assert _format_influence_line(payload) == (
        "- sections: couldn't reach the Clozn gateway on port 8080 (Connection refused)"
    )


def test_format_influence_line_no_sections_uses_the_notes_note():
    payload = {"run_id": "run_x", "sections": [], "note": "no section manifest on this run"}
    assert _format_influence_line(payload) == "- sections: no section manifest on this run"


def test_format_influence_line_empty_sections_with_no_note_still_one_honest_line():
    assert _format_influence_line({"sections": []}) == "- sections: no sections on this run"


def test_format_influence_line_never_raises_on_malformed_payload():
    assert _format_influence_line(None) == "- sections: no sections on this run"
    assert _format_influence_line("not a dict") == "- sections: no sections on this run"
    assert _format_influence_line({}) == "- sections: no sections on this run"
    assert _format_influence_line({"sections": "not a list"}) == "- sections: no sections on this run"


# ============================================================================================================
# ==================== 4. EngineSubstrate.score_prompt_tokens -- the raw-prompt scoring seam ==================
# ============================================================================================================
# The seam the raw-prompt section path (section 1, above) is built on: unlike score_tokens, this skips
# prompt assembly (_inject_block/_engine_tmpl) entirely -- the caller's `prompt` string rides to
# engine.score() UNCHANGED. Mirrors tests/test_engine_score.py's own score_tokens conventions
# (object.__new__(cs.EngineSubstrate), a fake with just a .score() method) -- no model, no GPU.

class _FakePromptScoreEngine:
    """Stands in for cloze_engine.EngineClient inside score_prompt_tokens: only `.score()` -- deliberately
    NO `.apply_template()` at all, unlike test_engine_score.py's `_FakeScoreEngine` -- because this seam
    never templates anything. If score_prompt_tokens ever regressed into calling apply_template, this
    fake would raise AttributeError instead of silently passing."""

    def __init__(self, reply=None):
        self.calls: list = []
        self._reply = reply if reply is not None else {"tokens": []}

    def score(self, **kw):
        self.calls.append(kw)
        return self._reply


def _bare_engine_substrate(engine):
    """EngineSubstrate via object.__new__ (mirrors test_engine_score.py's own helper) -- exercises
    score_prompt_tokens without a real engine/steer/health call."""
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = engine
    return sub


def test_score_prompt_tokens_passes_the_prompt_straight_through_untemplated():
    reply = {"tokens": [{"id": 1, "piece": "a", "logprob": -0.2}]}
    fe = _FakePromptScoreEngine(reply=reply)
    sub = _bare_engine_substrate(fe)
    out = sub.score_prompt_tokens("SYSTEM\n\nQ: what are whales?", [1, 2, 3], topk=5)
    assert fe.calls[-1]["prompt"] == "SYSTEM\n\nQ: what are whales?"     # byte-identical, no template
    assert fe.calls[-1]["continuation_ids"] == [1, 2, 3]
    assert fe.calls[-1]["topk"] == 5
    assert out == reply["tokens"]


def test_score_prompt_tokens_continuation_ids_take_precedence_over_text():
    fe = _FakePromptScoreEngine()
    sub = _bare_engine_substrate(fe)
    sub.score_prompt_tokens("p", [1, 2], continuation="ignored text")
    assert fe.calls[-1]["continuation_ids"] == [1, 2]
    assert "continuation" not in fe.calls[-1]


def test_score_prompt_tokens_continuation_text_fallback_when_no_ids():
    fe = _FakePromptScoreEngine()
    sub = _bare_engine_substrate(fe)
    sub.score_prompt_tokens("p", None, continuation="hello world")
    assert fe.calls[-1]["continuation"] == "hello world"
    assert "continuation_ids" not in fe.calls[-1]


def test_score_prompt_tokens_forwards_ids_as_ints():
    fe = _FakePromptScoreEngine()
    sub = _bare_engine_substrate(fe)
    sub.score_prompt_tokens("p", [1.0, 2.0, 3])
    assert fe.calls[-1]["continuation_ids"] == [1, 2, 3]


def test_score_prompt_tokens_tolerates_a_degraded_reply():
    fe = _FakePromptScoreEngine(reply={})
    sub = _bare_engine_substrate(fe)
    assert sub.score_prompt_tokens("p", [1]) == []
