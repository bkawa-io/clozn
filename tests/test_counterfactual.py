"""test_counterfactual -- model-free tests for research/counterfactual.py (EXPLAIN_THIS_ANSWER_SPEC.md
Milestone 3).

No model, no GPU, no torch: drives counterfactual.counterfactual() / counterfactual.dose_sweep() against a
FAKE substrate (mirrors test_receipts.py's FakeSub/FakeMem/FakeSteer) whose .chat() is a DETERMINISTIC
function of exactly which dial values are live at call time -- so a counterfactual's baseline-vs-override
delta is driven ONLY by whatever replay.py actually changed, never by randomness.

What's under test:
  * the BOTH-ARMS-GREEDY seam: counterfactual() calls sub.chat() exactly twice, both greedy (sample=False),
    both over the run's own stored messages -- never touching the run's stored sampled `response`.
  * the baseline arm uses the substrate's currently-live dials (untouched); the counterfactual arm uses
    the override -- and replay.py's restore-in-a-finally leaves the live substrate exactly as it found it.
  * delta math is receipts.receipt_metrics(baseline, counterfactual) EXACTLY -- no reimplementation.
  * the MANDATORY coherence axis (law #6): a clean arm reads not-degenerate; a scripted-degenerate arm
    (immediate word-3gram repetition) is flagged, even though causal_verified stays True (the override DID
    take hold -- it just derailed the model). causal_verified and coherence are orthogonal.
  * the unapplied-override honesty guard: an override for an axis the fake steer doesn't recognize never
    shows up in the replayed run's own recorded dial state -> causal_verified: False + an override_note,
    never a silently-claimed "no effect". A 0.0 override is never flagged (indistinguishable from "already
    off", the same convention receipts.py's own dial-ablation-to-zero relies on).
  * dose_sweep()'s response curve shape (delta + coherence per value), the derailment flag + which exact
    values crater, and that it costs exactly 2 * len(values) generations (each point fully independent).
  * never raises on garbage input (bad run / bad overrides / no substrate / bad dial / bad values).
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.replay.counterfactual as counterfactual    # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402


from clozn import receipts           # noqa: E402
import clozn.runs.store as runlog              # noqa: E402


# --- fakes (mirror test_replay.py / test_receipts.py's FakeSteer/FakeMem/FakeSub) ----------------------
# chat() is a pure function of the "warm"/"concise" dial VALUES (not just on/off), so overriding to a
# different value is guaranteed to change the reply -- and one scripted trapdoor (warm >= DEGENERATE_WARM)
# produces gibberish, giving the coherence axis something real to catch.

class FakeSteer:
    def __init__(self, strength=None, known_axes=None):
        self.strength = dict(strength or {})
        # axes this fake substrate actually "recognizes"; None means "recognizes everything" (the common
        # case). set() on an unrecognized name is silently ignored -- mirrors a real steer that doesn't
        # know an axis, and mirrors replay.py's own swallowed try/except around steer.set().
        self.known_axes = set(known_axes) if known_axes is not None else None

    def set(self, name, value):
        name = str(name)
        if self.known_axes is not None and name not in self.known_axes:
            return
        self.strength[name] = float(value)

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
    """chat() depends only on the current warm/concise dial VALUES -- no randomness. warm >=
    DEGENERATE_WARM is a scripted derailment: the model "degenerates" into immediate word repetition,
    exactly the failure mode the coherence axis exists to catch."""
    DEGENERATE_WARM = 3.0

    def __init__(self, mem=None, steer=None):
        self.memory = mem if mem is not None else FakeMem()
        self.steer = steer if steer is not None else FakeSteer()
        self.seen: list = []      # one entry per chat() call, in call order

    @property
    def calls(self):
        return len(self.seen)

    def chat(self, messages, max_new=256, sample=True):
        self.seen.append({"messages": messages, "sample": sample, "dials": dict(self.steer.strength)})
        warm = float(self.steer.strength.get("warm", 0.0) or 0.0)
        concise = float(self.steer.strength.get("concise", 0.0) or 0.0)
        if warm >= self.DEGENERATE_WARM:
            return "warm warm warm today"     # immediate 3-gram word repetition -- the trapdoor
        base = "Short answer." if concise > 0 else "A plain reply about the weather today."
        if warm > 0:
            base += " Hope that warms you right up!"
        return base


RUN = {"id": "run_parent0", "model": "clozn-qwen", "substrate": "QwenSubstrate",
       "messages": [{"role": "user", "content": "how should I phrase this email"}],
       "response": "THE STORED SAMPLED REPLY -- must never be used as anyone's baseline",
       "behavior": {"active_dials": {}}}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


# ============================================================================================== coherence
def test_coherence_proxy_flags_empty_repeat_char_runaway_and_script_switch():
    assert counterfactual._coherence("")["degenerate"] is True
    assert counterfactual._coherence("   ")["degenerate"] is True
    assert counterfactual._coherence("no no no thanks")["degenerate"] is True
    assert counterfactual._coherence("no no no thanks")["reason"] == "repeat-3gram"
    assert counterfactual._coherence("wow!!!!!")["degenerate"] is True
    assert counterfactual._coherence("wow!!!!!")["reason"] == "char-runaway"
    assert counterfactual._coherence("privet there")["degenerate"] is False   # ASCII -- not a script switch
    assert counterfactual._coherence("привет there")["degenerate"] is True     # Cyrillic letters -- real switch
    assert counterfactual._coherence("这是中文回复")["degenerate"] is True        # CJK letters -- real switch
    assert counterfactual._coherence("a perfectly normal reply, thanks!")["degenerate"] is False
    # emoji / curly quotes / em-dash are non-ASCII SYMBOLS+PUNCTUATION, NOT a script switch -- Gemma-2 is
    # emoji-heavy and coherent, and the old catch-all false-flagged it (found in mirror_bench cross-family).
    assert counterfactual._coherence("Sure! I'd love to help \U0001f60a✨")["degenerate"] is False
    assert counterfactual._coherence("It’s a great idea — truly.")["degenerate"] is False
    assert counterfactual._coherence("Enjoy your café visit!")["degenerate"] is False   # lone accent, below threshold


# ================================================================================ both-arms-greedy shape
def test_counterfactual_is_exactly_two_greedy_calls_over_the_runs_own_messages(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.2}))
    rec = counterfactual.counterfactual(RUN, {"warm": 0.9}, sub)
    assert rec is not None
    assert sub.calls == 2                                       # baseline + counterfactual, nothing more
    assert all(c["sample"] is False for c in sub.seen)           # BOTH arms greedy
    assert all(c["messages"] == RUN["messages"] for c in sub.seen)


def test_baseline_arm_uses_the_runs_actual_live_dials_not_the_override(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.2}))
    counterfactual.counterfactual(RUN, {"warm": 0.9}, sub)
    assert sub.seen[0]["dials"].get("warm") == 0.2               # baseline: untouched
    assert sub.seen[1]["dials"].get("warm") == 0.9               # counterfactual: overridden
    assert sub.steer.strength == {"warm": 0.2}                   # restored exactly (replay.py's own contract)


def test_counterfactual_reply_reflects_the_overridden_dial_value(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    rec = counterfactual.counterfactual(RUN, {"warm": 0.9}, sub)
    assert rec["overrides_applied"] == {"warm": 0.9}
    assert rec["baseline_reply"] == "A plain reply about the weather today."
    assert rec["counterfactual_reply"].endswith("warms you right up!")
    assert rec["has_effect"] is True
    assert rec["causal_verified"] is True


def test_counterfactual_never_uses_the_stored_sampled_reply_as_either_arm(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    rec = counterfactual.counterfactual(RUN, {"warm": 0.9}, sub)
    assert RUN["response"] not in (rec["baseline_reply"], rec["counterfactual_reply"])
    assert "sampled" in rec["note"].lower() and "baseline" in rec["note"].lower()


def test_counterfactual_cost_note_says_decode_time_cheap_vs_reprefill(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    rec = counterfactual.counterfactual(RUN, {"warm": 0.9}, sub)
    cn = rec["cost_note"].lower()
    assert "decode" in cn or "cheap" in cn
    assert "re-prefill" in cn or "prefix" in cn


def test_counterfactual_delta_matches_receipts_receipt_metrics_exactly(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    rec = counterfactual.counterfactual(RUN, {"concise": 0.7}, sub)
    assert rec["delta"] == receipts.receipt_metrics(rec["baseline_reply"], rec["counterfactual_reply"])


def test_counterfactual_can_show_no_effect_honestly(iso):
    # concise dial already active on the live substrate; overriding it to the SAME nonzero value changes
    # nothing observable in this FakeSub's chat() (it only branches on concise > 0) -- a real "no effect".
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"concise": 0.5}))
    rec = counterfactual.counterfactual(RUN, {"concise": 0.8}, sub)
    assert rec["causal_verified"] is True
    assert rec["has_effect"] is False
    assert rec["baseline_reply"] == rec["counterfactual_reply"] == "Short answer."


# ============================================================================ the coherence axis (law #6)
def test_coherence_is_clean_on_a_normal_counterfactual_arm(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    rec = counterfactual.counterfactual(RUN, {"warm": 0.9}, sub)
    assert rec["coherence"] == {"degenerate": False, "reason": ""}


def test_coherence_flags_a_scripted_degenerate_counterfactual_arm(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    rec = counterfactual.counterfactual(RUN, {"warm": FakeSub.DEGENERATE_WARM}, sub)
    assert rec["counterfactual_reply"] == "warm warm warm today"
    assert rec["coherence"]["degenerate"] is True
    assert rec["coherence"]["reason"] == "repeat-3gram"
    # the override DID take hold (mechanism proven) even though the result derailed (quality failed) --
    # a big delta here must read as "derailed", never silently as "just a bigger effect".
    assert rec["has_effect"] is True
    assert rec["causal_verified"] is True


# ================================================================================= unapplied-override guard
def test_unapplied_override_sets_causal_verified_false_with_a_note(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}, known_axes={"warm", "concise"}))
    rec = counterfactual.counterfactual(RUN, {"sarcasm": 0.8}, sub)
    assert rec is not None
    assert rec["causal_verified"] is False
    assert "override_note" in rec
    assert "sarcasm" in rec["override_note"]
    # nothing actually moved (the axis was never recognized) -- both arms come back identical, but for
    # the RIGHT, disclosed reason (mirrors receipts.py's internalized-mode card-ablation honesty guard).
    assert rec["baseline_reply"] == rec["counterfactual_reply"]
    assert rec["has_effect"] is False


def test_unapplied_override_names_only_the_axis_that_actually_failed(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}, known_axes={"warm"}))
    rec = counterfactual.counterfactual(RUN, {"warm": 0.9, "sarcasm": 0.8}, sub)
    assert rec["causal_verified"] is False
    assert "sarcasm" in rec["override_note"]
    assert "warm" not in rec["override_note"]
    assert rec["has_effect"] is True         # warm still took effect even though sarcasm silently didn't


def test_zero_value_override_is_never_flagged_as_unapplied(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.6}))
    rec = counterfactual.counterfactual(RUN, {"warm": 0.0}, sub)
    assert rec["causal_verified"] is True
    assert "override_note" not in rec


# ------------------------------------------------------------------------------------- never raises / bad input
def test_counterfactual_returns_none_on_bad_overrides(iso):
    sub = FakeSub()
    assert counterfactual.counterfactual(RUN, {}, sub) is None
    assert counterfactual.counterfactual(RUN, None, sub) is None
    assert counterfactual.counterfactual(RUN, "not-a-dict", sub) is None


def test_counterfactual_returns_none_on_empty_or_bad_run(iso):
    assert counterfactual.counterfactual(None, {"warm": 0.5}, FakeSub()) is None
    assert counterfactual.counterfactual({}, {"warm": 0.5}, FakeSub()) is None
    assert counterfactual.counterfactual("not a dict", {"warm": 0.5}, FakeSub()) is None


def test_counterfactual_returns_none_when_substrate_is_none(iso):
    assert counterfactual.counterfactual(RUN, {"warm": 0.5}, None) is None


# ===================================================================================== dose_sweep: the curve
def test_dose_sweep_returns_one_curve_point_per_value(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    out = counterfactual.dose_sweep(RUN, "warm", [0.0, 0.5, 1.0], sub)
    assert out["run_id"] == RUN["id"]
    assert out["dial"] == "warm"
    assert [pt["value"] for pt in out["curve"]] == [0.0, 0.5, 1.0]
    assert all("delta" in pt and "coherence" in pt and "causal_verified" in pt for pt in out["curve"])
    assert out["derailment"] is False
    assert out["derailed_at"] == []


def test_dose_sweep_flags_derailment_where_coherence_craters(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    values = [0.5, 1.0, FakeSub.DEGENERATE_WARM]
    out = counterfactual.dose_sweep(RUN, "warm", values, sub)
    assert out["derailment"] is True
    assert out["derailed_at"] == [FakeSub.DEGENERATE_WARM]
    clean = [pt for pt in out["curve"] if pt["value"] != FakeSub.DEGENERATE_WARM]
    assert all(pt["coherence"]["degenerate"] is False for pt in clean)
    bad = next(pt for pt in out["curve"] if pt["value"] == FakeSub.DEGENERATE_WARM)
    assert bad["coherence"]["degenerate"] is True
    assert bad["has_effect"] is True          # still a real, causally-verified change -- just a bad one
    assert bad["causal_verified"] is True


def test_dose_sweep_costs_exactly_two_generations_per_value(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    values = [0.0, 0.5, 1.0, 1.5]
    counterfactual.dose_sweep(RUN, "warm", values, sub)
    assert sub.calls == 2 * len(values)       # each point is a fully independent counterfactual() call


def test_dose_sweep_points_are_independent_yet_agree_on_the_unchanging_baseline(iso):
    sub = FakeSub(mem=FakeMem(1.0), steer=FakeSteer({}))
    out = counterfactual.dose_sweep(RUN, "warm", [0.0, 0.5], sub)
    baselines = {pt["baseline_reply"] for pt in out["curve"]}
    assert len(baselines) == 1                # same run + same untouched live dials -> same baseline every time


def test_dose_sweep_needs_a_dial_name(iso):
    sub = FakeSub()
    out = counterfactual.dose_sweep(RUN, "", [0.0, 1.0], sub)
    assert out["curve"] == []
    assert "no receipt" in out["note"].lower()
    assert sub.calls == 0


def test_dose_sweep_never_raises_on_garbage_input(iso):
    out = counterfactual.dose_sweep(None, "warm", [0.1], FakeSub())
    assert out["run_id"] is None
    assert out["curve"][0]["error"]
    out2 = counterfactual.dose_sweep(RUN, "warm", None, FakeSub())
    assert out2["curve"] == []
    out3 = counterfactual.dose_sweep("not a dict", "warm", [0.1], FakeSub())
    assert out3["run_id"] is None


def test_dose_sweep_degrades_when_substrate_is_none(iso):
    out = counterfactual.dose_sweep(RUN, "warm", [0.0, 0.5], None)
    assert all(pt.get("error") for pt in out["curve"])
    assert out["derailment"] is False
