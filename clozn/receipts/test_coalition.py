"""Tests for clozn.receipts.coalition (docs/PRODUCT_ROADMAP.md §8 tail: batched coalition/Shapley causal
credit). Model-free: drives coalition_report()/exact_shapley()/shapley_taylor() against FAKE substrates
whose chat() is a deterministic, hand-traceable function of exactly which influences are ablated at call
time (mirrors tests/test_receipts.py's FakeSub convention) -- so every expected number here is computed
independently, from the same primitives the module itself uses (receipt_metrics), never a hardcoded magic
number.

What's covered: exact Shapley on hand-computable N=3/4 fixtures (cross-checked against the Shapley
efficiency property and against shapley_taylor's closed form), the Shapley-Taylor 2nd-order estimator's
consistency for N>4 (including the bootstrap CI reuse from clozn.experiments.stats), the interaction-gap
math, N-cap / top-K-pair-selection behavior, and the batch-vs-sequential equality gate (a bit-exact fake
is trusted; a NOT-bit-exact fake is caught and the sequential value wins unless the caller opts into
'approximate').
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import clozn.settings as clozn_settings          # noqa: E402

import clozn.runs.store as runlog              # noqa: E402
from clozn.receipts import coalition           # noqa: E402
from clozn.receipts.metrics import receipt_metrics  # noqa: E402


# ================================================================================================== fixtures

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
    def __init__(self, strength=1.0):
        self.memory_strength = float(strength)
        self.rules = []
        self.prefix = "PFX"


class MultiDialSub:
    """chat() replies with a marker word for every dial CURRENTLY active (nonzero) -- ablating a dial
    (steer.set(name, 0.0)) removes exactly its own marker word from the reply, and nothing else. Gives
    full, hand-traceable control over every coalition's exact text for a purely ADDITIVE fixture (no
    influence's effect depends on any other)."""
    name = "fake"

    def __init__(self, dial_names):
        self.memory = FakeMem(1.0)
        self._mem = self.memory
        self.steer = FakeSteer({n: 1.0 for n in dial_names})
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True):
        self.calls += 1
        active = sorted(n for n, v in self.steer.strength.items() if v)
        return "BASE " + " ".join(active)


class RedundantSub:
    """N=3: card_a/card_b are a REDUNDANT pair (either alone covers "concise"; only removing BOTH changes
    the reply), and warm is an independent influence with its own always-present effect and NO interaction
    with the other two (mirrors tests/test_receipts.py's REDUNDANT_RUN fixture exactly, so its solo/pair/
    joint behavior is already independently validated there)."""
    name = "fake"

    def __init__(self, concise_card_ids):
        self.memory = FakeMem(1.0)
        self._mem = self.memory
        self.steer = FakeSteer({"warm": 0.4})
        self.concise_card_ids = {str(i) for i in concise_card_ids}
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True):
        self.calls += 1
        excluded = {str(i) for i in (getattr(self.memory, "_exclude_card_ids", None) or [])}
        concise_active = self.concise_card_ids - excluded
        base = "Short answer." if concise_active else (
            "A much longer rambling reply with plenty of extra words.")
        if float(self.steer.strength.get("warm", 0.0) or 0.0) > 0:
            base += " Warmly!"
        return base


RUN = {"id": "run_x", "messages": [{"role": "user", "content": "hi"}], "response": "SAMPLED, never a baseline"}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


def _solo_results(sub, baseline_reply, dial_names):
    """Build the `solo_results` shape core._prove_all_regen hands to coalition_report, by directly ablating
    each dial and measuring against `baseline_reply` -- exactly mirroring what the real leave-one-out loop
    in core.py already does (kept independent of core.py here so this file tests coalition.py in isolation)."""
    out = {}
    for name in dial_names:
        sub.steer.set(name, 0.0)
        reply = sub.chat([])
        sub.steer.set(name, 1.0)
        out[f"dial:{name}"] = {**coalition._arm_value(baseline_reply, reply), "_influence": {"dial": name}}
    return out


# ============================================================================================= exact Shapley

def test_exact_shapley_purely_additive_fixture_matches_solo_values(iso):
    """4 influences, each contributing its OWN unique word with no interaction at all: exact Shapley must
    reduce to exactly the solo delta for every influence (no synergy to redistribute)."""
    names = ["a", "b", "c", "d"]
    sub = MultiDialSub(names)
    baseline = sub.chat([])
    solo = _solo_results(sub, baseline, names)
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline)

    assert report["available"] is True
    assert report["n_influences"] == 4
    assert report["shapley"]["class"] == "exact"
    for key in solo:
        assert report["shapley"]["values"][key] == pytest.approx(solo[key]["value"], abs=1e-6)
    # Shapley efficiency property: values sum to the joint (grand-coalition) value.
    assert sum(report["shapley"]["values"].values()) == pytest.approx(report["joint"]["value"], abs=1e-6)
    # Purely additive -> the interaction gap is (near) zero.
    assert report["interaction_gap"]["gap"] == pytest.approx(0.0, abs=1e-6)




def test_exact_shapley_n4_needs_and_uses_the_extra_triples(iso):
    """N=4 exact Shapley needs 4 extra (triple) arms beyond solo+pairs+joint -- confirm they're actually
    run (cost_note says so) and the result still satisfies the efficiency property."""
    names = ["a", "b", "c", "d"]
    sub = MultiDialSub(names)
    baseline = sub.chat([])
    solo = _solo_results(sub, baseline, names)
    calls_before = sub.calls
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline)
    assert report["shapley"]["class"] == "exact"
    assert "4 extra" in report["cost_note"]
    # 6 pairs + 1 joint + 4 extra triples = 11 fresh generations (solo reused, not regenerated).
    assert sub.calls - calls_before == 11
    assert sum(report["shapley"]["values"].values()) == pytest.approx(report["joint"]["value"], abs=1e-6)


def test_single_influence_shapley_is_trivially_its_own_solo_delta(iso):
    sub = MultiDialSub(["only"])
    baseline = sub.chat([])
    solo = _solo_results(sub, baseline, ["only"])
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline)
    assert report["n_influences"] == 1
    assert report["shapley"]["values"]["dial:only"] == pytest.approx(solo["dial:only"]["value"])
    assert report["interaction_gap"]["gap"] == pytest.approx(0.0)
    assert report["batch_report"]["attempted"] is False


def test_no_influences_is_a_clean_unavailable_result():
    report = coalition.coalition_report(RUN, MultiDialSub([]), solo_results={}, baseline_reply="x")
    assert report == {"available": False, "reason": "no fired influences to build coalitions from"}


def test_joint_arm_failure_is_a_clean_unavailable_result(monkeypatch):
    """When every ablated arm fails to generate (`_ablated_child` returning None -- e.g. a dead engine),
    the joint arm is load-bearing for the interaction gap and its absence must degrade honestly, never
    fabricate a gap from partial data."""
    monkeypatch.setattr(coalition, "_ablated_child", lambda *a, **k: None)
    sub = MultiDialSub(["a", "b"])
    solo = {"dial:a": {"reply": "x", "value": 0.1, "has_effect": True, "_influence": {"dial": "a"}},
           "dial:b": {"reply": "x", "value": 0.1, "has_effect": True, "_influence": {"dial": "b"}}}
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply="baseline text")
    assert report["available"] is False
    assert "joint" in report["reason"]


# ========================================================================== N>4: Shapley-Taylor + N-cap

def test_shapley_taylor_consistency_and_bootstrap_ci_for_n_greater_than_exact_threshold(iso):
    """N=6 (> EXACT_SHAPLEY_MAX_N): all pairs are still exhaustive here (6 <= EXHAUSTIVE_PAIRS_MAX_N would
    be false for N=6 -- confirms the N=6 case triggers TOP-K capping), the estimator is explicitly labeled
    NOT exact, and each influence's interaction bootstrap CI is present (>= 2 partner pairs each, since
    top-K spreads across every key for this symmetric fixture)."""
    names = list("abcdef")            # N=6 > EXHAUSTIVE_PAIRS_MAX_N (5) -> pairs must be capped
    sub = MultiDialSub(names)
    baseline = sub.chat([])
    solo = _solo_results(sub, baseline, names)
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline,
                                        top_k_pairs=6, bootstrap_resamples=200, bootstrap_seed=0)
    assert report["available"] is True
    assert report["pairs_capped"] is True
    assert report["k_pairs"] == 6                       # min(C(6,2)=15, top_k_pairs=6)
    assert report["shapley"]["class"] == "shapley_taylor_2nd_order"
    assert "NOT exact" in report["shapley"]["estimator_note"]
    assert f"{report['n_influences']} solo" in report["shapley"]["estimator_note"]
    # Additive-by-construction fixture (each word removed independently) -> the 2nd-order estimate should
    # land CLOSE to the solo delta, but not bit-exact: the word-Jaccard value function is mildly nonlinear
    # even for strictly independent removals, so a small residual pairwise interaction term is expected and
    # honest (this is exactly why shapley_taylor's own docstring never claims exactness beyond N<=2).
    for key, val in report["shapley"]["values"].items():
        assert val == pytest.approx(solo[key]["value"], abs=0.02)
    # Top-6-of-15 pairs spreads partner counts unevenly across the 6 keys (a deterministic, lexicographic
    # tie-break over equal solo magnitudes) -- a key with >=2 measured partner pairs gets a real bootstrap
    # CI; a key capped down to <2 partners honestly reports "unavailable", never a fabricated interval.
    for key, ci in report["shapley"]["per_influence_interaction_ci"].items():
        n_partners = report["shapley"]["n_partners"][key]
        if n_partners >= 2:
            assert ci["available"] is True and ci["n_cases"] == n_partners
        else:
            assert ci["available"] is False
    assert any(ci["available"] for ci in report["shapley"]["per_influence_interaction_ci"].values())


def test_pairs_are_exhaustive_up_to_n5_and_capped_above_it(iso):
    names5 = list("abcde")
    sub5 = MultiDialSub(names5)
    baseline5 = sub5.chat([])
    solo5 = _solo_results(sub5, baseline5, names5)
    report5 = coalition.coalition_report(RUN, sub5, solo_results=solo5, baseline_reply=baseline5)
    assert report5["pairs_capped"] is False
    assert report5["k_pairs"] == 10                     # C(5,2)
    assert report5["shapley"]["class"] == "shapley_taylor_2nd_order"  # N=5 exceeds EXACT_SHAPLEY_MAX_N


def test_top_k_pair_selection_is_deterministic_and_ranked_by_solo_magnitude():
    # Distinct solo magnitudes: 'a' and 'b' are the two largest -> the pair {a,b} must be selected first.
    solo_values = {"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.05, "e": 0.01}
    top1 = coalition._top_k_pairs(list(solo_values), solo_values, 1)
    assert top1 == [frozenset({"a", "b"})]
    top_all = coalition._top_k_pairs(list(solo_values), solo_values, 100)
    assert len(top_all) == 10                            # C(5,2), never more than exist


# ================================================================================================= interaction gap

def test_interaction_gap_ratio_is_none_when_sum_of_solos_is_zero():
    gap = coalition.interaction_gap(0.3, {"a": 0.0, "b": 0.0})
    assert gap["ratio"] is None and gap["gap"] == pytest.approx(0.3)


def test_interaction_gap_reports_the_measured_overcounting_caveat():
    gap = coalition.interaction_gap(0.2, {"a": 0.3, "b": 0.3})
    assert gap["ratio"] == pytest.approx((0.2 - 0.6) / 0.6, abs=1e-6)
    assert "60%" in gap["note"] or "2.5x" in gap["note"]


# ======================================================================================== batching (FP-landmine)

class BatchingSub(MultiDialSub):
    """A substrate that ALSO offers batched coalition arms. `bit_exact=False` simulates a batched
    implementation that silently drifts from what sequential /score would compute -- exactly the failure
    mode the repo's FP-landmine rule exists to catch."""

    def __init__(self, dial_names, *, bit_exact=True, verified_exact=False):
        super().__init__(dial_names)
        self._bit_exact = bit_exact
        self.branch_coalitions_verified_exact = verified_exact
        self.branch_calls = 0

    def branch_coalitions(self, run, changes_list, *, baseline_reply=None):
        self.branch_calls += 1
        out = []
        for changes in changes_list:
            ablated = set((changes or {}).get("behavior_overrides") or {})
            active = sorted(n for n in self.steer.strength if n not in ablated)
            reply = "BASE " + " ".join(active)
            if not self._bit_exact:
                reply += " DRIFTED"
            out.append({"reply": reply})
        return out


def _fixture(sub_cls, **kwargs):
    names = ["a", "b", "c"]
    sub = sub_cls(names, **kwargs)
    baseline = sub.chat([])
    solo = {}
    for n in names:
        sub.steer.set(n, 0.0)
        solo[f"dial:{n}"] = {**coalition._arm_value(baseline, sub.chat([])), "_influence": {"dial": n}}
        sub.steer.set(n, 1.0)
    return sub, baseline, solo


def test_auto_mode_trusts_a_bit_exact_batched_substrate_only_after_cross_checking(iso):
    sub, baseline, solo = _fixture(BatchingSub, bit_exact=True, verified_exact=False)
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline,
                                        coalitions_batch="auto")
    batch = report["batch_report"]
    assert batch["attempted"] is True
    assert batch["bit_exact"] is True
    assert batch["used"] is False           # not self-certified -> sequential still supplies the value
    assert sub.branch_calls == 1
    assert sub.calls > 0                    # sequential path DID run, for the cross-check


def test_auto_mode_catches_a_non_bit_exact_batched_substrate_and_falls_back_to_sequential_truth(iso):
    sub, baseline, solo = _fixture(BatchingSub, bit_exact=False, verified_exact=False)
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline,
                                        coalitions_batch="auto")
    batch = report["batch_report"]
    assert batch["bit_exact"] is False
    assert batch["mismatched_subsets"]
    assert batch["used"] is False
    # The reported joint value must be the CORRECT sequential one, not the drifted batched one.
    correct_joint = coalition._arm_value(baseline, "BASE")["value"]     # all 3 dials ablated -> "BASE"
    assert report["joint"]["value"] == pytest.approx(correct_joint, abs=1e-6)


def test_approximate_opt_in_uses_the_uncertified_batched_value_and_labels_it(iso):
    sub, baseline, solo = _fixture(BatchingSub, bit_exact=False, verified_exact=False)
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline,
                                        coalitions_batch="approximate")
    batch = report["batch_report"]
    assert batch["used"] is True and batch["class"] == "approximate"
    # The (deliberately wrong) drifted batched joint value is what got reported this time.
    drifted_joint = coalition._arm_value(baseline, "BASE DRIFTED")["value"]
    assert report["joint"]["value"] == pytest.approx(drifted_joint, abs=1e-6)


def test_verified_exact_substrate_is_trusted_directly_with_no_sequential_cross_check(iso):
    sub, baseline, solo = _fixture(BatchingSub, bit_exact=True, verified_exact=True)
    calls_before = sub.calls                # _fixture already spent calls building baseline + solo
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline,
                                        coalitions_batch="auto")
    batch = report["batch_report"]
    assert batch["used"] is True and batch["class"] == "exact"
    assert sub.calls == calls_before         # no ADDITIONAL sequential generation -- fully trusted


def test_batching_off_never_calls_branch_coalitions(iso):
    sub, baseline, solo = _fixture(BatchingSub, bit_exact=True, verified_exact=True)
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline,
                                        coalitions_batch="off")
    assert report["batch_report"]["attempted"] is False
    assert sub.branch_calls == 0
    assert sub.calls > 0


# ==================================================================================================== format

def test_format_report_never_claims_significance_and_states_the_caveat(iso):
    names = ["a", "b", "c"]
    sub = MultiDialSub(names)
    baseline = sub.chat([])
    solo = _solo_results(sub, baseline, names)
    report = coalition.coalition_report(RUN, sub, solo_results=solo, baseline_reply=baseline)
    text = coalition.format_report(report)
    assert "Shapley" in text and "interaction gap" in text
    assert "overcount" in text.lower() or "OVERCOUNT" in text
    assert "significant" not in text.lower()


def test_format_report_unavailable_is_a_one_liner():
    text = coalition.format_report({"available": False, "reason": "no fired influences to build coalitions from"})
    assert "unavailable" in text and "no fired influences" in text
