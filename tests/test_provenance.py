"""Fixture tests for clozn.analysis.provenance's pure logic (no engine, no network).

trace_provenance() itself needs a live clozn-server started with --no-flash-attn; everything
below is the model-free seam: the dependence score and the verdict rules, both of which encode
findings that cost real measurements to establish.
"""
import math

import pytest

from clozn.analysis.provenance import context_dependence, verdict, verdict_label


# ------------------------------------------------------------------ context_dependence

def test_dependence_is_one_when_the_answer_collapses():
    # P_base ~ 0.92, P_cut ~ 0.001 -> the context carried it
    d = context_dependence(math.log(0.92), math.log(0.001))
    assert d > 0.99


def test_dependence_is_zero_when_the_answer_survives():
    # cutting context access left the probability untouched -> parametric
    assert context_dependence(math.log(0.5), math.log(0.5)) == 0.0


def test_dependence_is_zero_when_the_cut_HELPS():
    """Knockout can raise the answer's probability (removing a competitor helps). That is not
    negative dependence -- it is no evidence of context-carrying, so it clamps to 0."""
    assert context_dependence(math.log(0.4), math.log(0.6)) == 0.0


def test_dependence_is_graded_in_between():
    d = context_dependence(math.log(0.8), math.log(0.4))   # halved
    assert d == pytest.approx(0.5, abs=1e-6)


def test_dependence_is_stable_for_tiny_probabilities():
    """Computed in log space: a 1e-300 probability must not underflow to a wrong answer."""
    d = context_dependence(-690.0, -700.0)                 # both ~1e-300, ratio e^-10
    assert d == pytest.approx(1.0 - math.exp(-10.0), abs=1e-9)


# ------------------------------------------------------------------------- verdict

def test_verdict_requires_separation_from_control():
    """A huge raw effect with no separation from a matched random control earns NO verdict --
    the same discipline the causal tracer's FAILED_CONTROLS encodes."""
    assert verdict(0.99, best_ratio=2.9) == "INCONCLUSIVE"
    assert verdict(0.01, best_ratio=1.0) == "INCONCLUSIVE"


def test_verdict_tiers():
    assert verdict(0.95, best_ratio=50.0) == "CONTEXT_CARRIED"
    assert verdict(0.80, best_ratio=10.0) == "CONTEXT_CARRIED"   # boundary is inclusive
    assert verdict(0.50, best_ratio=10.0) == "MIXED"
    assert verdict(0.30, best_ratio=10.0) == "MIXED"             # boundary is inclusive
    assert verdict(0.10, best_ratio=10.0) == "PARAMETRIC"


def test_verdict_labels_are_conservative_and_fail_closed():
    assert verdict_label("CONTEXT_CARRIED") == "high measured context dependence"
    assert verdict_label("MIXED") == "mixed measured context dependence"
    assert verdict_label("PARAMETRIC") == "low measured context dependence"
    assert verdict_label("INCONCLUSIVE") == "inconclusive context-dependence measurement"
    assert verdict_label("future_enum") == "unknown provenance measurement"


def test_verdict_matches_the_measured_dissociation():
    """The live 7B result this module exists to express (notes §5h): an in-context lookup is
    CONTEXT_CARRIED, while 'the modern capital of Japan is Tokyo' survives the cut (PARAMETRIC)
    because the model knows it from its weights."""
    in_context = context_dependence(math.log(0.90), math.log(0.005))
    assert verdict(in_context, best_ratio=1154.0) == "CONTEXT_CARRIED"
    parametric = context_dependence(math.log(0.55), math.log(0.388))
    assert verdict(parametric, best_ratio=18.0) == "PARAMETRIC"
