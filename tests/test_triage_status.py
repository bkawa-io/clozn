"""Unit tests for clozn.triage.status: the controlled evidence-status enum and its derivation rule."""
from __future__ import annotations

import pytest

from clozn.triage import status


def test_states_matches_the_spec_vocabulary_exactly():
    assert status.STATES == {
        "observed", "matched", "mismatched", "eliminated", "reproduced", "correlated",
        "causally_supported", "inconclusive", "not_run", "unsupported",
    }


def test_root_cause_is_not_a_valid_status():
    assert "root_cause" not in status.STATES
    assert not status.is_valid_status("root_cause")


def test_matched_derives_to_eliminated():
    assert status.hypothesis_for("matched") == "eliminated"


def test_mismatched_derives_to_observed():
    assert status.hypothesis_for("mismatched") == "observed"


@pytest.mark.parametrize("raw", [
    "eliminated", "observed", "inconclusive", "not_run", "unsupported", "reproduced",
    "correlated", "causally_supported", "root_cause", "", None,
])
def test_hypothesis_for_rejects_anything_that_is_not_a_raw_comparison_result(raw):
    with pytest.raises(ValueError):
        status.hypothesis_for(raw)


def test_is_valid_status_accepts_every_member():
    for state in status.STATES:
        assert status.is_valid_status(state)
