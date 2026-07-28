"""The controlled evidence-status vocabulary for automatic regression triage.

notes/agent_roadmap/05-automatic-regression-triage.md requires "a controlled enum" and forbids
`root_cause` as a machine status -- a UI may summarize "likely cause" only when the rule engine
(clozn.triage.rules) permits it, never a step or observation directly.

THE ONE DISTINCTION THAT MATTERS
---------------------------------
`matched` / `mismatched` are RAW comparison results: did two recorded values agree? They carry no causal
claim on their own. `eliminated` / `observed` are the DERIVED per-hypothesis reading of that raw result:
a match ELIMINATES a dimension as a possible cause (it cannot explain a divergence it does not have); a
mismatch is merely OBSERVED -- a candidate cause, not a proven one. `hypothesis_for()` below is that
derivation, and it is deliberately mechanical (a lookup table, not a judgment call) so promoting a
mismatch beyond "observed" always requires a real intervention recorded as its own step, never a
re-reading of the same comparison.
"""
from __future__ import annotations

# All ten states named by the spec's "controlled enum" -- matches clozn.schemas.defs.clozn.triage.v1's
# EvidenceStatus enum verbatim. Any state used anywhere in this package must be a member of this set.
STATES = frozenset({
    "observed",
    "matched",
    "mismatched",
    "eliminated",
    "reproduced",
    "correlated",
    "causally_supported",
    "inconclusive",
    "not_run",
    "unsupported",
})

# Raw comparison outcome -> derived per-hypothesis verdict. Deliberately a two-entry table: these are the
# ONLY two raw outcomes a plain equality comparison can produce, and the mapping never needs a third case
# (a comparison that could not be attempted is `inconclusive`, produced directly by the step, never routed
# through this table).
_HYPOTHESIS_FOR_RAW = {"matched": "eliminated", "mismatched": "observed"}


def hypothesis_for(raw_result: str) -> str:
    """The derived per-hypothesis verdict for a raw `matched`/`mismatched` comparison.

    `matched` -> `eliminated` (the dimension cannot explain the divergence -- ruled out).
    `mismatched` -> `observed` (a real difference exists, but nothing has proven it caused anything yet).

    Raises ValueError for any other input: `inconclusive`, `not_run`, `unsupported`, `reproduced`,
    `correlated`, and `causally_supported` are not raw comparison outcomes -- they must be produced
    directly by the step or the rule engine that established them, never passed through this table.
    """
    try:
        return _HYPOTHESIS_FOR_RAW[raw_result]
    except KeyError:
        raise ValueError(
            "hypothesis_for() only accepts a raw comparison result ('matched' or 'mismatched'), "
            f"got {raw_result!r}"
        ) from None


def is_valid_status(status: str) -> bool:
    return status in STATES
