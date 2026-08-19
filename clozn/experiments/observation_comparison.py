"""Run-free comparison of one GeneratedObservation against its recorded parent.

An observation-first feature has exactly one Run: the immutable parent.  Its
counterfactual lives in a :class:`GeneratedObservation`, not in a second Run.
Materializing a throwaway child merely to reach ``diff_runs`` would hide child
creation behind an implementation detail and break the kernel invariant that a
Run appears only on an explicit materialization choice, so this module compares
the recorded answer suffix with the generated suffix directly.

The projection is deliberately narrower than the two-Run diff: it makes no
per-position distribution claims and never reports an "almost said" signal,
because a GeneratedObservation carries the counterfactual's own tokens, not a
second recorded distribution to compare against.  Everything it does report is
derivable from evidence already in hand.
"""
from __future__ import annotations

from collections.abc import Mapping
import difflib
from typing import Any

from .observations import GeneratedObservation

SCHEMA_VERSION = "clozn.observation-comparison.v1"
BASIS = "recorded_suffix_vs_generated_suffix"
SURFACE_SIMILARITY_LABEL = "surface similarity — wording, not meaning"

_UNAVAILABLE_REASONS = {
    "observation_not_completed": "the observation carries no completed generated evidence",
    "recorded_suffix_unavailable": "the parent has no recorded token pieces at this boundary",
}


def _recorded_pieces(run: Mapping[str, Any], position: int) -> list[str] | None:
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else None
    pieces = trace.get("tokens") if isinstance(trace, Mapping) else None
    if not isinstance(pieces, list) or not all(isinstance(piece, str) for piece in pieces):
        return None
    if position < 0 or position > len(pieces):
        return None
    return [str(piece) for piece in pieces[position:]]


def _generated_pieces(observation: GeneratedObservation) -> list[str] | None:
    steps = observation.generated_steps
    if not isinstance(steps, list) or not steps:
        return None
    pieces = [step.get("piece") for step in steps if isinstance(step, Mapping)]
    if len(pieces) != len(steps) or any(not isinstance(piece, str) for piece in pieces):
        return None
    if "".join(pieces) != observation.generated_suffix_text:
        # Token evidence that does not decode to the observed text cannot anchor a
        # divergence index.  Degrade to the text-only projection instead of guessing.
        return None
    return [str(piece) for piece in pieces]


def _unavailable(code: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "unavailable",
        "basis": BASIS,
        "reason_code": code,
        "reason": _UNAVAILABLE_REASONS.get(code, "the comparison is unavailable"),
    }


def observation_comparison(run: Mapping[str, Any], observation: GeneratedObservation) -> dict[str, Any]:
    """Compare a completed generated suffix with the parent's recorded suffix.

    Returns a typed ``unavailable`` projection rather than raising when either side
    is missing.  ``state`` is ``available`` only when both sides carry token pieces;
    a generated suffix with no usable per-token evidence degrades to
    ``token_evidence_unavailable`` with the surface comparison alone.
    """
    if not isinstance(observation, GeneratedObservation) or observation.status != "completed":
        return _unavailable("observation_not_completed")
    state_ref = observation.state_ref
    position = state_ref.position.index if state_ref is not None else None
    if not isinstance(position, int):
        return _unavailable("recorded_suffix_unavailable")
    recorded_pieces = _recorded_pieces(run, position)
    if recorded_pieces is None:
        return _unavailable("recorded_suffix_unavailable")

    recorded_text = "".join(recorded_pieces)
    generated_text = observation.generated_suffix_text
    generated_pieces = _generated_pieces(observation)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "available" if generated_pieces is not None else "token_evidence_unavailable",
        "basis": BASIS,
        "branch_point": {"answer_token_index": position},
        "recorded_suffix": {
            "token_count": len(recorded_pieces),
            "text_chars": len(recorded_text),
        },
        "generated_suffix": {
            "token_count": len(generated_pieces) if generated_pieces is not None else None,
            "text_chars": len(generated_text),
        },
        "identical_text": recorded_text == generated_text,
        "surface_similarity": {
            "value": round(difflib.SequenceMatcher(a=recorded_text, b=generated_text).ratio(), 4),
            "label": SURFACE_SIMILARITY_LABEL,
        },
    }
    if generated_pieces is None:
        result["reason_code"] = "generated_token_evidence_unavailable"
        return result

    common = 0
    for recorded, generated in zip(recorded_pieces, generated_pieces):
        if recorded != generated:
            break
        common += 1
    result["common_suffix_prefix_len"] = common
    if common == len(recorded_pieces) == len(generated_pieces):
        result["first_divergence"] = None
        return result
    if common < min(len(recorded_pieces), len(generated_pieces)):
        result["first_divergence"] = {
            "answer_token_index": position + common,
            "suffix_index": common,
            "kind": "token_mismatch",
            "recorded_piece": recorded_pieces[common],
            "generated_piece": generated_pieces[common],
        }
        return result
    result["first_divergence"] = {
        "answer_token_index": position + common,
        "suffix_index": common,
        "kind": "length_mismatch",
        "recorded_piece": recorded_pieces[common] if common < len(recorded_pieces) else None,
        "generated_piece": generated_pieces[common] if common < len(generated_pieces) else None,
    }
    return result


__all__ = ["BASIS", "SCHEMA_VERSION", "observation_comparison"]
