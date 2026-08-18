"""clozn/behavior/registry.py -- the versioned corrective-action registry (roadmap feature 08).

This is deliberately NOT a hand-maintained duplicate of clozn.replay.corrective.CORRECTION_PRESETS
-- the prompt_policy backend, already shipped and tested (4 of the spec's 6 default actions existed
before this feature; the other 2 were added alongside this module).

build_registry() assembles it into ONE reviewable catalog, matching the spec's action-registry
shape (id, label, description, backends in preference order, qualification, safe bounds, known
conflicts). An action id here IS the existing CORRECTION_PRESETS key (e.g. "less-verbose") --
never a parallel identifier a caller would have to translate.

HONESTY CONTRACT
-----------------
Every action's `backends` list enumerates outcomes for the vocabulary values this build can still
honestly offer: prompt_policy (always) and sampling_policy (declared `"type": "unsupported"` with
a `reason`, since no action recipe uses it yet -- see _sampling_policy_backend). The spec's fourth
value, control_vector (the tone-dial backend), was retired along with the rest of the
personalization layer and no longer appears here at all -- there is nothing left to honestly
report a `reason` about, so it is dropped rather than kept as a permanent stub.
"""
from __future__ import annotations

from clozn.replay.corrective import CORRECTION_PRESETS

SCHEMA = "clozn.action-registry.v1"
REGISTRY_VERSION = "1"

_LABELS = {
    "less-verbose": "More concise",
    "more-concrete": "More concrete",
    "use-context": "Use the provided sources",
    "ask-before-guessing": "Ask when unsure",
    "preserve-formatting": "Preserve my formatting",
    "stop-repeating": "Stop repeating yourself",
}

# Durable session/profile scoping was retired -- a kept correction only ever selects itself as its
# own parent run's revision (clozn.behavior.corrective_flow.keep_result), never a standing policy
# that could shape a later, unrelated request. See docs/CAPABILITIES.md.
SCOPES = ("once",)


def _prompt_policy_backend(action_id: str) -> dict:
    """Every registry action has one -- CORRECTION_PRESETS backs all 6 default actions."""
    return {
        "type": "prompt_policy",
        "recipe_version": "1",
        "parameters": {"preset": action_id},
        "qualification": "generic",
        "qualification_id": "clozn.prompt-policy.generic.v1",
        "available": True,
    }


def _sampling_policy_backend() -> dict:
    """No action in this registry version has a sampling-policy recipe (spec non-goal: "a
    general-purpose steering-vector laboratory"). Declared unsupported explicitly rather than
    omitted, so the vocabulary stays visible even where nothing uses it yet."""
    reason = "no sampling-policy action is implemented in this registry version"
    return {
        "type": "unsupported",
        "requested_type": "sampling_policy",
        "available": False,
        "reason": reason,
        "unavailability_reason": reason,
    }


def _action_entry(action_id: str) -> dict:
    return {
        "id": action_id,
        "label": _LABELS[action_id],
        "description": CORRECTION_PRESETS[action_id],
        "conflicts": [],
        "scopes": list(SCOPES),
        "eligibility": {
            "eligible": True,
            "note": "run-specific scope eligibility is returned by /runs/{id}/corrective-actions",
        },
        "evaluation_metrics": ["word_count", "repetition", "formatting"],
        "backends": [
            _prompt_policy_backend(action_id),
            _sampling_policy_backend(),
        ],
    }


def build_registry() -> dict:
    """The full clozn.action-registry.v1 document."""
    return {
        "schema_version": SCHEMA,
        "version": REGISTRY_VERSION,
        "actions": [_action_entry(action_id) for action_id in CORRECTION_PRESETS],
    }


def action_ids() -> list[str]:
    """Every action id currently in the registry, in vocabulary order -- convenience for callers
    (routes, CLI) that want the id list without building the full document."""
    return list(CORRECTION_PRESETS)
