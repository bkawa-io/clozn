"""clozn/behavior/registry.py -- the versioned corrective-action registry (roadmap feature 08).

This is deliberately NOT a hand-maintained duplicate of the two vocabularies that already exist:

  * clozn.replay.corrective.CORRECTION_PRESETS -- the prompt_policy backend, already shipped and
    tested (4 of the spec's 6 default actions existed before this feature; the other 2 were added
    alongside this module).
  * clozn.behavior.steering.axes.AXES -- the control_vector backend's dial catalog, already gated by
    per-exact-model calibration (clozn.behavior.steering.engine_adapter.EngineSteer.ceiling_for).

build_registry() assembles both into ONE reviewable catalog, matching the spec's action-registry
shape (id, label, description, backends in preference order, qualification, safe bounds, known
conflicts). An action id here IS the existing CORRECTION_PRESETS key (e.g. "less-verbose") --
never a parallel identifier a caller would have to translate.

HONESTY CONTRACT
-----------------
Every action's `backends` list enumerates outcomes for all four vocabulary values the spec
defines: prompt_policy, control_vector, sampling_policy, unsupported. A backend this build cannot
honestly offer is listed with `"type": "unsupported"` and a `reason`, never silently dropped -- a
reader of the registry sees the full vocabulary evaluated, never has to guess what wasn't tried.

`qualified` on a control_vector backend is only ever present when a live `steer` (an
EngineSteer-shaped duck type: `.ceiling_for(name) -> (max, calibrated)`) was supplied to
build_registry(): calibration is per exact model sha256 (see engine_adapter.py's own docstring), so
with no loaded model there is nothing honest to report and the field is omitted, never guessed as
True or False (SEAMS.md rule 2: omit, never null-pad).
"""
from __future__ import annotations

from clozn.behavior.steering import axes as _axes
from clozn.replay.corrective import CORRECTION_PRESETS

SCHEMA = "clozn.action-registry.v1"
REGISTRY_VERSION = "1"

# action id (== the existing CORRECTION_PRESETS key) -> the matching control-vector dial name in
# clozn.behavior.steering.axes.AXES, for the 2 of 6 actions a calibrated dial can also express.
# The other 4 (use-context, ask-before-guessing, preserve-formatting, stop-repeating) have no
# matching axis and stay prompt_policy-only -- that absence is the "unsupported" control_vector
# backend below, not an oversight.
_DIAL_FOR_ACTION = {
    "less-verbose": "concise",
    "more-concrete": "concrete",
}

_LABELS = {
    "less-verbose": "More concise",
    "more-concrete": "More concrete",
    "use-context": "Use the provided sources",
    "ask-before-guessing": "Ask when unsure",
    "preserve-formatting": "Preserve my formatting",
    "stop-repeating": "Stop repeating yourself",
}


def _prompt_policy_backend(action_id: str) -> dict:
    """Every registry action has one -- CORRECTION_PRESETS backs all 6 default actions."""
    return {
        "type": "prompt_policy",
        "recipe_version": "1",
        "parameters": {"preset": action_id},
        "qualification": "generic",
    }


def _control_vector_backend(action_id: str, steer) -> dict:
    dial = _DIAL_FOR_ACTION.get(action_id)
    if dial is None:
        return {"type": "unsupported",
                "reason": f"no control-vector dial exists for {action_id!r}"}
    axis = _axes.AXES.get(dial) or {}
    declared_max = float(axis.get("max", 1.5))
    backend = {
        "type": "control_vector",
        "recipe_version": "1",
        "parameters": {"dial": dial},
        "qualification": "model_build_exact",
        "safe_bounds": {"declared_max": declared_max},
    }
    if steer is not None and hasattr(steer, "ceiling_for"):
        try:
            ceiling, calibrated = steer.ceiling_for(dial)
        except Exception:
            # Can't establish live qualification -- degrade to the static shape above rather than
            # fail the whole registry or guess at a bool (SEAMS.md rule 2: omit, never null-pad).
            return backend
        backend["qualified"] = bool(calibrated)
        backend["safe_bounds"]["enforced_ceiling"] = float(ceiling)
        if not calibrated:
            backend["reason"] = (
                f"no calibration for this exact model; {dial!r} is capped at the generic "
                f"uncalibrated ceiling, not its declared max"
            )
    return backend


def _sampling_policy_backend() -> dict:
    """No action in this registry version has a sampling-policy recipe (spec non-goal: "a
    general-purpose steering-vector laboratory"). Declared unsupported explicitly rather than
    omitted, so the vocabulary stays visible even where nothing uses it yet."""
    return {"type": "unsupported",
            "reason": "no sampling-policy action is implemented in this registry version"}


def _action_entry(action_id: str, steer) -> dict:
    return {
        "id": action_id,
        "label": _LABELS[action_id],
        "description": CORRECTION_PRESETS[action_id],
        "conflicts": [],
        "backends": [
            _prompt_policy_backend(action_id),
            _control_vector_backend(action_id, steer),
            _sampling_policy_backend(),
        ],
    }


def build_registry(steer=None) -> dict:
    """The full clozn.action-registry.v1 document.

    `steer` is an optional duck-typed control-vector engine (EngineSteer-shaped) for the CURRENTLY
    loaded model. When given, control_vector backends report a live `qualified` bool for the exact
    model in force; when omitted (no worker loaded, or a caller that only wants the static catalog),
    that field is left out entirely. Never raises: a `steer` whose `ceiling_for` misbehaves degrades
    that one backend entry to its static (unqualified) shape rather than failing the whole registry.
    """
    return {
        "schema_version": SCHEMA,
        "version": REGISTRY_VERSION,
        "actions": [_action_entry(action_id, steer) for action_id in CORRECTION_PRESETS],
    }


def action_ids() -> list[str]:
    """Every action id currently in the registry, in vocabulary order -- convenience for callers
    (routes, CLI) that want the id list without building the full document."""
    return list(CORRECTION_PRESETS)
