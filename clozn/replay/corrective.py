"""Prompt-first corrective retries over the existing replay engine.

This module deliberately owns no route, preference, or steering state.  A retry is
a matched pair of greedy child runs: one plain baseline and one whose only extra
input is a named system instruction.  Keeping both arms makes the comparison an
observed regeneration rather than a claim about the stored (possibly sampled)
reply.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Any

from clozn import receipts
from clozn.behavior.compare import compare_metrics
from .replay import replay as replay_run


_PRESET_TEXT = {
    "less-verbose": (
        "For this reply, answer concisely. Preserve necessary caveats and requested details; "
        "remove repetition, preamble, and nonessential explanation."
    ),
    "more-concrete": (
        "For this reply, use specific examples, named steps, and concrete details. "
        "Do not invent facts; mark unknowns."
    ),
    "use-context": (
        "Use the supplied conversation and context as the primary evidence. Ground the answer "
        "in relevant details already provided; if needed information is absent, say what is missing."
    ),
    "ask-before-guessing": (
        "If missing information would materially change the answer, ask one concise clarifying "
        "question before attempting an answer. Do not guess missing facts."
    ),
    "preserve-formatting": (
        "Preserve the formatting conventions already present in the conversation (headings, lists, "
        "code fences, tables). Do not add or remove structural formatting the user did not ask for."
    ),
    "stop-repeating": (
        "Do not repeat information, phrases, or caveats already stated earlier in this reply or "
        "conversation. Say each point once."
    ),
}

# Public, immutable vocabulary.  Arbitrary caller-provided instructions are not a
# corrective preset and cannot cross this system-message seam.
CORRECTION_PRESETS: Mapping[str, str] = MappingProxyType(_PRESET_TEXT)


def inject_correction(messages: Sequence[Mapping[str, Any]], preset: str) -> list[dict[str, Any]]:
    """Return copied messages with one bounded corrective system instruction.

    Caller messages and nested tool payloads are deep-copied, never edited or
    reordered.  The correction follows any leading system messages, so existing
    system context stays first while the correction remains a distinct, auditable
    message rather than being concatenated into caller text.
    """
    if preset not in CORRECTION_PRESETS:
        raise ValueError(f"unknown corrective preset {preset!r}")
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise ValueError("messages must be a sequence of message objects")
    if any(not isinstance(message, Mapping) for message in messages):
        raise ValueError("messages must contain only message objects")

    copied = [deepcopy(dict(message)) for message in messages]
    insert_at = 0
    while insert_at < len(copied) and copied[insert_at].get("role") == "system":
        insert_at += 1
    copied.insert(insert_at, {
        "role": "system",
        "content": "Clozn corrective retry: " + CORRECTION_PRESETS[preset],
    })
    return copied


def _original_budget(run: Mapping[str, Any]) -> int:
    limits = ((run.get("context_receipt") or {}).get("limits") or {})
    value = limits.get("requested_max_tokens")
    return int(value) if isinstance(value, int) and 0 < value <= 16384 else 256


def _instruction_survived(child: Mapping[str, Any], instruction: str) -> bool:
    assembled = child.get("assembled_messages") or []
    if any(instruction in str(message.get("content") or "")
           for message in assembled if isinstance(message, Mapping)):
        return True
    return instruction in str(child.get("final_prompt") or "")


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _execution_identity(run: Mapping[str, Any], preset: str, backend: str, parameters: dict,
                        qualification: str, qualified: bool | None, fallback: bool,
                        baseline_reply: str, corrected_reply: str) -> dict:
    """The spec's per-revision execution identity: parent run, action id + backend + exact
    parameters, and before/after hashes -- reproducible/explainable later without re-reading the
    full replies. `ext` is contributed through Seam 3 (clozn.runs.identity_providers.
    behavior_intervention) rather than assembled here by hand, so a future caller of
    clozn.runs.identity.runtime_identity() that supplies the same `behavior_intervention` fact gets
    an identical shape with no code shared beyond the provider file."""
    from clozn.runs import identity_ext
    fact: dict[str, Any] = {
        "action_id": preset, "backend": backend, "registry_version": "1", "parameters": parameters,
        "qualification": qualification, "fallback": fallback,
    }
    if qualified is not None:
        fact["qualified"] = qualified
    return {
        "parent_run_id": run.get("id"),
        "action_id": preset,
        "backend": backend,
        "before_hash": _sha256(baseline_reply),
        "after_hash": _sha256(corrected_reply),
        "ext": identity_ext.collect({"behavior_intervention": fact}),
    }


def _prompt_blocks(presets) -> list[str]:
    selected = list(dict.fromkeys(str(value) for value in (presets or [])
                                  if str(value) in CORRECTION_PRESETS))
    if not selected:
        return []
    return ["Clozn active corrective response policy:\n" + "\n".join(
        f"- {CORRECTION_PRESETS[value]}" for value in selected
    )]


def _structured_failure(
    run: Mapping[str, Any],
    preset: str,
    requested_backend: str,
    budget: int,
    baseline: dict[str, Any],
    corrected: dict[str, Any],
) -> dict[str, Any]:
    """Return a persistable partial outcome without changing the legacy ``None`` contract."""
    return {
        "preset": preset,
        "instruction": CORRECTION_PRESETS[preset],
        "stored_original_reply": str(run.get("response") or ""),
        "baseline_reply": str(baseline.get("reply") or ""),
        "corrected_reply": str(corrected.get("reply") or ""),
        "delta": {},
        "changed": False,
        "intervention_observed": False,
        "comparison_note": (
            "matched greedy comparison did not complete; the stored original is context only"
        ),
        "max_tokens": budget,
        "baseline_child_id": baseline.get("run_id"),
        "corrected_child_id": corrected.get("run_id"),
        "child_ids": {
            "baseline": baseline.get("run_id"),
            "corrected": corrected.get("run_id"),
        },
        "child_outcomes": {"baseline": baseline, "corrected": corrected},
        "requested_backend": requested_backend,
        "backend": None,
        "executed_backend": None,
        "backend_fallback": False,
        "outcome": {"status": "execution_error"},
    }


def retry_compare(run: Mapping[str, Any], preset: str, sub, *,
                  backend: str | None = None,
                  structured: bool = False) -> dict[str, Any] | None:
    """Generate mandatory matched greedy baseline/corrected replay children.

    This is a request-local counterfactual debugging tool: the baseline is a plain greedy replay of
    the run, and the corrected arm is the SAME replay plus one bounded system instruction. Neither arm
    persists anything beyond the two child runs themselves -- there is no session/profile scope and no
    standing policy left behind for a later, unrelated request to pick up (durable, auto-applied
    corrections were retired; see docs/CAPABILITIES.md).

    Returns ``None`` when either existing replay operation fails, matching replay's
    established failure contract.  Invalid inputs and preset names raise
    ``ValueError`` before generation.  No dial or memory setting is changed here;
    replay remains the sole owner of temporary substrate state and restoration.

    ``backend`` chooses the corrected arm's mechanism. ``None`` (default) or ``"prompt_policy"`` is
    the only supported value -- the system-instruction preset. A calibrated named-dial backend
    (``"control_vector"``) existed here before named-dial personalization was retired; the corrected
    arm is always the prompt-instruction rewrite now.
    """
    if not isinstance(run, Mapping) or not run.get("id"):
        raise ValueError("run must be a stored run with an id")
    messages = run.get("messages")
    # Validate before either generation. The injected copy is intentionally not put
    # back into ``run``: replay must journal only caller-delivered messages.
    inject_correction(messages, preset)
    if backend not in (None, "prompt_policy"):
        raise ValueError("backend must be None or 'prompt_policy'")
    budget = _original_budget(run)
    requested_backend = backend or "prompt_policy"

    baseline_changes = {
        "greedy": True,
        "corrective_retry": {"arm": "baseline", "preset": preset},
    }
    try:
        baseline = replay_run(
            dict(run),
            baseline_changes,
            sub,
            max_new=budget,
        )
    except Exception as exc:
        if not structured:
            raise
        return _structured_failure(
            run, preset, requested_backend, budget,
            {"status": "error", "error": {"code": "generation_error", "message": str(exc)}},
            {"status": "not_run"},
        )
    if baseline is None:
        if structured:
            return _structured_failure(
                run, preset, requested_backend, budget,
                {"status": "error", "error": {"code": "generation_failed",
                                               "message": "baseline replay returned no run"}},
                {"status": "not_run"},
            )
        return None
    baseline_outcome = {
        "status": "success",
        "run_id": baseline.get("id"),
        "reply": str(baseline.get("response") or ""),
    }
    if structured and not baseline_outcome["run_id"]:
        baseline_outcome = {
            **baseline_outcome,
            "status": "error",
            "error": {"code": "missing_child_run",
                      "message": "baseline replay did not persist a child run id"},
        }
        return _structured_failure(
            run, preset, requested_backend, budget,
            baseline_outcome, {"status": "not_run"},
        )

    instruction = CORRECTION_PRESETS[preset]
    chosen_backend = "prompt_policy"
    backend_fallback = False

    corrected_changes = {
        "greedy": True,
        "corrective_retry": {
            "arm": "corrected",
            "preset": preset,
            "method": "system_instruction",
            "instruction": instruction,
        },
    }
    try:
        corrected = replay_run(
            dict(run),
            corrected_changes,
            sub,
            prompt_instructions=_prompt_blocks([preset]),
            max_new=budget,
        )
    except Exception as exc:
        if not structured:
            raise
        return _structured_failure(
            run, preset, requested_backend, budget,
            baseline_outcome,
            {"status": "error", "error": {"code": "generation_error", "message": str(exc)}},
        )
    if corrected is None:
        if structured:
            return _structured_failure(
                run, preset, requested_backend, budget,
                baseline_outcome,
                {"status": "error", "error": {"code": "generation_failed",
                                               "message": "corrected replay returned no run"}},
            )
        return None

    baseline_reply = str(baseline.get("response") or "")
    corrected_reply = str(corrected.get("response") or "")
    baseline_id = baseline.get("id")
    corrected_id = corrected.get("id")
    corrected_outcome = {
        "status": "success",
        "run_id": corrected_id,
        "reply": corrected_reply,
    }
    if structured and not corrected_id:
        corrected_outcome = {
            **corrected_outcome,
            "status": "error",
            "error": {"code": "missing_child_run",
                      "message": "corrected replay did not persist a child run id"},
        }
        return _structured_failure(
            run, preset, requested_backend, budget,
            baseline_outcome, corrected_outcome,
        )
    try:
        from .coherence import _coherence
        coherence = _coherence(corrected_reply)
    except Exception:
        coherence = {"degenerate": False, "reasons": []}

    intervention_observed = _instruction_survived(corrected, instruction)
    backend_parameters = {"preset": preset}
    qualification, qualified = "generic", None
    execution_identity = _execution_identity(
        run, preset, chosen_backend, backend_parameters, qualification=qualification,
        qualified=qualified, fallback=backend_fallback,
        baseline_reply=baseline_reply, corrected_reply=corrected_reply,
    )
    return {
        "preset": preset,
        "instruction": instruction,
        "stored_original_reply": str(run.get("response") or ""),
        "baseline_reply": baseline_reply,
        "corrected_reply": corrected_reply,
        "delta": {**receipts.receipt_metrics(baseline_reply, corrected_reply),
                  **compare_metrics(baseline_reply, corrected_reply)},
        "changed": baseline_reply != corrected_reply,
        "coherence": coherence,
        "intervention_observed": intervention_observed,
        "comparison_note": ("matched greedy baseline and candidate under the current runtime policy; "
                            "the stored original is context only"),
        "max_tokens": budget,
        "baseline_child_id": baseline_id,
        "corrected_child_id": corrected_id,
        "child_ids": {"baseline": baseline_id, "corrected": corrected_id},
        "child_outcomes": {
            "baseline": baseline_outcome,
            "corrected": corrected_outcome,
        },
        "requested_backend": requested_backend,
        "backend": chosen_backend,
        "executed_backend": chosen_backend,
        "backend_fallback": backend_fallback,
        "execution_identity": execution_identity,
        "outcome": {"status": "succeeded"},
    }
