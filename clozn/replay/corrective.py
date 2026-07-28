"""Prompt-first corrective retries over the existing replay engine.

This module deliberately owns no route, preference, or steering state.  A retry is
a matched pair of greedy child runs: one plain baseline and one whose only extra
input is a named system instruction.  Keeping both arms makes the comparison an
observed regeneration rather than a claim about the stored (possibly sampled)
reply.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Any

from clozn import receipts
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


def _prompt_blocks(presets) -> list[str]:
    selected = list(dict.fromkeys(str(value) for value in (presets or [])
                                  if str(value) in CORRECTION_PRESETS))
    if not selected:
        return []
    return ["Clozn active corrective response policy:\n" + "\n".join(
        f"- {CORRECTION_PRESETS[value]}" for value in selected
    )]


def retry_compare(run: Mapping[str, Any], preset: str, sub, *, scope: str = "once",
                  active_presets=()) -> dict[str, Any] | None:
    """Generate mandatory matched greedy baseline/corrected replay children.

    Returns ``None`` when either existing replay operation fails, matching replay's
    established failure contract.  Invalid inputs and preset names raise
    ``ValueError`` before generation.  No dial or memory setting is changed here;
    replay remains the sole owner of temporary substrate state and restoration.
    """
    if not isinstance(run, Mapping) or not run.get("id"):
        raise ValueError("run must be a stored run with an id")
    messages = run.get("messages")
    # Validate before either generation. The injected copy is intentionally not put
    # back into ``run``: replay must journal only caller-delivered messages.
    inject_correction(messages, preset)
    if scope not in {"once", "session", "profile"}:
        raise ValueError("scope must be once, session, or profile")
    budget = _original_budget(run)
    current_presets = list(dict.fromkeys(
        str(value) for value in (active_presets or []) if str(value) in CORRECTION_PRESETS
    ))
    candidate_presets = list(dict.fromkeys(current_presets + [preset]))

    baseline_changes = {
        "greedy": True,
        "corrective_retry": {"arm": "baseline", "preset": preset},
    }
    baseline = replay_run(dict(run), baseline_changes, sub,
                          prompt_instructions=_prompt_blocks(current_presets), max_new=budget)
    if baseline is None:
        return None

    instruction = CORRECTION_PRESETS[preset]
    corrected_changes = {
        "greedy": True,
        "corrective_retry": {
            "arm": "corrected",
            "preset": preset,
            "method": "system_instruction",
            "instruction": instruction,
            "scope": scope,
        },
    }
    corrected = replay_run(dict(run), corrected_changes, sub,
                           prompt_instructions=_prompt_blocks(candidate_presets),
                           max_new=budget)
    if corrected is None:
        return None

    baseline_reply = str(baseline.get("response") or "")
    corrected_reply = str(corrected.get("response") or "")
    baseline_id = baseline.get("id")
    corrected_id = corrected.get("id")
    try:
        from .counterfactual import _coherence
        coherence = _coherence(corrected_reply)
    except Exception:
        coherence = {"degenerate": False, "reasons": []}
    return {
        "preset": preset,
        "scope": scope,
        "active_presets_before": current_presets,
        "active_presets_candidate": candidate_presets,
        "instruction": instruction,
        "stored_original_reply": str(run.get("response") or ""),
        "baseline_reply": baseline_reply,
        "corrected_reply": corrected_reply,
        "delta": receipts.receipt_metrics(baseline_reply, corrected_reply),
        "changed": baseline_reply != corrected_reply,
        "coherence": coherence,
        "intervention_observed": _instruction_survived(corrected, instruction),
        "comparison_note": ("matched greedy baseline and candidate under the current runtime policy; "
                            "the stored original is context only"),
        "max_tokens": budget,
        "baseline_child_id": baseline_id,
        "corrected_child_id": corrected_id,
        "child_ids": {"baseline": baseline_id, "corrected": corrected_id},
    }
