"""The one effective-prompt preparation seam for context experiments."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from clozn.receipts.rederive import with_arm_conditions
from clozn.replay.span_bridge import resolve_context_receipt_source_set
from clozn.runs.context_units import protected_message_indices

from .interventions import DeleteSource
from .selections import ContextSelection


class EffectivePromptUnavailable(ValueError):
    """The exact prompt sent to the worker cannot be reconstructed."""

    def __init__(self, message: str, *, reason: str = "effective_prompt_unavailable"):
        super().__init__(message)
        self.reason = reason


def inject_block(messages: Sequence[Mapping[str, Any]], block: str | None) -> list[dict[str, Any]]:
    """Apply the same system-block injection used by the product worker seam."""
    copied = [dict(message) for message in messages]
    if not block:
        return copied
    for message in copied:
        if message.get("role") == "system":
            message["content"] = (str(message.get("content") or "") + "\n\n" + block).strip()
            return copied
    return [{"role": "system", "content": block}] + copied


def _messages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EffectivePromptUnavailable("effective prompt has no message list", reason="messages_unavailable")
    messages = [dict(item) for item in value if isinstance(item, Mapping)]
    if len(messages) != len(value):
        raise EffectivePromptUnavailable("effective prompt contains malformed messages", reason="messages_malformed")
    return messages


@dataclass(frozen=True)
class EffectivePrompt:
    """Detached worker input, retaining the block separately when required."""

    messages: tuple[dict[str, Any], ...]
    block: str | None
    basis: str
    basis_digest: str | None = None
    intervened_context_digest: str | None = None
    removed_source_ids: tuple[str, ...] = ()

    def worker_messages(self) -> list[dict[str, Any]]:
        return deepcopy([dict(message) for message in self.messages])

    def rendered_messages(self) -> list[dict[str, Any]]:
        return inject_block(self.worker_messages(), self.block)


def resolve_effective_prompt(run: Mapping[str, Any], intervention: DeleteSource | None = None) -> EffectivePrompt:
    """Resolve the exact message/block pair used by exact worker execution."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise EffectivePromptUnavailable("a recorded run with a non-empty id is required", reason="run_unavailable")
    conditions = with_arm_conditions(dict(run))
    if intervention is None:
        messages = _messages(conditions.get("messages"))
        return EffectivePrompt(
            messages=tuple(messages), block=conditions.get("block"),
            basis=str(conditions.get("block_source") or "none"),
        )
    if not isinstance(intervention, DeleteSource):
        raise EffectivePromptUnavailable("effective context preparation supports DeleteSource only",
                                         reason="intervention_unavailable")
    try:
        resolved = resolve_context_receipt_source_set(dict(run), list(intervention.source_ids))
    except Exception as exc:
        raise EffectivePromptUnavailable(
            f"canonical Context Receipt source resolution failed: {exc}",
            reason="intervention_unavailable",
        ) from exc
    ranges = resolved.get("exact_removed_ranges") or []
    protected = protected_message_indices(run.get("messages"))
    if any(isinstance(item, Mapping) and item.get("message_index") in protected for item in ranges):
        raise EffectivePromptUnavailable(
            "the current request and following message suffix are protected from source deletion",
            reason="protected_source",
        )
    basis = str(resolved.get("basis") or "")
    block = conditions.get("block") if basis != "assembled_messages" else None
    return EffectivePrompt(
        messages=tuple(_messages(resolved.get("messages"))),
        block=block,
        basis=basis,
        basis_digest=resolved.get("basis_digest") if isinstance(resolved.get("basis_digest"), str) else None,
        intervened_context_digest=(resolved.get("intervened_context_digest")
                                   if isinstance(resolved.get("intervened_context_digest"), str) else None),
        removed_source_ids=tuple(intervention.source_ids),
    )


def render_effective_prompt_for_retained(
    run: Mapping[str, Any], universe_ids: Sequence[str], retained_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Render a retained-source candidate through the canonical worker seam."""
    universe = tuple(universe_ids)
    retained = set(retained_ids)
    if any(source_id not in universe for source_id in retained):
        raise EffectivePromptUnavailable(
            "retained source IDs are outside the planned universe", reason="source_outside_universe"
        )
    removed = tuple(source_id for source_id in universe if source_id not in retained)
    intervention = DeleteSource(ContextSelection(removed)) if removed else None
    return resolve_effective_prompt(run, intervention).rendered_messages()


__all__ = [
    "EffectivePrompt", "EffectivePromptUnavailable", "inject_block",
    "render_effective_prompt_for_retained", "resolve_effective_prompt",
]
