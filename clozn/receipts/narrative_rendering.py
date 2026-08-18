"""Constrained and unconstrained narration calls."""
from __future__ import annotations

import re

from .fact_support import _as_dict, _as_list, _citable_facts


_CONSTRAINED_SYSTEM = (
    "You are explaining, after the fact, why a reply came out the way it did. You may rely ONLY on the "
    "measured facts listed below -- do not invent, assume, or recall anything else about the exchange "
    "(you have not been shown the question or the reply itself, on purpose). Every sentence you write must "
    "be grounded in one of these facts; write that fact's id in square brackets immediately after using it, "
    "for example [hesitation:2] or [concept:7]. If a section below says nothing applied, do not claim it did. "
    "If there are no facts at all, say plainly that no measured influence is on record for this reply -- "
    "that is a complete and correct answer, not a failure."
)

_NO_FACTS_LINE = "(no measured facts are on record for this reply)"


def _facts_lines(facts: list[dict]) -> str:
    if not facts:
        return _NO_FACTS_LINE
    return "\n".join(f"- [{f['id']}] ({f['category']}) {f['text']}" for f in facts)


def _constrained_messages(facts: list[dict]) -> list[dict]:
    user = ("Measured facts for this reply:\n" + _facts_lines(facts) +
            "\n\nUsing ONLY the facts above, explain briefly why the reply came out the way it did.")
    return [{"role": "system", "content": _CONSTRAINED_SYSTEM}, {"role": "user", "content": user}]


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def _extract_citations(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    return [m.strip() for m in _CITATION_RE.findall(text) if m.strip()]


def _safe_chat(sub, messages: list[dict]) -> str:
    try:
        chat = getattr(sub, "chat", None)
        if not callable(chat):
            return ""
        reply = chat(messages, max_new=256, sample=False)
        return reply if isinstance(reply, str) else str(reply)
    except Exception:
        return ""


def constrained_narration(explanation: dict, sub) -> dict:
    """Receipt-constrained narration that sees only measured facts."""
    facts = _citable_facts(explanation)
    valid_ids = {f["id"] for f in facts}
    text = _safe_chat(sub, _constrained_messages(facts))

    receipt_ids: list[str] = []
    seen: set[str] = set()
    for cid in _extract_citations(text):
        if cid in valid_ids and cid not in seen:
            seen.add(cid)
            receipt_ids.append(cid)
    return {"narration": text, "receipt_ids": receipt_ids}


_UNCONSTRAINED_QUESTION = "Why did you answer that way?"

_UNCONSTRAINED_NOTE = (
    "This is the model's UNCONSTRAINED, receipt-free guess at its own reasoning -- the confabulation "
    "sample research/FINDINGS.md's law #1 predicts will often be a fluent fabrication. It is CONTEXT FOR "
    "THE DIFF ONLY. Per EXPLAIN_THIS_ANSWER_SPEC.md's trap warning, this text must never be shown to a "
    "user as 'the answer' -- only confabulation_diff's flagged output, downstream of this, may be surfaced."
)


def unconstrained_why(run: dict, sub) -> dict:
    """Receipt-free self-narration sample used only as diff input."""
    run = _as_dict(run)
    messages = list(_as_list(run.get("messages")))
    response = run.get("response")
    if response:
        messages.append({"role": "assistant", "content": response if isinstance(response, str) else str(response)})
    messages.append({"role": "user", "content": _UNCONSTRAINED_QUESTION})

    text = _safe_chat(sub, messages)
    return {
        "unconstrained_text_context_only": text,
        "do_not_surface_as_answer": True,
        "role": "confabulation_sample",
        "note": _UNCONSTRAINED_NOTE,
    }
