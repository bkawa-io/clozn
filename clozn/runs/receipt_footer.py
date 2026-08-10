"""The in-band receipt footer and its contamination-safe stripping policy.

The footer is a bounded shoulder tap.  Its facts come from the same structured Turn Receipt signal
registry used by the everyday receipt route; it never embeds arbitrary source labels, prompt snippets, or
model output.  ``strip_footers`` remains deliberately assistant-only so a user or system message that
mentions or pastes a footer-like string is never rewritten.
"""
from __future__ import annotations

import re

from clozn.runs import confidence_spans, signals

MARK = "⟨clozn⟩"

# Match every footer shape this module emits: receipt-only, one/two controlled signal phrases, and any
# future controlled phrase on that one line.  The rule + exact marker are the ownership boundary.
_FOOTER_RE = re.compile(r"\n*---\n`" + re.escape(MARK) + r"`[^\n]*\s*$")


def _strip_text(s: str) -> str:
    return _FOOTER_RE.sub("", s).rstrip()


def strip_footers(messages: list) -> list:
    """Remove Clozn's own receipt footer from assistant messages, without mutating input messages."""
    out = []
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                message = dict(message)
                message["content"] = _strip_text(content)
            elif isinstance(content, list):
                message = dict(message)
                message["content"] = [
                    ({**part, "text": _strip_text(part["text"])}
                     if isinstance(part, dict) and isinstance(part.get("text"), str) else part)
                    for part in content
                ]
        out.append(message)
    return out


def summary(run: dict | None) -> dict:
    """Preserve the existing confidence-summary helper for callers outside the footer policy."""
    spans = confidence_spans.spans(run if isinstance(run, dict) else {})
    trace = run.get("trace") if isinstance(run, dict) else None
    trace = trace if isinstance(trace, dict) else {}
    tokens = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    confidences = [float(value) for value in (trace.get("confidence") or [])
                   if isinstance(value, (int, float))]
    mean = round(sum(confidences) / len(confidences), 2) if confidences else None
    n_shaky = sum(1 for span in spans if span.get("band") == "shaky")
    return {"n_tokens": len(tokens), "mean_conf": mean, "n_shaky": n_shaky,
            "line": confidence_spans.summarize(spans)}


def _footer_phrase(signal: dict) -> str | None:
    """Translate one structured signal through the code-owned phrase registry."""
    code = signal.get("code") if isinstance(signal, dict) else None
    if not isinstance(code, str):
        return None
    if code == "context_window_pressure":
        percentage = signal.get("percentage")
        if isinstance(percentage, int) and not isinstance(percentage, bool) and percentage >= 0:
            return f"prompt is {percentage}% of context window"
        return "context-window pressure"
    phrase = signals.FOOTER_PHRASES.get(code)
    return phrase if isinstance(phrase, str) else None


def footer(run: dict | None, link: str, *, mode: str = "exceptions", turn_receipt: dict | None = None) -> str:
    """Return the bounded footer for ``off``, ``exceptions``, or ``always`` mode.

    ``turn_receipt`` is accepted so a caller that already composed the read-side artifact can reuse it;
    otherwise this function performs the same pure projection locally.  The latter is still read-only and
    model/worker-free, and it happens only after a footer mode has opted into delivery.
    """
    if mode not in {"off", "exceptions", "always"} or not isinstance(run, dict):
        return ""
    if mode == "off":
        return ""
    try:
        if not isinstance(turn_receipt, dict):
            from clozn.runs.turn_receipt import build_turn_receipt
            receipt_run = dict(run)
            # Real journal records always have an id.  This private fallback keeps the pure helper
            # usable with small hard-signal fixtures without exposing a fabricated id to the user.
            receipt_run.setdefault("id", "footer")
            turn_receipt = build_turn_receipt(receipt_run)
        structured = turn_receipt.get("signals") if isinstance(turn_receipt, dict) else []
        attention = [
            item for item in (structured or [])
            if isinstance(item, dict) and item.get("level") == "attention"
        ]
        priority = {code: index for index, code in enumerate(signals.SIGNAL_PRIORITY)}
        attention.sort(key=lambda item: priority.get(item.get("code"), len(priority)))
        phrases = []
        for item in attention[:2]:
            phrase = _footer_phrase(item)
            if phrase and phrase not in phrases:
                phrases.append(phrase)
        if mode == "exceptions" and not phrases:
            return ""
        return f"\n\n---\n`{MARK}` {' · '.join(phrases + [f'receipt → {link}'])}"
    except Exception:
        # Footer delivery is additive.  An optional projection defect must never break the model reply.
        return ""


__all__ = ["MARK", "footer", "strip_footers", "summary"]
