"""History-safe handling for model-emitted ``<think>`` blocks.

Reasoning models commonly put private scratch text between literal ``<think>`` tags.  That text is
useful evidence, but it is not the assistant answer: returning it as ordinary content lets clients echo
it into the next request and lets tool parsers mistake speculative calls for real ones.  This module is
the single, pure policy used by API streams, the CLI, replay, and the run journal.

Only assistant/model output is sanitized.  User and system text is deliberately untouched, so asking a
model to explain the tag syntax still reaches the model verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


OPEN_TAG = "<think>"
CLOSE_TAG = "</think>"
REASONING_SCHEMA = "clozn.reasoning_trace.v1"


@dataclass(frozen=True)
class ThinkResult:
    public_text: str
    blocks: tuple[dict, ...]
    implicit_open: bool = False
    orphan_close_count: int = 0

    @property
    def stripped(self) -> bool:
        return bool(self.blocks or self.orphan_close_count)

    def journal(self, *, reasoning_steps: list[dict] | None = None,
                trace_alignment: str | None = None) -> dict:
        """Return the additive journal block, or ``{}`` when no think syntax was present."""
        if not self.stripped:
            return {}
        out = {
            "schema": REASONING_SCHEMA,
            "source": "model_think_tags",
            "stripped_from_response": True,
            "implicit_open": bool(self.implicit_open),
            "blocks": [dict(block) for block in self.blocks],
        }
        if self.orphan_close_count:
            out["orphan_close_count"] = int(self.orphan_close_count)
        if reasoning_steps is not None:
            out["trace_step_count"] = len(reasoning_steps)
        if trace_alignment:
            out["trace_alignment"] = str(trace_alignment)
        return out


def prompt_opens_think(final_prompt) -> bool:
    """Whether a rendered chat template leaves generation inside an open think block.

    Qwen reasoning templates can end the generation scaffold at ``<think>\n``; the generated text then
    contains reasoning followed only by ``</think>``.  Knowing this before the first token is what lets
    streaming clients remain truly streaming without leaking the leading reasoning.
    """
    if not isinstance(final_prompt, str):
        return False
    text = final_prompt.rstrip().lower()
    opened = text.rfind(OPEN_TAG)
    closed = text.rfind(CLOSE_TAG)
    return opened > closed


def _tag_suffix_len(text: str, candidates: Iterable[str]) -> int:
    """Length of the longest suffix that may become a structural tag on the next stream chunk."""
    lower = text.lower()
    best = 0
    for tag in candidates:
        upto = min(len(tag) - 1, len(lower))
        for n in range(upto, 0, -1):
            if lower.endswith(tag[:n]):
                best = max(best, n)
                break
    return best


class ThinkTagStream:
    """Incremental think-tag filter.

    ``feed`` returns only answer text safe to send to the client.  Tag fragments may be split across any
    number of transport chunks.  ``finish`` must be called once to flush a harmless partial tag or record
    an unclosed reasoning block; it returns ``(final_public_chunk, ThinkResult)``.
    """

    def __init__(self, *, implicit_open: bool = False):
        self._implicit_open = bool(implicit_open)
        self._in_think = bool(implicit_open)
        self._depth = 1 if implicit_open else 0
        self._implicit_redundant_open = bool(implicit_open)
        self._pending = ""
        self._reasoning: list[str] = []
        self._blocks: list[dict] = []
        self._drained_blocks = 0
        self._public: list[str] = []
        self._orphan_closes = 0
        self._finished = False

    def _emit(self, text: str) -> str:
        if text:
            self._public.append(text)
        return text

    def _close_block(self, *, closed: bool) -> None:
        text = "".join(self._reasoning)
        self._blocks.append({"text": text, "closed": bool(closed)})
        self._reasoning = []

    def feed(self, chunk) -> str:
        if self._finished:
            raise RuntimeError("ThinkTagStream.feed() called after finish()")
        self._pending += str(chunk or "")
        emitted: list[str] = []

        while self._pending:
            lower = self._pending.lower()
            if self._in_think:
                open_at = lower.find(OPEN_TAG)
                close_at = lower.find(CLOSE_TAG)
                candidates = [(open_at, "open"), (close_at, "close")]
                candidates = [(at, kind) for at, kind in candidates if at >= 0]
                if candidates:
                    at, kind = min(candidates, key=lambda item: item[0])
                    if at:
                        self._reasoning.append(self._pending[:at])
                    tag = OPEN_TAG if kind == "open" else CLOSE_TAG
                    self._pending = self._pending[at + len(tag):]
                    if kind == "open":
                        # Some prompt-prefilled templates still make the model echo the opener.  A first
                        # redundant opener before substantive reasoning is the same block, not nesting.
                        only_ws = not "".join(self._reasoning).strip()
                        if self._implicit_redundant_open and only_ws and self._depth == 1:
                            self._implicit_redundant_open = False
                        else:
                            self._reasoning.append(tag)
                            self._depth += 1
                    else:
                        self._depth -= 1
                        if self._depth <= 0:
                            self._close_block(closed=True)
                            self._in_think = False
                            self._depth = 0
                        else:
                            self._reasoning.append(tag)
                    continue
                keep = _tag_suffix_len(self._pending, (OPEN_TAG, CLOSE_TAG))
                cut = len(self._pending) - keep
                if cut:
                    self._reasoning.append(self._pending[:cut])
                    self._pending = self._pending[cut:]
                break

            open_at = lower.find(OPEN_TAG)
            close_at = lower.find(CLOSE_TAG)
            candidates = [(open_at, "open"), (close_at, "close")]
            candidates = [(at, kind) for at, kind in candidates if at >= 0]
            if candidates:
                at, kind = min(candidates, key=lambda item: item[0])
                if at:
                    emitted.append(self._emit(self._pending[:at]))
                tag_len = len(OPEN_TAG if kind == "open" else CLOSE_TAG)
                self._pending = self._pending[at + tag_len:]
                if kind == "open":
                    self._in_think = True
                    self._depth = 1
                    self._implicit_redundant_open = False
                else:
                    # A close without an open is still structural debris and must not enter history.
                    # Leading implicit reasoning is handled explicitly via ``implicit_open``.
                    self._orphan_closes += 1
                continue

            keep = _tag_suffix_len(self._pending, (OPEN_TAG, CLOSE_TAG))
            cut = len(self._pending) - keep
            if cut:
                emitted.append(self._emit(self._pending[:cut]))
                self._pending = self._pending[cut:]
            break

        return "".join(emitted)

    def drain_reasoning(self) -> list[str]:
        """Return newly completed reasoning blocks once for streaming evidence capture."""
        fresh = [str(block.get("text") or "") for block in self._blocks[self._drained_blocks:]]
        self._drained_blocks = len(self._blocks)
        return fresh

    def finish(self) -> tuple[str, ThinkResult]:
        if self._finished:
            raise RuntimeError("ThinkTagStream.finish() called more than once")
        self._finished = True
        tail = ""
        if self._in_think:
            self._reasoning.append(self._pending)
            self._pending = ""
            self._close_block(closed=False)
            self._in_think = False
        else:
            # An incomplete lookalike such as '<thi' is ordinary answer text, not a tag.
            tail = self._emit(self._pending)
            self._pending = ""
        return tail, ThinkResult(
            public_text="".join(self._public),
            blocks=tuple(dict(block) for block in self._blocks),
            implicit_open=self._implicit_open,
            orphan_close_count=self._orphan_closes,
        )


def sanitize_reply(text, *, implicit_open: bool = False, infer_implicit: bool = True) -> ThinkResult:
    """Sanitize a complete model reply.

    For old journal/client history that lacks the rendered prompt, a first unmatched ``</think>`` is
    treated as the close of a template-opened block.  This conservative inference prevents legacy Qwen
    scratch text from being fed back into generation.  Live streams use :func:`prompt_opens_think`
    instead, because inference after bytes have already been delivered would be too late.
    """
    raw = str(text or "")
    lower = raw.lower()
    if infer_implicit and not implicit_open:
        first_open = lower.find(OPEN_TAG)
        first_close = lower.find(CLOSE_TAG)
        implicit_open = first_close >= 0 and (first_open < 0 or first_close < first_open)
    stream = ThinkTagStream(implicit_open=implicit_open)
    stream.feed(raw)
    _, result = stream.finish()
    return result


def sanitize_messages(messages) -> list:
    """Copy a message list with think blocks removed from assistant string content only."""
    out = []
    for message in messages or []:
        if not isinstance(message, dict):
            out.append(message)
            continue
        item = dict(message)
        if item.get("role") == "assistant":
            # A caller may echo a separate `thinking` field with the transcript. It is evidence, not
            # assistant content, and no chat template should receive it on the next turn.
            item.pop("thinking", None)
            if isinstance(item.get("content"), str):
                item["content"] = sanitize_reply(item["content"]).public_text
        out.append(item)
    return out


def sanitize_steps(steps, *, implicit_open: bool = False) -> tuple[list[dict], list[dict], ThinkResult]:
    """Split raw token steps into public-answer steps and hidden reasoning/tag steps.

    A transport piece may straddle a tag boundary.  In that case its metadata is retained on both cloned
    records, with ``piece``/``text`` narrowed to the relevant text.  Token ids still name the original
    model token; this is evidence, not a claim that the visible substring was separately tokenized.
    """
    stream = ThinkTagStream(implicit_open=implicit_open)
    public_steps: list[dict] = []
    reasoning_steps: list[dict] = []
    for raw_step in steps or []:
        if not isinstance(raw_step, dict):
            continue
        piece = str(raw_step.get("piece", raw_step.get("text", "")))
        public_piece = stream.feed(piece)
        if public_piece:
            item = dict(raw_step)
            item["piece"] = public_piece
            item["text"] = public_piece
            public_steps.append(item)
        if public_piece != piece:
            hidden = dict(raw_step)
            hidden["piece"] = piece
            hidden["text"] = piece
            reasoning_steps.append(hidden)
    tail, result = stream.finish()
    if not result.stripped:
        # Pending near-tag fragments are ordinary text at EOF.  Return the exact original steps so they
        # are neither misclassified as reasoning nor accidentally merged into a neighbor.
        return [dict(step) for step in (steps or []) if isinstance(step, dict)], [], result
    if tail:
        # ``tail`` belongs to the final step's pending partial-tag prefix.  It was not emitted by feed,
        # so attach it to that step now (or create a metadata-free step if there was none).
        if public_steps:
            item = public_steps[-1]
            item["piece"] = str(item.get("piece", "")) + tail
            item["text"] = item["piece"]
        else:
            public_steps.append({"index": 0, "piece": tail, "text": tail, "alternatives": []})
    return public_steps, reasoning_steps, result
