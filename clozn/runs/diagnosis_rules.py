"""clozn/runs/diagnosis_rules.py -- D1: a deterministic diagnostic RULE ENGINE over one run's already-
persisted evidence. No LLM is ever the source of truth here: every finding is either a direct read of a
structured field (finish_reason, a token count, a differing setting between two runs) or a deterministic
text-pattern match over recorded text (a repeated sentence, an omitted segment, a missing output format).
This is the foundation of the "Why?" epic; a later slice turns these findings into plain-language
narratives (D2) and a guided-repair UI (D5) -- this module owns evidence and structural facts only, never
a causal story about why the model behaved as it did.

RELATIONSHIP TO `clozn.run_diagnosis.v1` (`clozn/runs/diagnosis.py`) -- READ THIS BEFORE EXTENDING EITHER
--------------------------------------------------------------------------------------------------------
`clozn.run_diagnosis.v1` (unregistered informal schema; not validated against `clozn.schemas`) answers
exactly two questions about ONE request: "why was it slow" and "was the reply cut off", from timing/
journal fields, using a THREE-value status vocabulary (`observed` / `not_observed` / `unavailable`) on a
small, FIXED, hand-named set of findings (`total_wall_time`, `model_load`, `prefill`, `generation`,
`context_pressure`, `context_allocation`, `cpu_spill`, `output_cutoff`, `client_auxiliary_calls`).

`clozn.diagnosis-findings.v1` (THIS module) is a different artifact answering a different question: a
FIXED CATALOG of PROMPT/EXECUTION PROBLEM rules (conflicting instructions, omitted input, missing output
format, evidence a source never cleared its measurement floor, run-to-run setting drift, and more -- see
`RULE_REGISTRY`), each ALWAYS producing exactly one entry no matter the outcome, with a FIVE-value status
vocabulary (`finding` / `not_observed` / `unavailable` / `pending` / `suppressed` -- `pending` and
`suppressed` have no equivalent in `clozn.run_diagnosis.v1` at all: `pending` names evidence that was
never even ATTEMPTED for this run -- e.g. no influence map, no comparison run supplied -- as distinct from
`unavailable`, evidence that was attempted and could not be read; `suppressed` is an explicit per-call
opt-out, covered below).

**Neither schema supersedes the other, today.** This module never reads, embeds, or produces a
`clozn.run_diagnosis.v1` document, and `clozn/runs/diagnosis.py` is untouched by this file's existence.
Rule 11 below (`output_stopped_length`) and `clozn.run_diagnosis.v1`'s own `output_cutoff` finding answer
an overlapping question (was the reply cut off by a token/context budget) from the SAME underlying run
fields, independently implemented -- this is disclosed duplication, not an oversight: composing this
module's rule engine INTO `diagnosis.diagnose()` (or vice versa) is left to whichever later slice actually
needs one unified "why" document (D2's job, not this one's), so that decision is made deliberately, once,
with both artifacts already stable, rather than as an unreviewed side effect here.

STABLE SPAN ADDRESSING: ONE SCHEME, NEVER A SECOND
------------------------------------------------------
Every rule whose evidence is fundamentally ABOUT recorded text (an omitted segment, a conflicting or
duplicated instruction, repeated source content, a missing output format, an instruction placed far from
the final request, a source's measured influence) cites a `kind: "text_span"` evidence entry whose
`address_id` comes from ONE `clozn.text-span-addresses.v1` document, built ONCE per `evaluate()` call via
`clozn.runs.text_span_addresses.build_persisted_text_span_addresses` (reused verbatim, never reimplemented
-- see that module's own docstring). `local_start`/`local_end` (when present) are a sentence-level
sub-range WITHIN that same addressed text, for a human reader's benefit only -- never a second addressing
authority a caller could resolve independently of the referenced `address_id`. A rule whose evidence is a
structured, non-text field (a token count, `finish_reason`, a differing sampling setting) cites a
`kind: "field"` entry (dotted path + exact value) instead; inventing a fake span for a non-text fact would
be exactly the kind of fabrication `docs/SEAMS.md` rule 3 forbids.

HONESTY RULES (all structural, enforced in code, not just documented)
--------------------------------------------------------------------------
  * Rules 8/9 (source-influence rules) return `status="pending"` when this run never recorded an influence
    map at all (the measurement was never attempted -- nothing to guess from), and `status="unavailable"`
    when one WAS attempted but is not readable (a failed/expired blob, a non-`ok` artifact status, a
    schema mismatch) -- never a fabricated finding from the absence of measurement.
  * `evaluate(run)` called twice on the byte-identical `run` dict (with the same `generated_at` override)
    returns byte-identical output -- no wall-clock read except the caller-overridable `generated_at`, no
    `set`/`dict` iteration order leaking into the result (every set-derived list is sorted before it is
    ever placed in output).
  * A rule that cannot evaluate for THIS run (a missing field, redacted text, a missing comparison run)
    reports `status="unavailable"` (or `"pending"` for the two specific evidence-never-attempted cases
    above) with a plain-English reason -- it is never silently absent from `findings`. `findings` always
    has exactly `len(RULE_REGISTRY)` entries, one per rule, in registry order, regardless of outcome.
  * `suppressed_rule_ids` (a plain parameter to `evaluate()`, never a config file -- config wiring is a
    later slice) does not remove a rule from `findings`; it flips that ONE entry to `status="suppressed"`
    without ever calling the rule's own evaluation logic, so the total rule-id set stays audit-stable.
  * Conservative by design: every text-pattern rule (conflicting/duplicate instructions, output format,
    instruction placement) matches on NARROW, explicit patterns and reports `confidence="pattern_match"`
    rather than a numeric score -- a missed detection is preferred over a false one (a false "conflicting
    instructions" finding erodes trust faster than a miss; see `_DIRECTIVE_RE`'s own restraint).
  * No causal vocabulary: every finding describes an OBSERVATION or a STRUCTURAL fact ("two instructions
    with opposite polarity were both delivered"; "this source never cleared the measurement floor"),
    never a claim that the observation explains the model's behavior -- that synthesis is D2's job.

STDLIB ONLY, SEQUENTIAL COMPOSITION
--------------------------------------
No imports beyond the standard library at module scope; numpy is never imported here at all (nothing in
this module does numeric work heavier than a ratio or a string-similarity ratio, both plain Python).
Composes, never reimplements, wherever an existing primitive actually fits: `clozn.runs.text_span_addresses`
(span addressing, the ONE addressing scheme -- see above) and `clozn.analysis.run_diff` (`compare_runs`,
for rule 12's run-to-run drift, mirroring how `clozn.runs.investigation` already composes that same
module). `clozn.runs.sections.drill_split` was considered for sentence splitting and deliberately NOT
reused: its `DRILL_MIN_CHARS=40` merge step is tuned for ablatable-PROMPT-SECTION granularity (never leave
a 3-character sliver no human would call an ablatable unit) and folds any shorter fragment into its
neighbor -- exactly the wrong behavior here, where a directive is routinely well under 40 characters
("Always be concise." is 19) and two adjacent short directives merging into one span would silently hide
a second, possibly conflicting, directive from every text-pattern rule below. `_split_sentences` is a
small, self-contained sentence/newline splitter with no minimum-length merge, disclosed here rather than
silently diverging from the sibling module's own choice.
"""
from __future__ import annotations

import difflib
import json
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from clozn import schemas
from clozn.runs import text_span_addresses as tsa

SCHEMA_VERSION = "clozn.diagnosis-findings.v1"

STATUS_VALUES = ("finding", "not_observed", "unavailable", "pending", "suppressed")
SEVERITY_VALUES = ("info", "low", "medium", "high")
CONFIDENCE_VALUES = ("exact", "pattern_match", "derived")

# Provisional -- see module docstring's "STABLE SPAN ADDRESSING" note on suggested_actions.kind: this
# module OWNS this vocabulary until D3 registers `clozn.corrective-flow.v1` with its own action-kind enum.
SUGGESTED_ACTION_KINDS = frozenset({
    "resend_context", "increase_context_budget", "reconcile_conflicting_instructions",
    "deduplicate_instructions", "deduplicate_source_content", "clarify_output_format",
    "move_instruction_near_request", "reinforce_low_effect_source", "resupply_below_floor_source",
    "restate_conversation_instruction", "increase_max_tokens", "reconfirm_run_configuration",
    "no_action_available",
})

# ------------------------------------------------------------------------------------- named thresholds
# Every number a rule branches on is named here, once, so a fixture or a future tuning pass has one place
# to read and change it -- never a magic literal buried in a rule body.
BUDGET_PRESSURE_RATIO = 0.9          # R02: prompt_tokens / context_window_tokens heuristic floor
NEAR_FLOOR_MULTIPLIER = 1.5          # R09: "just barely cleared" zone above thresholds.cell_abs_delta_nats
FAR_INSTRUCTION_MESSAGE_GAP = 3      # R07: intervening delivered messages before "far" fires
DUPLICATE_EXACT_RATIO = 1.0          # R04: normalized-text similarity counted as an exact duplicate
DUPLICATE_NEAR_RATIO = 0.90          # R04: normalized-text similarity counted as a near-duplicate

_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")
_NEWLINE_RE = re.compile(r"\n+")

_DIRECTIVE_RE = re.compile(
    r"^(please\s+)?(always|never|must\s+not|must|do\s+not|don't|never\s+ever)\b", re.IGNORECASE)
_NEGATIVE_MARKERS = ("never", "do not", "don't", "must not", "never ever")
_TRAILING_PUNCT_RE = re.compile(r"[\s.!?,:;]+$")
_LEADING_MARKER_RE = re.compile(
    r"^(please\s+)?(always|never\s+ever|never|must\s+not|must|do\s+not|don't)\b\s*", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")

_FORMAT_REQUESTS: tuple[tuple[str, re.Pattern], ...] = (
    ("json", re.compile(r"\b(in|as|with|using)\s+json\b|\bjson\s+format\b|\bvalid\s+json\b", re.IGNORECASE)),
    ("bulleted_list", re.compile(r"\bbullet(ed)?\s+(list|points?)\b", re.IGNORECASE)),
    ("numbered_list", re.compile(r"\bnumbered\s+list\b", re.IGNORECASE)),
    ("markdown_table", re.compile(r"\b(markdown\s+table|as\s+a\s+table)\b", re.IGNORECASE)),
    ("single_word", re.compile(r"\b(one[\s-]word|single[\s-]word)\s+answer\b", re.IGNORECASE)),
    ("yes_or_no", re.compile(r"\byes\s+or\s+no\b", re.IGNORECASE)),
)
_BULLET_LINE_RE = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_NUMBERED_LINE_RE = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


# =============================================================================================== helpers

def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _str(value: Any) -> "str | None":
    return value if isinstance(value, str) and value else None


def _int(value: Any) -> "int | None":
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_is_redacted(run: Mapping[str, Any]) -> bool:
    """Mirrors `clozn.runs.text_span_addresses._run_is_redacted`'s own PUBLIC-shape logic (that function
    is private to its module; duplicated here rather than reached into, the same choice
    `clozn.analysis.transplant`/`causal_bisect` document making for an analogous case)."""
    redaction = run.get("redaction")
    return (
        isinstance(redaction, Mapping) and redaction.get("status") in {"redacted", "literal_redacted"}
    ) or "redacted" in (run.get("flags") or [])


def _field_evidence(path: str, value: Any, note: "str | None" = None) -> dict:
    item: dict = {"kind": "field", "path": path, "value": value}
    if note:
        item["note"] = note
    return item


def _span_evidence(address_id: str, *, message_index: "int | None" = None,
                   local_start: "int | None" = None, local_end: "int | None" = None,
                   note: "str | None" = None) -> dict:
    item: dict = {"kind": "text_span", "address_id": address_id}
    if message_index is not None:
        item["message_index"] = message_index
    if local_start is not None:
        item["local_start"] = local_start
    if local_end is not None:
        item["local_end"] = local_end
    if note:
        item["note"] = note
    return item


def _action(kind: str, description: str) -> dict:
    assert kind in SUGGESTED_ACTION_KINDS, f"unknown suggested_action kind {kind!r}"   # authoring guard
    return {"kind": kind, "description": description}


def _finding_entry(rule_id: str, rule_name: str, *, status: str, summary: str, evidence: Sequence[dict] = (),
                   limitations: Sequence[str] = (), severity: "str | None" = None,
                   confidence: "str | None" = None, suggested_actions: Sequence[dict] = ()) -> dict:
    assert status in STATUS_VALUES, f"unknown status {status!r}"
    entry: dict = {"rule_id": rule_id, "rule_name": rule_name, "status": status, "summary": summary,
                  "evidence": list(evidence), "limitations": list(limitations)}
    if status == "finding":
        assert severity in SEVERITY_VALUES, f"finding {rule_id} needs a valid severity, got {severity!r}"
        assert confidence in CONFIDENCE_VALUES, f"finding {rule_id} needs a valid confidence, got {confidence!r}"
        entry["severity"] = severity
        entry["confidence"] = confidence
        entry["suggested_actions"] = list(suggested_actions)
    return entry


def _unavailable(rule_id: str, rule_name: str, reason: str, evidence: Sequence[dict] = ()) -> dict:
    return _finding_entry(rule_id, rule_name, status="unavailable", summary=reason, evidence=evidence)


def _pending(rule_id: str, rule_name: str, reason: str, evidence: Sequence[dict] = ()) -> dict:
    return _finding_entry(rule_id, rule_name, status="pending", summary=reason, evidence=evidence)


def _not_observed(rule_id: str, rule_name: str, summary: str, evidence: Sequence[dict] = ()) -> dict:
    return _finding_entry(rule_id, rule_name, status="not_observed", summary=summary, evidence=evidence)


# ================================================================================ text/sentence helpers

def _messages(run: Mapping[str, Any]) -> list:
    messages = run.get("messages")
    return [m for m in messages if isinstance(m, Mapping)] if isinstance(messages, list) else []


def _message_text(message: Mapping[str, Any]) -> "str | None":
    content = message.get("content")
    return content if isinstance(content, str) and content else None


def _final_user_index(messages: Sequence[Mapping[str, Any]]) -> "int | None":
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return None


def _address_for_message_index(document: "dict | None", index: int, message: Mapping[str, Any]) -> "str | None":
    """The `delivered_message`/`attached_source_span` address_id for `messages[index]`, matched against
    the shared span document's own PUBLIC `native_ref` fields only -- never a second computation of the
    address_id itself. Matches by `client_source_id` first (an explicit, stable client identity), else by
    the same `f"message-{index}"` native id `text_span_addresses` assigns positionally."""
    if not isinstance(document, dict):
        return None
    client_source_id = message.get("source_id")
    client_source_id = client_source_id if isinstance(client_source_id, str) and client_source_id else None
    fallback_id = f"message-{index}"
    for address in document.get("addresses") or []:
        if not isinstance(address, dict) or address.get("kind") not in (
                "delivered_message", "attached_source_span"):
            continue
        native_ref = _object(address.get("native_ref"))
        if client_source_id and native_ref.get("client_source_id") == client_source_id:
            return address.get("address_id")
        if not client_source_id and native_ref.get("id") == fallback_id and not native_ref.get(
                "client_source_id"):
            return address.get("address_id")
    return None


def _normalize_directive_text(sentence: str) -> str:
    return _WHITESPACE_RE.sub(" ", _TRAILING_PUNCT_RE.sub("", sentence.strip())).lower()


def _directive_polarity(sentence: str) -> "str | None":
    match = _DIRECTIVE_RE.match(sentence.strip())
    if not match:
        return None
    marker = match.group(0).lower()
    return "negative" if any(neg in marker for neg in _NEGATIVE_MARKERS) else "positive"


def _directive_subject(sentence: str) -> str:
    """The directive's normalized text with its leading polarity marker stripped -- "always answer in
    english." and "never answer in english." both reduce to "answer in english", which is what R03/R10
    group on to find opposite-polarity pairs about the SAME subject."""
    stripped = _LEADING_MARKER_RE.sub("", sentence.strip(), count=1)
    return _normalize_directive_text(stripped)


class _Directive:
    __slots__ = ("message_index", "role", "start", "end", "text", "polarity", "subject", "normalized")

    def __init__(self, message_index: int, role: str, start: int, end: int, text: str):
        self.message_index = message_index
        self.role = role
        self.start = start
        self.end = end
        self.text = text
        self.polarity = _directive_polarity(text)
        self.subject = _directive_subject(text)
        self.normalized = _normalize_directive_text(text)


def _split_sentences(text: str) -> list[tuple[int, int]]:
    """A sentence/line boundary splitter with NO minimum-length merge (see module docstring's explanation
    of why `clozn.runs.sections.drill_split` was not reused here) -- a boundary is a run of `.`/`!`/`?`
    immediately followed by whitespace or the end of the text, or a hard newline. Deterministic, stdlib
    `re` only. Always returns at least one span; an empty string returns `[(0, 0)]`."""
    if not text:
        return [(0, 0)]
    points = {0, len(text)}
    for match in _SENTENCE_END_RE.finditer(text):
        points.add(match.end())
    for match in _NEWLINE_RE.finditer(text):
        points.add(match.start())
        points.add(match.end())
    ordered = sorted(points)
    spans: list[tuple[int, int]] = []
    for a, b in zip(ordered, ordered[1:]):
        if b <= a:
            continue
        start, end = a, b
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans or [(0, len(text))]


def _extract_directives(messages: Sequence[Mapping[str, Any]], *, roles=("system", "user")) -> list:
    """Every directive-like sentence across `messages` (restricted to instruction-bearing roles by
    default -- an assistant reply is never a caller instruction), in message order. Sentence boundaries
    come from `_split_sentences`; a sentence counts as a directive only when `_DIRECTIVE_RE` matches its
    own leading marker -- deliberately narrow (see module docstring's conservative-by-design rule)."""
    directives: list = []
    for index, message in enumerate(messages):
        if message.get("role") not in roles:
            continue
        text = _message_text(message)
        if text is None:
            continue
        for start, end in _split_sentences(text):
            sentence = text[start:end]
            if _directive_polarity(sentence) is not None:
                directives.append(_Directive(index, message.get("role"), start, end, sentence))
    return directives


def _conflicting_pairs(directives: Sequence["_Directive"]) -> list:
    """Pairs of directives sharing the SAME normalized subject but OPPOSITE polarity, deduplicated by
    (subject, message_index) so the same repeated conflict is not reported once per sentence occurrence
    beyond what is needed to name every distinct message pair involved."""
    by_subject: dict = {}
    for directive in directives:
        if directive.subject:
            by_subject.setdefault(directive.subject, []).append(directive)
    pairs = []
    for subject, group in sorted(by_subject.items()):
        positives = [d for d in group if d.polarity == "positive"]
        negatives = [d for d in group if d.polarity == "negative"]
        if positives and negatives:
            pairs.append((subject, positives, negatives))
    return pairs


# =================================================================================== the shared context

class _Context:
    """Everything every rule needs, computed ONCE per `evaluate()` call -- never recomputed per rule, so
    two rules reading the same evidence always see the byte-identical derived state."""

    def __init__(self, run: Mapping[str, Any], comparison_run: "Mapping[str, Any] | None"):
        self.run = run
        self.comparison_run = comparison_run
        # "?" (never an empty string) mirrors clozn.analysis.run_diff's own convention for the identical
        # gap (`run_a.get("id") or "?"`) -- an explicit, unmistakable sentinel, never a fabricated id and
        # never a value that could fail this document's own run_id minLength.
        self.real_run_id = _str(run.get("id"))
        self.run_id = self.real_run_id or "?"
        self.redacted = _run_is_redacted(run)
        self.messages = _messages(run)
        self.final_user_index = _final_user_index(self.messages)
        self.context_receipt = _object(run.get("context_receipt"))
        self.span_document, self.span_error = self._build_span_document()
        self.directives = [] if self.redacted else _extract_directives(self.messages)

    def _build_span_document(self) -> tuple["dict | None", "str | None"]:
        if not self.real_run_id:
            return None, "the run record has no non-empty 'id', so no span document could be built"
        try:
            return tsa.build_persisted_text_span_addresses(dict(self.run)), None
        except Exception as exc:      # noqa: BLE001 -- never let evidence-building crash the engine
            return None, f"{type(exc).__name__}: {exc}"

    def address_for_message(self, index: int) -> "str | None":
        if not (0 <= index < len(self.messages)):
            return None
        return _address_for_message_index(self.span_document, index, self.messages[index])


# =========================================================================================== R01 - R12

def _rule_input_omitted_or_rejected(ctx: "_Context") -> dict:
    rule_id, rule_name = "R01", "input_omitted_or_rejected"
    if ctx.redacted:
        return _unavailable(rule_id, rule_name, "this run's text was redacted; omitted/rejected segments cannot be examined")
    receipt = ctx.context_receipt
    if not receipt:
        return _unavailable(rule_id, rule_name, "no context receipt was recorded for this run")

    excluded: dict = {}   # segment_id -> (reason, source: "omissions"|"delivered")
    omissions = receipt.get("omissions")
    if isinstance(omissions, list):
        for item in omissions:
            if isinstance(item, Mapping) and isinstance(item.get("segment_id"), str):
                excluded.setdefault(item["segment_id"], (item.get("reason"), "omissions"))
    delivered = receipt.get("delivered")
    if isinstance(delivered, list):
        for item in delivered:
            if isinstance(item, Mapping) and item.get("included") is False and isinstance(
                    item.get("segment_id"), str):
                excluded.setdefault(item["segment_id"], (item.get("reason"), "delivered"))

    if not omissions and not isinstance(delivered, list):
        return _unavailable(rule_id, rule_name,
                            "the context receipt has neither an omissions list nor a delivered segment "
                            "list, so omitted/rejected input cannot be examined")
    if not excluded:
        return _not_observed(rule_id, rule_name,
                             "no delivered input segment was recorded as omitted or excluded")

    evidence = []
    for segment_id in sorted(excluded):
        reason, source = excluded[segment_id]
        address_id = None
        for address in (ctx.span_document or {}).get("addresses") or []:
            native_ref = _object(address.get("native_ref"))
            if native_ref.get("segment_id") == segment_id:
                address_id = address.get("address_id")
                break
        if address_id:
            evidence.append(_span_evidence(address_id, note=f"reason={reason}"))
        else:
            evidence.append(_field_evidence(f"context_receipt.{source}", {"segment_id": segment_id, "reason": reason}))
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="medium", confidence="exact",
        summary=f"{len(excluded)} input segment(s) were omitted from the assembled context: "
               f"{', '.join(sorted(segment_id for segment_id in excluded))}.",
        evidence=evidence,
        suggested_actions=[_action("resend_context",
                                   "resend or re-attach the omitted segment(s) if they were meant to reach the model")],
        limitations=["a whole-request rejection (the request refused before any run was recorded, e.g. an "
                    "overlong prompt) cannot be observed here -- this rule only sees segments belonging "
                    "to a run that WAS recorded",
                    "reason codes reflect what the context-receipt builder recorded; several reason codes "
                    "are defined for forward compatibility and have no live producer in clozn today"])


def _rule_context_budget_pressure(ctx: "_Context") -> dict:
    rule_id, rule_name = "R02", "context_budget_pressure"
    receipt = ctx.context_receipt
    termination = _object(receipt.get("termination"))
    if termination.get("reason") == "context_limit":
        return _finding_entry(
            rule_id, rule_name, status="finding", severity="high", confidence="exact",
            summary="generation stopped because the context window was reached (context_receipt.termination.reason == 'context_limit').",
            evidence=[_field_evidence("context_receipt.termination.reason", "context_limit")],
            suggested_actions=[_action("increase_context_budget",
                                       "reduce prompt size or use a model/config with a larger context window")],
            limitations=["termination.reason reflects the engine's own report of why generation stopped; "
                        "it does not measure how much headroom would have been enough"])
    limits = _object(receipt.get("limits"))
    prompt_tokens, context_window = _int(limits.get("prompt_tokens")), _int(limits.get("context_window_tokens"))
    if prompt_tokens is None or not context_window:
        return _unavailable(rule_id, rule_name,
                            "prompt_tokens and context_window_tokens were not both recorded, so context-budget pressure is unavailable")
    ratio = prompt_tokens / context_window
    if ratio >= BUDGET_PRESSURE_RATIO:
        return _finding_entry(
            rule_id, rule_name, status="finding", severity="medium", confidence="derived",
            summary=f"the prompt used {prompt_tokens} of {context_window} context tokens ({ratio * 100:.1f}%), "
                   f"at or above the {BUDGET_PRESSURE_RATIO * 100:.0f}% pressure heuristic.",
            evidence=[_field_evidence("context_receipt.limits.prompt_tokens", prompt_tokens),
                     _field_evidence("context_receipt.limits.context_window_tokens", context_window)],
            suggested_actions=[_action("increase_context_budget",
                                       "reduce prompt size or use a model/config with a larger context window")],
            limitations=[f"a fixed {BUDGET_PRESSURE_RATIO * 100:.0f}% ratio heuristic, not a measured "
                        "allocation failure -- this measures capacity used, never latency or truncation"])
    return _not_observed(
        rule_id, rule_name,
        f"the prompt used {prompt_tokens} of {context_window} context tokens ({ratio * 100:.1f}%), "
        f"below the {BUDGET_PRESSURE_RATIO * 100:.0f}% pressure heuristic.",
        evidence=[_field_evidence("context_receipt.limits.prompt_tokens", prompt_tokens),
                 _field_evidence("context_receipt.limits.context_window_tokens", context_window)])


def _directive_evidence(ctx: "_Context", directive: "_Directive") -> dict:
    address_id = ctx.address_for_message(directive.message_index)
    if address_id:
        return _span_evidence(address_id, message_index=directive.message_index,
                              local_start=directive.start, local_end=directive.end)
    return _field_evidence(f"messages[{directive.message_index}]",
                           {"start": directive.start, "end": directive.end, "text": directive.text})


def _rule_conflicting_instructions(ctx: "_Context") -> dict:
    rule_id, rule_name = "R03", "conflicting_instructions"
    if ctx.redacted:
        return _unavailable(rule_id, rule_name, "this run's text was redacted; instructions cannot be compared")
    if not ctx.messages:
        return _unavailable(rule_id, rule_name, "no messages were recorded for this run")
    pairs = _conflicting_pairs(ctx.directives)
    if not pairs:
        return _not_observed(rule_id, rule_name,
                             f"no opposite-polarity instruction pairs were found among {len(ctx.directives)} "
                             f"detected directive(s).")
    evidence = []
    subjects = []
    for subject, positives, negatives in pairs:
        subjects.append(subject)
        for directive in positives + negatives:
            evidence.append(_directive_evidence(ctx, directive))
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="medium", confidence="pattern_match",
        summary=f"{len(pairs)} instruction subject(s) received both a positive and a negative directive: "
               f"{', '.join(subjects)}.",
        evidence=evidence,
        suggested_actions=[_action("reconcile_conflicting_instructions",
                                   "remove or restate one of the conflicting directives")],
        limitations=["matches only sentences with an explicit leading marker (always/never/must/do not/"
                    "don't) and an exact normalized-subject match after the marker is stripped -- "
                    "paraphrased or implicit conflicts are not detected"])


def _rule_duplicate_instructions(ctx: "_Context") -> dict:
    rule_id, rule_name = "R04", "duplicate_instructions"
    if ctx.redacted:
        return _unavailable(rule_id, rule_name, "this run's text was redacted; instructions cannot be compared")
    if not ctx.messages:
        return _unavailable(rule_id, rule_name, "no messages were recorded for this run")
    directives = ctx.directives
    seen_pairs: set = set()
    groups: list = []
    for i in range(len(directives)):
        for j in range(i + 1, len(directives)):
            a, b = directives[i], directives[j]
            if (a.message_index, a.start) == (b.message_index, b.start):
                continue
            ratio = difflib.SequenceMatcher(a=a.normalized, b=b.normalized).ratio()
            if ratio >= DUPLICATE_NEAR_RATIO:
                key = (i, j)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                kind = "duplicate" if ratio >= DUPLICATE_EXACT_RATIO else "near-duplicate"
                groups.append((kind, ratio, a, b))
    if not groups:
        return _not_observed(rule_id, rule_name,
                             f"no duplicate or near-duplicate instructions were found among "
                             f"{len(directives)} detected directive(s).")
    evidence = []
    exact = sum(1 for kind, *_ in groups if kind == "duplicate")
    near = len(groups) - exact
    for kind, ratio, a, b in groups:
        evidence.append(_directive_evidence(ctx, a))
        evidence.append(_directive_evidence(ctx, b))
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="low", confidence="pattern_match",
        summary=f"{exact} exact and {near} near-duplicate instruction pair(s) were found "
               f"(similarity >= {DUPLICATE_NEAR_RATIO:.2f}).",
        evidence=evidence,
        suggested_actions=[_action("deduplicate_instructions",
                                   "remove or merge the repeated directive(s)")],
        limitations=[f"near-duplicate uses a normalized-text similarity ratio (difflib), threshold "
                    f"{DUPLICATE_NEAR_RATIO:.2f} -- a wording-level measure, not a semantic one"])


def _rule_repeated_source_content(ctx: "_Context") -> dict:
    rule_id, rule_name = "R05", "repeated_source_content"
    if ctx.redacted:
        return _unavailable(rule_id, rule_name, "this run's text was redacted; source content cannot be compared")
    receipt = ctx.context_receipt
    delivered = receipt.get("delivered")
    if not isinstance(delivered, list) or not delivered:
        return _unavailable(rule_id, rule_name, "no delivered segment list was recorded for this run")
    by_hash: dict = {}
    for item in delivered:
        if not isinstance(item, Mapping):
            continue
        if not item.get("client_source_id"):
            continue     # scoped to explicit attached sources, not ordinary repeated conversational turns
        content_hash = item.get("content_hash")
        segment_id = item.get("segment_id")
        if isinstance(content_hash, str) and isinstance(segment_id, str):
            by_hash.setdefault(content_hash, []).append(segment_id)
    repeated = {h: ids for h, ids in by_hash.items() if len(ids) > 1}
    if not by_hash:
        return _unavailable(rule_id, rule_name,
                            "no attached source carried both a client_source_id and a content_hash, so "
                            "repeated source content cannot be examined")
    if not repeated:
        return _not_observed(rule_id, rule_name, "no attached source content repeated across segments")
    evidence = []
    for content_hash in sorted(repeated):
        for segment_id in sorted(repeated[content_hash]):
            address_id = None
            for address in (ctx.span_document or {}).get("addresses") or []:
                if _object(address.get("native_ref")).get("segment_id") == segment_id:
                    address_id = address.get("address_id")
                    break
            if address_id:
                evidence.append(_span_evidence(address_id, note=f"content_hash={content_hash}"))
            else:
                evidence.append(_field_evidence("context_receipt.delivered",
                                                {"segment_id": segment_id, "content_hash": content_hash}))
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="low", confidence="exact",
        summary=f"{len(repeated)} distinct source content hash(es) each appeared in more than one "
               f"attached segment.",
        evidence=evidence,
        suggested_actions=[_action("deduplicate_source_content",
                                   "remove the repeated attachment(s) or merge them into one")],
        limitations=["compares the receipt's own 16-hex content_hash, an exact-content signal -- "
                    "near-duplicate source text (paraphrased or partially overlapping) is not detected"])


def _format_request_evidence(ctx: "_Context", name: str, match: "re.Match") -> dict:
    """One `text_span` (preferred) or `field` evidence entry for a recognized output-format request,
    pointing at the exact matched phrase's location in the final user message -- never the format name
    alone, and never the reply text (evidence cites what was ASKED for, not a copy of the answer)."""
    address_id = ctx.address_for_message(ctx.final_user_index)
    if address_id:
        return _span_evidence(address_id, message_index=ctx.final_user_index,
                              local_start=match.start(), local_end=match.end(),
                              note=f"recognized output-format request: {name}")
    return _field_evidence(f"messages[{ctx.final_user_index}].content",
                           {"format": name, "start": match.start(), "end": match.end()})


def _rule_requested_format_absent(ctx: "_Context") -> dict:
    rule_id, rule_name = "R06", "requested_format_absent"
    if ctx.redacted:
        return _unavailable(rule_id, rule_name, "this run's text was redacted; the request/reply cannot be examined")
    if ctx.final_user_index is None:
        return _unavailable(rule_id, rule_name, "no final user message was recorded for this run")
    request_text = _message_text(ctx.messages[ctx.final_user_index])
    if request_text is None:
        return _unavailable(rule_id, rule_name, "the final user message has no recorded text")
    requested = [(name, pattern.search(request_text)) for name, pattern in _FORMAT_REQUESTS]
    requested = [(name, match) for name, match in requested if match is not None]
    if not requested:
        return _not_observed(rule_id, rule_name,
                             "no recognized output-format request was found in the final user message")
    request_evidence = [_format_request_evidence(ctx, name, match) for name, match in requested]
    reply = run_response(ctx.run)
    if reply is None:
        return _unavailable(rule_id, rule_name,
                            "an output-format request was recognized, but this run has no recorded reply text",
                            evidence=request_evidence)
    unmet = [name for name, _match in requested if not _format_satisfied(name, reply)]
    if not unmet:
        return _not_observed(
            rule_id, rule_name,
            f"the reply satisfies the recognized output-format request(s): "
            f"{', '.join(name for name, _m in requested)}.",
            evidence=request_evidence)
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="medium", confidence="pattern_match",
        summary=f"the reply does not satisfy the requested output format: {', '.join(unmet)}.",
        evidence=request_evidence,
        suggested_actions=[_action("clarify_output_format",
                                   "restate the desired output format explicitly, close to the final request")],
        limitations=["format detection covers a small, explicit set of requests (JSON, bulleted/numbered "
                    "list, markdown table, single-word, yes/no) -- an unrecognized phrasing is silently "
                    "not checked, never guessed at"])


def run_response(run: Mapping[str, Any]) -> "str | None":
    response = run.get("response")
    return response if isinstance(response, str) and response else None


def _format_satisfied(name: str, reply: str) -> bool:
    if name == "json":
        stripped = reply.strip()
        candidates = [stripped]
        fenced = _FENCED_JSON_RE.search(reply)
        if fenced:
            candidates.append(fenced.group(1).strip())
        for candidate in candidates:
            if not candidate:
                continue
            try:
                json.loads(candidate)
                return True
            except (ValueError, TypeError):
                continue
        return False
    if name == "bulleted_list":
        return len(_BULLET_LINE_RE.findall(reply)) >= 2
    if name == "numbered_list":
        return len(_NUMBERED_LINE_RE.findall(reply)) >= 2
    if name == "markdown_table":
        return len(_TABLE_ROW_RE.findall(reply)) >= 2
    if name == "single_word":
        return len(reply.strip().split()) == 1
    if name == "yes_or_no":
        return reply.strip().strip(".! ").lower() in ("yes", "no")
    return True


def _rule_instruction_far_from_request(ctx: "_Context") -> dict:
    rule_id, rule_name = "R07", "instruction_far_from_request"
    if ctx.redacted:
        return _unavailable(rule_id, rule_name, "this run's text was redacted; instruction placement cannot be examined")
    if ctx.final_user_index is None:
        return _unavailable(rule_id, rule_name, "no final user message was recorded for this run")
    earlier = [d for d in ctx.directives if d.message_index < ctx.final_user_index]
    if not earlier:
        return _not_observed(rule_id, rule_name,
                             "no directive-bearing message precedes the final user message")
    far = [d for d in earlier if ctx.final_user_index - d.message_index > FAR_INSTRUCTION_MESSAGE_GAP]
    if not far:
        return _not_observed(
            rule_id, rule_name,
            f"the nearest preceding directive is within {FAR_INSTRUCTION_MESSAGE_GAP} message(s) of the "
            f"final user message.")
    evidence = [_directive_evidence(ctx, d) for d in far]
    gaps = sorted({ctx.final_user_index - d.message_index for d in far})
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="low", confidence="derived",
        summary=f"{len(far)} directive(s) are more than {FAR_INSTRUCTION_MESSAGE_GAP} message(s) before "
               f"the final user message (gap(s): {', '.join(str(g) for g in gaps)}).",
        evidence=evidence,
        suggested_actions=[_action("move_instruction_near_request",
                                   "restate the instruction closer to the final request")],
        limitations=[f"a fixed {FAR_INSTRUCTION_MESSAGE_GAP}-message gap heuristic over message COUNT, "
                    "not rendered character distance or token distance"])


def _rule_request_conflicts_with_earlier_instructions(ctx: "_Context") -> dict:
    rule_id, rule_name = "R10", "request_conflicts_with_earlier_instructions"
    if ctx.redacted:
        return _unavailable(rule_id, rule_name, "this run's text was redacted; instructions cannot be compared")
    if ctx.final_user_index is None:
        return _unavailable(rule_id, rule_name, "no final user message was recorded for this run")
    final_directives = [d for d in ctx.directives if d.message_index == ctx.final_user_index]
    earlier_directives = [d for d in ctx.directives if d.message_index < ctx.final_user_index]
    if not final_directives:
        return _not_observed(rule_id, rule_name,
                             "the final user message contains no detected directive to compare")
    if not earlier_directives:
        return _not_observed(rule_id, rule_name, "no earlier message contains a detected directive")
    by_subject: dict = {}
    for directive in earlier_directives:
        if directive.subject:
            by_subject.setdefault(directive.subject, []).append(directive)
    conflicts = []
    for directive in final_directives:
        if not directive.subject:
            continue
        earlier_group = by_subject.get(directive.subject, [])
        opposite = [d for d in earlier_group if d.polarity != directive.polarity]
        if opposite:
            conflicts.append((directive, opposite))
    if not conflicts:
        return _not_observed(rule_id, rule_name,
                             "no final-request directive conflicts in polarity with an earlier instruction "
                             "on the same normalized subject")
    evidence = []
    subjects = []
    for directive, opposite in conflicts:
        subjects.append(directive.subject)
        evidence.append(_directive_evidence(ctx, directive))
        for other in opposite:
            evidence.append(_directive_evidence(ctx, other))
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="medium", confidence="pattern_match",
        summary=f"the final request conflicts in polarity with an earlier instruction on "
               f"{len(conflicts)} subject(s): {', '.join(subjects)}.",
        evidence=evidence,
        suggested_actions=[_action("restate_conversation_instruction",
                                   "confirm whether the earlier instruction still applies, or restate it")],
        limitations=["matches only sentences with an explicit leading marker and an exact "
                    "normalized-subject match -- paraphrased or implicit conflicts are not detected"])


def _link_context_span_address(ctx: "_Context", context_span_id: str) -> "str | None":
    for address in (ctx.span_document or {}).get("addresses") or []:
        native_ref = _object(address.get("native_ref"))
        if native_ref.get("collection") in ("influence.prompt_spans", "influence.prompt_sources") \
                and native_ref.get("id") == context_span_id:
            return address.get("address_id")
    return None


def _influence_readiness(run: Mapping[str, Any]) -> tuple:
    """(state, artifact_or_none, reason) where state is 'pending' (never attempted), 'unavailable'
    (attempted, not usable), or 'ok' (a real, status=='ok' clozn.context_answer_influence.v1 artifact)."""
    if "influence_map" not in run:
        return "pending", None, "this run never recorded an influence map"
    influence = run.get("influence_map")
    if isinstance(influence, Mapping) and isinstance(influence.get("unavailable"), str):
        return "unavailable", None, influence["unavailable"]
    if not isinstance(influence, Mapping) or not influence:
        return "unavailable", None, "the recorded influence evidence is not a usable object"
    schema_name = influence.get("schema_version") or influence.get("schema")
    if schema_name not in (tsa.INFLUENCE_SCHEMA, tsa.INFLUENCE_EXPORT_SCHEMA):
        return "unavailable", None, "the recorded influence evidence has an unsupported or missing schema"
    if influence.get("status") != "ok":
        error = _object(influence.get("error"))
        reason = _str(error.get("message")) or f"influence measurement status was {influence.get('status')!r}"
        return "unavailable", None, reason
    return "ok", influence, None


def _rule_source_below_measurement_floor(ctx: "_Context") -> dict:
    rule_id, rule_name = "R08", "source_below_measurement_floor"
    state, artifact, reason = _influence_readiness(ctx.run)
    if state == "pending":
        return _pending(rule_id, rule_name, reason)
    if state == "unavailable":
        return _unavailable(rule_id, rule_name, reason)
    links = [link for link in (artifact.get("links") or []) if isinstance(link, Mapping)]
    by_span: dict = {}
    for link in links:
        span_id = link.get("context_span_id")
        if isinstance(span_id, str):
            by_span.setdefault(span_id, []).append(link)
    below_floor = sorted(span_id for span_id, group in by_span.items()
                         if group and all(item.get("clears_floor") is False for item in group))
    if not by_span:
        return _unavailable(rule_id, rule_name, "the influence artifact recorded no links to evaluate")
    if not below_floor:
        return _not_observed(rule_id, rule_name,
                             "every measured source cleared the measurement floor for at least one answer span")
    evidence = []
    for span_id in below_floor:
        address_id = _link_context_span_address(ctx, span_id)
        if address_id:
            evidence.append(_span_evidence(address_id, note="never cleared the measurement floor"))
        else:
            evidence.append(_field_evidence("influence_map.links", {"context_span_id": span_id, "clears_floor": False}))
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="low", confidence="exact",
        summary=f"{len(below_floor)} source(s) were delivered and measured but never cleared the "
               f"influence measurement floor for any answer span.",
        evidence=evidence,
        suggested_actions=[_action("resupply_below_floor_source",
                                   "consider resupplying or emphasizing the source if it was meant to matter")],
        limitations=["clears_floor=False is a measured ABSENCE of a strong effect under one intervention "
                    "method (forced_score_intervention), never proof the source is irrelevant"])


def _rule_source_little_effect(ctx: "_Context") -> dict:
    rule_id, rule_name = "R09", "source_little_effect"
    state, artifact, reason = _influence_readiness(ctx.run)
    if state == "pending":
        return _pending(rule_id, rule_name, reason)
    if state == "unavailable":
        return _unavailable(rule_id, rule_name, reason)
    thresholds = _object(artifact.get("thresholds"))
    floor = thresholds.get("cell_abs_delta_nats")
    if not isinstance(floor, (int, float)) or isinstance(floor, bool):
        return _unavailable(rule_id, rule_name,
                            "the influence artifact has no numeric thresholds.cell_abs_delta_nats to "
                            "measure 'little effect' against")
    links = [link for link in (artifact.get("links") or []) if isinstance(link, Mapping)]
    by_span: dict = {}
    for link in links:
        span_id = link.get("context_span_id")
        if isinstance(span_id, str):
            by_span.setdefault(span_id, []).append(link)
    ceiling = float(floor) * NEAR_FLOOR_MULTIPLIER
    weak = sorted(span_id for span_id, group in by_span.items() if group and any(
        item.get("clears_floor") is True and isinstance(item.get("abs_delta_nats"), (int, float))
        and not isinstance(item.get("abs_delta_nats"), bool) and item["abs_delta_nats"] < ceiling
        for item in group) and not any(
        item.get("clears_floor") is True and isinstance(item.get("abs_delta_nats"), (int, float))
        and not isinstance(item.get("abs_delta_nats"), bool) and item["abs_delta_nats"] >= ceiling
        for item in group))
    if not by_span:
        return _unavailable(rule_id, rule_name, "the influence artifact recorded no links to evaluate")
    if not weak:
        return _not_observed(rule_id, rule_name,
                             "no source cleared the measurement floor only narrowly (within "
                             f"{NEAR_FLOOR_MULTIPLIER:g}x) at every answer span it cleared")
    evidence = []
    for span_id in weak:
        address_id = _link_context_span_address(ctx, span_id)
        if address_id:
            evidence.append(_span_evidence(address_id, note="cleared the floor narrowly"))
        else:
            evidence.append(_field_evidence("influence_map.links", {"context_span_id": span_id}))
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="low", confidence="derived",
        summary=f"{len(weak)} source(s) cleared the measurement floor, but only within "
               f"{NEAR_FLOOR_MULTIPLIER:g}x of it wherever they cleared it at all.",
        evidence=evidence,
        suggested_actions=[_action("reinforce_low_effect_source",
                                   "consider restating or emphasizing the source if a stronger effect was expected")],
        limitations=[f"'little effect' is a fixed {NEAR_FLOOR_MULTIPLIER:g}x-the-floor heuristic on "
                    "abs_delta_nats, not a statistical claim; effect is sign-only and not magnitude-"
                    "comparable across spans of different lengths"])


def _rule_output_stopped_length(ctx: "_Context") -> dict:
    rule_id, rule_name = "R11", "output_stopped_length"
    run = ctx.run
    finish_reason = run.get("finish_reason")
    termination = _object(ctx.context_receipt.get("termination"))
    term_reason = termination.get("reason") if isinstance(termination.get("reason"), str) else None

    if term_reason in ("max_tokens", "context_limit"):
        cause = ("the requested output-token limit was reached" if term_reason == "max_tokens"
                else "the model's context window was reached during generation")
        evidence = [_field_evidence("context_receipt.termination.reason", term_reason)]
        if isinstance(termination.get("generated_tokens"), int) and not isinstance(
                termination.get("generated_tokens"), bool):
            evidence.append(_field_evidence("context_receipt.termination.generated_tokens",
                                            termination["generated_tokens"]))
        return _finding_entry(
            rule_id, rule_name, status="finding", severity="high", confidence="exact",
            summary=f"generation stopped because {cause}.", evidence=evidence,
            suggested_actions=[_action("increase_max_tokens", "raise the output token budget")
                              if term_reason == "max_tokens" else
                              _action("increase_context_budget", "reduce prompt size or raise the context window")],
            limitations=[])
    if term_reason is not None:
        return _not_observed(rule_id, rule_name,
                             f"generation stopped for a recorded reason other than a length limit "
                             f"(context_receipt.termination.reason == {term_reason!r}).",
                             evidence=[_field_evidence("context_receipt.termination.reason", term_reason)])

    if finish_reason == "length":
        return _finding_entry(
            rule_id, rule_name, status="finding", severity="high", confidence="exact",
            summary="generation stopped at a token-budget limit (finish_reason == 'length'); no "
                   "context_receipt.termination evidence was recorded to separate output-cap from "
                   "context-window causes.",
            evidence=[_field_evidence("finish_reason", "length")],
            suggested_actions=[_action("increase_max_tokens", "raise the output token budget")],
            limitations=["without context_receipt.termination this rule cannot separate an output-token "
                        "cap from a context-window limit"])
    if isinstance(finish_reason, str):
        return _not_observed(rule_id, rule_name,
                             f"generation recorded a normal stop, not a length cutoff (finish_reason == {finish_reason!r}).",
                             evidence=[_field_evidence("finish_reason", finish_reason)])
    return _unavailable(rule_id, rule_name,
                        "no finish_reason or context_receipt.termination evidence was recorded for this run")


def _rule_run_to_run_drift(ctx: "_Context") -> dict:
    rule_id, rule_name = "R12", "run_to_run_drift"
    if ctx.comparison_run is None:
        return _pending(rule_id, rule_name, "no comparison run was supplied for this evaluation")
    from clozn.analysis import run_diff
    result = run_diff.compare_runs(dict(ctx.comparison_run), dict(ctx.run))
    if not result.get("ok"):
        return _unavailable(rule_id, rule_name,
                            f"the comparison run could not be diffed: {result.get('error')}")
    relevant = sorted(
        (d for d in result.get("differences") or []
         if isinstance(d, Mapping) and isinstance(d.get("dimension"), str)
         and (d["dimension"].startswith("identity.") or d["dimension"].startswith("generation."))
         and d.get("kind") not in ("unavailable", "diff_failed")),
        key=lambda d: d["dimension"])
    if not relevant:
        return _not_observed(rule_id, rule_name,
                             "no identity or generation-setting difference was found against the "
                             "supplied comparison run")
    evidence = [_field_evidence(d["dimension"],
                                {"value_a": d.get("value_a"), "value_b": d.get("value_b"), "kind": d.get("kind")})
               for d in relevant]
    dimensions = ", ".join(d["dimension"] for d in relevant)
    return _finding_entry(
        rule_id, rule_name, status="finding", severity="medium", confidence="exact",
        summary=f"{len(relevant)} identity/setting dimension(s) differ from the supplied comparison run: "
               f"{dimensions}.",
        evidence=evidence,
        suggested_actions=[_action("reconfirm_run_configuration",
                                   "confirm whether the setting/identity change was intended")],
        limitations=["reuses clozn.analysis.run_diff.compare_runs() verbatim -- see that module for the "
                    "exact dimension set and ranking; a difference is a structural fact, never a claim "
                    "about which run's behavior is 'correct'"])


RULE_REGISTRY: tuple[tuple[str, str, Any], ...] = (
    ("R01", "input_omitted_or_rejected", _rule_input_omitted_or_rejected),
    ("R02", "context_budget_pressure", _rule_context_budget_pressure),
    ("R03", "conflicting_instructions", _rule_conflicting_instructions),
    ("R04", "duplicate_instructions", _rule_duplicate_instructions),
    ("R05", "repeated_source_content", _rule_repeated_source_content),
    ("R06", "requested_format_absent", _rule_requested_format_absent),
    ("R07", "instruction_far_from_request", _rule_instruction_far_from_request),
    ("R08", "source_below_measurement_floor", _rule_source_below_measurement_floor),
    ("R09", "source_little_effect", _rule_source_little_effect),
    ("R10", "request_conflicts_with_earlier_instructions", _rule_request_conflicts_with_earlier_instructions),
    ("R11", "output_stopped_length", _rule_output_stopped_length),
    ("R12", "run_to_run_drift", _rule_run_to_run_drift),
)
RULE_IDS = tuple(rule_id for rule_id, _name, _fn in RULE_REGISTRY)


# =========================================================================================== public API

def evaluate(run: Mapping[str, Any], *, comparison_run: "Mapping[str, Any] | None" = None,
            suppressed_rule_ids: Sequence[str] = (), generated_at: "str | None" = None,
            validate: bool = True) -> dict:
    """Run the full `RULE_REGISTRY` against `run` and build one `clozn.diagnosis-findings.v1` document.

    Pure and deterministic: the same `run` (and `comparison_run`, and the same `generated_at` override)
    always produces byte-identical output. Never raises for a malformed or evidence-poor run -- every rule
    reports its own `unavailable`/`pending` state instead; only a genuinely unexpected internal error
    escapes (there is no historical precedent for suppressing those here, unlike the individual rules'
    own engine-call boundaries in sibling `clozn.analysis` modules, since this module never calls a live
    engine at all).

    `suppressed_rule_ids` (a plain parameter, not a config file) marks the named rule_id(s) `status:
    "suppressed"` WITHOUT calling their evaluation logic -- `findings` still carries exactly one entry per
    `RULE_REGISTRY` member, in registry order, always.
    """
    suppressed = sorted({str(r) for r in suppressed_rule_ids if isinstance(r, str)} & set(RULE_IDS))
    run = run if isinstance(run, Mapping) else {}
    comparison_run = comparison_run if isinstance(comparison_run, Mapping) else None
    ctx = _Context(run, comparison_run)

    findings = []
    for rule_id, rule_name, rule_fn in RULE_REGISTRY:
        if rule_id in suppressed:
            findings.append(_finding_entry(
                rule_id, rule_name, status="suppressed",
                summary=f"{rule_id} was suppressed for this evaluation by caller request.",
                evidence=[]))
            continue
        entry = rule_fn(ctx)
        assert entry.get("rule_id") == rule_id, f"{rule_id} handler returned a mismatched rule_id"
        findings.append(entry)

    status_counts = {status: 0 for status in STATUS_VALUES}
    for entry in findings:
        status_counts[entry["status"]] += 1

    document: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at if generated_at is not None else _now_iso(),
        "run_id": ctx.run_id,
        "redacted": ctx.redacted,
        "rule_registry": [{"rule_id": rule_id, "rule_name": rule_name} for rule_id, rule_name, _fn in RULE_REGISTRY],
        "suppressed_rule_ids": suppressed,
        "findings": findings,
        "summary": {"status_counts": status_counts},
    }
    if ctx.comparison_run is not None:
        comparison_id = _str(_object(ctx.comparison_run).get("id"))
        if comparison_id:
            document["comparison_run_id"] = comparison_id
    if validate:
        schemas.validate(document)
    return document
