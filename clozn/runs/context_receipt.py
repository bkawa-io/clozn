"""Context-delivery receipts: what reached the model, what changed on the way, what was omitted, and
why generation stopped (feature 06, ``clozn.context-receipt.v1``).

SHAPE HISTORY -- READ THIS BEFORE TOUCHING A KEY NAME
-------------------------------------------------------
Every run recorded before this module's 2026-07-27 rewrite carries a *different*, pre-schema-seam shape
under the same ``context_receipt`` field: ``{"schema": "clozn.context_receipt.v1", "delivered": {label,
meaning, messages}, "survived": {label, meaning, assembled_messages, final_prompt}, "input_truncated",
"input_policy", "output_cut_off", "limits", "warnings"}``. Note the underscore in the old schema value
and the ``"schema"`` key name (not ``"schema_version"``) -- that document was never registered under
``clozn/schemas/`` and carries no version-compatibility promise (its own prior docstring called it
"Phase 2.4", an internal capture, not a released artifact). It is never rewritten in place: history
stands as written (see :func:`read_receipt`, which detects and reads both shapes without migrating
either).

This rewrite is ADDITIVE, not a replacement, for every key that pre-existing code still reads at runtime:
``survived`` (with ``assembled_messages``/``final_prompt``), ``limits``, ``output_cut_off``,
``input_truncated``, ``input_policy``, and receipt-level ``warnings`` all keep their old shape and meaning
unchanged -- ``clozn.runs.diagnosis``'s cutoff/slow-run findings and two existing tests
(``tests/test_ollama_instrumented.py``, ``tests/test_replay.py``) read them directly and must keep working
on runs recorded today. The one exception is the top-level ``delivered`` key: nothing outside this
module's own CLI (``clozn/cli/commands/context.py``) and test file ever read it as the old
``{label, meaning, messages}`` object, so it is repurposed to the new schema's segment-array meaning. Full
message/prompt TEXT is deliberately not duplicated into the new segment arrays -- see "segment identity"
below -- so this repurposing loses nothing: ``run["messages"]``/``run["assembled_messages"]``/
``run["final_prompt"]`` remain the source of full content, governed by the run-level redact/delete
lifecycle in ``clozn.runs.mutations``, not by this artifact.

SEGMENT IDENTITY -- the contract features 07/08/10 build on
-------------------------------------------------------------
A segment is one user-visible unit; v1 makes that exactly one chat message (clozn has no sub-message
attachment/document-chunk concept -- see the schema file's top-level description). Its id is
``"seg_" + sha256(role + "\\n" + content)[:16]``, computed ONCE at the delivered boundary and carried
through unchanged -- never recomputed from content at a later stage, because recomputing would break
stability exactly when a transformation touches the content, which is the one case an id needs to
survive. It is content-derived, not run- or position-scoped, on purpose: the same logical turn (e.g. an
unchanged system prompt, a repeated regression-suite question) recurring across two different runs gets
the SAME segment_id, which is what a cross-run comparison (feature 10: "yesterday's segment X is missing
today") needs. A disambiguating ``_N`` suffix handles an exact (role, content) pair repeating within one
run's own delivered list. Segment records are metadata only -- source type/label/order/byte count/hash/
reason/redaction state -- never message content; see the schema file's top-level description for why.

ASSEMBLED CURRENTLY EQUALS DELIVERED
-------------------------------------
Since the 2026-07-27 "delete memory entirely" refactor, every current code path sets
``assembled_messages = list(messages)`` verbatim (``clozn/server/substrates.py``) -- there is currently no
transformation between delivered and assembled to observe. :func:`build_context_receipt` still does the
real (content-hash) matching work rather than assuming identity, so it stays correct if a future assembly
layer ever does diverge again, and so the existing direct-API test that constructs a genuinely different
``assembled_messages`` (see ``tests/test_context_receipt.py``) is still handled honestly: a delivered
segment whose exact (role, content) does not reappear in ``assembled_messages`` is marked
``included: False``; no attempt is made to guess that it was "modified into" some specific assembled
segment, because clozn has no content-similarity signal to back that claim (evidence before narration).

TERMINATION AND OMISSION VOCABULARIES ARE INTENTIONALLY LARGER THAN CLOZN'S LIVE DETECTION
---------------------------------------------------------------------------------------------
The schema's ``termination.reason`` and segment ``reason`` enums include codes clozn cannot currently
produce (``stop_sequence``, ``content_filter``, ``timeout``, and most of the omission codes -- clozn is a
raw relay with no attachment/RAG concept and no server-side history trimming; overlong prompts are
rejected outright, never silently trimmed). They are defined for forward compatibility and are simply
never emitted by :func:`normalize_termination` or :func:`build_context_receipt` today. A reason code that
never fires is honest; inventing detection logic with nothing real to detect would not be.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import sys
from copy import deepcopy

OUTPUT_TRUNCATED = "output_truncated"

SCHEMA_VERSION = "clozn.context-receipt.v1"
LEGACY_SCHEMA = "clozn.context_receipt.v1"          # pre-2026-07-27; never schema-governed, never rewritten


# ``clozn_sources`` is deliberately metadata-only.  These values describe why
# the client says a precisely addressed source exists; they do not change how
# the message is rendered or sent to the model.  Keep this small closed
# vocabulary at the public boundary so a typo never turns into an invented
# provenance category in a Context Receipt.
SOURCE_PROVENANCE_KINDS = frozenset({
    "message", "retrieved_document", "tool_result", "system_instruction",
    "conversation_turn", "client_supplied",
})


# ---------------------------------------------------------------------------------------------------
# segment identity
# ---------------------------------------------------------------------------------------------------

def segment_id(role, content, *, occurrence: int = 0) -> str:
    """The stable, content-derived id for one (role, content) message -- see the module docstring."""
    text = f"{role}\n{content if isinstance(content, str) else ''}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"seg_{digest}" if occurrence == 0 else f"seg_{digest}_{occurrence}"


def _content_hash(content) -> str:
    text = content if isinstance(content, str) else ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _canonical_source_id(*, segment: str, unicode_range: list[int], byte_range: list[int],
                         content_sha256: str, provenance_kind: str,
                         client_source_id: str | None,
                         parent_source_id: str | None) -> str:
    """A stable exact-source identity, never a client-controlled identifier.

    ``segment`` commits the role + complete message text (and the duplicate
    occurrence where applicable); the range and source-content digest commit
    the selected bytes.  Client labels are intentionally excluded, while a
    client identity/provenance/structural parent is included: changing any of
    those facts changes the claimed source, not merely its display name.  A
    child commits its *canonical* parent identity (not merely the parent client
    label), so changes to any structural ancestor also change the child ID.
    """
    canonical = {
        "byte_range": byte_range,
        "client_source_id": client_source_id,
        "content_sha256": content_sha256,
        "parent_source_id": parent_source_id,
        "provenance_kind": provenance_kind,
        "segment_id": segment,
        "unicode_range": unicode_range,
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return "src_" + hashlib.sha256(encoded).hexdigest()[:24]


def _byte_offset(text: str, unicode_offset: int) -> int:
    return len(text[:unicode_offset].encode("utf-8"))


def _range_pair(value, *, upper: int) -> list[int] | None:
    if not (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return None
    start, end = int(value[0]), int(value[1])
    if start < 0 or end <= start or end > upper:
        return None
    return [start, end]


def _canonical_message_sources(message: dict, *, segment: str, message_index: int,
                               content: str) -> list[dict]:
    """Canonicalize the private journal metadata for one exact message.

    The OpenAI adapter validates this input before it reaches the journal.  We
    nevertheless re-derive every byte offset and digest here because receipt
    capture is also a public, directly callable seam.  A malformed direct
    caller gets no speculative source record rather than a receipt that later
    claims it can faithfully delete unknown bytes.
    """
    raw_sources = message.get("_clozn_sources")
    if raw_sources is None:
        # Pre-span callers' top-level source_id/source_label remain metadata on
        # the legacy seg_ root.  Do not manufacture a duplicate src_ source.
        raw_sources = []
    if not isinstance(raw_sources, list):
        return []

    candidates: list[dict] = []
    client_ids: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            return []
        # A legacy normalized request had no explicit exact range.  Its
        # identity remains the enclosing seg_ root for compatibility.
        if "unicode_range" not in raw:
            continue
        unicode_range = _range_pair(raw.get("unicode_range"), upper=len(content))
        if unicode_range is None:
            return []
        byte_range = [_byte_offset(content, unicode_range[0]), _byte_offset(content, unicode_range[1])]
        supplied_bytes = raw.get("byte_range")
        if supplied_bytes is not None and _range_pair(supplied_bytes, upper=len(content.encode("utf-8"))) != byte_range:
            return []
        client_source_id = raw.get("client_source_id", raw.get("source_id"))
        if client_source_id is not None:
            if not isinstance(client_source_id, str) or not client_source_id or client_source_id in client_ids:
                return []
            client_ids.add(client_source_id)
        label = raw.get("label", raw.get("source_label"))
        if label is not None and (not isinstance(label, str) or not label):
            return []
        provenance_kind = raw.get("provenance_kind", "message")
        if not isinstance(provenance_kind, str) or provenance_kind not in SOURCE_PROVENANCE_KINDS:
            return []
        parent_client_source_id = raw.get("parent_source_id")
        if parent_client_source_id is not None and (
            not isinstance(parent_client_source_id, str) or not parent_client_source_id
        ):
            return []
        selected = content[unicode_range[0]:unicode_range[1]]
        source_sha256 = _content_sha256(selected)
        item = {
            "segment_id": segment,
            "message_index": message_index,
            "unicode_range": unicode_range,
            "byte_range": byte_range,
            "content_sha256": source_sha256,
            "provenance_kind": provenance_kind,
        }
        if client_source_id is not None:
            item["client_source_id"] = client_source_id
        if label is not None:
            item["source_label"] = label
        if parent_client_source_id is not None:
            # Resolved after all sibling source IDs exist.
            item["_parent_client_source_id"] = parent_client_source_id
        candidates.append(item)

    source_by_client_id = {
        item["client_source_id"]: item for item in candidates if item.get("client_source_id")
    }
    def _mint(item: dict, active: set[int]) -> str | None:
        existing = item.get("source_id")
        if isinstance(existing, str):
            return existing
        marker = id(item)
        if marker in active:
            return None
        active.add(marker)
        parent_client_id = item.get("_parent_client_source_id")
        parent_source_id = None
        if parent_client_id is not None:
            parent = source_by_client_id.get(parent_client_id)
            if parent is None or parent is item:
                return None
            parent_start, parent_end = parent["unicode_range"]
            child_start, child_end = item["unicode_range"]
            if not (parent_start <= child_start and child_end <= parent_end):
                return None
            parent_source_id = _mint(parent, active)
            if parent_source_id is None:
                return None
            item["parent_source_id"] = parent_source_id
        source_id = _canonical_source_id(
            segment=segment, unicode_range=item["unicode_range"], byte_range=item["byte_range"],
            content_sha256=item["content_sha256"], provenance_kind=item["provenance_kind"],
            client_source_id=item.get("client_source_id"), parent_source_id=parent_source_id,
        )
        item["source_id"] = source_id
        active.remove(marker)
        return source_id

    seen_source_ids: set[str] = set()
    for item in candidates:
        if _mint(item, set()) is None or item["source_id"] in seen_source_ids:
            return []
        seen_source_ids.add(item["source_id"])
        item.pop("_parent_client_source_id", None)

    # An overlap has one safe interpretation only: one source is an explicit
    # structural descendant of the other.  Do not infer ancestry from ranges.
    by_id = {item["source_id"]: item for item in candidates}

    def _is_ancestor(ancestor_id: str, child: dict) -> bool:
        parent_id = child.get("parent_source_id")
        seen: set[str] = set()
        while isinstance(parent_id, str) and parent_id and parent_id not in seen:
            if parent_id == ancestor_id:
                return True
            seen.add(parent_id)
            parent = by_id.get(parent_id)
            parent_id = parent.get("parent_source_id") if isinstance(parent, dict) else None
        return False

    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            left_start, left_end = left["unicode_range"]
            right_start, right_end = right["unicode_range"]
            if max(left_start, right_start) >= min(left_end, right_end):
                continue
            left_contains_right = left_start <= right_start and right_end <= left_end
            right_contains_left = right_start <= left_start and left_end <= right_end
            if not (
                (left_contains_right and _is_ancestor(left["source_id"], right))
                or (right_contains_left and _is_ancestor(right["source_id"], left))
            ):
                return []
    return candidates


def _delivered_segments(messages) -> list[dict]:
    """One segment per message in `messages`, in order, with a stable, content-derived id.

    ``source_id``/``source_label`` are private journal metadata produced only by the explicit
    ``clozn_sources`` request extension.  They never ride into the rendered prompt.
    """
    seen: dict[tuple[str, str], int] = {}
    out = []
    for index, message in enumerate(messages or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        key = (role, content if isinstance(content, str) else "")
        occurrence = seen.get(key, 0)
        seen[key] = occurrence + 1
        text = content if isinstance(content, str) else ""
        explicit_label = message.get("source_label")
        explicit_id = message.get("source_id")
        segment = {
            "segment_id": segment_id(role, content, occurrence=occurrence),
            "source_type": "message",
            "source_label": (
                explicit_label if isinstance(explicit_label, str) and explicit_label else role
            ),
            "original_order": index,
            "delivered_bytes": len(text.encode("utf-8")),
            "content_hash": _content_hash(content),
        }
        if isinstance(explicit_id, str) and explicit_id:
            segment["client_source_id"] = explicit_id
        sources = _canonical_message_sources(
            message, segment=segment["segment_id"], message_index=index, content=text,
        )
        if sources:
            # Precise source metadata is journal/receipt evidence only.  The
            # original message content remains the sole prompt-rendering input.
            segment["sources"] = sources
        out.append(segment)
    return out


def _assembled_segments(delivered_segments, delivered_messages, assembled_messages):
    """Match each assembled message back to the delivered segment with the SAME (role, content),
    consuming delivered occurrences in order so exact repeats match one-to-one. Returns
    (assembled_segment_dicts, set-of-matched-delivered-segment-ids)."""
    queues: dict[tuple[str, str], deque] = {}
    for seg, msg in zip(delivered_segments, (m for m in (delivered_messages or []) if isinstance(m, dict))):
        role = str(msg.get("role") or "")
        content = msg.get("content")
        key = (role, content if isinstance(content, str) else "")
        queues.setdefault(key, deque()).append(seg)

    matched: set[str] = set()
    seen: dict[tuple[str, str], int] = {}
    out = []
    for index, message in enumerate(assembled_messages or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        key = (role, content if isinstance(content, str) else "")
        queue = queues.get(key)
        matched_segment = None
        if queue:
            matched_segment = queue.popleft()
            sid = matched_segment["segment_id"]
            matched.add(sid)
        else:
            # No delivered segment has this exact content -- a genuinely new segment (or a modified one
            # clozn has no similarity signal to trace back to a specific delivered id; see module docstring).
            occurrence = seen.get(key, 0)
            seen[key] = occurrence + 1
            sid = segment_id(role, content, occurrence=occurrence)
        assembled = {
            "segment_id": sid,
            "source_type": "message",
            "source_label": (
                matched_segment.get("source_label", role) if matched_segment is not None else role
            ),
            "original_order": index,
            "included": True,
            "content_hash": _content_hash(content),
        }
        if matched_segment is not None and matched_segment.get("client_source_id"):
            assembled["client_source_id"] = matched_segment["client_source_id"]
        if matched_segment is not None and isinstance(matched_segment.get("sources"), list):
            # Exact (role, content) matching above proves these code-point and
            # UTF-8 offsets still identify the same bytes.  Update only the
            # view-local message index; canonical src_ IDs remain untouched.
            sources = []
            for source in matched_segment["sources"]:
                if not isinstance(source, dict):
                    continue
                copied = deepcopy(source)
                copied["message_index"] = index
                copied["segment_id"] = sid
                sources.append(copied)
            if sources:
                assembled["sources"] = sources
        out.append(assembled)
    return out, matched


# ---------------------------------------------------------------------------------------------------
# termination normalization
# ---------------------------------------------------------------------------------------------------

def normalize_termination(finish_reason, error, meta, limits) -> dict | None:
    """Best-effort normalization of why generation stopped into the schema's controlled vocabulary.

    Returns None when there is nothing to classify (no finish_reason and no error recorded) -- that is
    an *absence*, distinct from "unknown", which is reserved for "something happened but its cause
    could not be classified". Every branch that DOES classify keeps the raw backend value in
    `reason_raw` alongside the normalized code, per the spec's "store raw backend reason alongside
    normalized reason".

    Sources, all already captured elsewhere in the run (see the module docstring for what's live vs.
    vocabulary-only):
      * `meta["stream_failure"]` -- "client_disconnected" | "worker_disconnected"
        (clozn/server/app.py, sse.py, ndjson.py). NOTE: this reads the structured meta field, not
        `run["error"]`'s free-text string -- clozn.runs.diagnosis._cutoff_finding compares `error` to
        the bare literal "client_disconnected", which the actual stored error text
        ("client disconnected mid-stream: ...") never equals; that comparison is not repeated here.
      * `finish_reason` -- "length" | "stop" | "tool_calls" (clozn/server/routes/openai.py); the
        engine's own raw values ("eos"/"length"/"steps_exhausted") are already collapsed to "stop"/
        "length" before this layer (clozn/runs/trace.py), so `reason_raw` reflects clozn's own value,
        not the engine's -- the finer distinction cannot be recovered here.
      * `limits` -- prompt_tokens/context_window_tokens/requested_max_tokens/generated_tokens, to
        separate "the context window was reached" from "the requested output cap was reached" the same
        way clozn.runs.diagnosis._cutoff_finding already does narratively.
    """
    meta = meta if isinstance(meta, dict) else {}
    limits = limits if isinstance(limits, dict) else {}
    generated = limits.get("generated_tokens")
    base: dict = {}
    if isinstance(generated, int):
        base["generated_tokens"] = generated
    source = meta.get("finish_reason_source")
    if isinstance(source, str) and source:
        base["source"] = source
    raw_finish_reason = meta.get("finish_reason_raw")
    if not isinstance(raw_finish_reason, str) or not raw_finish_reason:
        raw_finish_reason = finish_reason

    stream_failure = meta.get("stream_failure")
    if stream_failure == "client_disconnected":
        return {**base, "reason": "client_cancelled", "reason_raw": stream_failure}
    if stream_failure == "worker_disconnected":
        return {**base, "reason": "worker_error", "reason_raw": stream_failure}

    if isinstance(error, str) and error:
        return {**base, "reason": "worker_error", "reason_raw": error}

    if finish_reason == "tool_calls":
        return {**base, "reason": "tool_call", "reason_raw": raw_finish_reason}

    if finish_reason == "length":
        prompt = limits.get("prompt_tokens")
        window = limits.get("context_window_tokens")
        maximum = limits.get("requested_max_tokens")
        hit_context = (isinstance(prompt, int) and isinstance(generated, int) and isinstance(window, int)
                       and prompt + generated >= window)
        hit_max = isinstance(generated, int) and isinstance(maximum, int) and generated >= maximum
        # context_limit takes priority when provably true, even if max_tokens is ALSO provably true: a
        # user who did not get their full requested output because the conversation was too long is the
        # more surprising and more actionable fact of the two.
        reason = "context_limit" if hit_context else ("max_tokens" if hit_max else "unknown")
        return {**base, "reason": reason, "reason_raw": raw_finish_reason}

    if finish_reason == "stop":
        return {**base, "reason": "eos", "reason_raw": raw_finish_reason}

    if isinstance(finish_reason, str) and finish_reason:
        return {**base, "reason": "unknown", "reason_raw": finish_reason}

    return None


# ---------------------------------------------------------------------------------------------------
# legacy (pre-2026-07-27) receipt logic -- unchanged behavior, kept for the additive top-level fields
# ---------------------------------------------------------------------------------------------------

def cutoff_warning(finish_reason, meta=None) -> dict | None:
    """Return one structured warning for a proven output cutoff, else ``None``."""
    if finish_reason != "length":
        return None
    meta = meta if isinstance(meta, dict) else {}
    warning = {
        "code": OUTPUT_TRUNCATED,
        "severity": "warning",
        "message": ("generation stopped at the output/context token budget; "
                    "the reply may be incomplete"),
    }
    maximum = meta.get("max_tokens")
    if isinstance(maximum, int) and maximum > 0:
        warning["requested_max_tokens"] = maximum
    return warning


def warnings_for(finish_reason, meta=None) -> list[dict]:
    warning = cutoff_warning(finish_reason, meta)
    return [warning] if warning else []


# ---------------------------------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------------------------------

def _apply_privacy(doc: dict, tier: str) -> dict:
    """Trim `doc` (already fully built, as if tier=="full") down to `tier`. "off" returns the minimal
    required-fields-only stub; "metadata_only"/"hashes_only" drop duplicated full TEXT (and, for
    hashes_only, segment labels/byte counts too) while always keeping hashes -- "store hashes even when
    privacy settings prevent full content retention" is the spec's own requirement."""
    if tier == "off":
        return {"schema_version": doc["schema_version"], "run_id": doc["run_id"], "privacy": "off"}

    out = dict(doc)
    if tier in ("metadata_only", "hashes_only"):
        survived = out.get("survived")
        if isinstance(survived, dict):
            survived = dict(survived)
            survived.pop("final_prompt", None)
            survived.pop("assembled_messages", None)
            survived["content_withheld_by_privacy_tier"] = tier
            out["survived"] = survived
    if tier == "hashes_only":
        for key in ("delivered", "assembled"):
            segments = out.get(key)
            if not isinstance(segments, list):
                continue
            trimmed = []
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                keep = {k: seg[k] for k in ("segment_id", "source_type", "original_order", "content_hash",
                                            "reason", "included") if k in seg}
                keep["redaction_state"] = "hash_only"
                trimmed.append(keep)
            out[key] = trimmed
    else:
        redaction_state = "full" if tier == "full" else "redacted"
        for key in ("delivered", "assembled"):
            segments = out.get(key)
            if not isinstance(segments, list):
                continue
            out[key] = [{**seg, "redaction_state": redaction_state} if isinstance(seg, dict) else seg
                        for seg in segments]
    rendered = out.get("rendered")
    if isinstance(rendered, dict):
        rendered = dict(rendered)
        rendered["content_available"] = (
            tier == "full"
            and isinstance((out.get("survived") or {}).get("final_prompt"), str)
        )
        out["rendered"] = rendered
    out["privacy"] = tier
    return out


# ---------------------------------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------------------------------

def build_context_receipt(*, messages=None, assembled_messages=None, final_prompt=None,
                          finish_reason=None, meta=None, trace=None, run_id=None, identity=None,
                          error=None, privacy=None) -> dict:
    """Build a no-inference receipt from the evidence captured for one run.

    Never raises: a schema-validation failure (a builder bug, or a genuinely missing `run_id`) is caught,
    reported on stderr, and returned as a best-effort document carrying `schema_validation_error` rather
    than propagating -- callers include `clozn.runs.store.record()`, whose own outer try/except would
    otherwise silently drop the ENTIRE run over a receipt-only defect. A broken receipt must cost its own
    field, never the run (same principle clozn.runs.identity_providers states for identity facets).

    """
    meta = meta if isinstance(meta, dict) else {}
    trace = trace if isinstance(trace, dict) else {}
    identity = identity if isinstance(identity, dict) else {}
    messages_list = messages if isinstance(messages, list) else []

    # ---- legacy-shaped fields, unchanged behavior (see module docstring) ----
    survived = {
        "label": "survived",
        "meaning": "post-assembly input retained as evidence of what reached generation",
        "assembled_messages": (list(assembled_messages) if isinstance(assembled_messages, list) else None),
        "final_prompt": final_prompt if isinstance(final_prompt, str) else None,
    }
    prompt_tokens = meta.get("prompt_tokens")
    n_ctx = meta.get("n_ctx")
    maximum = meta.get("max_tokens")
    generated = len(trace.get("tokens") or []) if isinstance(trace.get("tokens"), list) else None
    limits = {
        "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) else None,
        "context_window_tokens": n_ctx if isinstance(n_ctx, int) else None,
        "requested_max_tokens": maximum if isinstance(maximum, int) else None,
        "generated_tokens": generated,
    }
    warning = cutoff_warning(finish_reason, meta)

    # ---- new schema fields ----
    delivered_segments = _delivered_segments(messages_list)
    doc: dict = {"schema_version": SCHEMA_VERSION, "run_id": run_id if isinstance(run_id, str) else ""}

    template_fingerprint = identity.get("template_fingerprint")
    if isinstance(template_fingerprint, str) and template_fingerprint:
        doc["template_fingerprint"] = template_fingerprint
        doc["tokenizer_conflated_with_template"] = True

    if isinstance(n_ctx, int):
        doc["context_window_tokens"] = n_ctx
    if isinstance(maximum, int):
        doc["reserved_output_tokens"] = maximum

    transformations: list[dict] = []
    omissions: list[dict] = []
    if isinstance(assembled_messages, list):
        assembled_segments, matched_ids = _assembled_segments(
            delivered_segments, messages_list, assembled_messages)
        for seg in delivered_segments:
            seg["included"] = seg["segment_id"] in matched_ids
        doc["assembled"] = assembled_segments
    doc["delivered"] = delivered_segments

    rendered: dict = {}
    if isinstance(final_prompt, str) and final_prompt:
        rendered_bytes = final_prompt.encode("utf-8")
        rendered["sha256"] = hashlib.sha256(rendered_bytes).hexdigest()
        rendered["bytes"] = len(rendered_bytes)
        rendered["content_available"] = True
        if isinstance(template_fingerprint, str) and template_fingerprint:
            rendered["template_fingerprint"] = template_fingerprint
        all_ids = [seg["segment_id"] for seg in delivered_segments]
        if all_ids:
            transformations.append({
                "reason": "template_transformed",
                "segment_ids": all_ids,
                "detail": ("all delivered messages were rendered through the model's chat template "
                           "into the exact prompt string; see rendered.sha256"),
            })
    if isinstance(prompt_tokens, int):
        rendered["tokens"] = prompt_tokens
        rendered["token_count"] = prompt_tokens
        rendered["estimated"] = False           # the engine's own gen_started frame reports this exactly
    if rendered:
        doc["rendered"] = rendered

    doc["omissions"] = omissions
    doc["transformations"] = transformations

    termination = normalize_termination(finish_reason, error, meta, limits)
    if termination is not None:
        doc["termination"] = termination

    # ---- assemble, validate (never fatally), privacy-trim ----
    doc["survived"] = survived
    doc["limits"] = limits
    doc["input_truncated"] = False
    doc["input_policy"] = "overlong prompts are rejected, not silently truncated"
    doc["output_cut_off"] = warning is not None
    doc["warnings"] = [warning] if warning else []

    tier = privacy if privacy in ("full", "metadata_only", "hashes_only", "off") else None
    if tier is None:
        from clozn.runs import receipt_privacy
        tier = receipt_privacy.tier()
    doc = _apply_privacy(doc, tier)

    try:
        from clozn import schemas
        schemas.validate(doc)
    except Exception as exc:                    # noqa: BLE001 -- a receipt bug must cost the receipt, not the run
        doc["schema_validation_error"] = f"{type(exc).__name__}: {exc}"
        print(f"clozn: context receipt for run {run_id!r} failed schema validation: {exc}",
             file=sys.stderr)
    return doc


# ---------------------------------------------------------------------------------------------------
# reading -- dual-shape support for legacy (pre-2026-07-27) and new documents
# ---------------------------------------------------------------------------------------------------

def read_receipt(run: dict) -> dict:
    """A uniform view of `run`'s stored context receipt, regardless of which shape it is in.

    Returns {"shape": "new" | "legacy" | "absent" | "unrecognized", "receipt": <stored dict or {}>}.
    Legacy documents are read, never migrated in place -- see the module docstring's migration policy.
    """
    stored = run.get("context_receipt") if isinstance(run, dict) else None
    if not isinstance(stored, dict) or not stored:
        return {"shape": "absent", "receipt": {}}
    schema_version = stored.get("schema_version")
    if isinstance(schema_version, str) and schema_version.startswith("clozn.context-receipt."):
        return {"shape": "new", "receipt": stored}
    if stored.get("schema") == LEGACY_SCHEMA:
        return {"shape": "legacy", "receipt": stored}
    return {"shape": "unrecognized", "receipt": stored}
