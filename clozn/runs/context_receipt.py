"""Context-delivery receipts: what reached the model, what changed on the way, what was omitted, and
why generation stopped (``clozn.context-receipt.v1``).

The receipt is a schema-governed, metadata-only evidence artifact. ``delivered`` and ``assembled`` track
segments, ``rendered`` records the inference-boundary digest and prompt-token count, and ``termination``
records why generation stopped. Full message and prompt text remains on the run record under
``messages``, ``assembled_messages``, and ``final_prompt`` so one redaction lifecycle governs every
retained copy.

SEGMENT IDENTITY -- the contract features 07/08/10 build on
-------------------------------------------------------------
A segment is one user-visible message. Markdown-aware source spans may partition
eligible context inside that message for analysis. Its id is
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
Every current code path sets ``assembled_messages = list(messages)`` verbatim
(``clozn/server/substrates.py``) -- there is currently no
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
                         content_sha256: str, provenance_kind: str = "message") -> str:
    """Return the stable identity of one server-derived exact source span."""
    canonical = {
        "byte_range": byte_range,
        "content_sha256": content_sha256,
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
    """Canonicalize the server-derived flat source partition for one message."""
    raw_sources = message.get("_clozn_sources")
    if raw_sources is None:
        raw_sources = []
    if not isinstance(raw_sources, list):
        return []

    candidates: list[dict] = []
    for raw in raw_sources:
        if not isinstance(raw, dict):
            return []
        if "unicode_range" not in raw:
            return []
        unicode_range = _range_pair(raw.get("unicode_range"), upper=len(content))
        if unicode_range is None:
            return []
        byte_range = [_byte_offset(content, unicode_range[0]), _byte_offset(content, unicode_range[1])]
        selected = content[unicode_range[0]:unicode_range[1]]
        source_sha256 = _content_sha256(selected)
        item = {
            "segment_id": segment,
            "message_index": message_index,
            "unicode_range": unicode_range,
            "byte_range": byte_range,
            "content_sha256": source_sha256,
            "provenance_kind": "message",
        }
        item["source_id"] = _canonical_source_id(
            segment=segment, unicode_range=item["unicode_range"], byte_range=item["byte_range"],
            content_sha256=item["content_sha256"],
        )
        candidates.append(item)

    ordered = sorted(candidates, key=lambda item: item["unicode_range"])
    for left, right in zip(ordered, ordered[1:]):
        if left["unicode_range"][1] > right["unicode_range"][0]:
            return []
    if len({item["source_id"] for item in ordered}) != len(ordered):
        return []
    return ordered


def _delivered_segments(messages) -> list[dict]:
    """One segment per message, plus any server-derived flat source spans."""
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
        segment = {
            "segment_id": segment_id(role, content, occurrence=occurrence),
            "source_type": "message",
            "source_label": role,
            "original_order": index,
            "delivered_bytes": len(text.encode("utf-8")),
            "content_hash": _content_hash(content),
        }
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

def normalize_termination(finish_reason, error, meta, token_facts) -> dict | None:
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
      * `token_facts` -- prompt_tokens/context_window_tokens/requested_max_tokens/generated_tokens, to
        separate "the context window was reached" from "the requested output cap was reached" the same
        way clozn.runs.diagnosis._cutoff_finding already does narratively.
    """
    meta = meta if isinstance(meta, dict) else {}
    token_facts = token_facts if isinstance(token_facts, dict) else {}
    generated = token_facts.get("generated_tokens")
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
        prompt = token_facts.get("prompt_tokens")
        window = token_facts.get("context_window_tokens")
        maximum = token_facts.get("requested_max_tokens")
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
# cutoff evidence
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
    """Trim a metadata-only receipt down to ``tier``.

    Run content is never copied into this artifact. ``hashes_only`` removes descriptive segment
    metadata while retaining stable identities and hashes; ``off`` retains only the receipt marker.
    """
    if tier == "off":
        return {"schema_version": doc["schema_version"], "run_id": doc["run_id"], "privacy": "off"}

    out = dict(doc)
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
    # Automatic context-unit descriptors are added only to this detached
    # journal/evidence view.  The caller's message list has already crossed
    # the model boundary by the time receipt capture runs, and is never
    # mutated.  Keeping this at the receipt seam also makes direct receipt
    # construction and store-backed runs obey the same canonicalization path.
    from . import context_units
    receipt_messages = context_units.annotate_auto_sources(messages_list)

    prompt_tokens = meta.get("prompt_tokens")
    n_ctx = meta.get("n_ctx")
    maximum = meta.get("max_tokens")
    generated = len(trace.get("tokens") or []) if isinstance(trace.get("tokens"), list) else None
    token_facts = {
        "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) else None,
        "context_window_tokens": n_ctx if isinstance(n_ctx, int) else None,
        "requested_max_tokens": maximum if isinstance(maximum, int) else None,
        "generated_tokens": generated,
    }
    delivered_segments = _delivered_segments(receipt_messages)
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
            delivered_segments, receipt_messages, assembled_messages)
        for seg in delivered_segments:
            seg["included"] = seg["segment_id"] in matched_ids
        doc["assembled"] = assembled_segments
    doc["delivered"] = delivered_segments

    rendered: dict = {}
    if isinstance(final_prompt, str) and final_prompt:
        rendered_bytes = final_prompt.encode("utf-8")
        rendered["sha256"] = hashlib.sha256(rendered_bytes).hexdigest()
        rendered["bytes"] = len(rendered_bytes)
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
        rendered["estimated"] = False           # the engine's own gen_started frame reports this exactly
    if rendered:
        doc["rendered"] = rendered

    doc["omissions"] = omissions
    doc["transformations"] = transformations

    termination = normalize_termination(finish_reason, error, meta, token_facts)
    if termination is not None:
        doc["termination"] = termination

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
# reading
# ---------------------------------------------------------------------------------------------------

def read_receipt(run: dict) -> dict:
    """Return the canonical stored receipt or an explicit absence/error shape.

    Returns ``{"shape": "recorded" | "absent" | "unrecognized", "receipt": ...}``.
    Only schema-governed ``clozn.context-receipt.*`` documents are readable receipts.
    """
    stored = run.get("context_receipt") if isinstance(run, dict) else None
    if not isinstance(stored, dict) or not stored:
        return {"shape": "absent", "receipt": {}}
    schema_version = stored.get("schema_version")
    if schema_version == SCHEMA_VERSION:
        try:
            from clozn import schemas
            schemas.validate(stored, SCHEMA_VERSION)
        except schemas.ValidationError:
            return {"shape": "unrecognized", "receipt": stored}
        return {"shape": "recorded", "receipt": stored}
    return {"shape": "unrecognized", "receipt": stored}
