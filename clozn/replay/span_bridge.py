"""span_bridge.py -- the arbitrary-span bridge (C3 core slice, item 3).

B3 span addressing (`clozn.runs.text_span_addresses`) already gives every passage in a run a stable,
content-addressed identity: an `address_id`, a `(message_index-shaped) native_ref`, and a
`resolution.canonical` block carrying Unicode-code-point offsets plus SHA-256 hashes of both the
canonical basis text and the selected span. What it does NOT do is turn that address back into
something `clozn.replay.replay.replay()` can execute -- `replay()`'s only content-level ablation lever is
`changes["exclude_sections"]`, a list of NAMED sections against the run's persisted `sections` manifest.
An arbitrary address is neither named nor guaranteed to align with a section boundary.

This module is the bridge. `resolve_span_address()` re-derives a FRESH `clozn.text-span-addresses.v1`
document from the run's CURRENT stored content (never a cached/stale one) and looks the requested
`address_id` up in it. Because `address_id` is itself a hash over `{relation_key, native_ref,
canonical.basis_sha256, canonical.span_sha256, canonical.start, canonical.end}` (see
`text_span_addresses._address_id`), a content change since the id was minted changes the id too -- so a
stale id simply is not found in the fresh rebuild, and this function refuses
(`span_address_not_found_or_drifted`) rather than resolving to the WRONG text at the same coordinates.
This is deliberately one merged reason, not two: telling "this id never existed" apart from "the content
at this same logical spot moved" would need the relation_key reversed, which is a hash and not
reversible from the id alone. Both cases are equally disqualifying, so a caller a re-run
`clozn text-span-addresses RUN_ID` (or the investigation surface) gets a fresh id either way.

Spans anchored to an ordinary chat message (`native_ref.collection` in `{"run.messages",
"context_receipt.delivered"}`) resolve to a `message_index` this module can hand to `excise_spans()`,
via either of the manifest's two native-id shapes: the positional `f"message-{index}"` fallback (parsed
directly out of the id, no lookup needed -- pre-schema/legacy runs, which never stamp a `segment_id`), or
an opaque `segment_id` -- looked up against a `segment_id -> (message_index, content_hash)` index this
module builds by READING the run's OWN context receipt (`_segment_id_index()`, via
`clozn.runs.context_receipt.read_receipt`; the index reuses the receipt's own `original_order` field,
which is a message-list position recorded once at delivery time -- see that module's "segment identity"
section). A `segment_id` lookup is a lookup, not a proof: before trusting it, `_resolved_span_from_address()`
re-hashes the message CURRENTLY sitting at the resolved index (`clozn.runs.context_receipt._content_hash`,
reused rather than reimplemented -- two content-hash implementations that could drift apart would defeat
the purpose) and compares it against that segment's own recorded `content_hash`, refusing rather than
silently mis-mapping when the two disagree, or when the receipt records no usable hash to check against in
the first place. This makes the segment_id path self-verifying: a stale or corrupted `original_order`
almost always points at content with a different hash, and gets caught here rather than producing a
confident, wrong ablation. Two kinds are refused by construction, honestly, never guessed, regardless of
native-id shape:
  * `rendered_prompt_segment` (basis `rendered_prompt`, i.e. `final_prompt`) -- `replay()`'s only surface
    is a MESSAGE list (`chat(messages, ...)`), never a raw-prompt override; that is fork.py's territory,
    per replay.py's own module docstring. Splicing here would require inventing a capability replay does
    not have.
  * `answer_span` (basis `scored_answer`) -- a piece of the model's OWN reply, not prompt content. There
    is nothing to ablate out of a PROMPT here; that is a different kind of question this slice does not
    answer.

`excise_spans()` is the OTHER half: given one or more resolved spans, build a NEW messages list with each
span's [start, end) removed or replaced, right-to-left within a message so an earlier span's offsets in
the SAME message are never invalidated by a later span's removal -- exactly `replay.py`'s own
`_strip_message_parts` algorithm, generalized from a NAMED section's `parts` to an arbitrary offset list
(this module does not import that private helper; the two are independent, intentionally small,
implementations of the identical splice rule, kept apart because one is keyed by section name against a
persisted manifest and the other by raw offsets against a resolved address -- forcing them through one
shared function would need a manifest-shaped wrapper for every arbitrary-span caller for no shared
benefit). The result is an ordinary message list `clozn.replay.replay.replay()` already knows how to
consume via its `messages_override` parameter -- no change to replay.py itself was needed.

`pick_random_control_span()` builds the ONE piece of new machinery C3's five-arm-style discipline needs
that `clozn.receipts.forced`'s existing `_matched_length_filler` does not cover: a same-LENGTH,
NON-OVERLAPPING window elsewhere in the SAME basis text, chosen deterministically (SHA-256 over canonical
JSON, never Python's process-randomized `hash()` -- mirrors `clozn.analysis.causal_bisect.
_derive_single_site_seed`'s own reasoning) so the "did ANY similarly-sized change do this" control is
reproducible from the run id and the target span alone. `None` when the basis text is too short to fit a
disjoint window of the same length -- never a guessed overlapping or truncated one.

Stdlib-only, model-free, fully unit-testable against fixture run dicts -- no substrate, no generation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


_ID_MESSAGE_RE_PREFIX = "message-"


def _segment_id_index(run: dict) -> dict:
    """`segment_id -> (message_index, content_hash)` for every delivered segment on a NEW-shape context
    receipt (`clozn.runs.context_receipt.read_receipt` shape == "new"; legacy/absent/unrecognized shapes
    return `{}` -- those never stamp a `segment_id` in the first place, see
    `clozn.runs.text_span_addresses._context_projection`).

    `message_index` reuses the receipt's own `original_order` (falling back to the segment's position
    within `delivered` when `original_order` is missing/invalid), MIRRORING
    `clozn.runs.text_span_addresses._context_projection`'s own identical fallback -- so this index agrees
    with the message_index already baked into each address's canonical offsets, rather than computing a
    second, independently-derived mapping that could silently drift from it.

    `content_hash` is the segment's own recorded hash (`clozn.runs.context_receipt._content_hash`) -- kept
    alongside so `_resolved_span_from_address` can verify the message actually sitting at that index still
    matches before trusting this lookup. A lookup alone is not a proof (see this module's docstring)."""
    from clozn.runs.context_receipt import read_receipt
    view = read_receipt(run)
    if view["shape"] != "new":
        return {}
    delivered = view["receipt"].get("delivered")
    if not isinstance(delivered, list):
        return {}
    index: dict = {}
    for fallback_index, segment in enumerate(delivered):
        if not isinstance(segment, dict):
            continue
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            continue
        message_index = segment.get("original_order")
        if not isinstance(message_index, int) or isinstance(message_index, bool) or message_index < 0:
            message_index = fallback_index
        content_hash = segment.get("content_hash")
        index[segment_id] = (
            message_index,
            content_hash if isinstance(content_hash, str) and content_hash else None,
        )
    return index


def _parse_message_index(native_ref: dict, segment_index: dict) -> "int | None":
    """`native_ref["id"]` -> a message-list index. Two native-id shapes: the manifest's own positional
    fallback (`f"message-{index}"`, parsed directly, never present alongside a `segment_id`), or an opaque
    `segment_id`, looked up in `segment_index` (built by `_segment_id_index()` from the run's OWN context
    receipt -- see that function's and this module's docstrings). None when neither resolves, including a
    malformed "message-" id or a `segment_id` this run's receipt does not (or no longer) recognize --
    never guessed. Note this only resolves the INDEX; `_resolved_span_from_address` still verifies the
    segment_id path's content_hash before trusting it -- this function alone is not the safety property."""
    segment_id = native_ref.get("segment_id")
    if isinstance(segment_id, str) and segment_id:
        entry = segment_index.get(segment_id)
        return entry[0] if entry is not None else None
    raw_id = native_ref.get("id")
    if not isinstance(raw_id, str) or not raw_id.startswith(_ID_MESSAGE_RE_PREFIX):
        return None
    tail = raw_id[len(_ID_MESSAGE_RE_PREFIX):]
    if not tail.isdigit():
        return None
    return int(tail)


def _resolved_span_from_address(run: dict, address: dict, segment_index: dict) -> dict:
    """One already-matched, already-content-fresh address -> `{"ok": True, "span": ResolvedSpan}` or
    `{"ok": False, "reason": Reason}`. Never raises; every failure mode is a typed reason. `segment_index`
    is `_segment_id_index(run)`, built once by the caller and threaded through rather than rebuilt per
    address."""
    kind = address.get("kind")
    if kind not in ("delivered_message", "attached_source_span"):
        return {"ok": False, "reason": _reason(
            "span_basis_unsupported",
            f"span kind {kind!r} has no prompt-message basis this bridge can splice through "
            "replay()'s message-list surface (rendered_prompt_segment/answer_span/claim are out of "
            "scope for a prompt-span ablation in this slice)")}

    native_ref = address.get("native_ref") if isinstance(address.get("native_ref"), dict) else {}
    collection = native_ref.get("collection")
    message_index = None
    if collection in ("influence.prompt_spans", "influence.prompt_sources"):
        # Influence Query exposes projected IDs for measured fine/coarse source spans.  Those
        # addresses are authored against the source text, while replay needs the corresponding
        # message-list slot.  Resolve that slot only when the explicit source identity maps to one
        # message whose current content hash is the same canonical basis hash; never guess by index.
        source_id = native_ref.get("client_source_id")
        if not isinstance(source_id, str) or not source_id:
            influence = run.get("influence_map") if isinstance(run.get("influence_map"), dict) else {}
            source_records = influence.get("prompt_sources") if isinstance(influence, dict) else []
            source_keys = {native_ref.get("id"), native_ref.get("parent_id")}
            for source in source_records if isinstance(source_records, list) else []:
                if not isinstance(source, dict) or source.get("id") not in source_keys:
                    continue
                candidate = source.get("client_source_id")
                if isinstance(candidate, str) and candidate:
                    source_id = candidate
                    break
        messages = run.get("messages") if isinstance(run.get("messages"), list) else []
        matches = [
            index for index, message in enumerate(messages)
            if isinstance(message, dict) and isinstance(source_id, str) and source_id
            and message.get("source_id") == source_id
        ]
        if not matches:
            resolution = address.get("resolution") if isinstance(address.get("resolution"), dict) else {}
            canonical = resolution.get("canonical") if isinstance(resolution.get("canonical"), dict) else {}
            basis_sha256 = canonical.get("basis_sha256")
            if isinstance(basis_sha256, str) and basis_sha256:
                matches = [
                    index for index, message in enumerate(messages)
                    if isinstance(message, dict)
                    and isinstance(message.get("content"), str)
                    and hashlib.sha256(message["content"].encode("utf-8")).hexdigest() == basis_sha256
                ]
        if len(matches) != 1:
            return {"ok": False, "reason": _reason(
                "span_message_index_unresolvable",
                "the measured source span does not map to exactly one recorded message")}
        message_index = matches[0]
    elif collection not in ("run.messages", "context_receipt.delivered"):
        return {"ok": False, "reason": _reason(
            "span_basis_unsupported",
            f"native_ref.collection {collection!r} is not an ordinary chat message")}
    else:
        message_index = _parse_message_index(native_ref, segment_index)
    if message_index is None:
        return {"ok": False, "reason": _reason(
            "span_message_index_unresolvable",
            "this span's native reference uses an opaque segment id (or an unrecognized id shape) that "
            "this slice's bridge cannot map back to a message-list index")}

    resolution = address.get("resolution") if isinstance(address.get("resolution"), dict) else {}
    state = resolution.get("state")
    if state not in ("exact", "metadata_only"):
        return {"ok": False, "reason": _reason(
            "span_unavailable",
            f"span resolution state is {state!r} ({resolution.get('reason') or 'no further reason recorded'})")}

    canonical = resolution.get("canonical") if isinstance(resolution.get("canonical"), dict) else {}
    start, end = canonical.get("start"), canonical.get("end")
    basis_sha256 = canonical.get("basis_sha256")
    if (not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool) or end < start):
        return {"ok": False, "reason": _reason(
            "span_unavailable", "the resolved span carries no usable integer offsets")}

    messages = run.get("messages") if isinstance(run.get("messages"), list) else []
    if not (0 <= message_index < len(messages)) or not isinstance(messages[message_index], dict):
        return {"ok": False, "reason": _reason(
            "span_unavailable", f"message index {message_index} no longer exists on this run")}
    content = messages[message_index].get("content")
    content = content if isinstance(content, str) else ""
    if end > len(content):
        return {"ok": False, "reason": _reason(
            "span_unavailable", "the resolved span's offsets fall outside the current message content")}

    # segment_id self-verification: the segment_id -> message_index mapping above is a LOOKUP against the
    # run's own receipt, not a proof -- see _segment_id_index's docstring. Before trusting it, re-hash the
    # message CURRENTLY at message_index (clozn.runs.context_receipt._content_hash, reused rather than
    # reimplemented) and require it to match the segment's own recorded content_hash. A missing hash to
    # check against is treated the same as "unresolvable" (never silently trusted); an observed mismatch
    # is treated as drift, same vocabulary the full-hash check just below uses for the same failure mode.
    # This is what makes a stale/corrupted original_order fail loud instead of splicing the wrong message.
    segment_id = native_ref.get("segment_id")
    if isinstance(segment_id, str) and segment_id:
        from clozn.runs.context_receipt import _content_hash
        entry = segment_index.get(segment_id)
        recorded_content_hash = entry[1] if entry is not None else None
        if not recorded_content_hash:
            return {"ok": False, "reason": _reason(
                "span_message_index_unresolvable",
                "this segment's receipt entry carries no recorded content_hash this bridge can verify "
                "the segment_id -> message_index lookup against, so it cannot be trusted")}
        if _content_hash(content) != recorded_content_hash:
            return {"ok": False, "reason": _reason(
                "span_address_not_found_or_drifted",
                "the message content now sitting at this segment's recorded message-list position no "
                "longer matches the segment_id's recorded content_hash -- refusing rather than risk "
                "ablating the wrong message")}

    # Belt-and-suspenders re-verification: the freshly-rebuilt document already recomputed this hash from
    # CURRENT content (see module docstring), so this should always agree -- but this module never trusts
    # an upstream computation silently. A mismatch here is a bug elsewhere, and MUST refuse, not proceed.
    actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if isinstance(basis_sha256, str) and basis_sha256 and basis_sha256 != actual_hash:
        return {"ok": False, "reason": _reason(
            "span_address_not_found_or_drifted",
            "the message content's hash no longer matches this span's recorded canonical basis")}

    span = {
        "message_index": message_index,
        "start": start,
        "end": end,
        "basis_sha256_verified": True,
        "span_address_id": address.get("address_id"),
    }
    return {"ok": True, "span": span}


def _fresh_addresses(run: dict) -> list:
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses
    document = build_persisted_text_span_addresses(run, privacy="metadata_only")
    addresses = document.get("addresses")
    return addresses if isinstance(addresses, list) else []


def resolve_span_address(run: dict, address_id: str) -> dict:
    """`{"ok": True, "span": ResolvedSpan}` or `{"ok": False, "reason": Reason}` for one `address_id`
    against `run`'s CURRENT content. Never raises: a malformed `run` degrades to a `span_unavailable`
    refusal rather than an exception, mirroring every sibling read-only synthesis module in this
    codebase (`clozn.runs.investigation`, `clozn.receipts.explain`)."""
    try:
        addresses = _fresh_addresses(run)
    except Exception as exc:  # noqa: BLE001 -- a malformed run must refuse, never crash the caller
        return {"ok": False, "reason": _reason(
            "span_unavailable", f"could not rebuild this run's span addresses: {type(exc).__name__}: {exc}")}
    segment_index = _segment_id_index(run)
    for address in addresses:
        if isinstance(address, dict) and address.get("address_id") == address_id:
            return _resolved_span_from_address(run, address, segment_index)
    return {"ok": False, "reason": _reason(
        "span_address_not_found_or_drifted",
        f"no address {address_id!r} exists in this run's current span-address projection -- either it "
        "never existed, or the content it pointed at has changed since the id was minted")}


def resolve_source_spans(run: dict, source_id: str) -> dict:
    """Every resolvable span attached to `source_id` (a `client_source_id`) -> `{"ok": True, "spans":
    [ResolvedSpan, ...]}`, or `{"ok": False, "reason": Reason}` when the source has no addresses at all
    (`source_not_found`) or none of its addresses could be resolved to a message-anchored span
    (`source_has_no_resolvable_spans`, e.g. every occurrence is rendered_prompt/answer_span-anchored, or a
    segment_id occurrence whose content_hash no longer matches -- see `_resolved_span_from_address`). A
    source with SOME resolvable and some unresolvable occurrences still succeeds with the resolvable
    subset -- "omit this source" ablates what this bridge CAN reach, never silently claims to have removed
    what it could not."""
    try:
        addresses = _fresh_addresses(run)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": _reason(
            "span_unavailable", f"could not rebuild this run's span addresses: {type(exc).__name__}: {exc}")}
    matching = [
        a for a in addresses
        if isinstance(a, dict) and isinstance(a.get("native_ref"), dict)
        and a["native_ref"].get("client_source_id") == source_id
    ]
    if not matching:
        return {"ok": False, "reason": _reason(
            "source_not_found", f"no attached source with client_source_id {source_id!r} was found on this run")}
    segment_index = _segment_id_index(run)
    spans = []
    for address in matching:
        resolved = _resolved_span_from_address(run, address, segment_index)
        if resolved.get("ok"):
            spans.append(resolved["span"])
    if not spans:
        return {"ok": False, "reason": _reason(
            "source_has_no_resolvable_spans",
            f"source {source_id!r} was found but none of its occurrences resolve to a message-anchored "
            "span this bridge can splice (see span_basis_unsupported/span_message_index_unresolvable)")}
    return {"ok": True, "spans": spans}


def excise_spans(messages: list, spans: list, *, replacement=None) -> list:
    """`messages` with every `ResolvedSpan`'s `[start, end)` removed (`replacement=None`) or replaced
    (`replacement(n_chars) -> str`, e.g. `clozn.receipts.forced._matched_length_filler`). Multiple spans in
    the SAME message are spliced right-to-left (highest start first) so an earlier span's offsets stay
    valid; spans in different messages are independent. Never raises: an out-of-range message index, or a
    non-dict message, is left untouched -- same "never take the rest of the list down with one bad entry"
    discipline `replay.py`'s own `_strip_message_parts` uses. `messages` and its dicts are never mutated
    in place; only the touched messages are shallow-copied."""
    by_message: dict = {}
    for span in spans or []:
        if not isinstance(span, dict):
            continue
        idx = span.get("message_index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            continue
        by_message.setdefault(idx, []).append((span.get("start"), span.get("end")))

    out = [dict(m) if isinstance(m, dict) else m for m in (messages or [])]
    for idx, ranges in by_message.items():
        if not (0 <= idx < len(out)) or not isinstance(out[idx], dict):
            continue
        content = out[idx].get("content")
        content = content if isinstance(content, str) else ""
        for start, end in sorted(ranges, key=lambda pair: (pair[0] if isinstance(pair[0], int) else -1),
                                 reverse=True):
            if not isinstance(start, int) or isinstance(start, bool):
                continue
            if not isinstance(end, int) or isinstance(end, bool):
                continue
            start = max(0, min(start, len(content)))
            end = max(0, min(end, len(content)))
            if end < start:
                start, end = end, start
            fill = "" if replacement is None else replacement(end - start)
            content = content[:start] + fill + content[end:]
        out[idx]["content"] = content
    return out


def pick_random_control_span(run: dict, span: dict, *, extra: str = "") -> "dict | None":
    """A same-length window elsewhere in `span["message_index"]`'s current content, disjoint from `span`
    itself, chosen deterministically from a SHA-256 digest over `{run_id, message_index, start, end,
    extra}` (never Python's process-randomized `hash()` -- see module docstring). `None` when the message
    is too short to fit a disjoint same-length window -- never a guessed overlapping or shorter one."""
    idx = span.get("message_index")
    start, end = span.get("start"), span.get("end")
    if not isinstance(idx, int) or isinstance(idx, bool):
        return None
    if (not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool) or end < start):
        return None
    messages = run.get("messages") if isinstance(run.get("messages"), list) else []
    if not (0 <= idx < len(messages)) or not isinstance(messages[idx], dict):
        return None
    content = messages[idx].get("content")
    content = content if isinstance(content, str) else ""
    length = end - start
    if length <= 0 or len(content) < length:
        return None

    candidates = [
        i for i in range(0, len(content) - length + 1)
        if i + length <= start or i >= end
    ]
    if not candidates:
        return None

    key = {"run_id": run.get("id"), "message_index": idx, "start": start, "end": end, "extra": extra}
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    pick = int.from_bytes(digest[:4], byteorder="big") % len(candidates)
    chosen_start = candidates[pick]
    return {
        "message_index": idx,
        "start": chosen_start,
        "end": chosen_start + length,
        "basis_sha256_verified": True,
    }


def derive_seed(run: dict, *, purpose: str) -> int:
    """A deterministic uint32 seed from `{run_id, purpose}` -- SHA-256, never `hash()` (module docstring).
    Used to pin `sampler_change` arms to the SAME seed across baseline/no-op/treatment so an observed
    difference is attributable to the requested sampler override, not to seed variance."""
    key = {"run_id": run.get("id"), "purpose": purpose}
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return int.from_bytes(digest[:4], byteorder="big")


__all__ = [
    "derive_seed",
    "excise_spans",
    "pick_random_control_span",
    "resolve_source_spans",
    "resolve_span_address",
]
