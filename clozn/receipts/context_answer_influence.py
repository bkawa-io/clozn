"""Fast, generation-free context <-> answer influence maps.

The map teacher-forces a recorded run's *same continuation* once under the
recorded prompt and once per bounded context span with that span replaced by
the existing matched-length neutral filler.  A matrix cell is therefore::

    log p(recorded answer token | recorded context)
      - log p(recorded answer token | neutral replacement)

Positive values mean the context span supported that answer token relative to
the matched control; negative values mean it suppressed it.  The values are
signed log-probability deltas, never additive percentages and never a circuit
explanation.

This module deliberately has no generation, persistence, route, or rendering
dependencies.  It is duck-typed against ``sub.score_tokens`` through
``receipts.rederive.score_arm`` and is consequently model-free testable.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import time

from . import rederive
from .forced import _FORCED_MEAN_THRESHOLD, _forced_deltas, _matched_length_filler


SCHEMA = "clozn.context_answer_influence.v1"

# The spec-facing mode enum (notes/agent_roadmap/07-sources-evidence-lens.md's "Measurement modes"):
# presence_only / attention_proxy / forced_score_intervention / regeneration_intervention / unsupported.
# This module implements exactly one of those -- forced_score_intervention -- never attention_proxy
# (raw attention weight is not read anywhere on this path; every effect is a scored intervention).
MODE = "forced_score_intervention"

# The two evidence states a LINK can honestly carry once a map computes successfully. A link that never
# clears the measurement floor is still `observed` (the intervention ran; nothing hidden), never silently
# dropped -- "no clear source" must never be misread as "the source was irrelevant" (spec, verbatim).
EVIDENCE_STATE_CAUSALLY_SUPPORTED = "causally_supported"
EVIDENCE_STATE_OBSERVED = "observed"

# The persistent, product-facing caveat every displayed number here needs attached (spec's own required
# text, verbatim). Lives on `method` -- present whether the call ok'd or failed -- so a consumer reading
# ANY shape this module returns is one key away from the sentence that bounds every number in it.
_PERSISTENT_CAVEAT = (
    "Influence means this context changed the measured output under this intervention. It does not "
    "prove the document is correct or reveal hidden reasoning."
)

# The complete, closed set of `error.code` values this module ever emits (grep-verified against every
# `_failed(..., code=...)` call site below). Not enforced at runtime by this module itself -- enforced
# structurally by clozn/schemas/defs/clozn.context_answer_influence.v1.json's `error.code` enum, which a
# new code path that forgets to add itself here will fail against the moment it's validated at the
# persistence boundary (clozn/server/routes/influence_map.py). Keep this set and the schema's enum in sync.
ERROR_CODES = frozenset({
    "invalid_run",
    "no_text_context",
    "no_recorded_continuation",
    "scoring_unavailable",
    "invalid_baseline_score",
    "intervention_score_failed",
    "influence_map_error",
})

DEFAULT_MAX_CONTEXT_SPANS = 8
DEFAULT_TARGET_CHUNK_CHARS = 600
# Coarse-to-fine refinement: only the strongest coarse-clearing spans are ever split further, and only
# by this much, so the map's total score-call count stays bounded and the timing gate stays honest.
DEFAULT_MAX_REFINED_SOURCES = 2
DEFAULT_MAX_FINE_SPANS_PER_SOURCE = 3
DEFAULT_FINE_TARGET_CHUNK_CHARS = 150
_CONTROL_RECIPE = "clozn.matched_length_neutral_filler.v1"

_METHOD = {
    "name": "teacher_forced_matched_context_replacement",
    "mode": MODE,
    "generation_used": False,
    "baseline_reused": True,
    "measurement": "signed_logprob_delta_nats_per_recorded_answer_token",
    "sign": (
        "positive means the recorded context supported the answer token relative to the "
        "matched neutral replacement; negative means it suppressed the token"
    ),
    "segmentation": "coarse_source_aware_spans_v1_refined_coarse_to_fine_on_the_strongest_only",
    "redundancy_check": (
        "one bounded joint-ablation control on the two most strongly-clearing context spans, "
        "reported as an interaction delta, never as an additive percentage"
    ),
    "claim_limit": (
        "behavioral dependence under a controlled prompt intervention; not a percentage, "
        "attention explanation, internal mediation result, or circuit explanation"
    ),
    "caveat": _PERSISTENT_CAVEAT,
}


def _round(value: float) -> float:
    # Avoid a mixture of 0.0 and -0.0 in otherwise identical artifacts.
    rounded = round(float(value), 6)
    return 0.0 if rounded == 0 else rounded


def _digest(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _identity(run: dict, *, prompt_view: str) -> dict:
    captured = run.get("identity")
    captured = deepcopy(captured) if isinstance(captured, dict) else {}
    final_prompt = run.get("final_prompt")
    return {
        "run_id": run.get("id"),
        "model": run.get("model"),
        "substrate": run.get("substrate"),
        "captured": captured,
        "prompt_view": prompt_view,
        "final_prompt_sha256": (
            hashlib.sha256(final_prompt.encode("utf-8")).hexdigest()
            if isinstance(final_prompt, str)
            else None
        ),
    }


def _base_result(run: dict, *, prompt_view: str) -> dict:
    return {
        "schema": SCHEMA,
        "status": "unavailable",
        "available": False,
        "method": deepcopy(_METHOD),
        "identity": _identity(run, prompt_view=prompt_view),
    }


def _failed(run: dict, *, prompt_view: str, status: str, code: str, message: str,
            started: float, clock, **evidence) -> dict:
    out = _base_result(run, prompt_view=prompt_view)
    out.update(evidence)
    out.update({
        "status": status,
        "available": False,
        "error": {"code": code, "message": message},
        "timing": {"total_ms": _round(max(0.0, (clock() - started) * 1000.0))},
    })
    return out


def _text_ranges(text: str, *, target_chunk_chars: int) -> list[dict]:
    """Return deterministic sentence/hard-chunk ranges without changing source text.

    Ranges are half-open Python character offsets.  Leading/trailing whitespace is
    excluded from a range, while whitespace between grouped ranges is retained when
    the coarse span is assembled later.
    """
    if not text or not text.strip():
        return []

    sentence_ends = [
        match.end()
        for match in re.finditer(r"[.!?]+(?:[\"')\]]+)?(?=\s|$)", text)
    ]
    raw = []
    cursor = 0
    for end in sentence_ends:
        raw.append((cursor, end, "sentence"))
        cursor = end
    if cursor < len(text):
        raw.append((cursor, len(text), "sentence"))
    if not raw:
        raw = [(0, len(text), "chunk")]

    ranges = []
    target = max(64, int(target_chunk_chars))
    for raw_start, raw_end, natural_kind in raw:
        start, end = raw_start, raw_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        if end - start <= target:
            ranges.append({"start": start, "end": end, "kind": natural_kind})
            continue

        # A punctuation-free or very long sentence is split near whitespace.  The
        # hard upper bound is intentionally soft: a single word is never cut.
        part_start = start
        while end - part_start > target:
            candidate = part_start + target
            cut = candidate
            while cut > part_start and not text[cut - 1].isspace():
                cut -= 1
            if cut == part_start:
                cut = candidate
                while cut < end and not text[cut].isspace():
                    cut += 1
            part_end = cut
            while part_end > part_start and text[part_end - 1].isspace():
                part_end -= 1
            if part_end > part_start:
                ranges.append({"start": part_start, "end": part_end, "kind": "chunk"})
            part_start = cut
            while part_start < end and text[part_start].isspace():
                part_start += 1
        if part_start < end:
            ranges.append({"start": part_start, "end": end, "kind": "chunk"})
    return ranges


def _message_source(message: dict, index: int, *, prompt_view: str) -> dict | None:
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return None
    role = str(message.get("role") or "unknown")
    external_id = message.get("source_id", message.get("id"))
    name = message.get("name")
    return {
        "id": f"p.m{index:03d}",
        "target": "message",
        "message_index": index,
        "role": role,
        "name": str(name) if isinstance(name, (str, int)) else None,
        "external_source_id": (
            str(external_id) if isinstance(external_id, (str, int)) else None
        ),
        "source_kind": "assembled_message" if prompt_view == "assembled_messages" else "message",
        "start": 0,
        "end": len(content),
        "text": content,
    }


def _sources(messages: list, *, prompt_view: str, block: str | None) -> list[dict]:
    sources = []
    # A fallback prompt block is scored through score_arm's block argument.  Treat
    # it as a virtual system source so the map does not silently omit injected
    # memory on legacy runs without assembled_messages.
    if prompt_view != "assembled_messages" and isinstance(block, str) and block.strip():
        sources.append({
            "id": "p.b000",
            "target": "block",
            "message_index": None,
            "role": "system",
            "name": "prompt_block",
            "external_source_id": None,
            "source_kind": "prompt_block",
            "start": 0,
            "end": len(block),
            "text": block,
        })
    for index, message in enumerate(messages):
        source = _message_source(message, index, prompt_view=prompt_view)
        if source is not None:
            sources.append(source)
    return sources


def _select_sources(sources: list[dict], limit: int) -> tuple[list[dict], list[str]]:
    if len(sources) <= limit:
        return list(sources), []

    # Keep the earliest policy/system source, then the most recent context.  This
    # is bounded and deterministic; omitted IDs are surfaced rather than hidden.
    policy = next((s for s in sources if s["role"] in {"system", "developer"}), None)
    selected = []
    if policy is not None:
        selected.append(policy)
    if len(selected) >= limit:
        selected_ids = {source["id"] for source in selected[:limit]}
        chosen = [source for source in sources if source["id"] in selected_ids]
        omitted = [source["id"] for source in sources if source["id"] not in selected_ids]
        return chosen, omitted
    for source in reversed(sources):
        if source not in selected:
            selected.append(source)
        if len(selected) >= limit:
            break
    selected_ids = {source["id"] for source in selected}
    selected = [source for source in sources if source["id"] in selected_ids]
    omitted = [source["id"] for source in sources if source["id"] not in selected_ids]
    return selected, omitted


def _allocate_slots(sources: list[dict], units: dict[str, list[dict]], limit: int) -> dict[str, int]:
    allocation = {source["id"]: 1 for source in sources}
    remaining = max(0, limit - len(sources))
    while remaining:
        eligible = [
            source for source in sources
            if allocation[source["id"]] < len(units[source["id"]])
        ]
        if not eligible:
            break
        # Largest average characters per prospective partition wins.  Stable
        # source order is the tie-breaker because max() keeps the first item.
        chosen = max(
            eligible,
            key=lambda source: len(source["text"]) / (allocation[source["id"]] + 1),
        )
        allocation[chosen["id"]] += 1
        remaining -= 1
    return allocation


def _partition_units(units: list[dict], count: int) -> list[list[dict]]:
    count = min(max(1, count), len(units))
    # Contiguous, count-balanced partitioning is predictable and refinement-ready.
    return [
        units[(i * len(units)) // count:((i + 1) * len(units)) // count]
        for i in range(count)
    ]


def segment_context(messages: list, *, block: str | None = None,
                    prompt_view: str = "assembled_messages",
                    max_spans: int = DEFAULT_MAX_CONTEXT_SPANS,
                    target_chunk_chars: int = DEFAULT_TARGET_CHUNK_CHARS) -> dict:
    """Segment exact prompt sources into at most ``max_spans`` coarse spans.

    No span crosses a message/source boundary.  Every span retains exact source
    text plus half-open character offsets, role, and source identity.  Stable
    IDs (``p.mNNN.cNNN``) leave room for later fine descendants.
    """
    limit = max(1, int(max_spans))
    all_sources = _sources(messages, prompt_view=prompt_view, block=block)
    selected, omitted_ids = _select_sources(all_sources, limit)
    units = {
        source["id"]: _text_ranges(source["text"], target_chunk_chars=target_chunk_chars)
        for source in selected
    }
    selected = [source for source in selected if units[source["id"]]]
    allocation = _allocate_slots(selected, units, limit) if selected else {}

    spans = []
    selected_ids = {source["id"] for source in selected}
    for source in selected:
        groups = _partition_units(units[source["id"]], allocation[source["id"]])
        for coarse_index, group in enumerate(groups):
            start, end = group[0]["start"], group[-1]["end"]
            kind = group[0]["kind"] if len(group) == 1 else "chunk"
            spans.append({
                "id": f"{source['id']}.c{coarse_index:03d}",
                "parent_id": source["id"],
                "level": "coarse",
                "kind": kind,
                "target": source["target"],
                "message_index": source["message_index"],
                "role": source["role"],
                "source_kind": source["source_kind"],
                "start": start,
                "end": end,
                "text": source["text"][start:end],
                "child_unit_count": len(group),
            })

    public_sources = []
    for source in all_sources:
        public = {key: value for key, value in source.items() if key != "target"}
        public["selected"] = source["id"] in selected_ids
        public_sources.append(public)
    return {
        "sources": public_sources,
        "spans": spans,
        "selection": {
            "strategy": "earliest_policy_then_recent_sources_proportional_chunks_v1",
            "max_context_spans": limit,
            "selected_source_ids": [source["id"] for source in selected],
            "omitted_source_ids": omitted_ids,
            "measured_span_count": len(spans),
            "complete_for_selected_spans": True,
        },
    }


def _replace_span(messages: list, block: str | None, span: dict) -> tuple[list, str | None, dict]:
    replacement = _matched_length_filler(span["end"] - span["start"])
    copied = [dict(message) if isinstance(message, dict) else message for message in messages]
    copied_block = block
    if span["target"] == "block":
        original = copied_block or ""
        copied_block = original[:span["start"]] + replacement + original[span["end"]:]
    else:
        index = span["message_index"]
        original = copied[index]["content"]
        copied[index]["content"] = (
            original[:span["start"]] + replacement + original[span["end"]:]
        )
    control = {
        "context_span_id": span["id"],
        "kind": "matched_length_neutral_filler",
        "recipe": _CONTROL_RECIPE,
        "replacement_chars": len(replacement),
        "source_chars": span["end"] - span["start"],
        "length_preserved": len(replacement) == span["end"] - span["start"],
        "replacement_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
    }
    return copied, copied_block, control


def _refine_span(span: dict, *, target_chunk_chars: int, max_fine_spans: int) -> list[dict]:
    """Split one coarse span's own exact text into finer sentence/chunk sub-spans at a smaller
    target size, expressed in the same absolute source coordinates the coarse span already uses.

    Returns ``[]`` when the span is already atomic at the finer granularity -- coarse-to-fine
    refinement never manufactures a fine span that would just duplicate its coarse parent.
    """
    local_ranges = _text_ranges(span["text"], target_chunk_chars=target_chunk_chars)
    if len(local_ranges) <= 1:
        return []
    groups = _partition_units(local_ranges, min(max(1, int(max_fine_spans)), len(local_ranges)))
    if len(groups) <= 1:
        return []
    fine_spans = []
    for fine_index, group in enumerate(groups):
        local_start, local_end = group[0]["start"], group[-1]["end"]
        kind = group[0]["kind"] if len(group) == 1 else "chunk"
        fine_spans.append({
            "id": f"{span['id']}.f{fine_index:03d}",
            "parent_id": span["id"],
            "level": "fine",
            "kind": kind,
            "target": span["target"],
            "message_index": span["message_index"],
            "role": span["role"],
            "source_kind": span["source_kind"],
            "start": span["start"] + local_start,
            "end": span["start"] + local_end,
            "text": span["text"][local_start:local_end],
            "child_unit_count": len(group),
        })
    return fine_spans


def _replace_spans(messages: list, block: str | None, spans: list[dict]) -> tuple[list, str | None, list[dict]]:
    """Replace MULTIPLE bounded spans in one arm, each with its own independent matched-length
    neutral filler. Edits inside the same message/block text apply right-to-left so an earlier
    replacement never shifts a later span's recorded offsets. Used only by the bounded redundant-pair
    joint-ablation check -- every other arm still replaces exactly one span via ``_replace_span``.
    """
    copied = [dict(message) if isinstance(message, dict) else message for message in messages]
    copied_block = block
    controls: list[dict] = []

    def _apply(text: str, group: list[dict]) -> str:
        for span in sorted(group, key=lambda item: -item["start"]):
            replacement = _matched_length_filler(span["end"] - span["start"])
            text = text[:span["start"]] + replacement + text[span["end"]:]
            controls.append({
                "context_span_id": span["id"],
                "kind": "matched_length_neutral_filler",
                "recipe": _CONTROL_RECIPE,
                "replacement_chars": len(replacement),
                "source_chars": span["end"] - span["start"],
                "length_preserved": len(replacement) == span["end"] - span["start"],
                "replacement_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
            })
        return text

    by_message: dict[int, list[dict]] = {}
    block_spans = []
    for span in spans:
        if span["target"] == "block":
            block_spans.append(span)
        else:
            by_message.setdefault(span["message_index"], []).append(span)
    for index, group in by_message.items():
        copied[index]["content"] = _apply(copied[index]["content"], group)
    if block_spans:
        copied_block = _apply(copied_block or "", block_spans)

    order = [span["id"] for span in spans]
    controls.sort(key=lambda control: order.index(control["context_span_id"]))
    return copied, copied_block, controls


def _validated_tokens(tokens: list) -> list[dict] | None:
    if not isinstance(tokens, list) or not tokens:
        return None
    out = []
    for index, token in enumerate(tokens):
        if not isinstance(token, dict):
            return None
        logprob = token.get("logprob")
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
            return None
        if not math.isfinite(float(logprob)):
            return None
        out.append({
            "index": index,
            "token_id": token.get("id"),
            "piece": str(token.get("piece", "")),
            "logprob": _round(logprob),
        })
    return out


def _answer_spans(baseline: list[dict], recorded_answer: str) -> tuple[list[dict], dict]:
    scored_text = "".join(token["piece"] for token in baseline)
    cursor = 0
    spans = []
    for token in baseline:
        start, end = cursor, cursor + len(token["piece"])
        spans.append({
            "id": f"a.t{token['index']:04d}",
            "level": "token",
            "token_index": token["index"],
            "token_id": token["token_id"],
            "start": start,
            "end": end,
            "text": token["piece"],
        })
        cursor = end
    return spans, {
        "recorded_text": recorded_answer,
        "scored_text": scored_text,
        "scored_text_matches_recorded": scored_text == recorded_answer,
        "offset_basis": "scored_text",
    }


def _summary(prompt_spans: list[dict], answer_spans: list[dict], links: list[dict],
            refined_span_ids: frozenset[str] = frozenset()) -> dict:
    answer_to_context = []
    no_clear = []
    for answer in answer_spans:
        candidates = [link for link in links if link["answer_span_id"] == answer["id"]]
        candidates.sort(key=lambda link: (-link["abs_delta_nats"], link["context_span_id"]))
        clear = [link for link in candidates if link["clears_floor"]]
        if not clear:
            no_clear.append(answer["id"])
        # A refined coarse parent's own row stays in the full measured matrix/links and still counts
        # toward "clear_source" (never silently downgraded) -- but its finer children are strictly
        # more specific about the SAME source text, so the TOP-RANKED display list prefers them and
        # only falls back to the parent when no fine child individually cleared the floor.
        displayed = [link for link in clear if link["context_span_id"] not in refined_span_ids] or clear
        answer_to_context.append({
            "answer_span_id": answer["id"],
            "clear_source": bool(clear),
            "top_context_span_ids": [link["context_span_id"] for link in displayed[:3]],
        })

    context_to_answer = []
    for context in prompt_spans:
        candidates = [link for link in links if link["context_span_id"] == context["id"]]
        candidates.sort(key=lambda link: (-link["abs_delta_nats"], link["answer_span_id"]))
        clear = [link for link in candidates if link["clears_floor"]]
        context_to_answer.append({
            "context_span_id": context["id"],
            "clear_effect": bool(clear),
            "top_answer_span_ids": [link["answer_span_id"] for link in clear[:5]],
        })
    return {
        "has_any_clear_source": len(no_clear) < len(answer_spans),
        "no_clear_source": len(no_clear) == len(answer_spans),
        "answer_span_ids_without_clear_source": no_clear,
        "answer_to_context": answer_to_context,
        "context_to_answer": context_to_answer,
    }


def _redundancy_check(sub, conditions, messages, block, *, prompt_spans: list[dict], matrix: list[list],
                      answer_spans: list[dict], baseline_tokens, baseline: list[dict], floor: float,
                      clock) -> dict:
    """One bounded joint-ablation control on the two most strongly-clearing DISTINCT context spans.

    Both are replaced simultaneously in a single extra arm and the joint delta is compared, per answer
    token, against the plain SUM of their two individually-measured deltas.  A near-zero interaction
    means the two spans behaved roughly additively when replaced together; a large interaction means
    they were redundant/overlapping (interacting) sources for that token.  This is a single honest
    measurement of ONE pair, never a search over every pair, and never a claim that the two spans are
    the model's only causes or that their effects sum to a percentage of anything.
    """
    ranked = sorted(
        (
            (max((abs(delta) for delta in row), default=0.0), span["id"], index)
            for index, (span, row) in enumerate(zip(prompt_spans, matrix))
        ),
        key=lambda item: (-item[0], item[1]),
    )
    clearing = [item for item in ranked if item[0] >= floor]
    if len(clearing) < 2:
        return {"performed": False, "reason": "fewer than two context spans clear the measurement floor"}

    (_mag_a, id_a, index_a), (_mag_b, id_b, index_b) = clearing[0], clearing[1]
    span_a, span_b = prompt_spans[index_a], prompt_spans[index_b]
    row_a, row_b = matrix[index_a], matrix[index_b]

    arm_started = clock()
    joint_messages, joint_block, joint_controls = _replace_spans(messages, block, [span_a, span_b])
    joint_tokens, joint_ok = rederive.score_arm(
        sub, conditions, messages=joint_messages, block=joint_block,
        steer_strengths=conditions.get("steer_strengths") or {},
    )
    score_ms = _round(max(0.0, (clock() - arm_started) * 1000.0))
    context_span_ids = [id_a, id_b]
    if not joint_ok:
        return {"performed": False, "reason": "the joint-control score call did not complete",
                "context_span_ids": context_span_ids, "score_ms": score_ms}

    joint_deltas = _forced_deltas(baseline_tokens, joint_tokens)
    validated_joint = _validated_tokens(joint_tokens)
    if joint_deltas is None or validated_joint is None or len(validated_joint) != len(baseline):
        return {"performed": False, "reason": "the joint-control scorer returned a misaligned result",
                "context_span_ids": context_span_ids, "score_ms": score_ms}

    per_token = []
    for answer_index, answer_span in enumerate(answer_spans):
        individual_sum = _round(row_a[answer_index] + row_b[answer_index])
        joint_delta = _round(joint_deltas[answer_index])
        per_token.append({
            "answer_span_id": answer_span["id"],
            "individual_sum_nats": individual_sum,
            "joint_delta_nats": joint_delta,
            "interaction_nats": _round(joint_delta - individual_sum),
        })
    return {
        "performed": True,
        "context_span_ids": context_span_ids,
        "controls": joint_controls,
        "per_answer_token": per_token,
        "score_ms": score_ms,
        "claim_limit": (
            "one bounded interaction measurement between the two most strongly-clearing measured "
            "context spans; a near-zero interaction means their individually-measured effects were "
            "roughly additive under joint replacement -- it is not evidence the two spans are the "
            "model's only causes, and never a percentage of total explanation"
        ),
    }


def context_answer_influence(run: dict, sub, *, max_context_spans: int = DEFAULT_MAX_CONTEXT_SPANS,
                             min_abs_delta_nats: float = _FORCED_MEAN_THRESHOLD,
                             target_chunk_chars: int = DEFAULT_TARGET_CHUNK_CHARS,
                             max_refined_sources: int = DEFAULT_MAX_REFINED_SOURCES,
                             max_fine_spans_per_source: int = DEFAULT_MAX_FINE_SPANS_PER_SOURCE,
                             fine_target_chunk_chars: int = DEFAULT_FINE_TARGET_CHUNK_CHARS,
                             check_redundant_pair: bool = True,
                             clock=time.perf_counter) -> dict:
    """Build one portable ``clozn.context_answer_influence.v1`` evidence object.

    Successful maps make ``1 + len(prompt_spans)`` score calls: one baseline (reused for every arm)
    and one matched-control arm per context span, coarse first.  Spans whose coarse arm clears the
    measurement floor are automatically refined into up to ``max_fine_spans_per_source`` finer
    sub-spans each (bounded to the ``max_refined_sources`` strongest), each scored with one more
    reused-baseline arm and appended to the same matrix.  When at least two spans clear the floor, one
    additional bounded joint-ablation arm checks the strongest redundant pair.  Every call uses
    ``rederive.score_arm`` and the same stored continuation; generation is never invoked.  Failures are
    structured ``unavailable``/``error`` objects and never raise into a receipt caller.
    """
    started = clock()
    run = run if isinstance(run, dict) else {}
    conditions = rederive.with_arm_conditions(run)
    prompt_view = conditions.get("block_source") or "none"
    if prompt_view == "prompt_block":
        prompt_view = "messages_plus_prompt_block"
    elif prompt_view == "none":
        prompt_view = "messages"

    try:
        if not run:
            return _failed(
                run, prompt_view=prompt_view, status="error", code="invalid_run",
                message="a recorded run is required", started=started, clock=clock,
            )
        messages = conditions.get("messages")
        block = conditions.get("block")
        if not isinstance(messages, list):
            messages = []

        segmented = segment_context(
            messages, block=block, prompt_view=prompt_view,
            max_spans=max_context_spans, target_chunk_chars=target_chunk_chars,
        )
        prompt_spans = segmented["spans"]
        prompt_evidence = {
            "prompt_sources": segmented["sources"],
            "prompt_spans": [
                {key: value for key, value in span.items() if key != "target"}
                for span in prompt_spans
            ],
            "selection": segmented["selection"],
        }
        if not prompt_spans:
            return _failed(
                run, prompt_view=prompt_view, status="unavailable", code="no_text_context",
                message="the recorded run has no text context spans to measure",
                started=started, clock=clock, **prompt_evidence,
            )
        if conditions.get("continuation_ids") is None and not conditions.get("response"):
            return _failed(
                run, prompt_view=prompt_view, status="unavailable", code="no_recorded_continuation",
                message="the run has neither recorded continuation token IDs nor response text",
                started=started, clock=clock, **prompt_evidence,
            )

        baseline_started = clock()
        baseline_tokens, baseline_ok = rederive.score_arm(
            sub, conditions, messages=messages, block=block,
            steer_strengths=conditions.get("steer_strengths") or {},
        )
        baseline_ms = max(0.0, (clock() - baseline_started) * 1000.0)
        baseline = _validated_tokens(baseline_tokens) if baseline_ok else None
        recorded_ids = conditions.get("continuation_ids")
        if baseline is not None and recorded_ids is not None and len(baseline) != len(recorded_ids):
            baseline = None
        if baseline is None:
            code = "scoring_unavailable" if not baseline_ok else "invalid_baseline_score"
            return _failed(
                run, prompt_view=prompt_view, status="unavailable", code=code,
                message=("teacher-forced score_tokens is unavailable or the baseline could not be scored"
                         if not baseline_ok else
                         "the baseline scorer returned no finite, aligned token log-probabilities"),
                started=started, clock=clock, timing_detail={"baseline_ms": _round(baseline_ms)},
                **prompt_evidence,
            )

        answer_spans, answer = _answer_spans(baseline, str(conditions.get("response") or ""))

        def _score_one(span: dict) -> tuple[bool, list | None, dict, float]:
            arm_messages, arm_block, control = _replace_span(messages, block, span)
            arm_started = clock()
            arm_tokens, arm_ok = rederive.score_arm(
                sub, conditions, messages=arm_messages, block=arm_block,
                steer_strengths=conditions.get("steer_strengths") or {},
            )
            arm_ms = max(0.0, (clock() - arm_started) * 1000.0)
            raw_deltas = _forced_deltas(baseline_tokens, arm_tokens) if arm_ok else None
            validated_arm = _validated_tokens(arm_tokens) if arm_ok else None
            if raw_deltas is None or validated_arm is None or len(validated_arm) != len(baseline):
                return False, None, control, arm_ms
            row = [_round(delta) for delta in raw_deltas]
            control["counterfactual_logprobs"] = [token["logprob"] for token in validated_arm]
            return True, row, control, arm_ms

        matrix = []
        controls = []
        intervention_times = []
        for span in prompt_spans:
            ok, row, control, arm_ms = _score_one(span)
            intervention_times.append({"context_span_id": span["id"], "score_ms": _round(arm_ms)})
            if not ok:
                return _failed(
                    run, prompt_view=prompt_view, status="error", code="intervention_score_failed",
                    message=(f"matched-control scoring failed or did not align token-for-token for "
                             f"context span {span['id']}"),
                    started=started, clock=clock,
                    failed_context_span_id=span["id"],
                    completed_context_span_ids=[item["context_span_id"] for item in controls],
                    prompt_sources=prompt_evidence["prompt_sources"],
                    prompt_spans=prompt_evidence["prompt_spans"],
                    selection=prompt_evidence["selection"],
                    answer=answer,
                    answer_spans=answer_spans,
                )
            matrix.append(row)
            controls.append(control)

        floor = max(0.0, float(min_abs_delta_nats))

        # Coarse-to-fine refinement: rank coarse spans by their own strongest cell, then split only
        # the top max_refined_sources that CLEAR the floor into up to max_fine_spans_per_source finer
        # sub-spans each (skipped when a span is already atomic at the finer grain). Automatic,
        # source-aware, and bounded -- refinement never runs on a span that didn't already show a
        # measured effect, and it never grows the map without limit.
        refined_span_ids: list[str] = []
        strongest_first = sorted(
            (
                (max((abs(delta) for delta in row), default=0.0), span["id"], index)
                for index, (span, row) in enumerate(zip(prompt_spans, matrix))
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for magnitude, _span_id, coarse_index in strongest_first:
            if len(refined_span_ids) >= max(0, int(max_refined_sources)):
                break
            if magnitude < floor:
                break
            coarse_span = prompt_spans[coarse_index]
            fine_spans = _refine_span(
                coarse_span, target_chunk_chars=fine_target_chunk_chars,
                max_fine_spans=max_fine_spans_per_source,
            )
            if len(fine_spans) < 2:
                continue
            for fine_span in fine_spans:
                ok, row, control, arm_ms = _score_one(fine_span)
                intervention_times.append({"context_span_id": fine_span["id"], "score_ms": _round(arm_ms)})
                if not ok:
                    return _failed(
                        run, prompt_view=prompt_view, status="error", code="intervention_score_failed",
                        message=(f"matched-control scoring failed or did not align token-for-token for "
                                 f"fine context span {fine_span['id']}"),
                        started=started, clock=clock,
                        failed_context_span_id=fine_span["id"],
                        completed_context_span_ids=[item["context_span_id"] for item in controls],
                        prompt_sources=prompt_evidence["prompt_sources"],
                        prompt_spans=[
                            {key: value for key, value in span.items() if key != "target"}
                            for span in prompt_spans
                        ],
                        selection=prompt_evidence["selection"],
                        answer=answer,
                        answer_spans=answer_spans,
                    )
                prompt_spans.append(fine_span)
                matrix.append(row)
                controls.append(control)
            refined_span_ids.append(coarse_span["id"])
        refined_span_ids = frozenset(refined_span_ids)

        links = []
        for context_index, (span, row) in enumerate(zip(prompt_spans, matrix)):
            for answer_index, (answer_span, delta) in enumerate(zip(answer_spans, row)):
                magnitude = _round(abs(delta))
                clears = magnitude >= floor
                links.append({
                    "context_span_id": span["id"],
                    "answer_span_id": answer_span["id"],
                    "context_index": context_index,
                    "answer_index": answer_index,
                    "delta_nats": delta,
                    "abs_delta_nats": magnitude,
                    "effect": "supports" if delta > 0 else "suppresses" if delta < 0 else "neutral",
                    "clears_floor": clears,
                    # The exact claim this ONE number licenses (roadmap rule 4's explicit-states list):
                    # a link that never clears the floor was still measured -- `observed`, never silently
                    # dropped and never "source_irrelevant" (that claim is not available from a null
                    # result, see `caveat` on `method` above).
                    "evidence_state": (
                        EVIDENCE_STATE_CAUSALLY_SUPPORTED if clears else EVIDENCE_STATE_OBSERVED
                    ),
                })

        # Bounded check: does the strongest redundant pair of context spans behave additively or
        # overlap when replaced together? One extra score call at most, only when >= 2 spans clear.
        redundancy = (
            _redundancy_check(
                sub, conditions, messages, block, prompt_spans=prompt_spans, matrix=matrix,
                answer_spans=answer_spans, baseline_tokens=baseline_tokens, baseline=baseline,
                floor=floor, clock=clock,
            )
            if check_redundant_pair else
            {"performed": False, "reason": "redundant-pair check disabled for this call"}
        )
        redundancy_calls = 1 if "score_ms" in redundancy else 0

        thresholds = {
            "cell_abs_delta_nats": _round(floor),
            "source_clear_rule": "absolute signed cell delta meets or exceeds cell_abs_delta_nats",
            "calibration": "fixed_default_not_model_calibrated",
        }
        final_prompt_spans_public = [
            {key: value for key, value in span.items() if key != "target"}
            for span in prompt_spans
        ]
        selection = dict(prompt_evidence["selection"])
        selection["refinement"] = {
            "strategy": "refine_strongest_clearing_coarse_spans_v1",
            "max_refined_sources": max(0, int(max_refined_sources)),
            "max_fine_spans_per_source": max(0, int(max_fine_spans_per_source)),
            "fine_target_chunk_chars": max(1, int(fine_target_chunk_chars)),
            "refined_context_span_ids": sorted(refined_span_ids),
            "fine_span_count": sum(1 for span in prompt_spans if span.get("level") == "fine"),
        }
        timing = {
            "baseline_ms": _round(baseline_ms),
            "interventions": intervention_times,
            "interventions_total_ms": _round(sum(item["score_ms"] for item in intervention_times)),
            "redundancy_check_ms": redundancy.get("score_ms", 0.0),
            "total_ms": _round(max(0.0, (clock() - started) * 1000.0)),
            "score_calls": 1 + len(prompt_spans) + redundancy_calls,
        }
        result = _base_result(run, prompt_view=prompt_view)
        result.update({
            "status": "ok",
            "available": True,
            **prompt_evidence,
            "prompt_spans": final_prompt_spans_public,
            "selection": selection,
            "offsets": {
                "unit": "unicode_code_points",
                "interval": "half_open",
                "prompt_basis": "each prompt source's exact text",
                "answer_basis": "answer.scored_text",
            },
            "answer": answer,
            "answer_spans": answer_spans,
            "continuation": {
                "text_exact": True,
                "token_ids_exact": not conditions.get("retokenized", True),
                "retokenized": bool(conditions.get("retokenized", True)),
                "kind": ("recorded_token_ids" if conditions.get("continuation_ids") is not None
                         else "recorded_response_text"),
            },
            "baseline": {
                "logprobs": [token["logprob"] for token in baseline],
                "scored_once": True,
            },
            "controls": controls,
            "thresholds": thresholds,
            "matrix": matrix,
            "matrix_shape": [len(prompt_spans), len(answer_spans)],
            "matrix_complete": True,
            "links": links,
            "summary": _summary(prompt_spans, answer_spans, links, refined_span_ids),
            "redundancy_check": redundancy,
            "timing": timing,
        })
        # Timing varies by machine and is not evidence identity.  Everything else,
        # including raw baseline/control scores and all links, is committed.  redundancy_check carries
        # its own score_ms (machine-dependent) nested inside it, so it needs the same stripping as the
        # top-level timing block -- otherwise wall-clock jitter would make artifact_sha256 non-reproducible
        # for a byte-identical run.
        artifact_payload = {key: value for key, value in result.items() if key != "timing"}
        if isinstance(artifact_payload.get("redundancy_check"), dict):
            artifact_payload["redundancy_check"] = {
                key: value for key, value in artifact_payload["redundancy_check"].items()
                if key != "score_ms"
            }
        result["artifact_sha256"] = _digest(artifact_payload)
        return result
    except Exception as exc:
        return _failed(
            run, prompt_view=prompt_view, status="error", code="influence_map_error",
            message=f"context-answer influence mapping failed: {type(exc).__name__}: {exc}",
            started=started, clock=clock,
        )


# A verb-first alias reads naturally at call sites while keeping the evidence
# object's canonical name prominent.
build_context_answer_influence = context_answer_influence
