"""Run trace normalization and engine event folding."""
from __future__ import annotations

import math


# --------------------------------------------------------------------------- trace (per-token timeline)
# The Run Inspector's timeline wants, per generated token: what was committed, how sure the model was
# (confidence 0..1), and what it nearly said instead (alternatives). Two code paths already carry that:
# the CLI's stream_ar and the engine chat capture. Both hand us the same per-token "step" shape; keep the
# mapping in one pure place so the on-disk trace schema stays a single contract.
TRACE_KEYS = (
    "tokens",
    "confidence",
    "alternatives",
    "token_ids",
    "logprobs",
    "topk_entropy",
    "steps",
    # Model-emitted <think> tokens are evidence, but not answer tokens.  They live beside (never inside)
    # the public timeline so Replay can inspect them without continuation/fork/tool consumers ingesting
    # them as assistant content.
    "reasoning_steps",
    "workspace_readouts",
)


def _float_or_none(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _int_or_none(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _rounded_prob(x):
    v = _float_or_none(x)
    return round(v, 4) if v is not None else None


def _logprob(prob):
    p = _float_or_none(prob)
    if p is None or p <= 0:
        return None
    return round(math.log(p), 6)


def _entropy_from_probs(probs):
    vals = [_float_or_none(p) for p in (probs or [])]
    vals = [p for p in vals if p is not None and p > 0]
    if not vals:
        return None
    return round(-sum(p * math.log(p) for p in vals), 6)


def _clean_alt(a) -> dict | None:
    """Normalize one alternative, preserving token id/text/prob/logprob when they are real."""
    if not isinstance(a, dict):
        return None
    piece = str(a.get("piece", a.get("text", "")))
    prob = _rounded_prob(a.get("prob", a.get("confidence", a.get("conf"))))
    item = {"piece": piece, "text": piece}
    token_id = _int_or_none(a.get("token_id", a.get("id")))
    if token_id is not None:
        item["token_id"] = token_id
    if prob is not None:
        item["prob"] = prob
        lp = _logprob(prob)
        if lp is not None:
            item["logprob"] = lp
    elif _float_or_none(a.get("logprob")) is not None:
        item["logprob"] = round(float(a["logprob"]), 6)
    return item


def _clean_alts(alts) -> list[dict]:
    """Normalize a step's alternatives to rich alt dicts; junk entries are dropped."""
    out = []
    for a in alts or []:
        item = _clean_alt(a)
        if item is not None:
            out.append(item)
    return out


def _clean_step(s, fallback_index: int) -> dict | None:
    """Normalize one raw token step into the v2 schema while keeping v1 aliases readable."""
    if not isinstance(s, dict):
        return None
    piece = str(s.get("piece", s.get("token", s.get("text", ""))))
    index = _int_or_none(s.get("index", s.get("pos")))
    if index is None:
        index = int(fallback_index)
    prob = _rounded_prob(s.get("prob", s.get("conf", s.get("confidence"))))
    step = {"index": index, "piece": piece, "text": piece}
    token_id = _int_or_none(s.get("token_id", s.get("id")))
    if token_id is not None:
        step["token_id"] = token_id
    if prob is not None:
        step["prob"] = prob
        step["confidence"] = prob
        lp = _logprob(prob)
        if lp is not None:
            step["logprob"] = lp
    elif _float_or_none(s.get("logprob")) is not None:
        step["logprob"] = round(float(s["logprob"]), 6)
    alts = _clean_alts(s.get("alts", s.get("alternatives")))
    step["alternatives"] = alts
    for k in ("entropy", "topk_entropy", "wall_ms", "dt_ms"):
        v = _float_or_none(s.get(k))
        if v is not None:
            step[k] = round(v, 6 if k in ("entropy", "topk_entropy") else 3)
    return step


def _steps_from_parallel(trace: dict) -> list[dict]:
    """Reconstruct v2 `steps` from a ready trace dict's parallel arrays."""
    tokens = trace.get("tokens") if isinstance(trace, dict) else None
    if not isinstance(tokens, list):
        return []
    confidence = trace.get("confidence") if isinstance(trace.get("confidence"), list) else []
    alternatives = trace.get("alternatives") if isinstance(trace.get("alternatives"), list) else []
    token_ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else []
    topk_entropy = trace.get("topk_entropy") if isinstance(trace.get("topk_entropy"), list) else []
    out = []
    for i, piece in enumerate(tokens):
        raw = {"index": i, "piece": piece, "alts": alternatives[i] if i < len(alternatives) else []}
        if i < len(confidence):
            raw["conf"] = confidence[i]
        if i < len(token_ids):
            raw["token_id"] = token_ids[i]
        if i < len(topk_entropy) and topk_entropy[i] is not None:
            raw["topk_entropy"] = topk_entropy[i]
        step = _clean_step(raw, i)
        if step is not None:
            out.append(step)
    return out


def steps_to_trace(steps) -> dict:
    """Map per-token steps -> the run's trace dict with v1 arrays plus rich v2 `steps`."""
    steps = [s for s in (steps or []) if isinstance(s, dict)]
    if not steps:
        return {}
    rich = []
    for i, s in enumerate(steps):
        step = _clean_step(s, i)
        if step is not None:
            rich.append(step)
    if not rich:
        return {}
    tokens = [s.get("piece", "") for s in rich]
    confidence = [s.get("prob", 0.0) for s in rich]
    alternatives = [s.get("alternatives", []) for s in rich]
    token_ids = [s.get("token_id") for s in rich]
    logprobs = [s.get("logprob") for s in rich]
    topk_entropy = [s.get("topk_entropy") for s in rich]
    trace = {"tokens": tokens, "confidence": confidence, "steps": rich}
    if any(alternatives):
        trace["alternatives"] = alternatives
    if any(v is not None for v in token_ids):
        trace["token_ids"] = token_ids
    if any(v is not None for v in logprobs):
        trace["logprobs"] = logprobs
    if any(v is not None for v in topk_entropy):
        trace["topk_entropy"] = topk_entropy
    return trace


def accumulate_ar_events(events) -> list[dict]:
    """Fold the engine's autoregressive SSE frames into ordered per-token steps."""
    by_pos: dict = {}
    order: list = []
    for obj in events or []:
        if not isinstance(obj, dict):
            continue
        typ = obj.get("type")
        if typ == "tokens_committed":
            for it in obj.get("items", []):
                pos = it.get("pos")
                if pos not in by_pos:
                    order.append(pos)
                try:
                    conf = round(float(it.get("conf", 0.0)), 4)
                except (TypeError, ValueError):
                    conf = 0.0
                step = {
                    "pos": pos,
                    "index": _int_or_none(pos),
                    "id": it.get("id"),
                    "piece": str(it.get("piece", "")),
                    "conf": conf,
                    "alts": [],
                }
                for k in ("wall_ms", "dt_ms"):
                    if it.get(k) is not None:
                        step[k] = it.get(k)
                by_pos[pos] = step
        elif typ == "step_lens":
            positions = obj.get("positions") or [None]
            pieces, probs = obj.get("pieces", []), obj.get("probs", [])
            ids = obj.get("ids") or [None] * len(pieces)
            try:
                k = int(obj.get("k") or (len(probs) // max(1, len(positions))))
            except (TypeError, ValueError):
                k = len(probs)
            for row, pos in enumerate(positions):
                step = by_pos.get(pos)
                if not step:
                    continue
                start, end = row * k, row * k + k
                chosen_piece = step.get("piece")
                chosen_id = _int_or_none(step.get("id"))
                alts = []
                for tid, piece, prob in zip(ids[start:end], pieces[start:end], probs[start:end]):
                    token_id = _int_or_none(tid)
                    if (chosen_id is not None and token_id == chosen_id) or str(piece) == str(chosen_piece):
                        continue
                    alts.append({"token_id": token_id, "piece": str(piece), "prob": prob})
                    if len(alts) >= 3:
                        break
                step["alts"] = alts
                topk_entropy = _entropy_from_probs(probs[start:end])
                if topk_entropy is not None:
                    step["topk_entropy"] = topk_entropy
    return [by_pos[p] for p in sorted(order, key=lambda x: (x is None, x))]


def finish_reason_from_frames(frames) -> str | None:
    """Pluck the generation's stop cause from the engine's SSE frames."""
    reason = None
    for obj in frames or []:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "gen_finished" and isinstance(obj.get("reason"), str):
            reason = "stop" if obj["reason"] == "eos" else "length"
        if isinstance(obj.get("finish_reason"), str):
            reason = obj["finish_reason"]
        ch = obj.get("choices")
        if (
            isinstance(ch, list)
            and ch
            and isinstance(ch[0], dict)
            and isinstance(ch[0].get("finish_reason"), str)
        ):
            reason = ch[0]["finish_reason"]
    return reason


def raw_finish_reason_from_frames(frames) -> str | None:
    """Preserve the worker's own terminal reason before OpenAI normalization.

    ``gen_finished.reason`` carries values such as ``eos`` and ``steps_exhausted``.  Later summary
    frames intentionally normalize those to ``stop``/``length`` for protocol compatibility, so this
    helper reads only the worker event and never substitutes a public value.
    """
    reason = None
    for obj in frames or []:
        if (
            isinstance(obj, dict)
            and obj.get("type") == "gen_finished"
            and isinstance(obj.get("reason"), str)
            and obj["reason"]
        ):
            reason = obj["reason"]
    return reason


def generation_timing_from_frames(frames) -> dict:
    """Return versioned worker phases plus legacy aggregate aliases from the last terminal frame.

    New workers send ``timing`` with nanosecond durations from their own steady clock. Old workers only
    send ``wall_ms``/``tok_per_s``; that legacy path remains accepted. Invalid or absent fields are
    omitted, never converted into a zero-duration phase.
    """
    timing = {}
    for obj in frames or []:
        if not isinstance(obj, dict) or obj.get("type") != "gen_finished":
            continue
        worker_timing = obj.get("timing")
        if (
            isinstance(worker_timing, dict)
            and worker_timing.get("schema_version") == "clozn.worker-timing.v1"
            and worker_timing.get("unit") == "nanoseconds"
            and isinstance(worker_timing.get("phases"), list)
        ):
            clean_phases = []
            for raw_phase in worker_timing["phases"]:
                if not isinstance(raw_phase, dict) or not isinstance(raw_phase.get("name"), str):
                    continue
                duration_ns = raw_phase.get("duration_ns")
                if (
                    not isinstance(duration_ns, int) or isinstance(duration_ns, bool)
                    or duration_ns < 0
                ):
                    continue
                phase = {
                    "name": raw_phase["name"],
                    "owner": (
                        raw_phase.get("owner")
                        if isinstance(raw_phase.get("owner"), str)
                        else "clozn_worker"
                    ),
                    "duration_ns": duration_ns,
                    "measurement": "measured",
                    "aggregation": (
                        raw_phase.get("aggregation")
                        if raw_phase.get("aggregation") in {
                            "exclusive", "overlapping", "context_only"
                        }
                        else "exclusive"
                    ),
                }
                start_ns = raw_phase.get("start_ns")
                if isinstance(start_ns, int) and not isinstance(start_ns, bool) and start_ns >= 0:
                    phase["start_ns"] = start_ns
                if isinstance(raw_phase.get("scope"), str):
                    phase["scope"] = raw_phase["scope"]
                if isinstance(raw_phase.get("includes"), list):
                    phase["includes"] = [
                        str(value) for value in raw_phase["includes"] if isinstance(value, str)
                    ]
                clean_phases.append(phase)
            if clean_phases:
                timing["worker_timing"] = {
                    "schema_version": "clozn.worker-timing.v1",
                    "unit": "nanoseconds",
                    "clock": (
                        worker_timing.get("clock")
                        if isinstance(worker_timing.get("clock"), str)
                        else "steady_clock"
                    ),
                    "clock_owner": "clozn_worker",
                    "phases": clean_phases,
                }
                by_name = {phase["name"]: phase for phase in clean_phases}
                if "prefill" in by_name:
                    timing["prefill_duration_ms"] = by_name["prefill"]["duration_ns"] / 1_000_000
                if "decode" in by_name:
                    timing["generation_duration_ms"] = by_name["decode"]["duration_ns"] / 1_000_000

            worker_metrics = worker_timing.get("metrics")
            if isinstance(worker_metrics, dict):
                prompt_rate = worker_metrics.get("prompt_tokens_per_second")
                decode_rate = worker_metrics.get("decode_tokens_per_second")
                if (
                    isinstance(prompt_rate, (int, float)) and not isinstance(prompt_rate, bool)
                    and math.isfinite(float(prompt_rate)) and prompt_rate >= 0
                ):
                    timing["prompt_tokens_per_second"] = prompt_rate
                if (
                    isinstance(decode_rate, (int, float)) and not isinstance(decode_rate, bool)
                    and math.isfinite(float(decode_rate)) and decode_rate >= 0
                ):
                    timing["generation_tokens_per_second"] = decode_rate
        wall_ms = obj.get("wall_ms")
        new_tokens = obj.get("new_tokens")
        steps_total = obj.get("steps_total")
        tok_per_s = obj.get("tok_per_s")
        if ("generation_duration_ms" not in timing
                and isinstance(wall_ms, (int, float)) and not isinstance(wall_ms, bool)
                and math.isfinite(float(wall_ms)) and wall_ms >= 0):
            timing["generation_duration_ms"] = wall_ms
        if isinstance(new_tokens, int) and not isinstance(new_tokens, bool) and new_tokens >= 0:
            timing["generation_tokens"] = new_tokens
        if isinstance(steps_total, int) and not isinstance(steps_total, bool) and steps_total >= 0:
            timing["generation_steps"] = steps_total
        if ("generation_tokens_per_second" not in timing
                and isinstance(tok_per_s, (int, float)) and not isinstance(tok_per_s, bool)
                and math.isfinite(float(tok_per_s)) and tok_per_s >= 0):
            timing["generation_tokens_per_second"] = tok_per_s
    return timing


def _norm_trace(trace) -> dict:
    """Coerce whatever a caller passes for `trace` into the stored shape."""
    if isinstance(trace, list):
        return steps_to_trace(trace)
    if isinstance(trace, dict):
        if isinstance(trace.get("steps"), list):
            norm = steps_to_trace(trace["steps"])
        else:
            norm = {
                k: trace[k]
                for k in ("tokens", "confidence", "alternatives", "token_ids", "logprobs", "topk_entropy")
                if k in trace
            }
            steps = _steps_from_parallel(norm)
            if steps:
                norm["steps"] = steps
        if "workspace_readouts" in trace:
            norm["workspace_readouts"] = trace["workspace_readouts"]
        if isinstance(trace.get("reasoning_steps"), list):
            norm["reasoning_steps"] = [
                step for i, raw in enumerate(trace["reasoning_steps"])
                if (step := _clean_step(raw, i)) is not None
            ]
        return {k: norm[k] for k in TRACE_KEYS if k in norm}
    return {}


def _normalize_workspace_readouts(rid: str, readouts) -> list[dict]:
    """Keep explicit readouts, filling the run id when a provider leaves it blank."""
    out = []
    for r in readouts or []:
        if not isinstance(r, dict):
            continue
        item = dict(r)
        item.setdefault("type", "workspace_readout")
        if not item.get("run_id"):
            item["run_id"] = rid
        if not item.get("provider_type") or not item.get("readout_kind"):
            try:
                from clozn.readouts import workspace_lens

                fields = workspace_lens.taxonomy_fields(item.get("provider"), item.get("readout_kind"))
                if fields.get("provider_type") and not item.get("provider_type"):
                    item["provider_type"] = fields["provider_type"]
                if fields.get("readout_kind") and not item.get("readout_kind"):
                    item["readout_kind"] = fields["readout_kind"]
            except Exception:
                pass
        out.append(item)
    return out


def _with_workspace_readouts(rid: str, trace: dict, workspace_provider=None) -> dict:
    """Attach explicit/provider Workspace Lens readouts to token traces."""
    if not isinstance(trace, dict) or not trace.get("tokens"):
        return trace
    if trace.get("workspace_readouts"):
        trace = dict(trace)
        trace["workspace_readouts"] = _normalize_workspace_readouts(rid, trace["workspace_readouts"])
        return trace
    if workspace_provider is None:
        return trace
    try:
        readouts = workspace_provider(rid, trace)
        readouts = _normalize_workspace_readouts(rid, readouts)
        if readouts:
            trace = dict(trace)
            trace["workspace_readouts"] = readouts
    except Exception:
        pass
    return trace
