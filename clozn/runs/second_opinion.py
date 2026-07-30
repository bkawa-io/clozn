"""clozn/runs/second_opinion.py -- E4: model second opinion (`clozn.model-second-opinion.v1`).

Same request across a SECOND, already-preloaded model; compare disagreement against the original run.
This module owns the pure-ish comparison logic; `clozn/server/routes/second_opinion.py` owns path
parsing, request validation, and worker selection (via `clozn.server.model_routing`).

WHY THE ANCHOR ARM (arm_a) NEVER CALLS A WORKER
--------------------------------------------------
"The original run stays the anchor" (owner's spec, verbatim) is enforced structurally, not just by
convention: `build_anchor_arm` reads only the run's own already-persisted fields (`response`, `identity`,
`context_receipt`, `timing`) and never touches a `sub`/`engine`. Two consequences, both deliberate:

  * A second opinion can be produced even when the ORIGINAL model is no longer loaded -- only the
    SECOND model needs to be a ready, preloaded worker.
  * arm_a can never independently fail the way a live call can. Its only degraded states are the ones
    the run's own recorded evidence already carries (`redacted`, `empty`, or -- on a very old record --
    simply `unavailable`), mirroring `clozn.runs.claims.build_answer_claims`'s own segmentation-state
    vocabulary for the identical underlying fact rather than inventing a second one.

WHY NO CLAIM-LEVEL SUPPORT COMPARISON (E2) IN v1
---------------------------------------------------
E1 (`clozn.answer-claims.v1`) and E2 (`clozn.claim-support.v1`) both key their text-span addressing off a
REAL, persisted run id -- `clozn.runs.claims.build_answer_claims` requires `run["id"]` and anchors every
claim's `text_span` to that id's `derived.claims` collection. The second-opinion arm's answer is
deliberately NEVER persisted as a run (this module's own discipline above, and the owner's spec: "never a
replacement" for the original run) -- so there is no real run id to anchor arm_b's claims to. Minting a
synthetic id to satisfy that signature would produce span addresses that resolve against nothing
(`GET /runs/<synthetic-id>/span-addresses` would 404 forever), which is exactly the structural dishonesty
the span-address contract exists to prevent. `compatibility.qualified_evidence` records this omission
explicitly (`state: "anchor_only"`) rather than silently leaving it out unremarked. The anchor's OWN
claim-support is unaffected and already reachable at `GET /runs/<id>/claim-support`; this module does not
duplicate it. This is the first thing to build in a v2 that gives the second-opinion arm its own
(non-run-anchored) span-address basis.

WHY "AGREEMENT" IS A LABELED LEXICAL PROXY, NOT SEMANTIC AGREEMENT
-----------------------------------------------------------------------
clozn is stdlib-only (no embeddings, no NLI model), and this module makes zero extra model calls beyond
the one second-opinion generation itself -- adding an LLM-as-judge call would be a THIRD, uncontrolled
generation with its own bias, cost, and latency, out of scope for v1. `comparison.agreement` reuses
`clozn.receipts.metrics.receipt_metrics`'s existing word-type Jaccard distance -- the SAME number
`POST /runs/<id>/retry`'s compare view already surfaces -- honestly labeled `lexical_overlap_heuristic`
with an explicit caveat that it is NOT a semantic or entailment judgment. This is the same crude-on-
purpose discipline `clozn.behavior.compare`'s own module docstring states for the corrective-retry compare
view; this module does not invent a new, less-honest number.

TOKEN PROBABILITIES: DELIBERATELY NEVER EXPOSED HERE
---------------------------------------------------------
Per the owner's spec, bolded: "NEVER present token probabilities from different models as calibrated
confidence." Two models' logprobs are not on a shared scale -- different vocabularies, different training
objectives, different calibration. This module never calls `/score` on either arm and never reads or
forwards a single per-token logprob/probability from either model. The only per-arm numbers here are
`finish_reason`, `latency_ms`, and token/word counts -- none of them a probability, all of them equally
meaningful (or meaningless) regardless of which two models are being compared. `clozn.model-second-
opinion.v1` has no field shaped to carry one. If a future version wants per-token detail, it must not
render it as a shared "confidence" axis between arms -- read this docstring again the day that work
starts.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

SCHEMA_VERSION = "clozn.model-second-opinion.v1"

_DEFAULT_MAX_NEW = 256


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_is_redacted(run: dict) -> bool:
    redaction = run.get("redaction")
    return bool(
        (isinstance(redaction, dict) and redaction.get("status") in {"redacted", "literal_redacted"})
        or "redacted" in (run.get("flags") or [])
    )


def _budget(run: dict) -> int:
    """Reuse the ORIGINAL run's own requested output cap for arm_b, mirroring
    `clozn.replay.corrective._original_budget` -- a second opinion answers the same request under the
    same output budget, not an arbitrary new one."""
    limits = ((run.get("context_receipt") or {}).get("limits") or {})
    value = limits.get("requested_max_tokens")
    if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 16384:
        return value
    return _DEFAULT_MAX_NEW


def _messages_basis(run: dict) -> tuple[list, str]:
    """The sanitized, delivered messages both the anchor's own recorded input and arm_b's request are
    built from, plus a stable content hash -- the concrete, checkable basis for
    `delivered_input.identical_across_arms` (see the module docstring: this implementation never varies
    the delivered messages per arm, so the hash is computed once and shared)."""
    from clozn.runs.think_tags import sanitize_messages
    messages = sanitize_messages(run.get("messages") or [])
    encoded = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return messages, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_anchor_arm(run: dict) -> dict:
    """arm_a -- the run's own recorded evidence. Never calls a worker; see module docstring."""
    arm: dict = {"role": "anchor", "run_id": str(run.get("id") or "")}
    model_id = run.get("model")
    if isinstance(model_id, str) and model_id:
        arm["model_id"] = model_id

    identity = run.get("identity") if isinstance(run.get("identity"), dict) else {}
    worker_identity: dict = {}
    for key in ("model_sha256", "template_fingerprint", "engine_build"):
        value = identity.get(key)
        if isinstance(value, str) and value:
            worker_identity[key] = value
    if worker_identity:
        arm["worker_identity"] = worker_identity

    response = run.get("response")
    response = response if isinstance(response, str) else None
    if _run_is_redacted(run):
        arm["status"] = "redacted"
    elif response is None:
        arm["status"] = "unavailable"
    elif response == "":
        arm["status"] = "empty"
    else:
        arm["status"] = "ok"
        arm["response_text"] = response

    finish_reason = run.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason:
        arm["finish_reason"] = finish_reason

    timing = run.get("timing") if isinstance(run.get("timing"), dict) else {}
    duration_ms = timing.get("duration_ms")
    if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
        arm["latency_ms"] = float(duration_ms)

    limits = (run.get("context_receipt") or {}).get("limits") or {}
    prompt_tokens = limits.get("prompt_tokens")
    if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
        arm["prompt_tokens"] = prompt_tokens
    generated_tokens = limits.get("generated_tokens")
    if isinstance(generated_tokens, int) and not isinstance(generated_tokens, bool):
        arm["generated_tokens"] = generated_tokens

    return arm


def _selection_worker_identity(selection) -> dict:
    runtime_key = selection.runtime_key or {}
    identity = selection.worker_identity or {}
    out: dict = {}
    template_fingerprint = runtime_key.get("template_fingerprint")
    if isinstance(template_fingerprint, str) and template_fingerprint:
        out["template_fingerprint"] = template_fingerprint
    artifact_sha = runtime_key.get("gguf_artifact_sha256")
    if isinstance(artifact_sha, str) and artifact_sha:
        out["model_sha256"] = artifact_sha
    engine_build = runtime_key.get("engine_build")
    if isinstance(engine_build, str) and engine_build:
        out["engine_build"] = engine_build
    backend = runtime_key.get("backend")
    if isinstance(backend, str) and backend:
        out["backend"] = backend
    context_size = runtime_key.get("context_size")
    if isinstance(context_size, int) and not isinstance(context_size, bool) and context_size > 0:
        out["context_size"] = context_size
    worker_id = identity.get("worker_id")
    if isinstance(worker_id, str) and worker_id:
        out["worker_id"] = worker_id
    generation = identity.get("worker_generation")
    if isinstance(generation, int) and not isinstance(generation, bool):
        out["worker_generation"] = generation
    return out


def run_second_opinion_arm(selection, *, requested_model_id: str, messages: list, budget: int) -> dict:
    """arm_b -- one fresh completion from `selection`'s already-resolved, identity-qualified worker,
    given `messages` (the SAME sanitized, delivered messages arm_a's own recorded input was built
    from). A failed attempt here is recorded as a typed outcome and returned -- it never raises, and the
    caller never needs to discard arm_a's evidence because of it (see module docstring)."""
    arm: dict = {
        "role": "second_opinion", "requested_model_id": requested_model_id,
        "model_id": selection.model_id,
    }
    worker_identity = _selection_worker_identity(selection)
    if worker_identity:
        arm["worker_identity"] = worker_identity

    if not messages:
        arm.update(status="refused", refusal={
            "code": "no_delivered_messages",
            "message": "this run has no recorded delivered messages to send to a second model",
        })
        return arm

    sub = selection.sub
    steps: list = []
    call_kw = {"max_new": budget, "sample": False, "trace_out": steps}
    t0 = time.time()
    try:
        while True:
            try:
                reply = sub.chat(messages, **call_kw)
                break
            except TypeError as exc:
                # A fake/older substrate that predates one of these kwargs -- drop the named one and
                # retry, exactly `clozn.replay.replay.replay`'s own progressive-degrade loop. Never
                # loses the reply over an instrumentation kwarg the substrate simply doesn't support.
                dropped = next((k for k in ("trace_out",) if k in call_kw and k in str(exc)), None)
                if dropped is None:
                    raise
                del call_kw[dropped]
    except Exception as exc:
        arm.update(status="generation_error", refusal={
            "code": "generation_error", "message": f"{type(exc).__name__}: {exc}",
        })
        return arm

    latency_ms = round((time.time() - t0) * 1000.0, 1)
    reply = reply if isinstance(reply, str) else str(reply)

    finish_reason = None
    if hasattr(sub, "last_finish_reason"):
        try:
            finish_reason = sub.last_finish_reason()
        except Exception:
            finish_reason = None

    arm.update(status="ok", response_text=reply, latency_ms=latency_ms)
    if isinstance(finish_reason, str) and finish_reason:
        arm["finish_reason"] = finish_reason
    if steps:
        arm["generated_tokens"] = len(steps)
    return arm


def _template_compat(arm_a: dict, arm_b: dict) -> dict:
    a_fp = (arm_a.get("worker_identity") or {}).get("template_fingerprint")
    b_fp = (arm_b.get("worker_identity") or {}).get("template_fingerprint")
    out: dict = {"method": "template_fingerprint_compare"}
    if isinstance(a_fp, str) and a_fp:
        out["arm_a_template_fingerprint"] = a_fp
    if isinstance(b_fp, str) and b_fp:
        out["arm_b_template_fingerprint"] = b_fp
    if not (a_fp and b_fp):
        out["state"] = "unknown"
    elif a_fp == b_fp:
        out["state"] = "same"
    else:
        out["state"] = "differs"
        out["caveat"] = (
            "the second model renders chat messages under a different template fingerprint than the "
            "original run recorded. Both arms received the same delivered messages (see "
            "delivered_input), but what each model actually SAW after templating differs -- a "
            "disagreement below is not attributable to model weights alone."
        )
    return out


def _context_limit_compat(arm_a: dict, arm_b: dict) -> dict:
    prompt_tokens = arm_a.get("prompt_tokens")
    window = (arm_b.get("worker_identity") or {}).get("context_size")
    out: dict = {"method": "arm_a_recorded_prompt_tokens_vs_arm_b_context_window"}
    if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
        out["arm_a_prompt_tokens_estimate"] = prompt_tokens
    if isinstance(window, int) and not isinstance(window, bool) and window > 0:
        out["arm_b_context_window_tokens"] = window
    known = (isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool)
            and isinstance(window, int) and not isinstance(window, bool) and window > 0)
    if not known:
        out["state"] = "unknown"
    elif prompt_tokens >= window:
        out["state"] = "exceeds_estimate"
        out["caveat"] = (
            "the original run's own recorded prompt token count already meets or exceeds the second "
            "model's context window. This is an ESTIMATE ONLY -- measured under the first model's own "
            "tokenizer, not the second model's (different tokenizers segment identical text into "
            "different token counts) -- but a second-opinion generation attempted despite this may "
            "truncate or fail."
        )
    else:
        out["state"] = "within_estimate"
    return out


def _tools_schema_compat(run: dict) -> dict:
    contract = run.get("output_contract")
    if not contract:
        return {"state": "none_used"}
    out: dict = {
        "state": "used_not_replayed",
        "caveat": (
            "the original request used a tool/structured-output contract; the second-opinion arm is a "
            "plain chat completion and does not replay that contract, so its answer may not be directly "
            "comparable in form (e.g. natural-language text vs. a tool call or JSON payload)."
        ),
    }
    mode = contract.get("mode") if isinstance(contract, dict) else None
    if isinstance(mode, str) and mode:
        out["requested_mode"] = mode
    return out


_QUALIFIED_EVIDENCE_NOTE = (
    "claim-level support comparison (E2) is available for the anchor run alone via "
    "GET /runs/<id>/claim-support; the second-opinion arm is not a persisted run in v1, so no "
    "source-influence measurement exists to qualify its claims. See clozn/runs/second_opinion.py's "
    "module docstring for why."
)


def _build_comparison(arm_a: dict, arm_b: dict) -> dict | None:
    if arm_a.get("status") != "ok" or arm_b.get("status") != "ok":
        return None
    a_text, b_text = arm_a.get("response_text") or "", arm_b.get("response_text") or ""
    if not a_text or not b_text:
        return None

    from clozn.behavior.compare import compare_metrics
    from clozn.receipts.metrics import receipt_metrics

    metrics = receipt_metrics(a_text, b_text)
    extra = compare_metrics(a_text, b_text)
    words = metrics.get("words") or [0, 0]
    return {
        "agreement": {
            "method": "lexical_overlap_heuristic",
            "lexical_difference_percent": int(metrics.get("changed") or 0),
            "caveat": (
                "word-type Jaccard DISTANCE between the two answers (0 = identical word-type sets, 100 "
                "= completely disjoint) -- the SAME heuristic clozn's corrective-retry compare view "
                "already uses. This is NOT a semantic, entailment, or truth-preserving judgment: two "
                "answers phrased differently but agreeing in substance can score high here, and two "
                "answers sharing boilerplate wording while disagreeing in substance can score low. No "
                "embedding or NLI model is used, and no token probability from either model factors "
                "into this number."
            ),
        },
        "format_changed": bool(extra.get("format_changed")),
        "length": {"arm_a_words": int(words[0]), "arm_b_words": int(words[1])},
    }


def build_second_opinion(run: dict, selection, *, requested_model_id: str) -> dict:
    """Build and validate one `clozn.model-second-opinion.v1` document. `selection` is an already-
    resolved, identity-qualified `clozn.server.model_routing.ModelSelection` for the SECOND model
    (never the run's own model -- see the route module). Never raises on arm_b's behalf: any failure
    generating arm_b is captured inside `arm_b` itself by `run_second_opinion_arm`."""
    run_id = str(run.get("id") or "")
    messages, basis_sha256 = _messages_basis(run)
    budget = _budget(run)

    arm_a = build_anchor_arm(run)
    arm_b = run_second_opinion_arm(
        selection, requested_model_id=requested_model_id, messages=messages, budget=budget)

    compatibility = {
        "chat_template": _template_compat(arm_a, arm_b),
        "context_limit": _context_limit_compat(arm_a, arm_b),
        "tools_or_schema": _tools_schema_compat(run),
        "qualified_evidence": {"state": "anchor_only", "note": _QUALIFIED_EVIDENCE_NOTE},
    }

    document: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "run_id": run_id,
        "delivered_input": {
            "message_count": len(messages),
            "sha256": basis_sha256,
            "identical_across_arms": True,
        },
        "arm_a": arm_a,
        "arm_b": arm_b,
        "compatibility": compatibility,
    }

    comparison = _build_comparison(arm_a, arm_b)
    if comparison is not None:
        document["comparison"] = comparison

    from clozn import schemas
    schemas.validate(document)
    return document


__all__ = [
    "SCHEMA_VERSION",
    "build_anchor_arm",
    "build_second_opinion",
    "run_second_opinion_arm",
]
