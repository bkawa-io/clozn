"""Torch-free product adapter for the supervised C++ GGUF worker.

The app module owns live server state and route injection. This module reads that state through
``ctx`` so request-scoped worker selection and test doubles remain late-bound.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import os
import sys
import threading
import time
from typing import Any

from clozn.experiments.effective_prompt import inject_block
from clozn.server.config import REPO_ROOT, DEMO                        # noqa: F401
from clozn.server import app as ctx   # the seam: live server state + patchable helpers (see docstring)
from clozn.server.request_context import RequestContext   # backlog #2: per-request isolation (see EngineSubstrate)


def _minimal_context_batch_workers() -> int:
    """Bounded opt-in concurrency for independent engine context slots."""
    raw = os.environ.get("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "2")
    try:
        return max(1, min(8, int(raw)))
    except (TypeError, ValueError):
        return 2


def _experimental_native_reference_match_arms_enabled() -> bool:
    """Opt into the native-many wire path only for explicit measurements.

    The worker advertises the capability so benchmark tooling can discover it,
    but the product proof path remains scalar until a real-GGUF parity suite
    has qualified the execution regime.
    """
    return os.environ.get("CLOZN_ENABLE_NATIVE_REFERENCE_MATCH_ARMS", "").lower() in {
        "1", "true", "yes", "on",
    }


def _experimental_parent_anchor_enabled() -> bool:
    """Opt into the Phase-B parent-anchor path; never enable it for proof arms."""
    return os.environ.get("CLOZN_ENABLE_NATIVE_PARENT_ANCHOR", "").lower() in {
        "1", "true", "yes", "on",
    }



def _engine_discovery_context(engine_health=None, configured=None) -> dict:
    """Combine worker-reported build facts with supervisor-owned executable identity."""
    out = {}
    if isinstance(engine_health, Mapping):
        for field in ("backend", "engine_version", "build_id", "llama_cpp_commit"):
            value = engine_health.get(field)
            if value not in (None, ""):
                out[field] = value
    if isinstance(configured, Mapping):
        out.update({key: value for key, value in configured.items() if value not in (None, "")})
    return out


class EngineSubstrate:
    """PURE-ENGINE substrate: chat on the C++ GGUF runtime, NO PyTorch model resident. THIS is the class
    that brings the v1 text API and analyses onto the native runtime. It carries no memory,
    personalization, mechanistic readout, or steering state."""

    name = "engine"
    # The score-many seam is a scheduling wrapper around the same scalar
    # ``/score`` contract (including bounded concurrency), not a native model
    # batch. The explicit strategy declaration keeps that distinction visible.
    score_tokens_many_proof_grade = True

    @property
    def score_tokens_many_execution_strategy(self):
        return "scalar_http_concurrent" if _minimal_context_batch_workers() > 1 else "scalar_serial"

    # IDENTITY LAZY RE-RESOLUTION (engine-down pressure test finding #2): a down-at-startup engine pays
    # this ~2s connect-refused tax (this host's control fact) at most once per cooldown window on lazy
    # re-resolve attempts (see _maybe_reresolve_identity) -- never on every request, so a persistently-dead
    # engine never adds latency to ordinary calls.
    _IDENTITY_RETRY_COOLDOWN_S = 30.0

    def __init__(self, engine=None):
        if engine is None:
            raise RuntimeError("engine substrate needs a supervised GGUF worker")
        self.engine = engine
        # T0.2: reflect the ACTUALLY-LOADED GGUF, not a hardcoded Qwen assumption. Derive the family from
        # the engine's /health model file (best-effort -- never blocks boot if the engine isn't up yet).
        # run_meta() re-derives this lazily too, so the run record is correct even when the engine comes
        # up after the substrate.
        #
        # model_family/model_id/model_sha256 are PROPERTIES (below), backed by the _val fields here: a
        # startup-time engine outage must not permanently wedge identity resolution for the rest of the
        # process's life -- every read retries _resolve_identity() (cooldown-gated) while unresolved, and
        # never re-fetches once it resolves. See _resolve_identity/_maybe_reresolve_identity.
        self._model_family_val = None
        self._model_id_val = None
        self._model_sha256_val = None
        self._identity_lock = threading.Lock()
        self._identity_last_attempt = 0.0
        self._native_reference_match_arms = False
        self.last_native_reference_match_metrics = None
        self._request: RequestContext | None = None
        self._resolve_identity()

    def _resolve_identity(self):
        """One best-effort attempt to derive model_family/model_id/model_sha256 from the engine's /health.
        Never blocks boot or raises: an unreachable worker leaves identity unresolved. Called once at
        construction and retried lazily while the identity remains unresolved."""
        self._identity_last_attempt = time.time()
        h = {}
        try:
            h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
            capabilities = (h or {}).get("capabilities") if isinstance(h, dict) else {}
            self._native_reference_match_arms = bool(
                isinstance(capabilities, dict) and capabilities.get("reference_match_arms") is True
            )
            fam, _info = _engine_model_info((h or {}).get("model", ""))
            self._model_family_val = fam
            self._model_id_val = _info["model_id"]
        except Exception:
            return
        sha256 = str((h or {}).get("model_sha256") or "") or None
        if not sha256:
            return
        self._model_sha256_val = sha256

    def _maybe_reresolve_identity(self):
        """Retry _resolve_identity() iff identity is still unresolved (no model_sha256 yet) AND the
        cooldown has elapsed since the last attempt. A no-op once resolved and within the cooldown
        window so a persistently-down engine doesn't add a health() round-trip to every request."""
        if self._model_sha256_val:
            return
        if time.time() - self._identity_last_attempt < self._IDENTITY_RETRY_COOLDOWN_S:
            return
        with self._identity_lock:               # double-checked, so concurrent callers don't stack retries
            if self._model_sha256_val:
                return
            if time.time() - self._identity_last_attempt < self._IDENTITY_RETRY_COOLDOWN_S:
                return
            self._resolve_identity()

    @property
    def model_family(self):
        self._maybe_reresolve_identity()
        return self._model_family_val

    @property
    def model_id(self):
        self._maybe_reresolve_identity()
        return self._model_id_val

    @property
    def model_sha256(self):
        self._maybe_reresolve_identity()
        return self._model_sha256_val

    # ---- per-request context -------------------------------------------------------------------------
    # chat()/chat_stream() each start with self._new_request(), then write everything the call learns
    # about itself onto that one object. Public accessors below expose the completed request state.
    def _new_request(self) -> RequestContext:
        """Start this call's RequestContext and atomically publish it as the current request."""
        self._request = RequestContext()
        return self._request

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None,
             reference_tokens=None, stop=None):
        """One stateless chat completion on the engine.

        `sample`: the caller's request to sample (True), force greedy (False), or override this request's
        sampling fields with a dict. REPRODUCE_AND_PROVE_PLAN S5: `sample=True` (the default) resolves via
        ctx._resolve_sampling against the persisted "sampling" setting (default ON, with
        temperature=0.8/top_p=0.9/top_k=40/repeat_penalty=1.1 and a fresh seed every turn); the setting
        off degrades to greedy, byte-identical to pre-S5 behavior.
        `sample=False` ALWAYS decodes greedy (temperature 0) regardless of the setting -- this is what the
        receipt/replay/forced-scoring stack relies on: it forces the STORED token ids over this generation
        (replay.py passes `sample=False` for every `{"greedy": True}` change spec, which every receipt
        path uses), so a sampled interactive run's receipts are computed exactly like a greedy run's.

        `reference_tokens` (optional): the baseline reply's committed token ids. When present, the engine
        EARLY-STOPS this generation at the first token that differs from the reference (prove-all ablated
        arms) -- so the reply is a bit-exact PREFIX of what full generation would produce, plus a divergence
        verdict stashed for last_divergence(). This is a pure termination check -- decode/sampling (greedy
        or not) are otherwise untouched, so a diverged reply is still a bit-exact prefix either way.

        REQUEST ISOLATION: this call publishes one RequestContext through self._new_request(); all
        generation metadata, divergence state, trace data, and termination facts belong to that object."""
        req = self._new_request()
        samp = ctx._resolve_sampling(sample)
        req.sampling = samp
        req.generation_meta = ctx._engine_generation_meta(max_new, stream=False, sample=samp, stop=stop)
        # (A topic-gated memory-card block was composed and injected here until the 2026-07-27 cards cut.
        # Nothing prepends a system block on this path now, so the messages reach the template as given.)
        template_usage = {}
        gateway_phases = []
        template_started_ns = time.monotonic_ns()
        try:
            prompt = ctx._engine_tmpl(self.engine, messages, usage_out=template_usage)
        finally:
            gateway_phases.append({
                "name": "prompt_template", "owner": "clozn_gateway",
                "duration_ns": max(0, time.monotonic_ns() - template_started_ns),
                "measurement": "measured", "aggregation": "exclusive",
            })
        if mem_out is not None:
            # final_prompt = the EXACT rendered string the model saw (backlog #5); assembled_messages is its
            # pre-template form. Both recorded so the run is inspectable at either level.
            mem_out.update(assembled_messages=list(messages), final_prompt=prompt)
        kw = {}
        if reference_tokens:                                # prove-all early-stop: halt when the answer changes
            kw["reference_tokens"] = [int(t) for t in reference_tokens if t is not None]
        usage = {}
        traced_kw = {"sample": samp, "stop": stop}
        try:
            import inspect
            params = inspect.signature(ctx._engine_complete_traced).parameters.values()
            if ("usage_out" in inspect.signature(ctx._engine_complete_traced).parameters
                    or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)):
                traced_kw["usage_out"] = usage
        except Exception:
            pass
        dispatch_started_ns = time.monotonic_ns()
        try:
            reply_raw, steps, finish, divinfo = ctx._engine_complete_traced(
                self.engine, prompt, max_new, kw, **traced_kw)
        finally:
            gateway_phases.append({
                "name": "worker_dispatch", "owner": "clozn_gateway",
                "duration_ns": max(0, time.monotonic_ns() - dispatch_started_ns),
                "measurement": "measured", "aggregation": "overlapping",
            })
            from clozn.runs.perf_spans import timing_document
            req.gateway_timing = timing_document(gateway_phases)
        if isinstance(usage.get("prompt_tokens"), int):
            req.prompt_tokens = usage["prompt_tokens"]
        if isinstance(usage.get("completion_tokens"), int):
            req.completion_tokens = usage["completion_tokens"]
        if isinstance(usage.get("termination"), dict):
            req.termination = dict(usage["termination"])
        req.generation_timing = dict(usage.get("generation_timing") or {})
        raw_reason = usage.get("raw_finish_reason")
        req.finish_reason_raw = raw_reason if isinstance(raw_reason, str) else None
        req.finish_reason = finish                          # stash for last_finish_reason() (the log path)
        req.diverged, req.diverged_at = divinfo             # stash for last_divergence()
        # (The anchored-memory loop guard lived here -- it only ran when an anchored bag had actually
        # been injected this turn, which no longer happens. Generic repetition detection survives as
        # clozn/runs/degeneracy.py, used by runs/signals.py.)
        if trace_out is not None:
            trace_out.extend(steps)
        req.trace = list(steps)
        if mem_out is not None:
            req.prompt_manifest = dict(mem_out)
        return reply_raw.strip()

    def last_divergence(self):
        """The early-stop verdict from the most recent chat(): (diverged, diverged_at). (None, None) when
        the last chat carried no reference_tokens. Read by replay to record whether an ablated arm's reply
        was truncated at the point it provably changed."""
        request = getattr(self, "_request", None)
        if request is None:
            return None, None
        return request.diverged, request.diverged_at

    def score_tokens(self, messages, continuation_ids=None, *, continuation=None, block=None, topk=0):
        """Teacher-forced per-token logprob under an explicit context block.

        Assembles the prompt EXACTLY like chat() (inject_block + ctx._engine_tmpl -- the loaded
        model's own chat template), but from the caller's ``block`` rather than live state. Calls with
        different explicit blocks, the same messages, and the same continuation are directly
        comparable. No sampling occurs.

        `block`: an explicit system-context block string, or None to omit it.
        `continuation_ids`: the PRIMARY continuation form (token ids, e.g. from a stored trace) --
        takes precedence over `continuation` when both are given (mirrors EngineClient.score).
        `continuation`: a TEXT fallback (S3's rederive.py, for a run whose trace lacks per-token ids) --
        the engine retokenizes it independently of the prompt, which can drift at the prompt/
        continuation BPE boundary (flagged `boundary_approximate` by /score itself; see
        REPRODUCE_AND_PROVE_PLAN.md's tokenization-boundary caveat).
        Returns [{"id", "piece", "logprob"}, ...] (+ "topk" per token when topk>0), one entry per
        continuation token, in the SAME order as continuation_ids (or the engine's own retokenization
        of `continuation` text).
        """
        assembled = inject_block(messages, block)
        prompt = ctx._engine_tmpl(self.engine, assembled)
        kw = {}
        if continuation_ids is not None:
            kw["continuation_ids"] = [int(t) for t in continuation_ids]
        elif continuation is not None:
            kw["continuation"] = str(continuation)
        r = self.engine.score(prompt=prompt, topk=int(topk), **kw)
        return r.get("tokens", [])

    def score_tokens_many(self, arms, *, cancel=None):
        """Score arms through the trusted scalar worker route with bounded concurrency."""
        from clozn.experiments.multi_arm import concurrent_many, serial_many

        workers = _minimal_context_batch_workers()
        if workers <= 1:
            return serial_many(self.score_tokens, arms, cancel=cancel)
        return concurrent_many(
            self.score_tokens, arms, cancel=cancel, max_workers=workers,
        )
    def probe_reference_match(self, messages, reference_token_ids, *, generation_contract,
                              explicit_conditions=None):
        """Run one non-journaling exact recorded-answer probe.

        This deliberately does not call :meth:`chat`: chat resolves live
        sampling/settings and publishes request state. Every decode condition comes from the supplied
        contract while the worker's normal template and generation path remain the source of truth.
        The method returns a detached evidence dictionary;
        it creates no child run and writes no runlog state.
        """
        from clozn.runs.answer_preservation import (
            ExactAnswerPreservationError,
            classify_reference_match,
        )

        if not isinstance(generation_contract, dict):
            raise ExactAnswerPreservationError("generation_contract must be a mapping")
        mode = generation_contract.get("decode_mode")
        max_new = generation_contract.get("max_new")
        stop = generation_contract.get("stop")
        expected = generation_contract.get("expected_termination")
        if (mode not in {"greedy", "sample"} or isinstance(max_new, bool)
                or not isinstance(max_new, int) or max_new < 1):
            raise ExactAnswerPreservationError("generation_contract is incomplete")
        if not isinstance(stop, list) or any(not isinstance(item, str) for item in stop):
            raise ExactAnswerPreservationError("generation_contract.stop must be a list of strings")
        if (not isinstance(expected, dict) or not isinstance(expected.get("reason"), str)
                or not isinstance(expected.get("reason_raw"), str)):
            raise ExactAnswerPreservationError("generation_contract.expected_termination is incomplete")
        if not isinstance(reference_token_ids, list) or not reference_token_ids or any(
                isinstance(value, bool) or not isinstance(value, int) for value in reference_token_ids):
            raise ExactAnswerPreservationError("reference_token_ids must be a non-empty integer list")

        sampling = generation_contract.get("sampling")
        if mode == "sample":
            if not isinstance(sampling, dict):
                raise ExactAnswerPreservationError("sampled exact probes require a complete sampler")
            required = ("temperature", "top_p", "top_k", "repeat_penalty", "seed")
            if any(key not in sampling for key in required):
                raise ExactAnswerPreservationError("sampled exact probes require all sampler fields")
            if (isinstance(sampling["temperature"], bool) or not isinstance(sampling["temperature"], (int, float))
                    or isinstance(sampling["top_p"], bool) or not isinstance(sampling["top_p"], (int, float))
                    or isinstance(sampling["repeat_penalty"], bool) or not isinstance(sampling["repeat_penalty"], (int, float))
                    or isinstance(sampling["top_k"], bool) or not isinstance(sampling["top_k"], int)
                    or isinstance(sampling["seed"], bool) or not isinstance(sampling["seed"], int)):
                raise ExactAnswerPreservationError("sampled exact probes have invalid sampler fields")
            sample = {
                "on": True,
                "temperature": float(sampling["temperature"]),
                "top_p": float(sampling["top_p"]),
                "top_k": int(sampling["top_k"]),
                "repeat_penalty": float(sampling["repeat_penalty"]),
                "seed": int(sampling["seed"]),
            }
        else:
            sample = None

        conditions = dict(explicit_conditions or {})
        block = conditions.get("block")
        assembled = inject_block(messages, block) if block is not None else list(messages)
        template_usage = {}
        try:
            prompt = ctx._engine_tmpl(self.engine, assembled, usage_out=template_usage)
        except TypeError:
            prompt = ctx._engine_tmpl(self.engine, assembled)
        kw = {"reference_tokens": [int(value) for value in reference_token_ids]}
        usage = {}
        traced_kwargs = {"sample": sample, "stop": stop}
        try:
            import inspect
            signature = inspect.signature(ctx._engine_complete_traced)
            if "usage_out" in signature.parameters or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()):
                traced_kwargs["usage_out"] = usage
        except Exception:
            traced_kwargs["usage_out"] = usage
        reply, steps, finish, divinfo = ctx._engine_complete_traced(
            self.engine, prompt, max_new, kw, **traced_kwargs)
        # ``accumulate_ar_events`` uses the normalized trace key ``id``; accept the older
        # ``token_id`` spelling as well so the scalar exact-probe path does not discard a
        # perfectly valid real-worker token trace.
        generated_ids = [step.get("token_id", step.get("id"))
                         for step in steps if isinstance(step, dict)]
        if not generated_ids or any(isinstance(value, bool) or not isinstance(value, int) for value in generated_ids):
            return {"status": "unavailable", "reason": "probe_token_trace_unavailable"}
        diverged, diverged_at = divinfo if isinstance(divinfo, tuple) else (None, None)
        result = classify_reference_match(
            reference_token_ids, generated_ids, diverged=diverged, diverged_at=diverged_at,
            termination=usage.get("termination"), finish_reason=finish,
            expected_termination=expected, max_new=max_new,
        )
        result.update({
            "generated_token_ids": list(generated_ids),
            "finish_reason": finish,
            "termination": dict(usage.get("termination") or {}),
            "reply": reply,
        })
        return result

    def probe_reference_match_many(self, arms, *, cancel=None, proof_grade=True):
        """Batch seam for exact recorded-output probes.

        Exact probes retain the scalar early-stop and termination semantics;
        the default path merely schedules independent arms and never calls
        ``chat`` or publishes a RequestContext.  The native engine adapter is
        opt-in and explicitly non-proof-grade until real-GGUF parity is proven.
        """
        from clozn.experiments.multi_arm import concurrent_many, serial_many
        from clozn.runs.answer_preservation import classify_reference_match

        self.last_native_reference_match_metrics = None
        native_arms = [dict(arm) for arm in arms]
        if (not proof_grade and (_experimental_native_reference_match_arms_enabled()
                                 or _experimental_parent_anchor_enabled())
                and self._native_reference_match_arms
                and native_arms
                and callable(getattr(self.engine, "reference_match_arms", None))):
            prepared = []
            native_supported = True
            render_started_ns = time.perf_counter_ns()
            common_reference = native_arms[0].get("reference_token_ids") if native_arms else None
            common_contract = native_arms[0].get("generation_contract") if native_arms else None
            parent_anchor_prompts = {
                arm.get("parent_anchor_prompt") for arm in native_arms
                if isinstance(arm.get("parent_anchor_prompt"), str)
            }
            parent_anchor_prompt = (
                next(iter(parent_anchor_prompts))
                if len(parent_anchor_prompts) == 1 and _experimental_parent_anchor_enabled()
                else None
            )
            if _experimental_parent_anchor_enabled() and len(parent_anchor_prompts) != 1:
                native_supported = False
            for arm in native_arms:
                contract = arm.get("generation_contract")
                conditions = arm.get("explicit_conditions") or {}
                if (not isinstance(contract, dict) or contract.get("decode_mode") != "greedy"
                        or arm.get("reference_token_ids") != common_reference
                        or contract != common_contract):
                    native_supported = False
                    break
                messages = list(arm.get("messages") or [])
                block = conditions.get("block")
                assembled = inject_block(messages, block) if block is not None else messages
                prompt = ctx._engine_tmpl(self.engine, assembled)
                prepared.append({"arm_id": len(prepared), "prompt": prompt})
            if native_supported:
                try:
                    native_kwargs = {
                        "reference_token_ids": list(native_arms[0]["reference_token_ids"]),
                        "generation_contract": dict(native_arms[0]["generation_contract"]),
                    }
                    if parent_anchor_prompt is not None:
                        native_kwargs["parent_anchor_prompt"] = parent_anchor_prompt
                    response = self.engine.reference_match_arms(prepared, **native_kwargs)
                    rows = response.get("results") if isinstance(response, dict) else None
                    if isinstance(rows, list) and len(rows) == len(prepared):
                        self.last_native_reference_match_metrics = dict(response.get("metrics") or {})
                        if parent_anchor_prompt is not None:
                            self.last_native_reference_match_metrics["parent_anchor_enabled"] = True
                            self.last_native_reference_match_metrics["proof_grade"] = False
                        if not self.last_native_reference_match_metrics.get("prompt_rendering_time_ns"):
                            self.last_native_reference_match_metrics[
                                "prompt_rendering_time_ns"
                            ] = max(0, time.perf_counter_ns() - render_started_ns)
                        output = []
                        for row in rows:
                            raw = dict(row.get("result") or {})
                            generated = raw.get("generated_token_ids")
                            if not isinstance(generated, list) or any(
                                    isinstance(token, bool) or not isinstance(token, int)
                                    for token in generated):
                                raise ValueError("native reference-match row has no integer token trace")
                            expected = native_arms[int(row["arm_id"])].get("generation_contract", {}).get(
                                "expected_termination")
                            classified = classify_reference_match(
                                list(native_arms[0]["reference_token_ids"]), generated,
                                diverged=raw.get("diverged"),
                                diverged_at=raw.get("diverged_at"),
                                termination=raw.get("termination"),
                                finish_reason=raw.get("finish_reason"),
                                expected_termination=expected,
                                max_new=native_arms[0]["generation_contract"]["max_new"],
                            )
                            classified.update({
                                "generated_token_ids": list(generated),
                                "finish_reason": raw.get("finish_reason"),
                                "termination": dict(raw.get("termination") or {}),
                                "reply": raw.get("reply", ""),
                            })
                            output.append({"arm_index": int(row["arm_id"]), "result": classified})
                        return output
                except Exception:
                    # A stale/unsupported worker must degrade to the existing
                    # scalar path rather than turning an experiment into proof.
                    self.last_native_reference_match_metrics = None

        workers = _minimal_context_batch_workers()
        if workers <= 1:
            return serial_many(self.probe_reference_match, native_arms, cancel=cancel)
        return concurrent_many(self.probe_reference_match, native_arms, cancel=cancel, max_workers=workers)

    def last_stream_trace(self):
        """Per-token trace from the most recent completed or closed chat stream."""
        request = getattr(self, "_request", None)
        return list(request.trace) if request is not None else []

    def last_finish_reason(self):
        """The stop cause ("stop"|"length"|...) from the most recent chat()/chat_stream, or None. Same
        stash-and-read contract as last_stream_trace: the handler reads it AFTER generation, so the run
        logs WHY the engine stopped instead of a hard-coded 'stop'."""
        request = getattr(self, "_request", None)
        return request.finish_reason if request is not None else None

    def last_finish_reason_raw(self):
        """The worker event's raw stop cause before protocol normalization, when available."""
        request = getattr(self, "_request", None)
        return request.finish_reason_raw if request is not None else None

    def last_prompt_tokens(self):
        """The prompt token count the engine's own `gen_started` frame reported for the most recent
        chat_stream, or None (a non-streaming chat() call, an engine build too old to send the field, or
        nothing streamed yet). Same stash-and-read contract as last_stream_trace/last_finish_reason --
        exposed for request accounting when the worker reports it; never guessed when absent."""
        request = getattr(self, "_request", None)
        return request.prompt_tokens if request is not None else None

    def run_meta(self):
        """Reproducibility metadata -- WHAT produced a run -- for the run record. Fetched once from
        /health (model file -> quant, engine mode) and cached; the STATIC baseline here is the honest
        greedy default (temperature 0) -- the ACTUAL regime a specific reply used (greedy, or S5 sampled
        with its params + seed) rides in the current RequestContext, filled by chat()/chat_stream() and
        merged below, so a call made before any generation still reports the honest baseline rather than a
        guess. Health-derived fields are omitted when unavailable rather than guessed. Never raises:
        metadata never breaks a run."""
        health_meta = getattr(self, "_run_meta", None)
        if health_meta is None:
            health_meta = {}
            h: dict = {}
            mp = ""
            try:
                h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
                mp = str((h or {}).get("model", ""))
                if mp:
                    health_meta["model_file"] = mp.replace("\\", "/").rsplit("/", 1)[-1]
                    q = _quant_from_name(health_meta["model_file"])
                    if q:
                        health_meta["quant"] = q
                    # T0.2: which model actually produced this run (derived from the loaded GGUF, not a
                    # hardcoded id). family is the registry key; model_id the friendly HF name when known.
                    fam, info = _engine_model_info(mp)
                    if fam:
                        health_meta["family"] = fam
                    if info.get("model_id"):
                        health_meta["model_id"] = info["model_id"]
                if (h or {}).get("mode"):
                    health_meta["mode"] = h["mode"]
                for k in ("n_ctx", "device", "gpu_layers"):
                    v = (h or {}).get(k)
                    if v is not None:
                        health_meta[k] = v
            except Exception:
                pass
            # roadmap S4.3: immutable reproduction identity, assembled from the SAME /health fetch above --
            # never a second round trip, never a fresh file hash on the request path. Prefers the engine's
            # own reported model_sha256 (computed once at boot, see clozn.runs.identity's module
            # docstring); only falls back to hashing model_path itself when the engine didn't report one,
            # and even that fallback is cached process-wide by _identity_meta_val (below) plus
            # clozn.runs.identity's own on-disk cache. A failure here must not cost the health_meta this
            # call already earned, so it gets its own try/except rather than sharing the one above.
            try:
                from clozn.runs import identity as run_identity
                self._identity_meta_val = run_identity.runtime_identity(
                    model_path=mp or None,
                    model_sha256_hint=(h or {}).get("model_sha256"),
                    apply_template_fn=getattr(self.engine, "apply_template", None),
                    engine_health=h if isinstance(h, dict) else None,
                    extra_context=_engine_discovery_context(
                        h, getattr(self, "_engine_artifact_context", None)
                    ),
                )
            except Exception:
                self._identity_meta_val = {}
            self._run_meta = dict(health_meta)
        meta = ctx._engine_generation_meta()
        meta.update(dict(health_meta))
        request = getattr(self, "_request", None)
        meta.update(dict(request.generation_meta) if request is not None else {})
        meta.update(dict(getattr(request, "generation_timing", None) or {}))
        gateway_timing = getattr(request, "gateway_timing", None)
        if isinstance(gateway_timing, dict) and gateway_timing.get("phases"):
            meta["gateway_timing"] = dict(gateway_timing)
        raw_reason = getattr(request, "finish_reason_raw", None)
        if isinstance(raw_reason, str) and raw_reason:
            meta["finish_reason_raw"] = raw_reason
            meta["finish_reason_source"] = "worker_event"
        prompt_tokens = request.prompt_tokens if request is not None else None
        if isinstance(prompt_tokens, int):
            meta["prompt_tokens"] = prompt_tokens
        completion_tokens = getattr(request, "completion_tokens", None)
        if isinstance(completion_tokens, int):
            meta["completion_tokens"] = completion_tokens
            if isinstance(prompt_tokens, int):
                meta["total_tokens"] = prompt_tokens + completion_tokens
        termination = getattr(request, "termination", None)
        if isinstance(termination, dict) and termination:
            meta["termination"] = dict(termination)
        return dict(meta)

    def identity_meta(self) -> dict:
        """The run record's top-level `identity` block (roadmap S4.3): model_sha256, model_path,
        model_size_bytes, template_fingerprint, engine_build, clozn_version, captured_at -- whichever of
        those this process could actually establish. Computed once per process inside run_meta()'s single
        /health fetch (see its comment) and cached in self._identity_meta_val, so calling this never adds
        a network round trip or a file read beyond what run_meta() already pays for. Calls run_meta()
        first if it hasn't run yet this process; never raises."""
        if getattr(self, "_identity_meta_val", None) is None:
            try:
                self.run_meta()
            except Exception:
                pass
        return dict(getattr(self, "_identity_meta_val", None) or {})

    def chat_stream(self, messages, max_new=256, mem_out=None, sample=True, stop=None):
        """Streaming twin of chat(): the SAME prompt assembly
        (kept in lockstep -- see chat()'s comments; do not let this drift from that logic), but opens the engine's
        /v1/completions with stream:True (mirrors _engine_complete_traced's request) and yields text as
        the engine commits it, instead of waiting on one blocking call. This is what makes /v1/chat/
        completions's SSE branch (_sse_chat, gated on `getattr(SUB, "chat_stream", None)`) fire on the
        pure-engine substrate too -- before this existed, a streaming request here silently fell through
        to one non-streamed chat() reply. mem_out: as in chat() -- records what was actually assembled
        and sent this turn (assembled_messages/final_prompt).

        Per-token trace (the B3 contract): every parsed SSE frame is
        collected, then folded into the request context via runlog.accumulate_ar_events once the
        stream ends -- normal completion OR an early GeneratorExit (the consumer stopped early) -- so a
        partial stream still logs whatever trace it managed. Wrapped so any parse hiccup just leaves it
        [], same as the non-streaming path's fallback.

        SAMPLING (S5): `sample` has the same bool-or-override-dict contract as chat(). The OpenAI SSE route
        uses the dict form so explicit temperature/top_p/top_k/repeat_penalty/seed values affect the stream;
        ordinary callers use True and inherit the persisted defaults. The master setting off degrades to
        greedy, byte-identical to pre-S5 behavior.

        REQUEST ISOLATION + CANCELLATION: this call's own RequestContext (self._new_request()) owns all
        generated state. The context also carries a cancellation Event: sse.py's caller sets it (via
        self._request.cancel()) the instant it detects the CLIENT is gone (a failed write to the far end),
        and the read loop below checks it between worker frames as a second, belt-and-suspenders stop
        alongside the GeneratorExit an explicit `gen.close()` throws at the `yield` below -- either one
        aborts the worker's chunked send promptly instead of draining a reply nobody will read."""
        import urllib.error
        import urllib.request
        import clozn.runs.store as runlog
        req = self._new_request()
        samp = ctx._resolve_sampling(sample)
        req.sampling = samp
        req.generation_meta = ctx._engine_generation_meta(max_new, stream=True, sample=samp, stop=stop)
        template_usage = {}
        gateway_phases = []
        template_started_ns = time.monotonic_ns()
        try:
            prompt = ctx._engine_tmpl(self.engine, messages, usage_out=template_usage)
        finally:
            gateway_phases.append({
                "name": "prompt_template", "owner": "clozn_gateway",
                "duration_ns": max(0, time.monotonic_ns() - template_started_ns),
                "measurement": "measured", "aggregation": "exclusive",
            })
        if mem_out is not None:
            # final_prompt = the EXACT rendered string the model saw (backlog #5); kept in lockstep with chat().
            mem_out.update(assembled_messages=list(messages), final_prompt=prompt)
        kw = {}
        body = dict(kw); body["prompt"] = prompt; body["max_tokens"] = int(max_new)
        if stop:
            body["stop"] = list(stop)
        if samp and samp.get("on"):
            body["temperature"] = float(samp["temperature"])
            body["rep_penalty"] = float(samp["repeat_penalty"])
            body["top_k"] = int(samp["top_k"])
            body["top_p"] = float(samp["top_p"])
            body["seed"] = int(samp["seed"])
        else:
            body["temperature"] = 0.0; body["rep_penalty"] = 1.0; body["seed"] = 0
        body["stream"] = True
        wreq = urllib.request.Request(self.engine.base + "/v1/completions",
                                      data=json.dumps(body).encode("utf-8"),
                                      headers={"Content-Type": "application/json"})
        frames = []
        dispatch_started_ns = time.monotonic_ns()
        try:
            resp = urllib.request.urlopen(wreq, timeout=getattr(self.engine, "timeout", 600))
        except urllib.error.HTTPError as he:
            # Surface the worker's own error text instead of a bare status code.
            try:
                detail = json.loads(he.read()).get("error") or str(he)
            except Exception:
                detail = str(he)
            raise RuntimeError(f"engine: {detail}")
        try:
            for raw in resp:
                if req.is_cancelled():          # CANCELLATION: the caller already gave up on this request
                    break                        # (client gone) -- stop pulling from the worker between frames
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                frames.append(obj)
                if req.engine_req is None:
                    # StreamEnvelope (server_shared.hpp) stamps `req` on EVERY frame of this stream, so
                    # the first one parsed already carries it -- capture once here rather than waiting
                    # for a specific frame type. This is the req_ <-> worker-req correlation
                    # request_context.py's new_request_id() describes: routes/engine.py's POST /cancel
                    # reads it off self._request to resolve a gateway id to the worker's own.
                    engine_req = obj.get("req")
                    if engine_req:
                        req.engine_req = str(engine_req)
                if req.prompt_tokens is None and obj.get("type") == "gen_started":
                    # The worker's own accounting of how many prompt tokens this generation evaluated,
                    # captured once from the first gen_started frame and never derived or guessed here.
                    prompt_tokens = obj.get("prompt_tokens")
                    if isinstance(prompt_tokens, int):
                        req.prompt_tokens = prompt_tokens
                if isinstance(obj.get("completion_tokens"), int):
                    req.completion_tokens = obj["completion_tokens"]
                if obj.get("type") == "gen_finished" and isinstance(obj.get("new_tokens"), int):
                    req.completion_tokens = obj["new_tokens"]
                if isinstance((obj.get("usage") or {}).get("completion_tokens"), int):
                    req.completion_tokens = obj["usage"]["completion_tokens"]
                if isinstance((obj.get("usage") or {}).get("prompt_tokens"), int):
                    req.prompt_tokens = obj["usage"]["prompt_tokens"]
                if isinstance(obj.get("termination"), dict):
                    req.termination = dict(obj["termination"])
                if obj.get("type") == "tokens_committed":
                    for it in obj.get("items") or []:
                        piece = it.get("piece", "")
                        if piece:
                            yield piece
        finally:
            # ALWAYS release the engine connection -- whether the stream ran to [DONE] or the consumer
            # stopped early (this `finally` also runs when the caller .close()s us mid-stream, via a
            # GeneratorExit at the `yield` above); guarded so a close() hiccup can never mask a
            # propagating GeneratorExit -- it must reach the caller, never be swallowed here. (The
            # engine-side crash-on-disconnect is a separate C++-side task; this just closes cleanly.)
            try:
                resp.close()
            except Exception:
                pass
            try:
                req.trace = runlog.accumulate_ar_events(frames)
            except Exception:
                req.trace = []
            try:
                req.finish_reason = runlog.finish_reason_from_frames(frames)
                req.finish_reason_raw = runlog.raw_finish_reason_from_frames(frames)
            except Exception:
                req.finish_reason = None
                req.finish_reason_raw = None
            try:
                req.generation_timing = runlog.generation_timing_from_frames(frames)
            except Exception:
                req.generation_timing = {}
            gateway_phases.append({
                "name": "worker_dispatch", "owner": "clozn_gateway",
                "duration_ns": max(0, time.monotonic_ns() - dispatch_started_ns),
                "measurement": "measured", "aggregation": "overlapping",
            })
            from clozn.runs.perf_spans import timing_document
            req.gateway_timing = timing_document(gateway_phases)
            # (A streaming-twin loop guard lived here. It only ran when anchored memory had actually
            # ridden the turn -- mem_out["anchored"] -- which no longer happens, so it was dead code
            # after the anchored removal. Generic repetition detection survives in
            # clozn/runs/degeneracy.py and still reaches run records through runs/signals.py.)
            if mem_out is not None:
                req.prompt_manifest = dict(mem_out)


def _quant_from_name(name):
    """Pull the GGUF quant tag (Q4_K_M, Q8_0, IQ4_XS, F16, ...) out of a model filename, or None. GGUF
    files name their quantization in the basename, so this is the one bit of repro metadata we can read
    for free (no engine change) off /health's model path."""
    import re
    m = re.search(r"(IQ\d+[A-Z0-9_]*|Q\d+(?:_[A-Z0-9]+)+|Q\d+|BF16|F16|F32)", str(name), re.IGNORECASE)
    return m.group(1).upper() if m else None


# --- engine model registry (T0.2) ---------------------------------------------------------------------
# The engine substrate reflects the ACTUALLY-LOADED GGUF, not a hardcoded "Qwen2.5-7B" id/assumption.
# This tiny registry keys a friendly model_id for run_meta off the loaded model's family (derived from
# its /health filename), with a sensible default (None) for any unrecognized GGUF. This is NOT a
# framework -- it is the minimal table that avoids a hardcoded-Qwen coupling in the engine path.
_ENGINE_MODELS = {
    "qwen2.5-7b":   {"model_id": "Qwen/Qwen2.5-7B-Instruct"},
    "qwen2.5-0.5b": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct"},
    "qwen3.5-9b":   {"model_id": "Qwen/Qwen3.5-9B"},
    "llama-3.1-8b": {"model_id": "meta-llama/Llama-3.1-8B-Instruct"},
    "llama-3.2-1b": {"model_id": "meta-llama/Llama-3.2-1B-Instruct"},
    "llama-3.2-3b": {"model_id": "meta-llama/Llama-3.2-3B-Instruct"},
    "gemma4-e4b":   {"model_id": "google/gemma-4-E4B-it"},
    "ministral3-3b": {"model_id": "mistralai/Ministral-3-3B-Instruct-2512"},
}
_ENGINE_MODEL_DEFAULT = {"model_id": None}  # unknown GGUF: nothing pinned


def _model_family_from_name(name):
    """Coarse model family key ('qwen2.5-7b', 'llama-3.2-1b', ...) from a GGUF filename, or None -- the
    engine substrate looks up per-model assumptions in _ENGINE_MODELS by this key instead of hardcoding
    Qwen's. Same free derive-off-/health-filename trick as _quant_from_name (no engine change needed)."""
    import re
    s = str(name or "").lower()
    m = re.search(r"qwen[._]?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)b", s)
    if m:
        return f"qwen{m.group(1)}-{m.group(2)}b"
    m = re.search(r"llama[._-]?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)b", s)
    if m:
        return f"llama-{m.group(1)}-{m.group(2)}b"
    if re.search(r"gemma[._-]?4[._-]?e4b", s):
        return "gemma4-e4b"
    m = re.search(r"ministral[._-]?3[._-]?(\d+(?:\.\d+)?)b", s)
    if m:
        return f"ministral3-{m.group(1)}b"
    return None


def _engine_model_info(name):
    """(family, {model_id}) for the loaded GGUF -- the engine substrate's per-model assumptions -- or
    (None, the default with nothing pinned) for an unrecognized model."""
    fam = _model_family_from_name(name)
    return fam, dict(_ENGINE_MODELS.get(fam, _ENGINE_MODEL_DEFAULT))


def _engine_complete_traced(engine, prompt, max_tokens, kw, sample=None, usage_out=None, stop=None):
    """Generate on the engine and ALSO capture a per-token trace (issue B3), returning
    (reply, steps, finish, divinfo).

    The engine's non-streaming /v1/completions carries only the final text -- no per-token confidence. To
    populate the Run Inspector timeline we ask the SAME request with stream:True and fold its per-token
    `tokens_committed`/`step_lens` frames into steps via runlog.accumulate_ar_events. Generation defaults to
    greedy (temperature 0) so the reassembled text is identical to the blocking call; passing a
    ctx._resolve_sampling() dict as `sample` (S5) switches BOTH the streaming attempt and the fallback below to
    the same temperature/rep_penalty/seed, so the two paths always agree. Either way we only capture
    ALONGSIDE; the client still receives the same single JSON reply (this streams engine<->server, never to
    the client). Any streaming hiccup falls back to the plain complete() so a run is never lost -- just
    without a trace. (AR GGUFs only; a diffusion engine commits out of reading order and emits no such
    per-token stream.)

    `divinfo` is (diverged, diverged_at) when the request carried `reference_tokens` (prove-all early-stop):
    diverged True/False is the engine's verdict, else (None, None). The reply on a diverged run is a
    BIT-EXACT PREFIX of the full generation (the engine only adds a stop check -- decode is unchanged).

    `sample`: None (or a falsy dict) -- greedy, temperature=0/rep_penalty=1/seed=0/top_k=0/top_p=1,
    byte-identical to pre-S5 behavior. A ctx._resolve_sampling() dict -- temperature/repeat_penalty/seed
    plus top_k/top_p ride the request as the engine's own SampleConfig keys;
    the engine's sampler (engine/core/src/sample.cpp) truncates to top-k then the top-p nucleus before
    the draw. Greedy
    (temperature 0) ignores all of them -- the argmax path is untouched, receipts stay exact."""
    on = bool(sample and sample.get("on"))
    temperature = float(sample["temperature"]) if on else 0.0
    rep_penalty = float(sample["repeat_penalty"]) if on else 1.0
    top_k = int(sample["top_k"]) if on else 0
    top_p = float(sample["top_p"]) if on else 1.0
    seed = int(sample["seed"]) if on else 0
    import urllib.request
    body = dict(kw); body["prompt"] = prompt; body["max_tokens"] = int(max_tokens)
    if stop:
        body["stop"] = list(stop)
    body["temperature"] = temperature; body["rep_penalty"] = rep_penalty; body["seed"] = seed
    body["top_k"] = top_k; body["top_p"] = top_p
    body["stream"] = True
    try:
        req = urllib.request.Request(engine.base + "/v1/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        frames, text = [], ""
        diverged, diverged_at = None, None
        with urllib.request.urlopen(req, timeout=getattr(engine, "timeout", 600)) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                frames.append(obj)
                if (usage_out is not None and obj.get("type") == "gen_started"
                        and isinstance(obj.get("prompt_tokens"), int)):
                    usage_out["prompt_tokens"] = obj["prompt_tokens"]
                if usage_out is not None and isinstance(obj.get("completion_tokens"), int):
                    usage_out["completion_tokens"] = obj["completion_tokens"]
                if usage_out is not None and isinstance((obj.get("usage") or {}).get("completion_tokens"), int):
                    usage_out["completion_tokens"] = obj["usage"]["completion_tokens"]
                if usage_out is not None and isinstance((obj.get("usage") or {}).get("prompt_tokens"), int):
                    usage_out["prompt_tokens"] = obj["usage"]["prompt_tokens"]
                if usage_out is not None and isinstance(obj.get("termination"), dict):
                    usage_out["termination"] = dict(obj["termination"])
                if usage_out is not None and obj.get("type") == "gen_finished" and isinstance(obj.get("new_tokens"), int):
                    usage_out["completion_tokens"] = obj["new_tokens"]
                ch = obj.get("choices")                     # the final OpenAI-style frame carries the full text
                if ch and isinstance(ch, list) and ch[0].get("text"):
                    text = ch[0]["text"]
                if "diverged" in obj:                       # early-stop verdict rides the final frame
                    diverged = obj.get("diverged")
                    diverged_at = obj.get("diverged_at")
        import clozn.runs.store as runlog
        steps = runlog.accumulate_ar_events(frames)
        finish = runlog.finish_reason_from_frames(frames)   # the engine's real stop cause (else None)
        if not text:                                        # no final frame text -> reassemble from the pieces
            text = "".join(s.get("piece", "") for s in steps)
        if steps or text:
            if usage_out is not None:
                timing = runlog.generation_timing_from_frames(frames)
                if timing:
                    usage_out["generation_timing"] = timing
                raw_reason = runlog.raw_finish_reason_from_frames(frames)
                if raw_reason:
                    usage_out["raw_finish_reason"] = raw_reason
            return text, steps, finish, (diverged, diverged_at)
    except Exception:
        pass
    # Fallback: the original blocking path, reply preserved, trace simply empty. The non-streaming
    # /v1/completions carries the same `diverged`/`diverged_at` when a reference was sent. Same
    # temperature/rep_penalty/seed as the streaming attempt above -- the fallback must never silently
    # decode under a DIFFERENT regime than the one recorded in the run's meta.
    r = engine.complete(prompt, max_tokens=max_tokens, temperature=temperature, rep_penalty=rep_penalty,
                        top_k=top_k, top_p=top_p, seed=seed, stop=stop, **kw)
    prompt_tokens = (r.get("usage") or {}).get("prompt_tokens") if isinstance(r, dict) else None
    if usage_out is not None and isinstance(prompt_tokens, int):
        usage_out["prompt_tokens"] = prompt_tokens
    completion_tokens = (r.get("usage") or {}).get("completion_tokens") if isinstance(r, dict) else None
    if usage_out is not None and isinstance(completion_tokens, int):
        usage_out["completion_tokens"] = completion_tokens
    if usage_out is not None and isinstance(r, dict) and isinstance(r.get("termination"), dict):
        usage_out["termination"] = dict(r["termination"])
    ch = r.get("choices") if isinstance(r, dict) else None
    finish = ch[0].get("finish_reason") if (ch and isinstance(ch[0], dict)) else None
    divinfo = (r.get("diverged"), r.get("diverged_at")) if isinstance(r, dict) else (None, None)
    if not (isinstance(ch, list) and ch and isinstance(ch[0], dict)):
        # A reply with no usable `choices` is a worker protocol violation, not a generation. Never turn
        # a malformed response into plausible-looking model text.
        raise ValueError(
            "engine reply has no usable 'choices' -- refusing to synthesize text from "
            f"{type(r).__name__}"
        )
    return ch[0].get("text", ""), [], finish, divinfo
