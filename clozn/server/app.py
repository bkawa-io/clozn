"""The single public Clozn product gateway.

``clozn serve`` hosts this Torch-free server in a background thread after its private C++ model worker is
healthy. All product HTTP surfaces—OpenAI compatibility, Studio, runs, receipts,
experiments, and readouts—live here. Training and calibration are offline lab
jobs and cannot be selected as alternate serving substrates.
"""
import json
import logging
import os
import secrets
import subprocess
import time

os.environ.setdefault("CLOZN_RUNTIME_KIND", "product")
RUNTIME_KIND = os.environ["CLOZN_RUNTIME_KIND"]

from clozn.server.config import HERE, REPO_ROOT, DEMO, CLOZN_DIR   # noqa: E402 (side effects: sys.path/env/stdout)
from clozn.server.http_policy import (                            # noqa: E402
    DEFAULT_ALLOW_HEADERS, client_gone, max_request_bytes, origin_allowed, request_origin, send_cors_headers,
)
from clozn.server.request_gate import RequestGate, rejection_response   # noqa: E402

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: E402

_log = logging.getLogger(__name__)

POST_GATE = RequestGate.from_env()

# POST_GATE protects worker-shaped state outside the request-local generation paths. Checkpoint,
# replay, and journal operations are serialized unless a route has been explicitly audited as
# worker-independent. Everything outside _GENERATION_POST_PATHS below reads or writes state this
# process cannot prove is safe to run concurrently with generation (see WorkerGateRegistry.acquire_all's
# docstring), so this remains a DELIBERATELY SMALL allowlist of POST paths audited to touch NEITHER the
# substrate's generation state NOR anything else worker-scoped:
#   /cancel       -- cancellation control does not select a worker or start generation.
# Every other POST -- fork/checkpoint/replay/journal and everything else NOT in _GENERATION_POST_PATHS
# below -- still reads or writes worker-shaped state and stays fully serialized through POST_GATE. The
# risk of a wrong "this is safe" call here is a corrupted run record, so the bar is "audited to touch
# nothing substrate-shaped", not a coarse heuristic like "looks read-only".
_GATE_EXEMPT_POSTS = frozenset({"/cancel"})

# RT-05: the closed set of routes that resolve a model through
# clozn.server.model_routing.select_for_handler. do_POST does NOT acquire POST_GATE for
# these; select_for_handler acquires a PER-WORKER generation gate itself,
# AFTER selection (and any RT-04 cold-load coalescing) succeeds -- see that
# function's docstring for why gating must happen there, not here. This lets
# different workers generate concurrently without contending for one global
# turn. Every other POST path -- including every route this process
# does not itself understand the worker-scope of (fork, checkpoint, replay,
# journal, ...) -- still goes through POST_GATE below, and in managed mode
# ALSO drains every configured worker's generation gate first (see
# WorkerGateRegistry.acquire_all's docstring): a POST this process cannot
# prove is worker-agnostic must never run concurrently with generation on
# a worker it might touch.
_GENERATION_POST_PATHS = frozenset({
    "/v1/chat/completions",
})

try:
    from clozn_engine import EngineClient, EngineError
    ENGINE = None
except Exception:
    ENGINE = None
    class EngineError(RuntimeError):   # keeps imports usable when the worker SDK is unavailable
        pass

# Product startup always installs the router. ``None`` keeps route modules importable for isolated tests.
MODEL_ROUTER = None
PUBLIC_MODEL_ID = None


class EngineUnavailable(RuntimeError):
    """Raised when the engine is completely UNREACHABLE (connection refused / host down -- a raw
    urllib.error.URLError, never even got an HTTP response), as opposed to EngineError (the engine
    responded, but with a non-2xx JSON error). Routes translate this to a consistent 502 response."""


def _git_commit():
    """Best-effort build/repro id. Returns None when this checkout is not a git repo or git is unavailable."""
    if getattr(_git_commit, "_read", False):
        return getattr(_git_commit, "_value", None)
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL, timeout=2)
        val = out.decode("utf-8", "replace").strip() or None
    except Exception:
        val = None
    _git_commit._value = val
    _git_commit._read = True
    return val


def _without_unknowns(d):
    """Drop only unknown values; keep honest falsy repro values like 0 temperature or seed."""
    return {k: v for k, v in (d or {}).items() if v is not None}


def _openai_finish_reason(fr):
    """OpenAI-compatible responses need a concrete finish_reason string even if a substrate cannot provide one."""
    return fr if isinstance(fr, str) and fr else "stop"


def _qwen_generation_meta(max_new=None, sample=True, stream=None):
    return _without_unknowns({
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "sampler_mode": "sample" if sample else "greedy",
        "sampling": "sample" if sample else "greedy",
        "temperature": 0.7 if sample else 0.0,
        "top_p": 0.9 if sample else None,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 3,
        # Self-describing decode block (REPRODUCE_AND_PROVE_PLAN S2). seed is honestly null: HF
        # generate() sets no fixed seed on this path, so a sampled Qwen run is NOT exactly
        # reproducible -- the deliberate contrast with the engine's reproducible {"seed": 0}.
        "decode": {"mode": "sample" if sample else "greedy",
                   "temperature": 0.7 if sample else 0.0,
                   "top_p": 0.9 if sample else None,
                   "seed": None},
        "max_tokens": int(max_new) if max_new is not None else None,
        "stream": bool(stream) if stream is not None else None,
    })


# REPRODUCE_AND_PROVE_PLAN S5: settings-exposed interactive-chat sampling (llama.cpp's canonical
# defaults -- model-agnostic, since clozn serves any GGUF). "sampling" is the master on/off; OFF (or a
# caller that explicitly asked for greedy, e.g. every receipt/replay/forced-scoring call) always decodes
# temperature 0, byte-identical to pre-S5 behavior. The FULL regime (top_k/top_p included) is enforced by
# the engine's sampler -- see _resolve_sampling's HONESTY paragraph.
_SAMPLING_DEFAULTS = {
    "sampling": True,
    "sample_temperature": 0.8,
    "sample_top_p": 0.9,
    "sample_top_k": 40,
    "sample_repeat_penalty": 1.1,
}


def _resolve_sampling(want_sample):
    """Whether THIS engine generation should sample, and under what params -- S5. `want_sample` is either
    a bool or a per-request override dict (used by the OpenAI compatibility route for temperature/top_p/
    top_k/repeat_penalty/seed). Returns None (greedy, temperature 0 -- exactly pre-S5 behavior) when the
    request is falsy, its temperature is zero, OR a default boolean request is disabled by the persisted
    "sampling" master switch. An explicit override dict is per-request and wins over that switch. Otherwise
    this returns {"on": True, "temperature", "top_p", "top_k", "repeat_penalty", "seed"}. Missing fields
    use the persisted defaults; a missing seed gets a fresh per-turn value.

    Critical for the receipt/replay/rederive stack: `want_sample=False` (replay.py's
    `sampled = not changes.get("greedy")`, always False for receipts) short-circuits to None BEFORE the
    setting is even read -- so turning interactive sampling on/off can never make a receipt's forced-greedy
    regen non-deterministic. Only the caller's own request gates that.

    HONESTY: the full regime -- temperature/repeat_penalty/top_k/top_p/seed -- reaches the engine's
    sampler (engine/core/serve/server_shared.hpp's sample_from -> engine/core/src/sample.cpp: a
    temperature-scaled softmax, repetition penalty, then top-k + top-p nucleus truncation before an
    inverse-CDF draw). So a sampled reply follows the configured llama.cpp distribution. The
    greedy path (temperature 0 / this returning None) is the argmax and ignores every one of them --
    top_k/top_p included -- so receipts and forced-greedy replay stay bit-exact."""
    if not want_sample:
        return None
    overrides = want_sample if isinstance(want_sample, dict) else {}
    import clozn.settings as settings
    # The persisted switch controls Clozn's own default behavior. An OpenAI request that explicitly names
    # sampling fields is a per-call contract and must win; otherwise we would accept temperature/top_p and
    # silently run greedy because somebody toggled Studio's global setting earlier.
    if not overrides and not bool(settings.get_setting("sampling", _SAMPLING_DEFAULTS["sampling"])):
        return None
    resolved = {
        "on": True,
        "temperature": float(overrides.get("temperature", settings.get_setting(
            "sample_temperature", _SAMPLING_DEFAULTS["sample_temperature"]))),
        "top_p": float(overrides.get("top_p", settings.get_setting(
            "sample_top_p", _SAMPLING_DEFAULTS["sample_top_p"]))),
        "top_k": int(overrides.get("top_k", settings.get_setting(
            "sample_top_k", _SAMPLING_DEFAULTS["sample_top_k"]))),
        "repeat_penalty": float(overrides.get("repeat_penalty", settings.get_setting(
            "sample_repeat_penalty", _SAMPLING_DEFAULTS["sample_repeat_penalty"]))),
        # A FRESH seed every turn (not a fixed one) -- what makes a sampled reply vary turn to turn while
        # still being independently reproducible: reissuing the same generation request through the
        # configured gateway/private-worker path with this seed+params reproduces the text (mode.decode
        # records it on the run).
        "seed": int(overrides.get("seed", secrets.randbits(63))),
    }
    return resolved if resolved["temperature"] > 0 else None


def _sampling_settings():
    """The persisted S5 sampling settings (on/off + the four params) for the GET/POST /sampling/mode
    route -- what's actually configured, not what one specific turn resolved to. Unlike _resolve_sampling,
    this never generates a seed (a GET must never mutate anything)."""
    import clozn.settings as settings
    return {k: settings.get_setting(k, v) for k, v in _SAMPLING_DEFAULTS.items()}


def _engine_generation_meta(max_new=None, stream=None, sample=None, stop=None):
    """Reproducibility metadata for one engine generation (REPRODUCE_AND_PROVE_PLAN S2/S5).

    `sample`: None (the default) always yields the honest greedy regime -- byte-identical to every
    pre-S5 caller (run_meta()'s static baseline, and any chat()/chat_stream() call that resolved to
    greedy). Pass a _resolve_sampling() dict to flip the decode block to the real sampled regime this
    generation actually used."""
    on = bool(sample and sample.get("on"))
    if on:
        top = {"sampler_mode": "sample", "sampling": "sample", "temperature": sample["temperature"],
               "repetition_penalty": sample["repeat_penalty"], "seed": sample["seed"]}
        decode = {"mode": "sample", "temperature": sample["temperature"], "top_p": sample["top_p"],
                  "top_k": sample["top_k"], "repeat_penalty": sample["repeat_penalty"],
                  "seed": sample["seed"]}
    else:
        top = {"sampler_mode": "greedy", "sampling": "greedy", "temperature": 0.0,
               "repetition_penalty": 1.0, "seed": 0}
        # Self-describing decode block (REPRODUCE_AND_PROVE_PLAN S2): the honest regime this run was
        # produced under, so re-derivation/forced-scoring is exact-by-construction. Engine chat is greedy
        # (temperature 0), seed 0 -- the actual values passed, not a guess.
        decode = {"mode": "greedy", "temperature": 0.0, "seed": 0}
    return _without_unknowns({
        **top,
        "decode": decode,
        "max_tokens": int(max_new) if max_new is not None else None,
        "stream": bool(stream) if stream is not None else None,
        # An empty stop list is a resolved no-stop contract, not an unknown
        # value.  Persisting it lets exact recorded-answer studies distinguish
        # an intentional no-stop decode from a missing generation contract.
        "stop": list(stop) if stop is not None else ([] if max_new is not None else None),
    })


def _engine_tmpl(engine, messages, usage_out=None):
    """Render chat `messages` to a prompt string using the ENGINE-LOADED MODEL'S OWN chat template
    (POST /apply_template -> llama_chat_apply_template over the GGUF's tokenizer.chat_template). THIS is
    what makes the engine paths model-agnostic: whatever GGUF the engine has loaded, its messages are
    formatted in that model's native template (Qwen ChatML, Llama-3 headers, Gemma turns, ...), instead
    of a hardcoded model-specific string.

    Errors propagate deliberately (no silent Qwen fallback): a model with no embedded template, or an
    worker without /apply_template must surface -- silently mis-formatting the prompt is the
    exact bug this removes. Callers that need a soft degrade catch EngineError themselves."""
    if usage_out is not None and hasattr(engine, "apply_template_info"):
        info = engine.apply_template_info(messages)
        prompt_tokens = info.get("prompt_tokens") if isinstance(info, dict) else None
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            usage_out["prompt_tokens"] = prompt_tokens
        return info["prompt"]
    return engine.apply_template(messages)


SUB = None         # the active substrate object
SUBNAME = "engine"  # product server has one substrate; PyTorch model work lives in the lab

# ------- substrates: extracted to clozn/server/substrates.py; re-exported here (the seam) -----------
# app.py remains the canonical module: routes read ctx.<name> and tests patch cs.<name> on THIS module.
from clozn.server.substrates import (                                                    # noqa: E402
    EngineSubstrate,      # Qwen/Dream lab substrates now live in clozn/lab/substrates.py
    _quant_from_name, _model_family_from_name,
    _engine_model_info, _engine_complete_traced, _ENGINE_MODELS, _ENGINE_MODEL_DEFAULT,
)

# ------- route families: registered in dispatch order; each exposes try_get/try_post(handler, path, ...) --
# returning True once it has written a response, so do_GET/do_POST can stop at the first family that
# claims a path. Unknown POST routes return a normal 404 after every family declines them.
#
# runs.try_get_fallback (the generic GET /runs/<id>) is registered LAST in _GET_ROUTES, deliberately --
# every more-specific /runs/<id>/<suffix> family must get first refusal, since they all share the
# "/runs/" prefix the fallback also matches.
import types as _types                                                # noqa: E402

from clozn.server import static as _static_routes                     # noqa: E402
from clozn.server.routes import health as _health_routes              # noqa: E402
from clozn.server.routes import runs as _runs_routes                  # noqa: E402
from clozn.server.routes import openai as _openai_routes              # noqa: E402

from clozn.server.routes import _autoload as _route_autoload          # noqa: E402

_runs_fallback_routes = _types.SimpleNamespace(try_get=_runs_routes.try_get_fallback)

# Route families in the checked-in allowlist. A source file or marker alone cannot expose an
# endpoint. The manifest is loaded once at import, not per request.
_autoloaded_routes = _route_autoload.discover()

# Manifested GET families are spliced in BEFORE _runs_fallback_routes, never appended after it: the
# fallback matches the whole "/runs/" prefix, so anything landing behind it would be shadowed -- and
# shadowed as a wrong-shaped 200 rather than a 404, which is far harder to spot. See _autoload.py.
_GET_ROUTES = [
    _static_routes,
    _health_routes,
    _openai_routes,
    _runs_routes,
    *_route_autoload.with_try_get(_autoloaded_routes),
    _runs_fallback_routes,
]
_POST_ROUTES = [
    _openai_routes,
    *_route_autoload.with_try_post(_autoloaded_routes),
]
_DELETE_ROUTES = [_runs_routes]


def active_sub(h):
    """Return the substrate explicitly selected for this request, if any."""
    return getattr(h, "_route_sub", None)


def active_subname(h):
    routed = getattr(h, "_route_subname", None)
    return routed if routed is not None else "engine"


def active_engine(h):
    routed = getattr(h, "_route_engine", None)
    if routed is not None:
        return routed
    sub = active_sub(h)
    return getattr(sub, "engine", None) if sub is not None else None


def active_kind(h):
    return RUNTIME_KIND


def make_handler():
    """Build the public gateway handler. Model workers are attached per request by the router."""
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype, extra_headers=None):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            send_cors_headers(self)
            if extra_headers:                            # additive + optional: no caller passed this before,
                for k, v in extra_headers.items():        # so today's callers get byte-identical output
                    self.send_header(str(k), str(v))
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _json(self, code, o, extra_headers=None):
            serialize_started_ns = time.monotonic_ns()
            payload = json.dumps(o)
            serialize_duration_ns = max(0, time.monotonic_ns() - serialize_started_ns)
            rid = getattr(self, "_last_logged_run_id", None)
            if rid:
                try:
                    from clozn.runs.attachments import attach_performance_phase
                    attach_performance_phase(rid, {
                        "name": "gateway_serialize", "owner": "clozn_gateway",
                        "duration_ns": serialize_duration_ns,
                        "measurement": "measured",
                        # The run record's wall timer stops when record() is called, immediately before
                        # serialization. Keep this measured phase visible, but do not pretend it is
                        # inside that earlier wall interval when computing unaccounted time.
                        "aggregation": "context_only",
                    })
                except Exception:
                    pass
            self._send(code, payload, "application/json", extra_headers=extra_headers)

        def _record_gateway_phase(self, name, duration_ns, *, owner="clozn_gateway",
                                  aggregation="exclusive"):
            """Accumulate a measured gateway phase for the run this request will eventually journal."""
            try:
                duration_ns = max(0, int(duration_ns))
                phases = getattr(self, "_gateway_timing_phases", None)
                if not isinstance(phases, list):
                    phases = []
                    self._gateway_timing_phases = phases
                for phase in phases:
                    if (
                        phase.get("name") == name and phase.get("owner") == owner
                        and phase.get("aggregation") == aggregation
                    ):
                        phase["duration_ns"] += duration_ns
                        return
                phases.append({
                    "name": str(name), "owner": str(owner), "duration_ns": duration_ns,
                    "measurement": "measured", "aggregation": aggregation,
                })
            except Exception:
                pass

        def _reject_untrusted_origin(self):
            origin = request_origin(self)
            if origin and not origin_allowed(origin):
                self._json(403, {"error": "browser origin is not allowed"})
                return True
            return False

        def do_OPTIONS(self):
            """CORS preflight without opening localhost to arbitrary web origins."""
            origin = request_origin(self)
            if origin and not origin_allowed(origin):
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(204)
            send_cors_headers(self)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            requested = self.headers.get("Access-Control-Request-Headers")
            self.send_header("Access-Control-Allow-Headers", requested or DEFAULT_ALLOW_HEADERS)
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _client(self, ua):
            ua = (ua or "").lower()
            for k, v in (("cursor", "Cursor"), ("vscode", "VS Code"),
                         ("python-requests", "script"), ("httpx", "script"),
                         ("openai-python", "script"), ("curl", "curl"), ("mozilla", "browser")):
                if k in ua:
                    return v
            return ua[:24] or "unknown"

        def _log_run(self, source, messages, response, model, started, error=None, trace=None,
                     mem_out=None, finish_reason=None, finish_reason_fallback=None, extra_meta=None,
                     output_contract=None):
            """Persist this interaction as an inspectable run (never let logging break the request).
            mem_out: the {final_prompt, assembled_messages, actual_prompt_tokens?} record the generation
            path filled in -- what the substrate actually did with this turn.
            extra_meta (optional): additional {key: value} pairs folded into the run's `meta` block on top
            of the substrate/build-commit fields this method already assembles -- e.g. sse.py
            tags a failed stream with {"stream_failure": "client_disconnected" | "worker_disconnected"} so
            the Runs page (and anyone reading the record later) can tell those two failure modes apart
            instead of both just showing an opaque `error` string. None-valued entries are dropped (same
            "don't record a guess" rule _without_unknowns applies elsewhere).
            Returns the new run's id (str) on success, else None -- any failure along the way is swallowed,
            never raised. The M5 any-client bridge surfaces this id to the caller; None means "nothing to
            surface", not an error the request should see."""
            try:
                import clozn.runs.store as runlog
                extra_meta = dict(extra_meta or {})
                routing_artifact = getattr(self, "_model_routing_artifact", None)
                if isinstance(routing_artifact, dict):
                    # One request-scoped immutable routing decision, recorded on
                    # every success/failure journal path below.  Routes cannot
                    # accidentally reuse another model's evidence because the
                    # same handler override also supplies request_sub.
                    extra_meta["model_routing"] = json.loads(
                        json.dumps(routing_artifact)
                    )
                    resolved = (
                        routing_artifact.get("result", {})
                        .get("receipt", {})
                        .get("resolved_model_id")
                    )
                    if isinstance(resolved, str) and resolved:
                        model = resolved
                request_sub = active_sub(self)
                request_subname = active_subname(self)
                from clozn.runs.association import request_client, request_project, request_session
                session_key = request_session(self.headers)
                client_key, client_key_source = request_client(self.headers)
                project_key = request_project(self.headers)
                mo = mem_out or {}
                from clozn.runs.think_tags import prompt_opens_think, sanitize_messages, sanitize_reply
                messages = sanitize_messages(messages)
                think = sanitize_reply(response, implicit_open=prompt_opens_think(mo.get("final_prompt")))
                response = think.public_text
                reasoning = think.journal()
                meta = None
                try:                                          # engine: {model_file, quant, mode, sampling}
                    if request_sub is not None and hasattr(request_sub, "run_meta"):
                        meta = request_sub.run_meta() or None
                except Exception:
                    meta = None
                meta = dict(meta or {})
                try:
                    prompt_tokens = (
                        request_sub.last_prompt_tokens()
                        if request_sub is not None
                        and hasattr(request_sub, "last_prompt_tokens") else None
                    )
                    if isinstance(prompt_tokens, int):
                        meta.setdefault("prompt_tokens", prompt_tokens)
                except Exception:
                    pass
                actual_prompt_tokens = mo.get("actual_prompt_tokens")
                if isinstance(actual_prompt_tokens, int) and not isinstance(actual_prompt_tokens, bool):
                    meta["prompt_tokens"] = actual_prompt_tokens
                if extra_meta:
                    meta.update({k: v for k, v in extra_meta.items() if v is not None})
                gateway_phases = list(getattr(self, "_gateway_timing_phases", None) or [])
                substrate_gateway = meta.get("gateway_timing")
                if gateway_phases or (
                    isinstance(substrate_gateway, dict) and substrate_gateway.get("phases")
                ):
                    from clozn.runs.perf_spans import merge_timing_documents, timing_document
                    handler_timing = timing_document(gateway_phases)
                    meta["gateway_timing"] = merge_timing_documents(
                        substrate_gateway, handler_timing
                    )
                # (memory prompt_token_cost -- the with-vs-without-block token delta -- was computed
                # here; there is no block to charge for since the 2026-07-27 cards cut.)
                git = _git_commit()
                if git:
                    meta.setdefault("build_git_commit", git)
                if finish_reason:
                    meta.setdefault("finish_reason_source", "substrate")
                elif finish_reason_fallback:
                    meta.setdefault("finish_reason_source", "fallback")
                    meta.setdefault("finish_reason_fallback", finish_reason_fallback)
                # The pre-template message list the substrate actually handed to the renderer. It was
                # gated on prompt-memory mode; with memory gone the substrate records it unconditionally.
                assembled_messages = mo.get("assembled_messages")
                # backlog #5: the EXACT rendered chat-template string the engine produced (mem_out fills it
                # on the engine chat paths). None -> consumers fall back to assembled_messages.
                final_prompt = mo.get("final_prompt")
                # roadmap S4.3: immutable reproduction identity (model_sha256, template_fingerprint,
                # engine_build, clozn_version) -- see clozn.runs.identity's module docstring for why this
                # never adds a file hash or a network round trip to THIS request (EngineSubstrate.
                # identity_meta() piggybacks run_meta()'s single cached /health fetch). Substrates that
                # don't implement identity_meta() (the Torch lab adapters) simply log no identity block.
                identity = None
                try:
                    if request_sub is not None and hasattr(request_sub, "identity_meta"):
                        identity = request_sub.identity_meta() or None
                except Exception:
                    identity = None
                request_started = getattr(self, "_request_wall_started", None)
                if isinstance(request_started, (int, float)) and request_started <= time.time():
                    started = min(started, request_started)
                record_kwargs = dict(
                    source=source, client=self._client(self.headers.get("User-Agent", "")),
                    model=str(model), substrate=request_subname, messages=messages, response=response,
                    started=started, error=error,
                    trace=trace, finish_reason=finish_reason, meta=meta,
                    assembled_messages=assembled_messages, final_prompt=final_prompt,
                    identity=identity,
                    reasoning=reasoning, session_key=session_key,
                    client_key=client_key, client_key_source=client_key_source,
                    project_key=project_key,
                    output_contract=output_contract,
                )
                rid = runlog.record(**record_kwargs)
                self._last_logged_run_id = rid
                return rid                        # M5 bridge: the run id, for callers that want to surface it
            except Exception as exc:
                # Keep compatibility callers alive, but leave an actionable
                # request-scoped reason for the transport layer. A generated
                # reply must not be mistaken for a durable run merely because
                # this best-effort seam swallowed the failure.
                self._last_recording_error = f"{type(exc).__name__}: {exc}"
                _log.warning("run recording failed: %s", self._last_recording_error)
                return None

        def do_GET(self):
            self._last_logged_run_id = None
            if self._reject_untrusted_origin():
                return
            p = self.path.split("?")[0]
            for mod in _GET_ROUTES:
                try:
                    if mod.try_get(self, p):
                        return
                except EngineUnavailable as e:
                    self._json(502, {"error": str(e)})
                    return
                except Exception as e:
                    # Same guard as _dispatch_post: a route exception must become an HTTP error,
                    # never a dropped TCP connection. Traceback to the log, real error to the client.
                    import traceback
                    traceback.print_exc()
                    self._json(500, {"error": f"{type(e).__name__}: {e}", "route": p})
                    return
            self._json(404, {"error": "GET " + p})

        def do_DELETE(self):
            if self._reject_untrusted_origin():
                return
            p = self.path.split("?", 1)[0].rstrip("/") or "/"
            for mod in _DELETE_ROUTES:
                try:
                    if mod.try_delete(self, p):
                        return
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    self._json(500, {
                        "error": f"{type(exc).__name__}: {exc}",
                        "route": p,
                    })
                    return
            self._json(404, {"error": "DELETE " + p})

        def do_POST(self):
            from clozn.server.model_routing import clear_handler_selection
            clear_handler_selection(self)
            self._last_logged_run_id = None
            self._last_recording_error = None
            self._request_wall_started = time.time()
            self._gateway_timing_phases = []
            if self._reject_untrusted_origin():
                return
            try:
                from clozn.runs.association import validate_request_headers
                validate_request_headers(self.headers)
            except Exception as exc:
                field = getattr(exc, "field", "association")
                self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error",
                                            "param": field, "code": "invalid_association_id"}})
                return
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                self._json(501, {"error": "chunked request bodies are not supported"})
                return
            raw_length = self.headers.get("Content-Length")
            try:
                n = int(raw_length or 0)
            except (TypeError, ValueError):
                self.close_connection = True
                self._json(400, {"error": "invalid Content-Length"})
                return
            if n < 0:
                self.close_connection = True
                self._json(400, {"error": "invalid Content-Length"})
                return
            limit = max_request_bytes()
            if n > limit:
                self.close_connection = True
                self._json(413, {"error": f"request body exceeds the {limit}-byte limit"})
                return
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                # malformed JSON (bad syntax, wrong Content-Length, truncated body, ...) -- a clean 400,
                # not an uncaught JSONDecodeError. No route ever gets a chance to run against garbage.
                self._json(400, {"error": "invalid JSON body"})
                return
            if not isinstance(body, dict):
                # every route does body.get(...) unguarded -- a valid-JSON-but-non-dict body (e.g. `[1,2,3]`
                # or `"hi"`) would otherwise blow up as an AttributeError deep inside whichever route matched.
                self._json(400, {"error": "JSON body must be an object"})
                return
            p = self.path.split("?")[0].rstrip("/") or "/"
            if p in _GATE_EXEMPT_POSTS:
                # See _GATE_EXEMPT_POSTS' module-level comment: these paths are audited to touch
                # nothing the gate protects, so they run uncontended -- notably, they never sit in the
                # bounded queue behind a slow generation request either.
                try:
                    self._dispatch_post(p, body)
                finally:
                    clear_handler_selection(self)
                return
            if p in _GENERATION_POST_PATHS:
                # See _GENERATION_POST_PATHS' module-level comment: gating for these happens inside
                # select_for_handler, AFTER model selection/cold-load succeeds, keyed to the ONE worker
                # this specific request resolved to. Acquiring anything here, before that resolution,
                # would serialize the very requests RT-04's cold-load coalescing needs to see arrive
                # concurrently. clear_handler_selection releases whatever permit select_for_handler
                # acquired, exactly once, whether the route succeeded, raised, or never reached selection.
                try:
                    self._dispatch_post(p, body)
                finally:
                    clear_handler_selection(self)
                return
            queue_started_ns = time.monotonic_ns()
            rejected = POST_GATE.acquire(cancel_check=lambda: client_gone(self))
            self._record_gateway_phase(
                "gateway_queue", time.monotonic_ns() - queue_started_ns,
                aggregation="exclusive",
            )
            if rejected:
                self._reject_post_gate(rejected)
                return
            # RT-05 safety net: this POST is not one of the known generation paths, so this process does
            # not know which (if any) single worker it touches -- fork/checkpoint/replay/journal all land
            # here. In managed mode it must still never run concurrently with generation on a worker it
            # might touch (checkpoint harvest, ...) -- see
            # WorkerGateRegistry.acquire_all's docstring. Draining every configured worker's turn gives
            # this bucket the required "nothing else is running" guarantee.
            router = MODEL_ROUTER
            registry = getattr(router, "gate", None) if router is not None else None
            exclusive_release = None
            tracked_calls = []
            tracker_factory = (
                getattr(router, "worker_call_tracker", None)
                if router is not None else None
            )
            if callable(tracker_factory):
                # Unclassified mutations (checkpoint, replay, journal,
                # and similar routes) are serialized through POST_GATE, but a
                # cold generation can arrive concurrently because generation
                # routes intentionally bypass that global gate. Reserve every
                # currently-ready worker before waiting so a capacity load
                # cannot evict one while this mutation is in flight.
                for model in (router.runtime_status().get("models") or []):
                    if model.get("state") != "ready":
                        continue
                    tracker = tracker_factory(model["model_id"])
                    tracker.__enter__()
                    tracked_calls.append(tracker)
            if registry is not None:
                exclusive_started_ns = time.monotonic_ns()
                exclusive_rejected, exclusive_release = registry.acquire_all(
                    cancel_check=lambda: client_gone(self)
                )
                self._record_gateway_phase(
                    "gateway_queue", time.monotonic_ns() - exclusive_started_ns,
                    aggregation="exclusive",
                )
                if exclusive_rejected:
                    for tracker in reversed(tracked_calls):
                        tracker.__exit__(None, None, None)
                    POST_GATE.release()
                    self._reject_post_gate(exclusive_rejected)
                    return
            try:
                self._dispatch_post(p, body)
            finally:
                if exclusive_release is not None:
                    exclusive_release()
                for tracker in reversed(tracked_calls):
                    tracker.__exit__(None, None, None)
                POST_GATE.release()
                clear_handler_selection(self)

        def _reject_post_gate(self, rejected):
            # "cancelled": client_gone(self) fired while this request was queued -- the requester is
            # confirmed gone, so this response is best-effort only (mirrors "full"/"timeout": nothing
            # here relies on it actually being delivered). 499 (nginx's convention for "client closed
            # the request") -- there is no standard status for this, but it is the least-ambiguous
            # existing one for observability (access logs, metrics) even if nobody reads the body.
            status, payload, extra_headers = rejection_response(rejected)
            self._json(status, payload, extra_headers=extra_headers)

        def _dispatch_post(self, p, body):
            for mod in _POST_ROUTES:
                try:
                    if mod.try_post(self, p, body):
                        return
                except EngineUnavailable as e:
                    self._json(502, {"error": str(e)})
                    return
                except Exception as e:
                    # Without this, an exception inside a route module rode all the way up to
                    # socketserver, which closed the TCP connection with NO HTTP response at all --
                    # the client saw RemoteDisconnected, undiagnosable. (Pressure-test finding:
                    # three live repros -- a firing guard whose counter-injection lacked its layer
                    # calibration, and /runs/<id>/jlens with a non-numeric layer or topk.) The
                    # traceback still lands in the gateway log; the client now gets a real error.
                    import traceback
                    traceback.print_exc()
                    self._json(500, {"error": f"{type(e).__name__}: {e}", "route": p})
                    return
            self._json(404, {"error": f"unknown route: {p}"})

    return H


def build_server(*, host: str, port: int, engine, model_router, public_model_id=None):
    """Build the public HTTP server around the supervisor-owned model router.

    The globals are intentional: route modules resolve the active request through
    this module, while the returned server owns only socket and thread lifetime.
    """
    global SUB, SUBNAME, RUNTIME_KIND, ENGINE, MODEL_ROUTER, PUBLIC_MODEL_ID
    if engine is None:
        raise RuntimeError("the product gateway needs a private model worker")
    if model_router is None:
        raise RuntimeError("the product gateway needs a model router")
    os.environ["CLOZN_RUNTIME_KIND"] = "product"
    RUNTIME_KIND = "product"
    SUBNAME = "engine"
    ENGINE = engine
    MODEL_ROUTER = model_router
    PUBLIC_MODEL_ID = public_model_id or None
    SUB, managed_engine = model_router.control_pair()
    if managed_engine is not None:
        ENGINE = managed_engine
    # RT-05: warm the run store's SQLite schema single-threaded, before
    # ThreadingHTTPServer starts dispatching request handler threads.
    try:
        import clozn.runs.store as _runlog_warmup
        _runlog_warmup.list_runs(limit=1)
    except Exception:
        pass
    return ThreadingHTTPServer((host, int(port)), make_handler())
