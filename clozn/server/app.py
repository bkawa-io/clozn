"""The single public Clozn product gateway.

``clozn serve`` starts this Torch-free process after its private C++ model worker is
healthy.  All product HTTP surfaces—OpenAI compatibility, Studio, runs, memory,
receipts, steering, and readouts—live here.  Training and calibration are offline lab
jobs and cannot be selected as alternate serving substrates.
"""
import argparse
import json
import os
import secrets
import subprocess
import sys
import time

os.environ.setdefault("CLOZN_RUNTIME_KIND", "product")
RUNTIME_KIND = os.environ["CLOZN_RUNTIME_KIND"]

# `python -m clozn.server.app` executes this file as `__main__`, a SEPARATE module object from
# `clozn.server.app` in sys.modules terms. Every route module below does `from clozn.server import app
# as ctx` to read the shared SUB/SUBNAME/ENGINE state live -- if that import were left to run normally,
# it would import (and re-execute) THIS FILE A SECOND TIME under its real dotted name, creating a second,
# disconnected copy of all this module's state: main() would mutate the __main__ copy's SUB/SUBNAME
# while every route handler reads the OTHER copy's untouched defaults (SUB=None, SUBNAME="qwen") --
# invisible to the test suite (which only ever imports the dotted name, never runs this as __main__) but
# fatal to a live `-m` boot. Aliasing sys.modules here, before any of the routes/* imports at the bottom
# of this file run, makes `from clozn.server import app` resolve to THIS SAME object either way.
sys.modules.setdefault("clozn.server.app", sys.modules[__name__])

from clozn.server.config import HERE, REPO_ROOT, DEMO, CLOZN_DIR   # noqa: E402 (side effects: sys.path/env/stdout)
from clozn.server.http_policy import (                            # noqa: E402
    DEFAULT_ALLOW_HEADERS, client_gone, max_request_bytes, origin_allowed, request_origin, send_cors_headers,
)
from clozn.server.request_gate import RequestGate                 # noqa: E402

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: E402

POST_GATE = RequestGate.from_env()

# POST_GATE (request_gate.py) exists because EngineSubstrate keeps generation metadata, tone-dial
# strengths, and prompt-mode memory cards on ONE shared object -- concurrent POST dispatch could let two
# requests corrupt each other's evidence (see request_gate.py's module docstring). Backlog #2 ("request
# isolation + cancellation") moved the generation-metadata piece onto a per-call RequestContext (see
# clozn/server/request_context.py + EngineSubstrate.chat/chat_stream in substrates.py), which narrows --
# but does NOT eliminate -- the reason the gate exists: self.steer.strength and the memory-card store are
# STILL live, shared, mutable state that a concurrent /steer/* or /memory/* write could tear out from under
# an in-flight generation read. So the gate stays broad by default; this is a DELIBERATELY SMALL allowlist
# of the only two POST paths audited to touch NEITHER the substrate's generation state NOR steer/memory:
#   /capture/tier -- a bare settings-file flag (clozn.runs.capture_mode), unrelated to SUB entirely.
#   /substrate    -- POST always 410s (routes/health.py); it never reaches the substrate at all.
# Every other POST -- every /memory/* and /steer/* mutation, plus every generation endpoint -- still reads
# or writes that shared state and stays fully serialized. The risk of a wrong "this is safe" call here is a
# corrupted run record or a torn dial read, so the bar is "audited to touch nothing substrate-shaped", not
# a coarse heuristic like "looks read-only".
_GATE_EXEMPT_POSTS = frozenset({"/capture/tier", "/substrate", "/cancel"})

try:
    from clozn_engine import EngineClient, EngineError
    # Product startup has exactly one model worker.  Requiring the supervisor-provided port here avoids
    # silently constructing clients for two guessed ports and makes a directly-launched gateway fail
    # closed.  Offline imports/tests still work with ENGINE=None.
    _ENGINE_PORT = os.environ.get("CLOZN_ENGINE_PORT")
    ENGINE = EngineClient(port=int(_ENGINE_PORT)) if _ENGINE_PORT else None
except Exception:
    ENGINE = None
    class EngineError(RuntimeError):   # fallback so `except EngineError` resolves even if the SDK import failed
        pass


class EngineUnavailable(RuntimeError):
    """Raised when the engine is completely UNREACHABLE (connection refused / host down -- a raw
    urllib.error.URLError, never even got an HTTP response), as opposed to EngineError (the engine
    responded, but with a non-2xx JSON error). Routes that dispatch through Substrate.handle() (the
    generic /steer/* + /memory/* fallback in _dispatch_post below) raise/catch this specially so a dead
    engine answers with the same clean 502 shape every other engine-touching route uses, instead of a
    bare URLError leaking to the generic 500 fallback (see the engine-down pressure test's finding #1)."""


def _engine_unreachable_message() -> str:
    """The one canonical "the engine isn't there" message, shared by every caller that needs to tell a
    dead engine apart from some other failure (Substrate._ensure_steer, the receipt/rederive/swap_receipt
    routes, EngineSubstrate.jlens) -- so the wording is identical everywhere, not re-typed per call site."""
    base = getattr(ENGINE, "base", None) or "the engine"
    return f"engine not reachable at {base} -- is it running?"


def _engine_reachable(sub=None) -> bool:
    """True iff there's a real engine client to ask (`sub.engine`, falling back to the module-level
    ENGINE) AND a live GET /health round-trip on it actually connects. Used by routes whose underlying op
    (receipts.receipt/rederive/replay -- all documented "never raise into the caller") swallows a
    dead-engine URLError into an ambiguous None/False result: probing here on that ambiguous path (never
    on the happy path) lets the route tell "the engine is down" apart from "bad request/other failure"
    without threading exceptions through that deliberate no-raise contract.

    Returns True (i.e. "don't blame the engine") when there's no engine client available at all to probe,
    OR it has no `.health` method -- a duck-typed test substrate with no `.engine` (or one that doesn't
    implement health(), like several route tests' fakes), or a runtime with none configured -- so a
    failure that has nothing to do with engine connectivity keeps its ordinary generic message instead of
    being mislabeled "engine not reachable"."""
    eng = getattr(sub, "engine", None) if sub is not None else None
    if eng is None:
        eng = ENGINE
    if eng is None or not hasattr(eng, "health"):
        return True
    try:
        eng.health()
        return True
    except Exception:
        return False


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


# REPRODUCE_AND_PROVE_PLAN S5: settings-exposed interactive-chat sampling (Ollama/llama.cpp's canonical
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
    inverse-CDF draw). So a sampled reply lands on the same Ollama distribution the user knows. The
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
        # still being independently reproducible: re-POSTing /v1/completions with this same seed+params
        # reproduces the text (mode.decode records it on the run).
        "seed": int(overrides.get("seed", secrets.randbits(63))),
    }
    return resolved if resolved["temperature"] > 0 else None


def _sampling_settings():
    """The persisted S5 sampling settings (on/off + the four params) for the GET/POST /sampling/mode
    route -- what's actually configured, not what one specific turn resolved to. Unlike _resolve_sampling,
    this never generates a seed (a GET must never mutate anything)."""
    import clozn.settings as settings
    return {k: settings.get_setting(k, v) for k, v in _SAMPLING_DEFAULTS.items()}


def _engine_generation_meta(max_new=None, stream=None, sample=None):
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
    })


def _pers(name):
    return os.path.join(CLOZN_DIR, name)


# The honest J-lens provenance for the Run Inspector panel (J3), sourced from ~/.clozn/jlens/manifest.json
# (the sidecar the engine loaded). The `note` is verbatim from JLENS_ENGINE_PLAN.md -- a disposition, not a
# verified thought; a linear lens always emits something. Read once (cached); missing/broken manifest still
# yields the honest note (fit_model/layers just stay unfilled). Never raises.
_JLENS_NOTE = ("fitted linear Jacobian lens, transferred to this GGUF; a per-token 'disposed to say' read, "
               "NOT the model's literal thought — a linear lens always emits something.")


def _jlens_provenance():
    cached = getattr(_jlens_provenance, "_cached", None)
    if cached is not None:
        return dict(cached)
    prov = {"kind": "jacobian_lens", "fit_model": None, "layers": [], "note": _JLENS_NOTE}
    try:
        with open(os.path.join(_pers("jlens"), "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        prov["layers"] = [int(x) for x in (m.get("layers") or [])]
        model = str(m.get("model", "")).split("/")[-1].replace("-Instruct", "")
        fo = m.get("fitted_on") or {}
        bits = [b for b in ["HF", fo.get("quant"),
                            (f"{fo.get('n_prompts')} prompts" if fo.get("n_prompts") else None)] if b]
        if model:
            prov["fit_model"] = f"{model} ({', '.join(bits)})" if bits else model
    except Exception:
        pass
    _jlens_provenance._cached = dict(prov)
    return dict(prov)


def _jlens_run_text(run):
    """The text a run's J-lens should read + a source label the panel shows so it's honest about WHAT was
    read. Prefer the stored `response` (the reply whose tokens the chips annotate); else the last user
    message (a prompt-only run). ('' , 'none') when there's nothing to read."""
    if not isinstance(run, dict):
        return "", "none"
    resp = run.get("response")
    if isinstance(resp, str) and resp.strip():
        return resp, "response"
    msgs = run.get("messages") if isinstance(run.get("messages"), list) else []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user" and str(m.get("content", "")).strip():
            return str(m["content"]), "last_user_message"
    return "", "none"


def _jlens_workspace_readouts(res, run_id):
    """Map a J-lens readout into the protocol's workspace_readout shape (SPEC.md / WORKSPACE_LENS.md),
    provider_type 'jacobian_lens', readout_kind 'token' -- one event per position, so the same readout can
    ride the existing event spine. Additive/opt-in (request `protocol:true`); the panel uses tokens+readouts."""
    layer = res.get("layer")
    toks = res.get("tokens") or []
    rows = res.get("readouts") or []
    out = []
    for i, row in enumerate(rows):
        out.append({
            "type": "workspace_readout", "run_id": run_id,
            "token_index": i, "token_text": (toks[i] if i < len(toks) else None),
            "layer": layer, "position": i,
            "provider": f"jacobian_lens_l{layer}", "provider_type": "jacobian_lens",
            "readout_kind": "token",
            "top_readouts": [{"label": r.get("piece"), "score": r.get("score")} for r in (row or [])],
        })
    return out


def _jlens_envelope(res, run_id, text_source, want_protocol=False):
    """Shape EngineSubstrate.jlens's normalized dict into the J3 frontend contract. `res` is either
    {available:False, reason} or {available:True, layer, available_layers, n_tokens, tokens, readouts}
    (+ an `error` string when the layer was unknown). Carries the honest provenance block when available.
    HTTP is always 200 -- absence is a clean, renderable state, never an error."""
    if not res.get("available"):
        return {"available": False, "run_id": run_id, "reason": res.get("reason", "J-lens unavailable")}
    out = {"available": True, "run_id": run_id,
           "layer": res.get("layer"), "available_layers": res.get("available_layers", []),
           "text_source": text_source, "n_tokens": int(res.get("n_tokens", 0) or 0),
           "tokens": res.get("tokens", []), "readouts": res.get("readouts", []),
           "provenance": _jlens_provenance()}
    if res.get("error"):                       # unknown layer etc. -- surfaced cleanly (layers already listed)
        out["error"] = res["error"]
    if want_protocol:                          # bonus: also ride the workspace_readout protocol shape
        out["workspace_readouts"] = _jlens_workspace_readouts(res, run_id)
    return out


ENGINE_STEER = None        # lazy EngineSteer on the one GGUF engine -- tone dials work on any AR GGUF


def _engine_steer():
    global ENGINE_STEER
    if ENGINE_STEER is None and ENGINE is not None:
        from clozn.behavior.steering.engine_adapter import EngineSteer
        ENGINE_STEER = EngineSteer(ENGINE)
    return ENGINE_STEER


ENGINE_CONCEPT_STEER = None   # lazy ConceptSteer(ENGINE) -- Tier-1 #1's any-concept dial (dir(c)), see
                              # clozn/behavior/steering/concept_dir.py. Mirrors _engine_steer() above; a
                              # SEPARATE mechanism (dir(c), zero calibration) from the tone-dial EngineSteer.


def _engine_concept_steer():
    global ENGINE_CONCEPT_STEER
    if ENGINE_CONCEPT_STEER is None and ENGINE is not None:
        from clozn.behavior.steering.concept_dir import ConceptSteer
        ENGINE_CONCEPT_STEER = ConceptSteer(ENGINE)
    return ENGINE_CONCEPT_STEER


def _qwen_tmpl(messages):
    """Render chat messages into Qwen's chat-template STRING (ChatML). LEGACY: kept only as a documented
    reference / last-ditch fallback -- the engine generation paths now template PER-MODEL via
    _engine_tmpl (the GGUF's own embedded chat template), so a non-Qwen model gets its correct format.
    The torch QwenSubstrate never used this (it applies the HF tokenizer's template internally)."""
    sysmsg = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    for m in messages:
        if m.get("role") == "system" and m.get("content"):
            sysmsg = m["content"]
    s = f"<|im_start|>system\n{sysmsg}<|im_end|>\n"
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            s += f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


def _engine_tmpl(engine, messages, usage_out=None):
    """Render chat `messages` to a prompt string using the ENGINE-LOADED MODEL'S OWN chat template
    (POST /apply_template -> llama_chat_apply_template over the GGUF's tokenizer.chat_template). THIS is
    what makes the engine paths model-agnostic: whatever GGUF the engine has loaded, its messages are
    formatted in that model's native template (Qwen ChatML, Llama-3 headers, Gemma turns, ...), instead
    of a hardcoded Qwen string. Replaces _qwen_tmpl on every EngineSubstrate generation path.

    Errors propagate deliberately (no silent Qwen fallback): a model with no embedded template, or an
    engine too old to expose /apply_template, must surface -- silently mis-formatting the prompt is the
    exact bug this removes. Callers that need a soft degrade catch EngineError themselves."""
    if usage_out is not None and hasattr(engine, "apply_template_info"):
        info = engine.apply_template_info(messages)
        prompt_tokens = info.get("prompt_tokens") if isinstance(info, dict) else None
        if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool):
            usage_out["prompt_tokens"] = prompt_tokens
        return info["prompt"]
    return engine.apply_template(messages)


def _model_scoped_path(name):
    """Per-exact-GGUF product state path, with a legacy fallback for lab/tests."""
    sub = globals().get("SUB")
    digest = getattr(sub, "model_sha256", None) if sub is not None else None
    if digest:
        return _pers(os.path.join("models", str(digest), name))
    return _pers(name)


def _disk_dials():
    """Saved tone-dial values for the active exact GGUF."""
    path = _model_scoped_path("studio_personality.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return {k: float(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def _dial_calibration():
    """The curated per-model dial calibration (research/dial_autocalibrate.py's sweep -- see that module's
    docstring), read from ~/.clozn -- NEVER research/runs/dial_autocalibrate.json directly (that raw
    research file carries full curves + sample_replies per dial, meant for a human to eyeball, not to be
    re-parsed on every /steer/axes call; a curator step distills it down to just what a slider needs and
    persists THAT here). Missing/broken file -> {} (never raise): calibration is optional enrichment, so
    every caller must behave EXACTLY as it did before this existed when the file isn't there yet.

    Returns {dial_name: {"usable_max", "usable_range", "derail_point", "works"}, ...}. Tolerant of either a
    flat {dial_name: {...}} file, or one shaped like the raw research JSON ({"dials": {dial_name: {...}}},
    with "range_valid" instead of "works") -- whichever shape the curated file ends up in, this keeps
    working. A per-entry parse problem drops just that one dial (skipped, not crashed on)."""
    path = _model_scoped_path("dial_calibration.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        table = raw.get("dials", raw) if isinstance(raw, dict) else {}
        out = {}
        for name, c in table.items():
            if not isinstance(c, dict):
                continue
            works = c.get("works", c.get("range_valid"))
            out[name] = {
                "usable_max": c.get("usable_max"),
                "usable_range": c.get("usable_range"),
                "derail_point": c.get("derail_point"),
                "works": bool(works),
            }
        return out
    except Exception:
        return {}


def _with_calibration(axis, c):
    """Merge one _dial_calibration() entry into one /steer/axes axis dict, IN PLACE (also returned, so this
    reads as an expression in a list comprehension) -- the one spot that decides what a calibrated slider
    looks like to the studio UI. No entry for this dial (c falsy/missing) -> the axis is untouched except
    for "calibrated": False added: NO behavior change for a dial nobody has calibrated on this model (Law
    #6 -- the uncalibrated case must render exactly as it always has). An entry -> "max" becomes the
    CALIBRATED usable_max, falling back to the axis's own already-declared max when usable_max itself is
    None (a dial swept but never found usable), plus "usable_range"/"derail_point"/"works" for the UI to
    grey out a dead dial or show its working range."""
    if not c:
        # UNCALIBRATED -> advertise the ceiling that EngineSteer.set() will actually ENFORCE, not the
        # axis's declared max. These two used to disagree: the slider offered 1.5 while nothing had
        # ever swept this model, and a user who dragged it there got repetition loops. (Measured on
        # Qwen2.5-7B-Q4: coherent at 0.25, already degenerate at 0.5, pure loops at 1.5 while looking
        # like a 50% length win.) A slider must never offer a value the API will silently clamp.
        from clozn.behavior.steering.engine_adapter import UNSAFE_STRENGTH
        axis["calibrated"] = False
        axis["declared_max"] = axis.get("max")
        axis["max"] = min(float(axis.get("max", 1.5)), float(UNSAFE_STRENGTH))
        axis["uncalibrated"] = (
            f"No calibration for this exact model, so this dial is capped at {UNSAFE_STRENGTH} "
            f"instead of its declared {axis['declared_max']}. Calibrate this model to unlock its "
            f"real range."
        )
        return axis
    axis["max"] = c["usable_max"] if c.get("usable_max") is not None else axis["max"]
    axis["usable_range"] = c.get("usable_range")
    axis["derail_point"] = c.get("derail_point")
    axis["works"] = bool(c.get("works"))
    axis["calibrated"] = True
    return axis


def _library_dial_names() -> set:
    """The set of custom-dial names that are SHIPPED-LIBRARY dials (research/deploy_dial_library.py's
    one-time registration), read from ~/.clozn/studio_library.json's keys -- a file distinct from the
    user's own studio_custom_<name>.json, so a library dial can never be mistaken for something the user
    made. Missing/broken file -> empty set (never raise): before the library is deployed (or on any
    substrate that has never loaded studio_library.json), /steer/axes must behave EXACTLY as it always
    has -- every steer.custom entry tags "custom": True and none tag "library" -- the same Law-#6-style
    backward compat _dial_calibration() already gives the calibration merge."""
    path = _pers("studio_library.json")
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, dict) else set()
    except Exception:
        return set()


# ------- memory assembly: extracted to clozn/server/memory_assembly.py; re-exported (the seam) ------
from clozn.server.memory_assembly import (                                               # noqa: E402
    _SUSPICIOUS, QUOTE_SPAN_MAX, PROMPT_GATE_MIN,
    _risk_of, _dial_suggestion, _provenance_of, _memory_mode, _last_user,
    _prompt_gate, _prompt_relevance, _prompt_mem_cards, _prompt_block_for,
    PromptBlockDecision, _capture_prompt_decision, _baseline_prompt_tokens,
    _inject_block,
    _mem_migrate, _export_markdown, _runs_for_card, _mem_sync_rules,
)

# ------- profiles: named persona bundles -> cards + dials on the LIVE substrate ---------------------
# profiles.py is the model-free CRUD + compile layer (source bundles: card texts, dial settings, custom-
# dial recipes, fact pairs -- see its docstring). This is the thin wiring that hands it the live objects
# a switch needs (SUB._mem for cards/rules, SUB.steer for dials) and reports what actually happened.

def _active_profile_name():
    """The name of the last-switched-to profile, or None (nothing switched yet this install). Persisted
    in studio_settings.json alongside memory_mode -- one small settings file, not a new one."""
    import clozn.settings as settings
    return settings.get_setting("active_profile")


def _profiles_switch(sub, p) -> dict:
    """Apply profile bundle `p` to the live substrate `sub`: global cards REPLACE the studio's global
    set (a profile switch is a replacement, never a merge -- disjoint personas must not bleed), while
    app/project cards remain as request-selected overlays,
    dials replace via profiles.apply_dials (steer.clear() then set()), and the prompt-mode/internalized
    resync goes through the SAME _start_retrain machinery every other card mutation uses: instant in
    prompt mode (the cards ARE the memory there), a backgrounded consolidate() in internalized mode.

    Facts: the slot-memory store (clozn/lab/slotmem_qwen) is a lab-only research module -- it was
    never wired to change the product reply, and reorg Stage B removed the product-side facts surface
    entirely (no SlotBox, no /facts/* routes here). A profile's facts still travel in the bundle but
    are never compiled into anything on the product server -- reported honestly via `facts_note`,
    never silently dropped. Returns {name, prompt_block, cards:{removed,added}, dials, resync, facts_note}."""
    import clozn.memory.cards as memory_cards

    # 1) CARDS: delete the current GLOBAL set, then create the profile's cards fresh as global+active.
    #    App/project cards are independent overlays and survive switches byte-for-byte. Deleting the old
    #    globals (not just disabling them) still keeps disjoint profile personas isolated.
    from clozn.memory.scope import global_scope, scope_for_card
    removed = 0
    for c in memory_cards.list_cards():
        if scope_for_card(c)["kind"] == "global" and memory_cards.delete(c["id"]):
            removed += 1
    added = 0
    for c in p.get("cards", []):
        if c.get("status", "active") != "active":     # a disabled card in the bundle stays inert here too
            continue
        if memory_cards.create(c["text"], status="active", kind="preference",
                               evidence=f"profile:{p['name']}", scope=global_scope()) is not None:
            added += 1

    # 2) SYNC the memory mechanism from the new active set. force=True: the pre-check inside
    #    _start_retrain compares m.rules to memory_cards.active_texts(), and since we just rewrote the
    #    store out from under it, that comparison alone isn't trustworthy for a switch -- force skips it
    #    and always resyncs, exactly as the mode-switch catch-up (POST /memory/mode) already does.
    resync = {"retraining": False}
    m = getattr(sub, "_mem", None)
    if m is not None:
        resync = sub._start_retrain(m, "profile-switch", None, force=True)

    # 3) DIALS: replace via profiles.apply_dials (clear() then set(); custom-dial recipes recompute if
    #    not already present) -- persist exactly like /steer/set and /steer/custom already do, so the
    #    switched-to persona survives a restart the same way a manually-set dial would.
    dials = {"applied": {}, "customs_added": []}
    steer = getattr(sub, "steer", None)
    if steer is not None:
        from clozn.profiles import store as profiles
        dials = profiles.apply_dials(p, steer)
        try:
            if hasattr(steer, "save_state"):
                steer.save_state(getattr(sub, "_pers_steer", None) or
                                 _model_scoped_path("studio_personality.json"))
            if dials["customs_added"] and hasattr(steer, "save_custom"):
                steer.save_custom(_pers(f"studio_custom_{getattr(sub, 'name', SUBNAME)}.json"))
        except Exception:
            pass

    # 4) FACTS: reorg Stage B pulled the facts/slot-memory tier off the product surface entirely (it's a
    #    lab-only research module now, clozn/lab/slotmem_qwen). A profile's fact pairs still travel in
    #    the bundle (export/import stays lossless), but the product server never compiles them into a
    #    live store -- facts_note says so honestly, always.
    import clozn.settings as settings
    settings.set_setting("active_profile", p["name"])

    facts_note = None
    if p.get("facts"):
        facts_note = (f"{len(p['facts'])} fact(s) travel in the bundle but are not compiled anywhere -- "
                      "the facts/slot-memory tier is a lab-only research module, not part of the product.")
    return {"name": p["name"], "prompt_block": prompt_block_preview(p),
            "cards": {"removed": removed, "added": added}, "dials": dials,
            "resync": resync, "facts_note": facts_note}


def prompt_block_preview(p) -> str:
    """The system block this profile WOULD inject (profiles.prompt_block) -- for the switch response's
    receipt only; the live chat path still compiles fresh from the card store every gated-in turn."""
    from clozn.profiles import store as profiles
    return profiles.prompt_block(p)


ARGS = None
SUB = None         # the active substrate object
SUBNAME = "engine"  # product server has one substrate; PyTorch model work lives in the lab

# One process-wide time-travel SnapshotStore: the bounded, CPU-offloaded ring of per-turn
# KV snapshots. Built lazily from timetravel.get_config() (cap / byte-budget). Only ever holds real KV
# payloads when the `timetravel_snapshots` gate is ON (the RAM rule); branch RECORDING (the transcript
# transform -> child run) does not need it and works regardless. None until first requested.
SNAPSHOTS = None


def _snap_store():
    """The process-wide time-travel SnapshotStore, built lazily with the persisted ring config. Never
    raises -- a config hiccup falls back to the module defaults."""
    global SNAPSHOTS
    if SNAPSHOTS is None:
        try:
            import clozn.replay.timetravel as timetravel
            cfg = timetravel.get_config()
            SNAPSHOTS = timetravel.SnapshotStore(cap=cfg["cap"], budget_mb=cfg["budget_mb"])
        except Exception:
            import clozn.replay.timetravel as timetravel
            SNAPSHOTS = timetravel.SnapshotStore()
    return SNAPSHOTS


# ------- substrates: extracted to clozn/server/substrates.py; re-exported here (the seam) -----------
# app.py remains the canonical module: routes read ctx.<name> and tests patch cs.<name> on THIS module.
from clozn.server.substrates import (                                                    # noqa: E402
    Substrate, EngineSubstrate, _EngineMemory,      # Qwen/Dream lab substrates now live in clozn/lab/substrates.py
    _quant_from_name, _model_family_from_name,
    _engine_model_info, _engine_complete_traced, _ENGINE_MODELS, _ENGINE_MODEL_DEFAULT,
)

# ------- route families: registered in dispatch order; each exposes try_get/try_post(handler, path, ...) --
# returning True once it has written a response, so do_GET/do_POST can stop at the first family that
# claims a path -- exactly the first-match-wins order the old inline if/elif chain had. POST falls
# through, after every family, to the generic SUB.handle(path, body) substrate dispatch inlined at the
# end of do_POST (the catch-all for /memory/*, /steer/* -- those live in the Substrate classes above, not
# in a route module, since they're substrate-polymorphic domain dispatch, not per-path HTTP routing).
#
# runs.try_get_fallback (the generic GET /runs/<id>) is registered LAST in _GET_ROUTES, deliberately --
# every more-specific /runs/<id>/<suffix> family (receipts' /export, runs' own /timeline etc.) must get
# first refusal, since they all share the "/runs/" prefix the fallback also matches.
import types as _types                                                # noqa: E402

from clozn.server import static as _static_routes                     # noqa: E402
from clozn.server.routes import health as _health_routes              # noqa: E402
from clozn.server.routes import runs as _runs_routes                  # noqa: E402
from clozn.server.routes import memory as _memory_routes              # noqa: E402
from clozn.server.routes import receipts as _receipts_routes          # noqa: E402
from clozn.server.routes import replay as _replay_routes              # noqa: E402
from clozn.server.routes import corrective_retries as _corrective_retry_routes  # noqa: E402
from clozn.server.routes import timetravel as _timetravel_routes      # noqa: E402
from clozn.server.routes import profiles as _profiles_routes          # noqa: E402
from clozn.server.routes import preferences as _preferences_routes    # noqa: E402
from clozn.server.routes import feedback as _feedback_routes          # noqa: E402
from clozn.server.routes import ollama as _ollama_routes              # noqa: E402
from clozn.server.routes import openai as _openai_routes              # noqa: E402
from clozn.server.routes import engine as _engine_routes              # noqa: E402
from clozn.server.routes import guard as _guard_routes                # noqa: E402 (persisted /guard/mode)
from clozn.server.routes import models as _models_routes              # noqa: E402 (local GGUF inventory)
from clozn.server.routes import readouts as _readouts_routes          # noqa: E402
# Inspector route families: fork-at-token, journal actuary, model diff (F8).
from clozn.server.routes import provenance as _provenance_routes       # noqa: E402
from clozn.server.routes import causal_trace as _causal_trace_routes   # noqa: E402
from clozn.server.routes import fork as _fork_routes                   # noqa: E402
from clozn.server.routes import journal as _journal_routes             # noqa: E402
from clozn.server.routes import diff as _diff_routes                   # noqa: E402
from clozn.server.routes import receipt_link as _receipt_link_routes   # noqa: E402 (ambient delivery ch.1)
from clozn.server.routes import influence_map as _influence_map_routes # noqa: E402
from clozn.server.routes import contracts as _contracts_routes         # noqa: E402 (hook vocab + replay, P4.2)

_runs_fallback_routes = _types.SimpleNamespace(try_get=_runs_routes.try_get_fallback)

_GET_ROUTES = [_static_routes, _health_routes, _runs_routes, _memory_routes, _receipts_routes,
              _timetravel_routes, _profiles_routes, _ollama_routes, _openai_routes, _engine_routes,
              _guard_routes, _models_routes,
              _journal_routes, _diff_routes, _receipt_link_routes,
              _influence_map_routes, _contracts_routes, _runs_fallback_routes]
_POST_ROUTES = [_health_routes, _memory_routes, _receipts_routes,
               _corrective_retry_routes, _replay_routes,
               _timetravel_routes, _profiles_routes, _preferences_routes, _feedback_routes,
               _ollama_routes, _openai_routes, _engine_routes, _guard_routes, _readouts_routes,
               _fork_routes, _journal_routes, _diff_routes,
               _receipt_link_routes, _influence_map_routes, _contracts_routes,
               _provenance_routes, _causal_trace_routes]


def active_sub(h):
    """The substrate for this request: the handler's injected substrate if it has one, else the module
    global SUB (product default; what tests monkeypatch as cs.SUB). getattr-based so a plain/mock handler
    with no injection resolves to the global -- routes never touch the raw global directly."""
    inj = getattr(h, "_inj_sub", None)
    return inj if inj is not None else SUB


def active_subname(h):
    inj = getattr(h, "_inj_subname", None)
    return inj if inj is not None else SUBNAME


def active_kind(h):
    inj = getattr(h, "_inj_kind", None)
    return inj if inj is not None else RUNTIME_KIND


def make_handler(sub=None, subname=None, runtime_kind=None):
    """Build the HTTP handler class. sub/subname/runtime_kind are INJECTABLE: when None they fall back to
    this module's globals (SUB/SUBNAME/RUNTIME_KIND) -- what `clozn serve` sets and what tests monkeypatch
    as cs.SUB -- so the product path and its tests are unchanged. `clozn lab` passes its own, so it owns
    its handler + substrate WITHOUT reaching in to mutate this module's globals. Handler code reads
    _sub()/_subname()/_kind(); route modules read h.sub / h.subname / h.runtime_kind (same source)."""
    def _sub():
        return sub if sub is not None else SUB

    def _subname():
        return subname if subname is not None else SUBNAME

    def _kind():
        return runtime_kind if runtime_kind is not None else RUNTIME_KIND

    class H(BaseHTTPRequestHandler):
        # The injected substrate/name/kind for this handler (None on the product path, which uses the
        # module globals). Route modules resolve them via ctx.active_sub(h)/active_subname(h)/active_kind(h),
        # which getattr-fall-back to the globals when unset -- so a plain/mock handler with no injection
        # still resolves to cs.SUB (what the product sets and tests monkeypatch). Handler-internal code
        # uses the _sub()/_subname()/_kind() closures above (same source).
        _inj_sub, _inj_subname, _inj_kind = sub, subname, runtime_kind

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
            self._send(code, json.dumps(o), "application/json", extra_headers=extra_headers)

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
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            requested = self.headers.get("Access-Control-Request-Headers")
            self.send_header("Access-Control-Allow-Headers", requested or DEFAULT_ALLOW_HEADERS)
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _client(self, ua):
            ua = (ua or "").lower()
            for k, v in (("open-webui", "Open WebUI"), ("openwebui", "Open WebUI"), ("cursor", "Cursor"),
                         ("vscode", "VS Code"), ("python-requests", "script"), ("httpx", "script"),
                         ("openai-python", "script"), ("curl", "curl"), ("mozilla", "browser")):
                if k in ua:
                    return v
            return ua[:24] or "unknown"

        def _workspace_lens_provider(self, messages, response, error=None):
            """Return a provider callback for real concept readouts, if the brain stack is live.

            Preferred path: C++ engine activations -> SAE concepts (`engine_concepts`).
            Fallback path: loaded Python Qwen activations -> SAE/probe concepts (`sae/probe`).
            No mock data is attached to real runs from here.
            """
            if error or not response or not (_sub() and getattr(_sub(), "brain", None)):
                return None
            text = str(response or _last_user(messages) or "").strip()[:300]
            if not text:
                return None

            def provider(rid, norm_trace):
                from clozn.readouts import workspace_lens
                if not norm_trace or not norm_trace.get("tokens"):
                    return []
                if ENGINE is not None:
                    try:
                        data = _sub().brain.concepts_from_engine(text, ENGINE, 15)
                        return workspace_lens.readouts_from_concepts(
                            rid, norm_trace, data, provider="engine_concepts", layer=data.get("layer"))
                    except Exception:
                        pass
                try:
                    data = _sub().brain.concepts_only(text)
                    return workspace_lens.readouts_from_concepts(
                        rid, norm_trace, data, provider="sae/probe", layer=15)
                except Exception:
                    return []

            return provider

        def _log_run(self, source, messages, response, model, started, error=None, trace=None,
                     mem_out=None, finish_reason=None, finish_reason_fallback=None, extra_meta=None,
                     output_contract=None):
            """Persist this interaction as an inspectable run (never let logging break the request).
            mem_out (prompt mode): the {applied, gate, strength?} record the generation path filled --
            what memory ACTUALLY rode this turn (the topic gate may have omitted the block).
            extra_meta (optional): additional {key: value} pairs folded into the run's `meta` block on top
            of the substrate/build-commit/capture-tier fields this method already assembles -- e.g. sse.py
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
                from clozn.runs.association import request_client, request_project, request_session
                session_key = request_session(self.headers)
                client_key, client_key_source = request_client(self.headers)
                project_key = request_project(self.headers)
                mem = getattr(_sub(), "_mem", None) if _sub() else None
                mo = mem_out or {}
                from clozn.runs.think_tags import prompt_opens_think, sanitize_messages, sanitize_reply
                messages = sanitize_messages(messages)
                think = sanitize_reply(response, implicit_open=prompt_opens_think(mo.get("final_prompt")))
                response = think.public_text
                reasoning = think.journal()
                mode = mo.get("mode") or _memory_mode()
                if mode == "prompt":
                    # cards_applied == what was INJECTED this turn -- the per-turn honesty prompt mode
                    # buys (internalized can only report the whole active set). applied_ids ride along so
                    # the Run Inspector can offer per-card receipts. A path that filled nothing (or
                    # errored before generating) honestly records an empty application.
                    applied = [c for c in (mo.get("applied") or []) if isinstance(c, dict)]
                    strength = mo.get("strength",
                                      getattr(mem, "memory_strength", 1.0) if mem is not None else 1.0)
                    memd = {"cards_applied": [c.get("text", "") for c in applied],
                            "applied_ids": [c.get("id") for c in applied],
                            "strength": float(strength),
                            "has_prefix": (getattr(mem, "prefix", None) is not None) if mem is not None else False,
                            "mode": mode, "proposed_cards": []}
                    applied_scope_kinds = [c.get("scope_kind") for c in applied]
                    if any(kind in {"global", "app", "project"} for kind in applied_scope_kinds):
                        memd["applied_scope_kinds"] = [
                            kind if kind in {"global", "app", "project"} else "global"
                            for kind in applied_scope_kinds
                        ]
                    rel = [c.get("relevance") for c in applied]   # per-card topic cosine, aligned with cards_applied
                    if any(r is not None for r in rel):           # omit entirely when the embedder was unavailable
                        memd["relevance"] = [round(float(r), 4) if r is not None else None for r in rel]
                    if mo.get("gate") is not None:
                        memd["gate"] = round(float(mo["gate"]), 4)
                    if mo.get("prompt_block"):
                        memd["prompt_block"] = str(mo["prompt_block"])
                    for key in ("candidate_cards", "omitted_cards", "selection_stage", "omission_reason",
                                "prompt_token_cost_unavailable_reason"):
                        if key in mo:
                            value = mo[key]
                            memd[key] = ([dict(card) for card in value if isinstance(card, dict)]
                                         if key.endswith("_cards") and isinstance(value, list) else value)
                    if applied:                                  # bump exactly the cards that rode this turn
                        try:
                            import clozn.memory.cards as memory_cards
                            for c in applied:
                                if c.get("id"):
                                    memory_cards.bump_usage(c["id"])
                        except Exception:
                            pass
                elif mem is not None:
                    # INTERNALIZED: cards_applied == the ACTIVE-card texts. Post-D2, SUB._mem.rules is kept
                    # in sync with the active cards (see _mem_sync_rules), so reading .rules still reports
                    # exactly what shaped the reply. Reading SUB.memory would miss the dream cards -- use
                    # _mem (self.memory on qwen, self.dmem on dream). Only ACTIVE cards feed the prefix.
                    cards = getattr(mem, "rules", None) or getattr(mem, "cards", None) or []
                    memd = {"cards_applied": list(cards),
                            "strength": float(getattr(mem, "memory_strength", 1.0)),
                            "has_prefix": getattr(mem, "prefix", None) is not None,
                            "mode": mode, "proposed_cards": []}
                    if cards:                                    # record that the active cards influenced a run
                        try:
                            import clozn.memory.cards as memory_cards
                            for c in memory_cards.list_cards(status="active"):
                                memory_cards.bump_usage(c["id"])
                        except Exception:
                            pass
                else:
                    memd = {"mode": mode}                        # runlog records the mode on EVERY run
                # only meaningfully-nonzero dials (|v| >= 0.05); steer.active() drops exact-zeros but a
                # slider nudged to a hair (e.g. 0.02) still slips through and would clutter the record.
                if "active_dials" in mo:
                    # Some accepted surfaces (currently raw legacy text completions) build steering
                    # explicitly and report exactly what reached the worker. This prevents a live dial
                    # setting from being journaled as applied when that surface could not materialize it.
                    dials = dict(mo.get("active_dials") or {})
                else:
                    dials = _sub().steer.active() if (_sub() and hasattr(_sub(), "steer")) else {}
                dials = {k: v for k, v in dials.items() if abs(float(v)) >= 0.05}
                meta = None
                try:                                          # engine: {model_file, quant, mode, sampling}
                    if _sub() is not None and hasattr(_sub(), "run_meta"):
                        meta = _sub().run_meta() or None
                except Exception:
                    meta = None
                meta = dict(meta or {})
                active_profile = _active_profile_name()
                if active_profile:
                    meta.setdefault("active_profile", active_profile)
                try:
                    prompt_tokens = (_sub().last_prompt_tokens()
                                     if _sub() is not None and hasattr(_sub(), "last_prompt_tokens") else None)
                    if isinstance(prompt_tokens, int):
                        meta.setdefault("prompt_tokens", prompt_tokens)
                except Exception:
                    pass
                actual_prompt_tokens = mo.get("actual_prompt_tokens")
                if isinstance(actual_prompt_tokens, int) and not isinstance(actual_prompt_tokens, bool):
                    meta["prompt_tokens"] = actual_prompt_tokens
                if extra_meta:
                    meta.update({k: v for k, v in extra_meta.items() if v is not None})
                if mode == "prompt":
                    if not memd.get("cards_applied"):
                        memd["prompt_token_cost"] = 0
                    else:
                        baseline_tokens = mo.get("baseline_prompt_tokens")
                        prompt_tokens = mo.get("actual_prompt_tokens", meta.get("prompt_tokens"))
                        if (isinstance(baseline_tokens, int) and not isinstance(baseline_tokens, bool)
                                and isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool)
                                and prompt_tokens >= baseline_tokens):
                            memd["baseline_prompt_tokens"] = baseline_tokens
                            memd["prompt_token_cost"] = prompt_tokens - baseline_tokens
                git = _git_commit()
                if git:
                    meta.setdefault("build_git_commit", git)
                if finish_reason:
                    meta.setdefault("finish_reason_source", "substrate")
                elif finish_reason_fallback:
                    meta.setdefault("finish_reason_source", "fallback")
                    meta.setdefault("finish_reason_fallback", finish_reason_fallback)
                try:                                          # CAPTURE TIER: record it, and drop the trace at light
                    from clozn.runs import capture_mode
                    tier = capture_mode.tier()
                    meta = {**meta, "capture_tier": tier}
                    if not capture_mode.captures_trace(tier):
                        trace = None                          # light: text + finish_reason + metadata only
                except Exception:
                    pass
                workspace_provider = self._workspace_lens_provider(messages, response, error)
                assembled_messages = mo.get("assembled_messages") if mode == "prompt" else None
                # backlog #5: the EXACT rendered chat-template string the engine produced (mem_out fills it
                # on the engine chat paths). Captured in ANY memory mode -- the internalized/engine path
                # still renders a prompt even without a block. None -> consumers fall back to assembled_messages.
                final_prompt = mo.get("final_prompt")
                # roadmap S4.3: immutable reproduction identity (model_sha256, template_fingerprint,
                # engine_build, clozn_version) -- see clozn.runs.identity's module docstring for why this
                # never adds a file hash or a network round trip to THIS request (EngineSubstrate.
                # identity_meta() piggybacks run_meta()'s single cached /health fetch). Substrates that
                # don't implement identity_meta() (the Torch lab adapters) simply log no identity block.
                identity = None
                try:
                    if _sub() is not None and hasattr(_sub(), "identity_meta"):
                        identity = _sub().identity_meta() or None
                except Exception:
                    identity = None
                rid = runlog.record(source=source, client=self._client(self.headers.get("User-Agent", "")),
                                    model=str(model), substrate=_subname(), messages=messages, response=response,
                                    memory=memd, behavior={"active_dials": dials}, started=started, error=error,
                                    trace=trace, finish_reason=finish_reason, meta=meta,
                                    assembled_messages=assembled_messages, final_prompt=final_prompt,
                                    workspace_provider=workspace_provider, identity=identity,
                                    reasoning=reasoning, session_key=session_key,
                                    client_key=client_key, client_key_source=client_key_source,
                                    project_key=project_key,
                                    output_contract=output_contract)
                self._maybe_snapshot_turn(rid, messages, trace, error)
                return rid                        # M5 bridge: the run id, for callers that want to surface it
            except Exception:
                return None                        # logging must never break the request -- no id to surface

        def _maybe_snapshot_turn(self, rid, messages, trace, error):
            """TIME-TRAVEL (#6): when the snapshot gate is ON, register this turn in the bounded ring so the
            Run Inspector's rewind/branch reflects real recorded turns and the ring's bounded eviction runs
            in production. Fully guarded + gated OFF by default (the RAM rule). NOTE: the studio chat path is
            STATELESS (SelfTeach._generate builds its own cache via generate() and discards it), so v1
            records a DESCRIPTOR-only snapshot (turn index + token count, zero offloaded bytes) -- honest,
            and enough for the branch bookkeeping. Capturing the live KV payload here (the re-prefill fast
            path) needs the generation path to hand back its cache: the documented next rung."""
            try:
                if not rid or error:
                    return
                import clozn.replay.timetravel as timetravel
                if not timetravel.enabled():
                    return
                store = _snap_store()
                if store is None:
                    return
                turn = max(0, len(timetravel.message_turns(messages)) - 1)   # this reply's turn index
                # `trace` is a raw per-token step LIST here (runlog normalizes it later) -> len == tokens;
                # tolerate a pre-normalized {tokens:[...]} dict too. 0 when no trace was captured.
                if isinstance(trace, list):
                    n_tok = len(trace)
                elif isinstance(trace, dict):
                    n_tok = len(trace.get("tokens", []) or [])
                else:
                    n_tok = 0
                store.snapshot_turn(rid, turn, n_tok=n_tok, meta={"stateless": True})
            except Exception:
                pass

        def do_GET(self):
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

        def do_POST(self):
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
                # See _GATE_EXEMPT_POSTS' module-level comment: these two paths are audited to touch
                # nothing the gate protects, so they run uncontended -- notably, they never sit in the
                # bounded queue behind a slow generation request either.
                self._dispatch_post(p, body)
                return
            rejected = POST_GATE.acquire(cancel_check=lambda: client_gone(self))
            if rejected:
                # "cancelled": client_gone(self) fired while this request was queued -- the requester is
                # confirmed gone, so this response is best-effort only (mirrors "full"/"timeout": nothing
                # here relies on it actually being delivered). 499 (nginx's convention for "client closed
                # the request") -- there is no standard status for this, but it is the least-ambiguous
                # existing one for observability (access logs, metrics) even if nobody reads the body.
                status = {"full": 429, "timeout": 503, "cancelled": 499}[rejected]
                message = {"full": "request queue is full",
                           "timeout": "request timed out while waiting for the model",
                           "cancelled": "client disconnected while queued"}[rejected]
                self._json(status, {"error": {"message": message, "type": "server_busy"}},
                           extra_headers={"Retry-After": "1"})
                return
            try:
                self._dispatch_post(p, body)
            finally:
                POST_GATE.release()

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
            # Generic fallback: whatever the active substrate's own per-action dispatch serves
            # (/memory/*, /steer/* -- Substrate._memory/_steer above, substrate-polymorphic domain
            # dispatch, not per-path HTTP routing, so it stays here rather than in a route module).
            try:
                routed_body = body
                if p in ("/memory/add", "/memory/scope") and isinstance(body, dict):
                    # The public body selects only the kind. Exact opaque keys come from validated
                    # request headers and cross the substrate boundary as an unforgeable Python object;
                    # JSON callers cannot inject this private value themselves.
                    from clozn.server.generation_gateway import request_memory_scope
                    routed_body = dict(body)
                    routed_body["_memory_scope"] = request_memory_scope(self)
                r = _sub().handle(p, routed_body) if _sub() else None
                if r is None:
                    return self._json(409, {"error": f"'{p}' isn't served by the '{_subname()}' substrate",
                                            "need": "dream" if p == "/denoise" else "qwen", "active": _subname()})
                self._json(200, r)
            except EngineUnavailable as e:
                # the engine is down (see Substrate._ensure_steer) -- the same clean 502 shape every
                # other engine-touching route uses, not the generic 500 below.
                self._json(502, {"error": str(e)})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def main():
    global ARGS, SUB, SUBNAME, RUNTIME_KIND
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ARGS = ap.parse_args()
    if ENGINE is None:
        ap.error("CLOZN_ENGINE_PORT is missing; launch the product with `clozn serve <model>`")
    os.environ["CLOZN_RUNTIME_KIND"] = "product"
    RUNTIME_KIND = "product"
    SUBNAME = "engine"
    print("clozn gateway: connecting to private model worker ...", flush=True)
    SUB = EngineSubstrate()
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), make_handler())
    print(f"\n  Clozn -> http://{ARGS.host}:{ARGS.port}/\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
