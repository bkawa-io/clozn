"""commands.models -- model discovery (`clozn models`), fetch (`clozn pull`), and the fit planner
(`clozn plan`): "will it fit?" answered by reading only a GGUF's header -- a few MB at the front of the
file -- never the multi-GB tensor payload, never a model load, never the GPU.

resolve_model()/_flags_for()/_friendly() are also used by the run/serve/explain command modules to turn a
model name/path argument into a resolved GGUF + its launch flags.

HOME/CloznError live on `clozn.cli.main`; every function here that needs either does
`from clozn.cli import main as ctx` INSIDE the function body (never at module level -- main.py imports
this module at its own module level, so a module-level back-reference would deadlock the first time
something imports clozn.cli.commands.models before clozn.cli.main; see engine_process.py's docstring for
the full trace).

DISCOVERY IS SHARED WITH THE SERVER: `_model_dirs`/`_scan_models` below are now thin re-exports of
clozn.models.inventory.model_dirs/scan_models (the server's GET /models/local route uses the same two
functions -- see that module's docstring for why the scan itself had to move out of this CLI-only module
rather than the server importing clozn.cli). Kept as the same private names here so every existing
`from clozn.cli.commands.models import _model_dirs, _scan_models` call site (e.g. doctor.py) keeps working
unchanged; behavior is byte-identical, not just equivalent (clozn.models.inventory.HOME/REPO/ENGINE_CORE
resolve to the exact same strings this module's own copies used to).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from clozn.cli import formatting as fmt
from clozn.cli.engine_process import find_engine
from clozn.models.inventory import model_dirs as _model_dirs, quant_label as _quant_label, scan_models as _scan_models

# Known models: a filename fragment -> friendly name + launch flags. mask/eos => diffusion; chat => wrap the
# prompt in the chat template; AR models need no special flags (the engine auto-detects mode from the GGUF).
KNOWN = [
    ("qwen3.5-9b",            "qwen3.5",    {"chat": True}),
    ("qwen2.5-7b-instruct",   "qwen",      {"chat": True}),
    ("qwen2.5-0.5b-instruct", "qwen-0.5b", {"chat": True}),
    ("meta-llama-3.1-8b-instruct", "llama", {"chat": True, "tmpl": "llama3"}),
    ("ministral-3-3b-instruct-2512", "ministral", {"chat": True, "tmpl": "mistral"}),
    ("gemma-4-e4b-it",        "gemma4",     {"chat": True, "tmpl": "gemma"}),
    ("mistral-7b-instruct",   "mistral",   {"chat": True, "tmpl": "mistral"}),
    ("llama-3.2-1b-instruct", "llama-1b",  {"chat": True, "tmpl": "llama3"}),
    ("llama-3.2-3b-instruct", "llama-3b",  {"chat": True, "tmpl": "llama3"}),
    ("gemma-2-2b-it",         "gemma-2b",  {"chat": True, "tmpl": "gemma"}),
]

# Models `clozn pull` knows how to fetch: name -> (HF repo, file). Verified ungated single-file GGUFs.
# Anything else: `clozn pull owner/repo/file.gguf`.
PULLABLE = {
    "qwen-0.5b": ("bartowski/Qwen2.5-0.5B-Instruct-GGUF",    "Qwen2.5-0.5B-Instruct-Q8_0.gguf"),
    "qwen":      ("bartowski/Qwen2.5-7B-Instruct-GGUF",      "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    # Wave 1 qualification checkpoints. Keep these exact: J-lenses and calibrated dials are tied to
    # the base checkpoint, tokenizer, activation dimensions, and qualified GGUF digest.
    "qwen3.5":   ("unsloth/Qwen3.5-9B-GGUF",                  "Qwen3.5-9B-Q4_K_M.gguf"),
    "llama":     ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    "gemma4":    ("ggml-org/gemma-4-E4B-it-GGUF",             "gemma-4-E4B-it-Q4_K_M.gguf"),
    "ministral": ("mistralai/Ministral-3-3B-Instruct-2512-GGUF",
                   "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf"),
    "mistral":   ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"),
    "llama-1b":  ("bartowski/Llama-3.2-1B-Instruct-GGUF",    "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    "llama-3b":  ("bartowski/Llama-3.2-3B-Instruct-GGUF",    "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
    "gemma-2b":  ("bartowski/gemma-2-2b-it-GGUF",            "gemma-2-2b-it-Q4_K_M.gguf"),
}


# ----------------------------------------------------------------------------- discovery
#
# _model_dirs/_scan_models are imported from clozn.models.inventory above (see this module's docstring) --
# no definitions here anymore, just the aliased names every existing call site already uses.

def _flags_for(path: str) -> dict:
    base = os.path.basename(path).lower()
    for frag, _name, flags in KNOWN:
        if frag in base:
            return dict(flags)
    # Unknown GGUF: assume autoregressive instruct (the common case); engine still auto-detects mode.
    return {"chat": "instruct" in base or "chat" in base}


def _friendly(path: str) -> str:
    base = os.path.basename(path).lower()
    for frag, name, _ in KNOWN:
        if frag in base:
            return name
    return os.path.splitext(os.path.basename(path))[0]


def _quant_tag(path: str) -> str:
    """Best-effort quant label parsed off a GGUF's filename stem (e.g. 'Q2_K', 'Q4_K_M', 'Q8_0'), for the
    disambiguation note only -- falls back to the file size when no recognizable tag is present.

    The tag parse itself is clozn.models.inventory.quant_label (shared with the server's inventory route);
    this wrapper only adds the size-string fallback, which belongs here and not in the shared function --
    that fallback exists purely for this module's disambiguation PRINTOUT sitting next to a real size
    column, whereas the server's inventory reports size_bytes as its own separate field (see quant_label's
    own docstring for why inventing "quant: 4.4G" there would be a mislabel, not a derived quant)."""
    label = _quant_label(path)
    return label if label is not None else f"{os.path.getsize(path) / 1e9:.1f}G"


def _pick_best_quant(hits: list[str], arg: str) -> str:
    """>1 file on disk matches the SAME known short name -> never silently pick one (a naive first-match
    over `_scan_models()`'s alphabetically-sorted list lands on Q2_K before Q4_K_M/Q8_0 for 'qwen' --
    exactly the worst quant on disk, where this project's own quant-ladder receipts measure prose
    degrading, ~27% argmax flips vs Q8). Picks the LARGEST file instead: a reliable proxy for
    bits-per-weight across quants of the same base model, since architecture/layer-count are fixed and
    only bit-width varies file size. Always prints which file and why, so the choice is visible rather
    than silent -- pass the exact filename (or an unambiguous fragment) to get a different one."""
    if len(hits) == 1:
        return hits[0]
    best = max(hits, key=os.path.getsize)
    tags = ", ".join(f"{_quant_tag(h)}{' <- picked' if h == best else ''}" for h in hits)
    print(
        f"{fmt.DIM}- '{arg}' matched {len(hits)} files on disk ({tags}); "
        f"picked the highest-precision quant automatically. Pass the exact filename to choose a "
        f"different one.{fmt.RST}",
        file=sys.stderr,
    )
    return best


def resolve_model(arg: str) -> str:
    """A path, a known short name, or a fuzzy filename fragment -> an absolute GGUF path.

    Never a SILENT pick among multiple matches (see _pick_best_quant): the historical bug here was the
    exact-known-short-name branch returning the first alphabetical match, which for 'qwen' with
    Q2_K/Q4_K_M/Q8_0 all on disk meant the worst quant loaded with zero indication.
    """
    from clozn.cli import main as ctx
    if arg.lower().endswith(".gguf") and os.path.isfile(arg):
        return os.path.abspath(arg)
    models = _scan_models()
    if not models:
        raise ctx.CloznError("no GGUF models found. Put .gguf files in ~/.clozn/models or set CLOZN_MODELS=<dir>.")
    # exact known short-name: collect every file matching this fragment, not just the first.
    for frag, name, _ in KNOWN:
        if arg.lower() == name:
            hits = [m for m in models if frag in os.path.basename(m).lower()]
            if hits:
                return _pick_best_quant(hits, arg)
    # fuzzy: filename contains the arg
    hits = [m for m in models if arg.lower() in os.path.basename(m).lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise ctx.CloznError(f"'{arg}' is ambiguous: {', '.join(_friendly(h) for h in hits)}. Be more specific.")
    avail = ", ".join(sorted({_friendly(m) for m in models}))
    raise ctx.CloznError(f"model '{arg}' not found. Available: {avail}.")


def cmd_models(_args):
    from clozn.cli import main as ctx
    models = _scan_models()
    try:
        _, _, gpu = find_engine()
        eng = f"{fmt.BOLD}{'GPU' if gpu else 'CPU'} build{fmt.RST} found"
    except ctx.CloznError:
        eng = f"{fmt.BOLD}no engine built{fmt.RST} (run: cd engine/core && build_gpu.bat)"
    print(f"engine: {eng}")
    if not models:
        print(f"\nno models found. dirs searched: {', '.join(_model_dirs()) or '(none)'}")
        print("put .gguf files in ~/.clozn/models, or set CLOZN_MODELS=<dir>.")
        return
    print(f"\n{'NAME':<14} {'SIZE':>7}  {'KIND':<11} PATH")
    for m in models:
        size = f"{os.path.getsize(m)/1e9:.1f}G"
        flags = _flags_for(m)
        kind = "diffusion" if "mask" in flags else "autoregress"
        print(f"{_friendly(m):<14} {size:>7}  {kind:<11} {m}")
    print(f"\nrun one:  clozn run {_friendly(models[0])} \"your prompt\"")


def _rm(p):
    try:
        os.remove(p)
    except Exception:
        pass


def cmd_pull(args):
    from clozn.cli import main as ctx
    spec = args.model
    if spec in PULLABLE:
        repo, file = PULLABLE[spec]
    elif spec.endswith(".gguf") and spec.count("/") >= 2:
        parts = spec.split("/"); repo, file = "/".join(parts[:-1]), parts[-1]
    else:
        raise ctx.CloznError(f"don't know how to pull '{spec}'. Known: {', '.join(PULLABLE)}. "
                             f"Or give an explicit  owner/repo/file.gguf")
    dest_dir = os.path.join(ctx.HOME, "models"); os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, file)
    if os.path.isfile(dest):
        print(f"already have {file} ({os.path.getsize(dest) / 1e9:.1f}G)"); return
    url = f"https://huggingface.co/{repo}/resolve/main/{file}?download=true"
    print(f"{fmt.DIM}pulling{fmt.RST} {file}  {fmt.DIM}from {repo}{fmt.RST}", file=sys.stderr)
    tmp = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clozn/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0; t0 = time.time(); last = 0.0
            with open(tmp, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); done += len(b)
                    now = time.time()
                    if now - last > 0.4 or done == total:
                        last = now
                        sp = done / 1e6 / max(0.1, now - t0)
                        head = (f"{fmt._confbar(done / total)} {done / total * 100:5.1f}%  "
                                f"{done / 1e9:.2f}/{total / 1e9:.2f} GB") if total else f"{done / 1e9:.2f} GB"
                        sys.stderr.write(f"\r  {head}  {sp:4.0f} MB/s   "); sys.stderr.flush()
        sys.stderr.write("\n")
        os.replace(tmp, dest)
    except urllib.error.HTTPError as e:
        _rm(tmp)
        raise ctx.CloznError(f"{repo}/{file} not found on HuggingFace (404)." if e.code == 404
                             else f"download failed (HTTP {e.code}).")
    except Exception as e:
        _rm(tmp)
        raise ctx.CloznError(f"download failed: {e}")
    print(f"saved {_friendly(dest)} ({os.path.getsize(dest) / 1e9:.1f}G).  "
          f"run it:  clozn run {_friendly(dest)} \"hello\"")


# ------------------------------------------------------------------ plan (fit check, no download, no load)
# `clozn plan <name|path|url>` answers "will it fit?" by reading only a GGUF's header -- a few MB at the
# front of the file -- never the multi-GB tensor payload, never a model load, never the GPU. Local models
# read straight off disk; a name `clozn pull` knows but hasn't fetched yet is read remotely over HTTP Range
# against HuggingFace, so the fit check happens BEFORE the download it's trying to save you from.

def _fmt_ctx(n) -> str:
    if not n:
        return "?"
    return f"{n // 1024}k" if n % 1024 == 0 else str(n)


def _detect_vram_gb():
    """Best-effort local VRAM budget via `nvidia-smi` -- a driver metadata query, not a CUDA context: it
    doesn't allocate anything or run compute, so it's consistent with this being a CPU-only, no-GPU-use
    planner. Returns None (caller falls back to a default) if nvidia-smi isn't there or times out."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return round(float(out.stdout.strip().splitlines()[0]) / 1024, 1)
    except Exception:
        pass
    return None


def format_plan(name: str, header: dict, file_size_bytes: int, report: dict, vram_gb: float,
                ctx_for_estimate: int = 8192, source_note: str = "") -> str:
    """Pure dict(s)->text render of a fit_planner report (no I/O -- testable with canned dicts), e.g.:
    'Qwen2.5 7B Instruct Q4_K_M -- 28 layers, 32k ctx, 4.4 GB file
     on 16 GB VRAM: FITS (~est 5.3 GB at 8k ctx, KV-q8)'"""
    quant = header.get("quant", "?")
    n_layers = header.get("n_layers")
    size_gb = (file_size_bytes or 0) / 1e9
    verdict = f"{fmt.BOLD}FITS{fmt.RST}" if report["fits"] else f"{fmt.BOLD}WON'T FIT{fmt.RST}"
    lines = [
        f"{name} {quant} — {n_layers if n_layers is not None else '?'} layers, "
        f"{_fmt_ctx(header.get('context_length'))} ctx, {size_gb:.1f} GB file"
        + (f"  {fmt.DIM}{source_note}{fmt.RST}" if source_note else ""),
        f"on {vram_gb:g} GB VRAM: {verdict}  (~est {report['est_vram_gb']:.1f} GB at "
        f"{_fmt_ctx(ctx_for_estimate)} ctx, KV-q8)",
    ]
    if report.get("offload_hint"):
        lines.append(f"  {report['offload_hint']}")
    lines.append(f"{fmt.DIM}{report['note']}{fmt.RST}")
    if header.get("quant_source") and header["quant_source"] != "general.file_type":
        lines.append(f"{fmt.DIM}quant is a guess from the dominant tensor type ({header['quant_source']}).{fmt.RST}")
    return "\n".join(lines)


# --------------------------------------------------------------------- throughput predictor (model-free)
# `clozn plan` also renders a decode-tok/s ROOFLINE: physics from throughput_predictor.py (memory-bandwidth
# bound decode -- see that module's docstring), rendered here the same way format_plan renders fit_report.

def format_throughput(est: dict) -> str:
    """Pure dict->text render of a throughput_predictor.predict_throughput() estimate (no I/O -- testable
    with canned dicts). Appended under format_plan's fit-check output in `clozn plan`."""
    n_params = est.get("n_params") or 0
    weight_bytes = est.get("weight_bytes") or 0.0
    total_bytes = est.get("total_bytes_per_token") or 0.0
    if not n_params or not total_bytes:
        return (f"{fmt.DIM}predicted decode throughput: unavailable -- header didn't carry enough "
                f"tensor/shape info to estimate.{fmt.RST}")

    weight_gb = weight_bytes / 1e9
    kv_mb = (est.get("kv_bytes_per_token") or 0.0) / 1e6
    total_mb = total_bytes / 1e6
    bw = est["bandwidth_gb_s"]
    tok_s = est["predicted_tok_s"]
    bpw_avg = weight_bytes * 8 / n_params

    lines = [
        f"predicted decode throughput: {fmt.BOLD}~{tok_s:.0f} tok/s{fmt.RST}  "
        f"{fmt.DIM}(roofline @ {bw:g} GB/s assumed effective bandwidth){fmt.RST}",
        f"  weight bytes/token: {weight_gb:.2f} GB  ({n_params / 1e9:.2f}B params @ ~{bpw_avg:.2f} bpw, "
        f"exact from the GGUF's tensor shapes/types -- not the file size)",
        f"  KV bytes/token:     {kv_mb:.1f} MB  (at {_fmt_ctx(est['ctx_for_estimate'])} ctx, "
        f"{est['kv_bytes_per_element']:g}-byte/elem cache; grows as context fills -- this is the worst case "
        f"at that context depth, not a constant)",
        f"  total: {total_mb:.1f} MB/token  ->  {bw:g} GB/s / {total_mb:.1f} MB = ~{tok_s:.0f} tok/s",
        f"{fmt.DIM}ROOFLINE, not a promise: this is a memory-bandwidth-bound estimate that ignores "
        f"compute-bound prefill, batching, and real kernel efficiency (hardware rarely sustains 100% of "
        f"theoretical bandwidth) -- it predicts the ORDER of tok/s, not the exact number. Calibrate "
        f"against a live run for the real one (`--calibrate`, not yet wired -- see cmd_plan).{fmt.RST}",
    ]
    if est.get("unknown_params"):
        n_tensors = sum(est["unknown_type_tensor_counts"].values())
        lines.append(f"{fmt.DIM}note: {est['unknown_params']:,} params across {n_tensors} tensor(s) of "
                     f"unrecognized ggml_type were excluded from the weight-byte total (extend "
                     f"BPW_BY_GGML_TYPE in throughput_predictor.py).{fmt.RST}")
    return "\n".join(lines)


def cmd_plan(args):
    from clozn.cli import fit_planner            # stdlib+urllib only; imported lazily like other clozn.* siblings
    from clozn.cli import throughput_predictor as tp   # stdlib-only, model-free (see its module docstring)
    from clozn.cli import main as ctx

    vram_gb = args.vram if args.vram is not None else (_detect_vram_gb() or 16.0)
    spec = args.model

    # gguf_header_from_path/url raise ValueError ("not a GGUF file: ...") on a malformed header, or its
    # subclass NeedMoreBytes on a truncated one (the header didn't fit even at max_bytes) -- both are
    # facts about the FILE, not a bug in this command, so they get the same clean one-line CloznError
    # exit every other bad-input path here already uses; main() only catches CloznError, so anything
    # else here would otherwise surface as a raw traceback.
    try:
        if spec.startswith("http://") or spec.startswith("https://"):
            header = fit_planner.gguf_header_from_url(spec)
            size = header.get("file_size_bytes") or 0
            name = header.get("name") or spec.rsplit("/", 1)[-1]
            source_note = f"remote, not downloaded: {spec}"
        else:
            path = spec if (spec.lower().endswith(".gguf") and os.path.isfile(spec)) else None
            if path is None:
                try:
                    path = resolve_model(spec)
                except ctx.CloznError:
                    if spec not in PULLABLE:
                        raise
                    repo, file = PULLABLE[spec]
                    url = f"https://huggingface.co/{repo}/resolve/main/{file}"
                    print(f"{fmt.DIM}- '{spec}' isn't downloaded yet -- reading its header straight off "
                         f"HuggingFace (no download){fmt.RST}", file=sys.stderr)
                    header = fit_planner.gguf_header_from_url(url)
                    size = header.get("file_size_bytes") or 0
                    name = spec
                    source_note = f"not downloaded -- `clozn pull {spec}` fetches it: {url}"
                    path = None
            if path is not None:
                header = fit_planner.gguf_header_from_path(path)
                size = header["file_size_bytes"]
                name = _friendly(path)
                source_note = path
    except ValueError as e:
        raise ctx.CloznError(f"couldn't read '{spec}' as a GGUF: {e}")

    report = fit_planner.fit_report(header, size, vram_gb)
    print(format_plan(name, header, size, report, vram_gb, source_note=source_note))

    bandwidth_gb_s = args.bandwidth_gb_s if getattr(args, "bandwidth_gb_s", None) is not None \
        else tp.DEFAULT_BANDWIDTH_GB_S
    est = tp.predict_throughput(header, bandwidth_gb_s=bandwidth_gb_s)
    print()
    print(format_throughput(est))

    if getattr(args, "calibrate", False):
        _cmd_plan_calibrate_stub(name)


def _cmd_plan_calibrate_stub(name: str):
    """DEFERRED seam for `clozn plan --calibrate`: boot the engine on this GGUF, run a short fixed-length
    decode, measure the ACTUAL tok/s, and solve for this machine's real effective bandwidth
    (bandwidth_gb_s = measured_tok_s * total_bytes_per_token / 1e9) so future `clozn plan` calls on this
    machine can default to a calibrated constant instead of the generic RTX-5080-class guess.

    Deliberately NOT implemented in this pass: the engine is being rebuilt in a concurrent workstream
    (see docs/ROADMAP.md), and this predictor must stay pure-CPU / model-free until that lands. TODO
    (next pass, once the engine is free):
      1. resolve_model(name) -> path, launch the engine the same way `clozn run` does (engine_process.py)
      2. run a short, fixed-length decode (e.g. 64-128 tokens, greedy, no --heat) and time it
      3. measured_tok_s = tokens_generated / wall_clock_seconds
      4. back out bandwidth_gb_s from measured_tok_s and this run's own total_bytes_per_token, then
         report BOTH the roofline prediction and the calibrated number side by side
      5. optionally cache the calibrated bandwidth_gb_s under ~/.clozn/config.json for future `plan` runs
    """
    print()
    print(f"{fmt.BOLD}--calibrate: DEFERRED{fmt.RST} (not implemented in this pass)")
    print(f"{fmt.DIM}this would boot the engine on {name}, run a short live decode, measure the ACTUAL "
          f"tok/s, and use it to correct the assumed bandwidth above for this machine. Left unwired "
          f"deliberately -- the engine is being rebuilt in a concurrent workstream; see the TODO in "
          f"cmd_plan._cmd_plan_calibrate_stub() for the plan once it's free.{fmt.RST}")
