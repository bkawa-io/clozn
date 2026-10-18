"""commands.models -- model discovery (`clozn models`) and fetch (`clozn pull`).

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
