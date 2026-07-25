"""clozn.models.inventory -- local GGUF discovery, shared by the CLI (`clozn models`, resolve_model in
clozn.cli.commands.models) and the server's read-only inventory route (GET /models/local,
clozn.server.routes.models).

WHY THIS IS ITS OWN TOP-LEVEL PACKAGE, NOT clozn.cli.commands.models
----------------------------------------------------------------------------------------------
clozn/server/ imports NOTHING from clozn/cli/ (verified before this module was added -- grep the tree).
That's deliberate layering, not an accident: the server is the product-serving process and must not drag
in CLI-only machinery (argument parsing, CloznError's console-facing messages, engine-launch subprocess
code, ...). Before this module existed, `_scan_models`/`_model_dirs` lived only in
clozn.cli.commands.models, so a server route wanting the same listing would have had to either import that
CLI module (breaking the layering) or silently reimplement (and inevitably drift from) the directory
search. This module is the extraction point: both sides import the SAME functions from here.

HOME/REPO/ENGINE_CORE are re-derived here as PURE path constants (os.path.expanduser / __file__-relative),
not imported from clozn.cli.main.HOME / clozn.cli.engine_process.REPO,ENGINE_CORE -- importing those would
recreate exactly the clozn.cli dependency this extraction exists to avoid. They resolve to the identical
strings on the same machine (same expressions), so clozn.cli.commands.models.resolve_model's observable
behavior is unchanged now that it calls through to model_dirs()/scan_models() here instead of its own
former private copies.

`inventory()`'s sha256 field is populated ONLY from clozn.runs.identity's persistent hash cache -- see
cached_model_sha256's own docstring for why a GET route must never hash a multi-GB file inline. This
mirrors clozn.runs.identity's own "mandatory cache, cheap lookup" discipline (that module's own docstring
section "WHY THE HOT PATH NEVER PAYS FOR THIS").
"""
from __future__ import annotations

import glob
import json
import os
import re

HOME = os.path.expanduser("~/.clozn")
# repo root: clozn/models/inventory.py -> clozn/models -> clozn -> repo root (3 dirname calls), the same
# depth-from-root clozn.cli.engine_process.REPO computes from clozn/cli/engine_process.py.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_CORE = os.path.join(REPO, "engine", "core")

_QUANT_TAG_RE = re.compile(r"(?:^|[.\-_])((?:IQ|Q|F|BF)\d[A-Z0-9_]*)$", re.IGNORECASE)


def model_dirs() -> list[str]:
    """Every directory to search for local GGUFs, in priority order, de-duplicated: CLOZN_MODELS (a
    PATH-separated env var), config.json's "model_dirs" list, then ~/.clozn/models, <repo>/models, and the
    engine build's own models/ dir. Mirrors clozn.cli.commands.models's former private _model_dirs()
    exactly (that function now delegates here). A missing/unreadable config.json degrades to just the
    env-var + fallback dirs, never raises."""
    dirs = []
    if os.environ.get("CLOZN_MODELS"):
        dirs += os.environ["CLOZN_MODELS"].split(os.pathsep)
    cfg = os.path.join(HOME, "config.json")
    if os.path.isfile(cfg):
        try:
            with open(cfg, encoding="utf-8") as f:
                dirs += json.load(f).get("model_dirs", [])
        except Exception:
            pass
    dirs += [os.path.join(HOME, "models"), os.path.join(REPO, "models"), os.path.join(ENGINE_CORE, "models")]
    seen, out = set(), []
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d))
        if d not in seen and os.path.isdir(d):
            seen.add(d)
            out.append(d)
    return out


def scan_models() -> list[str]:
    """Every *.gguf file across model_dirs(), as absolute paths, sorted + de-duplicated. Mirrors
    clozn.cli.commands.models's former private _scan_models() exactly."""
    found = []
    for d in model_dirs():
        found += glob.glob(os.path.join(d, "*.gguf"))
    return sorted(set(found))


def quant_label(path: str) -> str | None:
    """Best-effort quant tag parsed off a GGUF's filename stem (e.g. 'Q4_K_M', 'Q8_0', 'F16'), or None
    when the name carries no recognizable tag -- never a fabricated guess. clozn.cli.commands.models's
    `_quant_tag` wraps this with its own size-string fallback for its disambiguation printout (where no
    separate size column exists); the inventory route below reports size_bytes as its own field, so a
    "quant" of "4.4G" would be a mislabeled size, not a derived quant -- this function stays honestly
    nullable instead."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _QUANT_TAG_RE.search(stem)
    return m.group(1).upper() if m else None


def inventory() -> list[dict]:
    """The local GGUF inventory as plain JSON-safe dicts: {path, filename, size_bytes, quant, sha256}.
    `quant` is quant_label()'s best-effort tag, or None when the filename carries no recognizable one.
    `sha256` is populated ONLY from clozn.runs.identity.cached_model_sha256 -- None whenever this exact
    file version hasn't been hashed by some other process yet (engine boot, a CLI command, ...); this
    function itself never hashes a file, so a request for this listing is always fast regardless of how
    many multi-GB GGUFs are on disk. A file that vanishes between scan_models() and the size read (a race
    with a delete/eject) is skipped rather than raising."""
    from clozn.runs.identity import cached_model_sha256

    out = []
    for path in scan_models():
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        out.append({
            "path": path,
            "filename": os.path.basename(path),
            "size_bytes": size,
            "quant": quant_label(path),
            "sha256": cached_model_sha256(path),
        })
    return out
