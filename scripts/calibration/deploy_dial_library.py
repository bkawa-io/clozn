"""deploy_dial_library.py -- register the 27 library-only dials as live studio custom dials, ONCE.

research/dial_library_shipped.json is the human-curated, per-model-calibrated 33-dial tone library (see
that file's own "_about"/"_provenance"/"_deploy" fields). ~/.clozn/dial_calibration.json (deployed
separately, NOT touched by this script) already caps every one of the 33 by name -- both the 6 that
happen to share a name with a steering.AXES built-in (warm, playful, formal, concise, poetic, concrete;
already live, already calibrated) and the 27 that don't exist as a dial ANYWHERE yet. This script's only
job is that second half: REGISTER the 27 library-only dials on the live studio substrate as custom dials
(SteeringControl.add_custom -- the diff-of-means direction, a few forward passes per dial over
steering.SEED_PROMPTS, hence GPU / the loaded 7B) so they actually show up as sliders.

WHY CUSTOM, NOT A NEW BUILT-IN: add_custom is exactly the mechanism SteeringControl already exposes for a
dial that isn't a static AXES entry (identical recipe: mean(+pole) - mean(-pole) over the seeds -> a unit
direction; see steering.SteeringControl.add_custom, untouched by this script). The one thing a SHIPPED
dial must not do is read as a user's OWN "make your own dial" creation on the Behavior page (that UI
already tags every steer.custom entry "yours" + gives it a delete button). So this script persists the 27
to their OWN file, ~/.clozn/studio_library.json -- separate from studio_custom_<name>.json (the user's
file, never written by this script) -- and clozn_server.py's QwenSubstrate boot now ALSO loads that file
(steer.load_custom), so the library survives a studio restart exactly like a user custom does.
clozn_server.py's /steer/axes reads studio_library.json's KEYS (_library_dial_names) to flag those
entries "library": true instead of "custom": true -- see that module for the read side; this script only
ever WRITES studio_library.json, never the steering source itself (whose add_custom/save_custom/load_custom/
custom-dict shape are all unmodified -- this script is just another caller of that existing recipe).

IDEMPOTENT: a name already present in ~/.clozn/studio_library.json (i.e. a previous run of this exact
script already registered it) is skipped -- no wasted GPU compute recomputing an unchanged direction.
Reruns after a partial run (killed partway through), or after dial_library_shipped.json gains new
entries beyond today's 33, only compute what's missing. A library name that collides with an EXISTING
USER custom dial (same name, already in ~/.clozn/studio_custom_qwen.json from the Behavior page's "make
your own dial" panel) is SKIPPED WITH A WARNING, never silently overwritten -- pass --force to override
(this replaces that dial's direction under the shared name; the user's original recipe is gone).

NEEDS THE LOADED MODEL/GPU (add_custom runs real forward passes) -- this constructs a full QwenSubstrate
exactly like clozn_server.py's own boot does (same load7b() nf4 backbone, same steering.SteeringControl,
same studio_custom_qwen.json / studio_library.json loads), so run it in the same environment the studio
itself runs in:

    clozn .venv python research/deploy_dial_library.py

--check (no model, no GPU, no ~/.clozn writes -- reads only the two small JSON files this script cares
about) prints the plan: which of the 27 are already deployed, which would be (re)computed, and any name
collisions with a user's own custom dial -- so the plan can be sanity-checked before paying the GPU cost.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root -- clozn_server/steering moved into the clozn/ package

from clozn.server import app as cs      # noqa: E402  -- cheap: no model load at import time (mirrors test_dial_*.py)
from clozn.lab.substrates import QwenSubstrate   # noqa: E402  -- lab adapter (relocated); import is torch-free, .name/() only
from clozn.behavior.steering.axes import AXES      # noqa: E402  -- AXES-only; no CUDA/model needed

SHIPPED_LIBRARY_PATH = os.path.join(HERE, "..", "..", "clozn", "data", "dial_library_shipped.json")


def load_shipped_library(path: str = SHIPPED_LIBRARY_PATH) -> list[dict]:
    """The 33 curated dials -- {"dials": [{name, category, pos, neg, ship_range, note}, ...]}. Raises
    loudly (never caught here) on a missing/malformed file: this is a one-shot deploy script, not a
    request-serving path, so a bad shipped file should stop the run rather than silently ship nothing."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dials = data.get("dials") if isinstance(data, dict) else None
    if not isinstance(dials, list) or not dials:
        raise ValueError(f"{path}: expected a top-level {{'dials': [...]}} non-empty list")
    return dials


def library_only_dials(dials: list[dict]) -> list[dict]:
    """The subset of `dials` NOT already a steering.AXES built-in name -- the 27 of 33 this script
    registers. Computed from AXES itself (never a hardcoded count), so this stays correct if AXES or the
    shipped library ever changes. The 6 that DO overlap (warm/playful/formal/concise/poetic/concrete) are
    already live built-ins, already capped by ~/.clozn/dial_calibration.json -- re-registering them as
    customs would create a second, redundant definition under the same name, so this script leaves them
    alone entirely (they are already "deployed" in every sense that matters)."""
    return [d for d in dials if d["name"] not in AXES]


def already_deployed_names() -> set:
    """Names a PREVIOUS run of this script already persisted, read straight off the KEYS of
    ~/.clozn/studio_library.json -- no model needed, so --check can answer this with nothing loaded.
    Missing/broken file -> empty set (a fresh install, or nothing deployed yet)."""
    path = cs._pers("studio_library.json")
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, dict) else set()
    except Exception:
        return set()


def existing_user_custom_names() -> set:
    """Names already in the USER's own custom-dial file (~/.clozn/studio_custom_qwen.json -- written by
    the Behavior page's "make your own dial" panel / POST /steer/custom), read directly with no model
    needed. Used only to detect a name COLLISION before deploying a library dial of the same name -- this
    script never writes to this file. Missing/broken file -> empty set."""
    path = cs._pers(f"studio_custom_{QwenSubstrate.name}.json")
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, dict) else set()
    except Exception:
        return set()


def plan(to_register: list[dict]):
    """Partition `to_register` (the 27 library-only dial specs) into (to_add, skip_deployed,
    skip_collision) -- PURE, no model/GPU: reads only the two small JSON files on disk. Shared by --check
    (prints this and stops) and deploy() (acts on it), so the preview and the real run can never disagree
    about what would happen."""
    deployed = already_deployed_names()
    user_names = existing_user_custom_names()
    to_add, skip_deployed, skip_collision = [], [], []
    for d in to_register:
        name = d["name"]
        if name in deployed:
            skip_deployed.append(name)
        elif name in user_names:
            skip_collision.append(name)
        else:
            to_add.append(d)
    return to_add, skip_deployed, skip_collision


def _print_plan(to_add, skip_deployed, skip_collision):
    print(f"[plan] {len(to_add)} to register, {len(skip_deployed)} already deployed, "
          f"{len(skip_collision)} collide with an existing user dial", flush=True)
    if to_add:
        print("  register: " + ", ".join(d["name"] for d in to_add), flush=True)
    if skip_deployed:
        print("  already deployed (skip): " + ", ".join(skip_deployed), flush=True)
    if skip_collision:
        print("  COLLIDES with a user custom (skip; rerun with --force to overwrite): "
              + ", ".join(skip_collision), flush=True)


def deploy(force: bool = False) -> dict:
    """Boot the real QwenSubstrate (loads the nf4 7B -- the GPU cost), register every not-yet-deployed
    library-only dial via steer.add_custom, and persist the full up-to-date set to
    ~/.clozn/studio_library.json. Returns a receipt: {"added", "skipped_deployed", "skipped_collision",
    "total_library_dials"} -- never just printed and discarded."""
    dials = load_shipped_library()
    to_register = library_only_dials(dials)
    to_add, skip_deployed, skip_collision = plan(to_register)
    if force and skip_collision:
        by_name = {d["name"]: d for d in to_register}
        to_add = to_add + [by_name[n] for n in skip_collision]
        skip_collision = []
    _print_plan(to_add, skip_deployed, skip_collision)

    print("[deploy] booting the studio substrate (loads the nf4 7B -- may take a minute)...", flush=True)
    sub = QwenSubstrate()      # __init__ already loads studio_custom_qwen.json + studio_library.json
    sub._ensure_steer()           # calibrate steer.base/resid_norm before any add_custom call (matches
                                   # how the live /steer/custom endpoint always runs _ensure_steer first)

    for d in to_add:
        print(f"[deploy] registering '{d['name']}' ({d.get('category', '?')})...", flush=True)
        sub.steer.add_custom(d["name"], d["pos"], d["neg"], float(d["ship_range"][1]))

    # Persist the FULL up-to-date library set (freshly added + already-deployed from a prior run) to its
    # OWN file -- never via steer.save_custom() (that dumps the WHOLE steer.custom dict, user dials
    # included); this writes ONLY genuinely-library names, so the file stays the authoritative "these are
    # shipped-library" set for both the next boot's load (QwenSubstrate.__init__) and /steer/axes's
    # "library" flag (clozn_server._library_dial_names).
    payload = {}
    for d in to_register:
        name = d["name"]
        if name in skip_collision:
            continue                          # a real user dial of this name -- never claimed as "library"
        info = sub.steer.custom.get(name)
        if info is None:
            continue                          # defensive; shouldn't happen for anything in to_add/deployed
        payload[name] = {"pos": info["pos"], "neg": info["neg"], "max": info["max"],
                         "source": "library", "category": d.get("category", "?")}
    lib_path = cs._pers("studio_library.json")
    os.makedirs(os.path.dirname(lib_path) or ".", exist_ok=True)
    with open(lib_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[deploy] done -- {len(payload)} library dial(s) live, persisted to {lib_path}", flush=True)

    return {"added": [d["name"] for d in to_add], "skipped_deployed": skip_deployed,
            "skipped_collision": skip_collision, "total_library_dials": len(payload)}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Register the 27 library-only dials from dial_library_shipped.json as live studio "
                    "custom dials, persisted to ~/.clozn/studio_library.json (kept separate from the "
                    "user's own studio_custom_<name>.json). Needs the loaded 7B -- run on the GPU box.")
    ap.add_argument("--check", action="store_true",
                    help="print the plan only (no model, no GPU, no ~/.clozn writes) and exit")
    ap.add_argument("--force", action="store_true",
                    help="also overwrite a name that collides with an existing USER custom dial "
                         "(use with care -- replaces that dial's direction under the shared name)")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    dials = load_shipped_library()
    to_register = library_only_dials(dials)
    if args.check:
        overlap = sorted(set(d["name"] for d in dials) & set(AXES))
        print(f"[check] {len(dials)} shipped dials, {len(to_register)} library-only "
              f"({len(overlap)} already steering.AXES built-ins: {', '.join(overlap)}).", flush=True)
        _print_plan(*plan(to_register))
        print("[check] no model loaded, nothing written. Run without --check on the GPU box to deploy.",
              flush=True)
        return None
    return deploy(force=args.force)


if __name__ == "__main__":
    main()
