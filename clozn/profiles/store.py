"""profiles.py -- PORTABLE PERSONA BUNDLES: named profiles ("friend", "work") as SOURCE bundles.

The design law (from the portability discussion + don't-fuse): store SOURCES, not compiled forms.
A profile is pure text/JSON -- card texts, dial settings, custom-dial recipes (pole descriptions),
fact (cue, answer) pairs -- so it is:
  * PORTABLE across model upgrades: vectors are a cache; recompile on any model (keys/values/dials
    re-derive deterministically from the sources -- the GPT-2 -> Qwen slotmem port proved the move).
  * INSTANT to switch (no retrain): dials apply immediately; cards compile to a prompt block; facts
    recompile into a slot store in seconds.
  * ISOLATED by construction: each persona's facts/cards live in its own bundle -- work never bleeds
    into friend.

This module is deliberately MODEL-FREE (stdlib only): CRUD + export/import + the compile helpers,
which take live objects (a SteeringControl-like, a SlotMem-like) by duck type. Wiring into the studio
server/UI is a later step; the format is the product.

Bundle layout (~/.clozn/profiles/<name>.json), version-tagged for future migration:
{
  "version": 1, "name": "work", "description": "...",
  "cards":        [{"text": "...", "status": "active"}, ...],       # dispositions (say-it tier)
  "dials":        {"concise": 0.8, "warm": -0.2},                    # built-in dial settings
  "response_policies": ["less-verbose"],                              # prompt-first retry defaults
  "custom_dials": [{"name","pos","neg","max"}, ...],                 # show-it recipes (recompilable)
  "facts":        [{"cue": "...", "answer": " ..."}, ...],           # slot-store sources (recompilable)
  "created_at": ..., "updated_at": ...
}
"""
from __future__ import annotations

import json
import os
import re
import time

from clozn._io import atomic_write_json

VERSION = 1
DEFAULT_DIR = os.path.join(os.path.expanduser("~"), ".clozn", "profiles")

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _now() -> float:
    return time.time()


def new_profile(name: str, description: str = "") -> dict:
    """A fresh, empty bundle. Names are slug-safe (they become filenames)."""
    if not _NAME_RE.match(name or ""):
        raise ValueError(f"profile name must match {_NAME_RE.pattern!r}, got {name!r}")
    return {"version": VERSION, "name": name, "description": description,
            "cards": [], "dials": {}, "response_policies": [], "custom_dials": [], "facts": [],
            "created_at": _now(), "updated_at": _now()}


def validate(p: dict) -> dict:
    """Shape-check (and lightly normalize) a bundle; raises ValueError on junk. Returns the bundle.

    Bundles are hand-editable/importable (that's the whole point -- see the module docstring), so every
    field below is guarded the same way `cards` already was: a wrong-shaped container degrades to empty,
    and a wrong-shaped/incomplete ENTRY within it is silently skipped rather than raising an uncaught
    AttributeError/KeyError/TypeError out of a plain dict/list comprehension (round-2 pressure test #2)."""
    if not isinstance(p, dict) or not _NAME_RE.match(str(p.get("name") or "")):
        raise ValueError("profile needs a slug-safe 'name'")
    if int(p.get("version", 1)) > VERSION:
        raise ValueError(f"profile version {p.get('version')} is newer than supported {VERSION}")
    p.setdefault("version", VERSION)
    p.setdefault("description", "")
    cards_in = p.get("cards")
    cards_in = cards_in if isinstance(cards_in, list) else []   # a non-list container degrades to empty too
    # Cards are CARRIED, never applied: memory cards were cut from the product on 2026-07-27, and a
    # bundle keeps them only so an export/import round-trip stays lossless. The global-vs-app/project
    # scope filter that used to run here went with the card store -- there is no longer an overlay tier
    # for a card to belong to, so every recorded card is preserved verbatim.
    p["cards"] = [{"text": str(c.get("text", c) if isinstance(c, dict) else c),
                   "status": str(c.get("status", "active")) if isinstance(c, dict) else "active"}
                  for c in cards_in if (c.get("text") if isinstance(c, dict) else c)]

    dials_in = p.get("dials")
    dials_out = {}
    if isinstance(dials_in, dict):
        for k, v in dials_in.items():
            try:
                dials_out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue                       # a non-numeric dial value is dropped, not fatal
    p["dials"] = dials_out

    from clozn.replay.corrective import CORRECTION_PRESETS
    policies_in = p.get("response_policies")
    policies_in = policies_in if isinstance(policies_in, list) else []
    p["response_policies"] = list(dict.fromkeys(
        str(value) for value in policies_in if str(value) in CORRECTION_PRESETS
    ))

    custom_in = p.get("custom_dials")
    custom_in = custom_in if isinstance(custom_in, list) else []
    custom_out = []
    for d in custom_in:
        if not isinstance(d, dict) or not d.get("name") or not d.get("pos") or not d.get("neg"):
            continue                           # not a dict, or missing a required recipe field -> skip
        try:
            custom_out.append({"name": str(d["name"]), "pos": str(d["pos"]), "neg": str(d["neg"]),
                               "max": float(d.get("max", 0.5))})
        except (TypeError, ValueError):
            continue
    p["custom_dials"] = custom_out

    facts_in = p.get("facts")
    facts_in = facts_in if isinstance(facts_in, list) else []
    facts_out = []
    for f in facts_in:
        if not isinstance(f, dict) or not f.get("cue") or not f.get("answer"):
            continue                           # not a dict, or missing cue/answer -> skip
        facts_out.append({"cue": str(f["cue"]), "answer": str(f["answer"])})
    p["facts"] = facts_out

    p.setdefault("created_at", _now())
    p["updated_at"] = _now()
    return p


class ProfileStore:
    """A directory of profile bundles. save/load/list/delete/export/import -- all plain JSON files."""

    def __init__(self, root: str = DEFAULT_DIR):
        self.root = root

    def _path(self, name: str) -> str:
        # Validate at the choke point: save() goes through validate() but load()/switch/export did NOT,
        # so an unguarded `name` like "../config" traversed to ~/.clozn/*.json. Every caller route already
        # catches ValueError, so a bad name degrades to a clean 400/404 rather than a file-system escape.
        if not _NAME_RE.match(name or ""):
            raise ValueError(f"profile name must match {_NAME_RE.pattern!r}, got {name!r}")
        return os.path.join(self.root, name + ".json")

    def save(self, p: dict) -> str:
        """Persist a bundle. Atomic (see clozn._io): validate() runs first (so a junk bundle raises
        ValueError before any file touch), and the write itself is temp-file-then-rename, so a failure
        partway through a save can never truncate/corrupt an existing bundle already at this path."""
        p = validate(dict(p))
        path = self._path(p["name"])
        atomic_write_json(path, p, indent=2, ensure_ascii=False)
        return path

    def load(self, name: str) -> dict:
        with open(self._path(name), encoding="utf-8") as f:
            return validate(json.load(f))

    def list(self) -> list[dict]:
        if not os.path.isdir(self.root):
            return []
        out = []
        for fn in sorted(os.listdir(self.root)):
            if fn.endswith(".json"):
                try:
                    out.append(self.load(fn[:-5]))
                except Exception:
                    continue                      # a corrupt bundle never breaks the listing
        return out

    def delete(self, name: str) -> bool:
        try:
            os.remove(self._path(name))
            return True
        except OSError:
            return False

    # export/import are just save/load at an arbitrary path -- the bundle IS the portable artifact.
    def export(self, name: str, dest: str) -> str:
        p = self.load(name)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(p, f, indent=2, ensure_ascii=False)
        return dest

    def import_(self, src: str, rename: str | None = None) -> dict:
        with open(src, encoding="utf-8") as f:
            p = validate(json.load(f))
        if rename:
            p["name"] = rename
            p = validate(p)
        self.save(p)
        return p


# ------------------------------------------------------------------ compile: sources -> live model
# Each compile step is cheap and re-runnable on ANY model -- that's the portability contract.

def prompt_block(p: dict) -> str:
    """Cards -> the system block (same wording the prefix was trained to imitate -- see
    Active cards only; empty string when there are none (callers omit the block entirely)."""
    texts = [c["text"] for c in p.get("cards", []) if c.get("status", "active") == "active"]
    if not texts:
        return ""
    return ("You are a helpful assistant talking with a returning user. Here is what you know "
            "about them; use it naturally to tailor how you respond:\n"
            + "\n".join("- " + t for t in texts))


def apply_dials(p: dict, steer) -> dict:
    """Set the profile's dial values on a live SteeringControl-like (duck-typed: .set/.clear, and
    .add_custom for recipes not already computed). Returns {applied:{...}, customs_added:[...]}.
    Clears existing dials first -- switching personas REPLACES the tone, never blends it."""
    steer.clear()
    added = []
    have = set(getattr(steer, "vecs", {}) or {})
    for d in p.get("custom_dials", []):
        if d["name"] not in have and hasattr(steer, "add_custom"):
            steer.add_custom(d["name"], d["pos"], d["neg"], d.get("max", 0.5))
            added.append(d["name"])
    for name, val in p.get("dials", {}).items():
        steer.set(name, val)
    return {"applied": dict(p.get("dials", {})), "customs_added": added}


def compile_facts(p: dict, slotmem, gate: bool = False) -> dict:
    """Recompile the profile's facts into a slot store on the CURRENT model (duck-typed: .write).
    Fresh entries assumed (caller clears or supplies an empty store -- persona isolation means one
    store per profile). gate=False by default: a profile's facts were curated; store them all."""
    written = skipped = 0
    for f in p.get("facts", []):
        r = slotmem.write(f["cue"], f["answer"], gate=gate)
        if isinstance(r, dict) and r.get("written") is False:
            skipped += 1
        else:
            written += 1
    if hasattr(slotmem, "calibrate_gate"):
        slotmem.calibrate_gate()                 # per-model recalibration is part of the port
    return {"written": written, "skipped": skipped}


def switch(p: dict, steer=None, slotmem=None) -> dict:
    """Apply a whole profile to a live substrate: dials (instant) + facts (seconds) + the prompt
    block (returned for the caller's chat path). The one-call persona switch."""
    out = {"name": p["name"], "prompt_block": prompt_block(p)}
    if steer is not None:
        out["dials"] = apply_dials(p, steer)
    if slotmem is not None:
        out["facts"] = compile_facts(p, slotmem)
    return out
