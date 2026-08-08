"""Native engine activation steering adapter."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from clozn._io import atomic_write_json

from . import axes
from . import jlens_transport

# Measured cliff (2026-07-24 usage-testing sweep, `concise` dial, temp 0, same prompt): 0.0/0.2/0.4/0.6
# all render clean prose (596/829/405/360 chars, max repeated 6-gram = 1); 0.8 renders a repetition
# loop (max repeated 6-gram = 8), broken capitalization, AND factual corruption (baseline
# "syn-propanethial-S-oxide" became "sulfonic acid") -- A1.12's repetition attractor reaching the
# shipped path. Deliberately NOT clamped here (that's a product-behavior call the user hasn't made) --
# see EngineSteer.set()'s warning instead.
UNSAFE_STRENGTH = 0.6


class EngineSteer:
    """Tone dials on the native engine using harvested residual directions.

    J-TRANSPORT (optional, off by default -- see jlens_transport.py / notes/JLENS_SAE_FINDINGS.md
    finding #1): these diff-of-means tone directions are built purely from harvested activations
    and, unlike concept_dir.py's dir(c), never touch J at all. `j_transport=True` (or a later
    `enable_j_transport()` call) turns on an explicit extra step, applied to the FINAL summed
    vector right before it is returned to the caller (steer_vector()) or sent to /intervene
    (generate()): J-transport it via jlens_transport.transport_direction, preserving this vector's
    own calibrated norm (norm="preserve" -- unlike dir(c)'s unit convention, THIS vector's
    magnitude already IS the injection strength, sent at coef=1.0). Whether it actually happened
    is never assumed -- see self.last_j_transport (recorded after every steer_vector()/generate()
    call) and the module docstring's HONESTY CONTRACT: a missing/wrong-model/wrong-shape J
    degrades to the ORIGINAL vector unchanged, `last_j_transport["applied"] is False`, never a
    silent substitution."""

    def __init__(self, engine_client, layer=None, *, j_transport=False, jlens_dir=None,
                 jlens_model_id=None, jlens_model_sha256=None, jlens_artifact_root=None,
                 jlens_k=None):
        self.ec = engine_client
        self.layer = layer
        self.vecs = {}
        self.custom = {}
        self.strength = {}
        # This exact model's dial calibration, installed by load_calibration(). EMPTY IS THE SAFE
        # DEFAULT and is what an unswept model keeps: ceiling_for() then enforces UNSAFE_STRENGTH
        # instead of the axis's declared max. Nothing here is inferred from another model -- a
        # calibration is only ever valid for the exact GGUF it was swept on.
        self.calibration: dict = {}
        self.base, self.resid_norm = 1.0, 0.0
        self.ready = False
        # J-transport config: OFF by default (see class docstring) -- enable_j_transport() flips
        # it on later too, e.g. once the active model's identity is known. `jlens_model_sha256` is
        # the strongest identity EngineSubstrate can actually offer (the running engine's own
        # /health "model_sha256" -- see jlens_transport.resolve_compact_jlens's model_sha256 tier);
        # `jlens_model_id` (a plain name string) is the weaker fallback for the legacy flat-manifest
        # sidecar that carries no compatible_gguf_sha256 at all.
        self._j_transport = bool(j_transport)
        self._jlens_dir = jlens_dir
        self._jlens_model_id = jlens_model_id
        self._jlens_model_sha256 = jlens_model_sha256
        self._jlens_artifact_root = jlens_artifact_root
        self._jlens_k = int(jlens_k) if jlens_k is not None else jlens_transport.DEFAULT_K
        self._jlens_cache: dict = {}          # {(dir, layer, k): CompactJLens}, shared across calls
        self.last_j_transport: dict | None = None   # the HONESTY flag: last transport_direction() result

    def enable_j_transport(self, *, jlens_dir=None, model_id=None, model_sha256=None,
                           artifact_root=None, k=None):
        """Turn on J-transport (see class docstring) after construction -- e.g. once the running
        engine's model identity is known (server/app.py builds EngineSteer before the engine has
        necessarily reported /health). Passing nothing just flips the flag on with whatever was
        set at __init__ (CLOZN_JLENS_DIR / ~/.clozn/jlens / ~/.clozn/artifacts by default -- see
        jlens_transport.resolve_compact_jlens)."""
        self._j_transport = True
        if jlens_dir is not None:
            self._jlens_dir = jlens_dir
        if model_id is not None:
            self._jlens_model_id = model_id
        if model_sha256 is not None:
            self._jlens_model_sha256 = model_sha256
        if artifact_root is not None:
            self._jlens_artifact_root = artifact_root
        if k is not None:
            self._jlens_k = int(k)

    def disable_j_transport(self):
        self._j_transport = False

    def _maybe_transport(self, vec: np.ndarray) -> np.ndarray:
        """Apply the optional J-transport step to a FINAL, fully-composed steer vector. Records
        the outcome on self.last_j_transport (the caller-visible honesty flag) every time this
        runs, whether or not transport was actually enabled/available -- so `last_j_transport is
        None` unambiguously means "J-transport wasn't even requested this call", and
        `last_j_transport["applied"]` is always the ground truth for "requested AND happened"."""
        if not self._j_transport:
            self.last_j_transport = None
            return vec
        result = jlens_transport.transport_direction(
            vec, jlens_dir=self._jlens_dir, layer=self.layer,
            model_sha256=self._jlens_model_sha256, model_id=self._jlens_model_id,
            artifact_root=self._jlens_artifact_root,
            expected_d_model=int(np.asarray(vec).shape[0]), k=self._jlens_k,
            cache=self._jlens_cache, norm="preserve")
        self.last_j_transport = result
        return np.asarray(result["vector"], dtype=np.float32)

    def _resolve_layer(self):
        """Choose a model-relative midpoint when no calibrated layer was supplied.

        Older code silently inherited Qwen2.5-7B's layer 14 for every unknown
        GGUF.  The worker now exposes its actual layer count, so an unqualified
        model starts from its own midpoint.  Wave-specific calibration may still
        replace this value with an empirically validated layer.
        """
        if self.layer is not None:
            return int(self.layer)
        try:
            health = self.ec.health()
            n_layer = int(health.get("n_layer") or 0)
        except Exception:
            n_layer = 0
        self.layer = max(1, n_layer // 2) if n_layer else 0
        return self.layer

    def load_library(self, path):
        """Load shipped dial metadata into self.custom without harvesting directions eagerly."""
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for name, v in data.items():
                self.custom[name] = {
                    "pos": v["pos"],
                    "neg": v["neg"],
                    "max": float(v.get("max", 0.5)),
                    "poles": [name, "neutral"],
                }
        except Exception:
            pass

    def compute(self, seeds=None) -> dict:
        seeds = axes.SEED_PROMPTS if seeds is None else seeds
        self._resolve_layer()
        # First-use signal: harvesting every base + library dial direction is ~30-40s of sequential
        # engine round-trips and is otherwise SILENT, so a first dial-touching turn reads as a hang
        # (a known usage-test papercut). stderr + ASCII-only (Windows cp1252).
        import sys
        n_dials = len(axes.AXES) + sum(1 for n in self.custom if n not in self.vecs)
        print(f"[steer] harvesting {n_dials} tone-dial directions; first use, ~30-40s...",
              file=sys.stderr, flush=True)
        norms = []
        for name, ax in axes.AXES.items():
            pos = np.mean(
                [self.ec.harvest(ax["pos"] + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
                axis=0,
            )
            neg = np.mean(
                [self.ec.harvest(ax["neg"] + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
                axis=0,
            )
            norms += [float(np.linalg.norm(pos)), float(np.linalg.norm(neg))]
            d = pos - neg
            self.vecs[name] = d / (float(np.linalg.norm(d)) + 1e-8)
        for name, entry in self.custom.items():
            if name in self.vecs:
                continue
            pos = np.mean(
                [self.ec.harvest(entry["pos"] + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
                axis=0,
            )
            neg = np.mean(
                [self.ec.harvest(entry["neg"] + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
                axis=0,
            )
            d = pos - neg
            self.vecs[name] = d / (float(np.linalg.norm(d)) + 1e-8)
        self.resid_norm = sum(norms) / len(norms)
        self.base = 0.08 * self.resid_norm
        self.ready = True
        return {"resid_norm": round(self.resid_norm, 1), "base": round(self.base, 1), "axes": list(self.vecs)}

    def add_custom(self, name, pos, neg, mx=0.5, seeds=None) -> dict:
        """Register a user-defined engine dial using the same diff-of-means recipe."""
        seeds = axes.SEED_PROMPTS if seeds is None else seeds
        self._resolve_layer()
        name = name.strip()[:24]
        pos_v = np.mean(
            [self.ec.harvest(pos + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
            axis=0,
        )
        neg_v = np.mean(
            [self.ec.harvest(neg + "\n\n" + s, layer=self.layer).activations.mean(0) for s in seeds],
            axis=0,
        )
        d = pos_v - neg_v
        self.vecs[name] = d / (float(np.linalg.norm(d)) + 1e-8)
        self.custom[name] = {"pos": pos, "neg": neg, "max": float(mx), "poles": [name, "neutral"],
                              "source": "user"}
        return self.custom[name]

    def remove_custom(self, name):
        self.custom.pop(name, None)
        self.vecs.pop(name, None)
        self.strength.pop(name, None)

    def save_custom(self, path):
        """Persist only user-created dials, not shipped-library metadata. Atomic for the same reason
        save_state is: open("w") truncates before a serialization failure can raise, and load_custom's
        except-pass would then silently eat the user's custom dials."""
        atomic_write_json(path, {k: {"pos": v["pos"], "neg": v["neg"], "max": v["max"]}
                                 for k, v in self.custom.items() if v.get("source") == "user"})

    def load_custom(self, path):
        if not os.path.isfile(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                self.add_custom(k, v["pos"], v["neg"], float(v.get("max", 0.5)))
        except Exception:
            pass

    def ceiling_for(self, name):
        """The strength ceiling actually enforced for one dial, and why. Returns (max, calibrated).

        THREE cases, in order:
          1. This exact model has a calibration entry with a usable_max -> use it. That number came
             from a coherence-gated sweep ON THIS MODEL and is the only trustworthy ceiling.
          2. Calibrated but swept-and-found-unusable (usable_max None) -> fall back to case 3 rather
             than the axis's optimistic declared max.
          3. NO calibration for this model -> UNSAFE_STRENGTH, not the axis's declared max.

        Case 3 is the fix for a real fail-open: axes.AXES declares max 1.5 for every dial, that 1.5
        was only ever validated on the model it was calibrated against, and the uncalibrated path
        served it anyway. Measured on Qwen2.5-7B-Q4 with no calibration: 0.25 was coherent but inert,
        0.5 already showed a repeated-4gram rate of 0.215, and 1.5 produced pure repetition loops
        while LOOKING like a 50% length reduction. Offering 1.5 to a model nobody swept is offering a
        number known to break some models.

        UNSAFE_STRENGTH (0.6) is a CEILING, NOT A GUARANTEE -- it is this codebase's own documented
        "validated working range" bound, and the 7B measurement above suggests some models derail
        below it. It is used here because inventing a lower number without a sweep would be making up
        evidence. The real answer is a per-model calibration; this only stops the worst case."""
        entry = self.calibration.get(name) or {}
        usable = entry.get("usable_max")
        if usable is not None:
            return float(usable), True
        return float(UNSAFE_STRENGTH), False

    def load_calibration(self, table):
        """Install this exact model's dial calibration ({dial: {"usable_max", ...}}). Anything falsy
        or malformed leaves the table empty, which means every dial takes the uncalibrated ceiling --
        the safe direction to fail."""
        self.calibration = dict(table) if isinstance(table, dict) else {}
        return self.calibration

    def set(self, name, value):
        """Set a dial's strength (capped to the ceiling from ceiling_for()) and return a warning
        string (or None) -- the single choke point every caller (`/steer/set`, `/steer/check`,
        replay nudges) goes through, so the message reaches all of them without duplication.

        Clamping to the uncalibrated ceiling IS now this function's job. That was previously deferred
        as "a product-behavior call the user hasn't made"; the call has since been made -- fail
        closed. A request above the ceiling is honoured up to the ceiling and warned about, never
        silently applied at full strength."""
        declared = float((axes.AXES.get(name) or self.custom.get(name) or {}).get("max", 1.5))
        ceiling, calibrated = self.ceiling_for(name)
        mx = min(declared, ceiling)
        requested = float(value)
        capped = max(-mx, min(mx, requested))
        # Warn ONLY when the uncalibrated ceiling is what bit -- i.e. this dial's own declared max
        # would have allowed the request and the missing calibration is the reason it didn't. Clamping
        # to a dial's OWN declared max was always silent and stays silent; that is not the bug being
        # fixed here, and making it noisy would change a contract callers already depend on.
        if not calibrated and ceiling < declared and abs(requested) > ceiling + 1e-9:
            msg = (f"'{name}' requested at {requested:+.2f}, clamped to {capped:+.2f}: no calibration "
                   f"exists for this exact model, so its declared max ({declared}) is not trusted and "
                   f"the ceiling is {UNSAFE_STRENGTH}. Calibrate this model to unlock its real range.")
            print(f"[steer] {msg}", file=sys.stderr, flush=True)
            self.strength[name] = capped
            return msg
        self.strength[name] = capped
        warning = None
        if abs(capped) > UNSAFE_STRENGTH:
            # Only reachable now when THIS model's own calibration set a usable_max above 0.6 -- the
            # uncalibrated path can no longer get here, it clamps at UNSAFE_STRENGTH above. So this is
            # no longer "you may be about to break your model": it is "your sweep said this is fine,
            # here is the generic caution anyway". Deliberately not a clamp: a measured usable_max on
            # this exact model outranks the generic bound.
            warning = (
                f"'{name}' at {capped:+.2f} is above the generic validated band (0.4-0.6), but within "
                f"THIS model's calibrated usable_max. Degradation (repetition loops, factual drift) is "
                f"where the generic bound comes from (sweep on 'concise', temp 0 -- clean through 0.6, "
                f"repeated-6-gram loop + broken capitalization + factual corruption at 0.8). Your "
                f"model's own sweep takes precedence; lower it if you see garbled output."
            )
            print(f"[steer] WARNING: {warning}", file=sys.stderr, flush=True)
        return warning

    @staticmethod
    def _text(r):
        ch = r.get("choices") if isinstance(r, dict) else None
        if ch:
            return ch[0].get("text") or ch[0].get("message", {}).get("content") or ""
        return (r.get("text") or "") if isinstance(r, dict) else str(r)

    def generate(self, prompt, strength=None, max_new=70):
        """Generate through the engine with active dials applied."""
        if not self.ready:
            self.compute()
        s = (self.strength if getattr(self, "_engaged", False) else {}) if strength is None else strength
        active = {k: v for k, v in s.items() if v and k in self.vecs}
        if not active:
            return self._text(self.ec.complete(prompt, max_tokens=max_new))
        vec = np.zeros_like(next(iter(self.vecs.values())))
        for k, v in active.items():
            vec = vec + self.base * float(v) * self.vecs[k]
        vec = self._maybe_transport(vec)
        return self._text(self.ec.intervene(prompt, vector=vec.tolist() if hasattr(vec, "tolist") else list(vec),
                                             coef=1.0, layer=self.layer, max_tokens=max_new))

    def steer_vector(self, strength):
        """Return the summed, pre-scaled tone direction for active dials, or None. When
        J-transport is enabled (see class docstring), this is where it is applied -- right before
        the caller puts the result on the wire (EngineSubstrate.chat's kw["steer_vec"]). Check
        self.last_j_transport after calling this to see whether it actually happened."""
        if not self.ready:
            self.compute()
        active = {k: v for k, v in (strength or {}).items() if v and k in self.vecs}
        if not active:
            self.last_j_transport = None
            return None
        vec = np.zeros_like(next(iter(self.vecs.values())))
        for k, v in active.items():
            vec = vec + self.base * float(v) * self.vecs[k]
        n = float(np.linalg.norm(vec))
        cap = self.base * 1.3
        if n > cap:
            vec = vec * (cap / n)
        vec = self._maybe_transport(vec)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def clear(self):
        self.strength = {}

    def engage(self):
        self._engaged = True

    def disengage(self):
        self._engaged = False

    def active(self):
        return {k: v for k, v in self.strength.items() if v}

    def save_state(self, path):
        # Atomic: a crash or a concurrent writer mid-write used to leave a truncated dial file, which
        # load_state's except-pass then silently swallowed -- the user's dials would just be gone, with
        # no error anywhere. Same failure shape as the settings/cards data-loss fix; same remedy.
        atomic_write_json(path, self.strength)

    def load_state(self, path):
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.strength = {k: float(v) for k, v in json.load(f).items()}
            except Exception:
                pass
