"""concept_dir.py -- the any-concept dial: dir(c) = normalize(J_l^T @ W_U[c]).

Wraps the VALIDATED research primitive (../clozn-jlens-work/scripts/dirc.py; live-checked at L21,
scale 0.25-0.5: raises logprob(c) +6..9 nat over baseline, content-specific vs a random equal-norm
write; scale>=1 over-injects and loses coherence -- see
../clozn-jlens-work/artifacts/dirc_selfconsistency_result.txt and j5a_swap_result.txt) into a
product module: type ANY concept word, get a steer direction, with ZERO contrastive pos/neg
calibration prompts (the retired tone-dial mechanism needed a harvested diff-of-means over a
whole pole pair; dir(c) needs neither a pole nor any harvesting). This module builds and caches
the vector, which rides the same steer_vec/coef/layer wire contract every steering write uses:
clozn_engine.EngineClient.intervene / .score, and EngineSubstrate.chat's kw["steer_vec"].

Math (identical convention to dirc.py / oracle.py -- both are lab-only and NOT imported here; the
~15 lines below are the product's own copy of the same formulas, since the formulas themselves
are just numpy, not a restricted asset):
  * J_l is a [d_model, d_model] fp32 matrix such that `transport(h, J) = h @ J_l.T` reproduces
    `J_l @ h` for h a column vector (matches jlens.lens.JacobianLens.transport / oracle.py).
  * W_U[c] is row `c` of the model's unembed/lm_head matrix ([vocab, d_model]); `unembed(x) =
    rmsnorm(x) @ W_U.T`.
  * dir_c(token_id, layer) = normalize(J_l.T @ W_U[token_id]), optionally scaled to a realistic
    residual magnitude (`scale * typical_norm`).

*** THE BLOCKER, AND THE FIX ***
J_l ships in the product today: ~/.clozn/jlens/J_layer{L}.f16 (the SAME raw fp16 sidecar the C++
engine's JlensServe::load reads for /jlens -- see engine/core/serve/server_shared.hpp). This
module reads it directly with plain numpy (load_jlens_jacobians below) -- no engine round trip
needed for J_l.

W_U (the model's unembed/lm_head matrix) was NOT exposed to product Python code anywhere: the
engine reads it straight out of the loaded GGUF's own (quantized) tensors, inside its own C++
process (JlensServe::load's out_head/out_norm), and no route handed any of it back -- /jlens
computes a full read+transport+unembed server-side and only ever returns TOP-K TOKENS for a real
residual it harvested from real text. The FIX: the engine now exposes
`POST /jlens/unembed_row {token_id}` -> `{vector: W_U[token_id]}` (engine/core/serve/
routes_jlens.cpp, added specifically to close this gap) -- ONE row (d_model floats), extracted
server-side via ggml_get_rows (which dequantizes whatever GGUF quant type the head tensor is), so
the full [vocab, d_model] matrix (~2 GB fp32) never has to leave the engine process. That is now
the DEFAULT path: ConceptDirSource.unembed_row() / fetch_unembed_row_from_engine() call it, and
ConceptSteer.compute() uses it automatically via the engine_client it already holds (for
resolve_token_id's /score round trip) -- no extra configuration needed to make dir(c) work
end-to-end against a running clozn-server with a J-lens sidecar loaded.
The OLDER lab-export path (a directory holding norm_weight.npy [d_model] fp32, lm_head_weight.npy
[vocab, d_model] fp32, unembed_meta.json {"rms_norm_eps": eps} -- the same shape
../clozn-jlens-work/artifacts already has) still works and still WINS if explicitly configured
(`unembed_dir=` / CLOZN_DIRC_UNEMBED_DIR) -- useful for offline dev/tests with no engine running,
or for the self-consistency verification that needs the FULL matrix (read_through_lens/rank_of).
`UnembedUnavailable` (see BLOCKER_NOTE) now only fires when NEITHER an explicit lab export NOR a
usable engine_client is available -- a genuine "no W_U source at all" degrade, not the normal case.

House honesty style (mirrors clozn/receipts/quant_receipts.py): never raise out of the
PRODUCT-FACING calls (ConceptSteer.compute/.steer_toward/.steer_vector) -- they return a
labeled `{"ok": False, "blocked": "...", "note": "..."}` dict instead. The lower-level math/loader
functions (dir_c, dir_c_from_row, load_unembed, load_jlens_jacobians, fetch_unembed_row_from_engine)
DO raise (ValueError / FileNotFoundError / UnembedUnavailable) -- they are the seam
ConceptSteer.compute() catches, kept raising for anyone building custom orchestration on top who
wants a normal exception instead of a checked dict.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_LAYER = 21                                  # the validated tap (J5a swap-receipts, dirc self-consistency)
DEFAULT_STRENGTH = 0.35                              # midpoint of the validated 0.25-0.5 operating range
VALIDATED_SCALE_RANGE = (0.25, 0.5)

# Per-layer median residual-row L2 norm at the validated tap(s), Qwen2.5-7B-Instruct Q4_K_M -- the
# realistic injection magnitude a UNIT dir(c) should be multiplied by (`coef` on the engine's
# steer wire, which itself computes `coef * vector` server-side -- see
# engine/core/serve/routes_state.cpp / routes_whitebox.cpp's `coef * raw_vec[i]`). Sourced from
# ../clozn-jlens-work/scripts/run_j5a_swap.py's MEDIAN_NORM (measured over cached hf_hidden
# activations, ../clozn-jlens-work/artifacts/dirc_selfconsistency_results.json's norm_calibration).
# Model/layer-specific: only valid for the fitted J-lens model (today: Qwen2.5-7B-Instruct).
#
# THE HOLE THIS MODULE'S PER-MODEL CALIBRATION CLOSES: the table above (and VALIDATED_SCALE_RANGE) is
# PINNED to one exact model -- a different GGUF has its own residual-activation scale entirely (the same
# lesson research/dial_autocalibrate_engine.py's own docstring documents for the diff-of-means tone
# dials: "a raw number does not transfer even to a different quantization/layer"). `load_concept_dial_
# calibration` / `save_concept_dial_calibration` below are the per-exact-model equivalent of
# clozn.server.app._dial_calibration()'s ~/.clozn/dial_calibration.json for THIS dial (dir(c)), written
# to the SAME _model_scoped_path-style per-digest location
# (~/.clozn/models/<model_sha256>/concept_dial_calibration.json) by
# scripts/calibration/concept_dial_autocalibrate.py's live sweep. `ConceptSteer` reads the per-model file
# when one is configured and present, and falls back to the constants below when it is absent -- NEVER
# silently: every resolved (median_resid_norm, scale_range) pair carries a `source` string
# ("per_model_calibration" | "global_default" | "explicit") so a caller can always tell which one fired
# (see ConceptSteer._resolve_median_norm/_resolve_scale_range and steer_toward's own "norm_source" field).
VALIDATED_MEDIAN_RESID_NORM = {16: 40.71, 21: 146.68, 25: 343.14}


# ============================================================================== the blocker + loaders

class UnembedUnavailable(RuntimeError):
    """No W_U (unembed/lm_head) source is available: neither an explicit lab export
    (unembed_dir=/CLOZN_DIRC_UNEMBED_DIR) NOR a usable engine_client with /jlens/unembed_row (the
    engine route added to close this gap -- see the module docstring's BLOCKER-AND-FIX section).
    ConceptSteer.compute() always tries the engine route by default; this only fires when that
    ALSO fails (no engine_client given, or the round trip itself errored) and there is no lab
    export configured either."""


BLOCKER_NOTE = (
    "dir(c) = normalize(J_l^T @ W_U[c]) needs W_U (the model's unembed/lm_head matrix row for "
    "the target token). The DEFAULT in-product source is the engine's POST /jlens/unembed_row "
    "{token_id} route (engine/core/serve/routes_jlens.cpp) -- ConceptSteer already calls this "
    "automatically via the engine_client it holds; this note only appears when that ALSO failed "
    "(no engine_client, the engine has no J-lens sidecar loaded, or the round trip errored) and "
    "no lab export is configured either. Set unembed_dir= (or CLOZN_DIRC_UNEMBED_DIR) to point at "
    "a directory holding norm_weight.npy [d_model] fp32 + lm_head_weight.npy [vocab, d_model] "
    "fp32 + unembed_meta.json {rms_norm_eps} (e.g. ../clozn-jlens-work/artifacts) as an explicit "
    "override/fallback -- there is no baked-in default for that path, only the engine route is "
    "on by default. (This was a hard blocker before the engine route "
    "shipped; now only degrades here on an actual engine/config failure.)"
)


@dataclass
class UnembedWeights:
    """norm_weight [d_model] fp32, lm_head_weight [vocab, d_model] fp32, rms eps."""
    norm_weight: np.ndarray
    lm_head_weight: np.ndarray
    eps: float = 1e-6


def _default_jlens_dir() -> str:
    """~/.clozn/jlens, honoring CLOZN_JLENS_DIR -- the SAME env var name the C++ engine already
    checks (see engine/core/serve/routes_jlens.cpp's 400 body: "start with --jlens <dir> or set
    CLOZN_JLENS_DIR"), so one env var configures both sides consistently."""
    env = os.environ.get("CLOZN_JLENS_DIR")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".clozn", "jlens")


def _default_unembed_dir() -> Optional[str]:
    """No baked-in default (see BLOCKER_NOTE) -- only an explicit env var."""
    return os.environ.get("CLOZN_DIRC_UNEMBED_DIR") or None


def load_jlens_manifest(jlens_dir: Optional[str] = None) -> dict:
    d = jlens_dir or _default_jlens_dir()
    path = os.path.join(d, "manifest.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jlens_jacobians(jlens_dir: Optional[str] = None, layers=None) -> dict:
    """J_l for every requested fitted layer, read straight from the product J-lens sidecar
    (~/.clozn/jlens/J_layer{L}.f16 by default) -- the SAME raw fp16 [d_model, d_model] file
    engine/core/serve/server_shared.hpp's JlensServe::load reads for /jlens (see its "ggml layout
    Jl(k,m)=J[m,k]" comment: a flat row-major buffer with J[m,k] at offset m*d_model+k -- exactly
    what np.fromfile(...).reshape(d_model, d_model) recovers, matching dirc.py/oracle.py's
    `h @ J.T` transport convention). Upcast to fp32 for the CPU matmul.

    Raises FileNotFoundError / ValueError (never silently returns a wrong-shaped matrix) on a
    missing manifest/sidecar, an unfitted layer, or a byte-count mismatch. Returns {layer: ndarray}.
    """
    d = jlens_dir or _default_jlens_dir()
    manifest = load_jlens_manifest(d)
    d_model = int(manifest["d_model"])
    available = [int(x) for x in manifest.get("layers", [])]
    want = [int(x) for x in layers] if layers is not None else available
    out = {}
    for layer in want:
        if layer not in available:
            raise ValueError(f"layer {layer} is not a fitted J-lens layer (available: {available})")
        path = os.path.join(d, f"J_layer{layer}.f16")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"missing J-lens sidecar {path!r} for layer {layer}")
        flat = np.fromfile(path, dtype="<f2")
        if flat.size != d_model * d_model:
            raise ValueError(
                f"{path!r}: expected {d_model * d_model} fp16 values ({d_model}x{d_model}), got {flat.size}")
        out[layer] = flat.reshape(d_model, d_model).astype(np.float32)
    return out


def load_unembed(unembed_dir: Optional[str] = None) -> UnembedWeights:
    """Load W_U from a directory holding norm_weight.npy/lm_head_weight.npy/unembed_meta.json (the
    SAME 3-file shape ../clozn-jlens-work/scripts/dirc.py's load_unembed already produces).
    Raises UnembedUnavailable (see BLOCKER_NOTE) when no directory is configured (neither
    `unembed_dir` nor CLOZN_DIRC_UNEMBED_DIR) or the directory is missing the required files --
    this is the ONE call in this module that names the real blocker instead of degrading quietly.
    """
    d = unembed_dir or _default_unembed_dir()
    if not d:
        raise UnembedUnavailable(BLOCKER_NOTE)
    norm_path = os.path.join(d, "norm_weight.npy")
    head_path = os.path.join(d, "lm_head_weight.npy")
    meta_path = os.path.join(d, "unembed_meta.json")
    if not (os.path.isfile(norm_path) and os.path.isfile(head_path)):
        raise UnembedUnavailable(
            f"unembed_dir {d!r} is missing norm_weight.npy / lm_head_weight.npy -- {BLOCKER_NOTE}")
    norm_weight = np.load(norm_path).astype(np.float32)
    lm_head_weight = np.load(head_path).astype(np.float32)
    eps = 1e-6
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as f:
                eps = float(json.load(f).get("rms_norm_eps", eps))
        except Exception:
            pass
    if lm_head_weight.ndim != 2 or norm_weight.shape != (lm_head_weight.shape[1],):
        raise ValueError(
            f"unembed shape mismatch: norm_weight {norm_weight.shape} vs lm_head_weight {lm_head_weight.shape}")
    return UnembedWeights(norm_weight=norm_weight, lm_head_weight=lm_head_weight, eps=eps)


# ============================================================================== per-model concept-dial calibration
#
# The per-exact-model equivalent of clozn.server.app._dial_calibration()'s ~/.clozn/dial_calibration.json,
# but for dir(c) instead of the diff-of-means tone dials -- see the module docstring's THE HOLE THIS
# MODULE'S PER-MODEL CALIBRATION CLOSES section above VALIDATED_MEDIAN_RESID_NORM. Written by
# scripts/calibration/concept_dial_autocalibrate.py's live sweep (deferred from this module's own tests --
# it needs a running engine + a loaded J-lens sidecar, exactly like that script's own live path); this
# module only owns the schema, the read, and the missing-file=no-op fallback discipline, all of which stay
# model-free and are unit-tested directly (tests/test_concept_dir.py).

CONCEPT_CALIBRATION_SCHEMA = "clozn.concept_dial_calibration.v1"


def concept_calibration_path(model_sha256: Optional[str] = None) -> str:
    """~/.clozn/models/<model_sha256>/concept_dial_calibration.json when a digest is given -- the SAME
    per-exact-GGUF scoping convention clozn.server.app._model_scoped_path uses for the tone-dial
    calibration file -- else the legacy ~/.clozn/concept_dial_calibration.json root (mirrors
    _model_scoped_path's own "no digest known yet" fallback). This module stays import-light (no
    clozn.server.app dependency: it must keep working standalone via --selftest/--demo, with no server
    process at all), so the digest is a plain string argument here, never read off a live substrate --
    callers that DO have a live substrate (the server) pass its `model_sha256` through explicitly."""
    base = os.path.join(os.path.expanduser("~"), ".clozn")
    if model_sha256:
        return os.path.join(base, "models", str(model_sha256), "concept_dial_calibration.json")
    return os.path.join(base, "concept_dial_calibration.json")


def load_concept_dial_calibration(model_sha256: Optional[str] = None, *,
                                  path: Optional[str] = None) -> Optional[dict]:
    """This exact model's per-layer concept-dial calibration, or None when there is nothing usable to
    read (missing file, unreadable JSON, wrong/missing schema tag, or no valid layer entries) -- NEVER
    raises, mirroring clozn.server.app._dial_calibration()'s missing-file=no-op discipline exactly: every
    caller (ConceptSteer) must fall back to the global VALIDATED_* constants exactly as it did before this
    existed, so an uncalibrated model's dir(c) behaves EXACTLY as it always has.

    Returns {layer:int -> {"median_resid_norm": float, "usable_scale_range": (lo, hi) or None,
    "derail_point": float or None, "works": bool}}, or None. A malformed individual layer entry is
    skipped (not fatal to the rest of the file), matching clozn.server.app._dial_calibration()'s own
    per-entry tolerance."""
    p = path or concept_calibration_path(model_sha256)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict) or raw.get("schema") != CONCEPT_CALIBRATION_SCHEMA:
            return None
        layers = raw.get("layers")
        if not isinstance(layers, dict) or not layers:
            return None
        out: dict = {}
        for key, entry in layers.items():
            try:
                layer = int(key)
            except (TypeError, ValueError):
                continue
            if not isinstance(entry, dict):
                continue
            median_norm = entry.get("median_resid_norm")
            if isinstance(median_norm, bool) or not isinstance(median_norm, (int, float)):
                continue
            scale_range = entry.get("usable_scale_range")
            if (isinstance(scale_range, (list, tuple)) and len(scale_range) == 2
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in scale_range)):
                scale_range = (float(scale_range[0]), float(scale_range[1]))
            else:
                scale_range = None
            out[layer] = {
                "median_resid_norm": float(median_norm),
                "usable_scale_range": scale_range,
                "derail_point": entry.get("derail_point"),
                "works": bool(entry.get("works", True)),
            }
        return out or None
    except Exception:
        return None


def save_concept_dial_calibration(model_sha256: Optional[str], layers: dict, *,
                                  path: Optional[str] = None, note: Optional[str] = None) -> str:
    """Atomically write one model's per-layer concept-dial calibration -- the counterpart
    scripts/calibration/concept_dial_autocalibrate.py's live sweep calls once it has measured this
    model's own median residual norm + usable dir(c) scale range at each fitted J-lens layer.

    `layers` is {layer -> {"median_resid_norm" (required), "usable_scale_range" ([lo, hi], optional),
    "derail_point" (optional), "works" (optional, default True), "n_samples" (optional)}}. Raises
    ValueError up front on a structurally bad entry (fail fast, before any file write) -- this is the
    ONE call in this pair that names a bad calibration input instead of writing a file `load_concept_
    dial_calibration` would then have to silently reject. Returns the path written."""
    validated: dict = {}
    for key, entry in layers.items():
        layer = int(key)
        if not isinstance(entry, dict) or entry.get("median_resid_norm") is None:
            raise ValueError(f"layer {layer}: calibration entry must be a dict with 'median_resid_norm'")
        median_norm = float(entry["median_resid_norm"])
        scale_range = entry.get("usable_scale_range")
        if scale_range is not None:
            if len(scale_range) != 2:
                raise ValueError(f"layer {layer}: usable_scale_range must be [lo, hi]")
            scale_range = [float(scale_range[0]), float(scale_range[1])]
        derail_point = entry.get("derail_point")
        validated[str(layer)] = {
            "median_resid_norm": median_norm,
            "usable_scale_range": scale_range,
            "derail_point": (float(derail_point) if derail_point is not None else None),
            "works": bool(entry.get("works", True)),
            "n_samples": (int(entry["n_samples"]) if entry.get("n_samples") is not None else None),
        }
    out = {
        "schema": CONCEPT_CALIBRATION_SCHEMA,
        "model_sha256": model_sha256,
        "layers": validated,
        "measured_ts": time.time(),
        "note": note,
    }
    dest = path or concept_calibration_path(model_sha256)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, dest)
    return dest


# ============================================================================== the math (dir(c) itself)

def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    variance = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 / np.sqrt(variance + eps)) * weight.astype(np.float32)


def _transport(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """h @ J.T -- J_l @ h for each row of h (h: [n, d_model], J: [d_model, d_model])."""
    return h.astype(np.float32) @ J.astype(np.float32).T


def _unembed(x: np.ndarray, unembed_weights: UnembedWeights) -> np.ndarray:
    normed = _rmsnorm(x, unembed_weights.norm_weight, unembed_weights.eps)
    return normed @ unembed_weights.lm_head_weight.astype(np.float32).T


def dir_c(token_id: int, layer: int, J_by_layer: dict, unembed_weights: UnembedWeights, *,
          scale: float = 1.0, typical_norm: Optional[float] = None) -> np.ndarray:
    """The injection direction: normalize(J_l^T @ W_U[token_id]), magnitude-scaled.

    `typical_norm`: when given, the returned vector has norm `scale * typical_norm` (a realistic
    injection magnitude at `layer`). When None (the default), the returned vector has norm
    `scale` directly -- a unit direction at scale=1.0, which is what the self-consistency check
    uses (RMSNorm-then-argmax is invariant to any positive rescaling of its input) and what
    ConceptSteer sends over the wire (paired with a separate `coef` the engine multiplies in).

    Raises ValueError for an unfitted layer, an out-of-range token_id, or a degenerate (~zero)
    raw direction -- never silently returns a wrong/garbage vector.
    """
    if layer not in J_by_layer:
        raise ValueError(f"layer {layer} has no loaded J_l (loaded: {sorted(J_by_layer)})")
    J = J_by_layer[layer]
    lm_head = unembed_weights.lm_head_weight
    if not (0 <= int(token_id) < lm_head.shape[0]):
        raise ValueError(f"token_id {token_id} is out of range for vocab size {lm_head.shape[0]}")
    if lm_head.shape[1] != J.shape[0]:
        raise ValueError(f"d_model mismatch: J_l is {J.shape}, W_U is {lm_head.shape}")
    w_c = lm_head[int(token_id)].astype(np.float32)
    raw = J.T @ w_c
    norm = float(np.linalg.norm(raw))
    if norm < 1e-12:
        raise ValueError(f"degenerate dir(c) for token {token_id} at layer {layer} (norm~0)")
    unit = raw / norm
    magnitude = scale if typical_norm is None else scale * typical_norm
    return (unit * magnitude).astype(np.float32)


def dir_c_from_row(w_c: np.ndarray, layer: int, J_by_layer: dict, *,
                   scale: float = 1.0, typical_norm: Optional[float] = None) -> np.ndarray:
    """dir(c) = normalize(J_l^T @ w_c) starting from an ALREADY-EXTRACTED W_U row (e.g. from
    fetch_unembed_row_from_engine's ONE-row engine round trip) instead of the full
    [vocab, d_model] lm_head matrix dir_c() needs. This is the engine-route path's entry point
    into the SAME math (same scale/typical_norm convention as dir_c() -- see its docstring); the
    two differ only in where w_c comes from, which is exactly the point: the product path never
    needs to hold the full unembed matrix in Python at all.

    Raises ValueError for an unfitted layer, a d_model-mismatched row, or a degenerate (~zero)
    raw direction -- never silently returns a wrong/garbage vector.
    """
    if layer not in J_by_layer:
        raise ValueError(f"layer {layer} has no loaded J_l (loaded: {sorted(J_by_layer)})")
    J = J_by_layer[layer]
    row = np.asarray(w_c, dtype=np.float32).reshape(-1)
    if row.shape[0] != J.shape[0]:
        raise ValueError(f"d_model mismatch: J_l is {J.shape}, w_c row is {row.shape}")
    raw = J.T @ row
    norm = float(np.linalg.norm(raw))
    if norm < 1e-12:
        raise ValueError("degenerate dir(c) (norm~0)")
    unit = raw / norm
    magnitude = scale if typical_norm is None else scale * typical_norm
    return (unit * magnitude).astype(np.float32)


def fetch_unembed_row_from_engine(engine_client, token_id: int) -> np.ndarray:
    """The DEFAULT in-product W_U source: POST /jlens/unembed_row -> W_U[token_id], the ONE
    unembed/lm_head row dir(c) needs (see engine/core/serve/routes_jlens.cpp, added to close the
    gap this module's BLOCKER_NOTE used to describe). `engine_client` is a clozn_engine.EngineClient
    (or anything duck-typed against its `.unembed_row(token_id) -> {"vector": [float,...], ...}`).

    Propagates whatever `engine_client.unembed_row` itself raises on a network/HTTP failure (e.g.
    EngineError -- "J-lens not loaded", connection refused, ...) unchanged; raises
    UnembedUnavailable itself only when the call SUCCEEDS but the response is missing/malformed
    `vector` (a server contract violation, not a normal failure mode). Either way, the caller
    (ConceptDirSource.unembed_row / ConceptSteer.compute) treats ANY exception here as "no W_U for
    this token right now" and degrades to a labeled blocked dict -- never raises out to product UI.
    """
    r = engine_client.unembed_row(int(token_id))
    vec = r.get("vector") if isinstance(r, dict) else None
    if not isinstance(vec, list) or not vec:
        raise UnembedUnavailable(
            f"engine /jlens/unembed_row returned no usable 'vector' for token {token_id}: {r!r}")
    return np.asarray(vec, dtype=np.float32)


def read_through_lens(vec: np.ndarray, layer: int, J_by_layer: dict,
                      unembed_weights: UnembedWeights) -> np.ndarray:
    """unembed(J_l @ vec) for a single [d_model] vector -- the self-consistency readout: logits
    [vocab] from pushing `vec` through the SAME J_l used to build it, then the model's own
    final-norm + unembed."""
    h = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    transported = _transport(h, J_by_layer[layer])
    logits = _unembed(transported, unembed_weights)
    return logits[0]


def rank_of(logits: np.ndarray, token_id: int) -> int:
    """0-based rank of `token_id` in `logits` (0 == top-1)."""
    return int((logits > logits[int(token_id)]).sum())


class ConceptDirSource:
    """Lazily loads + caches J_l (product sidecar) and W_U (see BLOCKER_NOTE) for one process."""

    def __init__(self, jlens_dir: Optional[str] = None, unembed_dir: Optional[str] = None):
        self.jlens_dir = jlens_dir
        self.unembed_dir = unembed_dir
        self._manifest: Optional[dict] = None
        self._J: dict = {}
        self._unembed: Optional[UnembedWeights] = None

    def manifest(self) -> dict:
        if self._manifest is None:
            self._manifest = load_jlens_manifest(self.jlens_dir)
        return self._manifest

    def available_layers(self) -> list:
        try:
            return [int(x) for x in self.manifest().get("layers", [])]
        except Exception:
            return []

    def jacobians(self, layers=None) -> dict:
        want = [int(x) for x in layers] if layers is not None else self.available_layers()
        missing = [layer for layer in want if layer not in self._J]
        if missing:
            self._J.update(load_jlens_jacobians(self.jlens_dir, layers=missing))
        return self._J

    def unembed(self) -> UnembedWeights:
        """The FULL [vocab, d_model] lab-export matrix (norm_weight/lm_head_weight/eps) -- only
        ever needed for the offline self-consistency check (read_through_lens/rank_of) or explicit
        lab dev, never for the product's own dir(c) build (see unembed_row). Raises
        UnembedUnavailable (never degrades silently) if no unembed_dir/CLOZN_DIRC_UNEMBED_DIR is
        configured -- see BLOCKER_NOTE."""
        if self._unembed is None:
            self._unembed = load_unembed(self.unembed_dir)
        return self._unembed

    def unembed_available(self) -> bool:
        try:
            self.unembed()
            return True
        except UnembedUnavailable:
            return False

    def unembed_row(self, token_id: int, engine_client=None) -> np.ndarray:
        """W_U[token_id] -- the ONE row dir(c) needs, never the full matrix. An EXPLICIT lab
        export (unembed_dir=/CLOZN_DIRC_UNEMBED_DIR) always wins, matching every other config knob
        in this module (see _default_unembed_dir/_default_jlens_dir) -- useful for offline dev/
        tests with no engine running. Otherwise -- the DEFAULT in-product path -- fetches the row
        from the running engine's /jlens/unembed_row via `engine_client`
        (fetch_unembed_row_from_engine); this is what makes dir(c) work with nothing but the
        shipped J-lens sidecar + a running clozn-server, no lab export needed at all.

        Raises UnembedUnavailable if neither an explicit export NOR an engine_client is available.
        An engine_client that IS given but fails propagates that failure unchanged -- the caller
        (ConceptSteer.compute) already wraps this call and turns any exception into a labeled
        blocked dict (see its docstring).
        """
        if self.unembed_available():
            return self.unembed().lm_head_weight[int(token_id)].astype(np.float32)
        if engine_client is not None:
            return fetch_unembed_row_from_engine(engine_client, token_id)
        raise UnembedUnavailable(BLOCKER_NOTE)


# ============================================================================== the product entry point

def _text_of(resp) -> str:
    """Extract generated text from an EngineClient .complete()/.intervene() response (OpenAI-ish
    {choices:[{text|message}]})."""
    ch = resp.get("choices") if isinstance(resp, dict) else None
    if ch:
        return ch[0].get("text") or (ch[0].get("message") or {}).get("content") or ""
    return (resp.get("text") or "") if isinstance(resp, dict) else str(resp)


class ConceptSteer:
    """Any-concept dial on a live engine client. Mirrors EngineSteer's persistent-dial surface
    (.set/.clear/.active/.steer_vector) so it drops into the same "dial UI" shape Fable already
    drives -- but each direction comes from dir(c), not diff-of-means harvesting, so there is no
    pos/neg calibration step: name a concept, get a direction.

    `engine_client` is a clozn_engine.EngineClient (or anything duck-typed against its
    `.score(prompt=, continuation=, topk=)` method, used only to resolve a concept WORD to its
    single vocab token id -- see resolve_token_id). Every product-facing method here NEVER raises;
    on any failure (an unresolvable/multi-token concept, an unfitted layer, or the unembed
    BLOCKER) it returns a labeled `{"ok": False, "blocked": "...", "note": "..."}` dict instead.
    """

    def __init__(self, engine_client, source: Optional[ConceptDirSource] = None,
                 layer: int = DEFAULT_LAYER, median_norm: Optional[float] = None,
                 model_sha256: Optional[str] = None, calibration: Optional[dict] = None):
        self.ec = engine_client
        self.source = source or ConceptDirSource()
        self.layer = int(layer)
        self.model_sha256 = model_sha256
        # Per-model concept-dial calibration (see the module docstring's THE HOLE ... section, and
        # load_concept_dial_calibration's own docstring for the missing-file=no-op discipline). `calibration`
        # lets a caller hand over an already-loaded dict directly (tests, or a caller that has it in memory
        # already); otherwise this loads ~/.clozn/models/<model_sha256>/concept_dial_calibration.json when
        # `model_sha256` is given, and stays None (falls straight through to the global VALIDATED_*
        # constants below) when it isn't -- so every EXISTING caller that never passes model_sha256 behaves
        # EXACTLY as it did before this existed (never silently wrong the other way either: `median_norm_
        # source`/`scale_range_source` always say which table actually fired).
        if calibration is not None:
            self._calibration = calibration
        elif model_sha256:
            self._calibration = load_concept_dial_calibration(model_sha256)
        else:
            self._calibration = None
        self._explicit_median_norm = float(median_norm) if median_norm is not None else None
        self.median_norm, self.median_norm_source = self._resolve_median_norm(self.layer)
        self.scale_range, self.scale_range_source = self._resolve_scale_range(self.layer)
        self.strength: dict = {}       # {concept: signed strength} -- the persistent dial state
        self._vecs: dict = {}          # {(concept, layer): unit dir(c) ndarray}
        self._token_ids: dict = {}     # {(concept, layer): token_id}

    # -- per-model calibration resolution ------------------------------------------------------

    def _resolve_median_norm(self, layer: int) -> tuple[Optional[float], str]:
        """(median_norm, source) for `layer`, in precedence order: an EXPLICIT ctor `median_norm`
        always wins ("explicit"); else this model's own per-model calibration file, when configured and
        it covers `layer` ("per_model_calibration"); else the global VALIDATED_MEDIAN_RESID_NORM table,
        pinned to Qwen2.5-7B-Instruct Q4_K_M ("global_default"); else (None, "unavailable") when nothing
        at all covers this layer. The `source` string is the "log which applied" this module's per-model
        calibration is required to never skip -- read it off self.median_norm_source, or off
        steer_toward()'s own returned "norm_source" field for one built vector."""
        if self._explicit_median_norm is not None:
            return self._explicit_median_norm, "explicit"
        cal = (self._calibration or {}).get(layer)
        if cal and cal.get("median_resid_norm") is not None:
            return float(cal["median_resid_norm"]), "per_model_calibration"
        global_val = VALIDATED_MEDIAN_RESID_NORM.get(layer)
        if global_val is not None:
            return float(global_val), "global_default"
        return None, "unavailable"

    def _resolve_scale_range(self, layer: int) -> tuple[tuple[float, float], str]:
        """(usable_scale_range, source) for `layer` -- this model's own per-model calibrated range when
        configured and present, else the global VALIDATED_SCALE_RANGE (0.25, 0.5), pinned to the same one
        fitted model as VALIDATED_MEDIAN_RESID_NORM. Unlike median_norm there is no 'unavailable' case:
        the global range is always a defined fallback (it just may be the wrong one for this model, which
        is exactly what a per-model measurement corrects)."""
        cal = (self._calibration or {}).get(layer)
        if cal and cal.get("usable_scale_range"):
            lo, hi = cal["usable_scale_range"]
            return (float(lo), float(hi)), "per_model_calibration"
        return VALIDATED_SCALE_RANGE, "global_default"

    # -- word -> token id --------------------------------------------------------------------

    def resolve_token_id(self, concept: str) -> dict:
        """Resolve a concept WORD to its single leading-space vocab token id via the engine's own
        tokenizer, reusing the EXISTING /score route (no new engine route needed): /score
        retokenizes its `continuation` text server-side and returns one entry per token (see
        clozn_engine.EngineClient.score's docstring). dir(c) needs exactly ONE vocab row, so a
        multi-token word is reported as unresolvable, not silently truncated to its first piece.

        Returns {"ok": True, "token_id": int, "piece": str} or {"ok": False, "note": "..."} --
        never raises (an engine round-trip can fail in many ways; all of them degrade here).
        """
        concept = (concept or "").strip()
        if not concept:
            return {"ok": False, "note": "empty concept"}
        try:
            r = self.ec.score(prompt="Consider the word:", continuation=" " + concept, topk=0)
        except Exception as e:
            return {"ok": False, "note": f"tokenization round-trip via /score failed: {e}"}
        toks = (r or {}).get("tokens") or []
        if len(toks) != 1:
            return {"ok": False,
                    "note": f"{concept!r} is not a single token ({len(toks)} pieces) -- dir(c) needs "
                            "exactly one vocab row; try a shorter/simpler word"}
        entry = toks[0]
        tid = entry.get("id") if isinstance(entry, dict) else None
        if not isinstance(tid, int):
            return {"ok": False, "note": "engine did not return an integer token id"}
        return {"ok": True, "token_id": tid, "piece": entry.get("piece")}

    # -- vector construction ------------------------------------------------------------------

    def compute(self, concept: str, *, layer: Optional[int] = None) -> dict:
        """Build (and cache) the UNIT dir(c) for `concept` at `layer` (default: self.layer).
        Never raises. Returns:
          {"ok": True, "concept", "layer", "token_id", "vector"} on success (`vector` is a plain
              list[float], unit norm -- pair it with a `coef` before sending to the engine; see
              steer_toward), or
          {"ok": False, "blocked": "token_resolution"|"unembed_unavailable"|"jlens_unavailable"|
              "bad_layer"|"error", "concept", "note"} on any failure.
        """
        layer = int(layer) if layer is not None else self.layer
        cache_key = (concept, layer)
        if cache_key in self._vecs:
            return {"ok": True, "concept": concept, "layer": layer,
                    "token_id": self._token_ids[cache_key], "vector": self._vecs[cache_key].tolist()}
        resolved = self.resolve_token_id(concept)
        if not resolved.get("ok"):
            return {"ok": False, "blocked": "token_resolution", "concept": concept, "layer": layer,
                    "note": resolved.get("note")}
        token_id = resolved["token_id"]
        # W_U[token_id]: the DEFAULT path is the engine's /jlens/unembed_row (via self.ec, the
        # same engine_client resolve_token_id already used) -- an explicit lab export
        # (unembed_dir=/CLOZN_DIRC_UNEMBED_DIR) wins if configured, see
        # ConceptDirSource.unembed_row. Any failure of either path (no engine_client, engine has
        # no J-lens loaded, connection error, ...) degrades here, never raises.
        try:
            w_c = self.source.unembed_row(token_id, engine_client=self.ec)
        except UnembedUnavailable as e:
            return {"ok": False, "blocked": "unembed_unavailable", "concept": concept, "layer": layer,
                    "token_id": token_id, "note": str(e)}
        except Exception as e:
            return {"ok": False, "blocked": "unembed_unavailable", "concept": concept, "layer": layer,
                    "token_id": token_id, "note": f"engine unembed-row fetch failed: {e}"}
        if layer not in self.source.available_layers():
            return {"ok": False, "blocked": "bad_layer", "concept": concept, "layer": layer,
                    "token_id": token_id,
                    "note": f"no fitted J-lens sidecar for layer {layer} (available: "
                            f"{self.source.available_layers()})"}
        try:
            J_by_layer = self.source.jacobians(layers=[layer])
        except Exception as e:
            return {"ok": False, "blocked": "jlens_unavailable", "concept": concept, "layer": layer,
                    "token_id": token_id, "note": str(e)}
        if layer not in J_by_layer:
            return {"ok": False, "blocked": "bad_layer", "concept": concept, "layer": layer,
                    "token_id": token_id,
                    "note": f"no fitted J-lens sidecar for layer {layer} (available: "
                            f"{self.source.available_layers()})"}
        try:
            vec = dir_c_from_row(w_c, layer, J_by_layer, scale=1.0, typical_norm=None)
        except Exception as e:
            return {"ok": False, "blocked": "error", "concept": concept, "layer": layer,
                    "token_id": token_id, "note": str(e)}
        self._vecs[cache_key] = vec
        self._token_ids[cache_key] = token_id
        return {"ok": True, "concept": concept, "layer": layer, "token_id": token_id, "vector": vec.tolist()}

    def steer_toward(self, concept: str, strength: float = DEFAULT_STRENGTH, *,
                     layer: Optional[int] = None) -> dict:
        """THE product entry point (mirrors the tone-dial shape: a name + a strength). Persists
        `concept`'s strength (so it composes with steer_vector() like a normal dial) AND returns
        an /intervene-ready payload in one call:
          {"ok": True, "concept", "token_id", "layer", "strength", "vector" (UNIT, list[float]),
           "coef" (float), "note"?, "norm_source"} -- feed straight into
           engine_client.intervene(prompt, vector=res["vector"], coef=res["coef"],
                                    layer=res["layer"], max_tokens=...),
           or engine_client.score(..., steer_vec=res["vector"],
                                   steer={"coef": res["coef"], "layer": res["layer"]}).
        `coef = strength * this layer's median residual norm` -- the realistic injection magnitude,
        resolved by `_resolve_median_norm` in precedence order: an explicit ctor `median_norm`, else
        this model's own per-model calibration (see load_concept_dial_calibration), else the global
        VALIDATED_MEDIAN_RESID_NORM table pinned to Qwen2.5-7B-Instruct Q4_K_M. `norm_source` on the
        returned dict says which one actually fired ("explicit" | "per_model_calibration" |
        "global_default") -- never silently wrong about which table a caller is trusting. The engine
        multiplies `coef * vector` itself, so `vector` stays a unit direction on the wire.
        On any failure, returns compute()'s `{"ok": False, "blocked", "note"}` shape (never
        raises), with `strength` folded in for the caller's bookkeeping.
        """
        built = self.compute(concept, layer=layer)
        if not built.get("ok"):
            built["strength"] = strength
            return built
        resolved_layer = built["layer"]
        if resolved_layer == self.layer:
            median_norm, norm_source = self.median_norm, self.median_norm_source
            scale_range = self.scale_range
        else:
            median_norm, norm_source = self._resolve_median_norm(resolved_layer)
            scale_range, _ = self._resolve_scale_range(resolved_layer)
        if not median_norm:
            return {"ok": False, "blocked": "no_norm_calibration", "concept": concept,
                    "layer": resolved_layer, "strength": strength,
                    "note": f"no validated median-residual-norm calibration for layer {resolved_layer}; "
                            "construct ConceptSteer(..., median_norm=...) explicitly for this layer, or "
                            "run scripts/calibration/concept_dial_autocalibrate.py to measure one"}
        self.strength[concept] = float(strength)
        coef = float(strength) * float(median_norm)
        note = None
        if abs(strength) >= 1.0:
            note = (f"validated operating point is scale in [{scale_range[0]}, {scale_range[1]}] at "
                    f"L{resolved_layer} (+6..9 nat over baseline logprob, content-specific vs an "
                    "equal-norm random write); |strength|>=1.0 over-injects and degrades coherence "
                    "(see j5a_swap_result.txt)")
        return {"ok": True, "concept": concept, "token_id": built["token_id"], "layer": resolved_layer,
                "strength": float(strength), "vector": built["vector"], "coef": coef, "note": note,
                "norm_source": norm_source}

    # -- persistent-dial surface (mirrors EngineSteer) ----------------------------------------

    def set(self, concept: str, value: float):
        """Mirror EngineSteer.set: persist a strength, lazily (no vector build here)."""
        self.strength[concept] = max(-1.5, min(1.5, float(value)))

    def clear(self):
        self.strength = {}

    def active(self) -> dict:
        return {k: v for k, v in self.strength.items() if v}

    def steer_vector(self, strength: Optional[dict] = None) -> Optional[list]:
        """Sum every ACTIVE concept's coef-scaled unit vector into ONE raw steer vector at
        self.layer -- the SAME return contract as EngineSteer.steer_vector (a plain list[float],
        pre-scaled, or None when nothing is active) -- so a caller can add this into
        EngineSubstrate.chat's kw["steer_vec"] alongside tone dials. A concept whose vector can't
        be built (see compute()) is SKIPPED here, not fatal -- call steer_toward() directly to see
        why any one concept failed."""
        active = {k: v for k, v in (strength if strength is not None else self.strength).items() if v}
        if not active:
            return None
        total = None
        for concept, value in active.items():
            built = self.compute(concept, layer=self.layer)
            if not built.get("ok"):
                continue
            # built["layer"] == self.layer always here (compute() was called with layer=self.layer
            # explicitly above), so self.median_norm -- already resolved with the full explicit ->
            # per-model-calibration -> global-default precedence (see _resolve_median_norm) -- is the
            # correct value for this concept's contribution, no re-resolution needed.
            median_norm = self.median_norm
            if not median_norm:
                continue
            contribution = np.asarray(built["vector"], dtype=np.float32) * (float(value) * float(median_norm))
            total = contribution if total is None else total + contribution
        return total.tolist() if total is not None else None


# ============================================================================== CLI: --selftest / --demo
#
# Mirrors engine/client/clozn_engine.py's own --selftest/--demo split: --selftest is offline (no
# engine, no GPU, safe as this module's default with no flags -- exercised in spirit by
# tests/test_concept_dir.py's fixture-based self-consistency checks, just runnable standalone
# here too). --demo is the LIVE smoke -- it needs only a running clozn-server with a J-lens
# sidecar loaded (`--jlens <dir>` / CLOZN_JLENS_DIR); W_U comes from that same server's
# /jlens/unembed_row route (see the BLOCKER-AND-FIX section above), so --unembed-dir is now
# OPTIONAL (pass it only to force the older lab-export path instead, e.g. for a model whose
# engine isn't running): `python -m clozn.analysis.concept_dir --demo --port 8095
#   --concept ocean --strength 0.35`

def _selftest() -> int:
    """Offline self-consistency proof on a synthetic fixture (orthogonal J + orthonormal W_U) --
    no engine, no GPU, no real J-lens/unembed files. See tests/test_concept_dir.py for the same
    check plus the full loader/ConceptSteer suite."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        jdir = os.path.join(tmp, "jlens")
        os.makedirs(jdir)
        d_model = 32
        rng = np.random.default_rng(0)
        J, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
        J = J.astype(np.float32)
        J.astype("<f2").tofile(os.path.join(jdir, "J_layer21.f16"))
        with open(os.path.join(jdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "selftest", "d_model": d_model, "vocab": d_model, "layers": [21],
                      "engine_default_tap_layer": 21}, f)
        udir = os.path.join(tmp, "unembed")
        os.makedirs(udir)
        W, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
        W = W.astype(np.float32)
        np.save(os.path.join(udir, "norm_weight.npy"), np.ones(d_model, dtype=np.float32))
        np.save(os.path.join(udir, "lm_head_weight.npy"), W)
        with open(os.path.join(udir, "unembed_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"rms_norm_eps": 1e-6}, f)

        source = ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)
        J_by_layer = source.jacobians()
        unembed_weights = source.unembed()
        ranks = []
        for c in range(d_model):
            vec = dir_c(c, 21, J_by_layer, unembed_weights, scale=1.0)
            logits = read_through_lens(vec, 21, J_by_layer, unembed_weights)
            ranks.append(rank_of(logits, c))
        ok = all(r == 0 for r in ranks)
        print(f"selftest {'OK' if ok else 'FAIL'}: {sum(r == 0 for r in ranks)}/{d_model} tokens "
              f"recovered dir(c) at exact top-1 through their own lens")
        return 0 if ok else 1


def _demo(args) -> int:
    """LIVE smoke -- deferred: needs a running clozn-server (with --jlens) and a configured unembed
    export; never invoked automatically."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    import sys as _sys
    _sys.path.insert(0, os.path.join(repo_root, "engine", "client"))
    from clozn_engine import EngineClient  # local import: only the --demo path needs the engine SDK

    ec = EngineClient(host=args.host, port=args.port)
    print(f"server: {ec.health().get('model')}")
    source = ConceptDirSource(jlens_dir=args.jlens_dir, unembed_dir=args.unembed_dir)
    steer = ConceptSteer(ec, source=source, layer=args.layer)
    built = steer.steer_toward(args.concept, args.strength)
    if not built.get("ok"):
        print(f"blocked: {built.get('blocked')} -- {built.get('note')}")
        return 1
    print(f"dir({args.concept!r}) at L{built['layer']}: token_id={built['token_id']} "
         f"coef={built['coef']:.2f}")
    base = ec.complete(args.prompt, max_tokens=args.max_tokens)
    swapped = ec.intervene(args.prompt, vector=built["vector"], coef=built["coef"],
                           layer=built["layer"], max_tokens=args.max_tokens)
    print("baseline:", _text_of(base))
    print("swapped :", _text_of(swapped))
    return 0


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="concept_dir.py -- any-concept dial (dir(c))")
    ap.add_argument("--selftest", action="store_true", help="offline self-consistency proof (default, safe)")
    ap.add_argument("--demo", action="store_true", help="LIVE smoke against a running clozn-server (DEFERRED)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--jlens-dir", default=None, help="default: ~/.clozn/jlens or CLOZN_JLENS_DIR")
    ap.add_argument("--unembed-dir", default=None, help="see BLOCKER_NOTE; default: CLOZN_DIRC_UNEMBED_DIR")
    ap.add_argument("--concept", default="ocean")
    ap.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    ap.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args(argv)
    if args.demo:
        return _demo(args)
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
