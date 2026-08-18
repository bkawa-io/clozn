"""jlens_transport.py -- J-transport an ALREADY-BUILT steer direction before it goes on the wire.

Measured in notes/JLENS_SAE_FINDINGS.md (finding #1, "J-transported SAE steering is 1.5-2x more
stable than raw"): SAE/concept decoder directions -- and, by the same argument, EngineSteer's own
diff-of-means tone-dial directions (axes.py) -- live in the full [d_model] residual space, but
~93% of a typical direction sits in J's NULL space: dimensions the remaining layers provably never
read. J-transport concentrates a direction into the ~7% that actually propagates to the model's
output. Measured on Qwen2.5-7B across 6 behavioral contrasts: raw directions collapse at coef ~40,
J-transported directions survive to coef ~60-80. Cosine between raw and transported is only
0.05-0.07 -- nearly orthogonal, i.e. genuinely different (and better) directions, not a rounding
tweak.

THIS module is the general-purpose, direction-agnostic step: "take a direction someone already
built (diff-of-means, dir(c), an SAE decoder column, anything living in one layer's [d_model]
residual space) and J-transport it." It is deliberately separate from concept_dir.py's dir(c),
which already bakes an equivalent transport into its OWN math (dir_c = normalize(J_l^T @
W_U[token])) -- dir(c) does not need this module; a caller building a direction some other way
(an SAE decoder column, or any other [d_model] residual-space direction) is exactly who this
module is for.

*** THE COMPACT FORM (why no dense [d_model, d_model] matrix is required at injection time) ***
notes/JLENS_SAE_FINDINGS.md's "Key technical notes":
    J-transport formula:            normalize(J^T @ raw_direction)
    Compact J-transport formula:    jt = Vh.T @ (S * (Vh @ dir))
`Vh` ([k, d_model], the top-k right-singular directions of J) and `S` ([k], their singular
values) are a truncated SVD of J -- a compact J is ~1.3 MB (k=50) vs ~25.7 MB dense fp16 for
Qwen2.5-7B's d_model=3584, and (per finding #5) is 98.4% output-aligned with the dense J^T
transport while steering 39% harder (no null-space dilution). `compact_transport` below takes
only (Vh, S) -- the dense J itself is never touched by the per-injection hot path. The ONE place
a dense matrix is still needed is `fit_compact_from_dense`, an explicit OFFLINE step that derives
(Vh, S) from an existing dense J-lens sidecar (~/.clozn/artifacts/jlens/<model>/J_layer{L}.f16 --
see concept_dir.py's load_jlens_jacobians) the FIRST time a given (model, layer) is used; its
result is cached (see resolve_compact_jlens's `cache` argument) so every subsequent transport for
that (model, layer) touches only the compact (Vh, S).

*** HONESTY CONTRACT (house style, mirrors concept_dir.py's ConceptSteer) ***
The product-facing entry points here (resolve_compact_jlens, transport_direction) NEVER raise and
NEVER fabricate: no J artifact for the active model -> a labeled no-op (`{"applied": False,
"reason": "no_jlens_artifact", ...}`, direction returned UNCHANGED); a J artifact that claims a
DIFFERENT model (or a mismatched residual width) -> REFUSED (`"reason": "wrong_model"` /
`"dim_mismatch"`), never silently substituted or broadcast; a genuine shape mismatch inside the
math itself -> refused (`"reason": "dimension_mismatch"`), never silently truncated/padded.
Whether transport actually happened is always the caller's to check via the returned `"applied"`
flag -- nothing upstream may assume it happened just because it was requested. The lower-level
math (`compact_transport`, `apply_compact_jlens`, `fit_compact_from_dense`) DOES raise
(`JTransportError`) on malformed input, exactly like concept_dir.py's dir_c/dir_c_from_row -- it is
the seam the product-facing calls catch.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import concept_dir

DEFAULT_K = 50  # notes finding #5: k=50, power=2 recovers 99.7% mean live-subspace containment;
                # k=50 compact transport measured 98.4% output-aligned vs dense J^T.


class JTransportError(ValueError):
    """Malformed compact-J input (shape mismatch, degenerate result). Raised only by the
    low-level math (compact_transport/apply_compact_jlens/fit_compact_from_dense); the
    product-facing resolve_compact_jlens/transport_direction catch this and degrade to a labeled
    dict instead -- see the module docstring's HONESTY CONTRACT."""


@dataclass
class CompactJLens:
    """A truncated-SVD compact J for one (model, layer): Vh [k, d_model] fp32 (top-k right
    singular vectors -- the live-subspace basis) + S [k] fp32 (their singular values). No dense
    [d_model, d_model] matrix is held here."""
    Vh: np.ndarray
    S: np.ndarray
    layer: int
    d_model: int
    model: Optional[str] = None
    source: dict = field(default_factory=dict)   # {"jlens_dir":, "fitted_from_dense": bool, ...}

    def __post_init__(self):
        self.Vh = np.asarray(self.Vh, dtype=np.float32)
        self.S = np.asarray(self.S, dtype=np.float32)
        if self.Vh.ndim != 2:
            raise JTransportError(f"Vh must be 2-D [k, d_model], got shape {self.Vh.shape}")
        if self.S.ndim != 1 or self.S.shape[0] != self.Vh.shape[0]:
            raise JTransportError(
                f"S must be 1-D [k] matching Vh's k ({self.Vh.shape[0]}), got shape {self.S.shape}")
        if int(self.d_model) != int(self.Vh.shape[1]):
            raise JTransportError(
                f"d_model={self.d_model} does not match Vh's column count {self.Vh.shape[1]}")

    @property
    def k(self) -> int:
        return int(self.Vh.shape[0])


# ============================================================================== the math (no dense J required)

def compact_transport(direction, Vh, S) -> np.ndarray:
    """jt = Vh.T @ (S * (Vh @ dir)) -- the COMPACT J-transport formula (see module docstring).
    Only (Vh, S) are touched; the dense [d_model, d_model] J never has to exist for this call.

    Raises JTransportError on any dimension mismatch (never silently broadcasts/truncates/pads --
    a d_model-mismatched direction is a caller bug or a wrong-model J, not something to guess
    through).
    """
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    Vh = np.asarray(Vh, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32).reshape(-1)
    if Vh.ndim != 2:
        raise JTransportError(f"Vh must be 2-D [k, d_model], got shape {Vh.shape}")
    k, d_model = Vh.shape
    if S.shape[0] != k:
        raise JTransportError(f"S has {S.shape[0]} entries but Vh has k={k} rows")
    if direction.shape[0] != d_model:
        raise JTransportError(
            f"direction has {direction.shape[0]} dims but this compact J is fitted for "
            f"d_model={d_model} -- refusing rather than truncating/broadcasting")
    projected = Vh @ direction     # [k]        -- dir's coordinates in the live-subspace basis
    scaled = S * projected         # [k]        -- weighted by each channel's singular value
    return Vh.T @ scaled           # [d_model]  -- back to residual space, concentrated in-subspace


def apply_compact_jlens(direction, compact: CompactJLens, *, norm: str = "unit") -> np.ndarray:
    """Run `direction` through `compact` (a CompactJLens) and rescale the result. `norm`:
      "unit"     -- normalize(J^T @ direction), exactly finding #1's formula (matches the unit
                    dir(c) convention concept_dir.py's ConceptSteer already sends over the wire,
                    a separate `coef` supplies the injection magnitude).
      "preserve" -- rescale the transported vector back to `direction`'s OWN norm. For a direction
                    whose magnitude is already a calibrated injection strength (e.g. EngineSteer's
                    diff-of-means tone vectors, which the engine applies at coef=1.0) -- J-transport
                    should rotate the direction into the live subspace WITHOUT discarding that
                    calibration.
      "none"     -- return the raw transported vector, no rescale (advanced/debug use).
    Raises JTransportError on a dimension mismatch (from compact_transport) or a degenerate
    (~zero-norm) result -- never silently returns a garbage/zero vector.
    """
    direction = np.asarray(direction, dtype=np.float32).reshape(-1)
    raw = compact_transport(direction, compact.Vh, compact.S)
    raw_norm = float(np.linalg.norm(raw))
    if raw_norm < 1e-12:
        raise JTransportError("J-transported direction is degenerate (norm~0); refusing to return a zero vector")
    if norm == "unit":
        return raw / raw_norm
    if norm == "preserve":
        original_norm = float(np.linalg.norm(direction))
        if original_norm < 1e-12:
            raise JTransportError("cannot preserve the norm of an all-zero input direction")
        return raw * (original_norm / raw_norm)
    if norm == "none":
        return raw
    raise JTransportError(f"unknown norm mode {norm!r}; expected 'unit', 'preserve', or 'none'")


def fit_compact_from_dense(J: np.ndarray, k: int = DEFAULT_K):
    """Derive a top-k compact (Vh, S) representation from a dense [d_model, d_model] J via SVD.

    An explicit OFFLINE / one-time step -- NOT part of the hot transport path (compact_transport /
    apply_compact_jlens need only the (Vh, S) this returns, never J itself again). Exists so an
    already-shipped DENSE J-lens sidecar (~/.clozn/artifacts/jlens/<model>/J_layer{L}.f16, read via
    concept_dir.load_jlens_jacobians) can back the compact transport path without a second artifact
    format. Uses a plain truncated SVD (numpy-only, no scipy) for correctness/simplicity; a
    production fitter would use randomized SVD for speed on large d_model (see
    notes/JLENS_SAE_FINDINGS.md finding #5) -- same output shape, this module doesn't care which
    produced (Vh, S).

    Returns (Vh [k, d_model] fp32, S [k] fp32). Raises JTransportError if J isn't square 2-D.
    """
    J = np.asarray(J, dtype=np.float32)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise JTransportError(f"J must be a square [d_model, d_model] matrix, got shape {J.shape}")
    d_model = J.shape[0]
    k = int(min(k, d_model))
    _, S, Vh = np.linalg.svd(J, full_matrices=False)
    return Vh[:k].astype(np.float32).copy(), S[:k].astype(np.float32).copy()


# ============================================================================== discovery: find/refuse a J artifact

def _resolve_jlens_dir(jlens_dir: Optional[str]) -> str:
    """Same resolution order as concept_dir._default_jlens_dir (explicit arg > CLOZN_JLENS_DIR >
    ~/.clozn/jlens) so ONE env var configures both dir(c) and this generic transport identically."""
    if jlens_dir:
        return jlens_dir
    return concept_dir._default_jlens_dir()


def _default_artifact_root() -> str:
    """~/.clozn/artifacts, honoring CLOZN_ARTIFACTS_DIR -- the SAME root
    clozn.cli.engine_process.spawn_engine already scans via contracts.find_compatible_artifact("jlens",
    ...) to pick the engine's own --jlens directory at launch. Tier 1/2 below search under
    `<this>/jlens/` for a model-scoped artifact (~/.clozn/artifacts/jlens/<model>/)."""
    return os.environ.get("CLOZN_ARTIFACTS_DIR") or os.path.join(
        os.path.expanduser("~"), ".clozn", "artifacts")


def _contract_model_block(manifest: dict) -> Optional[dict]:
    model = manifest.get("model")
    return model if isinstance(model, dict) else None


def _find_jlens_dir_by_sha256(model_sha256: str, *, jlens_dir: Optional[str] = None,
                              artifact_root: Optional[str] = None) -> dict:
    """Tier-2 discovery: match on the sidecar's OWN declared model.compatible_gguf_sha256 (the
    exact field clozn/artifacts/contracts.py's validate_artifact_manifest checks) without needing
    a full GGUF identity dict -- for callers that only ever learn the running model's digest (e.g.
    EngineSubstrate.model_sha256 from the engine's own /health, never a local GGUF file path to
    re-derive full metadata from). A real cryptographic match, not a name-string guess; still
    weaker than resolve_compact_jlens(model_identity=...) below (that path also cross-checks
    architecture/hidden_size/vocab/tokenizer and every artifact file's own checksum via
    contracts.validate_artifact_manifest). Legacy-style manifests (no "model" dict / no
    compatible_gguf_sha256 at all -- e.g. ~/.clozn/jlens) never match here; they only ever match
    via the model_id NAME check in resolve_compact_jlens.

    Returns {"ok": True, "jlens_dir":, "manifest":} or {"ok": False, "reason": ...}.
    """
    if jlens_dir:
        candidate_dirs = [jlens_dir]
    else:
        root = os.path.join(artifact_root or _default_artifact_root(), "jlens")
        if not os.path.isdir(root):
            return {"ok": False, "reason": "no_jlens_artifact"}
        candidate_dirs = sorted({
            os.path.dirname(os.path.join(dirpath, "manifest.json"))
            for dirpath, _dirnames, filenames in os.walk(root) if "manifest.json" in filenames
        })

    matches = []
    for d in candidate_dirs:
        manifest_path = os.path.join(d, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
        model_block = _contract_model_block(manifest) if isinstance(manifest, dict) else None
        compatible = model_block.get("compatible_gguf_sha256") if model_block else None
        if isinstance(compatible, list) and str(model_sha256).lower() in {
                str(x).lower() for x in compatible}:
            matches.append((d, manifest))
    if not matches:
        return {"ok": False, "reason": "wrong_model" if jlens_dir else "no_jlens_artifact"}
    if len(matches) > 1:
        return {"ok": False, "reason": "ambiguous_artifact",
                "note": f"multiple jlens artifacts claim GGUF sha256 {model_sha256}: "
                        f"{[d for d, _ in matches]}"}
    d, manifest = matches[0]
    return {"ok": True, "jlens_dir": d, "manifest": manifest}


def _verify_sidecar_checksum(jlens_dir: str, manifest: dict, layer: int) -> Optional[str]:
    """For a tier-2 (sha256-only) match: verify the ONE J_layer{L}.f16 file this call will
    actually read against the manifest's own declared checksum (manifest["files"][name]["sha256"]),
    reusing contracts.sha256_file rather than reimplementing hashing. Returns an error note string
    on a mismatch/missing declaration, or None when the file checks out (or the manifest simply
    doesn't declare per-file checksums, e.g. a hand-written test fixture -- nothing to verify)."""
    files = manifest.get("files")
    if not isinstance(files, dict):
        return None
    name = f"J_layer{layer}.f16"
    spec = files.get(name)
    if not isinstance(spec, dict) or not spec.get("sha256"):
        return None
    from clozn.artifacts import contracts
    path = os.path.join(jlens_dir, name)
    if not os.path.isfile(path):
        return f"declared file {name!r} is missing from {jlens_dir!r}"
    actual = contracts.sha256_file(path)
    expected = str(spec["sha256"]).lower()
    if actual.lower() != expected:
        return f"{name!r} checksum mismatch: manifest says {expected}, file hashes to {actual}"
    return None


def resolve_compact_jlens(*, jlens_dir: Optional[str] = None, layer: Optional[int] = None,
                          model_identity: Optional[dict] = None, model_sha256: Optional[str] = None,
                          model_id: Optional[str] = None, expected_d_model: Optional[int] = None,
                          artifact_root: Optional[str] = None, k: int = DEFAULT_K,
                          cache: Optional[dict] = None) -> dict:
    """Resolve a CompactJLens for one (jlens_dir, layer), honest about every way it can NOT be
    available. NEVER raises -- see the module docstring's HONESTY CONTRACT.

    Model-matching rigor, strongest first (pass the strongest identity your call site actually
    has -- do not fabricate a stronger one):
      `model_identity` -- a full GGUF identity dict, the EXACT shape clozn.artifacts.contracts.
          gguf_identity()/find_compatible_artifact() already validate against (architecture,
          hidden_size, layer_count, vocab_size, tokenizer_sha256, sha256). Delegates discovery AND
          validation entirely to contracts.find_compatible_artifact("jlens", model_identity,
          artifact_root, explicit_dir=jlens_dir) -- every artifact FILE's checksum is verified too,
          not just the manifest's claims. The rigorous path; use it whenever a local GGUF path is
          in hand (e.g. clozn.cli.engine_process/qualify already compute this identity).
      `model_sha256` -- just the loaded GGUF's digest (e.g. EngineSubstrate.model_sha256 from the
          running engine's own /health, which never hands back a full identity or file path). Matches
          against the manifest's own model.compatible_gguf_sha256 (see _find_jlens_dir_by_sha256)
          plus a checksum of the one sidecar file actually read -- a real cryptographic check, just
          without the full metadata cross-validation model_identity gives you.
      `model_id` -- a plain display-name string (e.g. "Qwen/Qwen2.5-7B-Instruct"). Compared against
          the manifest's own declared model name (manifest["model"], or
          manifest["model"]["source_id"] for a contract-style manifest). The WEAKEST check -- no
          cryptographic guarantee, just a name match -- kept for the legacy flat-manifest sidecar
          (~/.clozn/jlens) that carries no compatible_gguf_sha256/file-checksum data at all.
      (none of the above) -- NO wrong-model protection beyond the d_model/layer checks below.

    `expected_d_model`: the active model's own residual width (e.g. engine /health's n_embd).
    Refused (not silently broadcast) when it disagrees with the sidecar's declared d_model.

    Returns {"available": True, "compact": CompactJLens, "layer":, "model":, "jlens_dir":,
    "cached": bool} or {"available": False, "reason": "no_jlens_artifact"|"corrupt_manifest"|
    "wrong_model"|"dim_mismatch"|"layer_not_fitted"|"missing_sidecar_file"|"corrupt_sidecar"|
    "ambiguous_artifact"|"artifact_contract_error", "note"?, ...}. "no_jlens_artifact" is the
    ordinary NO-OP case (no J for this model at all -- e.g. every model today with no
    compact-eligible sidecar shipped yet); every other reason is a REFUSAL of something that exists
    but doesn't match or doesn't parse.
    """
    verify_file_checksum = False

    if model_identity is not None:
        from clozn.artifacts import contracts
        root = artifact_root or _default_artifact_root()
        try:
            found_dir = contracts.find_compatible_artifact(
                "jlens", model_identity, root, explicit_dir=jlens_dir)
        except contracts.ArtifactContractError as e:
            return {"available": False, "reason": "artifact_contract_error",
                    "jlens_dir": jlens_dir or root, "note": str(e)}
        if not found_dir:
            return {"available": False, "reason": "no_jlens_artifact", "jlens_dir": jlens_dir or root}
        d = found_dir
        try:
            manifest = concept_dir.load_jlens_manifest(d)
        except Exception as e:
            return {"available": False, "reason": "corrupt_manifest", "jlens_dir": d, "note": str(e)}
        model_block = _contract_model_block(manifest)
        manifest_model_name = model_block.get("source_id") if model_block else manifest.get("model")
        # already fully validated (manifest shape + every file's checksum) by
        # contracts.find_compatible_artifact above -- no further per-file check needed.

    elif model_sha256:
        found = _find_jlens_dir_by_sha256(str(model_sha256), jlens_dir=jlens_dir,
                                          artifact_root=artifact_root)
        if not found.get("ok"):
            return {"available": False, "reason": found.get("reason"), "note": found.get("note"),
                    "jlens_dir": jlens_dir}
        d = found["jlens_dir"]
        manifest = found["manifest"]
        model_block = _contract_model_block(manifest)
        manifest_model_name = model_block.get("source_id") if model_block else manifest.get("model")
        verify_file_checksum = True

    else:
        d = _resolve_jlens_dir(jlens_dir)
        manifest_path = os.path.join(d, "manifest.json")
        if not os.path.isfile(manifest_path):
            return {"available": False, "reason": "no_jlens_artifact", "jlens_dir": d}
        try:
            manifest = concept_dir.load_jlens_manifest(d)
        except Exception as e:
            return {"available": False, "reason": "corrupt_manifest", "jlens_dir": d, "note": str(e)}
        model_block = _contract_model_block(manifest)
        manifest_model_name = model_block.get("source_id") if model_block else manifest.get("model")
        if model_id and manifest_model_name and str(manifest_model_name) != str(model_id):
            return {"available": False, "reason": "wrong_model", "jlens_dir": d,
                    "expected_model": model_id, "artifact_model": manifest_model_name}

    manifest_d_model = manifest.get("d_model")
    if expected_d_model is not None and manifest_d_model is not None \
            and int(manifest_d_model) != int(expected_d_model):
        return {"available": False, "reason": "dim_mismatch", "jlens_dir": d,
                "expected_d_model": expected_d_model, "artifact_d_model": manifest_d_model}

    available_layers = [int(x) for x in (manifest.get("layers") or [])]
    if layer is not None:
        resolved_layer = int(layer)
    elif manifest.get("engine_default_tap_layer") is not None:
        resolved_layer = int(manifest["engine_default_tap_layer"])
    elif available_layers:
        resolved_layer = available_layers[0]
    else:
        resolved_layer = None
    if resolved_layer is None or resolved_layer not in available_layers:
        return {"available": False, "reason": "layer_not_fitted", "jlens_dir": d,
                "layer": resolved_layer, "available_layers": available_layers}

    # NOTE: only a SUCCESSFUL resolution (a fitted CompactJLens for a real, validated directory) is
    # cached -- a "not found"/refused outcome above is recomputed (a cheap directory/manifest
    # check, no SVD) on every call. Fine for today (every model ships with no compact-eligible
    # artifact at all, so this is a fast negative every time); worth adding a negative-result cache
    # too once a real per-request caller (e.g. EngineSteer on a hot chat loop) actually has a
    # matching artifact to find repeatedly.
    cache_key = (os.path.abspath(d), resolved_layer, int(k))
    if cache is not None and cache_key in cache:
        return {"available": True, "compact": cache[cache_key], "jlens_dir": d,
                "layer": resolved_layer, "model": manifest_model_name, "cached": True}

    if verify_file_checksum:
        checksum_error = _verify_sidecar_checksum(d, manifest, resolved_layer)
        if checksum_error:
            return {"available": False, "reason": "corrupt_sidecar", "jlens_dir": d,
                    "layer": resolved_layer, "note": checksum_error}

    try:
        J_by_layer = concept_dir.load_jlens_jacobians(d, layers=[resolved_layer])
        J = J_by_layer[resolved_layer]
    except (FileNotFoundError, ValueError) as e:
        return {"available": False, "reason": "missing_sidecar_file", "jlens_dir": d,
                "layer": resolved_layer, "note": str(e)}

    try:
        Vh, S = fit_compact_from_dense(J, k=k)
        compact = CompactJLens(Vh=Vh, S=S, layer=resolved_layer, d_model=J.shape[0],
                               model=manifest_model_name,
                               source={"jlens_dir": d, "fitted_from_dense": True})
    except JTransportError as e:
        return {"available": False, "reason": "corrupt_sidecar", "jlens_dir": d,
                "layer": resolved_layer, "note": str(e)}

    if cache is not None:
        cache[cache_key] = compact
    return {"available": True, "compact": compact, "jlens_dir": d, "layer": resolved_layer,
            "model": manifest_model_name, "cached": False}


# ============================================================================== the product entry point

def transport_direction(direction, *, jlens_dir: Optional[str] = None, layer: Optional[int] = None,
                        model_identity: Optional[dict] = None, model_sha256: Optional[str] = None,
                        model_id: Optional[str] = None, expected_d_model: Optional[int] = None,
                        artifact_root: Optional[str] = None, k: int = DEFAULT_K,
                        cache: Optional[dict] = None, norm: str = "preserve") -> dict:
    """THE explicit, optional J-transport step: run `direction` (a plain list/ndarray steer vector,
    already built by whatever mechanism -- diff-of-means, dir(c), an SAE column) through the
    active model's compact J when one is available, else return it UNCHANGED. NEVER raises (see
    the module docstring's HONESTY CONTRACT). `model_identity`/`model_sha256`/`model_id` are the
    three model-matching tiers resolve_compact_jlens documents (strongest first) -- pass whichever
    one your call site actually has.

    Returns:
      {"applied": True,  "vector": list[float], "layer":, "k":, "model":, "jlens_dir":, "norm":}
      {"applied": False, "vector": list[float] (SAME as input direction, unchanged), "reason": "...",
       "note"?: "..."}

    `applied` is the one field every caller MUST check -- this module never lets "J-transport was
    requested" be confused with "J-transport actually happened" (this codebase's house honesty
    rule: a transform must never be reported/assumed to have happened when it did not).
    """
    direction_list = direction.tolist() if isinstance(direction, np.ndarray) else [float(x) for x in direction]
    resolved = resolve_compact_jlens(jlens_dir=jlens_dir, layer=layer, model_identity=model_identity,
                                     model_sha256=model_sha256, model_id=model_id,
                                     expected_d_model=expected_d_model, artifact_root=artifact_root,
                                     k=k, cache=cache)
    if not resolved.get("available"):
        return {"applied": False, "vector": direction_list, "reason": resolved.get("reason"),
                "note": resolved.get("note"), "jlens_dir": resolved.get("jlens_dir")}
    try:
        jt = apply_compact_jlens(np.asarray(direction_list, dtype=np.float32),
                                 resolved["compact"], norm=norm)
    except JTransportError as e:
        return {"applied": False, "vector": direction_list, "reason": "dimension_mismatch",
                "note": str(e), "jlens_dir": resolved.get("jlens_dir")}
    return {"applied": True, "vector": jt.tolist(), "layer": resolved["layer"],
            "k": resolved["compact"].k, "model": resolved.get("model"),
            "jlens_dir": resolved.get("jlens_dir"), "norm": norm}
