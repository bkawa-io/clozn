"""Pure manifest parsing and artifact selection -- no I/O, no subprocess, no network.

parse_manifest() validates a fetched document against clozn.schemas' clozn.engine-manifest.v1 (Seam 2)
and rejects one whose protocol_version major this supervisor cannot speak, reusing clozn.protocol's own
worker-handshake check rather than duplicating the majors table. select_artifact() then picks the
narrowest compatible artifact for a clozn/setup/platform_detect.py platform profile. Both are ordinary
functions over plain dicts, deliberately: this is the part of feature 01 that tests exercise directly
against fixture manifests, with no download, no filesystem, no subprocess involved at all.
"""
from __future__ import annotations

from clozn import schemas
from clozn.protocol import check_worker_protocol
from clozn.setup.errors import ManifestError, SelectionError

SCHEMA_NAME = "clozn.engine-manifest.v1"

# clozn setup --backend accepts these plus "auto"; a manifest artifact's own "backend" field is always
# one of the three (schema-enforced), never "auto" -- that value only ever describes a REQUEST.
BACKEND_CHOICES = ("auto", "cpu", "cuda", "metal")


def parse_manifest(document: dict) -> dict:
    """Validate `document` against clozn.engine-manifest.v1 and its protocol_version against what this
    clozn can drive. Returns `document` unchanged on success (never mutated -- callers that want a
    trimmed/normalized view do that themselves). Raises ManifestError, never schemas.ValidationError/
    SchemaError directly, so every caller in this package catches exactly one exception type."""
    if not isinstance(document, dict):
        raise ManifestError(f"engine manifest must be a JSON object, got {type(document).__name__}")
    try:
        schemas.validate(document, SCHEMA_NAME)
    except (schemas.ValidationError, schemas.SchemaError) as error:
        raise ManifestError(f"engine manifest is invalid: {error}") from None

    protocol_version = document.get("protocol_version")
    ok, reason = check_worker_protocol(protocol_version)
    if not ok:
        raise ManifestError(f"engine manifest protocol_version is incompatible: {reason}")
    return document


def _matches_platform(artifact: dict, platform: dict) -> bool:
    return artifact.get("os") == platform.get("os") and artifact.get("arch") == platform.get("arch")


def _candidates_for_backend(artifacts: list, platform: dict, backend: str) -> list:
    return [a for a in artifacts if _matches_platform(a, platform) and a.get("backend") == backend]


def _best_cuda_candidate(candidates: list, cuda_major: "int | None") -> dict:
    """Deterministic tie-break among same-platform, same-backend ('cuda') artifacts that differ only by
    cuda_major: an exact match to the detected/requested driver major wins; otherwise the highest major
    that does not exceed it (an older runtime a newer driver can still load); otherwise -- no driver major
    known, or every artifact's major exceeds it -- the single highest major offered, on the theory that a
    manifest publishing more than one cuda_major without a detected driver to pick against should still
    resolve to SOMETHING rather than nothing, and the newest is the least surprising default."""
    by_major = sorted(candidates, key=lambda a: a.get("cuda_major") or 0)
    if cuda_major is not None:
        exact = [a for a in by_major if a.get("cuda_major") == cuda_major]
        if exact:
            return exact[0]
        not_newer = [a for a in by_major if (a.get("cuda_major") or 0) <= cuda_major]
        if not_newer:
            return not_newer[-1]
    return by_major[-1]


def select_artifact(manifest: dict, platform: dict, *, backend_pref: str = "auto") -> dict:
    """The single narrowest-compatible artifact from an already-parsed manifest for `platform`
    (clozn/setup/platform_detect.py's shape: os/arch/gpu_backend/cuda_major).

    backend_pref:
      "auto" (default) -- prefer platform['gpu_backend'] when the manifest offers it for this os/arch,
                            else fall back to "cpu".
      "cpu"/"cuda"/"metal" -- require exactly that backend; SelectionError if the manifest has none for
                            this os/arch.

    Raises SelectionError with the searched os/arch/backend spelled out -- never returns None, so a
    caller cannot forget to check for "no match" (roadmap rule 3: no silent fallback).
    """
    if backend_pref not in BACKEND_CHOICES:
        raise SelectionError(f"--backend must be one of {BACKEND_CHOICES}, got {backend_pref!r}")
    artifacts = manifest.get("artifacts") or []
    osname, arch = platform.get("os"), platform.get("arch")

    if backend_pref != "auto":
        candidates = _candidates_for_backend(artifacts, platform, backend_pref)
        if not candidates:
            raise SelectionError(
                f"no {backend_pref} artifact for {osname}/{arch} in this manifest "
                f"({len(artifacts)} artifact(s) total)")
    else:
        gpu_backend = platform.get("gpu_backend")
        candidates = _candidates_for_backend(artifacts, platform, gpu_backend) if gpu_backend else []
        if not candidates:
            candidates = _candidates_for_backend(artifacts, platform, "cpu")
        if not candidates:
            raise SelectionError(
                f"no compatible artifact for {osname}/{arch} in this manifest "
                f"({len(artifacts)} artifact(s) total; tried "
                f"{gpu_backend or 'no GPU backend detected'} then cpu)")

    if len(candidates) == 1:
        return dict(candidates[0])
    if candidates[0].get("backend") == "cuda":
        return dict(_best_cuda_candidate(candidates, platform.get("cuda_major")))
    # More than one same-platform/same-backend artifact with no documented tie-break (e.g. two cpu
    # builds) is a manifest authoring error, not something to silently pick one of -- fail loud.
    raise SelectionError(
        f"manifest is ambiguous: {len(candidates)} {candidates[0].get('backend')} artifacts for "
        f"{osname}/{arch} with no distinguishing field this selector understands")


def install_key(clozn_version: str, artifact: dict) -> str:
    """The registry key one resolved artifact install occupies: '<version>/<os>-<arch>-<backend>[N]',
    e.g. '1.0.0/windows-x86_64-cuda12'. Deterministic and collision-free for the fields the manifest
    schema actually constrains (a manifest may never legally publish two artifacts with the same
    os/arch/backend/cuda_major -- select_artifact already refuses that ambiguity above)."""
    backend = artifact.get("backend")
    tag = f"{backend}{artifact['cuda_major']}" if backend == "cuda" and artifact.get("cuda_major") else backend
    return f"{clozn_version}/{artifact.get('os')}-{artifact.get('arch')}-{tag}"
