"""Build and aggregate native Clozn engine release artifacts.

This module is deliberately stdlib-only.  A release runner uses ``stage`` to
collect the freshly built runtime files, then ``package`` to:

* execute ``clozn-server --version --json`` on the native runner;
* compare the binary identity with the Python/protocol/llama.cpp sources;
* add canonical BUILD-INFO.json and produce a deterministic ZIP; and
* write a small attestation beside the archive.

The aggregation runner uses ``manifest`` to re-measure every downloaded
archive and emit one deterministic ``clozn.engine-manifest.v1`` document.  It
requires an exact expected matrix, so a failed or missing matrix job cannot
silently produce a partial release.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse
import zipfile


RECORD_VERSION = "clozn.release-artifact.v1"
MANIFEST_VERSION = "clozn.engine-manifest.v1"
_BUILD_ID_RE = re.compile(r"^[A-Za-z0-9._+\-]+$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]*$")
_PROTOCOL_RE = re.compile(r"^[0-9]+\.[0-9]+$")
class ReleaseError(ValueError):
    """A fail-closed release contract violation."""


@dataclass(frozen=True, order=True)
class Cell:
    os: str
    arch: str
    backend: str
    cuda_major: int | None = None

    @classmethod
    def parse(cls, value: str) -> "Cell":
        parts = value.split("/")
        if len(parts) not in (3, 4):
            raise ReleaseError(
                f"matrix cell must be os/arch/backend[/cuda_major], got {value!r}"
            )
        os_name, arch, backend = parts[:3]
        cuda_major = None
        if len(parts) == 4:
            try:
                cuda_major = int(parts[3])
            except ValueError:
                raise ReleaseError(f"cuda_major must be an integer in {value!r}") from None
        if os_name not in {"linux", "windows", "macos"}:
            raise ReleaseError(f"unsupported release os {os_name!r}")
        if arch not in {"x86_64", "arm64"}:
            raise ReleaseError(f"unsupported release arch {arch!r}")
        if backend not in {"cpu", "cuda", "metal"}:
            raise ReleaseError(f"unsupported release backend {backend!r}")
        if backend == "cuda" and (cuda_major is None or cuda_major < 1):
            raise ReleaseError("cuda cells require a positive cuda_major")
        if backend != "cuda" and cuda_major is not None:
            raise ReleaseError(f"{backend} cells may not declare cuda_major")
        return cls(os_name, arch, backend, cuda_major)

    @classmethod
    def from_document(cls, document: dict) -> "Cell":
        suffix = f"/{document['cuda_major']}" if "cuda_major" in document else ""
        return cls.parse(
            f"{document.get('os')}/{document.get('arch')}/{document.get('backend')}{suffix}"
        )

    def label(self) -> str:
        suffix = f"/{self.cuda_major}" if self.cuda_major is not None else ""
        return f"{self.os}/{self.arch}/{self.backend}{suffix}"

    def platform_backend_key(self) -> tuple[str, str, str]:
        return self.os, self.arch, self.backend


@dataclass(frozen=True)
class SourceIdentity:
    version: str
    protocol_version: str
    llama_cpp_commit: str


def _literal_assignment(path: Path, name: str) -> object:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ReleaseError(f"cannot read release authority {path}: {error}") from None
    values = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            try:
                values.append(ast.literal_eval(node.value))
            except (TypeError, ValueError):
                raise ReleaseError(f"{path}: {name} must be a literal") from None
    if len(values) != 1:
        raise ReleaseError(f"{path}: expected exactly one literal assignment to {name}")
    return values[0]


def derive_source_identity(repo_root: Path) -> SourceIdentity:
    """Read the three existing release authorities without importing Clozn."""
    repo_root = repo_root.resolve()
    version = _literal_assignment(repo_root / "clozn" / "__init__.py", "__version__")
    protocol = _literal_assignment(repo_root / "clozn" / "protocol.py", "PROTOCOL_VERSION")
    commit = _literal_assignment(
        repo_root / "engine" / "core" / "third_party" / "bootstrap_llama.py", "COMMIT"
    )
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise ReleaseError(f"invalid Clozn version authority: {version!r}")
    if not isinstance(protocol, str) or not _PROTOCOL_RE.fullmatch(protocol):
        raise ReleaseError(f"invalid protocol version authority: {protocol!r}")
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise ReleaseError(f"invalid llama.cpp commit authority: {commit!r}")
    return SourceIdentity(version, protocol, commit)


def _canonical_json(document: object) -> str:
    return json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _read_json_object(path: Path, *, what: str) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseError(f"cannot read {what} {path}: {error}") from None
    if not isinstance(document, dict):
        raise ReleaseError(f"{what} {path} must contain a JSON object")
    return document


def read_engine_identity(engine: Path, *, timeout: float = 15.0) -> dict:
    """Execute the model-free build-info path and return its JSON object."""
    if not engine.is_file():
        raise ReleaseError(f"engine entrypoint does not exist: {engine}")
    try:
        result = subprocess.run(
            [str(engine.resolve()), "--version", "--json"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ReleaseError(f"could not execute engine build identity: {error}") from None
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise ReleaseError(
            f"engine build identity exited {result.returncode}: {detail[:500]}"
        )
    try:
        identity = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReleaseError(f"engine build identity was not valid JSON: {error}") from None
    if not isinstance(identity, dict):
        raise ReleaseError("engine build identity must be a JSON object")
    return identity


def validate_build_identity(
    identity: dict, source: SourceIdentity, cell: Cell, *, require_release_build: bool = True
) -> dict:
    """Validate and normalize the exact identity fields the manifest publishes."""
    expected_keys = {
        "engine_version",
        "build_id",
        "protocol_version",
        "backend",
        "llama_cpp_commit",
        "feature_flags",
    }
    missing = sorted(expected_keys - identity.keys())
    if missing:
        raise ReleaseError(f"engine build identity missing: {', '.join(missing)}")
    disagreements = []
    for field, expected in (
        ("engine_version", source.version),
        ("protocol_version", source.protocol_version),
        ("backend", cell.backend),
        ("llama_cpp_commit", source.llama_cpp_commit),
    ):
        if identity.get(field) != expected:
            disagreements.append(f"{field}={identity.get(field)!r}, expected {expected!r}")
    if disagreements:
        raise ReleaseError("engine build identity disagrees with release: " + "; ".join(disagreements))

    build_id = identity.get("build_id")
    if not isinstance(build_id, str) or not _BUILD_ID_RE.fullmatch(build_id):
        raise ReleaseError(f"engine build_id is malformed: {build_id!r}")
    if require_release_build and build_id == "development":
        raise ReleaseError("release artifacts may not use build_id 'development'")
    flags = identity.get("feature_flags")
    if (
        not isinstance(flags, dict)
        or not all(
            isinstance(key, str) and bool(key) and isinstance(value, bool)
            for key, value in flags.items()
        )
    ):
        raise ReleaseError("engine feature_flags must have non-empty names and boolean values")
    return {
        "engine_version": source.version,
        "build_id": build_id,
        "protocol_version": source.protocol_version,
        "backend": cell.backend,
        "llama_cpp_commit": source.llama_cpp_commit,
        "feature_flags": dict(sorted(flags.items())),
    }


def stage_runtime(bin_dir: Path, stage_root: Path, server_name: str, license_path: Path) -> None:
    """Copy one clean build's runtime directory into the release archive layout."""
    bin_dir, stage_root, license_path = bin_dir.resolve(), stage_root.resolve(), license_path.resolve()
    if not bin_dir.is_dir():
        raise ReleaseError(f"runtime bin directory does not exist: {bin_dir}")
    if not (bin_dir / server_name).is_file():
        raise ReleaseError(f"runtime entrypoint is missing: {bin_dir / server_name}")
    if not license_path.is_file():
        raise ReleaseError(f"license file does not exist: {license_path}")
    if stage_root.exists() and any(stage_root.iterdir()):
        raise ReleaseError(f"stage directory must be empty: {stage_root}")
    destination = stage_root / "bin"
    destination.mkdir(parents=True, exist_ok=True)
    for source in sorted(bin_dir.rglob("*")):
        if source.is_symlink():
            raise ReleaseError(f"runtime staging refuses symlinks: {source}")
        if source.is_file():
            relative = source.relative_to(bin_dir)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copy2(license_path, stage_root / "LICENSE")


def _file_measurement(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise ReleaseError(f"cannot measure archive {path}: {error}") from None
    if size < 1:
        raise ReleaseError(f"release archive is empty: {path}")
    return digest.hexdigest(), size


def _write_deterministic_zip(root: Path, output: Path, entrypoint: str) -> None:
    root, output = root.resolve(), output.resolve()
    entrypoint_path = Path(entrypoint)
    if entrypoint_path.is_absolute() or ".." in entrypoint_path.parts:
        raise ReleaseError(f"entrypoint must be an archive-relative safe path: {entrypoint!r}")
    if not (root / entrypoint_path).is_file():
        raise ReleaseError(f"entrypoint is not present in staging: {entrypoint}")
    if output.is_relative_to(root):
        raise ReleaseError("archive output may not be inside the staging tree")
    files = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReleaseError(f"archive refuses symlinks: {path}")
        if path.is_file():
            files.append(path)
    if not files:
        raise ReleaseError(f"staging tree is empty: {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
                relative = path.relative_to(root).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if relative == entrypoint_path.as_posix() else 0o644
                info.external_attr = (mode & 0xFFFF) << 16
                with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def package_artifact(
    *,
    repo_root: Path,
    stage_root: Path,
    archive_path: Path,
    record_path: Path,
    entrypoint: str,
    cell: Cell,
    url: str,
) -> dict:
    """Qualify, package, measure, and attest one native matrix cell."""
    source = derive_source_identity(repo_root)
    engine = stage_root / Path(entrypoint)
    identity = validate_build_identity(read_engine_identity(engine), source, cell)
    if not (stage_root / "LICENSE").is_file():
        raise ReleaseError("staging must include LICENSE")
    build_info = stage_root / "BUILD-INFO.json"
    build_info.write_text(_canonical_json(identity), encoding="utf-8", newline="\n")
    _write_deterministic_zip(stage_root, archive_path, entrypoint)
    sha256, size_bytes = _file_measurement(archive_path)

    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ReleaseError(f"release artifact URL must be absolute HTTPS: {url!r}")
    if Path(parsed_url.path).name != archive_path.name:
        raise ReleaseError("release artifact URL filename must match the measured archive")
    if archive_path.resolve().parent != record_path.resolve().parent:
        raise ReleaseError("archive and release attestation must be written to the same directory")

    artifact = {
        "os": cell.os,
        "arch": cell.arch,
        "backend": cell.backend,
        "url": url,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "entrypoint": entrypoint_path(entrypoint),
        "build_id": identity["build_id"],
        "llama_cpp_commit": identity["llama_cpp_commit"],
        "feature_flags": identity["feature_flags"],
    }
    if cell.cuda_major is not None:
        artifact["cuda_major"] = cell.cuda_major
    record = {
        "record_version": RECORD_VERSION,
        "status": "passed",
        "clozn_version": source.version,
        "protocol_version": source.protocol_version,
        "archive": archive_path.name,
        "artifact": artifact,
        "build_identity": identity,
    }
    record_path = record_path.resolve()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=record_path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical_json(record))
    try:
        os.replace(temporary, record_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return record


def entrypoint_path(value: str) -> str:
    if not isinstance(value, str):
        raise ReleaseError(f"entrypoint must be a string, got {type(value).__name__}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value or "\\" in value:
        raise ReleaseError(f"entrypoint must be a safe POSIX archive path: {value!r}")
    return path.as_posix()


def _validate_record(path: Path, document: dict, source: SourceIdentity) -> tuple[Cell, dict]:
    if document.get("record_version") != RECORD_VERSION:
        raise ReleaseError(f"{path}: unsupported release attestation version")
    if document.get("status") != "passed":
        raise ReleaseError(f"{path}: artifact did not pass native qualification")
    if document.get("clozn_version") != source.version:
        raise ReleaseError(f"{path}: attestation Clozn version disagrees with release")
    if document.get("protocol_version") != source.protocol_version:
        raise ReleaseError(f"{path}: attestation protocol version disagrees with release")
    artifact = document.get("artifact")
    identity = document.get("build_identity")
    if not isinstance(artifact, dict) or not isinstance(identity, dict):
        raise ReleaseError(f"{path}: attestation is missing artifact/build_identity objects")
    cell = Cell.from_document(artifact)
    normalized_identity = validate_build_identity(identity, source, cell)
    for field in ("build_id", "llama_cpp_commit", "feature_flags"):
        if artifact.get(field) != normalized_identity[field]:
            raise ReleaseError(f"{path}: artifact {field} disagrees with its engine build identity")
    entrypoint_path(artifact.get("entrypoint"))
    url = artifact.get("url")
    if not isinstance(url, str) or urlparse(url).scheme != "https":
        raise ReleaseError(f"{path}: artifact URL must use HTTPS")
    archive_name = document.get("archive")
    if (
        not isinstance(archive_name, str)
        or Path(archive_name).name != archive_name
        or Path(urlparse(url).path).name != archive_name
    ):
        raise ReleaseError(f"{path}: archive must be a local filename matching the artifact URL")
    archive = path.parent / archive_name
    measured_sha, measured_size = _file_measurement(archive)
    if artifact.get("sha256") != measured_sha or artifact.get("size_bytes") != measured_size:
        raise ReleaseError(f"{path}: archive measurement disagrees with attestation")
    return cell, dict(artifact)


def build_manifest(
    record_paths: list[Path], expected_cells: list[Cell], source: SourceIdentity
) -> dict:
    """Aggregate an exact matrix into a deterministic engine manifest."""
    if not record_paths:
        raise ReleaseError("no native artifact attestations were found")
    if not expected_cells:
        raise ReleaseError("the expected release matrix must be explicit")
    expected = set(expected_cells)
    if len(expected) != len(expected_cells):
        raise ReleaseError("expected release matrix contains duplicate cells")

    artifacts: list[tuple[Cell, dict]] = []
    seen_platform_backends: dict[tuple[str, str, str], Path] = {}
    seen_build_ids: dict[str, Path] = {}
    for path in sorted((item.resolve() for item in record_paths), key=str):
        document = _read_json_object(path, what="release attestation")
        cell, artifact = _validate_record(path, document, source)
        key = cell.platform_backend_key()
        if key in seen_platform_backends:
            raise ReleaseError(
                f"duplicate platform/backend {cell.label()} in {seen_platform_backends[key]} and {path}"
            )
        seen_platform_backends[key] = path
        build_id = artifact["build_id"]
        if build_id in seen_build_ids:
            raise ReleaseError(f"duplicate build_id {build_id!r} in release attestations")
        seen_build_ids[build_id] = path
        artifacts.append((cell, artifact))

    actual = {cell for cell, _ in artifacts}
    if actual != expected:
        missing = ", ".join(cell.label() for cell in sorted(expected - actual)) or "none"
        unexpected = ", ".join(cell.label() for cell in sorted(actual - expected)) or "none"
        raise ReleaseError(
            f"native matrix is incomplete or mixed (missing: {missing}; unexpected: {unexpected})"
        )
    return {
        "schema_version": MANIFEST_VERSION,
        "clozn_version": source.version,
        "protocol_version": source.protocol_version,
        "artifacts": [
            artifact
            for _, artifact in sorted(artifacts, key=lambda item: item[0])
        ],
    }


def write_manifest(
    output: Path, record_paths: list[Path], expected_cells: list[Cell], source: SourceIdentity
) -> dict:
    """Write only after the entire matrix validates, preserving any prior file on failure."""
    document = build_manifest(record_paths, expected_cells, source)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(_canonical_json(document))
    try:
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return document


def _repo_default() -> Path:
    return Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_repo_default())
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("version", help="print the one source-derived Clozn version")

    stage = commands.add_parser("stage", help="collect a native build into archive layout")
    stage.add_argument("--bin-dir", type=Path, required=True)
    stage.add_argument("--stage-root", type=Path, required=True)
    stage.add_argument("--server-name", required=True)
    stage.add_argument("--license", type=Path, required=True)

    package = commands.add_parser("package", help="qualify and package one native cell")
    package.add_argument("--stage-root", type=Path, required=True)
    package.add_argument("--archive", type=Path, required=True)
    package.add_argument("--record", type=Path, required=True)
    package.add_argument("--entrypoint", required=True)
    package.add_argument("--cell", type=Cell.parse, required=True)
    package.add_argument("--url", required=True)

    manifest = commands.add_parser("manifest", help="aggregate an exact native release matrix")
    manifest.add_argument("--records-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--expect", action="append", type=Cell.parse, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        source = derive_source_identity(args.repo_root)
        if args.command == "version":
            print(source.version)
        elif args.command == "stage":
            stage_runtime(args.bin_dir, args.stage_root, args.server_name, args.license)
        elif args.command == "package":
            package_artifact(
                repo_root=args.repo_root,
                stage_root=args.stage_root,
                archive_path=args.archive,
                record_path=args.record,
                entrypoint=args.entrypoint,
                cell=args.cell,
                url=args.url,
            )
        elif args.command == "manifest":
            paths = sorted(args.records_dir.rglob("*.release-artifact.json"))
            write_manifest(args.output, paths, args.expect, source)
        else:  # pragma: no cover - argparse guarantees the command set
            raise AssertionError(args.command)
    except ReleaseError as error:
        print(f"release error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
