"""safe_extract() -- path-traversal- and symlink-safe zip/tar extraction.

Every archive `clozn setup` extracts already passed a sha256 check against the manifest (see
clozn/setup/transport.py) before this module ever sees it -- but "the bytes are what the manifest
promised" says nothing about whether the ARCHIVE'S CONTENTS are safe to write to disk. This module never
trusts a member path or type, mirroring the `.relative_to(root)` idiom clozn/artifacts/contracts.py
already uses for lab-artifact payloads (validate_artifact_manifest), extended to cover extraction itself
-- that module validates files already on disk; this one is what puts them there. The repo has been bitten
by path traversal before; nothing here is optional.
"""
from __future__ import annotations

import os
import tarfile
import zipfile
from pathlib import Path

from clozn.setup.errors import ArchiveError

# A generous ceiling on total EXTRACTED bytes, independent of any manifest-declared (compressed) size --
# no real clozn-server build has ever approached this. It exists to bound a hostile or corrupt archive,
# not to be a tight per-artifact budget.
_MAX_TOTAL_EXTRACTED_BYTES = 8 * 1024 * 1024 * 1024   # 8 GiB
_MAX_MEMBER_COUNT = 20_000


def _safe_relative_path(name: str, root: Path) -> Path:
    """The resolved on-disk path for archive member `name`, guaranteed to be inside `root`. Raises
    ArchiveError on an empty/root name, an absolute or drive-qualified path, a '..' component, or a
    resolution that lands outside `root` even after all of the above pass (defense in depth: the
    explicit checks above should already catch everything, but a path that survives them and still
    escapes on resolve() is refused too, not trusted because it "looked" fine)."""
    if not name or name.strip().strip("/\\") == "":
        raise ArchiveError(f"archive contains an empty or root member name: {name!r}")
    # Archive tooling on any platform may store either separator; a member name crafted with backslashes
    # must be judged as a path on EVERY extraction platform, not just the one that treats '\\' specially.
    candidate = Path(name.replace("\\", "/"))
    if candidate.is_absolute() or candidate.drive or candidate.root:
        raise ArchiveError(f"archive member has an absolute or rooted path: {name!r}")
    if ".." in candidate.parts:
        raise ArchiveError(f"archive member contains a '..' path-traversal component: {name!r}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        raise ArchiveError(f"archive member escapes the extraction root: {name!r}") from None
    return resolved


def _guard_member_count(count: int) -> None:
    if count > _MAX_MEMBER_COUNT:
        raise ArchiveError(f"archive has {count} members, over the {_MAX_MEMBER_COUNT} sanity cap")


def _guard_total_size(total: int) -> None:
    if total > _MAX_TOTAL_EXTRACTED_BYTES:
        raise ArchiveError(
            f"archive's extracted size exceeds the {_MAX_TOTAL_EXTRACTED_BYTES} byte sanity cap")


def _copy_stream(source, target, chunk_size: int = 1 << 20) -> None:
    while True:
        chunk = source.read(chunk_size)
        if not chunk:
            return
        target.write(chunk)


def _extract_zip(archive_path: str, root: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        _guard_member_count(len(infos))
        total = 0
        for info in infos:
            total += info.file_size
            _guard_total_size(total)
            # The upper 16 bits of external_attr carry the POSIX st_mode when the archive was built on
            # a POSIX system (the common case for a clozn-engine release archive); S_IFLNK is the
            # symlink bit. Zip has no portable symlink representation otherwise, so this is the one
            # signal available -- and it is checked BEFORE any bytes are written, never trusted after.
            posix_mode = (info.external_attr >> 16) & 0o170000
            if posix_mode == 0o120000:
                raise ArchiveError(f"archive member is a symlink, which is never allowed: {info.filename}")
            destination = _safe_relative_path(info.filename, root)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(destination, "wb") as target:
                _copy_stream(source, target)
            _apply_unix_permissions(destination, info.external_attr >> 16)


def _extract_tar(archive_path: str, root: Path, *, mode: str) -> None:
    with tarfile.open(archive_path, mode) as archive:
        members = archive.getmembers()
        _guard_member_count(len(members))
        total = 0
        for member in members:
            if member.issym() or member.islnk():
                raise ArchiveError(f"archive member is a symlink, which is never allowed: {member.name}")
            if member.isdev():
                raise ArchiveError(
                    f"archive member is a device/fifo/socket node, which is never allowed: {member.name}")
            total += max(0, member.size)
            _guard_total_size(total)
            destination = _safe_relative_path(member.name, root)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ArchiveError(f"archive member is not a regular file or directory: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ArchiveError(f"archive member could not be read: {member.name}")
            with source, open(destination, "wb") as target:
                _copy_stream(source, target)
            _apply_unix_permissions(destination, member.mode)


def _apply_unix_permissions(path: Path, mode_bits: int) -> None:
    """Best-effort: carry the archive's own permission bits (in particular +x) onto POSIX. A silent
    no-op on Windows and on any chmod failure -- clozn never relies on this succeeding; a .exe launches
    by extension/PE header, not a permission bit, and a POSIX build that loses +x here still fails
    loudly at qualify time (clozn/setup/install.py's qualify_entrypoint), not silently."""
    perm = mode_bits & 0o777
    if not perm:
        return
    try:
        os.chmod(path, perm)
    except OSError:
        pass


def safe_extract(archive_path: str, dest_dir: str) -> None:
    """Extract `archive_path` (.zip, .tar.gz, .tgz, or .tar -- inferred from its filename) into
    `dest_dir`, which must already exist. Every member is validated BEFORE a single byte is written: no
    absolute/rooted path, no '..' traversal, no symlink or device node, and a sanity cap on member count
    and total extracted bytes. Raises ArchiveError on any violation, a corrupt/truncated archive, or an
    unrecognized format. Callers extract into a throwaway staging directory (never a real install
    target) precisely so a failure here never touches anything real -- see clozn/setup/install.py."""
    root = Path(dest_dir)
    if not root.is_dir():
        raise ArchiveError(f"extraction root does not exist or is not a directory: {dest_dir}")
    lower = archive_path.lower()
    try:
        if lower.endswith(".zip"):
            _extract_zip(archive_path, root)
        elif lower.endswith(".tar.gz") or lower.endswith(".tgz"):
            _extract_tar(archive_path, root, mode="r:gz")
        elif lower.endswith(".tar"):
            _extract_tar(archive_path, root, mode="r:")
        else:
            raise ArchiveError(
                f"unrecognized archive format (expected .zip/.tar.gz/.tgz/.tar): {archive_path}")
    except (zipfile.BadZipFile, tarfile.TarError) as error:
        raise ArchiveError(f"archive is corrupt or truncated: {error}") from None
