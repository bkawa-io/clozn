"""safe_extract() must write only what a legitimate clozn-engine release archive would contain, and
reject everything a hostile or corrupt one might. Every fixture archive here is built in-memory/on the
fly with zipfile/tarfile -- no checked-in binary blobs, no network."""
from __future__ import annotations

import os
import tarfile
import zipfile

import pytest

from clozn.setup.archive import safe_extract
from clozn.setup.errors import ArchiveError


def _make_zip(path, members: dict, *, symlink: "tuple[str, str] | None" = None):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
        if symlink:
            link_name, target = symlink
            info = zipfile.ZipInfo(link_name)
            info.external_attr = (0o120777 << 16)   # S_IFLNK | 0777
            zf.writestr(info, target)


def _make_targz(path, members: dict, *, symlink: "tuple[str, str] | None" = None, executables=()):
    with tarfile.open(path, "w:gz") as tf:
        for name, content in members.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            if name in executables:
                info.mode = 0o755
            import io
            tf.addfile(info, io.BytesIO(data))
        if symlink:
            link_name, target = symlink
            info = tarfile.TarInfo(name=link_name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tf.addfile(info)


# --------------------------------------------------------------------------------------------- happy path

def test_safe_extract_zip_writes_expected_files(tmp_path):
    archive = tmp_path / "engine.zip"
    _make_zip(str(archive), {"bin/clozn-server.exe": "fake binary", "README.txt": "hi"})
    dest = tmp_path / "out"
    dest.mkdir()
    safe_extract(str(archive), str(dest))
    assert (dest / "bin" / "clozn-server.exe").read_text() == "fake binary"
    assert (dest / "README.txt").read_text() == "hi"


def test_safe_extract_targz_writes_expected_files_and_preserves_exec_bit(tmp_path):
    archive = tmp_path / "engine.tar.gz"
    _make_targz(str(archive), {"bin/clozn-server": "fake binary"}, executables=("bin/clozn-server",))
    dest = tmp_path / "out"
    dest.mkdir()
    safe_extract(str(archive), str(dest))
    extracted = dest / "bin" / "clozn-server"
    assert extracted.read_text() == "fake binary"
    if os.name != "nt":
        assert extracted.stat().st_mode & 0o100


# -------------------------------------------------------------------------------------------- path safety

def test_safe_extract_zip_rejects_dotdot_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    _make_zip(str(archive), {"../../evil.txt": "pwned"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArchiveError, match="traversal|escapes"):
        safe_extract(str(archive), str(dest))
    assert not (tmp_path.parent / "evil.txt").exists()


def test_safe_extract_zip_rejects_absolute_path_member(tmp_path):
    archive = tmp_path / "evil.zip"
    _make_zip(str(archive), {"/etc/passwd": "pwned"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArchiveError):
        safe_extract(str(archive), str(dest))


def test_safe_extract_zip_rejects_backslash_traversal_on_any_platform(tmp_path):
    """A member name using Windows-style separators must be judged as a path everywhere it might be
    extracted, not just on Windows."""
    archive = tmp_path / "evil.zip"
    _make_zip(str(archive), {"..\\..\\evil.txt": "pwned"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArchiveError):
        safe_extract(str(archive), str(dest))


def test_safe_extract_tar_rejects_dotdot_traversal(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _make_targz(str(archive), {"../evil.txt": "pwned"})
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArchiveError):
        safe_extract(str(archive), str(dest))


def test_safe_extract_zip_rejects_a_symlink_member(tmp_path):
    archive = tmp_path / "evil.zip"
    _make_zip(str(archive), {"bin/clozn-server.exe": "real"}, symlink=("bin/evil-link", "/etc/passwd"))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArchiveError, match="symlink"):
        safe_extract(str(archive), str(dest))


def test_safe_extract_tar_rejects_a_symlink_member(tmp_path):
    archive = tmp_path / "evil.tar.gz"
    _make_targz(str(archive), {"bin/clozn-server": "real"}, symlink=("bin/evil-link", "/etc/passwd"))
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArchiveError, match="symlink"):
        safe_extract(str(archive), str(dest))


def test_safe_extract_rejects_an_unrecognized_format(tmp_path):
    archive = tmp_path / "engine.rar"
    archive.write_bytes(b"not really a rar")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArchiveError, match="unrecognized"):
        safe_extract(str(archive), str(dest))


def test_safe_extract_rejects_a_corrupt_archive(tmp_path):
    archive = tmp_path / "engine.zip"
    archive.write_bytes(b"this is not a zip file at all, just garbage bytes")
    dest = tmp_path / "out"
    dest.mkdir()
    with pytest.raises(ArchiveError, match="corrupt"):
        safe_extract(str(archive), str(dest))


def test_safe_extract_requires_an_existing_destination_directory(tmp_path):
    archive = tmp_path / "engine.zip"
    _make_zip(str(archive), {"a.txt": "hi"})
    with pytest.raises(ArchiveError, match="does not exist"):
        safe_extract(str(archive), str(tmp_path / "missing"))
