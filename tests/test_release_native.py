import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.release.native_release import (
    Cell,
    ReleaseError,
    SourceIdentity,
    build_manifest,
    validate_build_identity,
    write_manifest,
    _write_deterministic_zip,
)


SOURCE = SourceIdentity(
    version="1.2.3",
    protocol_version="1.0",
    llama_cpp_commit="a" * 40,
)


def _identity(cell: Cell, build_id: str) -> dict:
    return {
        "engine_version": SOURCE.version,
        "build_id": build_id,
        "protocol_version": SOURCE.protocol_version,
        "backend": cell.backend,
        "llama_cpp_commit": SOURCE.llama_cpp_commit,
        "feature_flags": {"lora": True, "sae": False, "whitebox": True},
    }


def _record(tmp_path: Path, cell: Cell, name: str, *, build_id: str | None = None) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    archive = directory / f"clozn-engine-{name}.zip"
    payload = f"archive:{name}".encode()
    archive.write_bytes(payload)
    identity = _identity(cell, build_id or f"release-{name}")
    artifact = {
        "os": cell.os,
        "arch": cell.arch,
        "backend": cell.backend,
        "url": f"https://example.invalid/v1.2.3/{archive.name}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "entrypoint": "bin/clozn-server.exe" if cell.os == "windows" else "bin/clozn-server",
        "build_id": identity["build_id"],
        "llama_cpp_commit": identity["llama_cpp_commit"],
        "feature_flags": identity["feature_flags"],
    }
    if cell.cuda_major is not None:
        artifact["cuda_major"] = cell.cuda_major
    document = {
        "record_version": "clozn.release-artifact.v1",
        "status": "passed",
        "clozn_version": SOURCE.version,
        "protocol_version": SOURCE.protocol_version,
        "archive": archive.name,
        "artifact": artifact,
        "build_identity": identity,
    }
    record = directory / f"{name}.release-artifact.json"
    record.write_text(json.dumps(document), encoding="utf-8")
    return record


def test_manifest_is_deterministic_across_record_order(tmp_path):
    linux = Cell.parse("linux/x86_64/cpu")
    windows = Cell.parse("windows/x86_64/cpu")
    first = _record(tmp_path, windows, "windows")
    second = _record(tmp_path, linux, "linux")

    one = build_manifest([first, second], [linux, windows], SOURCE)
    two = build_manifest([second, first], [windows, linux], SOURCE)

    assert one == two
    assert [item["os"] for item in one["artifacts"]] == ["linux", "windows"]
    assert json.dumps(one, sort_keys=True) == json.dumps(two, sort_keys=True)


def test_duplicate_platform_backend_is_refused_even_for_cuda_variants(tmp_path):
    cuda_12 = Cell.parse("linux/x86_64/cuda/12")
    cuda_13 = Cell.parse("linux/x86_64/cuda/13")
    first = _record(tmp_path, cuda_12, "cuda12")
    second = _record(tmp_path, cuda_13, "cuda13")

    with pytest.raises(ReleaseError, match="duplicate platform/backend"):
        build_manifest([first, second], [cuda_12, cuda_13], SOURCE)


def test_build_identity_disagreement_is_refused():
    linux = Cell.parse("linux/x86_64/cpu")
    identity = _identity(linux, "release-linux")
    identity["engine_version"] = "9.9.9"
    identity["backend"] = "cuda"

    with pytest.raises(ReleaseError, match="engine build identity disagrees"):
        validate_build_identity(identity, SOURCE, linux)


def test_partial_matrix_failure_preserves_existing_manifest(tmp_path):
    linux = Cell.parse("linux/x86_64/cpu")
    windows = Cell.parse("windows/x86_64/cpu")
    record = _record(tmp_path, linux, "linux")
    output = tmp_path / "clozn-engine-manifest.json"
    output.write_text("previous complete release\n", encoding="utf-8")

    with pytest.raises(ReleaseError, match="missing: windows/x86_64/cpu"):
        write_manifest(output, [record], [linux, windows], SOURCE)

    assert output.read_text(encoding="utf-8") == "previous complete release\n"


def test_measured_archive_tampering_is_refused(tmp_path):
    linux = Cell.parse("linux/x86_64/cpu")
    record = _record(tmp_path, linux, "linux")
    document = json.loads(record.read_text(encoding="utf-8"))
    (record.parent / document["archive"]).write_bytes(b"tampered")

    with pytest.raises(ReleaseError, match="archive measurement disagrees"):
        build_manifest([record], [linux], SOURCE)


def test_artifact_identity_fields_must_match_attested_engine(tmp_path):
    linux = Cell.parse("linux/x86_64/cpu")
    record = _record(tmp_path, linux, "linux")
    document = json.loads(record.read_text(encoding="utf-8"))
    document = copy.deepcopy(document)
    document["artifact"]["feature_flags"]["lora"] = False
    record.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ReleaseError, match="feature_flags disagrees"):
        build_manifest([record], [linux], SOURCE)


def test_archive_packaging_is_deterministic_and_normalizes_metadata(tmp_path):
    stage = tmp_path / "stage"
    (stage / "bin").mkdir(parents=True)
    server = stage / "bin" / "clozn-server"
    server.write_bytes(b"native executable")
    (stage / "LICENSE").write_text("license\n", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    _write_deterministic_zip(stage, first, "bin/clozn-server")
    server.touch()
    _write_deterministic_zip(stage, second, "bin/clozn-server")

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["LICENSE", "bin/clozn-server"]
        assert archive.getinfo("bin/clozn-server").date_time == (1980, 1, 1, 0, 0, 0)
        assert (archive.getinfo("bin/clozn-server").external_attr >> 16) & 0o111


def test_feature_flag_names_must_be_nonempty():
    linux = Cell.parse("linux/x86_64/cpu")
    identity = _identity(linux, "release-linux")
    identity["feature_flags"][""] = True

    with pytest.raises(ReleaseError, match="non-empty names"):
        validate_build_identity(identity, SOURCE, linux)
