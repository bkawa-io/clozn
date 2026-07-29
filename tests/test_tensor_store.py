"""test_tensor_store -- clozn/analysis/tensor_store.py: content-addressed BINARY float32 tensor blobs
(slice 3.2's storage primitive). Mirrors clozn.runs.store's own blob-store test discipline: every store
is redirected to tmp_path via monkeypatching the module's TENSOR_ROOT global (never the real
~/.clozn/tensors -- tests/conftest.py's tripwire only guards a fixed settings-file list, not this
store, so isolation here is this file's own responsibility, same as every other store's tests).
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from clozn.analysis import tensor_store as ts  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_tensor_root(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "TENSOR_ROOT", str(tmp_path / "tensors"))


# ==================================================================================== store + load round trip

def test_store_and_load_round_trip():
    values = [1.0, -2.5, 3.25, 0.0, 100.125]
    ref = ts.store_tensor(values, shape=[5], provenance={"model": "reference", "layer": 4, "position": 2})
    assert "write_failed" not in ref
    assert ref["dtype"] == "float32"
    assert ref["shape"] == [5]
    assert len(ref["sha256"]) == 64

    loaded = ts.load_tensor(ref)
    assert loaded["ok"] is True
    assert loaded["shape"] == [5]
    assert loaded["dtype"] == "float32"
    assert loaded["provenance"] == {"model": "reference", "layer": 4, "position": 2}
    for expected, actual in zip(values, loaded["values"]):
        assert abs(expected - actual) < 1e-5   # float32 round trip, not exact float64 equality


def test_store_without_provenance_load_omits_it():
    ref = ts.store_tensor([1.0, 2.0], shape=[2])
    loaded = ts.load_tensor(ref)
    assert loaded["ok"] is True
    assert "provenance" not in loaded


def test_identical_values_dedupe_to_the_same_digest():
    ref_a = ts.store_tensor([1.0, 2.0, 3.0], shape=[3], provenance={"layer": 1})
    ref_b = ts.store_tensor([1.0, 2.0, 3.0], shape=[3], provenance={"layer": 99})
    assert ref_a["sha256"] == ref_b["sha256"]
    # provenance is NOT part of the content address (see module docstring) -- whichever call landed
    # first on disk wins the sidecar; loading by either ref still returns the same verified bytes.
    loaded = ts.load_tensor(ref_b)
    assert loaded["ok"] is True
    assert loaded["values"] == pytest.approx([1.0, 2.0, 3.0], abs=1e-5)


def test_multidimensional_shape_round_trips():
    values = list(range(12))
    ref = ts.store_tensor(values, shape=[3, 4])
    loaded = ts.load_tensor(ref)
    assert loaded["shape"] == [3, 4]
    assert loaded["values"] == pytest.approx([float(v) for v in values])


# ============================================================================================= validation

def test_wrong_dtype_raises():
    with pytest.raises(ValueError, match="float32"):
        ts.store_tensor([1.0], shape=[1], dtype="float64")


def test_shape_values_mismatch_raises():
    with pytest.raises(ValueError, match="elements"):
        ts.store_tensor([1.0, 2.0, 3.0], shape=[4])


# ============================================================================================= honesty: never a silent empty

def test_load_missing_ref_is_unavailable_not_empty():
    out = ts.load_tensor({})
    assert "ok" not in out
    assert "unavailable" in out


def test_load_none_ref_is_unavailable():
    out = ts.load_tensor(None)
    assert "unavailable" in out


def test_load_malformed_sha256_is_unavailable():
    out = ts.load_tensor({"sha256": "not-a-real-digest"})
    assert "unavailable" in out


def test_load_missing_blob_file_is_unavailable():
    fake_digest = "0" * 64
    out = ts.load_tensor({"sha256": fake_digest})
    assert "unavailable" in out
    assert "missing" in out["unavailable"]
    assert out["sha256"] == fake_digest


def test_load_corrupt_blob_digest_mismatch_is_unavailable():
    ref = ts.store_tensor([1.0, 2.0], shape=[2])
    blob_path = ts._blob_path(ref["sha256"])
    with open(blob_path, "wb") as handle:
        handle.write(b"\x00\x00\x00\x00\x00\x00\x00\x00")   # different bytes -> digest mismatch
    out = ts.load_tensor(ref)
    assert "unavailable" in out
    assert "corrupt" in out["unavailable"]
    assert out["sha256"] == ref["sha256"]


def test_load_missing_sidecar_refuses_to_guess_shape():
    ref = ts.store_tensor([1.0, 2.0, 3.0, 4.0], shape=[2, 2])
    os.remove(ts._sidecar_path(ref["sha256"]))
    out = ts.load_tensor(ref)
    assert "unavailable" in out
    assert "sidecar" in out["unavailable"]
    assert "values" not in out


def test_load_reports_write_failure_from_the_ref_never_reads_disk():
    ref = {"sha256": "1" * 64, "write_failed": "OSError: disk full"}
    out = ts.load_tensor(ref)
    assert "unavailable" in out
    assert "disk full" in out["unavailable"]


def test_store_write_failure_is_recorded_on_the_ref_not_raised(monkeypatch):
    def _boom(path, data):
        raise OSError("simulated disk full")

    monkeypatch.setattr(ts, "_atomic_write_bytes", _boom)
    ref = ts.store_tensor([1.0, 2.0], shape=[2])
    assert "write_failed" in ref
    assert "disk full" in ref["write_failed"]

    loaded = ts.load_tensor(ref)
    assert "unavailable" in loaded
    assert "write failed" in loaded["unavailable"]


# ============================================================================================= digest exact

def test_digest_is_sha256_of_the_little_endian_float32_bytes():
    import array
    import hashlib
    import sys as _sys

    values = [1.5, -2.25, 3.0]
    ref = ts.store_tensor(values, shape=[3])
    arr = array.array("f", values)
    if _sys.byteorder != "little":
        arr.byteswap()
    expected = hashlib.sha256(arr.tobytes()).hexdigest()
    assert ref["sha256"] == expected
