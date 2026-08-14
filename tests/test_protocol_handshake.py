"""Golden-fixture guard for the worker<->supervisor handshake.

The protocol version + capability vocabulary live in THREE places by necessity (a C++ worker, a Python
supervisor, and a shared JSON contract Studio can read). This test is the single tripwire that fails the
moment any of the three drifts -- so the handshake can never silently disagree across the language gap.
"""
import json
import re
from pathlib import Path

import pytest

from clozn import protocol

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "protocol" / "fixtures" / "handshake.json"
CPP_HEADER = ROOT / "engine" / "core" / "serve" / "server_shared.hpp"
CPP_HEALTH = ROOT / "engine" / "core" / "serve" / "server_main.cpp"
CPP_CMAKE = ROOT / "engine" / "core" / "CMakeLists.txt"


@pytest.fixture(scope="module")
def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_python_constant_matches_fixture(fixture):
    assert protocol.PROTOCOL_VERSION == fixture["protocol_version"]
    assert set(protocol.SUPPORTED_MAJORS) == set(fixture["supported_majors"])


def test_cpp_header_version_matches_fixture(fixture):
    """The C++ worker pins the SAME version string -- extracted straight from the header source so a
    rebuild is never needed for this guard to catch a drift."""
    src = CPP_HEADER.read_text(encoding="utf-8")
    m = re.search(r'PROTOCOL_VERSION\s*=\s*"([^"]+)"', src)
    assert m, "PROTOCOL_VERSION constant not found in server_shared.hpp"
    assert m.group(1) == fixture["protocol_version"]


def test_stream_envelope_declared(fixture):
    """Every native-stream JSON frame is stamped with these keys (req = request id, seq = monotonic
    per-request counter). The engine stamps them via StreamEnvelope; a live stream is checked end to end
    by the product smoke -- here we pin the contract so a consumer knows the two keys to expect."""
    assert fixture["stream_frame_envelope"] == ["req", "seq"]
    src = CPP_HEALTH.read_text(encoding="utf-8")
    # the envelope is wired into the streaming blocks (guards against the frames silently going raw again)
    assert "StreamEnvelope env{id, write}" in src


def test_cpp_health_capability_keys_match_fixture(fixture):
    """The /health `capabilities` object advertises EXACTLY the fixture's capability set -- extracted from
    the `json capabilities{ ... };` literal so C++ can't add or drop a flag without updating the contract."""
    src = CPP_HEALTH.read_text(encoding="utf-8")
    block = re.search(r"json capabilities\{(.*?)\};", src, re.DOTALL)
    assert block, "capabilities literal not found in the /health handler"
    keys = set(re.findall(r'\{"(\w+)",', block.group(1)))
    assert keys == set(fixture["capabilities"])


def test_cpp_health_required_fields_match_fixture(fixture):
    """Every required handshake field is present in the C++ /health document, including the opaque
    process generation needed to reject checkpoint references after a worker restart."""
    src = CPP_HEALTH.read_text(encoding="utf-8")
    block = re.search(r'json h\{\{"status", "ok"\},(.*?)\};', src, re.DOTALL)
    assert block, "/health response literal not found in server_main.cpp"
    keys = {"status", *re.findall(r'\{"(\w+)",', block.group(1))}
    assert set(fixture["health_required_fields"]).issubset(keys)
    assert "worker_generation_id" in fixture["health_required_fields"]


def test_cpp_health_exposes_additive_engine_build_identity():
    """Direct one-worker serving must expose the same truthful build identity used by checkpoints.

    The fields are intentionally not part of ``health_required_fields``: an older worker may omit
    them and the Python gateway must then fail closed for exact replay rather than fabricate a build.
    """
    src = CPP_HEALTH.read_text(encoding="utf-8")
    block = re.search(r'json h\{\{"status", "ok"\},(.*?)\};', src, re.DOTALL)
    assert block, "/health response literal not found in server_main.cpp"
    keys = set(re.findall(r'\{"(\w+)",', block.group(1)))
    assert {"engine_version", "build_id", "backend", "llama_cpp_commit"}.issubset(keys)
    assert "json build = engine_build_info();" in src


def test_cpp_health_exposes_context_batch_and_memory_regime():
    src = CPP_HEALTH.read_text(encoding="utf-8")
    block = re.search(r'json h\{\{"status", "ok"\},(.*?)\};', src, re.DOTALL)
    assert block, "/health response literal not found in server_main.cpp"
    keys = set(re.findall(r'\{"(\w+)",', block.group(1)))
    assert {"n_ctx", "n_batch", "n_ubatch", "memory"}.issubset(keys)
    assert "cp.n_batch = n_batch_;" in (ROOT / "engine/core/src/model_ggml.cpp").read_text(encoding="utf-8")
    assert "cp.n_ubatch = n_ubatch_;" in (ROOT / "engine/core/src/model_ggml.cpp").read_text(encoding="utf-8")


def test_checkpoint_reference_contract_is_generation_scoped(fixture):
    checkpoint_ref = fixture["checkpoint_reference"]
    assert checkpoint_ref["fields"] == ["checkpoint_id", "worker_generation_id"]
    assert checkpoint_ref["checkpoint_id_semantics"] == "opaque_generation_scoped"
    assert checkpoint_ref["worker_generation_precondition"] == \
        "optional_for_in_process_compatibility"

    source = (ROOT / "engine" / "core" / "serve" / "checkpoint_store.hpp").read_text(
        encoding="utf-8")
    assert '"ckpt-" + worker_generation_id_' in source
    assert "std::deque<std::string> insertion_order_" in source
    assert "records_.erase(insertion_order_.front())" in source


def test_cpp_model_free_build_info_contract_is_embedded():
    """Setup qualification must be possible before a GGUF is parsed or loaded."""
    src = CPP_HEALTH.read_text(encoding="utf-8")
    required = {
        "engine_version", "build_id", "protocol_version", "backend",
        "llama_cpp_commit", "feature_flags",
    }
    block = re.search(r"static json engine_build_info\(\) \{(.*?)\n\}", src, re.DOTALL)
    assert block, "engine_build_info() not found in server_main.cpp"
    assert required.issubset(set(re.findall(r'\{"(\w+)",', block.group(1))))
    version_branch = src.index('std::string(argv[1]) == "--version"')
    usage_or_model = src.index("if (argc < 2)")
    assert version_branch < usage_or_model, "build-info path must run before model argument/loading"

    cmake = CPP_CMAKE.read_text(encoding="utf-8")
    for define in (
        "CLOZN_ENGINE_VERSION", "CLOZN_ENGINE_BUILD_ID",
        "CLOZN_ENGINE_BACKEND", "CLOZN_LLAMA_CPP_COMMIT",
    ):
        assert f'{define}="${{{define if define != "CLOZN_ENGINE_BACKEND" else "_clozn_engine_backend"}}}"' \
            in cmake


def test_supervisor_accepts_same_major():
    ok, reason = protocol.check_worker_protocol(protocol.PROTOCOL_VERSION)
    assert ok and reason == ""
    # A newer MINOR on the same major is still compatible (additive).
    major = protocol.parse_major(protocol.PROTOCOL_VERSION)
    ok2, _ = protocol.check_worker_protocol(f"{major}.99")
    assert ok2


def test_supervisor_refuses_incompatible_or_missing():
    for bad in (None, "", "0.9", "2.0", "999.0", "not-a-version", 1.0, {"v": 1}):
        ok, reason = protocol.check_worker_protocol(bad)
        assert not ok, f"expected refusal for {bad!r}"
        assert reason, "a refusal must carry an actionable reason"


def test_parse_major():
    assert protocol.parse_major("1.0") == 1
    assert protocol.parse_major("12.34") == 12
    assert protocol.parse_major("3") == 3
    # only strings with a numeric MAJOR slot parse; everything else is None
    for bad in (None, "", "x.0", "v1.0", "  .0", 5, 1.0, [], {"v": 1}):
        assert protocol.parse_major(bad) is None
