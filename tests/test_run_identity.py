"""test_run_identity -- roadmap S4.3: immutable reproduction identity on every run.

Model-free. Covers clozn/runs/identity.py's three entry points (model_sha256's mandatory cache,
template_fingerprint's fixed-conversation fingerprint, runtime_identity's omit-don't-fake assembly),
the EngineSubstrate.identity_meta()/run_meta() wiring in clozn/server/substrates.py (prefers the
engine's own /health model_sha256, falls back to hashing only when that's absent, computed once per
process), and clozn.runs.store.record()'s new `identity` field.
"""
from __future__ import annotations

import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

import clozn.runs.identity as identity   # noqa: E402
import clozn.runs.store as runlog        # noqa: E402
from clozn.server import app as cs       # noqa: E402


@pytest.fixture
def iso_cache(tmp_path, monkeypatch):
    """Point identity's on-disk hash cache at a throwaway file so tests never read/write the real
    ~/.clozn/cache/model_hashes.json."""
    monkeypatch.setattr(identity, "_CACHE_PATH", str(tmp_path / "model_hashes.json"))
    return tmp_path


@pytest.fixture
def store(tmp_path):
    """Redirect the run store to a temp dir for the duration of one test (mirrors test_runlog.py)."""
    original = runlog.RUNS_DIR
    runlog.RUNS_DIR = str(tmp_path / "runs")
    try:
        yield runlog
    finally:
        runlog.RUNS_DIR = original


# ============================================================================================ model_sha256

def test_model_sha256_matches_hashlib_for_a_small_file(tmp_path, iso_cache):
    path = tmp_path / "tiny.bin"
    data = b"clozn reproduction receipt fixture bytes"
    path.write_bytes(data)
    assert identity.model_sha256(str(path)) == hashlib.sha256(data).hexdigest()


def test_model_sha256_cache_hit_never_reopens_the_file(tmp_path, monkeypatch, iso_cache):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"weights-v1" * 1000)

    calls = {"n": 0}
    real_sha256_file = identity.sha256_file

    def counting(path_arg, chunk_size=identity._CHUNK_SIZE):
        calls["n"] += 1
        return real_sha256_file(path_arg, chunk_size=chunk_size)

    monkeypatch.setattr(identity, "sha256_file", counting)

    first = identity.model_sha256(str(path))
    second = identity.model_sha256(str(path))

    assert calls["n"] == 1                                  # second call hit the cache, never reopened the file
    assert first == second == hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_sha256_cache_invalidates_on_mtime_change(tmp_path, monkeypatch, iso_cache):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"version-1")

    calls = {"n": 0}
    real_sha256_file = identity.sha256_file

    def counting(path_arg, chunk_size=identity._CHUNK_SIZE):
        calls["n"] += 1
        return real_sha256_file(path_arg, chunk_size=chunk_size)

    monkeypatch.setattr(identity, "sha256_file", counting)

    first = identity.model_sha256(str(path))
    assert calls["n"] == 1

    # Same bytes, but the mtime moves forward -- e.g. a re-export that rewrites identical content.
    # mtime is part of the cache key on purpose, so this must still be treated as a new file version.
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    second = identity.model_sha256(str(path))
    assert calls["n"] == 2                                  # mtime changed -> re-hashed, not served from cache
    assert second == first                                  # content unchanged -> same digest


def test_model_sha256_missing_file_returns_none(tmp_path, iso_cache):
    assert identity.model_sha256(str(tmp_path / "does-not-exist.gguf")) is None


def test_model_sha256_empty_or_missing_path_returns_none(iso_cache):
    assert identity.model_sha256(None) is None
    assert identity.model_sha256("") is None


# ======================================================================================= template_fingerprint

def _chatml(messages):
    return "".join(f"<|{m['role']}|>{m['content']}" for m in messages)


def _llama_style(messages):
    return "".join(f"[{m['role'].upper()}] {m['content']}\n" for m in messages)


def test_template_fingerprint_is_stable_for_a_fixed_template():
    a = identity.template_fingerprint(_chatml)
    b = identity.template_fingerprint(_chatml)
    assert a == b
    assert isinstance(a, str) and len(a) == 16
    assert all(c in "0123456789abcdef" for c in a)


def test_template_fingerprint_differs_for_a_different_template():
    assert identity.template_fingerprint(_chatml) != identity.template_fingerprint(_llama_style)


def test_template_fingerprint_uses_the_fixed_canonical_conversation():
    seen = []

    def capture(messages):
        seen.append(messages)
        return "rendered"

    identity.template_fingerprint(capture)
    assert seen == [list(identity.CANONICAL_CONVERSATION)]


def test_template_fingerprint_none_function_returns_none():
    assert identity.template_fingerprint(None) is None


def test_template_fingerprint_swallows_a_raising_function():
    def boom(messages):
        raise RuntimeError("no template loaded")

    assert identity.template_fingerprint(boom) is None


def test_template_fingerprint_swallows_a_non_string_return():
    assert identity.template_fingerprint(lambda messages: {"not": "a string"}) is None
    assert identity.template_fingerprint(lambda messages: "") is None


# =========================================================================================== runtime_identity

def test_runtime_identity_omits_unavailable_keys():
    out = identity.runtime_identity()
    assert "model_path" not in out
    assert "model_sha256" not in out
    assert "model_size_bytes" not in out
    assert "template_fingerprint" not in out
    assert "engine_build" not in out
    assert "captured_at" in out
    # clozn_version IS honestly measurable here (the installed package) -- present, not fabricated.
    assert out.get("clozn_version")


def test_runtime_identity_prefers_the_hint_over_recomputing(tmp_path, monkeypatch, iso_cache):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"weights")

    def boom(_path):
        raise AssertionError("model_sha256() must not be called when a hint is supplied")

    monkeypatch.setattr(identity, "model_sha256", boom)

    out = identity.runtime_identity(model_path=str(path), model_sha256_hint="already-known-digest")
    assert out["model_sha256"] == "already-known-digest"
    assert out["model_path"] == os.path.abspath(str(path))
    assert out["model_size_bytes"] == len(b"weights")


def test_runtime_identity_falls_back_to_hashing_without_a_hint(tmp_path, iso_cache):
    path = tmp_path / "model.gguf"
    data = b"weights-v2"
    path.write_bytes(data)
    out = identity.runtime_identity(model_path=str(path))
    assert out["model_sha256"] == hashlib.sha256(data).hexdigest()


def test_runtime_identity_includes_template_fingerprint_engine_build_and_version():
    out = identity.runtime_identity(
        apply_template_fn=lambda messages: "rendered-prompt",
        engine_health={"engine_build": "clozn-server-2026.07"},
        clozn_version="9.9.9",
    )
    assert out["template_fingerprint"] == hashlib.sha256(b"rendered-prompt").hexdigest()[:16]
    assert out["engine_build"] == "clozn-server-2026.07"
    assert out["clozn_version"] == "9.9.9"


def test_runtime_identity_never_invents_an_engine_build():
    """/health today carries no build-hash field at all (see identity.py's _ENGINE_BUILD_KEYS
    comment) -- an engine_health dict that doesn't carry any candidate key must omit engine_build,
    not guess at one."""
    out = identity.runtime_identity(engine_health={"model": "x.gguf", "n_ctx": 4096})
    assert "engine_build" not in out


def test_runtime_identity_never_raises_on_a_hostile_apply_template_fn():
    out = identity.runtime_identity(apply_template_fn=lambda messages: 1 / 0)
    assert "template_fingerprint" not in out
    assert "captured_at" in out


# =========================================================================== EngineSubstrate wiring (BACKLOG 4.3)

class _FakeEngineForIdentity:
    """Stand-in engine exposing /health + /apply_template, mirroring test_engine_substrate.py's
    _HealthEngine but letting each test control model_sha256 presence and template rendering."""

    def __init__(self, model, extra_health=None, template_fn=None):
        self.base = "http://127.0.0.1:1"
        self.timeout = 0.2
        self._h = {"status": "ok", "model": model, "mode": "autoregressive"}
        if extra_health:
            self._h.update(extra_health)
        self._template_fn = template_fn

    def health(self):
        return dict(self._h)

    def apply_template(self, messages, add_assistant=True):
        if self._template_fn is None:
            raise RuntimeError("no template loaded")
        return self._template_fn(messages)


def test_engine_substrate_identity_meta_prefers_healths_own_model_sha256(iso_cache):
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = _FakeEngineForIdentity(
        "/models/qwen.gguf",
        extra_health={"model_sha256": "deadbeef" * 8},
        template_fn=lambda msgs: "<im>" + msgs[-1]["content"],
    )
    ident = sub.identity_meta()
    assert ident["model_sha256"] == "deadbeef" * 8
    assert ident["model_path"] == os.path.abspath("/models/qwen.gguf")
    assert "template_fingerprint" in ident
    assert "clozn_version" in ident and "captured_at" in ident


def test_engine_substrate_identity_meta_is_cached_after_first_call(iso_cache):
    sub = object.__new__(cs.EngineSubstrate)
    calls = {"n": 0}

    class _CountEngine:
        base = "x"

        def health(self):
            calls["n"] += 1
            return {"model": "m.gguf", "model_sha256": "abc123"}

    sub.engine = _CountEngine()
    first = sub.identity_meta()
    second = sub.identity_meta()
    assert first == second
    assert calls["n"] == 1                                  # /health fetched once, then cached for the process


def test_engine_substrate_identity_meta_falls_back_to_hashing_when_health_omits_sha256(tmp_path, iso_cache):
    model_path = tmp_path / "local.gguf"
    data = b"a-tiny-fake-gguf"
    model_path.write_bytes(data)

    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = _FakeEngineForIdentity(str(model_path))    # health carries no model_sha256 at all
    ident = sub.identity_meta()
    assert ident["model_sha256"] == hashlib.sha256(data).hexdigest()


def test_engine_substrate_identity_meta_never_raises_on_bad_health():
    sub = object.__new__(cs.EngineSubstrate)

    class _BoomEngine:
        base = "x"

        def health(self):
            raise RuntimeError("no engine")

    sub.engine = _BoomEngine()
    ident = sub.identity_meta()
    assert "model_sha256" not in ident
    assert "captured_at" in ident


# ==================================================================================== the run-record writer

def test_record_persists_the_identity_block_when_provided(store):
    ident = {"model_sha256": "a" * 64, "template_fingerprint": "0123456789abcdef",
             "clozn_version": "0.1.0", "captured_at": "2026-07-20T00:00:00+00:00"}
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                       identity=ident)
    rec = store.get_run(rid)
    assert rec["identity"] == ident


def test_record_defaults_identity_to_an_empty_dict_when_omitted(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey")
    assert store.get_run(rid)["identity"] == {}
