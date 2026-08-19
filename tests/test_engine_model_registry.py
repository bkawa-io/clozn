"""Engine model registry (T0.2): the EngineSubstrate reflects the ACTUALLY-LOADED GGUF instead of a
hardcoded "Qwen2.5-7B" id/assumption. The only Qwen-specific coupling the engine substrate carried was
the tone-dial steer TAP LAYER (mid-depth: 14 for Qwen-7B's 28 layers); a tiny family registry keys that
-- plus a friendly model_id for run_meta -- off the loaded model's /health filename, with a sensible
default for any unrecognized GGUF. Everything else the engine already calibrates per-model server-side.

Model-free: a fake engine reports a /health model path; no C++ server, no GPU. The LIVE cross-model proof
that the taps actually work on Llama is T0.3 (separate, manual).
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))

from clozn.server import app as cs          # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402




# ==================================================================================== family derivation

def test_model_family_from_name_qwen():
    assert cs._model_family_from_name("Qwen2.5-7B-Instruct-Q4_K_M.gguf") == "qwen2.5-7b"
    assert cs._model_family_from_name("Qwen2.5-0.5B-Instruct-Q4_K_M.gguf") == "qwen2.5-0.5b"


def test_model_family_from_name_llama():
    assert cs._model_family_from_name("Llama-3.2-1B-Instruct-Q4_K_M.gguf") == "llama-3.2-1b"
    assert cs._model_family_from_name(
        r"C:\Users\x\.clozn\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf") == "llama-3.2-3b"
    assert cs._model_family_from_name(
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf") == "llama-3.1-8b"


def test_model_family_from_name_wave_one_models():
    assert cs._model_family_from_name("Qwen3.5-9B-Q4_K_M.gguf") == "qwen3.5-9b"
    assert cs._model_family_from_name("gemma-4-E4B-it-Q4_K_M.gguf") == "gemma4-e4b"
    assert cs._model_family_from_name(
        "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf") == "ministral3-3b"


def test_model_family_from_name_unknown_is_none():
    assert cs._model_family_from_name("mistral-7b-instruct-q4_k_m.gguf") is None
    assert cs._model_family_from_name("") is None
    assert cs._model_family_from_name(None) is None


def test_engine_model_info_known_family():
    fam, info = cs._engine_model_info("/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    assert fam == "llama-3.2-1b"
    assert info["model_id"] == "meta-llama/Llama-3.2-1B-Instruct"

    fam, info = cs._engine_model_info("/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    assert fam == "llama-3.1-8b"
    assert info == {"model_id": "meta-llama/Llama-3.1-8B-Instruct"}


def test_engine_model_info_unknown_returns_default():
    fam, info = cs._engine_model_info("mistral-7b-instruct-q4_k_m.gguf")
    assert fam is None
    assert info == {"model_id": None}


# ==================================================================================== derive-from-engine at construction

class _HealthSteerEngine:
    """A fake engine exposing just /health (with a model path) + the apply_template/complete surface the
    EngineSubstrate constructor's downstream might touch. Never hits a real socket."""

    def __init__(self, model):
        self.base = "http://127.0.0.1:1"
        self.timeout = 0.2
        self._model = model

    def health(self):
        return {"status": "ok", "model": self._model, "mode": "autoregressive",
                "n_ctx": 4096, "device": "cuda", "gpu_layers": 99}

    def apply_template(self, messages, add_assistant=True):
        return cs._qwen_tmpl(messages)

    def complete(self, prompt, **params):
        return {"choices": [{"text": "ok", "finish_reason": "stop"}]}



@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


def _make_sub(monkeypatch, model):
    eng = _HealthSteerEngine(model)
    monkeypatch.setattr(cs, "ENGINE", eng)
    return cs.EngineSubstrate()





def test_run_meta_reports_family_and_model_id(iso, monkeypatch):
    sub = _make_sub(monkeypatch, "/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
    meta = sub.run_meta()
    assert meta["family"] == "llama-3.2-1b"
    assert meta["model_id"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert meta["model_file"] == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"


def test_run_meta_omits_family_for_unknown_model(iso, monkeypatch):
    sub = _make_sub(monkeypatch, "/models/mistral-7b-instruct-q4_k_m.gguf")
    meta = sub.run_meta()
    assert "family" not in meta        # unrecognized -> omitted, not guessed
    assert "model_id" not in meta
    assert meta["model_file"] == "mistral-7b-instruct-q4_k_m.gguf"   # the raw file still recorded
