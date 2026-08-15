"""Engine-side chat templating seam (T0.1): the engine applies the LOADED MODEL'S OWN chat template
(GGUF tokenizer.chat_template via llama_chat_apply_template) so clozn formats prompts per-model -- Qwen
ChatML, Llama-3 headers, Gemma turns, ... -- instead of a hardcoded Qwen string. Two model-free layers:

  * clozn_engine.EngineClient.apply_template -- the thin SDK wrapper (POST /apply_template body shape,
    add_assistant default, return value).
  * clozn_server._engine_tmpl -- the module helper the EngineSubstrate generation paths call in place of
    the old _qwen_tmpl; it just delegates to engine.apply_template (errors propagate, no silent Qwen
    fallback).

The C++ /apply_template route itself (engine/core/serve/clozn_server.cpp) and the actual per-model
formatting (Qwen ChatML byte-identical to the old _qwen_tmpl; Llama-3 <|begin_of_text|><|start_header_id|>)
are proved LIVE against a running clozn-server on both models -- not exercised by this offline suite.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))

from clozn_engine import EngineClient          # noqa: E402
from clozn.server import app as cs           # noqa: E402


# ==================================================================================== EngineClient.apply_template

def test_apply_template_sends_messages_and_add_assistant(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {"prompt": "P"})
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    out = ec.apply_template(msgs)
    assert seen["path"] == "/apply_template"
    assert seen["body"] == {"messages": msgs, "add_assistant": True}
    assert out == "P"


def test_apply_template_add_assistant_false(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {"prompt": ""})
    ec.apply_template([{"role": "user", "content": "hi"}], add_assistant=False)
    assert seen["body"]["add_assistant"] is False


def test_apply_template_returns_the_prompt_field(monkeypatch):
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post",
                        lambda path, body: {"prompt": "<|im_start|>user\nhi<|im_end|>\n", "template_source": "model"})
    assert ec.apply_template([{"role": "user", "content": "hi"}]) == "<|im_start|>user\nhi<|im_end|>\n"


def test_apply_template_info_returns_exact_worker_token_count(monkeypatch):
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: {
        "prompt": "rendered", "prompt_tokens": 17, "template_source": "model",
    })
    assert ec.apply_template_info([{"role": "user", "content": "hi"}]) == {
        "prompt": "rendered", "prompt_tokens": 17,
    }


def test_apply_template_info_accepts_an_older_worker_without_count(monkeypatch):
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: {"prompt": "legacy rendered"})
    assert ec.apply_template_info([{"role": "user", "content": "hi"}]) == {
        "prompt": "legacy rendered",
    }
    assert ec.apply_template([{"role": "user", "content": "hi"}]) == "legacy rendered"


def test_apply_template_info_rejects_a_malformed_worker_count(monkeypatch):
    from clozn_engine import EngineError

    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: {
        "prompt": "rendered", "prompt_tokens": True,
    })
    import pytest
    with pytest.raises(EngineError, match="prompt_tokens"):
        ec.apply_template_info([{"role": "user", "content": "hi"}])


def test_apply_template_info_can_opt_in_to_exact_token_ids(monkeypatch):
    ec = EngineClient(port=1)
    seen = {}
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {
        "prompt": "rendered", "prompt_tokens": 3, "prompt_token_ids": [11, 22, 33],
    })
    assert ec.apply_template_info(
        [{"role": "user", "content": "hi"}], include_token_ids=True) == {
        "prompt": "rendered", "prompt_tokens": 3, "prompt_token_ids": [11, 22, 33],
    }
    assert seen["body"]["include_token_ids"] is True


# ==================================================================================== _engine_tmpl helper

class _RecordingEngine:
    def __init__(self, rendered="RENDERED"):
        self.rendered = rendered
        self.calls = []

    def apply_template(self, messages, add_assistant=True):
        self.calls.append([dict(m) for m in messages])
        return self.rendered


def test_engine_tmpl_delegates_to_engine_apply_template():
    eng = _RecordingEngine(rendered="THE PROMPT")
    msgs = [{"role": "user", "content": "hello"}]
    assert cs._engine_tmpl(eng, msgs) == "THE PROMPT"
    assert eng.calls == [msgs]


def test_engine_tmpl_can_return_exact_template_token_usage():
    class InfoEngine:
        def apply_template_info(self, messages, add_assistant=True):
            return {"prompt": "THE PROMPT", "prompt_tokens": 9}

    usage = {}
    assert cs._engine_tmpl(InfoEngine(), [{"role": "user", "content": "hello"}], usage) == "THE PROMPT"
    assert usage == {"prompt_tokens": 9}


def test_engine_tmpl_propagates_errors_no_silent_fallback():
    """A model with no embedded template (or an engine too old to expose /apply_template) must SURFACE --
    _engine_tmpl deliberately does not fall back to the hardcoded Qwen _qwen_tmpl (that silent
    mis-format is exactly the bug this seam removes)."""
    class _BoomEngine:
        def apply_template(self, messages, add_assistant=True):
            raise RuntimeError("model has no embedded chat template")

    import pytest
    with pytest.raises(RuntimeError):
        cs._engine_tmpl(_BoomEngine(), [{"role": "user", "content": "hi"}])
