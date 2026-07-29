"""test_pair_compatibility -- clozn/analysis/pair_compatibility.py (`clozn.pair-compatibility.v1`): the
shared, versioned model-pair compatibility contract several features consume (diff-model, Experiments,
mechanistic diff, Studio Compare, causal bisect) instead of each reimplementing preflight checks.

Model-free / GPU-free throughout, mirroring tests/test_diff_model.py's own discipline: the relocated
tokenizer/template probes are exercised against small fake engines exposing only `.score`/
`.apply_template` (never a real engine); the structural (hash-based, no-engine) side is exercised against
plain synthetic identity dicts shaped like `clozn.artifacts.contracts.gguf_identity(...)`'s return value,
never a real GGUF file; `assess_gguf_pair` is exercised with `gguf_identity` itself monkeypatched out, so
no file ever touches disk.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

import clozn.analysis.pair_compatibility as pc  # noqa: E402
from clozn import schemas  # noqa: E402
from clozn.cli.commands import quant_check as qc  # noqa: E402


# ==================================================================================== fake engines
# Mirrors tests/test_diff_model.py's own fakes -- these are duplicated on purpose (not imported from that
# test module) so this file can stand alone as the regression net for the relocated functions.

class TokenizeEngine:
    def __init__(self, tokenize_fn):
        self.tokenize_fn = tokenize_fn

    def score(self, prompt=None, **kw):
        return {"tokens": self.tokenize_fn(kw.get("continuation", ""))}


def _word_tokens(text):
    return [{"id": 1000 + i, "piece": w} for i, w in enumerate(text.split())]


class TemplateEngine:
    def __init__(self, template):
        self.template = template

    def apply_template(self, messages):
        return self.template


def _identity(*, architecture=None, hidden_size=None, layer_count=None, vocab_size=None,
             tokenizer_sha256=None, chat_template_sha256=None, sha256=None, filename=None):
    """A synthetic `gguf_identity(...)`-shaped dict -- only the keys pair_compatibility actually reads,
    with a value simply absent (never None) when a case wants a field "not measured". Real gguf_identity
    always returns every key it computes; tests exercise the "unknown" path by omitting a key entirely,
    which is exactly what a caller who disabled include_file_hash or hit a partial header would produce."""
    out = {}
    if architecture is not None:
        out["architecture"] = architecture
    if hidden_size is not None:
        out["hidden_size"] = hidden_size
    if layer_count is not None:
        out["layer_count"] = layer_count
    if vocab_size is not None:
        out["vocab_size"] = vocab_size
    if tokenizer_sha256 is not None:
        out["tokenizer_sha256"] = tokenizer_sha256
    if chat_template_sha256 is not None:
        out["chat_template_sha256"] = chat_template_sha256
    if sha256 is not None:
        out["sha256"] = sha256
    if filename is not None:
        out["filename"] = filename
    return out


_FULL_A = dict(architecture="qwen2", hidden_size=3584, layer_count=28, vocab_size=152064,
              tokenizer_sha256="tok-aaa", chat_template_sha256="tpl-aaa",
              sha256="a" * 64, filename="a.gguf")
_FULL_B_SAME = dict(architecture="qwen2", hidden_size=3584, layer_count=28, vocab_size=152064,
                    tokenizer_sha256="tok-aaa", chat_template_sha256="tpl-aaa",
                    sha256="b" * 64, filename="b.gguf")


# ==================================================================================== check_tokenizer_compat
# (relocated from diff_model.py -- same behavior, exercised again here as this module's own regression net)

def test_check_tokenizer_compat_identical_is_compatible():
    sub_a = qc._EngineScoreSub(TokenizeEngine(_word_tokens))
    sub_b = qc._EngineScoreSub(TokenizeEngine(_word_tokens))
    out = pc.check_tokenizer_compat(sub_a, sub_b)
    assert out["compatible"] is True
    assert len(out["probes"]) == len(pc.TOKENIZER_PROBES)


def test_check_tokenizer_compat_different_ids_is_incompatible():
    def _shifted(text):
        return [{"id": 2000 + i, "piece": w} for i, w in enumerate(text.split())]

    sub_a = qc._EngineScoreSub(TokenizeEngine(_word_tokens))
    sub_b = qc._EngineScoreSub(TokenizeEngine(_shifted))
    out = pc.check_tokenizer_compat(sub_a, sub_b)
    assert out["compatible"] is False
    assert any(not p["ids_match"] for p in out["probes"])


def test_check_tokenizer_compat_never_raises_when_engine_blows_up():
    class BoomEngine:
        def score(self, prompt=None, **kw):
            raise RuntimeError("boom")

    sub_a = qc._EngineScoreSub(BoomEngine())
    sub_b = qc._EngineScoreSub(TokenizeEngine(_word_tokens))
    out = pc.check_tokenizer_compat(sub_a, sub_b)
    assert out["compatible"] is False


def test_tokenizer_refusal_message_names_the_failed_probes_and_suggests_same_family():
    out = {"compatible": False, "probes": [{"probe": "code_snippet", "ids_match": False, "pieces_match": False}]}
    msg = pc.tokenizer_refusal_message(out)
    assert "code_snippet" in msg
    assert "meaningless" in msg
    assert "same-tokenizer-family" in msg or "same tokenizer" in msg.lower()


# ==================================================================================== check_template_match

def test_check_template_match_identical():
    sub_a = qc._EngineScoreSub(TemplateEngine("SAME"))
    sub_b = qc._EngineScoreSub(TemplateEngine("SAME"))
    out = pc.check_template_match(sub_a, sub_b)
    assert out["match"] is True


def test_check_template_match_different():
    sub_a = qc._EngineScoreSub(TemplateEngine("A"))
    sub_b = qc._EngineScoreSub(TemplateEngine("B"))
    out = pc.check_template_match(sub_a, sub_b)
    assert out["match"] is False


def test_check_template_match_never_raises_when_apply_template_blows_up():
    class BoomEngine:
        def apply_template(self, messages):
            raise RuntimeError("no embedded template")

    sub_a = qc._EngineScoreSub(BoomEngine())
    sub_b = qc._EngineScoreSub(TemplateEngine("X"))
    out = pc.check_template_match(sub_a, sub_b)
    assert out["match"] is False


# ==================================================================================== writable_layer_range

def test_writable_layer_range_normal():
    assert pc.writable_layer_range(28) == {"min": 1, "max_exclusive": 28}


def test_writable_layer_range_none_for_missing_or_non_positive():
    assert pc.writable_layer_range(None) is None
    assert pc.writable_layer_range(0) is None
    assert pc.writable_layer_range(-1) is None
    assert pc.writable_layer_range("28") is None


def test_writable_layer_range_excludes_bool():
    # isinstance(True, int) is True in Python -- writable_layer_range must not treat a stray bool as a
    # layer count (mirrors clozn/schemas/_validator.py's own int-vs-bool discipline).
    assert pc.writable_layer_range(True) is None


# ==================================================================================== assess(): dimensions

def test_assess_all_dimensions_same_is_fully_compatible():
    doc = pc.assess(_FULL_A, _FULL_B_SAME, label_a="ref", label_b="cand")
    assert doc["schema_version"] == pc.SCHEMA_VERSION
    assert doc["tokenizer"] == {"state": "exact", "method": "hash"}
    assert doc["template"] == {"state": "same", "method": "hash"}
    assert doc["architecture"]["state"] == "same"
    assert doc["layer_count"]["state"] == "same"
    assert doc["hidden_size"]["state"] == "same"
    assert doc["vocab_size"]["state"] == "same"
    assert doc["verdict"]["overall"] == "compatible"
    assert doc["verdict"]["reasons"] == []
    assert doc["verdict"]["operations"]["per_token_comparison"]["permitted"] is True
    assert doc["verdict"]["operations"]["residual_transplant"]["permitted"] is True
    assert doc["writable_layers"]["model_a"] == {"min": 1, "max_exclusive": 28}
    assert doc["writable_layers"]["model_b"] == {"min": 1, "max_exclusive": 28}
    # schemas.validate already ran inside assess(); re-running here pins the contract explicitly.
    schemas.validate(doc)


def test_assess_omits_label_when_not_given():
    doc = pc.assess(_FULL_A, _FULL_B_SAME)
    assert "label" not in doc["model_a"]
    assert doc["model_a"]["filename"] == "a.gguf"
    assert doc["model_a"]["sha256"] == "a" * 64


def test_assess_tokenizer_differs_by_hash_blocks_only_per_token_comparison():
    """The split-gating property the module exists for: a tokenizer mismatch refuses per-token
    comparison but leaves residual transplant alone (it only cares about hidden_size)."""
    identity_b = dict(_FULL_B_SAME, tokenizer_sha256="tok-different")
    doc = pc.assess(_FULL_A, identity_b)
    assert doc["tokenizer"]["state"] == "differs"
    assert doc["verdict"]["overall"] == "incompatible"
    ops = doc["verdict"]["operations"]
    assert ops["per_token_comparison"]["permitted"] is False
    assert ops["residual_transplant"]["permitted"] is True   # hidden_size still matches
    assert "tokenizer" in ops["per_token_comparison"]["reason"].lower()


def test_assess_hidden_size_differs_blocks_only_residual_transplant():
    """The mirror image: hidden_size differing blocks the transplant but per-token comparison (which only
    needs the tokenizer) is unaffected."""
    identity_b = dict(_FULL_B_SAME, hidden_size=4096)
    doc = pc.assess(_FULL_A, identity_b)
    assert doc["hidden_size"]["state"] == "differs"
    assert doc["verdict"]["overall"] == "compatible_with_caveats"
    ops = doc["verdict"]["operations"]
    assert ops["per_token_comparison"]["permitted"] is True
    assert ops["residual_transplant"]["permitted"] is False
    assert "hidden_size" in ops["residual_transplant"]["reason"]
    assert any("hidden_size" in r for r in doc["verdict"]["reasons"])


def test_assess_missing_fields_are_unknown_and_values_omitted_not_nulled():
    identity_a = _identity(architecture="qwen2")   # everything else genuinely unmeasured
    identity_b = _identity(architecture="qwen2")
    doc = pc.assess(identity_a, identity_b)
    assert doc["tokenizer"] == {"state": "unknown", "method": "unknown"}
    assert doc["template"] == {"state": "unknown", "method": "unknown"}
    assert doc["hidden_size"] == {"state": "unknown"}   # no value_a/value_b key at all -- never null
    assert "value_a" not in doc["hidden_size"]
    assert "value_b" not in doc["hidden_size"]
    assert doc["writable_layers"] == {}   # neither side's layer_count known
    ops = doc["verdict"]["operations"]
    assert ops["per_token_comparison"]["permitted"] is False
    assert ops["residual_transplant"]["permitted"] is False
    assert doc["verdict"]["overall"] == "compatible_with_caveats"   # unknown is not a confirmed refusal


def test_assess_writable_layers_reports_only_the_known_side():
    identity_a = dict(_FULL_A)
    identity_b = _identity()   # layer_count unknown on B
    doc = pc.assess(identity_a, identity_b)
    assert doc["writable_layers"] == {"model_a": {"min": 1, "max_exclusive": 28}}


def test_assess_tokenizer_probe_overrides_hash_and_records_failed_probes():
    probe = {"compatible": False, "probes": [
        {"probe": "plain_english", "ids_match": False, "pieces_match": False},
        {"probe": "digits_arithmetic", "ids_match": True, "pieces_match": True},
    ]}
    # hashes agree (would say "exact" on their own) -- the probe result must win.
    doc = pc.assess(_FULL_A, _FULL_B_SAME, tokenizer_compat=probe)
    assert doc["tokenizer"]["state"] == "differs"
    assert doc["tokenizer"]["method"] == "probe"
    assert doc["tokenizer"]["failed_probes"] == ["plain_english"]


def test_assess_template_probe_overrides_hash():
    doc = pc.assess(_FULL_A, _FULL_B_SAME, template_match=False)
    assert doc["template"]["state"] == "differs"
    assert doc["template"]["method"] == "probe"
    assert doc["template"]["caveat"]   # generic caveat auto-attached


def test_assess_template_caller_supplied_policy_and_caveat_pass_through():
    doc = pc.assess(_FULL_A, _FULL_B_SAME, template_match=False,
                    template_policy="reference", template_caveat=pc.TEMPLATE_DIFFER_REFERENCE_CAVEAT)
    assert doc["template"]["policy_applied"] == "reference"
    assert doc["template"]["caveat"] == pc.TEMPLATE_DIFFER_REFERENCE_CAVEAT


def test_assess_template_no_caveat_when_it_matches():
    doc = pc.assess(_FULL_A, _FULL_B_SAME, template_match=True)
    assert doc["template"]["state"] == "same"
    assert "caveat" not in doc["template"]


def test_assess_generated_at_default_and_override():
    doc_default = pc.assess(_FULL_A, _FULL_B_SAME)
    assert doc_default["generated_at"].endswith("Z")
    doc_fixed = pc.assess(_FULL_A, _FULL_B_SAME, generated_at="2026-01-01T00:00:00Z")
    assert doc_fixed["generated_at"] == "2026-01-01T00:00:00Z"


def test_assess_validate_false_skips_schema_check(monkeypatch):
    calls = []
    monkeypatch.setattr(pc.schemas, "validate", lambda doc: calls.append(doc))
    pc.assess(_FULL_A, _FULL_B_SAME, validate=False)
    assert calls == []
    pc.assess(_FULL_A, _FULL_B_SAME, validate=True)
    assert len(calls) == 1


# ==================================================================================== helper predicates

def test_may_per_token_compare_and_may_residual_transplant():
    doc = pc.assess(_FULL_A, _FULL_B_SAME)
    assert pc.may_per_token_compare(doc) is True
    assert pc.may_residual_transplant(doc) is True

    bad_tok = pc.assess(_FULL_A, dict(_FULL_B_SAME, tokenizer_sha256="different"))
    assert pc.may_per_token_compare(bad_tok) is False

    bad_hidden = pc.assess(_FULL_A, dict(_FULL_B_SAME, hidden_size=1))
    assert pc.may_residual_transplant(bad_hidden) is False


def test_predicate_helpers_never_raise_on_a_malformed_report():
    assert pc.may_per_token_compare({}) is False
    assert pc.may_residual_transplant({}) is False


# ==================================================================================== assess_gguf_pair

def test_assess_gguf_pair_delegates_to_gguf_identity_and_assess(monkeypatch):
    """Model-free: gguf_identity itself is monkeypatched out, so this never touches a real file."""
    calls = []

    def fake_gguf_identity(path, *, include_file_hash=True):
        calls.append((path, include_file_hash))
        return dict(_FULL_A) if "a" in path else dict(_FULL_B_SAME)

    monkeypatch.setattr(pc, "gguf_identity", fake_gguf_identity)
    doc = pc.assess_gguf_pair("path/to/a.gguf", "path/to/b.gguf", label_a="ref", label_b="cand")

    assert calls == [("path/to/a.gguf", True), ("path/to/b.gguf", True)]
    assert doc["model_a"]["label"] == "ref"
    assert doc["verdict"]["overall"] == "compatible"


def test_assess_gguf_pair_include_file_hash_false_is_threaded_through(monkeypatch):
    calls = []

    def fake_gguf_identity(path, *, include_file_hash=True):
        calls.append(include_file_hash)
        return dict(_FULL_A)

    monkeypatch.setattr(pc, "gguf_identity", fake_gguf_identity)
    pc.assess_gguf_pair("a.gguf", "b.gguf", include_file_hash=False)
    assert calls == [False, False]
