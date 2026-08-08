"""tests/test_generation_guard.py -- model-free unit tests for clozn/server/generation_guard.py's PURE
pieces: opt-in spec parsing, per-model guard-threshold calibration (load/schema/fallback), layer selection
(explicit -> calibration default -> discovered-valid, never a hardcoded 16), per-concept trigger-set/
threshold resolution (calibrated vs uncalibrated), the topk floor, the jlens-based disposition read
(generalized to a trigger SET), the engine-agnostic control loop (run_guarded_generation), the receipt
builder, and -- critically -- the HONESTY CAVEAT wording itself (present-tense detect-and-correct only,
never lead-time/prediction/"before it's said"/"acts on intent"; see the module's own HONESTY LAW section).

No engine, no HTTP, no GPU: run_guarded_generation is driven entirely through injected fake callables. The
production adapter (guarded_chat_completion) is covered separately in tests/test_generation_guard_server.py
via a fake substrate/engine (never a live one).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clozn.settings as clozn_settings          # noqa: E402
from clozn.server import generation_guard as gg                  # noqa: E402


# ================================================================================ opt-in spec parsing

def test_parse_guard_spec_none_when_field_absent():
    assert gg.parse_guard_spec({}) is None
    assert gg.parse_guard_spec(None) is None


def test_parse_guard_spec_explicit_empty_dict_is_off():
    assert gg.parse_guard_spec({"clozn_guard": {}}) is None


def test_parse_guard_spec_explicit_false_is_off():
    assert gg.parse_guard_spec({"clozn_guard": False}) is None


def test_parse_guard_spec_explicit_none_is_off():
    assert gg.parse_guard_spec({"clozn_guard": None}) is None


def test_parse_guard_spec_empty_concepts_list_is_off():
    assert gg.parse_guard_spec({"clozn_guard": {"concepts": []}}) is None


def test_parse_guard_spec_valid_minimal_fills_defaults_and_leaves_layer_unset():
    """`layer` must be None (not a hardcoded default) unless the caller explicitly set it -- see the
    module docstring's LAYER SELECTION section; a stale hardcoded default is exactly the bug that broke
    moving from the 9B to the 7B (layer 16 isn't even a fitted J-lens layer there)."""
    spec = gg.parse_guard_spec({"clozn_guard": {"concepts": ["violence"]}})
    assert spec == {
        "concepts": ["violence"], "threshold": gg.DEFAULT_THRESHOLD,
        "counter_strength": gg.DEFAULT_COUNTER_STRENGTH, "max_fires": gg.DEFAULT_MAX_FIRES,
        "layer": None, "chunk_tokens": gg.DEFAULT_CHUNK_TOKENS, "topk": gg.DEFAULT_TOPK,
    }


def test_parse_guard_spec_honors_explicit_overrides():
    spec = gg.parse_guard_spec({"clozn_guard": {
        "concepts": ["violence", "self-harm"], "threshold": 2.5, "counter_strength": -0.8,
        "max_fires": 5, "layer": 21, "chunk_tokens": 16, "topk": 4,
    }})
    assert spec["concepts"] == ["violence", "self-harm"]
    assert spec["threshold"] == 2.5
    assert spec["counter_strength"] == -0.8
    assert spec["max_fires"] == 5
    assert spec["layer"] == 21
    assert spec["chunk_tokens"] == 16
    assert spec["topk"] == 4


@pytest.mark.parametrize("bad, why", [
    ({"clozn_guard": "not-a-dict"}, "must be an object"),
    ({"clozn_guard": {"concepts": "violence"}}, "concepts must be a list"),
    ({"clozn_guard": {"concepts": [""]}}, "non-empty strings"),
    ({"clozn_guard": {"concepts": [123]}}, "non-empty strings"),
    ({"clozn_guard": {"concepts": ["x"], "threshold": "high"}}, "threshold must be a number"),
    ({"clozn_guard": {"concepts": ["x"], "counter_strength": True}}, "counter_strength must be a number"),
    ({"clozn_guard": {"concepts": ["x"], "max_fires": 0}}, "max_fires must be a positive integer"),
    ({"clozn_guard": {"concepts": ["x"], "max_fires": 1.5}}, "max_fires must be a positive integer"),
    ({"clozn_guard": {"concepts": ["x"], "layer": "16"}}, "layer must be an integer"),
    ({"clozn_guard": {"concepts": ["x"], "chunk_tokens": 0}}, "chunk_tokens must be a positive integer"),
    ({"clozn_guard": {"concepts": ["x"], "topk": -1}}, "topk must be a positive integer"),
])
def test_parse_guard_spec_rejects_malformed_values(bad, why):
    with pytest.raises(ValueError, match=why):
        gg.parse_guard_spec(bad)


def test_parse_guard_spec_ignores_a_stale_persisted_setting(monkeypatch, tmp_path):
    """Regression: Clozn no longer persists a server-wide guard default (retired -- see
    docs/CAPABILITIES.md). A settings file left over from an old Clozn version that still carries the
    retired `generation_guard` key must be completely inert -- a request that omits `clozn_guard` always
    takes the ordinary, unguarded path, regardless of anything ever written to the settings store."""
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    clozn_settings.set_setting("generation_guard", {"concepts": ["violence"]})
    assert gg.parse_guard_spec({}) is None
    spec = gg.parse_guard_spec({"clozn_guard": {"concepts": ["self-harm"]}})
    assert spec["concepts"] == ["self-harm"]


# ================================================================================ per-model calibration load/schema

def _write_calibration(path, *, model_sha256="abc123", default_layer=14,
                      concepts=None, schema=gg.GUARD_CALIBRATION_SCHEMA):
    concepts = concepts if concepts is not None else {
        "violence": {"layer": 14, "threshold": 10.07, "trigger_ids": [101, 102, 103],
                    "trigger_pieces": [" kill", " knife", " blood"], "catch": 1.0, "fp": 0.0,
                    "n_battery": 12, "note": "small-battery calibration"},
    }
    payload = {"schema_version": schema, "model_sha256": model_sha256, "default_layer": default_layer,
              "concepts": concepts}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload


def test_guard_calibration_path_scopes_per_exact_digest():
    p = gg.guard_calibration_path("abc123")
    assert p == os.path.join(os.path.expanduser("~"), ".clozn", "models", "abc123",
                             "guard_threshold_calibration.json")


def test_guard_calibration_path_falls_back_to_legacy_root_without_a_digest():
    p = gg.guard_calibration_path(None)
    assert p == os.path.join(os.path.expanduser("~"), ".clozn", "guard_threshold_calibration.json")


def test_load_guard_calibration_missing_file_returns_none(tmp_path):
    assert gg.load_guard_calibration("abc123", path=str(tmp_path / "nope.json")) is None


def test_load_guard_calibration_wrong_schema_returns_none(tmp_path):
    p = tmp_path / "cal.json"
    _write_calibration(str(p), schema="something.else.v1")
    assert gg.load_guard_calibration("abc123", path=str(p)) is None


def test_load_guard_calibration_corrupt_json_returns_none(tmp_path):
    p = tmp_path / "cal.json"
    p.write_text("{not valid json", encoding="utf-8")
    assert gg.load_guard_calibration("abc123", path=str(p)) is None


def test_load_guard_calibration_no_concepts_returns_none(tmp_path):
    p = tmp_path / "cal.json"
    _write_calibration(str(p), concepts={})
    assert gg.load_guard_calibration("abc123", path=str(p)) is None


def test_load_guard_calibration_valid_file_reads_the_real_artifact_shape(tmp_path):
    """Matches runs/experiments/guard_signal_qwen2.5-7b.json's own emitted artifact shape exactly."""
    p = tmp_path / "cal.json"
    _write_calibration(str(p))
    cal = gg.load_guard_calibration("abc123", path=str(p))
    assert cal["model_sha256"] == "abc123"
    assert cal["default_layer"] == 14
    assert cal["concepts"]["violence"] == {
        "layer": 14, "threshold": 10.07, "trigger_ids": [101, 102, 103],
        "trigger_pieces": [" kill", " knife", " blood"], "catch": 1.0, "fp": 0.0,
        "n_battery": 12, "note": "small-battery calibration", "topk": None,
    }


def test_load_guard_calibration_reads_forward_compatible_topk(tmp_path):
    p = tmp_path / "cal.json"
    _write_calibration(str(p), concepts={
        "violence": {"layer": 14, "threshold": 10.07, "trigger_ids": [101], "topk": 96},
    })
    cal = gg.load_guard_calibration("abc123", path=str(p))
    assert cal["concepts"]["violence"]["topk"] == 96


def test_load_guard_calibration_skips_malformed_concept_entries_keeps_the_rest(tmp_path):
    p = tmp_path / "cal.json"
    _write_calibration(str(p), concepts={
        "violence": {"layer": 14, "threshold": 10.07, "trigger_ids": [101, 102]},
        "broken_layer": {"layer": "14", "threshold": 1.0, "trigger_ids": [1]},
        "broken_no_triggers": {"layer": 14, "threshold": 1.0, "trigger_ids": []},
        "broken_shape": "not-a-dict",
    })
    cal = gg.load_guard_calibration("abc123", path=str(p))
    assert list(cal["concepts"]) == ["violence"]


def test_load_guard_calibration_bad_default_layer_degrades_to_none(tmp_path):
    p = tmp_path / "cal.json"
    payload = _write_calibration(str(p))
    payload["default_layer"] = "fourteen"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    cal = gg.load_guard_calibration("abc123", path=str(p))
    assert cal["default_layer"] is None
    assert "violence" in cal["concepts"]   # the rest of the file is still usable


# ================================================================================ layer selection

def test_engine_jlens_layers_reads_health_jlens_layers():
    class FakeEngine:
        def health(self):
            return {"jlens": {"on": True, "layers": [21, 2, 14, 25]}}
    assert gg.engine_jlens_layers(FakeEngine()) == [2, 14, 21, 25]


def test_engine_jlens_layers_empty_when_no_jlens_block():
    class FakeEngine:
        def health(self):
            return {"model": "x"}
    assert gg.engine_jlens_layers(FakeEngine()) == []


def test_engine_jlens_layers_empty_on_engine_failure():
    class FakeEngine:
        def health(self):
            raise RuntimeError("down")
    assert gg.engine_jlens_layers(FakeEngine()) == []


def test_nearest_layer_picks_closest():
    assert gg._nearest_layer(16, [2, 14, 21, 25]) == 14   # |16-14|=2 < |16-21|=5
    assert gg._nearest_layer(16, [16, 21]) == 16
    assert gg._nearest_layer(16, []) is None


def test_nearest_layer_ties_break_to_the_smaller_layer():
    assert gg._nearest_layer(10, [8, 12]) == 8   # both distance 2 -- smaller wins, deterministic


def test_resolve_guard_layer_explicit_always_wins():
    spec = {"layer": 99}   # even an engine-invalid layer -- honored as-is, never second-guessed
    layer, source = gg.resolve_guard_layer(spec, None, [2, 14, 21, 25])
    assert (layer, source) == (99, "explicit")


def test_resolve_guard_layer_uses_calibration_default_when_available():
    spec = {"layer": None}
    cal = {"default_layer": 14, "concepts": {}}
    layer, source = gg.resolve_guard_layer(spec, cal, [2, 14, 21, 25])
    assert (layer, source) == (14, "calibration_default")


def test_resolve_guard_layer_ignores_calibration_default_when_engine_lacks_it():
    """The calibration says layer 14, but THIS engine doesn't report it as fitted -- fall through to
    discovery rather than trusting a stale calibration default blindly."""
    spec = {"layer": None}
    cal = {"default_layer": 14, "concepts": {}}
    layer, source = gg.resolve_guard_layer(spec, cal, [21, 25])
    assert source == "discovered_valid"
    assert layer in (21, 25)


def test_resolve_guard_layer_16_invalid_falls_through_to_discovery():
    """THE bug this whole update fixes: a hardcoded 16 that isn't even a fitted layer on this model
    (the 7B's fitted layers are [2, 14, 21, 25]) must never be used -- discovery picks the nearest valid
    one instead (14, distance 2, beats 21's distance 5)."""
    spec = {"layer": None}
    layer, source = gg.resolve_guard_layer(spec, None, [2, 14, 21, 25])
    assert (layer, source) == (14, "discovered_valid")


def test_resolve_guard_layer_unavailable_when_engine_reports_no_layers_at_all():
    spec = {"layer": None}
    layer, source = gg.resolve_guard_layer(spec, None, [])
    assert layer is None
    assert source == "unavailable"


# ================================================================================ per-concept signal resolution

def test_resolve_concept_signal_calibrated_at_the_polled_layer():
    cal = {"default_layer": 14, "concepts": {
        "violence": {"layer": 14, "threshold": 10.07, "trigger_ids": [101, 102],
                    "trigger_pieces": [" kill", " knife"], "catch": 1.0, "fp": 0.0, "n_battery": 12,
                    "note": "small battery", "topk": None},
    }}
    signal = gg.resolve_concept_signal("violence", cal, poll_layer=14)
    assert signal["calibrated"] is True
    assert signal["threshold"] == 10.07
    assert signal["trigger_ids"] == {101, 102}
    assert signal["trigger_pieces"] == [" kill", " knife"]
    assert signal["catch"] == 1.0 and signal["fp"] == 0.0


def test_resolve_concept_signal_calibrated_at_a_different_layer_is_not_applied():
    """A calibration exists for 'violence' at layer 14, but this request polls layer 21 -- never
    misapplied to a different layer's readout."""
    cal = {"default_layer": 14, "concepts": {
        "violence": {"layer": 14, "threshold": 10.07, "trigger_ids": [101], "trigger_pieces": [" kill"],
                    "topk": None},
    }}
    signal = gg.resolve_concept_signal("violence", cal, poll_layer=21)
    assert signal["calibrated"] is False
    assert signal["trigger_ids"] is None
    assert "layer 14" in signal["note"] and "polling layer 21" in signal["note"]


def test_resolve_concept_signal_uncalibrated_concept_gets_the_documented_note():
    signal = gg.resolve_concept_signal("unicorn-sparkles", None, poll_layer=14)
    assert signal["calibrated"] is False
    assert signal["threshold"] == gg.DEFAULT_THRESHOLD
    assert signal["note"] == gg.UNCALIBRATED_NOTE


def test_resolve_concept_signal_uncalibrated_uses_the_request_fallback_threshold():
    signal = gg.resolve_concept_signal("unicorn-sparkles", None, poll_layer=14, fallback_threshold=3.3)
    assert signal["calibrated"] is False
    assert signal["threshold"] == 3.3


class _FakeConceptSteer:
    """Fakes concept_dir.ConceptSteer's two relevant entry points: .compute (full dir(c), needed for a
    CALIBRATED/correctable concept) and .resolve_token_id (tokenize-only, needed for an UNCALIBRATED
    concept's annotate-only fallback)."""

    def __init__(self, compute_ok=None, resolve_ok=None, token_ids=None):
        self.compute_ok = set(compute_ok or [])
        self.resolve_ok = set(resolve_ok if resolve_ok is not None else (compute_ok or []))
        self.token_ids = dict(token_ids or {})
        self.compute_calls = []
        self.resolve_calls = []

    def compute(self, concept, layer=None):
        self.compute_calls.append((concept, layer))
        if concept in self.compute_ok:
            return {"ok": True, "concept": concept, "layer": layer,
                    "token_id": self.token_ids.get(concept, 1), "vector": [0.1, 0.2]}
        return {"ok": False, "blocked": "unembed_unavailable", "concept": concept,
                "note": f"no dir({concept}) available"}

    def resolve_token_id(self, concept):
        self.resolve_calls.append(concept)
        if concept in self.resolve_ok:
            return {"ok": True, "token_id": self.token_ids.get(concept, 2), "piece": " " + concept}
        return {"ok": False, "note": f"cannot tokenize {concept!r}"}


def test_resolve_guard_signals_calibrated_concept_never_calls_resolve_token_id():
    cal = {"default_layer": 14, "concepts": {
        "violence": {"layer": 14, "threshold": 10.07, "trigger_ids": [101], "trigger_pieces": [" kill"],
                    "topk": None},
    }}
    steer = _FakeConceptSteer(compute_ok=["violence"])
    signals, reason = gg.resolve_guard_signals(steer, {"concepts": ["violence"], "threshold": 0.0}, cal, 14)
    assert reason is None
    assert signals["violence"]["calibrated"] is True
    assert steer.compute_calls == [("violence", 14)]
    assert steer.resolve_calls == []   # never needed -- the calibration already carries a trigger set


def test_resolve_guard_signals_fails_closed_when_a_calibrated_concept_cannot_build_dirc():
    cal = {"default_layer": 14, "concepts": {
        "violence": {"layer": 14, "threshold": 10.07, "trigger_ids": [101], "trigger_pieces": [" kill"],
                    "topk": None},
    }}
    steer = _FakeConceptSteer(compute_ok=[])   # dir(c) build fails for everything
    signals, reason = gg.resolve_guard_signals(steer, {"concepts": ["violence"], "threshold": 0.0}, cal, 14)
    assert signals is None
    assert "violence" in reason and "unavailable" in reason


def test_resolve_guard_signals_uncalibrated_concept_resolves_via_token_id():
    steer = _FakeConceptSteer(resolve_ok=["mild-topic"], token_ids={"mild-topic": 55})
    signals, reason = gg.resolve_guard_signals(
        steer, {"concepts": ["mild-topic"], "threshold": 0.0}, None, 14,
    )
    assert reason is None
    assert signals["mild-topic"]["calibrated"] is False
    assert signals["mild-topic"]["trigger_ids"] == {55}
    assert steer.compute_calls == []   # never needed -- this concept can't fire anyway


def test_resolve_guard_signals_uncalibrated_concept_degrades_softly_not_fatally():
    """An uncalibrated concept that can't even be tokenized does NOT refuse the whole request -- it was
    never eligible to fire a correction, so it just loses its annotation (with a note), and a calibrated
    sibling concept still resolves fine."""
    cal = {"default_layer": 14, "concepts": {
        "violence": {"layer": 14, "threshold": 10.07, "trigger_ids": [101], "trigger_pieces": [" kill"],
                    "topk": None},
    }}
    steer = _FakeConceptSteer(compute_ok=["violence"], resolve_ok=[])   # "mystery" can't even tokenize
    signals, reason = gg.resolve_guard_signals(
        steer, {"concepts": ["violence", "mystery"], "threshold": 0.0}, cal, 14,
    )
    assert reason is None
    assert signals["violence"]["calibrated"] is True
    assert signals["mystery"]["calibrated"] is False
    assert signals["mystery"]["trigger_ids"] is None
    assert "could not be resolved" in signals["mystery"]["note"]


# ================================================================================ topk floor

def test_resolve_guard_topk_no_calibrated_concepts_uses_spec_topk():
    signals = {"x": {"calibrated": False}}
    assert gg.resolve_guard_topk({"topk": 8}, signals) == 8


def test_resolve_guard_topk_raises_floor_when_any_concept_is_calibrated():
    signals = {"violence": {"calibrated": True, "topk": None}, "x": {"calibrated": False}}
    assert gg.resolve_guard_topk({"topk": 8}, signals) == gg.CALIBRATION_TOPK_FLOOR


def test_resolve_guard_topk_never_lowers_a_higher_spec_topk():
    signals = {"violence": {"calibrated": True, "topk": None}}
    assert gg.resolve_guard_topk({"topk": 200}, signals) == 200


def test_resolve_guard_topk_honors_a_per_concept_topk_above_the_floor():
    signals = {"violence": {"calibrated": True, "topk": 128}}
    assert gg.resolve_guard_topk({"topk": 8}, signals) == 128


# ================================================================================ concept_activation (jlens read)

def test_concept_activation_single_id_backward_compatible():
    jl = {"readouts": [[{"id": 1, "score": 0.1}], [{"id": 5, "score": 9.5}, {"id": 1, "score": -2.0}]]}
    assert gg.concept_activation(jl, 5) == pytest.approx(9.5)


def test_concept_activation_trigger_set_takes_the_max_over_present_ids():
    jl = {"readouts": [[{"id": 101, "score": 3.0}, {"id": 102, "score": 9.0}, {"id": 999, "score": 50.0}]]}
    assert gg.concept_activation(jl, {101, 102, 103}) == pytest.approx(9.0)


def test_concept_activation_trigger_set_absent_entirely_is_none():
    jl = {"readouts": [[{"id": 1, "score": 9.5}]]}
    assert gg.concept_activation(jl, {101, 102}) is None


def test_concept_activation_empty_trigger_set_is_none():
    jl = {"readouts": [[{"id": 1, "score": 9.5}]]}
    assert gg.concept_activation(jl, set()) is None
    assert gg.concept_activation(jl, None) is None


def test_concept_activation_no_readouts_is_none():
    assert gg.concept_activation({"readouts": []}, 5) is None
    assert gg.concept_activation({}, 5) is None
    assert gg.concept_activation(None, 5) is None


def test_concept_activation_malformed_score_is_skipped_not_fatal():
    jl = {"readouts": [[{"id": 101, "score": "high"}, {"id": 102, "score": 4.0}]]}
    assert gg.concept_activation(jl, {101, 102}) == pytest.approx(4.0)


def test_concept_activation_specific_position():
    jl = {"readouts": [[{"id": 5, "score": 1.0}], [{"id": 5, "score": 2.0}]]}
    assert gg.concept_activation(jl, 5, position=0) == pytest.approx(1.0)
    assert gg.concept_activation(jl, 5, position=1) == pytest.approx(2.0)


# ================================================================================ run_guarded_generation (the loop)

def _fake_generate(pieces_by_call):
    calls = []

    def generate_chunk(prompt_so_far, max_new, *, counter=None):
        calls.append({"prompt_len": len(prompt_so_far), "max_new": max_new, "counter": counter})
        return pieces_by_call[len(calls) - 1]

    generate_chunk.calls = calls
    return generate_chunk


def test_guard_loop_no_fire_never_corrects():
    gen = _fake_generate(["clean chunk one ", "clean chunk two"])

    def read_disposition(text):
        return {"violence": None}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition,
        build_counter=lambda c: (_ for _ in ()).throw(
            AssertionError("build_counter must never be called when nothing fires")),
        base_text="PROMPT: ", max_tokens=48, chunk_tokens=24, concepts=["violence"],
        correctable_concepts={"violence"}, thresholds={"violence": 1.0},
        counter_strength=-0.5, max_fires=3, layer=14,
    )
    assert out["text"] == "clean chunk one clean chunk two"
    assert out["fires"] == []
    assert out["n_fires"] == 0
    assert out["cap_reached"] is False
    assert out["n_chunks"] == 2
    assert len(gen.calls) == 2


def test_guard_loop_fires_and_replaces_the_flagged_chunk():
    gen = _fake_generate(["BAD content here", "safe corrected content", " clean chunk two"])

    def read_disposition(text):
        if "BAD content" in text:
            return {"violence": 9.0}
        return {"violence": None}

    built_counters = []

    def build_counter(concept):
        built_counters.append(concept)
        return {"concept": concept, "vector": [0.1], "coef": -5.0}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition, build_counter=build_counter,
        base_text="PROMPT: ", max_tokens=48, chunk_tokens=24, concepts=["violence"],
        correctable_concepts={"violence"}, thresholds={"violence": 1.0},
        counter_strength=-0.5, max_fires=3, layer=14,
    )
    assert out["text"] == "safe corrected content clean chunk two"
    assert out["n_fires"] == 1
    assert out["cap_reached"] is False
    assert built_counters == ["violence"]
    fire = out["fires"][0]
    assert fire["concept"] == "violence"
    assert fire["pre_activation"] == pytest.approx(9.0)
    assert fire["post_activation"] is None
    assert fire["counter_strength"] == -0.5
    assert fire["chunk_index"] == 0
    assert fire["token_position"] == 0
    assert fire["layer"] == 14
    assert fire["threshold"] == 1.0
    assert fire["calibrated"] is True
    assert out["max_observed_activation"]["violence"] == pytest.approx(9.0)


def test_guard_loop_uncalibrated_concept_never_fires_even_above_its_displayed_threshold():
    """The core policy decision: a concept NOT in correctable_concepts is annotate-only -- it can cross
    its own 'threshold' entry and still never trigger a correction."""
    gen = _fake_generate(["TRIGGER word here", "more TRIGGER word", "still more TRIGGER"])

    def read_disposition(text):
        return {"mystery": 99.0} if "TRIGGER" in text else {"mystery": None}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition,
        build_counter=lambda c: (_ for _ in ()).throw(
            AssertionError("an uncalibrated concept must never call build_counter")),
        base_text="", max_tokens=72, chunk_tokens=24, concepts=["mystery"],
        correctable_concepts=set(), thresholds={"mystery": 0.0},
        counter_strength=-0.5, max_fires=3, layer=14,
    )
    assert out["n_fires"] == 0
    assert out["cap_reached"] is False
    assert out["max_observed_activation"]["mystery"] == pytest.approx(99.0)   # still annotated
    assert "TRIGGER" in out["text"]   # never corrected


def test_guard_loop_records_post_activation_when_correction_does_not_fully_clear_it():
    gen = _fake_generate(["BAD content", "still a bit BAD"])

    def read_disposition(text):
        if "BAD" in text:
            return {"violence": 9.0 if "still a bit" not in text else 3.0}
        return {"violence": None}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition,
        build_counter=lambda c: {"vector": [0.1], "coef": -1.0},
        base_text="", max_tokens=24, chunk_tokens=24, concepts=["violence"],
        correctable_concepts={"violence"}, thresholds={"violence": 1.0},
        counter_strength=-0.5, max_fires=3, layer=14,
    )
    assert out["n_fires"] == 1
    assert out["fires"][0]["pre_activation"] == pytest.approx(9.0)
    assert out["fires"][0]["post_activation"] == pytest.approx(3.0)


def test_guard_loop_cap_reached_stops_correcting_and_says_so():
    gen = _fake_generate(["BAD one", "corrected one", "BAD two", "plain rest"])

    def read_disposition(text):
        if "BAD" in text:
            return {"violence": 9.0}
        return {"violence": None}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition,
        build_counter=lambda c: {"vector": [0.1], "coef": -1.0},
        base_text="", max_tokens=72, chunk_tokens=24, concepts=["violence"],
        correctable_concepts={"violence"}, thresholds={"violence": 1.0},
        counter_strength=-0.5, max_fires=1, layer=14,
    )
    assert out["n_fires"] == 1
    assert out["cap_reached"] is True
    assert "BAD two" in out["text"]
    assert "plain rest" in out["text"]
    assert len(gen.calls) == 4


def test_guard_loop_never_exceeds_max_fires_even_with_many_triggering_chunks():
    pieces = ["BAD"] * 10
    gen = _fake_generate(pieces)

    def read_disposition(text):
        return {"violence": 9.0} if text.rstrip().endswith("BAD") else {"violence": None}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition,
        build_counter=lambda c: {"vector": [0.1], "coef": -1.0},
        base_text="", max_tokens=24 * 10, chunk_tokens=24, concepts=["violence"],
        correctable_concepts={"violence"}, thresholds={"violence": 1.0},
        counter_strength=-0.5, max_fires=2, layer=14,
    )
    assert out["n_fires"] == 2
    assert out["cap_reached"] is True


# ================================================================================ build_receipt

def _resolved_signals_calibrated():
    return {"violence": {
        "calibrated": True, "threshold": 10.07, "trigger_ids": {101, 102}, "trigger_pieces": [" kill", " knife"],
        "topk": None, "catch": 1.0, "fp": 0.0, "n_battery": 12, "calibration_note": "small battery", "note": None,
    }}


def test_build_receipt_per_concept_breakdown_calibrated():
    spec = gg.parse_guard_spec({"clozn_guard": {"concepts": ["violence"]}})
    result = {"text": "clean text", "fires": [], "n_fires": 0, "cap_reached": False, "n_chunks": 1,
             "max_observed_activation": {"violence": 5.0}}
    receipt = gg.build_receipt(result, spec, layer=14, layer_source="discovered_valid", topk=64,
                              resolved_signals=_resolved_signals_calibrated())
    entry = receipt["concepts"]["violence"]
    assert entry["calibrated"] is True
    assert entry["layer"] == 14
    assert entry["threshold"] == 10.07
    assert entry["trigger_ids"] == [101, 102]
    assert entry["trigger_pieces"] == [" kill", " knife"]
    assert entry["max_observed_activation"] == 5.0
    assert entry["catch"] == 1.0 and entry["fp"] == 0.0
    assert receipt["layer"] == 14
    assert receipt["layer_source"] == "discovered_valid"
    assert receipt["topk"] == 64
    assert receipt["n_fires"] == 0
    assert "cap_note" not in receipt
    assert receipt["caveat"] == gg.GUARD_CAVEAT


def test_build_receipt_per_concept_breakdown_uncalibrated():
    spec = gg.parse_guard_spec({"clozn_guard": {"concepts": ["mystery"]}})
    result = {"text": "x", "fires": [], "n_fires": 0, "cap_reached": False, "n_chunks": 1,
             "max_observed_activation": {"mystery": None}}
    signals = {"mystery": {"calibrated": False, "threshold": gg.DEFAULT_THRESHOLD, "trigger_ids": None,
                          "trigger_pieces": None, "note": gg.UNCALIBRATED_NOTE}}
    receipt = gg.build_receipt(result, spec, layer=14, layer_source="discovered_valid", topk=8,
                              resolved_signals=signals)
    entry = receipt["concepts"]["mystery"]
    assert entry["calibrated"] is False
    assert entry["trigger_ids"] is None
    assert entry["note"] == gg.UNCALIBRATED_NOTE
    assert "catch" not in entry


def test_build_receipt_includes_cap_note_when_capped():
    spec = gg.parse_guard_spec({"clozn_guard": {"concepts": ["violence"], "max_fires": 1}})
    result = {"text": "x", "fires": [{"chunk_index": 0, "token_position": 0, "concept": "violence",
                                      "pre_activation": 9.0, "post_activation": 9.0,
                                      "counter_strength": -0.5, "layer": 14, "threshold": 10.07,
                                      "calibrated": True}],
             "n_fires": 1, "cap_reached": True, "n_chunks": 3, "max_observed_activation": {"violence": 9.0}}
    receipt = gg.build_receipt(result, spec, layer=14, layer_source="calibration_default", topk=64,
                              resolved_signals=_resolved_signals_calibrated())
    assert receipt["cap_reached"] is True
    assert receipt["cap_note"] == gg.GUARD_CAP_NOTE


# ================================================================================ THE HONESTY CAVEAT ITSELF

_FORBIDDEN_OVERCLAIM_PHRASES = [
    "before it's said", "before it is said", "reads intent early", "acts on intent",
    "predicts the", "predict the", "prediction of intent", "intent-before-speech capability",
    "foreknowledge of", "sees it coming",
]

_REQUIRED_PRESENT_TENSE_PHRASE = "detects and corrects during generation"


def test_caveat_contains_the_required_present_tense_phrase_verbatim():
    assert _REQUIRED_PRESENT_TENSE_PHRASE in gg.GUARD_CAVEAT


def test_caveat_contains_the_a1_1_numbers_honestly():
    assert "100%" in gg.GUARD_CAVEAT
    assert "5%" in gg.GUARD_CAVEAT
    assert "10/10" in gg.GUARD_CAVEAT
    assert "1/20" in gg.GUARD_CAVEAT
    assert "layer 16" in gg.GUARD_CAVEAT
    assert "-0.5" in gg.GUARD_CAVEAT
    assert "median_lead_time_tokens = 0" in gg.GUARD_CAVEAT
    assert "4/10" in gg.GUARD_CAVEAT


def test_caveat_states_the_small_battery_calibration_is_not_a_reliability_claim():
    lowered = gg.GUARD_CAVEAT.lower()
    assert "small-battery" in lowered
    assert "not a public reliability claim" in lowered
    assert "calibrated" in lowered   # references the receipt's own field name


def test_caveat_never_claims_lead_time_or_prediction_as_a_capability():
    lowered = gg.GUARD_CAVEAT.lower()
    for phrase in _FORBIDDEN_OVERCLAIM_PHRASES:
        assert phrase not in lowered, f"caveat must never claim: {phrase!r}"


def test_cap_note_also_carries_no_overclaim():
    lowered = gg.GUARD_CAP_NOTE.lower()
    for phrase in _FORBIDDEN_OVERCLAIM_PHRASES:
        assert phrase not in lowered, f"cap note must never claim: {phrase!r}"


def test_uncalibrated_note_also_carries_no_overclaim_and_explains_the_decision():
    lowered = gg.UNCALIBRATED_NOTE.lower()
    for phrase in _FORBIDDEN_OVERCLAIM_PHRASES:
        assert phrase not in lowered, f"uncalibrated note must never claim: {phrase!r}"
    assert "never fires a correction" in lowered or "never fire" in lowered


def test_module_docstring_states_the_present_tense_framing():
    """The module's own documentation must state the present-tense framing. (Unlike GUARD_CAVEAT/
    GUARD_CAP_NOTE -- the actual API-facing strings, scanned above for the forbidden phrases themselves --
    the docstring's HONESTY LAW section legitimately QUOTES those same forbidden phrases as instructive
    examples of what never to claim, so it is deliberately not scanned for their absence here.)"""
    doc = " ".join((gg.__doc__ or "").split()).lower()   # collapse line-wrap whitespace before matching
    assert "present-tense" in doc or "present tense" in doc
    assert "detects and corrects during generation" in doc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
