"""Unit tests for the per-token trace capture (issue B3).

The Run Inspector timeline needs, per generated token: the committed piece, the model's confidence, and
the alternatives it weighed. Two paths feed that -- the CLI's stream_ar and the engine-chat SSE capture --
and both route their per-token "steps" through the SAME pure helpers in runlog:

  - runlog.accumulate_ar_events(frames)  folds raw SSE frames -> ordered steps (shared by CLI + server)
  - runlog.steps_to_trace(steps)         maps steps -> the stored {tokens, confidence, alternatives}
  - runlog._norm_trace(trace)            coerces whatever record() is handed into the stored shape

These are model-free, so we exercise them with fabricated engine frames and fabricated CLI steps -- no
GGUF, no HF, no network.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # research/ (runlog lives here)
import clozn.runs.store as runlog
from clozn.readouts import workspace_lens


# --------------------------------------------------------------------------- fixtures

def fake_engine_frames():
    """The frames the engine streams for a 3-token AR completion: a tokens_committed per token plus a
    step_lens (top-k) for the uncertain ones. Position order is intentionally not first-seen-sorted."""
    return [
        {"type": "tokens_committed", "items": [{"pos": 0, "id": 9, "conf": 0.95, "piece": "The"}]},
        {"type": "step_lens", "positions": [0], "pieces": ["The", "A", "This"], "probs": [0.95, 0.03, 0.02]},
        {"type": "tokens_committed", "items": [{"pos": 1, "id": 5, "conf": 0.42, "piece": " cat"}]},
        {"type": "step_lens", "positions": [1], "pieces": [" cat", " dog", " fox"], "probs": [0.42, 0.31, 0.2]},
        {"type": "tokens_committed", "items": [{"pos": 2, "id": 7, "conf": 0.88, "piece": " sat"}]},
        # a final OpenAI-style frame (carries the assembled text, no per-token data) -- must be ignored
        {"id": "cmpl-x", "object": "text_completion",
         "choices": [{"text": "The cat sat", "index": 0, "finish_reason": "stop"}]},
    ]


def fake_cli_steps():
    """The step shape the CLI's stream_ar returns after accumulation: {pos, piece, conf, alts}."""
    return [
        {"pos": 0, "piece": "Hello", "conf": 0.99, "alts": []},
        {"pos": 1, "piece": " there", "conf": 0.55, "alts": [{"piece": " world", "prob": 0.4}]},
    ]


# --------------------------------------------------------------------------- accumulate_ar_events

def test_accumulate_pairs_commits_with_lens_in_order():
    steps = runlog.accumulate_ar_events(fake_engine_frames())
    assert [s["piece"] for s in steps] == ["The", " cat", " sat"]        # first-seen position order
    assert [s["pos"] for s in steps] == [0, 1, 2]
    # confidences carried through from tokens_committed
    assert steps[0]["conf"] == 0.95 and steps[1]["conf"] == 0.42
    # step_lens folded in as alternatives, chosen token excluded, capped at 3
    assert steps[1]["alts"] == [{"token_id": None, "piece": " dog", "prob": 0.31},
                                {"token_id": None, "piece": " fox", "prob": 0.2}]
    assert steps[2]["alts"] == []                                        # no step_lens for pos 2


def test_accumulate_empty_and_garbage():
    assert runlog.accumulate_ar_events([]) == []
    assert runlog.accumulate_ar_events(None) == []
    # malformed frames are skipped, not fatal
    assert runlog.accumulate_ar_events([1, "x", {"type": "other"}, {"type": "step_lens"}]) == []


def test_accumulate_preserves_token_id_and_derives_topk_entropy():
    """Backlog #2: accumulate_ar_events must stop dropping tokens_committed's `id`, and must derive a
    topk_entropy per position from step_lens's recorded top-k probs (-sum(p*ln p), chosen token included) --
    a TOP-K APPROXIMATION, never the true full-vocab entropy the engine never sends."""
    steps = runlog.accumulate_ar_events(fake_engine_frames())
    assert [s["id"] for s in steps] == [9, 5, 7]                     # real engine ids, not dropped
    # pos 0 and 1 had a step_lens row -> a real (approximate) topk_entropy; pos 2 had none -> no key at all
    assert steps[0]["topk_entropy"] == 0.232166
    assert steps[1]["topk_entropy"] == 1.049305
    assert "topk_entropy" not in steps[2]                             # never fabricated for a missing top-k


def test_steps_to_trace_derives_logprob_as_ln_conf():
    """logprobs is DERIVED (log of that token's own confidence), not a separately-emitted value -- so it
    must equal math.log(conf) exactly (rounded), for every committed token in the fabricated frames."""
    import math
    steps = runlog.accumulate_ar_events(fake_engine_frames())
    trace = runlog.steps_to_trace(steps)
    expected = [round(math.log(c), 6) for c in trace["confidence"]]
    assert trace["logprobs"] == expected == [-0.051293, -0.867501, -0.127833]


# --------------------------------------------------------------------------- finish_reason_from_frames

def test_finish_reason_reads_the_final_frame():
    """The engine's real stop cause rides the final OpenAI-style frame's choices[0].finish_reason -- the
    very frame accumulate_ar_events ignores. finish_reason_from_frames plucks it back."""
    assert runlog.finish_reason_from_frames(fake_engine_frames()) == "stop"


def test_finish_reason_length_when_truncated():
    frames = [
        {"type": "tokens_committed", "items": [{"pos": 0, "conf": 0.9, "piece": "hi"}]},
        {"id": "cmpl-x", "object": "text_completion",
         "choices": [{"text": "hi", "index": 0, "finish_reason": "length"}]},
    ]
    assert runlog.finish_reason_from_frames(frames) == "length"


def test_finish_reason_accepts_state_stream_final_frame():
    """The state-stream protocol carries it top-level on a 'final' frame, not under choices -- accept both."""
    assert runlog.finish_reason_from_frames([{"kind": "final", "text": "x", "finish_reason": "length"}]) == "length"


def test_finish_reason_from_gen_finished_event():
    """The AR gen_finished event (raw reason: eos|length|steps_exhausted) is captured and mapped like the
    engine's finish_reason(), so the stop cause survives even when the trailing OpenAI frame isn't present."""
    assert runlog.finish_reason_from_frames([{"type": "gen_finished", "reason": "eos"}]) == "stop"
    assert runlog.finish_reason_from_frames([{"type": "gen_finished", "reason": "length"}]) == "length"
    assert runlog.finish_reason_from_frames([{"type": "gen_finished", "reason": "steps_exhausted"}]) == "length"


def test_raw_worker_finish_reason_is_preserved_before_openai_normalization():
    frames = [
        {"type": "gen_finished", "reason": "steps_exhausted"},
        {"choices": [{"finish_reason": "length"}]},
    ]
    assert runlog.finish_reason_from_frames(frames) == "length"
    assert runlog.raw_finish_reason_from_frames(frames) == "steps_exhausted"


def test_generation_timing_from_worker_terminal_event():
    frames = [{"type": "gen_finished", "new_tokens": 17, "wall_ms": 425.5,
               "steps_total": 18, "tok_per_s": 40.0, "reason": "eos"}]
    assert runlog.generation_timing_from_frames(frames) == {
        "generation_duration_ms": 425.5,
        "generation_tokens": 17,
        "generation_steps": 18,
        "generation_tokens_per_second": 40.0,
    }


def test_finish_reason_none_when_absent_or_garbage():
    assert runlog.finish_reason_from_frames([]) is None
    assert runlog.finish_reason_from_frames(None) is None
    assert runlog.finish_reason_from_frames([1, "x", {"type": "tokens_committed"}]) is None
    # a choices frame that carries text but no finish_reason field -> still None (nothing to pluck)
    assert runlog.finish_reason_from_frames([{"choices": [{"text": "hi", "index": 0}]}]) is None


# --------------------------------------------------------------------------- steps_to_trace

def test_engine_steps_to_trace_full_shape():
    steps = runlog.accumulate_ar_events(fake_engine_frames())
    trace = runlog.steps_to_trace(steps)
    assert set(trace) == {"tokens", "confidence", "alternatives", "steps",
                          "token_ids", "logprobs", "topk_entropy"}
    assert trace["tokens"] == ["The", " cat", " sat"]
    assert trace["confidence"] == [0.95, 0.42, 0.88]
    assert trace["steps"][1]["index"] == 1
    assert trace["steps"][1]["token_id"] == 5
    assert trace["steps"][1]["prob"] == 0.42
    assert trace["steps"][1]["logprob"] == -0.867501
    assert "entropy" not in trace["steps"][1]           # engine frames expose top-k, not the true full entropy
    # topk_entropy IS derivable here -- step_lens recorded a top-3 for every position, so it's a real
    # (if approximate) value, not null, at every index.
    assert trace["steps"][1]["topk_entropy"] == 1.049305
    assert "wall_ms" not in trace["steps"][1] and "dt_ms" not in trace["steps"][1]
    # parallel arrays: one alternatives entry per token (empty where none were recorded)
    assert len(trace["alternatives"]) == 3
    assert trace["alternatives"][1] == [{"piece": " dog", "text": " dog", "prob": 0.31, "logprob": -1.171183},
                                        {"piece": " fox", "text": " fox", "prob": 0.2, "logprob": -1.609438}]
    assert trace["alternatives"][2] == []
    # v2 parallel arrays: real per-token ids/logprobs/topk_entropy -- index-aligned with tokens/confidence.
    assert trace["token_ids"] == [9, 5, 7]
    assert trace["logprobs"] == [-0.051293, -0.867501, -0.127833]
    assert trace["topk_entropy"] == [0.232166, 1.049305, None]      # pos 2 had no step_lens -> null, not fabricated


def test_cli_steps_to_trace_has_tokens_confidence_and_alts():
    trace = runlog.steps_to_trace(fake_cli_steps())
    assert trace["tokens"] == ["Hello", " there"]
    assert trace["confidence"] == [0.99, 0.55]
    assert "alternatives" in trace                                       # at least one step had alts
    assert trace["alternatives"] == [[], [{"piece": " world", "text": " world", "prob": 0.4,
                                           "logprob": -0.916291}]]


def test_steps_without_alternatives_omit_the_key():
    """The non-streaming engine path (and any path with confidence but no top-k) must not store a wall of []."""
    steps = [{"piece": "a", "conf": 0.7, "alts": []}, {"piece": "b", "conf": 0.6, "alts": []}]
    trace = runlog.steps_to_trace(steps)
    assert trace["tokens"] == ["a", "b"]
    assert trace["confidence"] == [0.7, 0.6]
    assert "alternatives" not in trace


def test_steps_to_trace_omits_all_null_new_arrays_but_keeps_derivable_logprobs():
    """token_ids/topk_entropy are all-null here (no ids, no top-k captured) -> each array is omitted
    entirely, same rule as `alternatives`; `logprobs` IS derivable (every step has a real conf) -> kept."""
    import math
    steps = [{"piece": "a", "conf": 0.7, "alts": []}, {"piece": "b", "conf": 0.6, "alts": []}]
    trace = runlog.steps_to_trace(steps)
    assert "token_ids" not in trace
    assert "topk_entropy" not in trace
    assert trace["logprobs"] == [round(math.log(0.7), 6), round(math.log(0.6), 6)]


def test_steps_to_trace_omits_logprobs_when_no_step_has_a_usable_confidence():
    """No step here carries a confidence at all -> nothing to derive a logprob from -> `logprobs` (like
    `token_ids`/`topk_entropy`) is omitted rather than stored as a wall of nulls."""
    steps = [{"piece": "a", "alts": []}, {"piece": "b", "alts": []}]
    trace = runlog.steps_to_trace(steps)
    assert "logprobs" not in trace and "token_ids" not in trace and "topk_entropy" not in trace
    assert trace["confidence"] == [0.0, 0.0]           # v1 array's existing, unchanged default-to-zero contract


def test_empty_steps_give_clean_empty_trace():
    for empty in ([], None, [1, "x", {"no": "dict-fields"}][:0]):
        assert runlog.steps_to_trace(empty) == {}
    # a list of purely non-dict junk also collapses to empty
    assert runlog.steps_to_trace([1, "x", None]) == {}


def test_legacy_step_keys_accepted():
    """steps_to_trace also accepts the stored key names (token/confidence/alternatives), not just piece/conf/alts."""
    steps = [{"token": "x", "confidence": 0.5, "alternatives": [{"piece": "y", "prob": 0.3}]}]
    trace = runlog.steps_to_trace(steps)
    assert trace["tokens"] == ["x"] and trace["confidence"] == [0.5]
    assert trace["alternatives"] == [[{"piece": "y", "text": "y", "prob": 0.3, "logprob": -1.203973}]]


def test_bad_confidence_defaults_to_zero_not_crash():
    trace = runlog.steps_to_trace([{"piece": "x", "conf": "not-a-number", "alts": []}])
    assert trace["confidence"] == [0.0]


def test_steps_to_trace_preserves_real_timing_without_inventing_missing_fields():
    trace = runlog.steps_to_trace([
        {"index": 2, "token_id": 99, "piece": "x", "prob": 0.25, "dt_ms": 12.3456, "wall_ms": 40}
    ])
    step = trace["steps"][0]
    assert step["index"] == 2 and step["token_id"] == 99
    assert step["prob"] == 0.25 and step["logprob"] == -1.386294
    assert step["dt_ms"] == 12.346 and step["wall_ms"] == 40.0
    assert "entropy" not in step


# --------------------------------------------------------------------------- _norm_trace (record's coercion)

def test_norm_trace_accepts_raw_step_list():
    """record(trace=<step list>) must normalize -- the server hands the raw engine steps straight through."""
    norm = runlog._norm_trace(fake_cli_steps())
    assert norm["tokens"] == ["Hello", " there"]
    assert "confidence" in norm


def test_norm_trace_keeps_ready_dict_but_only_known_keys():
    ready = {"tokens": ["a"], "confidence": [0.5], "alternatives": [[]], "junk": 1}
    norm = runlog._norm_trace(ready)
    assert set(norm) == {"tokens", "confidence", "alternatives", "steps"}
    assert "junk" not in norm
    assert norm["steps"][0]["index"] == 0


def test_norm_trace_empty_and_absent_stay_empty():
    assert runlog._norm_trace(None) == {}
    assert runlog._norm_trace([]) == {}
    assert runlog._norm_trace({}) == {}
    assert runlog._norm_trace("garbage") == {}


def test_norm_trace_preserves_new_parallel_arrays_from_a_ready_dict_without_steps():
    """A ready dict trace (record() called directly with pre-built parallel arrays, no `steps`) keeps the
    v2 arrays it was explicitly given, and folds them into the reconstructed `steps` list too."""
    ready = {"tokens": ["a", "b"], "confidence": [0.9, 0.4], "token_ids": [11, None],
             "topk_entropy": [0.05, None]}
    norm = runlog._norm_trace(ready)
    assert norm["token_ids"] == [11, None]
    assert norm["topk_entropy"] == [0.05, None]
    assert norm["steps"][0]["token_id"] == 11
    assert norm["steps"][0]["topk_entropy"] == 0.05
    assert "token_id" not in norm["steps"][1]        # null in the array -> key simply absent on that step
    assert "topk_entropy" not in norm["steps"][1]
    assert norm["steps"][0]["logprob"] is not None   # still derived per-step from confidence, unconditionally


def test_norm_trace_legacy_dict_without_new_keys_has_no_key_error():
    """A trace dict shaped exactly like every run persisted BEFORE this change (no token_ids/logprobs/
    topk_entropy at all) must still normalize cleanly -- the new arrays are simply absent, never a crash.
    This is the backward-compat contract: an old trace still renders, it just has less detail."""
    legacy = {"tokens": ["a", "b"], "confidence": [0.9, 0.4], "alternatives": [[], []]}
    norm = runlog._norm_trace(legacy)
    assert "token_ids" not in norm and "logprobs" not in norm and "topk_entropy" not in norm
    assert norm["steps"][0]["piece"] == "a"


def test_norm_trace_legacy_step_dicts_without_new_keys_have_no_key_error():
    """A raw step list shaped like the pre-change contract ({piece, conf, alts} only, no token_id/
    topk_entropy) must still run through steps_to_trace/_clean_step without a KeyError or crash."""
    legacy_steps = [{"piece": "a", "conf": 0.9, "alts": []}, {"piece": "b", "conf": 0.4, "alts": []}]
    norm = runlog._norm_trace(legacy_steps)
    assert norm["tokens"] == ["a", "b"]
    assert "token_ids" not in norm and "topk_entropy" not in norm
    assert "token_id" not in norm["steps"][0] and "topk_entropy" not in norm["steps"][0]


# --------------------------------------------------------------------------- end-to-end through record()

def test_record_persists_engine_trace(tmp_path, monkeypatch):
    """A full engine-chat-style record() call: raw steps in -> a stored run whose trace has the timeline."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    steps = runlog.accumulate_ar_events(fake_engine_frames())
    rid = runlog.record(source="engine_chat", client="test", model="clozn-qwen (engine)",
                        messages=[{"role": "user", "content": "hi"}], response="The cat sat", trace=steps)
    assert rid
    run = runlog.get_run(rid)
    assert run["trace"]["tokens"] == ["The", " cat", " sat"]
    assert run["trace"]["confidence"] == [0.95, 0.42, 0.88]
    # the low-confidence flag is derived from trace.confidence (0.42 < 0.3 is False, so not flagged here)
    assert "low-confidence" not in run["flags"]


def test_record_does_not_auto_mock_workspace_readouts(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    steps = runlog.accumulate_ar_events(fake_engine_frames())
    rid = runlog.record(source="engine_chat", client="test", model="clozn-qwen (engine)",
                        messages=[{"role": "user", "content": "hi"}], response="The cat sat", trace=steps)
    assert "workspace_readouts" not in runlog.get_run(rid)["trace"]


def test_record_attaches_provider_workspace_readouts(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    steps = runlog.accumulate_ar_events(fake_engine_frames())

    def provider(rid, trace):
        return workspace_lens.readouts_from_concepts(
            rid, trace, {"considered": [{"label": "dragon/fear/RPG", "rel": 0.82}], "layer": 15},
            provider="engine_concepts", layer=15)

    rid = runlog.record(source="engine_chat", client="test", model="clozn-qwen (engine)",
                        messages=[{"role": "user", "content": "hi"}], response="The cat sat",
                        trace=steps, workspace_provider=provider)
    readouts = runlog.get_run(rid)["trace"]["workspace_readouts"]
    assert len(readouts) == 3
    assert readouts[0]["type"] == "workspace_readout"
    assert readouts[0]["run_id"] == rid
    assert readouts[0]["provider"] == "engine_concepts"
    assert readouts[0]["provider_type"] == "engine_concepts"
    assert readouts[0]["readout_kind"] == "concept"
    assert {"label", "score"} <= set(readouts[0]["top_readouts"][0])
    assert all("entropy" in r for r in readouts)


def test_ready_workspace_readouts_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    ready = {
        "tokens": ["x"],
        "confidence": [0.8],
        "workspace_readouts": [{
            "type": "workspace_readout",
            "run_id": "external",
            "token_index": 0,
            "token_text": "x",
            "layer": 1,
            "position": 0,
            "top_readouts": [{"label": "uncertainty", "score": 0.2}],
            "entropy": 0.1,
            "provider": "fixture",
        }],
    }
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "q"}], response="x", trace=ready)
    readout = runlog.get_run(rid)["trace"]["workspace_readouts"][0]
    assert readout["provider"] == "fixture"
    assert readout["provider_type"] == "mock"
    assert readout["readout_kind"] == "risk"


def test_record_low_confidence_flag_from_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    steps = [{"piece": "um", "conf": 0.1, "alts": []}]                   # very unsure
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "q"}],
                        response="um", trace=steps)
    assert "low-confidence" in runlog.get_run(rid)["flags"]


def test_record_hf_chat_leaves_trace_empty(tmp_path, monkeypatch):
    """The HF chat paths pass no trace -> the stored run has a clean empty {} (documented behavior)."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    rid = runlog.record(source="studio_chat", messages=[{"role": "user", "content": "hey"}],
                        response="hello")                                # no trace= at all
    assert runlog.get_run(rid)["trace"] == {}


# --------------------------------------------------------------------------- finish_reason on the record

def test_record_persists_finish_reason_and_truncated_flag(tmp_path, monkeypatch):
    """finish_reason is stored on the run, and 'length' (cut off at the token cap) raises a 'truncated' flag."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    rid = runlog.record(source="engine_chat", messages=[{"role": "user", "content": "hi"}],
                        response="The cat s", finish_reason="length")
    run = runlog.get_run(rid)
    assert run["finish_reason"] == "length"
    assert "truncated" in run["flags"]


def test_record_stop_is_not_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    rid = runlog.record(source="engine_chat", messages=[{"role": "user", "content": "hi"}],
                        response="Done.", finish_reason="stop")
    run = runlog.get_run(rid)
    assert run["finish_reason"] == "stop"
    assert "truncated" not in run["flags"]


def test_record_without_finish_reason_stores_none(tmp_path, monkeypatch):
    """The HF paths don't compute a stop cause -> the field is present and None, never fabricated."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    rid = runlog.record(source="studio_chat", messages=[{"role": "user", "content": "hey"}], response="hi")
    assert runlog.get_run(rid)["finish_reason"] is None


def test_record_persists_meta_block(tmp_path, monkeypatch):
    """Reproducibility metadata (model_file/quant/mode) rides the run record; absent -> a clean {}."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path))
    meta = {"model_file": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", "quant": "Q4_K_M", "mode": "autoregressive"}
    rid = runlog.record(source="engine_chat", messages=[{"role": "user", "content": "hi"}],
                        response="hey", meta=meta)
    assert runlog.get_run(rid)["meta"] == meta
    rid2 = runlog.record(source="studio_chat", messages=[{"role": "user", "content": "x"}], response="y")
    assert runlog.get_run(rid2)["meta"] == {}          # no meta -> clean empty, never fabricated
