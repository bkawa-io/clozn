"""Model-free tests for the static, versioned hook/intervention vocabulary (roadmap Phase 4.2)."""
from __future__ import annotations

from clozn.experiments.stats import REPLAY_CLASSES
from clozn.receipts.hook_vocabulary import SCHEMA, hook_vocabulary


def test_schema_is_versioned():
    doc = hook_vocabulary()
    assert doc["schema"] == SCHEMA == "clozn.hook_vocabulary.v1"


def test_returns_a_fresh_copy_every_call():
    first = hook_vocabulary()
    first["hooks"].append({"tampered": True})
    first["replay_classes"]["vocabulary"].append("invented")
    second = hook_vocabulary()
    assert len(second["hooks"]) == 4
    assert "invented" not in second["replay_classes"]["vocabulary"]


def test_names_the_four_eval_callback_interception_points():
    doc = hook_vocabulary()
    names = {hook["name"] for hook in doc["hooks"]}
    assert names == {"l_out-<il>", "kq_soft_max-<il>", "kqv_out-<il>", "ffn_out-<il>"}


def test_l_out_documents_the_layer_zero_read_write_asymmetry():
    l_out = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "l_out-<il>")
    assert "final via embeddings" in l_out["read"]["layer_zero_sentinel"]
    assert "[1, n_layer)" in l_out["write"]["layer_range"]
    assert "REJECTED" in l_out["write"]["layer_range"]


def test_l_out_capture_plane_range_and_known_gap_are_cited():
    l_out = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "l_out-<il>")
    assert "1..n_layer-1" in l_out["read"]["capture_plane_layer_range"]
    assert "n_layer-2" in l_out["read"]["known_gap_last_layer"]
    assert "PRE-edit state" in l_out["read"]["write_capture_interaction"]


def test_l_out_pre_post_norm_position_is_honestly_unspecified():
    l_out = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "l_out-<il>")
    assert "UNSPECIFIED" in l_out["read"]["pre_post_norm_position"]


def test_kq_soft_max_documents_flash_attn_gate_and_shared_capability_flag():
    kq = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "kq_soft_max-<il>")
    assert "--no-flash-attn" in kq["materialization_constraint"]
    assert "capabilities.attn_knockout" in kq["capability_flag"]
    assert "no separate advertised capability" in kq["capability_flag"]


def test_kq_soft_max_layer_ranges_differ_between_knockout_and_l_out_write():
    doc = hook_vocabulary()
    kq = next(hook for hook in doc["hooks"] if hook["name"] == "kq_soft_max-<il>")
    l_out = next(hook for hook in doc["hooks"] if hook["name"] == "l_out-<il>")
    assert "[0, n_layer)" in kq["knockout"]["layer_range"]
    assert "IS valid here" in kq["knockout"]["layer_range"]
    assert "[1, n_layer)" in l_out["write"]["layer_range"]


def test_kq_soft_max_renormalize_default_discrepancy_is_flagged():
    kq = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "kq_soft_max-<il>")
    text = kq["knockout"]["renormalize_default_discrepancy"]
    assert "FALSE" in text and "TRUE" in text


def test_kq_soft_max_client_head_gap_is_flagged():
    kq = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "kq_soft_max-<il>")
    assert "NO `head` field" in kq["knockout"]["client_head_gap"]


def test_kqv_out_documents_additive_composition_and_no_flash_attn_gate():
    kqv = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "kqv_out-<il>")
    assert "CONTRIBUTES" in kqv["stream_semantics"]
    assert "COMPOSE" in kqv["stream_semantics"]
    assert "additive_writes_probe.py" in kqv["stream_semantics"]
    assert "flash attention ON" in kqv["materialization_constraint"]


def test_kqv_out_head_write_layer_and_head_ranges_are_cited():
    kqv = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "kqv_out-<il>")
    assert "[0, n_layer)" in kqv["head_write"]["layer_range"]
    assert "[0, n_head)" in kqv["head_write"]["head_range"]


def test_kqv_out_head_write_documents_the_2026_07_28_honesty_fix():
    kqv = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "kqv_out-<il>")
    text = kqv["head_write"]["honesty_fix_2026_07_28"]
    assert "head_write_applied\": true" in text
    assert "silently dropped" in text
    assert "head_write_landed()" in text
    assert "400" in text


def test_ffn_out_is_named_and_documents_additive_composition():
    doc = hook_vocabulary()
    names = {hook["name"] for hook in doc["hooks"]}
    assert "ffn_out-<il>" in names
    ffn = next(hook for hook in doc["hooks"] if hook["name"] == "ffn_out-<il>")
    assert "CONTRIBUTES" in ffn["stream_semantics"]
    assert "COMPOSE" in ffn["stream_semantics"]
    assert "test_ffn_hook.cpp" in ffn["stream_semantics"]
    assert "ffn_hook_probe.py" in ffn["stream_semantics"]


def test_ffn_out_architecture_coverage_names_present_and_absent_families():
    ffn = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "ffn_out-<il>")
    text = ffn["architecture_coverage"]
    assert "qwen2.cpp" in text and "deepseek2.cpp" in text
    assert "mamba" in text and "rwkv" in text
    assert "UNSPECIFIED" in text


def test_ffn_out_write_layer_range_includes_zero_unlike_residual_write():
    ffn = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "ffn_out-<il>")
    assert "[0, n_layer)" in ffn["ffn_write"]["layer_range"]
    assert "STATICALLY" in ffn["ffn_write"]["layer_range"]


def test_ffn_out_write_tracks_landed_state_for_architecture_coverage_gap():
    ffn = next(hook for hook in hook_vocabulary()["hooks"] if hook["name"] == "ffn_out-<il>")
    text = ffn["ffn_write"]["landed_tracking"]
    assert "ffn_write_landed()" in text
    assert "ffn_write_applied:true" in text or "ffn_write_applied" in text


def test_stream_semantics_summary_makes_overwrite_vs_contribute_impossible_to_miss():
    doc = hook_vocabulary()
    summary = doc["stream_semantics_summary"]
    rows = {row["hook"]: row for row in summary["table"]}
    assert rows["l_out-<il>"]["semantics"] == "OVERWRITES"
    assert rows["l_out-<il>"]["composes_across_layers"] is False
    assert rows["kqv_out-<il>"]["semantics"] == "CONTRIBUTES (additive)"
    assert rows["kqv_out-<il>"]["composes_across_layers"] is True
    assert rows["ffn_out-<il>"]["semantics"] == "CONTRIBUTES (additive)"
    assert rows["ffn_out-<il>"]["composes_across_layers"] is True
    # top-level, not buried inside one hook's entry -- a skimming reader must be able to find it
    assert "stream_semantics_summary" in doc


def test_checkpoint_restore_mechanism_finding_is_honest():
    cb = hook_vocabulary()["checkpoint_branch"]
    assert "does NOT call GgmlAdapter::load_checkpoint" in cb["restore"]["mechanism_finding"]
    assert "KV-blob fast restore" in cb["restore"]["mechanism_finding"]
    assert "bit-identical GREEDY" in cb["restore"]["adapter_correctness_bar"]


def test_branch_batched_kv_usage_is_marked_unspecified():
    cb = hook_vocabulary()["checkpoint_branch"]
    assert "UNSPECIFIED" in cb["branch"]["mechanism_finding"]


def test_replay_classes_reuse_stats_vocabulary_exactly():
    section = hook_vocabulary()["replay_classes"]
    assert tuple(section["vocabulary"]) == REPLAY_CLASSES
    assert set(section["operation_classes"]) == {
        "score.teacher_forced", "intervention.attention_knockout",
        "capture.residual.layer_output", "intervention.residual.replace_rows",
        "checkpoint_restore_or_branch_default_greedy", "checkpoint_restore_or_branch_sampled",
    }
    for label in section["operation_classes"].values():
        assert label.split(" -- ")[0].strip() in REPLAY_CLASSES


def test_replay_classes_distinguished_from_client_enum():
    section = hook_vocabulary()["replay_classes"]
    text = section["distinct_from_client_replay_class"]
    assert "request_replay" in text and "re_prefilled" in text
    assert "must not be conflated" in text


def test_not_covered_lists_the_unread_state_route():
    not_covered = hook_vocabulary()["not_covered"]
    assert any("/state" in item for item in not_covered)
    assert any("head" in item.lower() for item in not_covered)
    assert any("steer" in item.lower() for item in not_covered)


def test_ar_mode_gate_is_documented():
    doc = hook_vocabulary()
    assert "autoregressive" in doc["gate"]["ar_mode_required"]


def test_document_is_json_serializable():
    import json
    json.dumps(hook_vocabulary())
