from __future__ import annotations

import json

import pytest

from clozn.experiments.shared_parent import (
    SharedParentParityError,
    SharedParentSessionClient,
    SharedParentSessionError,
    assert_evidence_parity,
)


class _Engine:
    def __init__(self):
        self.probes = []
        self.promotions = []
        self.closed = []
        self.version = 0

    def reference_match_persistent_create(self, prompt, *, reference_token_ids, generation_contract):
        assert prompt == "PARENT"
        assert list(reference_token_ids) == [1, 2]
        assert generation_contract["decode_mode"] == "greedy"
        return {
            "session_id": "session-1",
            "parent_version": 0,
            "parent_prompt_digest": "parent-digest",
            "runtime_identity": {"worker_generation_id": "worker-1", "n_batch": 256, "n_ubatch": 256},
            "telemetry": {"parent_refill_rows_after_initial_create": 0},
        }

    def reference_match_persistent_probe(self, session_id, *, expected_parent_version, children):
        assert session_id == "session-1"
        assert expected_parent_version == self.version
        self.probes.append([dict(child) for child in children])
        return {
            "results": [
                {"candidate_id": child["candidate_id"], "candidate_rank": child["candidate_rank"],
                 "native_preserves": child["candidate_rank"] == 0}
                for child in children
            ],
            "telemetry": {"probe_count": len(children)},
        }

    def reference_match_persistent_promote(self, session_id, *, expected_parent_version, candidate_id):
        assert session_id == "session-1"
        assert expected_parent_version == self.version
        self.promotions.append(candidate_id)
        self.version += 1
        return {"parent_version": self.version, "parent_prompt_digest": f"digest-{self.version}",
                "telemetry": {"promoted_child_count": self.version}}

    def reference_match_persistent_close(self, session_id):
        self.closed.append(session_id)
        return {"closed": True}


CONTRACT = {
    "decode_mode": "greedy",
    "sampling": None,
    "max_new": 3,
    "stop": [],
    "expected_termination": {"reason": "eos", "reason_raw": "eos"},
}


def _client():
    engine = _Engine()
    client = SharedParentSessionClient(engine, (1, 2), CONTRACT)
    return client, engine


def _children():
    return [
        {"candidate_id": "child-a", "candidate_rank": 1, "prompt": "A"},
        {"candidate_id": "child-b", "candidate_rank": 0, "prompt": "B"},
    ]


def test_create_establishes_deterministic_parent_version_zero():
    client, _engine = _client()
    response = client.create("PARENT")
    assert response["parent_version"] == 0
    assert client.parent_version == 0


def test_probe_binds_expected_parent_version_and_rejects_stale():
    client, engine = _client()
    client.create("PARENT")
    with pytest.raises(SharedParentSessionError) as exc:
        client.probe_round(_children(), expected_parent_version=7)
    assert exc.value.code == "stale_parent_state"
    assert engine.probes == []


def test_failed_round_leaves_parent_identity_unchanged():
    client, _engine = _client()
    client.create("PARENT")
    before = client.report()
    client.probe_round(_children())
    after = client.report()
    assert after["parent_version"] == before["parent_version"] == 0
    assert after["parent_prompt_digest"] == before["parent_prompt_digest"]


def test_explicit_promotion_increments_parent_version_without_client_selection():
    client, engine = _client()
    client.create("PARENT")
    client.probe_round(_children())
    client.promote("child-b", exact_preserved=True)
    assert client.parent_version == 1
    assert engine.promotions == ["child-b"]


def test_old_child_cannot_be_promoted_after_parent_changes_and_double_promotion_rejected():
    client, _engine = _client()
    client.create("PARENT")
    client.probe_round(_children())
    client.promote("child-b", exact_preserved=True)
    with pytest.raises(SharedParentSessionError) as exc:
        client.promote("child-a", exact_preserved=True)
    assert exc.value.code == "stale_candidate"
    with pytest.raises(SharedParentSessionError) as exc:
        client.promote("child-b", exact_preserved=True)
    assert exc.value.code == "stale_candidate"


def test_close_invalidates_later_operations():
    client, engine = _client()
    client.create("PARENT")
    client.close()
    assert engine.closed == ["session-1"]
    with pytest.raises(SharedParentSessionError) as exc:
        client.probe_round(_children())
    assert exc.value.code == "session_closed"


def test_cancellation_does_not_logically_promote_a_child():
    client, engine = _client()
    client.create("PARENT")
    client.probe_round(_children())
    client.cancel_round()
    assert client.parent_version == 0
    assert engine.promotions == []
    with pytest.raises(SharedParentSessionError):
        client.promote("child-b", exact_preserved=True)


def test_unverified_evidence_prevents_promotion():
    client, engine = _client()
    client.create("PARENT")
    client.probe_round(_children())
    with pytest.raises(SharedParentSessionError) as exc:
        client.promote("child-b", exact_preserved=False)
    assert exc.value.code == "promotion_requires_exact_preservation"
    assert engine.promotions == []


def test_native_scalar_disagreement_surfaces_typed_parity_failure():
    with pytest.raises(SharedParentParityError) as exc:
        assert_evidence_parity([{"status": "matched", "matched_token_count": 2}],
                               [{"status": "diverged", "matched_token_count": 1}])
    assert exc.value.code == "shared_parent_parity_failure"
    assert exc.value.mismatches[0]["arm_index"] == 0


def test_candidate_rank_is_supplied_by_reducer_and_not_invented_by_runtime():
    client, engine = _client()
    client.create("PARENT")
    children = [
        {"candidate_id": "ordered-late", "candidate_rank": 9, "prompt": "L"},
        {"candidate_id": "ordered-early", "candidate_rank": 3, "prompt": "E"},
    ]
    client.probe_round(children)
    assert engine.probes[-1] == children


def test_report_serialization_is_deterministic():
    client, _engine = _client()
    client.create("PARENT")
    payload_a = json.dumps(client.report(), sort_keys=True, separators=(",", ":"))
    payload_b = json.dumps(client.report(), sort_keys=True, separators=(",", ":"))
    assert payload_a == payload_b
