"""Focused tests for the verified-child replay seam retained by v1."""
from __future__ import annotations

import pytest

from clozn.replay import replay as replay_child
import clozn.runs.store as runlog


RUN = {
    "id": "run_parent0",
    "client": "studio",
    "model": "fixture-model",
    "substrate": "engine",
    "messages": [{"role": "user", "content": "original"}],
}
CHANGES = {
    "experiment": {
        "experiment_id": "exp_fixture",
        "arm_id": "arm_fixture",
        "observation_id": "obs_fixture",
        "intervention": {"kind": "delete_source", "source_ids": ["src_fixture"]},
    },
    "greedy": True,
}


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))


class FakeSubstrate:
    def __init__(self):
        self.calls = []

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls.append({
            "messages": [dict(message) for message in messages],
            "max_new": max_new,
            "sample": sample,
        })
        if trace_out is not None:
            trace_out.extend([{"piece": "kept", "id": 7, "conf": 0.9}])
        if mem_out is not None:
            mem_out.update(
                assembled_messages=[dict(message) for message in messages],
                final_prompt="rendered reduced prompt",
            )
        return "preserved answer"

    def last_finish_reason(self):
        return "stop"

    def run_meta(self):
        return {"sampler_mode": "greedy"}

    def identity_meta(self):
        return {"model_sha256": "a" * 64}


def test_verified_override_records_an_immutable_child(isolated_store):
    substrate = FakeSubstrate()
    messages = [{"role": "user", "content": "retained source and question"}]

    child = replay_child(
        RUN,
        CHANGES,
        substrate,
        messages_override=messages,
        sampling_override=False,
        max_new=19,
    )

    assert child is not None
    assert child["parent_run_id"] == RUN["id"]
    assert child["changes_applied"] == CHANGES
    assert child["messages"] == messages
    assert child["response"] == "preserved answer"
    assert child["finish_reason"] == "stop"
    assert child["trace"]["token_ids"] == [7]
    assert substrate.calls == [{"messages": messages, "max_new": 19, "sample": False}]
    assert RUN["messages"][0]["content"] == "original"


def test_recorded_sampling_contract_is_forwarded(isolated_store):
    substrate = FakeSubstrate()
    sampling = {
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "seed": 7,
    }
    child = replay_child(
        RUN,
        {"experiment": CHANGES["experiment"]},
        substrate,
        messages_override=[{"role": "user", "content": "reduced"}],
        sampling_override=sampling,
    )
    assert child is not None
    assert substrate.calls[0]["sample"] == sampling


@pytest.mark.parametrize("changes", [
    {},
    {"memory_off": True},
    {"exclude_sections": ["notes"]},
    {"experiment": CHANGES["experiment"], "behavior_off": True},
])
def test_removed_branch_controls_are_rejected(isolated_store, changes):
    assert replay_child(
        RUN,
        changes,
        FakeSubstrate(),
        messages_override=[{"role": "user", "content": "reduced"}],
        sampling_override=False,
    ) is None


def test_resolved_messages_and_decoding_contract_are_required(isolated_store):
    assert replay_child(RUN, CHANGES, FakeSubstrate(), sampling_override=False) is None
    assert replay_child(
        RUN,
        CHANGES,
        FakeSubstrate(),
        messages_override=[{"role": "user", "content": "reduced"}],
    ) is None


def test_generation_failure_does_not_create_a_child(isolated_store):
    class BrokenSubstrate(FakeSubstrate):
        def chat(self, *args, **kwargs):
            raise RuntimeError("worker stopped")

    assert replay_child(
        RUN,
        CHANGES,
        BrokenSubstrate(),
        messages_override=[{"role": "user", "content": "reduced"}],
        sampling_override=False,
    ) is None
    assert runlog.list_runs() == []
