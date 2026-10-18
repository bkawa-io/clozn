"""Model-free contracts for the live v1 acceptance harness."""
from __future__ import annotations

from pathlib import Path
import sys


SMOKE_DIR = Path(__file__).resolve().parents[1] / "scripts" / "smoke"
sys.path.insert(0, str(SMOKE_DIR))
try:
    import v1_acceptance as smoke
finally:
    sys.path.remove(str(SMOKE_DIR))


class FakeClient:
    def __init__(self):
        self.materializations = 0

    def post(self, path, body):
        if path.endswith("/minimal-context/jobs"):
            return 202, {"job_id": "job_one", "state": "queued"}
        if path.endswith("/materialize"):
            self.materializations += 1
            return 201, {
                "state": "completed",
                "child_run_id": "run_child",
            }
        raise AssertionError(f"unexpected POST {path}: {body}")

    def get(self, path):
        if path.endswith("/minimal-context/jobs/job_one"):
            return 200, {
                "state": "completed",
                "result": {
                    "schema_version": "clozn.minimal-context-search-result.v2",
                    "result_id": "mcres_one",
                    "status": "completed",
                    "certificate": "INCLUSION_MINIMUM",
                    "stopping_reason": "inclusion_minimum_proven",
                    "universe": {"source_ids": ["src_a", "src_b"]},
                    "best": {
                        "retained_source_ids": ["src_a"],
                        "removed_source_ids": ["src_b"],
                        "observation_status": "exact_preserved",
                    },
                    "reduction": {
                        "original_prompt_token_cost": 20,
                        "retained_prompt_token_cost": 12,
                        "percent": 40.0,
                    },
                },
            }
        if path.endswith("/minimal-context/results"):
            return 200, {"results": [{"result_id": "mcres_one"}]}
        if path.endswith("/minimal-context/results/mcres_one"):
            return 200, {
                "result_id": "mcres_one",
                "current_binding": {"status": "current", "reason": None},
            }
        if path == "/runs/run_parent":
            return 200, {"id": "run_parent", "response": "answer"}
        if path == "/runs/run_child":
            return 200, {
                "id": "run_child", "parent_run_id": "run_parent", "response": "answer",
            }
        raise AssertionError(f"unexpected GET {path}")


def test_minimal_context_acceptance_requires_durable_idempotent_child():
    client = FakeClient()

    report = smoke._minimal_context(
        client, run_id="run_parent", expected_source_ids=["src_a", "src_b"],
        job_timeout=1,
    )

    assert client.materializations == 2
    assert report == {
        "result_id": "mcres_one",
        "certificate": "INCLUSION_MINIMUM",
        "stopping_reason": "inclusion_minimum_proven",
        "source_count": 2,
        "retained_source_count": 1,
        "removed_source_count": 1,
        "original_prompt_tokens": 20,
        "retained_prompt_tokens": 12,
        "reduction_percent": 40.0,
        "child_run_id": "run_child",
    }
