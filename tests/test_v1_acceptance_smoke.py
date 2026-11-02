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
    def post(self, path, body):
        assert body == {"max_new_executions": 16}
        if path.endswith("/context-investigation/lean-context/jobs"):
            return 202, {"job_id": "job_one", "state": "queued"}
        raise AssertionError(f"unexpected POST {path}: {body}")

    def get(self, path):
        if path.endswith("/context-investigation/lean-context/jobs/job_one"):
            return 200, {
                "state": "completed",
                "result": {
                    "schema_version": "clozn.lean-context-result.v1",
                    "search_id": "search_one",
                    "status": "completed",
                    "universe": {"source_ids": ["src_a", "src_b"]},
                    "best": {
                        "retained_source_ids": ["src_a"],
                        "removed_source_ids": ["src_b"],
                        "rendered_prompt_token_cost": 12,
                    },
                    "original": {"rendered_prompt_token_cost": 20},
                    "reduction": {
                        "removed_prompt_tokens": 8,
                    },
                    "generation_comparison": {
                        "status": "completed",
                        "full_context": {"text": "answer"},
                        "lean_context": {"text": "answer"},
                        "exact_match": True,
                    },
                    "budget": {"used_new_executions": 3},
                },
            }
        raise AssertionError(f"unexpected GET {path}")


def test_lean_context_acceptance_requires_reduction_and_fresh_answers():
    report = smoke._lean_context(
        FakeClient(), run_id="run_parent", expected_source_ids=["src_a", "src_b"],
        job_timeout=1,
    )

    assert report == {
        "search_id": "search_one",
        "source_count": 2,
        "retained_source_count": 1,
        "removed_source_count": 1,
        "original_prompt_tokens": 20,
        "retained_prompt_tokens": 12,
        "reduction_percent": 40.0,
        "used_new_executions": 3,
        "fresh_answers_exact_match": True,
    }
