#!/usr/bin/env python3
"""Deterministic Minimal Context launch demo.

This fixture exercises the real bounded solver and certificate accounting in
the benchmark harness. It prints measured candidate/probe counts rather than
shipping a screenshot or asserting that a 50 -> 3 result exists.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bench.minimal_context_multi_arm import run_case  # noqa: E402


def main() -> int:
    row = run_case(
        source_count=50, prompt_size="medium", answer_size="short",
        mode="exact", execution="batch", search_seed=0,
    )
    print(json.dumps({
        "demo": "minimal_context_50_to_3",
        "status": row["status"],
        "source_count": row["source_count"],
        "candidate_retained_source_count": row["candidate_retained_source_count"],
        "certificate_kind": row["certificate_kind"],
        "probe_count": row["probe_count"],
        "search_probe_count": row["search_probe_count"],
        "certification_probe_count": row["certification_probe_count"],
        "lower_cardinality_candidate_count": row["lower_cardinality_candidate_count"],
        "time_to_exact_certificate_s": row["time_to_exact_certificate_s"],
        "cache_reuse_rate": row["cache_reuse_rate"],
        "result_id": row["result_id"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
