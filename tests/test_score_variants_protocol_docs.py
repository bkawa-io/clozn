"""Keep the documented related-prompt wire identity aligned across language boundaries."""

import json
from pathlib import Path

from clozn_engine._score_variants_contract import (
    SCORE_VARIANTS_CAPABILITY,
    SCORE_VARIANTS_ENDPOINT,
    SCORE_VARIANTS_EVIDENCE_USE,
    SCORE_VARIANTS_NUMERICAL_REGIME,
    SCORE_VARIANTS_QUALIFIED,
    SCORE_VARIANTS_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "protocol/fixtures/score_variants.json"


def test_score_variants_wire_identity_is_documented_and_shared():
    document = (ROOT / "protocol/SCORE_VARIANTS.md").read_text(encoding="utf-8")
    readme = (ROOT / "protocol/README.md").read_text(encoding="utf-8")
    generated_header = (
        ROOT / "engine/core/include/clozn/score_variants_contract.generated.hpp"
    ).read_text(
        encoding="utf-8"
    )
    route = (ROOT / "engine/core/serve/routes_score_variants.cpp").read_text(encoding="utf-8")
    client = (ROOT / "engine/client/clozn_engine/__init__.py").read_text(encoding="utf-8")
    handshake = json.loads((ROOT / "protocol/fixtures/handshake.json").read_text(encoding="utf-8"))
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))["contract"]

    assert contract["schema_version"] == SCORE_VARIANTS_SCHEMA
    assert contract["endpoint"] == SCORE_VARIANTS_ENDPOINT
    assert contract["capability"] == SCORE_VARIANTS_CAPABILITY
    assert f'SCHEMA_VERSION = "{SCORE_VARIANTS_SCHEMA}"' in generated_header
    assert f'ENDPOINT = "{SCORE_VARIANTS_ENDPOINT}"' in generated_header
    assert "from ._score_variants_contract import" in client
    assert "score_variants_contract::ENDPOINT" in route
    assert SCORE_VARIANTS_CAPABILITY in handshake["capabilities"]
    assert SCORE_VARIANTS_SCHEMA in document
    assert f"`POST {SCORE_VARIANTS_ENDPOINT}`" in document
    assert "SCORE_VARIANTS.md" in readme


def test_score_variants_document_preserves_screening_only_invariant():
    document = (ROOT / "protocol/SCORE_VARIANTS.md").read_text(encoding="utf-8")
    generated_header = (
        ROOT / "engine/core/include/clozn/score_variants_contract.generated.hpp"
    ).read_text(
        encoding="utf-8"
    )
    route = (ROOT / "engine/core/serve/routes_score_variants.cpp").read_text(encoding="utf-8")

    assert SCORE_VARIANTS_NUMERICAL_REGIME == "screening_unqualified"
    assert SCORE_VARIANTS_EVIDENCE_USE == "screening"
    assert SCORE_VARIANTS_QUALIFIED is False
    for literal in ("screening_unqualified", "screening"):
        assert literal in document
        assert literal in generated_header
    for symbol in ("NUMERICAL_REGIME", "EVIDENCE_USE", "QUALIFIED"):
        assert f"score_variants_contract::{symbol}" in route
    assert "trusted observations" in document
    assert "scalar-remeasured" in document


def test_generated_score_variants_contract_is_current():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/protocol/generate_score_variants_contract.py"),
         "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
