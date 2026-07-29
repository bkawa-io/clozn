from pathlib import Path

from scripts.docs import check_drift


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_clean_current_docs_pass(tmp_path):
    _write(tmp_path / "README.md", "# Product\n\nSee [capabilities](docs/CAPABILITIES.md).\n")
    _write(tmp_path / "docs" / "CAPABILITIES.md", "# Capabilities\n\nhttps://github.com/bkawa-io/clozn\n")

    assert check_drift.check_docs(tmp_path) == []
    assert check_drift.main([str(tmp_path)]) == 0


def test_high_confidence_current_user_drift_is_reported(tmp_path):
    _write(
        tmp_path / "README.md",
        """# Product

Run `clozn lab qwen`.
Old home: https://github.com/old-owner/clozn
See [missing](docs/MISSING.md).
""",
    )
    _write(
        tmp_path / "docs" / "ADAPTERS.md",
        """# Adapters

## Not yet built

- `clozn diff-adapter` compares the two arms.
""",
    )

    violations = check_drift.check_docs(tmp_path)
    assert {violation.code for violation in violations} == {
        "adapter-contradiction",
        "broken-link",
        "obsolete-owner",
        "removed-command",
    }
    assert check_drift.main([str(tmp_path)]) == 1


def test_archival_allowlist_preserves_historical_claims(tmp_path):
    _write(tmp_path / "README.md", "# Current\n")
    _write(
        tmp_path / "docs" / "DESIGN.md",
        """# Historical design

`clozn memory`, https://github.com/old-owner/clozn, [deleted](old/path.md)

## Not yet built
- `clozn diff-adapter`
""",
    )
    _write(
        tmp_path / "docs" / "research" / "old.md",
        "`clozn lab qwen` and [a deleted result](missing.json)\n",
    )

    assert check_drift.check_docs(tmp_path) == []


def test_repository_current_user_docs_are_clean():
    root = Path(__file__).resolve().parents[1]
    assert check_drift.check_docs(root) == []
