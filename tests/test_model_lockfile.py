"""tests/test_model_lockfile.py -- clozn/models/lockfile.py (feature 02, GitHub Action for model-change
gating: the `clozn.model-lock.v1` lockfile parser/validator).

Model-free and network-free throughout: `load_lockfile` never opens a socket, so these tests never need
to mock one. Reuses the same fixtures tests/test_schema_contracts.py already walks under
tests/fixtures/schemas/clozn.model-lock.v1/ so the two suites never disagree about what a valid/invalid
lockfile looks like.
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
FIXTURES = os.path.join(HERE, "fixtures", "schemas", "clozn.model-lock.v1")

import pytest  # noqa: E402

from clozn.models.lockfile import LockfileError, load_lockfile, model_roles, pinned_model  # noqa: E402


def _fixture(name: str) -> str:
    return os.path.join(FIXTURES, name)


# ================================================================================================== load_lockfile

def test_load_lockfile_valid_two_models():
    document = load_lockfile(_fixture("valid__two_models.json"))
    assert document["schema_version"] == "clozn.model-lock.v1"
    assert set(document["models"]) == {"baseline", "candidate"}


def test_load_lockfile_valid_minimal_single_model():
    document = load_lockfile(_fixture("valid__minimal_single_model.json"))
    assert set(document["models"]) == {"baseline"}


def test_load_lockfile_missing_file():
    with pytest.raises(LockfileError, match="could not read lockfile"):
        load_lockfile(os.path.join(FIXTURES, "does-not-exist.json"))


def test_load_lockfile_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LockfileError, match="not valid JSON"):
        load_lockfile(str(path))


def test_load_lockfile_missing_sha256_rejected():
    with pytest.raises(LockfileError, match="does not conform"):
        load_lockfile(_fixture("invalid__missing_sha256.json"))


def test_load_lockfile_truncated_sha256_rejected():
    with pytest.raises(LockfileError, match="does not conform"):
        load_lockfile(_fixture("invalid__truncated_sha256.json"))


def test_load_lockfile_wrong_schema_version_rejected():
    with pytest.raises(LockfileError, match="does not conform"):
        load_lockfile(_fixture("invalid__wrong_schema_version.json"))


def test_load_lockfile_non_https_url_rejected():
    """Caught by the schema's own `pattern` today (this fixture never reaches load_lockfile's separate,
    defense-in-depth HTTPS check -- see that check's docstring), but the resulting LockfileError must
    still say plainly that the document did not conform."""
    with pytest.raises(LockfileError, match="does not conform"):
        load_lockfile(_fixture("invalid__non_https_url.json"))


def test_load_lockfile_explicit_https_check_fires_independently_of_the_schema_pattern(tmp_path, monkeypatch):
    """load_lockfile's own HTTPS check (distinct from the schema's `pattern`) is defense-in-depth: prove
    it actually fires, by making `clozn.schemas.validate` a no-op so only load_lockfile's own loop can
    catch a non-HTTPS url. Without this test, that second check could silently rot into dead code."""
    import clozn.models.lockfile as lockfile_module
    document = json.loads(open(_fixture("valid__minimal_single_model.json"), encoding="utf-8").read())
    document["models"]["baseline"]["url"] = "http://example.com/models/base.gguf"
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    monkeypatch.setattr(lockfile_module.schemas, "validate", lambda document, name=None: None)
    with pytest.raises(LockfileError, match="url must be HTTPS"):
        load_lockfile(str(path))


# =================================================================================================== model_roles

def test_model_roles_sorted():
    document = load_lockfile(_fixture("valid__two_models.json"))
    assert model_roles(document) == ["baseline", "candidate"]


def test_model_roles_empty_for_no_models():
    assert model_roles({"schema_version": "clozn.model-lock.v1", "models": {}}) == []


# ================================================================================================== pinned_model

def test_pinned_model_returns_the_named_role():
    document = load_lockfile(_fixture("valid__two_models.json"))
    entry = pinned_model(document, "candidate")
    assert entry["url"].startswith("https://")
    assert len(entry["sha256"]) == 64


def test_pinned_model_raises_for_unknown_role():
    document = load_lockfile(_fixture("valid__minimal_single_model.json"))
    with pytest.raises(LockfileError, match="no model pinned for role 'candidate'"):
        pinned_model(document, "candidate")
