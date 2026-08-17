"""Durable storage for derived Minimal Context search results.

Direct experimental evidence remains in :class:`ObservationStore`.  This
store contains only the immutable, reference-bearing search result document
and uses the existing runs SQLite/blob conventions.
"""
from __future__ import annotations

from contextlib import closing
import json
import time
from collections.abc import Mapping
from typing import Any

from clozn import schemas
from clozn.experiments.state import canonical_json
from clozn.runs import store as run_store


class MinimalContextResultStoreError(ValueError):
    """A derived Minimal Context result could not be stored or read safely."""


class MinimalContextResultIntegrityError(MinimalContextResultStoreError):
    """An immutable result ID was presented with different content."""


def _result_type():
    # Avoid importing the recipe during module initialization; the recipe owns
    # the result model and does not depend on this store.
    from .minimal_context import MinimalContextResult
    return MinimalContextResult


class MinimalContextResultStore:
    """SQLite index plus content-addressed JSON result artifacts."""

    def __init__(self, *, runs_store: Any = run_store):
        self.runs_store = runs_store

    def _ensure(self) -> None:
        self.runs_store._ensure()
        # Derived result documents are an additive artifact family, like the
        # existing standalone replay-result stores.  Keep their table
        # bootstrap local so the shared run migration ledger remains stable.
        with closing(self.runs_store._connect()) as db, db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS minimal_context_results (
                    result_id TEXT PRIMARY KEY,
                    search_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    base_execution_fingerprint TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS minimal_context_results_run_idx "
                "ON minimal_context_results(run_id, created_ts DESC, result_id DESC)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS minimal_context_results_search_idx "
                "ON minimal_context_results(search_id, created_ts DESC, result_id DESC)"
            )

    def _load_row(self, row) -> Any:
        if row is None:
            return None
        ref = json.loads(row["payload_json"])
        artifact = self.runs_store._load_blob(ref, kind="minimal context result")
        if not isinstance(artifact, Mapping) or artifact.get("unavailable"):
            raise MinimalContextResultStoreError(
                f"result artifact {row['result_id']!r} is unavailable"
            )
        try:
            result = _result_type().from_dict(artifact)
            schemas.validate(dict(artifact), "clozn.minimal-context-search-result.v2")
        except Exception as exc:
            raise MinimalContextResultStoreError(
                f"result artifact {row['result_id']!r} failed validation"
            ) from exc
        if result.result_id != row["result_id"]:
            raise MinimalContextResultIntegrityError("result artifact ID disagrees with its SQLite row")
        if result.search_id != row["search_id"] or result.run_id != row["run_id"]:
            raise MinimalContextResultIntegrityError("result artifact binding disagrees with its SQLite row")
        if result.base_execution_fingerprint != row["base_execution_fingerprint"]:
            raise MinimalContextResultIntegrityError("result artifact fingerprint disagrees with its SQLite row")
        return result

    @staticmethod
    def _document(value: Any) -> tuple[Any, dict[str, Any]]:
        result_type = _result_type()
        result = value if isinstance(value, result_type) else result_type.from_dict(value)
        document = result.to_dict()
        schemas.validate(document, "clozn.minimal-context-search-result.v2")
        return result, document

    def put(self, value: Any, *, now: float | None = None) -> str:
        result, document = self._document(value)
        self._ensure()
        artifact_ref = self.runs_store._store_blob(document, kind="minimal context result")
        if artifact_ref.get("write_failed"):
            raise MinimalContextResultStoreError("minimal context result artifact could not be written")
        payload_json = canonical_json(artifact_ref)
        stamp = float(now if now is not None else time.time())
        with closing(self.runs_store._connect()) as db, db:
            row = db.execute(
                "SELECT * FROM minimal_context_results WHERE result_id = ?",
                (result.result_id,),
            ).fetchone()
            if row is not None:
                prior = self._load_row(row)
                if prior.to_json() != result.to_json():
                    raise MinimalContextResultIntegrityError(
                        "result_id already names a different immutable result"
                    )
                return result.result_id
            db.execute(
                "INSERT INTO minimal_context_results "
                "(result_id, search_id, run_id, base_execution_fingerprint, created_ts, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (result.result_id, result.search_id, result.run_id,
                 result.base_execution_fingerprint, stamp, payload_json),
            )
        return result.result_id

    def get(self, result_id: str) -> Any | None:
        if not isinstance(result_id, str) or not result_id:
            return None
        self._ensure()
        with closing(self.runs_store._connect()) as db:
            row = db.execute(
                "SELECT * FROM minimal_context_results WHERE result_id = ?", (result_id,)
            ).fetchone()
        return self._load_row(row)

    def list_for_run(self, run_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        if not isinstance(run_id, str) or not run_id:
            return []
        self._ensure()
        bounded = max(0, min(int(limit), 200))
        with closing(self.runs_store._connect()) as db:
            rows = db.execute(
                "SELECT * FROM minimal_context_results WHERE run_id = ? "
                "ORDER BY created_ts DESC, result_id DESC LIMIT ?",
                (run_id, bounded),
            ).fetchall()
        summaries: list[dict[str, Any]] = []
        for row in rows:
            result = self._load_row(row)
            summaries.append({
                "schema_version": "clozn.minimal-context-search-result.v2",
                "result_id": result.result_id,
                "run_id": result.run_id,
                "search_id": result.search_id,
                "status": result.status,
                "search_status": result.search_status,
                "certificate": result.certificate,
                "stopping_reason": result.stopping_reason,
                "best": result.best.to_dict() if result.best else None,
                "reduction": dict(result.reduction),
                "base_execution_fingerprint": result.base_execution_fingerprint,
            })
        return summaries

    def latest_for_search(self, search_id: str) -> Any | None:
        if not isinstance(search_id, str) or not search_id:
            return None
        self._ensure()
        with closing(self.runs_store._connect()) as db:
            row = db.execute(
                "SELECT * FROM minimal_context_results WHERE search_id = ? "
                "ORDER BY created_ts DESC, result_id DESC LIMIT 1", (search_id,)
            ).fetchone()
        return self._load_row(row)


def current_binding(result: Any, run: Mapping[str, Any] | None) -> dict[str, Any]:
    """Classify a stored result against the currently loaded parent Run.

    Historical results remain readable when this returns ``stale`` or
    ``run_unavailable``; the route layer uses the classification to gate
    actions, not to hide the historical document.
    """
    if run is None:
        return {"status": "run_unavailable", "reason": "the recorded Run is unavailable"}
    if not isinstance(run, Mapping) or run.get("id") != result.run_id:
        return {"status": "stale", "reason": "result belongs to another Run"}
    try:
        from clozn.experiments.state import ExecutionState
        state = ExecutionState.from_run(run)
    except Exception as exc:
        return {"status": "stale", "reason": f"current execution binding is unavailable: {exc}"}
    if state.execution_fingerprint != result.base_execution_fingerprint:
        return {"status": "stale", "reason": "the current execution fingerprint changed"}
    try:
        from clozn.runs.context_search_universe import plan_context_search_universe
        policy = result.universe.get("policy") if isinstance(result.universe, Mapping) else {}
        max_units = int(policy.get("max_units", len(result.universe.get("source_ids") or [])))
        current = plan_context_search_universe(run, run.get("context_units"), max_units=max_units)
    except Exception as exc:
        return {"status": "stale", "reason": f"the current Context Search Universe is unavailable: {exc}"}
    if current.get("universe_id") != result.universe.get("universe_id"):
        return {"status": "stale", "reason": "the current Context Search Universe changed"}
    if list(current.get("source_ids") or []) != list(result.universe.get("source_ids") or []):
        return {"status": "stale", "reason": "the current source universe changed"}
    return {"status": "current", "reason": None}


__all__ = [
    "MinimalContextResultIntegrityError", "MinimalContextResultStore",
    "MinimalContextResultStoreError", "current_binding",
]
