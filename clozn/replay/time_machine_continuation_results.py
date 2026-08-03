"""Immutable terminal receipts for exact appended-turn Time Machine attempts.

Failures do not fabricate child runs.  Successful receipts are stored here and embedded in the
real immutable child, giving both outcomes the same durable lookup/audit surface.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
import sqlite3

from clozn import schemas


RESULTS_DIR = os.path.join(os.path.expanduser("~/.clozn"), "time-machine")


class TimeMachineContinuationResultError(RuntimeError):
    """A continuation receipt could not be stored without violating immutability."""


def _path() -> str:
    return os.path.join(RESULTS_DIR, "continuations.sqlite3")


def _connect() -> sqlite3.Connection:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db = sqlite3.connect(_path(), timeout=30.0)
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE IF NOT EXISTS time_machine_continuations ("
        " continuation_id TEXT PRIMARY KEY,"
        " requested_run_id TEXT NOT NULL,"
        " source_turn INTEGER NOT NULL,"
        " status TEXT NOT NULL,"
        " payload_json TEXT NOT NULL"
        ")"
    )
    return db


def save(receipt: dict) -> dict:
    """Insert one terminal receipt; a retry with the same ID must be byte-identical."""
    schemas.validate(receipt, "clozn.time-machine-continuation.v1")
    continuation_id = receipt.get("continuation_id")
    status = receipt.get("status")
    if not isinstance(continuation_id, str) or not continuation_id:
        raise TimeMachineContinuationResultError("receipt needs continuation_id")
    if status not in {"completed", "unavailable", "failed", "cancelled"}:
        raise TimeMachineContinuationResultError("only a terminal continuation receipt can be stored")
    payload = json.dumps(
        receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with closing(_connect()) as db, db:
        row = db.execute(
            "SELECT payload_json FROM time_machine_continuations WHERE continuation_id=?",
            (continuation_id,),
        ).fetchone()
        if row is not None:
            if row["payload_json"] != payload:
                raise TimeMachineContinuationResultError(
                    "continuation_id already names a different immutable receipt")
            return json.loads(row["payload_json"])
        db.execute(
            "INSERT INTO time_machine_continuations "
            "(continuation_id, requested_run_id, source_turn, status, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                continuation_id,
                receipt["requested_run_id"],
                receipt["source_turn"],
                status,
                payload,
            ),
        )
    return json.loads(payload)


def get(continuation_id: str) -> dict | None:
    if not isinstance(continuation_id, str) or not continuation_id or not os.path.isfile(_path()):
        return None
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT payload_json FROM time_machine_continuations WHERE continuation_id=?",
            (continuation_id,),
        ).fetchone()
    return json.loads(row["payload_json"]) if row is not None else None


def latest_for_run(run_id: str, turn: int | None = None) -> dict | None:
    if not isinstance(run_id, str) or not run_id or not os.path.isfile(_path()):
        return None
    query = (
        "SELECT payload_json FROM time_machine_continuations WHERE requested_run_id=?"
        + (" AND source_turn=?" if turn is not None else "")
        + " ORDER BY rowid DESC LIMIT 1"
    )
    params = (run_id, int(turn)) if turn is not None else (run_id,)
    try:
        with closing(_connect()) as db:
            row = db.execute(query, params).fetchone()
    except (OSError, sqlite3.Error):
        return None
    return json.loads(row["payload_json"]) if row is not None else None
