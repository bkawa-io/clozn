"""Immutable terminal execution-fork receipts.

Failed controls are evidence, not generated child runs.  They therefore live in this tiny artifact
store rather than being fabricated as empty ``source=fork`` generations.  Successful receipts are
stored here too and are additionally embedded in their real child run.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
import sqlite3

from clozn import schemas

RESULTS_DIR = os.path.join(os.path.expanduser("~/.clozn"), "execution-forks")


class ExecutionForkResultError(RuntimeError):
    """A terminal receipt could not be stored without violating immutability."""


def _path() -> str:
    return os.path.join(RESULTS_DIR, "results.sqlite3")


def _connect() -> sqlite3.Connection:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db = sqlite3.connect(_path(), timeout=30.0)
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE IF NOT EXISTS execution_fork_results ("
        " execution_id TEXT PRIMARY KEY,"
        " parent_run_id TEXT NOT NULL,"
        " phase TEXT NOT NULL,"
        " payload_json TEXT NOT NULL"
        ")"
    )
    return db


def save(receipt: dict) -> dict:
    """Insert one terminal receipt. The same bytes are idempotent; different bytes never overwrite."""
    schemas.validate(receipt, "clozn.execution-fork.v1")
    execution_id = receipt.get("execution_id")
    phase = receipt.get("phase")
    if not isinstance(execution_id, str) or not execution_id:
        raise ExecutionForkResultError("terminal receipt needs execution_id")
    if phase not in {"completed", "failed", "cancelled"}:
        raise ExecutionForkResultError("only a terminal execution-fork receipt can be stored")
    payload = json.dumps(
        receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with closing(_connect()) as db, db:
        row = db.execute(
            "SELECT payload_json FROM execution_fork_results WHERE execution_id=?",
            (execution_id,),
        ).fetchone()
        if row is not None:
            if row["payload_json"] != payload:
                raise ExecutionForkResultError(
                    "execution_id already names a different immutable receipt")
            return json.loads(row["payload_json"])
        db.execute(
            "INSERT INTO execution_fork_results "
            "(execution_id, parent_run_id, phase, payload_json) VALUES (?, ?, ?, ?)",
            (execution_id, receipt["parent_run_id"], phase, payload),
        )
    return json.loads(payload)


def get(execution_id: str) -> dict | None:
    if not isinstance(execution_id, str) or not execution_id:
        return None
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT payload_json FROM execution_fork_results WHERE execution_id=?",
            (execution_id,),
        ).fetchone()
    return json.loads(row["payload_json"]) if row is not None else None

