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


def list_for_parent(parent_run_id: str) -> list[dict]:
    """Every stored terminal execution-fork receipt for `parent_run_id`, in deterministic
    (`execution_id` ascending) order.

    A pure read: unlike `_connect()` (used by `save`/`get`), this NEVER calls `os.makedirs` or opens
    the database in a mode that would create it -- `[]` on a fresh installation (no results directory,
    no sqlite file) leaves the filesystem exactly as it found it. This is the property
    `clozn.replay.rewind_fidelity`'s read-only projection depends on: asking about rewind fidelity on
    a run that has never had an execution-fork attempt must never conjure `~/.clozn/execution-forks/`
    into existence.

    A row whose payload is not valid JSON, or decodes to something other than an object, is silently
    skipped rather than raising -- "safely reject a malformed receipt" here means "do not let one
    corrupt row take down every other read for this parent." Schema-conformance (whether a receipt
    that DOES parse is trustworthy enough to count as exactness PROOF) is a policy decision left to the
    caller, not this storage-layer helper. Never mutates a stored row.
    """
    if not isinstance(parent_run_id, str) or not parent_run_id:
        return []
    if not os.path.isfile(_path()):
        return []
    with closing(sqlite3.connect(_path(), timeout=30.0)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT payload_json FROM execution_fork_results WHERE parent_run_id=? "
            "ORDER BY execution_id ASC",
            (parent_run_id,),
        ).fetchall()
    receipts: list[dict] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            receipts.append(payload)
    return receipts

