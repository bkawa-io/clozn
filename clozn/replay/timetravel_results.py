"""Immutable Answer Time Machine verification receipts.

The worker checkpoint itself is ephemeral.  This small store keeps the proof receipt so inspection can
show that a prompt-boundary verification was attempted, while the eligibility endpoint continues to
require a fresh proof before claiming that the current worker can replay exactly.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
import sqlite3

from clozn import schemas

RESULTS_DIR = os.path.join(os.path.expanduser("~/.clozn"), "time-machine")


class TimeMachineResultError(RuntimeError):
    """A verification receipt could not be stored without violating immutability."""


def _path() -> str:
    return os.path.join(RESULTS_DIR, "verifications.sqlite3")


def _connect() -> sqlite3.Connection:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db = sqlite3.connect(_path(), timeout=30.0)
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE IF NOT EXISTS time_machine_verifications ("
        " verification_id TEXT PRIMARY KEY,"
        " parent_run_id TEXT NOT NULL,"
        " turn_index INTEGER NOT NULL,"
        " lookup_run_id TEXT NOT NULL,"
        " payload_json TEXT NOT NULL"
        ")"
    )
    columns = {row[1] for row in db.execute("PRAGMA table_info(time_machine_verifications)")}
    if "lookup_run_id" not in columns:
        db.execute("ALTER TABLE time_machine_verifications ADD COLUMN lookup_run_id TEXT")
        db.execute(
            "UPDATE time_machine_verifications SET lookup_run_id=parent_run_id "
            "WHERE lookup_run_id IS NULL")
    return db


def save(receipt: dict) -> dict:
    """Insert one verification receipt; retries with the same ID must be byte-identical."""
    schemas.validate(receipt, "clozn.time-machine-verification.v1")
    verification_id = receipt.get("verification_id")
    if not isinstance(verification_id, str) or not verification_id:
        raise TimeMachineResultError("verification receipt needs verification_id")
    payload = json.dumps(
        receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with closing(_connect()) as db, db:
        row = db.execute(
            "SELECT payload_json FROM time_machine_verifications WHERE verification_id=?",
            (verification_id,),
        ).fetchone()
        if row is not None:
            if row["payload_json"] != payload:
                raise TimeMachineResultError(
                    "verification_id already names a different immutable receipt")
            return json.loads(row["payload_json"])
        db.execute(
            "INSERT INTO time_machine_verifications "
            "(verification_id, parent_run_id, turn_index, lookup_run_id, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                verification_id,
                receipt["parent_run_id"],
                receipt["turn"],
                receipt.get("requested_run_id") or receipt["parent_run_id"],
                payload,
            ),
        )
    return json.loads(payload)


def get(verification_id: str) -> dict | None:
    if not isinstance(verification_id, str) or not verification_id:
        return None
    with closing(_connect()) as db:
        row = db.execute(
            "SELECT payload_json FROM time_machine_verifications WHERE verification_id=?",
            (verification_id,),
        ).fetchone()
    return json.loads(row["payload_json"]) if row is not None else None


def latest_for_run(run_id: str, turn: int | None = None) -> dict | None:
    """Return the newest receipt for a run/turn, if the local receipt store is available."""
    if not isinstance(run_id, str) or not run_id:
        return None
    # GET /runs/<id>/time-machine is explicitly read-only.  Do not create a directory/database just
    # because a run has never had a verification action; a prior POST is what initializes the store.
    if not os.path.isfile(_path()):
        return None
    query = (
        "SELECT payload_json FROM time_machine_verifications "
        "WHERE lookup_run_id=?" + (" AND turn_index=?" if turn is not None else "") +
        " ORDER BY rowid DESC LIMIT 1"
    )
    params = (run_id, int(turn)) if turn is not None else (run_id,)
    try:
        with closing(_connect()) as db:
            row = db.execute(query, params).fetchone()
    except (OSError, sqlite3.Error):
        return None
    return json.loads(row["payload_json"]) if row is not None else None
