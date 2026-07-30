"""F2: the session/agent trace API -- cumulative, cross-turn evidence over one session's persisted runs.

`clozn.runs.sessions` (F1) is the anchor entity; this module is a purely DERIVED view over it and over
`clozn.runs.store`'s `runs` table. Nothing here writes a run, writes a session, or calls a live engine --
`build_trace()` is engine-free, model-free, and reads only already-persisted evidence (proven in
tests/test_session_trace.py by a substrate double whose `__getattr__` raises on any access, mirroring
tests/test_token_workbench.py's own "GET must never touch engine.X" pattern).

COMPOSITION, NEVER RE-DERIVATION
-----------------------------------
  * Ordering and pagination: `clozn.runs.sessions.list_session_runs` (F1's own cursor contract --
    `store.encode_cursor`/`decode_cursor`, oldest-first). This module never writes its own run-ordering
    SQL; it fetches a page of run ids from that function and re-reads each one directly (see "RAW RUN
    DICTS" below).
  * Turn-to-turn drift (identity/generation-setting changes, and what entered/left the delivered
    context): `clozn.analysis.run_diff.compare_runs(previous_turn, this_turn)`, called exactly ONCE per
    turn pair -- the SAME function D1's rule R12 (`clozn.runs.diagnosis_rules._rule_run_to_run_drift`)
    and D2's narratives already compose. This module never re-implements a diff.
  * Per-turn diagnostics: `clozn.runs.diagnosis_rules.evaluate(run, comparison_run=previous_turn)` --
    the exact function `clozn/server/routes/diagnosis_findings.py` already calls at GET-time (established
    precedent that this composition is GET-safe: it is pure Python/regex over already-persisted fields,
    no engine, no measurement). R08/R09 (the two rules that need a persisted influence map) naturally
    report `status="pending"` here -- see "RAW RUN DICTS" below for exactly why, structurally, not just
    by convention.

RAW RUN DICTS, NOT `store.get_run()`
----------------------------------------
`store.get_run()` unpacks BOTH content-addressed blobs a run may carry: the per-token generation trace
(`trace_ref` -> `trace`) and, when present, the influence map (`influence_map_ref` -> `influence_map`).
Neither is inline in `payload_json`; both cost a blob file read (`store._load_blob`). This module never
calls `store.get_run()` -- every run it reads comes back through `_raw_run()`/`_raw_runs_by_id()`, a thin
`SELECT payload_json ... ; json.loads()` with NO blob resolution. Two deliberate consequences, not
accidents:

  1. Zero blob I/O anywhere in this module, on any code path, including the session-wide
     "first went wrong" scan (`_first_went_wrong_candidates`) that can touch every linear turn in the
     session. `context_receipt`, `messages`, `response`, `identity`, `meta`, `finish_reason`, `error` are
     ALL inline in `payload_json` already (see `clozn.runs.store._pack`) -- only `trace`/`influence_map`
     are ever blobbed, and neither is read by anything this module composes (`diagnosis_rules.evaluate`
     never reads `run["trace"]` at all; `run_diff.compare_runs`'s embedded `model_diff.diff_runs` degrades
     gracefully to a text-only diff when `trace` is absent, and this module never inspects that dimension's
     enrichment detail regardless).
  2. `diagnosis_rules.evaluate()`'s influence-dependent rules (R08/R09) check `"influence_map" not in run`
     to decide `pending` vs a real evaluation -- a raw run dict NEVER carries that key (only
     `influence_map_ref`), so R08/R09 ALWAYS report `pending` here. This is the literal, structural
     implementation of the F2 spec's rule ("do not trigger influence measurement; rules that would need it
     surface as their honest pending status") -- not a flag this module checks and branches on, a
     property of the data it hands the rule engine.

DETERMINISM
-------------
`build_trace(..., generated_at=...)` is the ONE wall-clock read this module performs on its own behalf; it
is threaded into every `diagnosis_rules.evaluate(..., generated_at=generated_at)` call so those embedded
documents never introduce their own timestamp. `run_diff.compare_runs()` DOES stamp its own live
`generated_at` on its return value -- this module never embeds that field anywhere; only the timestamp-free
parts (`differences`, `findings`) are ever copied into this document (see `_turn_comparison`). Calling
`build_trace()` twice with the same session state and the same `generated_at` override therefore produces
byte-identical output (tests/test_session_trace.py proves this directly).

NO CAUSAL VOCABULARY
-----------------------
`first_went_wrong_candidates` names DETERMINISTIC heuristic candidates (first turn with a diagnostic
finding, first identity/generation-setting drift against the previous turn, first failed/cancelled run) --
never "the cause". Every string this module emits (and the module's own source, outside this one quoted
citation) stays free of causal vocabulary ("because", "caused", "causes", "causing", "due to",
"responsible for", "leads to", "results in", "the reason") -- verified by
tests/test_session_trace.py against both generated documents and this file's own source, the same
guard pattern `clozn/runs/diagnosis_narratives.py` and `clozn/analysis/mechanistic_diff.py` already use.

BRANCHES ARE NEVER FLATTENED
--------------------------------
`turns` is built exclusively from `sessions.list_session_runs`'s default (`source NOT IN ('replay',
'branch', 'fork')`) -- a fork/retry never appears there. `branches` is a SEPARATE list, grouped by
`parent_run_id`, scoped to parents present on the current page (see `_branches_for`); no run ever appears
in both lists.

FAILED AND CANCELLED RUNS STAY IN
-------------------------------------
`sessions.list_session_runs`'s SQL has no filter on `error` or `finish_reason` -- a failed or cancelled
run is a completely ordinary row there, and this module never adds one. "Failed or cancelled" (used only
for the `first_failed_run` candidate) means `run.get("error")` is a non-empty string, OR
`context_receipt.termination.reason` is `"client_cancelled"` or `"worker_error"` -- the exact vocabulary
`clozn.runs.context_receipt.normalize_termination` already uses; this module invents no new status enum.

REDACTED RUNS
---------------
Two different redaction modes exist (`clozn.runs.mutations.redact_run`), and they behave differently here
-- both honestly, neither a bug:

  * LITERAL redaction (`redact_run(rid, literals=[...])`) scrubs only matched substrings from text fields
    and leaves `session_key`/`client_key`/structure/numbers untouched -- the turn stays fully part of its
    session. `_is_redacted()` mirrors the same public-shape check `clozn.runs.diagnosis_rules.
    _run_is_redacted` uses (duplicated rather than reached into, the same choice that module documents
    making relative to `clozn.runs.text_span_addresses`'s own private copy) and marks the turn
    `redacted: true`; `context_usage`/`turn_comparison` still derive whatever `run_diff`/`diagnosis_rules`
    can honestly still read (numeric/structural fields survive; scrubbed text degrades to
    `unavailable`/`not_observed` from those modules' own redaction awareness, never a fabricated
    "unchanged").
  * FULL tombstone redaction (`redact_run(rid)`, no `literals`) is mutations.py's OWN documented contract
    to clear `session_key` along with every other identifying field (`_REMOVED_FIELDS` includes
    `session_key`) -- this module has no override for that, and none should be added: a fully-redacted
    run genuinely stops being part of any session's evidence trail from that point forward, exactly as it
    stops matching every OTHER `session_key`-scoped query in this codebase (`sessions.list_session_runs`,
    `store.find_runs`, ...). The turn simply no longer appears in `turns` (nor is it double-counted in
    `totals_through_this_page` or considered by `first_went_wrong_candidates`) -- this is the session's
    evidence trail correctly reflecting what full redaction already did everywhere else -- this module
    never silently drops a turn for merely having failed; see "FAILED AND CANCELLED RUNS STAY IN" above.

RUNNING TOTALS ARE SESSION-WIDE, NOT PAGE-LOCAL
----------------------------------------------------
Each turn's `cumulative` block is the running total from the SESSION's first linear turn through THIS
turn, regardless of which page it appears on -- computed by summing every prior linear turn's
`duration_ms` (a real SQLite column) and `context_receipt.limits.{prompt,generated}_tokens` (parsed from
`payload_json`, no blob I/O) up to the page's cursor boundary (`_prior_totals`), then accumulating forward
across the page itself. This costs one aggregate query plus one payload scan over every run PRECEDING the
requested page -- bounded by `store.KEEP` (1000 runs store-wide), and pure arithmetic, not model work.
"""
from __future__ import annotations

import json
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Mapping

from clozn import schemas
from clozn.analysis import run_diff

from . import association, sessions, store
from .diagnosis_rules import RULE_REGISTRY, evaluate as _evaluate_diagnosis_rules
from .summaries import _summary

SCHEMA_VERSION = "clozn.session-trace.v1"

DEFAULT_LIMIT = 50
_CANDIDATE_KINDS = ("first_finding", "first_settings_drift", "first_failed_run")
_FAILURE_TERMINATION_REASONS = ("client_cancelled", "worker_error")
_CONTEXT_USAGE_KEYS = ("prompt_tokens", "context_window_tokens", "requested_max_tokens", "generated_tokens")
_TOKEN_TOTAL_KEYS = ("prompt_tokens", "generated_tokens")
_CANDIDATE_SCAN_PAGE = 200   # internal walk chunk size for _first_went_wrong_candidates -- see its docstring


# ------------------------------------------------------------------------------------------------ helpers

def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _str(value: Any) -> "str | None":
    return value if isinstance(value, str) and value else None


def _int_or_none(value: Any) -> "int | None":
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_redacted(run: Mapping[str, Any]) -> bool:
    """Mirrors clozn.runs.diagnosis_rules._run_is_redacted's own public-shape check (private to that
    module; duplicated here rather than reached into -- see that module's own docstring for the identical
    choice relative to text_span_addresses)."""
    redaction = run.get("redaction")
    return (
        isinstance(redaction, Mapping) and redaction.get("status") in {"redacted", "literal_redacted"}
    ) or "redacted" in (run.get("flags") or [])


def _is_failed_or_cancelled(run: Mapping[str, Any]) -> bool:
    if _str(run.get("error")):
        return True
    termination = _object(_object(run.get("context_receipt")).get("termination"))
    return termination.get("reason") in _FAILURE_TERMINATION_REASONS


# ------------------------------------------------------------------------------------------- raw run reads
# See the module docstring's "RAW RUN DICTS" section: every run this module reads comes through here, NOT
# store.get_run(), so nothing in this module ever resolves a trace/influence-map blob.

def _raw_run(rid: "str | None") -> "dict | None":
    if not rid or not store._valid_rid(rid):
        return None
    store._ensure()
    with closing(store._connect()) as db:
        row = db.execute("SELECT payload_json FROM runs WHERE id = ?", (rid,)).fetchone()
    if row is None:
        return None
    try:
        doc = json.loads(row["payload_json"])
    except Exception:
        return None
    return doc if isinstance(doc, dict) else None


# ------------------------------------------------------------------------------------- per-turn projections

def _context_usage(run: Mapping[str, Any]) -> dict:
    limits = _object(_object(run.get("context_receipt")).get("limits"))
    out = {}
    for key in _CONTEXT_USAGE_KEYS:
        value = _int_or_none(limits.get(key))
        if value is not None:
            out[key] = value
    return out


def _timing(run: Mapping[str, Any]) -> dict:
    duration_ms = _int_or_none(_object(run.get("timing")).get("duration_ms"))
    return {"duration_ms": duration_ms} if duration_ms is not None else {}


def _highlights(findings_document: dict) -> dict:
    findings = [f for f in findings_document.get("findings") or []
               if isinstance(f, dict) and f.get("status") == "finding"]
    status_counts = _object(findings_document.get("summary")).get("status_counts")
    return {
        "findings": findings,
        "status_counts": dict(status_counts) if isinstance(status_counts, dict) else {},
    }


def _context_changes(differences: list) -> dict:
    segment_diffs = [
        d for d in differences
        if isinstance(d, dict) and isinstance(d.get("dimension"), str)
        and d["dimension"].startswith("context.delivered.segments.")
    ]
    new_segments = [d for d in segment_diffs if d.get("kind") == "added"]
    dropped_segments = [d for d in segment_diffs if d.get("kind") == "removed"]
    other = [
        d for d in differences
        if isinstance(d, dict) and isinstance(d.get("dimension"), str)
        and d["dimension"].startswith("context.")
        and not d["dimension"].startswith("context.delivered.segments.")
    ]
    return {"new_segments": new_segments, "dropped_segments": dropped_segments, "other_context_differences": other}


def _delivered_segment_count(run: Mapping[str, Any]) -> int:
    delivered = _object(run.get("context_receipt")).get("delivered")
    return len(delivered) if isinstance(delivered, list) else 0


def _turn_comparison(previous_run: Mapping[str, Any], run: Mapping[str, Any]) -> dict:
    """Compose (never re-derive) `run_diff.compare_runs()` for one adjacent turn pair. Only the
    timestamp-free parts of its result (`differences`, `findings`) are ever copied out -- see the module
    docstring's "DETERMINISM" section for why `compare_runs()`'s own `generated_at` is never embedded."""
    compared_to = _str(previous_run.get("id"))
    result = run_diff.compare_runs(dict(previous_run), dict(run))
    if not result.get("ok"):
        out: dict = {"available": False, "reason": _str(result.get("error")) or "comparison failed"}
        if compared_to:
            out["compared_to_run_id"] = compared_to
        return out
    differences = [d for d in result.get("differences") or [] if isinstance(d, dict)]
    settings_changes = [
        d for d in differences
        if isinstance(d.get("dimension"), str)
        and (d["dimension"].startswith("identity.") or d["dimension"].startswith("generation."))
    ]
    changes = _context_changes(differences)
    changes["carried_forward_segment_count"] = max(
        0, _delivered_segment_count(run) - len(changes["new_segments"]))
    out = {
        "available": True,
        "settings_changes": settings_changes,
        "context_changes": changes,
        "classifications": [c for c in result.get("findings") or [] if isinstance(c, dict)],
    }
    if compared_to:
        out["compared_to_run_id"] = compared_to
    return out


def _build_turn(run: dict, previous_run: "dict | None", cumulative: dict, generated_at: str) -> dict:
    entry: dict = {
        "run_id": run.get("id"),
        "recorded_ts": run.get("recorded_ts"),
        "created_at": run.get("created_at"),
        "source": run.get("source"),
        "client": run.get("client"),
        "model": run.get("model"),
        "prompt_summary": _summary(run).get("prompt_summary") or "",
        "response_summary": _summary(run).get("response_summary") or "",
        "redacted": _is_redacted(run),
    }
    if _str(run.get("finish_reason")):
        entry["finish_reason"] = run["finish_reason"]
    if _str(run.get("error")):
        entry["error"] = run["error"]
    usage = _context_usage(run)
    if usage:
        entry["context_usage"] = usage
    timing = _timing(run)
    if timing:
        entry["timing"] = timing

    cumulative["turn_count"] += 1
    cumulative["duration_ms_total"] += timing.get("duration_ms", 0)
    for key, total_key in zip(_TOKEN_TOTAL_KEYS, ("prompt_tokens_total", "generated_tokens_total")):
        cumulative[total_key] += usage.get(key, 0)
    entry["cumulative"] = dict(cumulative)

    findings_document = _evaluate_diagnosis_rules(
        run, comparison_run=previous_run, generated_at=generated_at)
    entry["diagnostic_highlights"] = _highlights(findings_document)

    if previous_run is not None:
        entry["turn_comparison"] = _turn_comparison(previous_run, run)
    return entry


# --------------------------------------------------------------------------------------- running totals seed

def _prior_totals(session_key_value: str, cursor: "str | None") -> dict:
    """The running-totals baseline for everything BEFORE the requested page -- see the module docstring's
    "RUNNING TOTALS ARE SESSION-WIDE" section. Zero for the first page (cursor=None)."""
    baseline = {"turn_count": 0, "duration_ms_total": 0, "prompt_tokens_total": 0, "generated_tokens_total": 0}
    if not cursor:
        return baseline
    after_ts, after_id = store.decode_cursor(cursor)
    store._ensure()
    with closing(store._connect()) as db:
        row = db.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(duration_ms), 0) AS duration_total FROM runs "
            "WHERE session_key = ? AND source NOT IN ('replay', 'branch', 'fork') "
            "AND (recorded_ts < ? OR (recorded_ts = ? AND id <= ?))",
            (session_key_value, after_ts, after_ts, after_id),
        ).fetchone()
        token_rows = db.execute(
            "SELECT payload_json FROM runs "
            "WHERE session_key = ? AND source NOT IN ('replay', 'branch', 'fork') "
            "AND (recorded_ts < ? OR (recorded_ts = ? AND id <= ?))",
            (session_key_value, after_ts, after_ts, after_id),
        ).fetchall()
    baseline["turn_count"] = int(row["n"] or 0)
    baseline["duration_ms_total"] = int(row["duration_total"] or 0)
    for token_row in token_rows:
        try:
            run = json.loads(token_row["payload_json"])
        except Exception:
            continue
        usage = _context_usage(run) if isinstance(run, dict) else {}
        baseline["prompt_tokens_total"] += usage.get("prompt_tokens", 0)
        baseline["generated_tokens_total"] += usage.get("generated_tokens", 0)
    return baseline


# ------------------------------------------------------------------------------------------------- branches

def _branches_for(session_key_value: str, turn_ids: list) -> list:
    if not turn_ids:
        return []
    store._ensure()
    placeholders = ",".join("?" for _ in turn_ids)
    with closing(store._connect()) as db:
        rows = db.execute(
            f"SELECT payload_json FROM runs WHERE session_key = ? AND parent_run_id IN ({placeholders}) "
            "ORDER BY recorded_ts ASC, id ASC",
            (session_key_value, *turn_ids),
        ).fetchall()
    by_parent: dict = {}
    for row in rows:
        try:
            run = json.loads(row["payload_json"])
        except Exception:
            continue
        if not isinstance(run, dict):
            continue
        parent = run.get("parent_run_id")
        if isinstance(parent, str) and parent:
            by_parent.setdefault(parent, []).append(_summary(run))
    return [{"parent_run_id": pid, "children": by_parent[pid]} for pid in turn_ids if pid in by_parent]


# ------------------------------------------------------------------------------- first-went-wrong candidates

def _candidate(kind: str, run: Mapping[str, Any], summary: str, **extra) -> dict:
    entry = {"kind": kind, "run_id": run.get("id"), "recorded_ts": run.get("recorded_ts"), "summary": summary}
    entry.update(extra)
    return entry


def _first_went_wrong_candidates(session_key_value: str, generated_at: str) -> list:
    """A session-wide, oldest-first scan for the three DETERMINISTIC heuristic candidates -- see the
    module docstring's "NO CAUSAL VOCABULARY" section. Walks linear turns via `sessions.list_session_runs`
    (internally paginated at `_CANDIDATE_SCAN_PAGE`) until all three kinds are found or the session is
    exhausted; typically short-circuits within the first few turns of a session that has any problem at
    all. Bounded worst case: `store.KEEP` (1000) runs store-wide, all raw-dict reads (see "RAW RUN DICTS"),
    zero blob I/O -- see the module docstring's "RUNNING TOTALS"/"RAW RUN DICTS" sections for the same
    cost argument applied here."""
    found: dict = {}
    previous_run: "dict | None" = None
    cursor: "str | None" = None
    while len(found) < len(_CANDIDATE_KINDS):
        page = sessions.list_session_runs(session_key_value, cursor=cursor, limit=_CANDIDATE_SCAN_PAGE)
        if not page["runs"]:
            break
        for summary in page["runs"]:
            run = _raw_run(summary.get("id"))
            if run is None:
                continue
            if "first_failed_run" not in found and _is_failed_or_cancelled(run):
                found["first_failed_run"] = _candidate(
                    "first_failed_run", run,
                    "this turn's run recorded an error or a cancelled/failed termination reason")
            if previous_run is not None and "first_settings_drift" not in found:
                result = run_diff.compare_runs(dict(previous_run), dict(run))
                if result.get("ok"):
                    drift_dims = sorted(
                        d["dimension"] for d in result.get("differences") or []
                        if isinstance(d, dict) and isinstance(d.get("dimension"), str)
                        and (d["dimension"].startswith("identity.") or d["dimension"].startswith("generation."))
                    )
                    if drift_dims:
                        found["first_settings_drift"] = _candidate(
                            "first_settings_drift", run,
                            "identity/generation setting(s) differ from the previous turn: "
                            + ", ".join(drift_dims),
                            compared_to_run_id=previous_run.get("id"))
            if "first_finding" not in found:
                findings_document = _evaluate_diagnosis_rules(
                    run, comparison_run=previous_run, generated_at=generated_at)
                fired = sorted(f["rule_id"] for f in findings_document.get("findings") or []
                               if isinstance(f, dict) and f.get("status") == "finding")
                if fired:
                    found["first_finding"] = _candidate(
                        "first_finding", run,
                        "diagnostic rule(s) reported a finding on this turn: " + ", ".join(fired),
                        rule_ids=fired)
            previous_run = run
            if len(found) == len(_CANDIDATE_KINDS):
                break
        cursor = page["next_cursor"]
        if cursor is None:
            break
    return [found[kind] for kind in _CANDIDATE_KINDS if kind in found]


# ------------------------------------------------------------------------------------------------- session

def _session_projection(session_document: dict) -> dict:
    out = {"id": session_document["id"], "privacy": session_document.get("privacy") or {}}
    if _str(session_document.get("title")):
        out["title"] = session_document["title"]
    if _str(session_document.get("client_key")):
        out["client_key"] = session_document["client_key"]
    return out


# ---------------------------------------------------------------------------------------------- public API

def build_trace(session_id, *, cursor: "str | None" = None, limit: int = DEFAULT_LIMIT,
                generated_at: "str | None" = None, materialize: bool = True) -> "dict | None":
    """The full `clozn.session-trace.v1` document for one page of one session's turns, or None when the
    session honestly does not exist (no explicit `sessions` row AND no member run -- see
    `clozn.runs.sessions.get_session`). `materialize=True` (the default, matching F1's own GET
    /sessions/<id> route) persists a legacy session's identity row on first view; it is a single cheap
    SQLite write, never measurement or model work.

    Deterministic given a fixed `generated_at` and unchanged underlying data -- see the module docstring's
    "DETERMINISM" section. Never touches a live substrate/engine -- see "COMPOSITION, NEVER RE-DERIVATION".
    """
    session_document = sessions.get_session(session_id, materialize=materialize)
    if session_document is None:
        return None
    key = session_document["id"]
    generated_at = generated_at if generated_at is not None else _now_iso()
    wanted_limit = max(1, min(1000, int(limit)))

    page = sessions.list_session_runs(key, cursor=cursor, limit=wanted_limit)

    previous_run: "dict | None" = None
    if cursor:
        _after_ts, after_id = store.decode_cursor(cursor)
        previous_run = _raw_run(after_id)

    cumulative = _prior_totals(key, cursor)
    turns = []
    turn_ids = []
    for summary in page["runs"]:
        run = _raw_run(summary.get("id"))
        if run is None:
            continue
        turn_ids.append(run["id"])
        turns.append(_build_turn(run, previous_run, cumulative, generated_at))
        previous_run = run

    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "session_id": key,
        "session": _session_projection(session_document),
        "page": {
            "cursor": cursor,
            "next_cursor": page["next_cursor"],
            "limit": wanted_limit,
            "count": len(turns),
        },
        "turns": turns,
        "branches": _branches_for(key, turn_ids),
        "totals_through_this_page": dict(cumulative),
        "diagnostic_rule_registry": [
            {"rule_id": rule_id, "rule_name": rule_name} for rule_id, rule_name, _fn in RULE_REGISTRY
        ],
        "first_went_wrong_candidates": _first_went_wrong_candidates(key, generated_at),
    }
    schemas.validate(document, SCHEMA_VERSION)
    return document


__all__ = ["SCHEMA_VERSION", "build_trace"]
