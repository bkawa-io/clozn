"""F6: the verify-before-save teaching loop -- clozn.correction-verification.v1, on top of F5
(`clozn/runs/corrections.py`) and D4 (`clozn/replay/controlled.py`'s exact, model-free comparison
primitive, the same one `compare-runs --test` executes).

THE FLOW THIS MODULE IS THE MIDDLE OF
--------------------------------------
    propose -> child retry -> compare vs target failure -> approve -> store at scope -> later runs show firing
       (F5)      (out of scope,       (THIS module)         (THIS      (F5's confirm       (F5's receipt
    draft_       already a run                              module)     mutation,          integration,
    correction    id by the time                                        composed not       apply_and_record/
    ())           this is called)                                       re-implemented)     resolve_corrections)

Only the middle two steps are this module's job, and it is deliberately narrow about that:

  * "Propose" is F5's `draft_correction()`, called by whatever later slice decides WHAT correction text
    to suggest (an LLM, a rule, a human typing in Studio) -- none of that lives here. This module accepts
    a `correction_id` that is already a drafted, unconfirmed row.
  * "Child retry" is likewise not generated here. Producing a retry run that has the drafted correction's
    content spliced into context is expensive (it needs a live model) and belongs to whichever route
    drives that generation -- this module accepts `child_run_id` as an already-persisted run id, exactly
    the same "accept it as input" boundary the owner's brief draws for the proposal text itself. F6 does
    not know or care HOW the child run was produced; it only compares what the child run's outcome
    actually was against what the target failure's outcome actually was, and records both ids so anyone
    can go re-read either run and check the claim independently -- receipts, not self-report.
  * "Compare vs target failure" reuses `clozn.replay.controlled.match_run()` verbatim -- the exact
    primitive `compare-runs --test`'s two-arm executor already calls to decide whether a controlled swap
    "recovered" a prior outcome. This module does not write a second comparison implementation: it calls
    the same pure, model-free function, feeding it (child_run, target_run, match_criterion) and reading
    its `available`/`matched` verdict. "Fixed" is operationalized exactly the way this codebase already
    operationalizes every outcome claim: an exact, disclosed criterion (`clozn.replay.controlled.
    SUPPORTED_MATCHERS`), never a semantic/quality judgment -- see that module's own "exact recorded
    evidence only; semantic similarity is never used" note.
  * "Approve" and "store at scope" collapse into one atomic transaction below (`verify_and_promote`):
    scope was already fixed at draft time (F5's `validate_scope`), so there is nothing left to "store" at
    promotion beyond the confirm mutation itself, which this module performs by calling
    `clozn.runs.corrections._confirm_row()` -- the SAME private helper `confirm_correction()` itself
    calls for a hand-confirm, composed here rather than re-derived, inside THIS module's own transaction
    so the verification event and the confirm event land in the identical COMMIT or neither does.
  * "Later runs show firing" needs nothing new: `resolve_corrections()` already treats a promoted
    correction exactly like a hand-confirmed one (both are just `confirmed_ts IS NOT NULL AND enabled = 1`
    rows), and `apply_and_record()` already folds it into a future run's `context_receipt`. This module
    adds no second path for that -- see F5's own module docstring for the structural argument.

THE GUARANTEE THIS MODULE EXISTS TO MAKE STRUCTURAL: "NO PROMOTION WITHOUT A COMPARISON"
------------------------------------------------------------------------------------------
Read this precisely, because `clozn.runs.corrections.confirm_correction()` already exists, does NOT
require a comparison, and must keep working exactly as it does today -- a user hand-writing their own
correction and confirming it is F5's legitimate, unrelated flow. F6's guarantee is narrower: a correction
promoted THROUGH THIS MODULE must carry the verification pair that justified it. Concretely, mechanism,
not convention:

  1. `verify_and_promote()` has NO default for `target_run_id`/`child_run_id` -- they are required
     keyword-only arguments. There is no call shape that reaches the confirm mutation without supplying
     both.
  2. Both are resolved through `clozn.runs.store.get_run()` BEFORE anything is written. A caller cannot
     promote against a fabricated or missing run id: get_run() returns an already-persisted, IMMUTABLE
     run or this function raises `TeachingLoopRunNotFoundError` and writes nothing.
  3. The confirm mutation happens ONLY inside the branch that already wrote the `verification_passed`
     event, in the SAME open `db` transaction (`with closing(store._connect()) as db, db:` -- the same
     commit-or-rollback contract every other corrections.py mutator uses). There is no code path in this
     module that calls the confirm mutation without having appended that event first, in that transaction.
     A crash between the two is impossible to observe as "confirmed but no verification event": SQLite
     either commits both rows or neither.
  4. The `confirmed` event this module writes ALSO carries `detail.target_run_id`/`child_run_id`/
     `match_criterion` (via `_confirm_row(..., detail=...)`) -- so the pair is discoverable two ways: from
     the adjacent `verification_passed` event, and from the `confirmed` event itself. Either is sufficient
     alone; a hand-confirm's `confirmed` event (via `confirm_correction()`) never carries this detail, so
     "promoted by the teaching loop" is itself a directly observable fact, not something the caller has to
     assert out-of-band.

What this mechanism does NOT prove, stated as plainly as F5's own module docstring states its analogous
limit: nothing here verifies that `child_run_id` was ACTUALLY produced by applying THIS correction's
`content` to the same context as `target_run_id` -- that causal link lives in whatever produced the child
run (a live replay with the draft's content spliced in), not in this module, which only ever sees two
already-persisted run ids and compares their recorded outcomes. A caller could, in principle, pass two
unrelated run ids that happen to differ under the chosen criterion and get a "promoted" result. Two things
narrow that gap without closing it: (a) the recorded pair is real and checkable -- unlike a bare
`verified: true`, anyone can re-read both runs and see whether the "child retry" story actually holds up,
and (b) `target_run_id`/`child_run_id` are exact, immutable run ids, never free text, so the claim cannot
be edited after the fact to match a different pair. This is the same shape of honest limit F5's
`apply_and_record()` docstring draws for its own single-writer guarantee -- convention closes the
remaining gap, not a mechanical impossibility.

"FAILED CORRECTIONS STAY DRAFTS" -- AND ARE NEVER SILENTLY DISCARDED EITHER
------------------------------------------------------------------------------
When the child run reproduces the target failure (or the match criterion's evidence is unavailable on
either run), `verify_and_promote()` writes exactly one `verification_failed` event -- carrying the same
target/child/criterion/comparison detail a `verification_passed` event would have carried -- and returns
without ever touching `confirmed_ts`/`enabled`. The correction remains precisely what it was: an inert,
unconfirmed draft, structurally unselectable by `resolve_corrections()` (F5's own SQL predicate, unchanged
by anything in this module). The attempt itself is not lost: `verification_failed` lands in the SAME
append-only `correction_events` ledger every other lifecycle event does, so `corrections.export_correction
()` shows it forever -- "clozn tried this and it did not work" is a first-class, permanent fact next to
"clozn tried this and it worked," never a discarded log line.

UNDO -- COMPOSED, NOT REIMPLEMENTED
--------------------------------------
A promotion this module makes is, by construction, indistinguishable in STATE from a hand confirm (same
`confirmed_ts`/`enabled` columns, same `confirmed` event type). `clozn.runs.corrections.undo_last_change()`
already reverts the most recent confirm/disable/enable transition inside its own single transaction; since
this module's promotion writes exactly one `confirmed` event as its last state-changing row, calling
`corrections.undo_last_change(correction_id)` after a teaching-loop promotion reverts it exactly the same
way it would a hand confirm -- back to drafted, `confirmed_ts` absent, atomically. This module adds no
second undo function; see `test_teaching_loop.py::test_undo_restores_prior_config_transactionally` for the
round trip. The `verification_passed` event that justified the promotion is untouched by the undo (this
ledger never rewrites history), so the record of what was verified and when survives a later undo exactly
as every other past event does.
"""
from __future__ import annotations

import time
from contextlib import closing

from clozn import schemas
from clozn.replay import controlled

from . import corrections, store

SCHEMA_NAME = "clozn.correction-verification.v1"


class TeachingLoopError(ValueError):
    """Base class for every typed, catchable F6 failure. Correction-identity/state problems (unknown,
    already confirmed, deleted) propagate as `clozn.runs.corrections.CorrectionNotFoundError`/
    `CorrectionStateError` directly rather than being re-wrapped here -- this module reuses F5's own
    internal row lookup (`corrections._require_row`), so those are the exact same exceptions a caller of
    F5 already knows to catch. `TeachingLoopError` and its subclasses cover failure modes F5 has no
    equivalent for: a malformed/unknown run id, an unknown match criterion, or a degenerate (target ==
    child) pair."""


class TeachingLoopValueError(TeachingLoopError):
    """A caller-supplied argument (run id shape, match_criterion, target == child) is malformed."""


class TeachingLoopRunNotFoundError(TeachingLoopError):
    """`target_run_id` or `child_run_id` does not resolve to a persisted run. Raised before anything is
    written -- a promotion can never be attempted against a fabricated or missing run id."""


def _require_run(run_id, *, label: str) -> dict:
    if not isinstance(run_id, str) or not store._valid_rid(run_id):
        raise TeachingLoopValueError(f"{label} must be an exact valid run id, got {run_id!r}")
    run = store.get_run(run_id)
    if run is None:
        raise TeachingLoopRunNotFoundError(f"{label} {run_id!r} was not found in the local journal")
    return run


def verify_and_promote(correction_id: str, *, target_run_id: str, child_run_id: str,
                       match_criterion: str = "exact_output", now: "float | None" = None) -> dict:
    """Verify a drafted correction against a target-failure/child-retry run pair, and promote it (confirm
    + record the pair) ONLY if the comparison shows the child run did not reproduce the target failure.
    Returns a schema-valid `clozn.correction-verification.v1` document either way -- a failed verification
    is a normal, successful RETURN of this function (it raises only for a malformed call), not an
    exception, because "clozn tried this and it did not work" is evidence to record, not an error.

    Raises `TeachingLoopValueError`/`TeachingLoopRunNotFoundError` for a malformed call (see above),
    `corrections.CorrectionNotFoundError` for an unknown correction_id, and
    `corrections.CorrectionStateError` if the correction is deleted or was already confirmed (a teaching-
    loop promotion targets a fresh draft; re-verifying an already-confirmed correction is not this
    function's job -- disable it and draft a new one, or undo first, per the module docstring).
    """
    if match_criterion not in controlled.SUPPORTED_MATCHERS:
        raise TeachingLoopValueError(
            f"match_criterion must be one of {sorted(controlled.SUPPORTED_MATCHERS)}, got "
            f"{match_criterion!r}"
        )
    target_run = _require_run(target_run_id, label="target_run_id")
    child_run = _require_run(child_run_id, label="child_run_id")
    if target_run["id"] == child_run["id"]:
        raise TeachingLoopValueError(
            "target_run_id and child_run_id must name different runs -- a retry cannot verify itself"
        )

    # The comparison itself: pure, model-free, computed over two already-persisted run dicts. This is
    # clozn.replay.controlled's own primitive, not a second implementation of it -- see the module
    # docstring's "compare vs target failure" section.
    comparison = controlled.match_run(child_run, target_run, match_criterion)
    now = time.time() if now is None else float(now)

    detail = {
        "target_run_id": target_run_id,
        "child_run_id": child_run_id,
        "match_criterion": match_criterion,
        "comparison": comparison,
    }

    store._ensure()
    with closing(store._connect()) as db, db:
        row = corrections._require_row(db, correction_id)
        if row["deleted_ts"] is not None:
            raise corrections.CorrectionStateError(
                f"correction {correction_id!r} is deleted and cannot be verified"
            )
        if row["confirmed_ts"] is not None:
            raise corrections.CorrectionStateError(
                f"correction {correction_id!r} is already confirmed; verify_and_promote() targets a "
                f"fresh, unconfirmed draft -- undo or draft a new correction to re-run the teaching loop"
            )

        fixed = bool(comparison.get("available")) and not comparison.get("matched")
        if fixed:
            corrections._append_event(
                db, correction_id, "verification_passed", run_id=child_run_id, detail=detail, ts=now
            )
            corrections._confirm_row(db, correction_id, now=now, detail={
                "promoted_by": "teaching_loop",
                "target_run_id": target_run_id,
                "child_run_id": child_run_id,
                "match_criterion": match_criterion,
            })
            verification, promoted = "passed", True
            reason = (
                f"child run {child_run_id!r} did not reproduce target failure {target_run_id!r} under "
                f"{match_criterion!r}; correction promoted"
            )
        else:
            corrections._append_event(
                db, correction_id, "verification_failed", run_id=child_run_id, detail=detail, ts=now
            )
            verification, promoted = "failed", False
            if not comparison.get("available"):
                reason = (
                    f"{match_criterion} evidence was unavailable on the target or child run "
                    f"({comparison.get('reason', 'no reason recorded')}); correction was not promoted"
                )
            else:
                reason = (
                    f"child run {child_run_id!r} reproduced the same outcome as target failure "
                    f"{target_run_id!r} under {match_criterion!r}; correction was not promoted"
                )
        row = corrections._fetch_row(db, correction_id)

    result = {
        "schema_version": SCHEMA_NAME,
        "correction_id": correction_id,
        "created_ts": now,
        "target_run_id": target_run_id,
        "child_run_id": child_run_id,
        "match_criterion": match_criterion,
        "comparison": comparison,
        "verification": verification,
        "promoted": promoted,
        "reason": reason,
        "correction": corrections._row_to_document(row),
    }
    schemas.validate(result, SCHEMA_NAME)
    return result


__all__ = [
    "SCHEMA_NAME",
    "TeachingLoopError", "TeachingLoopValueError", "TeachingLoopRunNotFoundError",
    "verify_and_promote",
]
