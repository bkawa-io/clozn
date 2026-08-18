"""Time-travel debugger surface: the snapshot gate + ring config + store stats
(GET/POST /timetravel/mode, POST /timetravel/stats), rewinding & branching from a turn
(POST /runs/<id>/branch) into a child run, and explicit prompt-boundary verification
(POST /runs/<id>/time-machine/verify and POST /runs/<id>/time-machine/branch). The snapshot ring holds KV state in CPU RAM, so it is behind
ONE persisted setting (`timetravel_snapshots`, DEFAULT OFF -- the RAM rule); branch RECORDING does NOT
depend on that gate, only holding live KV for the (future) re-prefill fast path does. Mechanical
extraction of clozn.server.app's old `_timetravel` handler method + the matching do_GET/do_POST branches;
behavior unchanged. -> clozn.replay (timetravel.py).
"""
from collections.abc import Mapping

from clozn.server import app as ctx


def _exact_branch_receipt(
    requested_run_id: str,
    turn: int,
    *,
    status: str,
    exact_replay: bool,
    reasons: list[dict],
    source_run_id: str | None = None,
    child_run_id: str | None = None,
    execution_fork_execution_id: str | None = None,
    capture: Mapping | None = None,
    execution_fork: Mapping | None = None,
) -> dict:
    """Build the closed Time Machine exact-child envelope without inventing missing lineage."""
    artifact = {
        "schema_version": "clozn.time-machine-branch.v1",
        "requested_run_id": requested_run_id,
        "source_turn": int(turn),
        "status": status,
        "exact_replay": bool(exact_replay),
        "fidelity": "exact_replay_eligible" if exact_replay else "unavailable",
        "reasons": list(reasons),
        "capture": dict(capture) if isinstance(capture, Mapping) else {},
        "execution_fork": dict(execution_fork) if isinstance(execution_fork, Mapping) else {},
    }
    for name, value in (
        ("source_run_id", source_run_id),
        ("child_run_id", child_run_id),
        ("execution_fork_execution_id", execution_fork_execution_id),
    ):
        if isinstance(value, str) and value:
            artifact[name] = value
    from clozn import schemas
    schemas.validate(artifact, "clozn.time-machine-branch.v1")
    return artifact


def try_get(h, p):
    if p.startswith("/runs/") and p.endswith("/time-machine"):
        run_id = p[len("/runs/"):-len("/time-machine")]
        import clozn.runs.store as runlog
        run = runlog.get_run(run_id)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        import clozn.replay.timetravel as timetravel
        h._json(200, timetravel.replay_eligibility(run, store=ctx._snap_store()))
        return True
    if p == "/timetravel/mode":      # #6: is per-turn KV snapshotting on? + ring config + store stats
        import clozn.replay.timetravel as timetravel
        out = {"enabled": timetravel.enabled(), **timetravel.get_config()}
        store = ctx._snap_store()
        if store is not None:
            out["store"] = store.stats()
        h._json(200, out)
        return True
    return False


def try_post(h, p, body):
    if p.startswith("/timetravel/"):   # #6: the time-travel snapshot gate + ring config (default OFF)
        import clozn.replay.timetravel as timetravel
        if p == "/timetravel/mode":       # read or set the on/off gate + ring config
            changed = False
            if "enabled" in body:
                timetravel.set_enabled(bool(body.get("enabled")))
                changed = True
            if "cap" in body or "budget_mb" in body:
                timetravel.set_config(cap=body.get("cap"), budget_mb=body.get("budget_mb"))
                changed = True
                cfg = timetravel.get_config()          # apply the (clamped) new ceilings to the LIVE store
                if ctx._snap_store() is not None:
                    ctx._snap_store().reconfigure(cap=cfg["cap"], budget_mb=cfg["budget_mb"])
            out = {"enabled": timetravel.enabled(), **timetravel.get_config()}
            store = ctx._snap_store()
            if store is not None:
                out["store"] = store.stats()
            out["changed"] = changed
            h._json(200, out)
            return True
        if p == "/timetravel/stats":      # just the store's honest memory receipt
            store = ctx._snap_store()
            h._json(200, {"enabled": timetravel.enabled(),
                         **(store.stats() if store is not None else {})})
            return True
        h._json(404, {"error": f"POST {p}"})
        return True
    if p.startswith("/runs/") and p.endswith("/time-machine/branch"):
        rid = p[len("/runs/"):-len("/time-machine/branch")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not isinstance(body, dict) or set(body) - {"turn"}:
            h._json(400, {
                "error": "exact Time Machine child replay accepts only {\"turn\": integer}",
                "code": "time_machine_branch_options_unsupported",
            })
            return True
        if "turn" not in body:
            h._json(400, {"error": "need a replay turn", "code": "turn_required"})
            return True
        try:
            turn = int(body["turn"])
        except (TypeError, ValueError):
            h._json(400, {"error": "turn must be an integer", "code": "turn_invalid"})
            return True
        if turn < 0:
            h._json(400, {"error": "turn must be non-negative", "code": "turn_invalid"})
            return True

        import clozn.replay.timetravel as timetravel
        turns = timetravel.completed_message_turns(run)
        source_run = timetravel.resolve_exact_turn_source_run(run, turn)
        if (
            source_run is None
            or turn < 0
            or turn >= len(turns)
            or turns[turn].get("assistant_idx") is None
        ):
            artifact = _exact_branch_receipt(
                rid,
                turn,
                status="unavailable",
                exact_replay=False,
                reasons=[{
                    "code": "historical_source_unavailable",
                    "message": (
                        "an exact organic source run with the requested assistant turn was not "
                        "found in session history"),
                }],
            )
            h._json(422, artifact)
            return True

        from clozn.server.model_routing import select_run_model_facts
        facts = select_run_model_facts(h, source_run, route="/runs/<id>/time-machine/branch")
        if facts is None:
            return True
        runtime, worker, engine, _sub = facts
        if engine is None or runtime is None or worker is None:
            h._json(503, {
                "error": "exact Time Machine child replay requires a ready identity-qualified product worker",
                "code": "time_machine_branch_worker_unavailable",
            })
            return True

        checkpoint_envelope = None
        try:
            from clozn.replay.checkpoint_pin_store import resolve_pin
            resolved_pin = resolve_pin(source_run.get("id"))
            if isinstance(resolved_pin, Mapping) and resolved_pin.get("ok") is True:
                candidate = resolved_pin.get("envelope")
                if isinstance(candidate, Mapping):
                    checkpoint_envelope = candidate
        except Exception:
            checkpoint_envelope = None

        from clozn.replay.checkpoint_capture import (
            CheckpointCaptureError,
            capture_parent_checkpoint,
        )
        try:
            capture = capture_parent_checkpoint(
                source_run,
                engine,
                runtime_identity=runtime,
                worker_identity=worker,
                checkpoint_envelope=checkpoint_envelope,
            )
        except CheckpointCaptureError as exc:
            artifact = _exact_branch_receipt(
                rid,
                turn,
                status="unavailable",
                exact_replay=False,
                source_run_id=source_run.get("id"),
                reasons=[{"code": "checkpoint_capture_request_invalid", "message": str(exc)}],
            )
            h._json(422, artifact)
            return True
        except Exception as exc:
            artifact = _exact_branch_receipt(
                rid,
                turn,
                status="unavailable",
                exact_replay=False,
                source_run_id=source_run.get("id"),
                reasons=[{
                    "code": "checkpoint_capture_failed",
                    "message": f"exact child replay capture failed: {type(exc).__name__}: {exc}",
                }],
                capture=capture if isinstance(capture, Mapping) else {},
                execution_fork={"phase": "failed"},
            )
            h._json(422, artifact)
            return True
        if not isinstance(capture, Mapping) or capture.get("status") != "available":
            reason = (capture.get("reasons") or [{}])[0] if isinstance(capture, Mapping) else {}
            artifact = _exact_branch_receipt(
                rid,
                turn,
                status="unavailable",
                exact_replay=False,
                source_run_id=source_run.get("id"),
                reasons=[{
                    "code": str(reason.get("code") or "checkpoint_unavailable"),
                    "message": str(reason.get("message") or "the exact checkpoint was unavailable"),
                }],
                capture=capture if isinstance(capture, Mapping) else {},
            )
            h._json(422, artifact)
            return True

        from clozn.replay.execution_fork import plan_execution_fork
        try:
            plan = plan_execution_fork(
                source_run,
                {"position": 0, "change": {"type": "none"}},
                checkpoint=capture.get("checkpoint_reference"),
                runtime_identity=runtime,
                worker_identity=worker,
            )
        except Exception as exc:
            artifact = _exact_branch_receipt(
                rid,
                turn,
                status="unavailable",
                exact_replay=False,
                source_run_id=source_run.get("id"),
                reasons=[{
                    "code": "exact_plan_failed",
                    "message": f"exact child replay planning failed: {type(exc).__name__}: {exc}",
                }],
                capture=capture,
            )
            h._json(422, artifact)
            return True
        if plan.get("classification") != "exact_execution_fork":
            reason = (plan.get("reasons") or [{}])[0]
            artifact = _exact_branch_receipt(
                rid,
                turn,
                status="unavailable",
                exact_replay=False,
                source_run_id=source_run.get("id"),
                reasons=[{
                    "code": str(reason.get("code") or "exact_plan_unavailable"),
                    "message": str(reason.get("message") or "the exact plan was unavailable"),
                }],
                capture=capture,
                execution_fork=plan,
            )
            h._json(422, artifact)
            return True

        from clozn.replay.execution_fork_execute import execute_exact_fork
        try:
            result = execute_exact_fork(
                source_run,
                plan,
                engine,
                runtime_identity=runtime,
                worker_identity=worker,
                reload_parent=runlog.get_run,
            )
        except Exception as exc:
            h._json(500, {
                "error": f"exact child replay could not be persisted: {type(exc).__name__}: {exc}",
                "code": "time_machine_branch_persistence_error",
            })
            return True

        receipt = result.get("receipt") if isinstance(result, Mapping) else None
        child = result.get("child") if isinstance(result, Mapping) else None
        phase = receipt.get("phase") if isinstance(receipt, Mapping) else None
        if phase == "completed" and isinstance(child, Mapping) and isinstance(child.get("id"), str):
            artifact = _exact_branch_receipt(
                rid,
                turn,
                status="completed",
                exact_replay=True,
                source_run_id=source_run.get("id"),
                child_run_id=child.get("id"),
                execution_fork_execution_id=receipt.get("execution_id"),
                reasons=[{
                    "code": "exact_child_replay_completed",
                    "message": (
                        "the unchanged control matched and an exact same-prompt child replay "
                        "was persisted"),
                }],
                capture=capture,
                execution_fork=receipt,
            )
            h._json(201, artifact)
            return True

        reason = (receipt.get("reasons") or [{}])[0] if isinstance(receipt, Mapping) else {}
        artifact = _exact_branch_receipt(
            rid,
            turn,
            status="failed",
            exact_replay=False,
            source_run_id=source_run.get("id"),
            execution_fork_execution_id=(
                receipt.get("execution_id") if isinstance(receipt, Mapping) else None),
            reasons=[{
                "code": str(reason.get("code") or "exact_child_replay_failed"),
                "message": str(reason.get("message") or "the exact child replay did not complete"),
            }],
            capture=capture,
            execution_fork=receipt if isinstance(receipt, Mapping) else {},
        )
        h._json(422, artifact)
        return True
    if p.startswith("/runs/") and p.endswith("/time-machine/verify"):
        rid = p[len("/runs/"):-len("/time-machine/verify")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not isinstance(body, dict) or set(body) - {"turn"}:
            h._json(400, {
                "error": "prompt-boundary verification accepts only {\"turn\": integer}",
                "code": "time_machine_verification_options_unsupported",
            })
            return True
        if "turn" not in body:
            h._json(400, {"error": "need a verification turn", "code": "turn_required"})
            return True
        try:
            turn = int(body["turn"])
        except (TypeError, ValueError):
            h._json(400, {"error": "turn must be an integer", "code": "turn_invalid"})
            return True

        import clozn.replay.timetravel as timetravel
        # Resolve an earlier turn only through its exact immutable session-run prefix.  A final-run
        # checkpoint cannot restore that earlier state, and an ambiguous/missing session history must
        # remain a typed unavailable receipt rather than a guessed worker selection.
        turns = timetravel.completed_message_turns(run)
        source_run = timetravel.resolve_exact_turn_source_run(run, turn)
        if not turns or turn < 0 or turn >= len(turns) or (turn != turns[-1]["turn"] and source_run is None):
            artifact = timetravel.verify_prompt_boundary(
                run, turn, None, runtime_identity={}, worker_identity={})
            try:
                from clozn.replay import timetravel_results
                artifact = timetravel_results.save(artifact)
            except Exception:
                pass
            h._json(422, artifact)
            return True

        from clozn.server.model_routing import select_run_model_facts
        facts = select_run_model_facts(
            h, source_run if source_run is not None else run,
            route="/runs/<id>/time-machine/verify")
        if facts is None:
            return True
        runtime, worker, engine, _sub = facts
        if engine is None or runtime is None or worker is None:
            h._json(503, {
                "error": "exact Time Machine verification requires a ready identity-qualified product worker",
                "code": "time_machine_verification_worker_unavailable",
            })
            return True
        # If this run (or the exact historical source run) has a durable checkpoint pin, hydrate
        # that export into the selected worker before proving the boundary.  The pin store verifies
        # blob and sidecar digests on read; a missing pin simply falls back to the normal recorded
        # history capture path.
        checkpoint_envelope = None
        try:
            from clozn.replay.checkpoint_pin_store import resolve_pin
            resolved_pin = resolve_pin((source_run or run).get("id"))
            if isinstance(resolved_pin, dict) and resolved_pin.get("ok") is True:
                candidate = resolved_pin.get("envelope")
                if isinstance(candidate, dict):
                    checkpoint_envelope = candidate
        except Exception:
            checkpoint_envelope = None
        artifact = timetravel.verify_prompt_boundary(
            run, turn, engine, runtime_identity=runtime, worker_identity=worker,
            source_run=source_run, requested_run_id=rid,
            checkpoint_envelope=checkpoint_envelope)
        try:
            from clozn.replay import timetravel_results
            artifact = timetravel_results.save(artifact)
        except Exception as exc:
            h._json(500, {
                "error": f"Time Machine verification receipt could not be persisted: {type(exc).__name__}: {exc}",
                "code": "time_machine_verification_persistence_error",
            })
            return True
        h._json(201 if artifact["status"] == "verified" else 422, artifact)
        return True
    if p.startswith("/runs/") and p.endswith("/branch"):   # #6: rewind & branch from a turn -> a child run
        rid = p[len("/runs/"):-len("/branch")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.server.model_routing import select_control_model_for_run
        selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/branch")
        if selection is None:
            return True   # typed clozn.model-routing.v1 refusal already written
        sub = selection.sub
        if not (sub and getattr(sub, "chat", None)):   # a branch regenerates through the product model
            h._json(503, {"error": "branch requires a ready product model worker"})
            return True
        if "turn" not in body:
            h._json(400, {"error": "need a branch turn"})
            return True
        try:
            turn = int(body.get("turn"))
        except (TypeError, ValueError):
            h._json(400, {"error": "turn must be an integer"})
            return True
        alt = body.get("alt_user")
        # greedy by default (the receipt path: a branch's future is attributable, not sampling dice).
        sample = bool(body.get("sample", False))
        try:
            import clozn.replay.timetravel as timetravel
            child = timetravel.branch(run, turn, sub, alt_user=alt, sample=sample,
                                      store=ctx._snap_store())
        except Exception as e:
            h._json(500, {"error": f"branch failed: {type(e).__name__}: {e}"})
            return True
        if child is None:                          # None == bad turn index or a generation failure
            h._json(400, {"error": "branch failed (turn out of range, or generation error)"})
            return True
        h._json(200, child)
        return True
    return False
