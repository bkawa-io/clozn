"""server/routes/snapshot.py -- FORK-PIN-01: POST /runs/<id>/snapshot/pin.

The gateway-level composition that makes durable pinning reachable from a running product server:
(1) reconstruct the run as a live, in-memory checkpoint on whichever worker currently serves its
model (clozn.replay.checkpoint_capture.capture_parent_checkpoint, using the shared run-scoped
model/runtime identity resolver);
(2) export it via the engine's new POST /v1/checkpoint/export; (3) persist it durably via
clozn.replay.checkpoint_pin_store.pin_checkpoint (content-addressed blob + SQLite metadata).

``clozn snapshot unpin``/``clozn snapshot list`` still need no live worker at all -- they are pure local
SQLite/blob operations -- and the CLI continues to call clozn.replay.checkpoint_pin_store directly for
those. Studio's read/manage surface is provided by the separate `clozn.server.routes.snapshots` module;
only `pin` needs a live engine (to materialize the checkpoint from the run's recorded history in the
first place).
"""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True

_PIN_SUFFIX = "/snapshot/pin"


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith(_PIN_SUFFIX)):
        return False
    run_id = p[len("/runs/"):-len(_PIN_SUFFIX)]

    import clozn.runs.store as runlog
    parent = runlog.get_run(run_id)
    if parent is None:
        h._json(404, {"error": "run not found"})
        return True

    note = None
    preview = False
    if isinstance(body, Mapping):
        raw_note = body.get("note")
        if raw_note is not None and not isinstance(raw_note, str):
            h._json(400, {"error": "note must be a string", "code": "snapshot_pin_invalid_note"})
            return True
        note = raw_note
        raw_preview = body.get("preview", False)
        if not isinstance(raw_preview, bool):
            h._json(400, {"error": "preview must be a boolean", "code": "snapshot_pin_invalid_preview"})
            return True
        preview = raw_preview
    elif body:
        h._json(400, {
            "error": "snapshot pin accepts only an optional {\"note\": str, \"preview\": bool} body",
            "code": "snapshot_pin_options_unsupported"})
        return True

    from clozn.server.model_routing import select_run_model_facts
    facts = select_run_model_facts(h, parent, route="/runs/<id>/snapshot/pin")
    if facts is None:
        return True  # the routing helper already wrote the routing-error response
    runtime, worker, engine, _sub = facts
    if engine is None or runtime is None or worker is None:
        h._json(503, {
            "error": "pinning a checkpoint requires a ready identity-qualified product worker",
            "code": "snapshot_pin_worker_unavailable",
        })
        return True

    from clozn.replay.checkpoint_capture import CheckpointCaptureError, capture_parent_checkpoint
    try:
        artifact = capture_parent_checkpoint(
            parent, engine, runtime_identity=runtime, worker_identity=worker)
    except CheckpointCaptureError as exc:
        h._json(400, {"error": str(exc), "code": "checkpoint_capture_request_invalid"})
        return True
    except Exception as exc:
        h._json(500, {
            "error": f"checkpoint capture failed: {type(exc).__name__}: {exc}",
            "code": "checkpoint_capture_receipt_error",
        })
        return True

    if artifact.get("status") != "available":
        # Honest, not an error: the run is simply not eligible for exact capture right now (see
        # capture_parent_checkpoint's own status/reasons vocabulary) -- the full artifact rides along
        # so a caller can see exactly why (unavailable vs failed, and the specific reason code).
        h._json(422, {
            "error": "the run is not eligible for a durable checkpoint pin",
            "code": "snapshot_pin_checkpoint_unavailable",
            "capture": artifact,
        })
        return True

    reference = artifact.get("checkpoint_reference")
    checkpoint_id = reference.get("checkpoint_id") if isinstance(reference, Mapping) else None
    source_generation = (
        reference.get("worker_generation_id") if isinstance(reference, Mapping) else None)
    if not (isinstance(checkpoint_id, str) and checkpoint_id
            and isinstance(source_generation, str) and source_generation):
        h._json(500, {
            "error": "captured checkpoint reference is missing checkpoint_id/worker_generation_id",
            "code": "snapshot_pin_receipt_incomplete",
        })
        return True

    try:
        export = engine.export_checkpoint(checkpoint_id, worker_generation_id=source_generation)
    except Exception as exc:
        h._json(502, {
            "error": f"checkpoint export failed: {type(exc).__name__}: {exc}",
            "code": "snapshot_pin_export_failed",
        })
        return True
    envelope = export.get("envelope") if isinstance(export, Mapping) else None
    if not isinstance(envelope, Mapping):
        h._json(502, {
            "error": "worker export returned no envelope",
            "code": "snapshot_pin_export_incomplete",
        })
        return True

    if preview:
        # Show the real byte cost BEFORE anything is durably written -- see the CLI's cmd_pin, which
        # calls this route once with preview:true to display size, then requires --yes before calling
        # it again with preview:false to actually persist. Nothing has touched the blob store or
        # SQLite at this point; the capture above lives only in the worker's OWN ephemeral memory
        # (exactly the existing checkpoint capture -- unchanged by this preview).
        h._json(200, {
            "ok": True, "preview": True, "run_id": run_id,
            "size_bytes": export.get("size_bytes"),
            "envelope_bytes": export.get("envelope_bytes"),
            "capture": artifact,
        })
        return True

    from clozn.replay.checkpoint_pin_store import PinStoreError, pin_checkpoint
    try:
        manifest = pin_checkpoint(
            run_id, envelope,
            checkpoint_id=checkpoint_id,
            source_worker_generation_id=source_generation,
            note=note,
        )
    except PinStoreError as exc:
        h._json(400, {"error": str(exc), "code": "snapshot_pin_store_error"})
        return True
    except Exception as exc:
        h._json(500, {
            "error": f"could not persist pin: {type(exc).__name__}: {exc}",
            "code": "snapshot_pin_store_error",
        })
        return True

    h._json(201, {"ok": True, "manifest": manifest, "capture": artifact})
    return True
