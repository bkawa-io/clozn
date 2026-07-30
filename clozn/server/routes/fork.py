"""Fork-at-token surface: POST /runs/<id>/fork -- FORK-02 compatibility wrapper. Tries the exact
execution-fork path first (checkpoint capture -> plan -> execute, composed read-only from
clozn.replay's proven FORK-00/01/CKPT-01 modules), degrades to the legacy text splice only when no
exact state is honestly available, and returns `unavailable` -- never a silent empty result -- when
neither path can run. -> clozn.replay.fork.compat_fork (orchestration) / clozn.replay.fork.fork (the
splice itself, unchanged).

Body (unchanged): {"position": <int, index into the reply's trace tokens>, "token": "<piece>" |
"token_id": <int>}. Existing callers keep working -- the request shape is untouched, and a successful
response is still the child run record at the top level: `outcome`
(exact_execution_fork | reconstructed_replay | unavailable) and `reasons` are ADDITIVE fields, not a
new envelope. See clozn/replay/fork.py's compat_fork docstring and docs/EXECUTION_FORK_CONTRACT.md for
the shared vocabulary this wrapper reuses rather than reinventing.

Status codes: 404 (no such run), 400 (bad position/token), 503 (no engine reachable at all -- neither
path can even be attempted, same trigger/message as before FORK-02), 200 (a child run was created --
exact or reconstructed), 422 (structurally `unavailable` -- neither path is honestly eligible for this
run/request), 500 (the eligible path's own generation/persistence step failed, mirroring the
pre-FORK-02 contract for that exact failure mode).

Which worker: the substrate resolution (`ctx.active_sub(h)`, and its "no engine at all" -> 503 gate) is
UNCHANGED from before FORK-02, so the multi-model-router 503 contract already tested elsewhere
(test_unselected_run_engine_routes_never_use_default_worker) still holds. The current worker's exact
runtime/worker identity is then looked up opportunistically (clozn.server.routes.execution_fork's
_sub_facts, the same derivation the dedicated execution-fork gateway uses) purely to decide whether the
exact path can be attempted; when it can't be derived, compat_fork simply degrades the outcome -- it
never widens who counts as "no worker ready" at the route level.

Registered in clozn/server/app.py: imported as `_fork_routes` and placed in `_POST_ROUTES`. Live
surface: the Observatory's fork-a-ghost flow (studio-frontend/src/data/api.ts -> Observatory.tsx). The
studio-frontend cutover to read the new `outcome` field is a separate wave (FORK-02's UI half).
"""
from clozn.server import app as ctx


def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/fork"):   # fork the reply at a token -> a child run
        rid = p[len("/runs/"):-len("/fork")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        # A fork teacher-forces a raw prompt prefix through the private worker seam.
        sub = ctx.active_sub(h)
        if not (sub and getattr(sub, "engine", None)):
            h._json(503, {"error": "fork requires a ready product model worker"})
            return True
        if "position" not in body:
            h._json(400, {"error": "need a fork position (an index into the reply's trace tokens)"})
            return True
        try:
            position = int(body.get("position"))
        except (TypeError, ValueError):
            h._json(400, {"error": "position must be an integer"})
            return True

        from clozn.server.routes.execution_fork import _sub_facts
        runtime_identity, worker_identity, _engine = _sub_facts(sub)

        import clozn.replay.fork as fork_mod
        try:
            result = fork_mod.compat_fork(
                run, sub, position, token=body.get("token"), token_id=body.get("token_id"),
                runtime_identity=runtime_identity, worker_identity=worker_identity)
        except ValueError as e:                          # validation: out-of-range / no trace / bad token
            h._json(400, {"error": str(e)})
            return True
        except Exception as e:
            h._json(500, {"error": f"fork failed: {type(e).__name__}: {e}"})
            return True
        if result is None:                                # generation/persistence failure (fork never raises these)
            h._json(500, {"error": "fork failed (generation error, or the run's prompt could not "
                         "be reconstructed)"})
            return True
        status = 422 if result.get("outcome") == "unavailable" else 200
        h._json(status, result)
        return True
    return False
