"""The C3 HTTP surface -- "point at a passage or a setting and ask whether it mattered."

Slice 1 (this morning, `746a723`) built the core with NO route on purpose: `clozn.runs.
investigation_experiment.plan_experiment()` is the model-free, mutation-free eligibility planner;
`clozn.receipts.investigation_experiment.run_experiment()` is the four-arm controlled executor, the
ONE place `analysis`/`causal_claim` are computed (see that module's own docstring for the full
argument). This file adds three things over that core, and computes NOTHING new:

    POST /runs/<id>/investigation-experiment/plan   -- sync, model-free: plan_experiment() verbatim.
    POST /runs/<id>/investigation-experiment         -- starts a bounded background job that runs
                                                          run_experiment() against the run's own model.
    GET  /runs/<id>/investigation-experiment/jobs/<job_id>          -- poll.
    POST /runs/<id>/investigation-experiment/jobs/<job_id>/cancel   -- cancel.

WHY A JOB, NOT A SYNCHRONOUS POST
----------------------------------
run_experiment() drives 2-3 full, blocking `sub.chat()` generations (no_op_replay, treatment, and --
when a matched control exists -- random_equal_effect_control), each a real greedy decode against a
live worker. That is materially heavier than a single generation, in the same weight class as
the controlled experiment executor and `clozn.analysis.tracer.trace` (one job) -- both of
which the token-workbench action surface (`clozn/server/routes/token_workbench_actions.py`) already
runs through `clozn.server.influence_jobs.JOBS` rather than blocking the HTTP thread, specifically so
a caller polls progress and can cancel instead of holding a connection open across multiple sequential
decodes. A synchronous POST here would either time out a real client on a slow worker or return
"complete" for arms that are, deceptively, still running -- exactly what the brief forbids. This file
reuses that SAME shared job registry (`kind="investigation_experiment"`) rather than inventing a
second lifecycle/cancellation story.

The `/plan` route stays synchronous and separate on purpose (requirement: plan and execute must be
distinguishable): `plan_experiment()` touches no substrate and generates nothing, so there is nothing
to make async. `/investigation-experiment` (execute) ALSO calls `plan_experiment()` first, synchronously,
before spending a worker selection or a job slot -- not a second eligibility ruling, the exact SAME
pure function `run_experiment()` itself calls first internally (see that module's docstring: "Plan...
which is ALWAYS consulted first"). A request that would refuse never reaches the job registry at all.

NO SECOND PATH TO A CAUSAL CLAIM
----------------------------------
This file never reads `document["causal_claim"]` or `document["analysis"]` to decide anything, and
never sets either key. `plan_experiment()`'s output only ever carries `phase`/`eligibility`/`plan` --
no causal fields exist yet at that phase. `run_experiment()`'s output is attached to the job VERBATIM
after a schema-conformance check; the only field of it this file inspects is `phase` (to decide the
JOB's own completed-vs-failed lifecycle state), never `analysis.effect_specific` or
`causal_claim.licensed`. Every response body this route produces is either the core's own document,
unmodified, or a typed refusal this file did not derive from the run's content.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4) -- no edit to clozn/server/app.py.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

SCHEMA_VERSION = "clozn.investigation-experiment.v1"
_MODEL_ROUTING_ROUTE = "/runs/<id>/investigation-experiment"
_MARKER = "/investigation-experiment"


def _split(p: str):
    """(run_id, trailing) for `/runs/<id>/investigation-experiment<trailing>`, where `trailing` is
    `""`, `"/plan"`, `"/jobs/<job_id>"`, or `"/jobs/<job_id>/cancel"` -- or None when `p` is not this
    route family at all (never claims a path it cannot fully parse)."""
    if not p.startswith("/runs/"):
        return None
    middle = p[len("/runs/"):]
    if _MARKER not in middle:
        return None
    run_id, _, rest = middle.partition(_MARKER)
    if not run_id:
        return None
    return run_id, rest


def _validated(document: dict):
    """`(document, None)` if `document` conforms to clozn.investigation-experiment.v1, else
    `(None, error_payload)` -- the one write-boundary check this file makes on every core-produced
    document before it leaves the process (mirrors investigation.py's own
    `investigation_contract_invalid` convention): a document the core's OWN schema rejects must fail
    loudly here, never sail through and mislead a caller into treating it as trustworthy evidence."""
    from clozn import schemas

    try:
        schemas.validate(document, SCHEMA_VERSION)
    except schemas.ValidationError as exc:
        return None, {
            "error": "investigation-experiment document failed its own schema",
            "code": "investigation_experiment_contract_invalid",
            "detail": str(exc),
        }
    return document, None


# ==================================================================================================== plan
def _plan(h, run_id: str, body: dict) -> bool:
    import clozn.runs.store as runlog

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    intervention = body.get("intervention")
    if not isinstance(intervention, dict):
        h._json(400, {"error": "body.intervention must be an object"})
        return True

    from clozn.runs.investigation_experiment import plan_experiment

    try:
        document = plan_experiment(run, intervention)
    except ValueError as exc:
        h._json(400, {"error": str(exc)})
        return True

    document, error = _validated(document)
    if error is not None:
        h._json(500, error)
        return True

    # A typed refusal is a successful ANSWER to "can this be run", not an HTTP failure -- mirrors
    # `clozn investigate-experiment`'s own CLI exit-0-in-either-case convention. `phase` in the body
    # already carries the distinguishing information; this route does not also encode it in a
    # domain-specific status code the way the execute route below does for a real action request.
    h._json(200, document)
    return True


# ================================================================================================= execute
def _job_worker(run: dict, intervention: dict, sub):
    """The JobControl-shaped `worker(control)` callable: composes `clozn.receipts.
    investigation_experiment.run_experiment()` read-only. Whatever document it returns is attached
    VERBATIM (after a schema-conformance check) -- this function never re-labels, re-derives, or
    inspects `analysis`/`causal_claim`.

    `run_experiment()` has no cooperative cancellation hook (each arm is one blocking `sub.chat()`
    call) -- the same documented caveat `fork_worker`/`causal_trace_worker`
    (`clozn/runs/token_workbench_actions.py`) carry: a cancel request is honored at the job-STATE
    level only (`control.attach_result()` raises `JobCancelled` if cancellation raced the result), but
    an arm already in flight keeps running to completion in this thread."""
    def worker(control):
        from clozn.receipts.investigation_experiment import run_experiment

        control.checkpoint(phase="running_arms", completed=0, total=1)
        try:
            document = run_experiment(run, intervention, sub)
        except Exception as exc:
            return {"state": "failed", "error": {
                "code": "investigation_experiment_job_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }}
        control.checkpoint(phase="validating", completed=1, total=1)

        document, error = _validated(document)
        if error is not None:
            return {"state": "failed", "error": {
                "code": error["code"],
                "message": f"investigation-experiment artifact failed its own schema: {error['detail']}",
            }}

        # Attach the document either way -- even a `phase: "failed"` document carries the plan and a
        # typed `error` a caller wants to inspect (`run_experiment`'s own "the plan survives onto a
        # failed document" discipline). This file reads only `phase` here, never `analysis`/
        # `causal_claim`, to decide the JOB's lifecycle -- a generation failure is this job's own
        # failure (mirrors `fork_worker`'s `child is None` -> job "failed"), not a completed-but-
        # inconclusive analysis the way `causal_trace_worker`'s `blocked` verdict is.
        control.attach_result(document)
        if document.get("phase") == "failed":
            arm_error = document.get("error") if isinstance(document.get("error"), dict) else {}
            return {"state": "failed", "error": {
                "code": "investigation_experiment_generation_failed",
                "message": arm_error.get("message") or "a generation arm failed",
            }}
        return {"state": "completed"}

    return worker


def _execute(h, run_id: str, body: dict) -> bool:
    import clozn.runs.store as runlog

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    intervention = body.get("intervention")
    if not isinstance(intervention, dict):
        h._json(400, {"error": "body.intervention must be an object"})
        return True

    from clozn.runs.investigation_experiment import plan_experiment

    try:
        preflight = plan_experiment(run, intervention)
    except ValueError as exc:
        h._json(400, {"error": str(exc)})
        return True

    if preflight.get("phase") != "planned":
        # Ineligible: refused BEFORE any worker is selected or any job slot is spent -- the exact
        # typed reason plan_experiment already reasoned about, surfaced verbatim (never a 500, never
        # a generic error; see this module's own docstring).
        preflight, error = _validated(preflight)
        if error is not None:
            h._json(500, error)
            return True
        h._json(422, preflight)
        return True

    from clozn.server.model_routing import select_control_model_for_run

    selection = select_control_model_for_run(h, run.get("model"), route=_MODEL_ROUTING_ROUTE)
    if selection is None:
        return True   # typed clozn.model-routing.v1 refusal already written
    sub = selection.sub
    if not (sub and callable(getattr(sub, "chat", None))):
        h._json(503, {
            "error": "investigation-experiment requires a ready product model worker",
            "code": "investigation_experiment_worker_unavailable",
        })
        return True

    from clozn.server.influence_jobs import JOBS, JobCapacityError

    worker = _job_worker(run, dict(intervention), sub)
    try:
        job = JOBS.start(run_id, worker, kind="investigation_experiment")
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc)})
        return True
    h._json(202, job)
    return True


# ==================================================================================================== jobs
def try_get(h, p):
    parsed = _split(p)
    if parsed is None:
        return False
    run_id, rest = parsed
    if not rest.startswith("/jobs/") or rest.endswith("/cancel"):
        return False
    job_id = rest[len("/jobs/"):]
    if not job_id or "/" in job_id:
        return False

    from clozn.server.influence_jobs import JOBS

    job = JOBS.get(run_id, job_id)
    if job is None:
        h._json(404, {"error": "investigation-experiment job not found"})
    else:
        h._json(200, job)
    return True


def try_post(h, p, body):
    parsed = _split(p)
    if parsed is None:
        return False
    run_id, rest = parsed
    body = body if isinstance(body, dict) else {}

    if rest.startswith("/jobs/") and rest.endswith("/cancel"):
        job_id = rest[len("/jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS

        job = JOBS.cancel(run_id, job_id)
        if job is None:
            h._json(404, {"error": "investigation-experiment job not found"})
        else:
            h._json(200, job)
        return True

    if rest == "/plan":
        return _plan(h, run_id, body)
    if rest == "":
        return _execute(h, run_id, body)
    return False
