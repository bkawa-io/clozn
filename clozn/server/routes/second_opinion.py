"""E4: model second opinion, over HTTP.

    GET  /runs/<id>/second-opinion/candidates
    POST /runs/<id>/second-opinion

The Studio "Would another model disagree?" question (`studio-frontend/src/data/askAnotherQuestion.ts`,
id `second_opinion`) is served independently from the mechanistic-diff action. The mechanistic route
now executes compatible teacher-forced comparisons through the managed registry; this second-opinion
route intentionally remains a free-generation experiment and keeps its own compatibility and failure
semantics.

MECHANISTIC-DIFF IS THE WRONG SUBSTRATE FOR THIS -- A FINDING, NOT A REUSE
------------------------------------------------------------------------------
Despite living one route family over, `_mechanistic_diff_action` (and the `clozn diff-model` CLI path it
mirrors) answers a DIFFERENT question: "do these two GGUFs disagree on a shared, teacher-forced
continuation," gated MANDATORILY on identical tokenization (`clozn.analysis.pair_compatibility`'s
`per_token_comparison` operation) because a token id means nothing shared across two different
vocabularies. "Would another model disagree" is the opposite case: the two models are typically NOT the
same tokenizer family -- that is exactly what makes a second opinion interesting -- and the comparison is
over each model's own FREE generation, not a forced continuation. Gating a second opinion on tokenizer
identity would refuse the most useful case (a genuinely different model family) and permit only the least
interesting one (two quants of the same weights, which quant-check already covers). `askAnotherQuestion.
ts`'s own doc comment already draws this line ("a live second model's own answer, run to compare against"
vs. "compare two models' internals"); this route implements the former. See `clozn.runs.second_opinion`'s
module docstring for the rest of this design.

WHY A PRE-FLIGHT REFUSAL IS A HARD HTTP ERROR BUT A POST-RESOLUTION FAILURE IS NOT
----------------------------------------------------------------------------------------
`select_control_model_for_run` already has an established, shared contract: on failure it writes a typed
`clozn.model-routing.v1` refusal directly to `h` and returns `None` (see every other run-scoped action in
this codebase -- fork, causal-trace, source-measure, retry). This route reuses that contract as-is for
resolving the SECOND model: an unknown model id, a not-ready worker, or no managed router at all is a
hard HTTP error, exactly like every sibling action. That is deliberately NOT where this feature's "one
arm's failure must not invalidate completed arms" honesty requirement lives -- there is no arm_a/arm_b
document to return yet at that point, only a request that could not even be attempted.

That requirement instead governs what happens AFTER a worker is successfully resolved and identity-
qualified: if the actual generation call then fails (engine error, timeout, degenerate response), the
route still returns 200 with the anchor's full evidence intact and the failure recorded as a typed
`arm_b.status` (`clozn.runs.second_opinion.run_second_opinion_arm` never raises on arm_b's behalf). This
is the real boundary: "can we even ask" is a request-shaped failure (4xx/5xx, matching every sibling
action); "we asked and it went wrong" is investigation evidence (200, structural per-arm status) --
never the reverse, and never collapsed into one.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4) -- no edit to clozn/server/app.py.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

_CANDIDATES_SUFFIX = "/second-opinion/candidates"
_ACTION_SUFFIX = "/second-opinion"


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_CANDIDATES_SUFFIX)):
        return False
    run_id = p[len("/runs/"):-len(_CANDIDATES_SUFFIX)]
    if not run_id:
        return False

    import clozn.runs.store as runlog

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    from clozn.server import app as ctx

    router = getattr(ctx, "MODEL_ROUTER", None)
    own_model = run.get("model")
    own_model = own_model if isinstance(own_model, str) else None
    if router is None:
        h._json(200, {"managed": False, "own_model_id": own_model, "candidates": []})
        return True

    from clozn.server.model_routing import peek_control_model_for_run

    candidates = []
    for model_id in router.model_ids():
        if model_id == own_model:
            continue
        sub = peek_control_model_for_run(h, model_id, route="/runs/<id>/second-opinion/candidates")
        candidates.append({"model_id": model_id, "ready": sub is not None})
    h._json(200, {"managed": True, "own_model_id": own_model, "candidates": candidates})
    return True


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith(_ACTION_SUFFIX)):
        return False
    run_id = p[len("/runs/"):-len(_ACTION_SUFFIX)]
    if not run_id:
        return False
    body = body if isinstance(body, dict) else {}

    import clozn.runs.store as runlog

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    second_model = body.get("model")
    if not isinstance(second_model, str) or not second_model.strip():
        h._json(400, {"error": "body.model must name the second (already-preloaded) model to ask"})
        return True
    second_model = second_model.strip()
    if second_model == run.get("model"):
        h._json(400, {
            "error": (
                "body.model is the same as this run's own model -- a second opinion needs a "
                "DIFFERENT model; asking the same model again is not a second opinion"),
        })
        return True

    from clozn.server import app as ctx

    if getattr(ctx, "MODEL_ROUTER", None) is None:
        h._json(503, {
            "error": (
                "model second opinion requires a managed multi-model gateway; this process is serving "
                "only one model, so a genuinely different, already-loaded second model cannot exist"),
            "code": "second_opinion_requires_managed_router",
        })
        return True

    from clozn.server.model_routing import select_control_model_for_run

    selection = select_control_model_for_run(h, second_model, route="/runs/<id>/second-opinion")
    if selection is None:
        return True   # typed clozn.model-routing.v1 refusal already written
    sub = selection.sub
    if not (sub and callable(getattr(sub, "chat", None))):
        h._json(503, {"error": "second opinion requires a ready product model worker"})
        return True

    from clozn.runs.second_opinion import build_second_opinion

    document = build_second_opinion(run, selection, requested_model_id=second_model)
    h._json(200, document)
    return True
