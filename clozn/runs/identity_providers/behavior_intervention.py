"""Which corrective action (if any) shaped a replayed reply -- roadmap feature 08's execution-
identity requirement, wired through Seam 3 (docs/SEAMS.md).

This is a facet provider, not a general chat-path hook: clozn.server.substrates.EngineSubstrate.
identity_meta() is cached per PROCESS (one /health fetch, reused for every request), so it has no
per-request fact to hand this provider -- threading one through would mean changing identity_meta()'s
signature, a widely shared method (clozn/server/app.py, clozn/replay/replay.py, clozn/server/routes/
openai.py, clozn/server/routes/corrective_retries.py all call it), for one feature's narrow need.

Instead, clozn.replay.corrective.retry_compare() -- the ONLY caller that actually knows which action
ran -- calls clozn.runs.identity_ext.collect() directly with a `behavior_intervention` fact it
already has in hand, and folds the result into its OWN response as `execution_identity`. This module
is that fact's shape-checker: it normalizes and type-guards whatever the caller put in `context[
"behavior_intervention"]`, the same way identity_ext's own worked example (engine_artifact) reads
`context["engine_health"]`.

Because identity_ext.collect() discovers every provider file automatically, this ALSO runs whenever
ANY caller of clozn.runs.identity.runtime_identity() supplies a `behavior_intervention` key in
`extra_context` -- e.g. if a future run-recording path threads the same fact through -- without this
file changing. Callers that don't have the fact simply never populate that key, and this provider
correctly contributes nothing (SEAMS.md rule 2: omit, never null-pad).
"""
from __future__ import annotations

NAME = "behavior_intervention"


def identity(context):
    fact = (context or {}).get("behavior_intervention")
    if not isinstance(fact, dict):
        return None

    action_id = fact.get("action_id")
    backend = fact.get("backend")
    if not isinstance(action_id, str) or not action_id:
        return None
    if not isinstance(backend, str) or not backend:
        return None

    out: dict = {"action_id": action_id, "backend": backend}

    registry_version = fact.get("registry_version")
    if isinstance(registry_version, str) and registry_version:
        out["registry_version"] = registry_version

    parameters = fact.get("parameters")
    if isinstance(parameters, dict) and parameters:
        out["parameters"] = dict(parameters)

    qualification = fact.get("qualification")
    if isinstance(qualification, str) and qualification:
        out["qualification"] = qualification

    # bool is intentionally its own check (never merged with the qualification string above) --
    # "qualified" answers a different question (was THIS exact model/build actually validated) than
    # "qualification" (what KIND of validation this backend claims at all).
    qualified = fact.get("qualified")
    if isinstance(qualified, bool):
        out["qualified"] = qualified

    fallback = fact.get("fallback")
    if isinstance(fallback, bool):
        out["fallback"] = fallback

    return out
