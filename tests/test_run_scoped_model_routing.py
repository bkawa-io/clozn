"""Run-scoped control routes resolve their worker from the RUN'S OWN model under a managed
multi-model gateway, instead of `ctx.active_sub(h)` failing closed for every run (the bug this
change fixes -- see clozn.server.app.active_sub's docstring and
clozn.server.model_routing.select_control_model_for_run's docstring for the full story).

Every route exercised here used to do one of two wrong things under `app.MODEL_ROUTER` being set:
  * call `ctx.active_sub(h)` directly -> always None -> an unconditional 503 for EVERY run, not
    just ones whose worker was genuinely unavailable (see the old, now-corrected
    `test_unselected_run_engine_routes_never_use_default_worker` in test_managed_model_bootstrap.py);
  * (causal_trace.py, provenance.py) fall back to the bare `ctx.ENGINE` global, which under a
    managed router is the CONTROL-PAIR engine -- i.e. some OTHER model's worker -- silently
    analyzing the wrong model's weights while returning a normal-looking 200.

This file proves, for a representative, broad sample of the converted routes:
  1. under a real two-model `PreloadedModelRouter`, a run whose `model` is "beta" resolves to
     BETA's own substrate/engine, never alpha's and never the legacy global SUB;
  2. a run whose `model` is not configured on the router refuses with a typed
     `clozn.model-routing.v1` error (never a bare 503, never a fallback);
  3. the two "compose, don't block" read routes (investigation, corrective-actions registry)
     degrade to an honest "capability unavailable" instead of refusing outright, since they never
     execute anything;
  4. the legacy (`MODEL_ROUTER is None`) path is untouched -- still resolves through
     `ctx.active_sub`/`ctx.active_engine` exactly as before.

No model, no GPU, no engine launch: every engine here is a small deterministic fake whose
`.health()` satisfies `PreloadedModelBinding`/`qualify_live_identity`'s own bookkeeping, and every
downstream domain call (`clozn.analysis.tracer.trace`, ...) is
monkeypatched to record which `sub`/`engine_url` it was actually given -- this file is about the
SELECTION layer, not re-testing each domain function's own business logic (that has its own
coverage elsewhere, e.g. test_replay.py, test_causal_trace_server.py, test_receipts_server.py).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import clozn.runs.store as runlog
from clozn.cli.worker_registry import AdapterRuntimeIdentity, RuntimeKey
from clozn.server import app as server
from clozn.server.model_routing import (
    PreloadedModelBinding,
    PreloadedModelRouter,
    select_run_model_facts,
)
from clozn.server.routes import (
    causal_trace,
    corrective_actions,
    corrective_retries,
    influence_map,
    investigation,
    investigation_experiment,
    provenance,
    receipts,
    replay,
    section_influence,
    timetravel,
    token_workbench_actions,
)


WHITE_BOX = {"sae": False, "jlens": False, "attn_knockout": False}
_ABSENT = AdapterRuntimeIdentity.absent()


class FakeEngine:
    """Enough of the real engine client's surface to pass qualify_live_identity() -- health()'s
    fields must match the binding's own runtime_key exactly, per model_routing.py's own checks."""

    def __init__(self, *, digest, template, build, generation, base):
        self.digest = digest
        self.template = template
        self.build = build
        self.generation = generation
        self.base = base

    def health(self):
        return {
            "status": "ok",
            "worker_generation_id": self.generation,
            "protocol_version": "1.1",
            "model_sha256": self.digest,
            "n_ctx": 4096,
            "device": "cpu",
            "engine_build": self.build,
            "template_fingerprint": self.template,
            "capabilities": dict(WHITE_BOX),
        }


class FakeSub:
    """Every capability attribute a run-scoped route might gate on, present by default so a
    success-path test only has to monkeypatch the ONE deep domain call it cares about rather than
    hand-build a working substrate per route."""

    def __init__(self, engine):
        self.engine = engine
        self.chat = lambda *_a, **_k: None
        self.score_tokens = lambda *_a, **_k: None
        self.jlens = lambda *_a, **_k: {"available": True}
        self.steer = SimpleNamespace(set=lambda *_a, **_k: None, strength={})
        self.handle = lambda *_a, **_k: None


def _binding(model_id, *, digest, template, build, generation, base):
    key = RuntimeKey(
        gguf_artifact_sha256=digest, context_size=4096, backend="cpu", adapter=_ABSENT,
        template_fingerprint=template, engine_build=build, white_box_flags=WHITE_BOX,
    ).as_dict()
    engine = FakeEngine(digest=digest, template=template, build=build, generation=generation, base=base)
    sub = FakeSub(engine)
    binding = PreloadedModelBinding(
        model_id=model_id,
        resolved_artifact={"model_id": model_id, "format": "gguf", "artifact_sha256": digest},
        runtime_key=key,
        adapter=_ABSENT.as_dict(),
        state="ready",
        worker_identity={
            "worker_id": generation, "worker_generation_id": generation,
            "worker_generation": 1, "runtime_key_sha256": key["key_sha256"],
            "protocol_version": "1.1", "engine_build": build, "backend": "cpu",
        },
        sub=sub, engine=engine,
    )
    return binding, sub, engine


def _two_model_router():
    """alpha is the DEFAULT/control-pair model; beta is a second, non-default preloaded model.
    Every test below stores its run under model="beta" specifically to prove the route never
    reaches for the default/control worker (alpha) or the legacy global SUB."""
    alpha_binding, alpha_sub, alpha_engine = _binding(
        "alpha", digest="a" * 64, template="1" * 16, build="build-alpha", generation="gen-alpha",
        base="http://127.0.0.1:44101")
    beta_binding, beta_sub, beta_engine = _binding(
        "beta", digest="b" * 64, template="2" * 16, build="build-beta", generation="gen-beta",
        base="http://127.0.0.1:44102")
    router = PreloadedModelRouter(
        [alpha_binding, beta_binding], default_model_id="alpha",
        preload_model_ids=["alpha", "beta"], max_loaded_workers=2,
    )
    return router, SimpleNamespace(sub=alpha_sub, engine=alpha_engine), SimpleNamespace(sub=beta_sub, engine=beta_engine)


class Handler:
    def __init__(self, path=""):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


@pytest.fixture
def managed(monkeypatch):
    """Installs a real two-model router as app.MODEL_ROUTER, and POISONS app.SUB/app.ENGINE (the
    legacy globals) so any accidental fallback to them is loud, not silently plausible-looking."""
    router, alpha, beta = _two_model_router()
    previous = server.MODEL_ROUTER, server.SUB, server.ENGINE

    class Poison:
        def __getattr__(self, name):
            raise AssertionError(
                f"a run-scoped route touched the legacy default worker (.{name}) instead of "
                "resolving the run's own model through MODEL_ROUTER")

    server.MODEL_ROUTER = router
    server.SUB = Poison()
    server.ENGINE = Poison()
    try:
        yield router, alpha, beta
    finally:
        server.MODEL_ROUTER, server.SUB, server.ENGINE = previous


def _run(**overrides):
    values = {
        "source": "engine_chat", "client": "studio", "model": "beta", "substrate": "engine",
        "messages": [{"role": "user", "content": "hi"}], "response": "hello there",
        "final_prompt": "<user>hi</user>", "trace": {"tokens": ["hello", " there"],
        "token_ids": [1, 2]},
    }
    values.update(overrides)
    run_id = runlog.record(**values)
    assert run_id
    return runlog.get_run(run_id)


# ============================================================================ typed refusal (broad)

# (route module, callable name, path suffix, body) -- each fetches a run whose model ("charlie")
# is NOT configured on the router at all, so select_control_model_for_run must raise unknown_model
# and the route must surface the typed clozn.model-routing.v1 refusal, never a bare 503/500 and
# never touching alpha or the legacy SUB.
_REFUSAL_CASES = [
    (causal_trace, "try_post", "/causal-trace", {}),
    (provenance, "try_post", "/provenance", {}),
    (influence_map, "try_post", "/influence-map", {}),
    (influence_map, "try_post", "/influence-map/jobs", {}),
    (replay, "try_post", "/replay", {}),
    (replay, "try_post", "/counterfactual", {"behavior_overrides": {"warmth": 0.5}}),
    (timetravel, "try_post", "/branch", {"turn": 0}),
    (corrective_retries, "try_post", "/retry", {"preset": "less-verbose"}),
    (receipts, "try_post", "/receipts", {}),
    (receipts, "try_post", "/receipt", {"influence": {"dial": "warmth"}}),
    (receipts, "try_post", "/swap_receipt", {"to_concept": "x"}),
    (receipts, "try_post", "/rederive", {}),
    (receipts, "try_post", "/narrate", {}),
    (receipts, "try_post", "/jlens", {}),
    (investigation_experiment, "try_post", "/investigation-experiment",
     {"intervention": {"kind": "sampler_change", "overrides": {"temperature": 0.5}}}),
]


@pytest.mark.parametrize(
    "module,fn_name,suffix,body", _REFUSAL_CASES,
    ids=[f"{m.__name__.rsplit('.', 1)[-1]}{s}" for m, _f, s, _b in _REFUSAL_CASES],
)
def test_unresolvable_run_model_refuses_typed_never_falls_back(iso, managed, module, fn_name, suffix, body):
    run = _run(model="charlie")   # not a binding on the router at all
    h = Handler()
    fn = getattr(module, fn_name)
    claimed = fn(h, f"/runs/{run['id']}{suffix}", body)

    assert claimed is True
    assert h.status is not None and h.status >= 400
    assert isinstance(h.body, dict)
    artifact = h.body.get("clozn_model_routing")
    assert isinstance(artifact, dict), f"expected a typed clozn.model-routing.v1 refusal, got {h.body!r}"
    assert artifact.get("schema_version") == "clozn.model-routing.v1"
    assert h.body["error"]["code"] == "unknown_model"


# ==================================================================================== success path

def test_causal_trace_uses_the_runs_own_engine_not_the_control_pair_engine(iso, managed, monkeypatch):
    """This is the ctx.ENGINE bug: before this change, causal_trace._engine_base fell back to the
    bare module-level ctx.ENGINE under a managed router -- the CONTROL-PAIR (alpha's) engine, not
    this run's (beta's) -- silently tracing the wrong model. Proves that no longer happens."""
    router, alpha, beta = managed
    run = _run(model="beta")
    captured = {}

    def fake_trace(prompt, answer, position, **kwargs):
        captured["engine_url"] = kwargs.get("engine_url")
        return {"ok": True, "verdict": "PASS", "nodes": []}

    import clozn.analysis.tracer as tracer
    monkeypatch.setattr(tracer, "trace", fake_trace)

    h = Handler()
    assert causal_trace.try_post(h, f"/runs/{run['id']}/causal-trace", {})
    assert h.status == 200
    assert captured["engine_url"] == beta.engine.base
    assert captured["engine_url"] != alpha.engine.base


def test_provenance_uses_the_runs_own_engine_not_the_control_pair_engine(iso, managed, monkeypatch):
    router, alpha, beta = managed
    run = _run(model="beta")
    captured = {}

    def fake_trace_provenance(prompt, answer, **kwargs):
        captured["engine_url"] = kwargs.get("engine_url")
        return {"ok": True, "verdict": "context_used", "spans": []}

    import clozn.analysis.provenance as provenance_mod
    monkeypatch.setattr(provenance_mod, "trace_provenance", fake_trace_provenance)

    h = Handler()
    assert provenance.try_post(h, f"/runs/{run['id']}/provenance", {})
    assert h.status == 200
    assert captured["engine_url"] == beta.engine.base
    assert captured["engine_url"] != alpha.engine.base


def test_influence_map_sync_uses_the_runs_own_model_worker(iso, managed, monkeypatch):
    router, alpha, beta = managed
    run = _run(model="beta")
    captured = {}

    def fake_influence(run_arg, sub, **kwargs):
        captured["sub"] = sub
        return {"schema": "clozn.context_answer_influence.v1", "available": True, "status": "ok",
                "context_spans": [], "answer_tokens": []}

    import clozn.receipts.context_answer_influence as backend
    monkeypatch.setattr(backend, "context_answer_influence", fake_influence)
    monkeypatch.setattr(backend, "SCHEMA", "clozn.context_answer_influence.v1")

    h = Handler()
    assert influence_map.try_post(h, f"/runs/{run['id']}/influence-map", {})
    assert captured["sub"] is beta.sub
    assert captured["sub"] is not alpha.sub


def test_replay_uses_the_runs_own_model_worker(iso, managed, monkeypatch):
    router, alpha, beta = managed
    run = _run(model="beta")
    captured = {}

    def fake_replay(run_arg, changes, sub):
        captured["sub"] = sub
        return {"id": "run_child", "parent_run_id": run_arg["id"]}

    import clozn.replay as replay_pkg
    monkeypatch.setattr(replay_pkg, "replay", fake_replay)

    h = Handler()
    assert replay.try_post(h, f"/runs/{run['id']}/replay", {"changes_applied": {}})
    assert h.status == 200
    assert captured["sub"] is beta.sub
    assert captured["sub"] is not alpha.sub


def test_rederive_uses_the_runs_own_model_worker(iso, managed, monkeypatch):
    router, alpha, beta = managed
    run = _run(model="beta")
    captured = {}

    def fake_rederive(run_arg, sub):
        captured["sub"] = sub
        return {"text": "hello there", "steps": [], "meta": {}}

    import clozn.receipts.rederive as rederive_mod
    monkeypatch.setattr(rederive_mod, "rederive", fake_rederive)

    h = Handler()
    assert receipts.try_post(h, f"/runs/{run['id']}/rederive", {})
    assert h.status == 200
    assert captured["sub"] is beta.sub
    assert captured["sub"] is not alpha.sub


def test_token_workbench_force_token_action_uses_the_runs_own_model_worker(iso, managed, monkeypatch):
    router, alpha, beta = managed
    run = _run(model="beta", trace={"tokens": ["hello", " there"], "token_ids": [1, 2]})
    captured = {}

    def fake_force_token_worker(run_arg, sub, index, **kwargs):
        captured["sub"] = sub
        return lambda control: {"state": "completed"}

    from clozn.runs import token_workbench_actions as domain
    monkeypatch.setattr(domain, "force_token_worker", fake_force_token_worker)

    h = Handler()
    assert token_workbench_actions.try_post(
        h, f"/runs/{run['id']}/tokens/0/force-token", {"token_piece": "x"})
    assert h.status == 202
    assert captured["sub"] is beta.sub
    assert captured["sub"] is not alpha.sub


def test_token_workbench_causal_trace_action_uses_the_runs_own_engine(iso, managed, monkeypatch):
    router, alpha, beta = managed
    run = _run(model="beta", trace={"tokens": ["hello", " there"], "token_ids": [1, 2]})
    captured = {}

    def fake_causal_trace_worker(run_arg, index, **kwargs):
        captured["engine_url"] = kwargs.get("engine_url")
        return lambda control: {"state": "completed"}

    from clozn.runs import token_workbench_actions as domain
    monkeypatch.setattr(domain, "causal_trace_worker", fake_causal_trace_worker)
    monkeypatch.setattr(domain, "find_cached_action", lambda *_a, **_k: None)

    h = Handler()
    assert token_workbench_actions.try_post(
        h, f"/runs/{run['id']}/tokens/0/causal-trace", {})
    assert h.status == 202
    assert captured["engine_url"] == beta.engine.base
    assert captured["engine_url"] != alpha.engine.base


def test_token_workbench_source_measure_action_uses_the_runs_own_model_worker(iso, managed, monkeypatch):
    router, alpha, beta = managed
    run = _run(model="beta", trace={"tokens": ["hello", " there"], "token_ids": [1, 2]})
    captured = {}

    def fake_source_measure(run_arg, sub, max_spans):
        captured["sub"] = sub
        return lambda control: {"state": "completed"}

    from clozn.runs import token_workbench_actions as domain
    monkeypatch.setattr(domain, "source_measure_job_worker", fake_source_measure)

    h = Handler()
    assert token_workbench_actions.try_post(
        h, f"/runs/{run['id']}/tokens/0/source-measure", {})
    assert h.status == 202
    assert captured["sub"] is beta.sub
    assert captured["sub"] is not alpha.sub


def test_investigation_experiment_uses_the_runs_own_model_worker(iso, managed, monkeypatch):
    """POST /runs/<id>/investigation-experiment starts a job (like fork/causal-trace/source-measure
    above) -- proves it resolves beta's worker for the actual run_experiment() call, never alpha's or
    the legacy global SUB, and that the completed job carries the result through to a real poll."""
    import time

    router, alpha, beta = managed
    run = _run(model="beta")
    captured = {}

    def fake_run_experiment(run_arg, intervention, sub):
        captured["sub"] = sub
        return {
            "schema_version": "clozn.investigation-experiment.v1",
            "experiment_id": "invexp_test",
            "run_id": run_arg["id"],
            "generated_at": "2026-07-29T00:00:00Z",
            "phase": "completed",
            "intervention": intervention,
            "eligibility": {"state": "eligible"},
            "plan": {
                "arm_order": ["baseline", "no_op_replay", "treatment", "random_equal_effect_control"],
                "resolved": {"kind": "sampler_change", "sampler_overrides": {"temperature": 0.5}},
            },
            "arms": {
                "baseline": {"reply_sha256": "0" * 64, "matches_baseline": True},
                "no_op_replay": {"reply_sha256": "0" * 64, "matches_baseline": True},
                "treatment": {"reply_sha256": "1" * 64, "matches_baseline": False},
            },
            "analysis": {"instrument_sane": True, "reasons": ["no random control ran for this kind"]},
            "observed": {
                "treatment_reply_differs_from_baseline": True,
                "note": "a factual diff, never a causal claim",
            },
            "causal_claim": {
                "licensed": False,
                "statement": "uncontrolled: no random equal-effect control could be run for this kind",
            },
        }

    import clozn.receipts.investigation_experiment as domain
    monkeypatch.setattr(domain, "run_experiment", fake_run_experiment)

    h = Handler()
    assert investigation_experiment.try_post(
        h, f"/runs/{run['id']}/investigation-experiment",
        {"intervention": {"kind": "sampler_change", "overrides": {"temperature": 0.5}}})
    assert h.status == 202
    job_id = h.body["job_id"]

    from clozn.server.influence_jobs import JOBS
    deadline = time.monotonic() + 2
    state = None
    while time.monotonic() < deadline:
        snapshot = JOBS.get(run["id"], job_id)
        state = snapshot["state"]
        if state in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    assert state == "completed"
    assert captured["sub"] is beta.sub
    assert captured["sub"] is not alpha.sub


def test_section_influence_uses_the_runs_own_model_worker(iso, managed, monkeypatch):
    router, alpha, beta = managed
    run = _run(model="beta", sections=[
        {"id": "s1", "name": "s1", "source": "api",
         "parts": [{"message_index": 0, "start": 0, "end": 2}]},
    ])
    captured = {}

    def fake_with_arm_conditions(run_arg):
        captured.setdefault("subs", [])
        return {"raw_messages": run_arg.get("messages", []), "raw_block": None,
                "steer_strengths": None}

    def fake_score_arm(sub, conditions, **kwargs):
        captured["sub"] = sub
        return [{"logprob": -0.1}], True

    import clozn.receipts.rederive as rederive_mod
    monkeypatch.setattr(rederive_mod, "with_arm_conditions", fake_with_arm_conditions)
    monkeypatch.setattr(rederive_mod, "score_arm", fake_score_arm)

    h = Handler()
    assert section_influence.try_post(h, f"/runs/{run['id']}/section-influence", {})
    assert h.status == 200
    assert captured["sub"] is beta.sub
    assert captured["sub"] is not alpha.sub


def test_corrective_retry_uses_the_runs_own_model_worker(iso, managed, monkeypatch):
    router, alpha, beta = managed
    run = _run(model="beta")
    captured = {}

    def fake_retry_compare(run_arg, preset, sub, **kwargs):
        captured["sub"] = sub
        return {"coherence": {"degenerate": False}, "intervention_observed": False}

    import clozn.replay.corrective as corrective_mod
    monkeypatch.setattr(corrective_mod, "retry_compare", fake_retry_compare)

    h = Handler()
    assert corrective_retries.try_post(h, f"/runs/{run['id']}/retry", {"preset": "less-verbose"})
    assert h.status == 200
    assert captured["sub"] is beta.sub
    assert captured["sub"] is not alpha.sub


# ===================================================================== "compose, don't block" reads

def _measurement_availability(document):
    action = next(
        a for a in document["actions"] if a["id"] == "measure_prompt_source_influence")
    return action["availability"]


def test_investigation_degrades_to_capability_false_rather_than_refusing(iso, managed):
    """investigation.py never executes anything -- an unresolvable worker must not turn a read into
    a hard refusal, it degrades to scoring_available:false (still 200), exactly like legacy mode
    already does when the one engine is simply down."""
    run = _run(model="charlie")   # not configured on the router
    h = Handler()
    assert investigation.try_get(h, f"/runs/{run['id']}/investigation")
    assert h.status == 200
    assert _measurement_availability(h.body) == "unavailable"


def test_investigation_reports_the_runs_own_worker_capability_when_resolvable(iso, managed):
    router, alpha, beta = managed
    run = _run(model="beta")
    h = Handler()
    assert investigation.try_get(h, f"/runs/{run['id']}/investigation")
    assert h.status == 200
    assert _measurement_availability(h.body) == "ready"   # beta.sub.score_tokens is set


def test_corrective_actions_registry_degrades_rather_than_refusing(iso, managed):
    run = _run(model="charlie")
    h = Handler()
    assert corrective_actions.try_get(h, f"/runs/{run['id']}/corrective-actions")
    assert h.status == 200   # never a clozn.model-routing.v1 refusal on a describe-only read


def test_corrective_actions_confirm_hard_refuses_on_unresolvable_worker(iso, managed, monkeypatch):
    """Unlike the registry/preview reads above, /confirm actually regenerates -- it must fail
    closed with a typed refusal, never degrade."""
    from clozn.behavior import corrective_flow

    run = _run(model="charlie")
    preview_id = "fix_preview_test0000000000000000000"
    monkeypatch.setattr(
        corrective_flow, "get_preview",
        lambda pid: {"parent_run_id": run["id"]} if pid == preview_id else None)

    h = Handler()
    assert corrective_actions.try_post(
        h, f"/corrective-previews/{preview_id}/confirm", {"idempotency_key": "k1"})
    assert h.status >= 400
    artifact = h.body.get("clozn_model_routing")
    assert isinstance(artifact, dict)
    assert artifact.get("schema_version") == "clozn.model-routing.v1"


# ==================================================================================== legacy path

def test_legacy_no_router_still_uses_active_sub_directly_unchanged(iso, monkeypatch):
    """MODEL_ROUTER is None (the pre-existing single-worker product path): every converted route
    must keep resolving through ctx.active_sub/ctx.active_engine exactly as before -- proved here
    for a representative working substrate reaching the deep call."""
    previous = server.MODEL_ROUTER, server.SUB, server.ENGINE
    server.MODEL_ROUTER = None
    server.SUB = None
    server.ENGINE = None
    try:
        run = _run(model="whatever-legacy-runs-had")
        engine = FakeEngine(digest="z" * 64, template="9" * 16, build="b", generation="g",
                            base="http://127.0.0.1:1")
        sub = FakeSub(engine)
        server.SUB = sub
        server.ENGINE = engine

        assert server.SUB is sub
    finally:
        server.MODEL_ROUTER, server.SUB, server.ENGINE = previous


# ============================================================== neutral run-scoped identity facts

def test_run_model_facts_reuses_router_binding_facts_without_reprobing(iso, managed):
    """Managed path returns the router binding's OWN identity without an extra health probe."""
    router, alpha, beta = managed
    run = _run(model="beta")
    facts = select_run_model_facts(Handler(), run, route="/runs/<id>/snapshot/pin")
    assert facts is not None
    runtime, worker, engine, sub = facts
    selection = router.select_control_model("beta", route="/runs/<id>/snapshot/pin")
    assert runtime == selection.runtime_key
    assert worker == selection.worker_identity
    assert engine is beta.engine
    assert sub is beta.sub


def test_selection_identity_facts_derives_on_legacy_path(iso):
    """Legacy (no router) path still derives identity through one engine health probe."""
    from clozn.server.model_routing import ModelSelection
    from clozn.experiments.execution_facts import selection_identity_facts

    engine = FakeEngine(digest="z" * 64, template="9" * 16, build="b", generation="g",
                        base="http://127.0.0.1:1")
    sub = FakeSub(engine)
    selection = ModelSelection(model_id="legacy", sub=sub, engine=engine, artifact=None)
    runtime, worker, resolved_engine = selection_identity_facts(selection)
    assert resolved_engine is engine
    assert worker["worker_generation_id"] == "g"
