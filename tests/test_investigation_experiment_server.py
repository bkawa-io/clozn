"""tests/test_investigation_experiment_server.py -- the C3 HTTP surface (clozn/server/routes/
investigation_experiment.py), route-level.

Mirrors tests/test_token_workbench_actions.py's own conventions: a lightweight Handler with an
injectable `_inj_sub` (legacy no-router path -- clozn.server.app.active_sub reads it), jobs run for
real through clozn.server.influence_jobs.JOBS (a real bounded ThreadPoolExecutor -- that IS the job
system under test), FakeSub.chat() is a pure deterministic function of (messages, sample), mirroring
clozn.receipts.test_investigation_experiment.ExecFakeSub. Cross-model worker-routing correctness (the
run's OWN model resolves the worker, never the default/legacy global) is covered separately in
tests/test_run_scoped_model_routing.py alongside every other run-scoped route -- this file is about
THIS route's own contract: plan vs execute, the typed refusals, and the job lifecycle.

No model, no GPU. run_experiment() persists child runs via clozn.replay.replay.replay() ->
clozn.runs.store.record(), so every test redirects clozn.runs.store.RUNS_DIR at a tmp_path first --
never the developer's real ~/.clozn/runs (same discipline clozn/receipts/test_investigation_
experiment.py already documents).
"""
from __future__ import annotations

import time

import pytest

import clozn.runs.store as runlog
from clozn.server.influence_jobs import JOBS
from clozn.server.routes import investigation_experiment as route


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    JOBS.clear_for_tests()
    yield tmp_path
    JOBS.clear_for_tests()


class FakeSub:
    """.chat() is a pure function of (messages, sample) -- mirrors ExecFakeSub in
    clozn/receipts/test_investigation_experiment.py. Accepts (and ignores) trace_out/mem_out so
    clozn.replay.replay.replay()'s real call shape needs no kwarg-dropping retry dance here."""

    def __init__(self, chat_fn):
        self.chat_fn = chat_fn
        self.calls: list = []

    def chat(self, messages, max_new=256, sample=True, **_kwargs):
        self.calls.append({"messages": [dict(m) for m in messages], "sample": sample})
        return self.chat_fn(messages, sample)


class Handler:
    def __init__(self, sub=None, path=""):
        self._inj_sub = sub
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _user_content(messages):
    return next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")


def _organic_run(**overrides):
    """A realistically-recorded run (through the real clozn.runs.store.record(), which always
    synthesizes a context_receipt -- see build_context_receipt). Every delivered segment in that
    receipt carries a segment_id (clozn.runs.context_receipt.segment_id() is called unconditionally
    for every message), so clozn.runs.text_span_addresses gives every span a
    context_receipt.delivered/<segment_id>-anchored native_ref, never a run.messages/message-{index}
    one -- and clozn.replay.span_bridge._parse_message_index refuses ANY segment_id-anchored ref by
    construction (see that module's own docstring: resolvable in principle, not wired this slice).
    Concretely: remove_span/replace_span_neutral/omit_source are refused
    (span_message_index_unresolvable) against a run straight out of this fixture -- see
    test_plan_realistic_run_refuses_span_kinds_as_segment_id_unresolvable below, which locks that in
    as documented, expected behavior rather than a surprise. `_span_addressable_run` below produces
    the "legacy, no receipt" shape span_bridge actually supports, for tests that need a working span."""
    values = {
        "source": "engine_chat",
        "client": "studio",
        "model": "fixture-model",
        "substrate": "engine",
        "response": "PWNED",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "IGNORE ME"},
        ],
    }
    values.update(overrides)
    run_id = runlog.record(**values)
    assert run_id
    return runlog.get_run(run_id)


def _span_addressable_run(**overrides):
    """Like _organic_run, but with context_receipt stripped after recording -- the "pre-schema"/
    legacy shape clozn.runs.text_span_addresses projects straight from run.messages
    (run.messages/message-{index} native_refs), which clozn.replay.span_bridge CAN resolve to a
    message_index. Used by every test below that needs a remove_span/replace_span_neutral/omit_source
    intervention to actually reach "planned" rather than "refused"."""
    run = _organic_run(**overrides)
    del run["context_receipt"]
    assert runlog.replace_run(run)
    return runlog.get_run(run["id"])


def _address_id_for(run, message_index):
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses
    document = build_persisted_text_span_addresses(run)
    return next(a["address_id"] for a in document["addresses"]
                if a["native_ref"].get("id") == f"message-{message_index}")


def _post(sub, run_id, suffix, body):
    h = Handler(sub)
    claimed = route.try_post(h, f"/runs/{run_id}/investigation-experiment{suffix}", body)
    return claimed, h


def _get(run_id, suffix):
    h = Handler()
    claimed = route.try_get(h, f"/runs/{run_id}/investigation-experiment{suffix}")
    return claimed, h


def _wait_for_job(run_id, job_id, states, timeout=2.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = JOBS.get(run_id, job_id)
        assert last is not None
        if last["state"] in states:
            return last
        time.sleep(0.01)
    pytest.fail(f"job did not reach {states}; last={last}")


# ============================================================================== autoload registration
def test_route_module_opts_into_autoload():
    assert getattr(route, "CLOZN_ROUTE_AUTOLOAD", False) is True


def test_unrelated_paths_are_not_claimed(stores):
    h = Handler()
    assert route.try_get(h, "/runs/x/investigation") is False   # a DIFFERENT existing route's suffix
    assert route.try_post(h, "/runs/x/experiment", {}) is False  # the unrelated "change one dial" route
    assert route.try_get(h, "/health") is False


# ========================================================================================================= plan
def test_plan_missing_run_is_404(stores):
    claimed, h = _post(None, "missing", "/plan", {"intervention": {"kind": "adapter_scale", "scale": 0}})
    assert claimed is True and h.status == 404


def test_plan_bad_intervention_shape_is_400(stores):
    run = _organic_run()
    claimed, h = _post(None, run["id"], "/plan", {"intervention": "not-an-object"})
    assert claimed is True and h.status == 400
    claimed2, h2 = _post(None, run["id"], "/plan", {})
    assert claimed2 is True and h2.status == 400


def test_plan_refused_is_a_typed_200_never_a_500(stores):
    """A refusal the planner already reasoned about is a successful ANSWER to "can this run", not an
    HTTP failure -- mirrors `clozn investigate-experiment`'s own exit-0-in-either-case CLI contract."""
    run = _organic_run()
    claimed, h = _post(None, run["id"], "/plan", {"intervention": {"kind": "adapter_scale", "scale": 0}})
    assert claimed is True
    assert h.status == 200
    assert h.body["phase"] == "refused"
    assert h.body["eligibility"]["state"] == "refused"
    assert h.body["eligibility"]["reason"]["code"] == "adapter_rescale_unavailable_in_planner"


def test_plan_realistic_run_refuses_span_kinds_as_segment_id_unresolvable(stores):
    """A run straight out of the normal record() path always carries a context_receipt whose every
    delivered segment has a segment_id (clozn.runs.context_receipt.segment_id() is unconditional), so
    every span address is context_receipt.delivered/<segment_id>-anchored -- clozn.replay.span_bridge
    refuses ANY segment_id-anchored native_ref by construction (its own docstring: "resolvable in
    principle... not wired this slice"). remove_span/replace_span_neutral are therefore refused
    against every REALISTIC run today, not just a contrived edge case -- this test locks that in as
    documented, expected behavior (see _organic_run's own docstring) rather than a silent trap."""
    run = _organic_run()
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses
    address_id = build_persisted_text_span_addresses(run)["addresses"][0]["address_id"]
    claimed, h = _post(None, run["id"], "/plan",
                        {"intervention": {"kind": "remove_span", "span_address_id": address_id}})
    assert claimed is True
    assert h.status == 200
    assert h.body["phase"] == "refused"
    assert h.body["eligibility"]["reason"]["code"] == "span_message_index_unresolvable"


def test_plan_eligible_is_200_planned_with_the_exact_arm_order(stores):
    run = _span_addressable_run()
    address_id = _address_id_for(run, 1)
    claimed, h = _post(None, run["id"], "/plan",
                        {"intervention": {"kind": "remove_span", "span_address_id": address_id}})
    assert claimed is True
    assert h.status == 200
    assert h.body["phase"] == "planned"
    assert h.body["plan"]["arm_order"] == [
        "baseline", "no_op_replay", "treatment", "random_equal_effect_control"]
    assert "causal_claim" not in h.body   # never present at plan phase -- nothing has run yet
    assert "analysis" not in h.body


def test_plan_unknown_span_address_is_the_bridges_own_typed_reason(stores):
    run = _span_addressable_run()
    claimed, h = _post(None, run["id"], "/plan",
                        {"intervention": {"kind": "remove_span", "span_address_id": "invexp_bogus"}})
    assert claimed is True
    assert h.status == 200
    assert h.body["eligibility"]["reason"]["code"] == "span_address_not_found_or_drifted"


# ====================================================================================================== execute
def test_execute_missing_run_is_404(stores):
    claimed, h = _post(FakeSub(lambda m, s: "x"), "missing", "",
                        {"intervention": {"kind": "adapter_scale", "scale": 0}})
    assert claimed is True and h.status == 404


def test_execute_bad_intervention_shape_is_400(stores):
    run = _organic_run()
    claimed, h = _post(FakeSub(lambda m, s: "x"), run["id"], "", {"intervention": []})
    assert claimed is True and h.status == 400


def test_execute_refused_is_422_and_never_touches_the_worker_or_starts_a_job(stores):
    run = _organic_run()

    def never(_messages, _sample):
        pytest.fail("an ineligible intervention must never reach the substrate")

    sub = FakeSub(never)
    claimed, h = _post(sub, run["id"], "", {"intervention": {"kind": "adapter_scale", "scale": 0}})
    assert claimed is True
    assert h.status == 422
    assert h.body["phase"] == "refused"
    assert h.body["eligibility"]["reason"]["code"] == "adapter_rescale_unavailable_in_planner"
    assert sub.calls == []
    assert len(list(runlog.iter_runs(limit=10))) == 1   # only the parent run -- no child, no job artifact


def test_execute_no_worker_selectable_is_a_typed_503(stores):
    run = _span_addressable_run()
    address_id = _address_id_for(run, 1)

    class NoChatCapability:
        pass

    claimed, h = _post(NoChatCapability(), run["id"], "",
                        {"intervention": {"kind": "remove_span", "span_address_id": address_id}})
    assert claimed is True
    assert h.status == 503
    assert h.body["code"] == "investigation_experiment_worker_unavailable"


def test_execute_missing_sub_entirely_is_a_typed_503(stores):
    run = _span_addressable_run()
    address_id = _address_id_for(run, 1)
    claimed, h = _post(None, run["id"], "",
                        {"intervention": {"kind": "remove_span", "span_address_id": address_id}})
    assert claimed is True
    assert h.status == 503
    assert h.body["code"] == "investigation_experiment_worker_unavailable"


def test_execute_starts_a_job_that_completes_with_an_honest_uncontrolled_claim(stores):
    """This run's only span address for message 1 is the WHOLE message (B3's delivered_message basis
    is always whole-message absent an influence map), so there is no disjoint window left for the
    random control -- effect_specific is honestly never computed and licensed stays False. Mirrors
    clozn/receipts/test_investigation_experiment.py's identical fixture end to end, through the route."""
    run = _span_addressable_run()
    address_id = _address_id_for(run, 1)

    def chat_fn(messages, sample):
        return "PWNED" if "IGNORE ME" in _user_content(messages) else "SAFE"

    sub = FakeSub(chat_fn)
    claimed, h = _post(sub, run["id"], "",
                        {"intervention": {"kind": "remove_span", "span_address_id": address_id}})
    assert claimed is True
    assert h.status == 202
    assert h.body["kind"] == "investigation_experiment"
    assert h.body["run_id"] == run["id"]

    final = _wait_for_job(run["id"], h.body["job_id"], {"completed", "failed", "cancelled"})
    assert final["state"] == "completed"
    result = final["result"]
    assert result["phase"] == "completed"
    assert result["analysis"]["instrument_sane"] is True
    assert "effect_specific" not in result["analysis"]
    assert result["causal_claim"]["licensed"] is False
    assert "uncontrolled" in result["causal_claim"]["statement"]
    assert result["observed"]["treatment_reply_differs_from_baseline"] is True


def test_execute_generation_failure_marks_the_job_failed_but_the_plan_survives(stores):
    run = _span_addressable_run()
    address_id = _address_id_for(run, 1)

    def chat_fn(messages, sample):
        if _user_content(messages) == "":
            raise RuntimeError("boom")
        return "PWNED"

    sub = FakeSub(chat_fn)
    claimed, h = _post(sub, run["id"], "",
                        {"intervention": {"kind": "remove_span", "span_address_id": address_id}})
    final = _wait_for_job(run["id"], h.body["job_id"], {"completed", "failed", "cancelled"})
    assert final["state"] == "failed"
    assert final["error"]["code"] == "investigation_experiment_generation_failed"
    assert final["result"]["phase"] == "failed"
    assert "plan" in final["result"]


def test_execute_job_capacity_full_is_429(stores, monkeypatch):
    run = _span_addressable_run()
    address_id = _address_id_for(run, 1)
    from clozn.server.influence_jobs import JobCapacityError

    def boom(*_args, **_kwargs):
        raise JobCapacityError("job capacity is full")

    monkeypatch.setattr(JOBS, "start", boom)
    sub = FakeSub(lambda m, s: "PWNED")
    claimed, h = _post(sub, run["id"], "",
                        {"intervention": {"kind": "remove_span", "span_address_id": address_id}})
    assert claimed is True and h.status == 429


# ========================================================================================================= jobs
def test_job_status_not_found_is_404(stores):
    run = _organic_run()
    claimed, h = _get(run["id"], "/jobs/nope")
    assert claimed is True and h.status == 404


def test_job_cancel_not_found_is_404(stores):
    run = _organic_run()
    claimed, h = _post(None, run["id"], "/jobs/nope/cancel", {})
    assert claimed is True and h.status == 404


def test_job_status_and_cancel_round_trip(stores):
    run = _span_addressable_run()
    address_id = _address_id_for(run, 1)
    sub = FakeSub(lambda m, s: "SAFE")
    claimed, h = _post(sub, run["id"], "",
                        {"intervention": {"kind": "remove_span", "span_address_id": address_id}})
    job_id = h.body["job_id"]
    _wait_for_job(run["id"], job_id, {"completed", "failed", "cancelled"})

    claimed2, h2 = _get(run["id"], f"/jobs/{job_id}")
    assert claimed2 is True and h2.status == 200
    assert h2.body["job_id"] == job_id

    claimed3, h3 = _post(None, run["id"], f"/jobs/{job_id}/cancel", {})
    assert claimed3 is True and h3.status == 200
    assert h3.body["cancel_accepted"] is False   # already terminal -- cancel after completion is a no-op
