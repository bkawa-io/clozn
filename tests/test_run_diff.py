"""test_run_diff.py -- model-free unit tests for analysis/run_diff.py (agent roadmap feature 10,
"What changed"). Synthetic run records only, in the same shape clozn/analysis/test_model_diff.py's own
`_run()` fixture uses: no engine, no network, no live server, no GPU.

Covers the spec's own test list (notes/agent_roadmap/10-run-change-explainer.md): model-only change,
template-only change, context omission with same model, sampling-only change, tool parser difference,
max-token termination, a privacy/missing-content-shaped comparison, and multiple simultaneous differences
staying separated -- plus this module's own forward-compatible identity["ext"] walk and the structural
rank/status orthogonality the roadmap task called load-bearing.
"""
from __future__ import annotations

import pytest

from clozn import schemas
from clozn.analysis import run_diff


# ============================================================================================ fixtures

def _run(rid, *, model_sha256=None, model_path=None, template_fingerprint=None, engine_build=None,
         clozn_version=None, ext=None, meta=None, messages=None, final_prompt=None, prompt_tokens=None,
         finish_reason=None, response="", output_contract=None, tokens=None):
    identity = {}
    if model_sha256 is not None:
        identity["model_sha256"] = model_sha256
    if model_path is not None:
        identity["model_path"] = model_path
    if template_fingerprint is not None:
        identity["template_fingerprint"] = template_fingerprint
    if engine_build is not None:
        identity["engine_build"] = engine_build
    if clozn_version is not None:
        identity["clozn_version"] = clozn_version
    if ext is not None:
        identity["ext"] = ext

    context_receipt = {}
    if messages is not None or final_prompt is not None or prompt_tokens is not None:
        context_receipt = {
            "delivered": {"messages": messages if messages is not None else []},
            "survived": {"final_prompt": final_prompt},
            "limits": {"prompt_tokens": prompt_tokens},
        }

    rec = {
        "id": rid,
        "identity": identity,
        "meta": dict(meta or {}),
        "messages": messages if messages is not None else [],
        "response": response,
        "finish_reason": finish_reason,
        "context_receipt": context_receipt,
        "output_contract": dict(output_contract) if output_contract is not None else {},
        "trace": {"tokens": list(tokens)} if tokens is not None else {},
    }
    return rec


def _validate(doc):
    schemas.validate(doc, "clozn.run-diff.v1")


def _dims(result):
    return {d["dimension"] for d in result["differences"]}


def _classes(result):
    return {f["classification"] for f in result["findings"]}


# ============================================================================= schema conformance (every case)

def test_a_full_comparison_validates_against_its_own_schema():
    a = _run("run_a", model_sha256="a" * 64, template_fingerprint="fp1", meta={"temperature": 0.2},
             messages=[{"role": "user", "content": "hi"}] * 3, response="hello there")
    b = _run("run_b", model_sha256="b" * 64, template_fingerprint="fp2", meta={"temperature": 0.9},
             messages=[{"role": "user", "content": "hi"}], response="hi", finish_reason="length")
    result = run_diff.compare_runs(a, b)
    assert result["ok"] is True
    _validate(result)


def test_two_identical_runs_produce_no_differences_and_still_validate():
    a = _run("run_a", model_sha256="a" * 64, response="same text")
    b = _run("run_b", model_sha256="a" * 64, response="same text")
    result = run_diff.compare_runs(a, b)
    assert result["differences"] == []
    assert result["findings"] == []
    _validate(result)


# =================================================================================== missing / malformed input

def test_missing_run_yields_an_honest_error_not_a_run_diff_document():
    result = run_diff.compare_runs(None, {"id": "run_b"})
    assert result["ok"] is False
    assert result["missing"] == ["a"]
    assert "schema_version" not in result       # an error shape is not asserted against the schema


def test_both_runs_missing_lists_both():
    result = run_diff.compare_runs({}, None)
    assert set(result["missing"]) == {"a", "b"}


# ========================================================================================= spec test: model-only

def test_model_only_change_is_observed_and_ranked_first():
    a = _run("run_a", model_sha256="a" * 64)
    b = _run("run_b", model_sha256="b" * 64)
    result = run_diff.compare_runs(a, b)
    assert "identity.model_sha256" in _dims(result)
    assert "model_changed" in _classes(result)
    finding = next(f for f in result["findings"] if f["classification"] == "model_changed")
    assert finding["status"] == "observed"
    assert finding["dimensions"] == ["identity.model_sha256"]
    diff = next(d for d in result["differences"] if d["dimension"] == "identity.model_sha256")
    assert diff["rank"] == 0        # first in _RANK_ORDER
    assert diff["kind"] == "changed"
    assert diff["value_a"] == "a" * 64 and diff["value_b"] == "b" * 64


# ======================================================================================= spec test: template-only

def test_template_only_change_is_observed():
    a = _run("run_a", model_sha256="same" * 16, template_fingerprint="fp_old")
    b = _run("run_b", model_sha256="same" * 16, template_fingerprint="fp_new")
    result = run_diff.compare_runs(a, b)
    assert "identity.model_sha256" not in _dims(result)   # same model -- no entry emitted at all
    assert "identity.template_fingerprint" in _dims(result)
    assert "template_changed" in _classes(result)
    assert "model_changed" not in _classes(result)


# =============================================================================== spec test: context omission

def test_context_omission_with_same_model_is_flagged():
    six = [{"role": "user", "content": f"turn {i}"} for i in range(6)]
    two = six[-2:]
    a = _run("run_a", model_sha256="m" * 64, messages=six)
    b = _run("run_b", model_sha256="m" * 64, messages=two)
    result = run_diff.compare_runs(a, b)
    diff = next(d for d in result["differences"] if d["dimension"] == "context.delivered.messages.count")
    assert diff["kind"] == "changed" and diff["value_a"] == 6 and diff["value_b"] == 2
    finding = next(f for f in result["findings"] if f["classification"] == "context_omission")
    assert finding["status"] == "observed"
    assert "4 fewer message(s)" in finding["summary"]


def test_context_growth_is_a_difference_but_not_an_omission_finding():
    a = _run("run_a", messages=[{"role": "user", "content": "hi"}])
    b = _run("run_b", messages=[{"role": "user", "content": "hi"}] * 5)
    result = run_diff.compare_runs(a, b)
    assert "context.delivered.messages.count" in _dims(result)
    assert "context_omission" not in _classes(result)


# ================================================================================= spec test: sampling-only

def test_sampling_only_change_is_observed():
    a = _run("run_a", meta={"temperature": 0.2, "top_p": 0.9, "seed": 7})
    b = _run("run_b", meta={"temperature": 0.9, "top_p": 0.9, "seed": 7})
    result = run_diff.compare_runs(a, b)
    assert _dims(result) == {"generation.temperature"}
    finding = next(f for f in result["findings"] if f["classification"] == "sampling_changed")
    assert finding["dimensions"] == ["generation.temperature"]
    assert finding["status"] == "observed"


def test_multiple_sampling_changes_are_reported_together_in_one_finding():
    a = _run("run_a", meta={"temperature": 0.2, "seed": 1, "max_tokens": 256})
    b = _run("run_b", meta={"temperature": 0.9, "seed": 2, "max_tokens": 256})
    result = run_diff.compare_runs(a, b)
    assert _dims(result) == {"generation.temperature", "generation.seed"}
    finding = next(f for f in result["findings"] if f["classification"] == "sampling_changed")
    assert sorted(finding["dimensions"]) == ["generation.seed", "generation.temperature"]


def test_sampling_key_absent_on_both_sides_is_never_reported():
    a = _run("run_a", meta={"temperature": 0.2})
    b = _run("run_b", meta={"temperature": 0.2})
    result = run_diff.compare_runs(a, b)
    assert _dims(result) == set()      # top_k/stop/seed/... are simply never mentioned


# ==================================================================================== spec test: tool parser

def test_tool_parse_failure_is_flagged():
    a = _run("run_a", output_contract={"schema": "clozn.output_contract.v2",
                                       "outcome": {"status": "parsed", "kind": "tool_call"}})
    b = _run("run_b", output_contract={"schema": "clozn.output_contract.v2",
                                       "outcome": {"status": "error", "code": "native_parse_failed"}})
    result = run_diff.compare_runs(a, b)
    diff = next(d for d in result["differences"] if d["dimension"] == "output.tool_call_status")
    assert diff["value_a"] == "parsed" and diff["value_b"] == "error"
    finding = next(f for f in result["findings"] if f["classification"] == "tool_parse_failed")
    assert finding["status"] == "observed"


def test_no_tool_call_on_either_side_is_not_applicable_not_unavailable():
    a = _run("run_a")
    b = _run("run_b")
    result = run_diff.compare_runs(a, b)
    assert "output.tool_call_status" not in _dims(result)


# ================================================================================ spec test: max-token termination

def test_max_token_termination_is_flagged():
    a = _run("run_a", finish_reason="stop")
    b = _run("run_b", finish_reason="length")
    result = run_diff.compare_runs(a, b)
    assert next(d for d in result["differences"] if d["dimension"] == "output.finish_reason")["value_b"] == "length"
    finding = next(f for f in result["findings"] if f["classification"] == "output_truncated")
    assert "run_b's output ended at the max-token limit" in finding["summary"]


def test_finish_reason_absent_on_both_sides_is_never_reported():
    a = _run("run_a")
    b = _run("run_b")
    result = run_diff.compare_runs(a, b)
    assert "output.finish_reason" not in _dims(result)


# ============================================================================ spec test: privacy / missing content

def test_privacy_restricted_comparison_labels_the_limitation_without_inventing_content():
    a = _run("run_a", messages=[{"role": "user", "content": "hi"}] * 4)
    b = {**_run("run_b"), "context_receipt": {}}     # nothing captured at all -- legacy/light-tier shaped
    result = run_diff.compare_runs(a, b)
    diff = next(d for d in result["differences"] if d["dimension"] == "context.delivered.messages.count")
    assert diff["kind"] == "unavailable"
    assert diff["value_a"] == 4
    assert "value_b" not in diff                 # the missing side is never invented as 0
    assert result["privacy_limited"] is True


def test_both_sides_missing_context_receipt_produces_no_entry_at_all():
    a = {**_run("run_a"), "context_receipt": {}}
    b = {**_run("run_b"), "context_receipt": {}}
    result = run_diff.compare_runs(a, b)
    assert "context.delivered.messages.count" not in _dims(result)
    assert result["privacy_limited"] is False


# ========================================================================= spec test: multiple simultaneous diffs

def test_multiple_simultaneous_differences_stay_separated_not_collapsed():
    a = _run("run_a", model_sha256="a" * 64, template_fingerprint="fp1", meta={"temperature": 0.2},
             messages=[{"role": "user", "content": "x"}] * 5, finish_reason="stop")
    b = _run("run_b", model_sha256="b" * 64, template_fingerprint="fp2", meta={"temperature": 0.9},
             messages=[{"role": "user", "content": "x"}], finish_reason="length")
    result = run_diff.compare_runs(a, b)
    classes = _classes(result)
    assert {"model_changed", "template_changed", "sampling_changed", "context_omission",
           "output_truncated"} <= classes
    # every finding is its own object with its own dimensions -- never merged into one summary
    assert len(result["findings"]) == len(classes)


# ==================================================================== forward-compatible identity["ext"] walk

def test_unknown_ext_namespace_added_removed_and_changed_all_surface():
    a = _run("run_a", ext={"totally_unknown_facet": {"x": 1}, "only_in_a": {"y": 1}})
    b = _run("run_b", ext={"totally_unknown_facet": {"x": 2}, "only_in_b": {"z": 1}})
    result = run_diff.compare_runs(a, b)
    dims = _dims(result)
    assert "identity.ext.totally_unknown_facet.x" in dims
    assert "identity.ext.only_in_a" in dims
    assert "identity.ext.only_in_b" in dims
    only_in_a = next(d for d in result["differences"] if d["dimension"] == "identity.ext.only_in_a")
    assert only_in_a["kind"] == "removed" and only_in_a["value_a"] == {"y": 1}
    only_in_b = next(d for d in result["differences"] if d["dimension"] == "identity.ext.only_in_b")
    assert only_in_b["kind"] == "added" and only_in_b["value_b"] == {"z": 1}


def test_unknown_ext_namespaces_never_feed_findings():
    a = _run("run_a", ext={"adapter": {"strength": 0.2}})
    b = _run("run_b", ext={"adapter": {"strength": 0.9}})
    result = run_diff.compare_runs(a, b)
    assert "identity.ext.adapter.strength" in _dims(result)
    assert result["findings"] == []     # no classification manufactured from a facet this differ can't read


def test_ext_namespace_equal_on_both_sides_produces_no_entry():
    a = _run("run_a", ext={"machine": {"gpu": "5080"}})
    b = _run("run_b", ext={"machine": {"gpu": "5080"}})
    result = run_diff.compare_runs(a, b)
    assert _dims(result) == set()


def test_ext_recursion_is_bounded_and_deep_values_are_compared_opaquely():
    deep_a = {"l1": {"l2": {"l3": {"l4": "a"}}}}
    deep_b = {"l1": {"l2": {"l3": {"l4": "b"}}}}
    a = _run("run_a", ext={"deep_facet": deep_a})
    b = _run("run_b", ext={"deep_facet": deep_b})
    result = run_diff.compare_runs(a, b)
    # depth cap means the differ stops decomposing at some point and reports an opaque "changed" node --
    # but it must still be present SOMEWHERE under identity.ext.deep_facet, never silently dropped.
    assert any(d["dimension"].startswith("identity.ext.deep_facet") for d in result["differences"])


def test_a_namespace_that_raises_during_comparison_is_isolated_not_fatal():
    class Uncomparable:
        def __eq__(self, other):
            raise RuntimeError("boom")
        def __hash__(self):
            return 0

    a = _run("run_a", model_sha256="m" * 64, ext={"broken": Uncomparable()})
    b = _run("run_b", model_sha256="m" * 64, ext={"broken": Uncomparable()})
    result = run_diff.compare_runs(a, b)
    assert result["ok"] is True          # the whole comparison survives
    broken = next(d for d in result["differences"] if d["dimension"] == "identity.ext.broken")
    assert broken["kind"] == "diff_failed"
    assert "RuntimeError" in broken["note"]


def test_no_ext_on_either_side_is_silent():
    a = _run("run_a")
    b = _run("run_b")
    result = run_diff.compare_runs(a, b)
    assert not any(d["dimension"].startswith("identity.ext") for d in result["differences"])


# ================================================================================== ranking / evidence structure

def test_rank_lives_only_on_differences_never_on_findings():
    a = _run("run_a", model_sha256="a" * 64, ext={"unknown": {"v": 1}})
    b = _run("run_b", model_sha256="b" * 64, ext={"unknown": {"v": 2}})
    result = run_diff.compare_runs(a, b)
    assert all("rank" in d for d in result["differences"])
    assert not any("rank" in f for f in result["findings"])
    model_rank = next(d for d in result["differences"] if d["dimension"] == "identity.model_sha256")["rank"]
    ext_rank = next(d for d in result["differences"] if d["dimension"].startswith("identity.ext"))["rank"]
    assert model_rank < ext_rank        # identity beats an unknown ext facet in presentation order


def test_findings_status_is_never_upgraded_by_rank():
    a = _run("run_a", model_sha256="a" * 64)
    b = _run("run_b", model_sha256="b" * 64)
    result = run_diff.compare_runs(a, b)
    assert all(f["status"] in {"observed", "eliminated", "reproduced", "correlated", "causally_supported"}
              for f in result["findings"])
    assert all(f["status"] != "causally_supported" for f in result["findings"])   # no replay ran


def test_ranking_block_documents_the_static_order():
    a = _run("run_a", model_sha256="a" * 64)
    b = _run("run_b", model_sha256="b" * 64)
    result = run_diff.compare_runs(a, b)
    assert result["ranking"]["order"][0] == "identity.model_sha256"
    assert result["ranking"]["order"][-1] == "identity.ext."
    assert "presentation-only" in result["ranking"]["note"]


# ============================================================================================= replay planner

def test_replay_planner_marks_available_swaps_from_the_diff():
    a = _run("run_a", template_fingerprint="fp1", meta={"temperature": 0.2},
             messages=[{"role": "user", "content": "hi"}] * 3)
    b = _run("run_b", template_fingerprint="fp2", meta={"temperature": 0.9},
             messages=[{"role": "user", "content": "hi"}])
    result = run_diff.compare_runs(a, b)
    plan = run_diff.plan_replay(a, b, result)
    by_swap = {c["swap"]: c for c in plan["candidates"]}
    assert by_swap["context"]["available"] is True
    assert by_swap["template"]["available"] is True
    assert by_swap["sampling"]["available"] is True
    assert plan["runs_required"] == 3
    assert "NOT performed" in plan["note"]


def test_replay_planner_marks_swaps_unavailable_when_nothing_differs():
    a = _run("run_a")
    b = _run("run_b")
    result = run_diff.compare_runs(a, b)
    plan = run_diff.plan_replay(a, b, result)
    assert all(not c["available"] for c in plan["candidates"])
    assert plan["runs_required"] == 0
    assert all(c["note"] for c in plan["candidates"])


def test_replay_planner_never_executes_anything():
    """No import of clozn.replay's execution primitive anywhere in this module -- this stays a pure
    planning function; execution is a deferred, separately-scoped slice (see the module docstring)."""
    import inspect
    assert "import clozn.replay" not in inspect.getsource(run_diff)
    assert "from clozn.replay" not in inspect.getsource(run_diff)
