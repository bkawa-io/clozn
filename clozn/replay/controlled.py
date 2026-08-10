"""Bounded controlled swaps for run comparison and regression triage.

This module is deliberately separate from :mod:`clozn.analysis.run_diff`:
the latter remains a pure, model-free comparison engine, while this module
owns the expensive intervention boundary.  A result is a
``clozn.run-change-test.v1`` artifact whose child run ids are evidence, not a
generated explanation.

The executor accepts a tiny runner protocol, so its budget accounting,
matching rules and causal classification are model-free testable.  The
product route supplies :class:`SubstrateReplayRunner`; tests may supply a
canned runner, but no code path invents a child run id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import math
import time
from typing import Callable

from clozn import schemas
from clozn.analysis import run_diff

SCHEMA_VERSION = "clozn.run-change-test.v1"
SUPPORTED_TESTS = ("context", "template", "sampling")
SUPPORTED_MATCHERS = ("exact_output", "tool_parse", "finish_reason", "token_budget")
DEFAULT_MAX_RUNS = 4
DEFAULT_MAX_SECONDS = 120.0


class ControlledTestError(ValueError):
    """Invalid request/runner contract, before a model run starts."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _tool_parse(run: dict):
    return _dict(_dict(run.get("output_contract")).get("outcome")).get("status")


def _generated_tokens(run: dict):
    receipt = _dict(run.get("context_receipt"))
    value = _dict(receipt.get("termination")).get("generated_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    value = _dict(receipt.get("limits")).get("generated_tokens")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    tokens = _dict(run.get("trace")).get("tokens")
    return len(tokens) if isinstance(tokens, list) else None


def _criterion_value(run: dict, criterion: str):
    if criterion == "exact_output":
        value = run.get("response")
        if not isinstance(value, str):
            return False, None
        return True, {
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "bytes": len(value.encode("utf-8")),
        }
    if criterion == "tool_parse":
        value = _tool_parse(run)
        return (True, value) if isinstance(value, str) and value else (False, None)
    if criterion == "finish_reason":
        value = run.get("finish_reason")
        return (True, value) if isinstance(value, str) and value else (False, None)
    if criterion == "token_budget":
        value = _generated_tokens(run)
        finish = run.get("finish_reason")
        if value is None:
            return False, None
        # Exact measured tuple, not an inferred quality/semantic score.
        return True, {"generated_tokens": value, "finish_reason": finish}
    raise ControlledTestError(
        f"unknown match criterion {criterion!r}; know: {list(SUPPORTED_MATCHERS)}"
    )


def match_run(run: dict, target: dict, criterion: str) -> dict:
    """Compare one arm to a recorded target using an explicit measurable rule."""
    have_run, actual = _criterion_value(run, criterion)
    have_target, expected = _criterion_value(target, criterion)
    if not have_run or not have_target:
        return {
            "available": False,
            "reason": f"{criterion} evidence is unavailable on "
                      f"{'the child run' if not have_run else 'the target run'}",
        }
    return {"available": True, "matched": actual == expected, "actual": actual, "expected": expected}


def recorded_sampling_config(run: dict):
    """Return the behavior-bearing sampler recorded for ``run``.

    ``False`` means the run was recorded greedily, a mapping means the complete sampled
    configuration is recoverable, and ``None`` means the historical sampler provenance is
    incomplete.  This is deliberately the single reader used by replay/checkpoint and the
    sampler-sensitivity planner; callers must not fill in current server defaults.
    """
    meta = _dict(run.get("meta"))
    decode = _dict(meta.get("decode"))
    source = {**meta, **decode}
    mode = source.get("mode") or source.get("sampler_mode") or source.get("sampling")
    temperature = source.get("temperature")
    if mode == "greedy" or temperature == 0 or temperature == 0.0:
        return False
    if not (mode in ("sample", "sampling") or isinstance(temperature, (int, float))):
        return None
    names = {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "repeat_penalty": "repeat_penalty",
        "seed": "seed",
    }
    values = {}
    for output, key in names.items():
        value = source.get(key)
        if output == "repeat_penalty" and value is None:
            value = source.get("repetition_penalty")
        if value is None or isinstance(value, bool):
            return None
        if output in ("top_k", "seed"):
            if not isinstance(value, int):
                return None
        elif not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            return None
        values[output] = value
    return values


# Backward-compatible private spelling for existing callers.  New code should use the public helper
# above so there is one definition of recorded sampler provenance.
_sampling_config = recorded_sampling_config


def _fixed_sampling(run: dict) -> bool:
    config = _sampling_config(run)
    return config is False or (isinstance(config, dict) and isinstance(config.get("seed"), int))


def _candidate_dials(run: dict) -> dict:
    value = _dict(run.get("behavior")).get("active_dials")
    return dict(value) if isinstance(value, dict) else {}


def _model_identity(run: dict):
    """Return the strongest recorded model identity available for a run.

    Controlled swaps hold the candidate model fixed.  If both source runs carry a model identity and
    those identities differ, running a context/template/sampling arm would confound the requested swap
    with a model change, so the arm must be reported unavailable rather than compared as if it were a
    causal test.
    """
    identity = _dict(run.get("identity"))
    value = identity.get("model_sha256") or run.get("model")
    return str(value) if value else None


def _max_tokens(run: dict) -> int:
    value = _dict(run.get("meta")).get("max_tokens")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 256


def _available(test: str, baseline: dict, candidate: dict, diff: dict) -> tuple[bool, str | None]:
    baseline_model, candidate_model = _model_identity(baseline), _model_identity(candidate)
    if baseline_model and candidate_model and baseline_model != candidate_model:
        return False, (
            "controlled swaps require one unchanged model identity; the reference and candidate "
            "runs use different models"
        )
    by_dim = {
        d.get("dimension"): d for d in diff.get("differences") or []
        if isinstance(d, dict) and d.get("dimension")
    }
    if test == "context":
        if not isinstance(baseline.get("messages"), list) or not isinstance(candidate.get("messages"), list):
            return False, "exact delivered messages are not retained on both runs"
        if baseline.get("messages") == candidate.get("messages"):
            return False, "delivered context is identical; there is nothing to swap"
        if _sampling_config(candidate) is None:
            return False, "candidate sampling configuration is not exactly recoverable"
        return True, None
    if test == "template":
        changed = "identity.template_fingerprint" in by_dim
        if not changed:
            return False, "template fingerprint did not change"
        if (run_diff._template_material(baseline) is None
                or run_diff._template_material(candidate) is None):
            return False, (
                "exact template material was not captured for both runs; "
                "a fingerprint cannot be executed"
            )
        return True, None
    if test == "sampling":
        sampling_a = _sampling_config(baseline)
        sampling_b = _sampling_config(candidate)
        if sampling_a is None or sampling_b is None:
            return False, (
                "one or both sampling configurations are not exactly recoverable "
                "(sampled runs require a recorded fixed seed)"
            )
        if sampling_a == sampling_b:
            return False, "exact sampling configuration did not change"
        if not isinstance(candidate.get("messages"), list):
            return False, "candidate delivered messages are not retained"
        return True, None
    raise ControlledTestError(f"unknown controlled test {test!r}")


def _validate_limits(max_runs, max_seconds) -> tuple[int, float]:
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 0:
        raise ControlledTestError("max_runs must be a non-negative integer")
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
            or not math.isfinite(float(max_seconds)) or float(max_seconds) < 0):
        raise ControlledTestError("max_seconds must be a finite non-negative number")
    return max_runs, float(max_seconds)


@dataclass
class ExecutionBudget:
    max_runs: int
    max_seconds: float
    clock: Callable[[], float] = time.monotonic
    started: float = field(init=False)
    runs_used: int = 0
    stop_reason: str | None = None

    def __post_init__(self):
        self.max_runs, self.max_seconds = _validate_limits(self.max_runs, self.max_seconds)
        self.started = self.clock()

    def elapsed(self) -> float:
        return max(0.0, self.clock() - self.started)

    def remaining_seconds(self) -> float:
        return max(0.0, self.max_seconds - self.elapsed())

    def can_start(self, runs_needed: int = 1) -> bool:
        if self.runs_used + runs_needed > self.max_runs:
            self.stop_reason = "budget_exhausted"
            return False
        if self.remaining_seconds() <= 0:
            self.stop_reason = "budget_exhausted"
            return False
        return True

    def start_run(self) -> float:
        if not self.can_start(1):
            raise ControlledTestError("execution budget exhausted")
        self.runs_used += 1
        return self.remaining_seconds()

    def snapshot(self) -> dict:
        doc = {
            "max_runs": self.max_runs,
            "max_seconds": self.max_seconds,
            "runs_used": self.runs_used,
            "duration_ms": int(round(self.elapsed() * 1000)),
            "remaining_runs": max(0, self.max_runs - self.runs_used),
            "remaining_seconds": round(self.remaining_seconds(), 6),
        }
        if self.stop_reason:
            doc["stop_reason"] = self.stop_reason
        return doc


def plan_change_tests(baseline: dict, candidate: dict, *, tests=None,
                      max_runs: int = DEFAULT_MAX_RUNS,
                      max_seconds: float = DEFAULT_MAX_SECONDS,
                      match_criterion: str = "exact_output",
                      clock: Callable[[], float] = time.monotonic) -> dict:
    """Return a schema-valid, zero-run plan artifact."""
    return execute_change_tests(
        baseline, candidate, runner=None, tests=tests, max_runs=max_runs,
        max_seconds=max_seconds, match_criterion=match_criterion,
        dry_run=True, clock=clock,
    )


def _planned_test(kind: str, available: bool, reason: str | None, budget: dict) -> dict:
    return {
        "kind": kind,
        "status": "not_run",
        "ran": False,
        "runs_used": 0,
        "duration_ms": 0,
        "budget": dict(budget),
        "evidence": [],
        "arms": [],
        "stop_reason": "planned" if available else "unavailable",
        "reason": (
            "planned only; no model run was started"
            if available else str(reason or "controlled test unavailable")
        ),
    }


def _runner_qualification(runner, kind: str, baseline: dict, candidate: dict,
                          match_criterion: str) -> dict:
    supported = getattr(runner, "supported_matchers", None)
    if supported is not None and match_criterion not in supported:
        return {
            "ok": False,
            "reason": (
                f"live runner cannot persist {match_criterion} evidence on replay children; "
                f"supported: {sorted(supported)}"
            ),
        }
    fn = getattr(runner, "qualify", None)
    if not callable(fn):
        return {"ok": True, "basis": "runner supplied no additional live identity qualification"}
    value = fn(kind, baseline, candidate)
    return value if isinstance(value, dict) else {"ok": False, "reason": "runner qualification failed"}


def _run_arm(runner, budget: ExecutionBudget, *, kind: str, arm: str,
             baseline: dict, candidate: dict, target: dict,
             match_criterion: str) -> dict:
    timeout_seconds = budget.start_run()
    started = budget.clock()
    try:
        result = runner.run_arm(
            kind, arm, baseline, candidate, timeout_seconds=timeout_seconds
        )
    except Exception as exc:  # one failed arm is explicit evidence, never a vanished test
        result = {"error": f"{type(exc).__name__}: {exc}"}
    duration_ms = int(round(max(0.0, budget.clock() - started) * 1000))
    child = result.get("run") if isinstance(result, dict) else None
    out = {
        "name": arm,
        "status": "error",
        "duration_ms": duration_ms,
        "match_target": "baseline" if target is baseline else "candidate",
    }
    if isinstance(child, dict) and child.get("id"):
        out["status"] = "completed"
        out["run_id"] = child["id"]
        out["match"] = match_run(child, target, match_criterion)
        if arm == "treatment":
            out["match_candidate"] = match_run(child, candidate, match_criterion)
    else:
        out["error"] = (
            result.get("error") if isinstance(result, dict) and result.get("error")
            else "runner returned no persisted child run"
        )
    if budget.remaining_seconds() <= 0:
        budget.stop_reason = "budget_exhausted"
        cancel = getattr(runner, "cancel_current", None)
        if callable(cancel):
            try:
                out["cancellation"] = cancel()
            except Exception as exc:
                out["cancellation"] = {
                    "requested": True, "terminated": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
    return out


def _classify_test(kind: str, arms: list[dict], baseline: dict, candidate: dict,
                   criterion: str) -> tuple[str, str]:
    if len(arms) != 2 or any(arm.get("status") != "completed" for arm in arms):
        return "inconclusive", "one or both controlled arms failed to produce a persisted child run"
    control, treatment = arms
    control_match, treatment_match = control.get("match") or {}, treatment.get("match") or {}
    if not control_match.get("available") or not treatment_match.get("available"):
        return "inconclusive", "the selected match criterion was unavailable on an arm or target"
    target_a = _criterion_value(baseline, criterion)
    target_b = _criterion_value(candidate, criterion)
    if target_a[0] and target_b[0] and target_a[1] == target_b[1]:
        return "inconclusive", (
            "baseline and candidate are identical under the selected match criterion; "
            "there is no measured regression for this test to recover"
        )
    if not control_match.get("matched"):
        return "inconclusive", (
            "the control arm did not reproduce the candidate under the selected exact criterion"
        )
    if treatment_match.get("matched"):
        if kind == "sampling" and not (_fixed_sampling(baseline) and _fixed_sampling(candidate)):
            return "reproduced", (
                "the sampling swap recovered the baseline, but an unfixed stochastic regime means "
                "this may only remove sampling variance; no parameter-level causal claim is made"
            )
        return "causally_supported", (
            f"control reproduced the candidate and the {kind} treatment recovered the baseline "
            f"under {criterion}, with the runner-qualified identity held constant"
        )
    # A treatment that still matches candidate is a real negative intervention.
    treatment_to_candidate = arms[1].get("match_candidate")
    if isinstance(treatment_to_candidate, dict) and treatment_to_candidate.get("matched"):
        return "eliminated", f"the {kind} swap did not move the selected outcome away from the candidate"
    return "observed", (
        f"the {kind} swap changed the selected outcome but did not recover the baseline; "
        "directional movement is not causal attribution"
    )


def execute_change_tests(baseline: dict, candidate: dict, *, runner, tests=None,
                         max_runs: int = DEFAULT_MAX_RUNS,
                         max_seconds: float = DEFAULT_MAX_SECONDS,
                         match_criterion: str = "exact_output",
                         dry_run: bool = False,
                         clock: Callable[[], float] = time.monotonic,
                         validate: bool = True) -> dict:
    """Plan or execute controlled swaps with strict run/deadline accounting."""
    if not isinstance(baseline, dict) or not baseline.get("id"):
        raise ControlledTestError("baseline run is missing an id")
    if not isinstance(candidate, dict) or not candidate.get("id"):
        raise ControlledTestError("candidate run is missing an id")
    if match_criterion not in SUPPORTED_MATCHERS:
        raise ControlledTestError(
            f"unknown match criterion {match_criterion!r}; know: {list(SUPPORTED_MATCHERS)}"
        )
    requested = list(tests or SUPPORTED_TESTS)
    unknown = sorted(set(requested) - set(SUPPORTED_TESTS))
    if unknown:
        raise ControlledTestError(f"unknown controlled test(s): {unknown}")
    if len(set(requested)) != len(requested):
        raise ControlledTestError("controlled test names must be unique")
    budget = ExecutionBudget(max_runs, max_seconds, clock=clock)
    diff = run_diff.compare_runs(baseline, candidate)
    if not diff.get("ok"):
        raise ControlledTestError(diff.get("error") or "run comparison failed")

    test_docs = []
    criterion_a = _criterion_value(baseline, match_criterion)
    criterion_b = _criterion_value(candidate, match_criterion)
    criterion_reason = None
    if not criterion_a[0] or not criterion_b[0]:
        criterion_reason = (
            f"{match_criterion} evidence is not recorded on both source runs; "
            "a recovery cannot be measured"
        )
    elif criterion_a[1] == criterion_b[1]:
        criterion_reason = (
            "baseline and candidate are identical under the selected match criterion; "
            "there is no measured regression to recover"
        )
    for kind in requested:
        available, reason = _available(kind, baseline, candidate, diff)
        if criterion_reason:
            available, reason = False, criterion_reason
        before = budget.snapshot()
        if dry_run or not available:
            test_docs.append(_planned_test(kind, available, reason, before))
            continue
        if runner is None:
            raise ControlledTestError("execution requested without a controlled-test runner")
        qualification = _runner_qualification(
            runner, kind, baseline, candidate, match_criterion)
        if not qualification.get("ok"):
            test_docs.append({
                **_planned_test(kind, False, qualification.get("reason") or
                                "live candidate identity could not be qualified", before),
                "stop_reason": "identity_unqualified",
                "qualification": qualification,
            })
            continue
        # Never start half of a two-arm test merely because one run remains.
        if not budget.can_start(2):
            test_docs.append({
                **_planned_test(kind, False, "budget_exhausted", budget.snapshot()),
                "stop_reason": "budget_exhausted",
                "reason": "budget_exhausted before both required arms could start",
            })
            continue

        test_started = budget.clock()
        runs_before = budget.runs_used
        control = _run_arm(
            runner, budget, kind=kind, arm="control", baseline=baseline,
            candidate=candidate, target=candidate, match_criterion=match_criterion,
        )
        arms = [control]
        if budget.can_start(1):
            treatment = _run_arm(
                runner, budget, kind=kind, arm="treatment", baseline=baseline,
                candidate=candidate, target=baseline, match_criterion=match_criterion,
            )
            arms.append(treatment)

        status, status_reason = _classify_test(
            kind, arms, baseline, candidate, match_criterion
        )
        evidence = [
            {"run_id": arm["run_id"], "arm": arm["name"]}
            for arm in arms if arm.get("run_id")
        ]
        test_doc = {
            "kind": kind,
            "status": status,
            "ran": True,
            "runs_used": budget.runs_used - runs_before,
            "duration_ms": int(round(max(0.0, budget.clock() - test_started) * 1000)),
            "budget": budget.snapshot(),
            "evidence": evidence,
            "arms": arms,
            "qualification": qualification,
            "reason": status_reason,
        }
        if budget.stop_reason:
            test_doc["stop_reason"] = budget.stop_reason
        test_docs.append(test_doc)

    causal = [t["kind"] for t in test_docs if t.get("status") == "causally_supported"]
    if len(causal) == 1:
        classification = causal[0]
    elif len(causal) > 1:
        classification = "entangled"
    else:
        classification = "undetermined"
    final_budget = budget.snapshot()
    if dry_run:
        overall_status = "planned"
    elif budget.stop_reason == "budget_exhausted":
        overall_status = "budget_exhausted"
    elif any(t.get("ran") for t in test_docs):
        overall_status = "completed"
    else:
        overall_status = "inconclusive"
    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "run_a": baseline["id"],
        "run_b": candidate["id"],
        "status": overall_status,
        "dry_run": bool(dry_run),
        "match_criterion": {
            "kind": match_criterion,
            "note": "exact recorded evidence only; semantic similarity is never used",
        },
        "budget": final_budget,
        "tests": test_docs,
        "summary": {
            "classification": classification,
            "causally_supported": causal,
            "entangled": len(causal) > 1,
        },
    }
    if validate:
        schemas.validate(document, SCHEMA_VERSION)
    return document


class SubstrateReplayRunner:
    """Product live-runner backed by the existing replay primitive."""

    supported_matchers = frozenset({"exact_output", "finish_reason", "token_budget"})

    def __init__(self, substrate):
        self.substrate = substrate
        self.last_run: dict | None = None

    def qualify(self, kind: str, _baseline: dict, candidate: dict) -> dict:
        wanted = _dict(candidate.get("identity"))
        try:
            actual = self.substrate.identity_meta() if hasattr(self.substrate, "identity_meta") else {}
        except Exception as exc:
            return {"ok": False, "reason": f"live identity lookup failed: {type(exc).__name__}: {exc}"}
        actual = _dict(actual)
        checked = {}
        for identity_field in ("model_sha256", "template_fingerprint"):
            if not wanted.get(identity_field) or not actual.get(identity_field):
                return {
                    "ok": False,
                    "reason": (
                        f"{identity_field} is unavailable on the candidate or live substrate"
                    ),
                }
            checked[identity_field] = {
                "candidate": wanted[identity_field],
                "live": actual[identity_field],
            }
            if wanted[identity_field] != actual[identity_field]:
                return {
                    "ok": False,
                    "reason": f"live {identity_field} does not match candidate",
                    "checked": checked,
                }
        for identity_field in ("engine_build",):
            if wanted.get(identity_field):
                checked[identity_field] = {
                    "candidate": wanted[identity_field],
                    "live": actual.get(identity_field),
                }
                if wanted[identity_field] != actual.get(identity_field):
                    return {"ok": False, "reason": f"live {identity_field} does not match candidate",
                            "checked": checked}
        wanted_adapter = _dict(_dict(wanted.get("ext")).get("adapter"))
        actual_adapter = _dict(_dict(actual.get("ext")).get("adapter"))
        if wanted_adapter:
            checked["identity.ext.adapter"] = {
                "candidate": wanted_adapter, "live": actual_adapter,
            }
            if wanted_adapter != actual_adapter:
                return {"ok": False, "reason": "live adapter identity does not match candidate",
                        "checked": checked}
        if kind == "template" and not hasattr(self.substrate, "replay_with_template"):
            return {"ok": False, "reason": "live substrate has no exact-template replay seam",
                    "checked": checked}
        return {"ok": True, "checked": checked}

    def _sampling_for(self, run: dict):
        value = _sampling_config(run)
        if value is None:
            raise ControlledTestError("sampling configuration is not exactly recoverable")
        return value

    def run_arm(self, kind: str, arm: str, baseline: dict, candidate: dict, *,
                timeout_seconds: float) -> dict:
        from clozn.replay import replay as replay_run

        if arm not in ("control", "treatment"):
            return {"error": f"unknown arm {arm!r}"}
        messages = candidate.get("messages")
        sampling = self._sampling_for(candidate)
        if kind == "context" and arm == "treatment":
            messages = baseline.get("messages")
        elif kind == "sampling" and arm == "treatment":
            sampling = self._sampling_for(baseline)
        elif kind == "template":
            fn = getattr(self.substrate, "replay_with_template", None)
            if not callable(fn):
                return {"error": "live substrate has no exact-template replay seam"}
            result = fn(
                candidate, run_diff._template_material(baseline) if arm == "treatment"
                else run_diff._template_material(candidate),
                timeout_seconds=timeout_seconds,
            )
            self.last_run = result if isinstance(result, dict) else None
            return {"run": self.last_run} if self.last_run else {"error": "template replay failed"}

        changes = {
            "behavior_off": True,
            "behavior_overrides": _candidate_dials(candidate),
            "controlled_test": {"kind": kind, "arm": arm},
        }
        engine = getattr(self.substrate, "engine", None)
        prior_timeout = getattr(engine, "timeout", None) if engine is not None else None
        if engine is not None and isinstance(prior_timeout, (int, float)):
            engine.timeout = max(0.05, min(float(prior_timeout), float(timeout_seconds)))
        try:
            child = replay_run(
                candidate, changes, self.substrate,
                messages_override=messages,
                sampling_override=sampling,
                max_new=_max_tokens(candidate),
            )
        finally:
            if engine is not None and isinstance(prior_timeout, (int, float)):
                engine.timeout = prior_timeout
        self.last_run = child if isinstance(child, dict) else None
        return {"run": self.last_run} if self.last_run else {"error": "replay failed"}

    def cancel_current(self) -> dict:
        req = getattr(self.substrate, "_request", None)
        requested = False
        if req is not None and hasattr(req, "cancel"):
            req.cancel()
            requested = True
        engine = getattr(self.substrate, "engine", None)
        worker_req = getattr(req, "engine_req", None) if req is not None else None
        terminated = False
        if worker_req and engine is not None and hasattr(engine, "cancel"):
            try:
                result = engine.cancel(worker_req)
                terminated = bool((result or {}).get("cancelled", True))
            except Exception:
                terminated = False
        return {
            "requested": requested,
            "terminated": terminated,
            "reason": (
                "worker cancellation acknowledged" if terminated
                else "local cancellation requested; worker termination was not independently acknowledged"
            ),
        }
