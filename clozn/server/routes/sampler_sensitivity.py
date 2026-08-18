"""HTTP routes for the explicit, bounded Sampler Sensitivity Probe."""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True
_PLAN_SUFFIX = "/sampler-sensitivity/plan"
_EXECUTE_SUFFIX = "/sampler-sensitivity"


def _error(h, status: int, code: str, message: str):
    h._json(status, {"error": message, "code": code})
    return True


def _run_id(path: str, suffix: str) -> str:
    return path[len("/runs/"):-len(suffix)]


def _request(body):
    if not isinstance(body, Mapping):
        from clozn.replay.sampler_sensitivity import SamplerSensitivityInputError
        raise SamplerSensitivityInputError("invalid_body", "body must be an object")
    allowed = {"position", "recipe", "seed_probes"}
    if set(body) - allowed:
        from clozn.replay.sampler_sensitivity import SamplerSensitivityInputError
        raise SamplerSensitivityInputError("invalid_body", "sampler sensitivity has unknown fields")
    return {
        "position": body.get("position", 0),
        "recipe": body.get("recipe", "nearby_v1"),
        "seed_probes": body.get("seed_probes", 0),
    }


def try_post(h, p, body):
    is_plan = p.startswith("/runs/") and p.endswith(_PLAN_SUFFIX)
    is_execute = (
        p.startswith("/runs/") and p.endswith(_EXECUTE_SUFFIX)
        and not p.endswith(_PLAN_SUFFIX)
    )
    if not (is_plan or is_execute):
        return False

    import clozn.runs.store as runlog
    from clozn.replay.sampler_sensitivity import (
        SamplerSensitivityInputError,
        execute_sampler_sensitivity,
        plan_sampler_sensitivity,
    )

    suffix = _PLAN_SUFFIX if is_plan else _EXECUTE_SUFFIX
    parent = runlog.get_run(_run_id(p, suffix))
    if parent is None:
        return _error(h, 404, "run_not_found", "run not found")
    try:
        request = _request(body)
        plan = plan_sampler_sensitivity(parent, **request)
    except SamplerSensitivityInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except Exception:
        return _error(h, 500, "sampler_sensitivity_contract_invalid", "sampler sensitivity plan could not be composed")

    if is_plan:
        h._json(200, plan)
        return True

    if plan["execution"]["state"] != "ready":
        try:
            result = execute_sampler_sensitivity(parent, None, plan)
        except Exception:
            return _error(h, 500, "sampler_sensitivity_contract_invalid", "sampler sensitivity result could not be composed")
        h._json(422, result)
        return True

    # Parent-scoped routing: the request has no model/worker/runtime override.  The selected
    # substrate is the only live dependency and is resolved through the neutral run-scoped helper.
    from clozn.server.model_routing import select_run_model_facts

    facts = select_run_model_facts(
        h, parent, route="/runs/<id>/sampler-sensitivity")
    if facts is None:
        return True
    runtime_identity, worker_identity, engine, sub = facts
    if engine is None or runtime_identity is None or worker_identity is None:
        return _error(
            h, 503, "sampler_sensitivity_worker_unavailable",
            "Sampler Sensitivity requires an identity-qualified parent model worker",
        )

    try:
        result = execute_sampler_sensitivity(
            parent,
            sub,
            plan,
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            reload_parent=runlog.get_run,
            cancel_check=getattr(h, "_execution_fork_cancelled", None),
        )
    except SamplerSensitivityInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except Exception:
        return _error(h, 500, "sampler_sensitivity_contract_invalid", "sampler sensitivity result could not be composed")

    summary = result.get("summary") or {}
    if summary.get("children_created", 0) > 0:
        status = 201
    elif result.get("execution", {}).get("state") == "cancelled":
        status = 409
    else:
        status = 422
    h._json(status, result)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_post"]
