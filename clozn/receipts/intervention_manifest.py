"""Server-side validator + replay executor for clozn-client's ``clozn.intervention_manifest.v1`` JSON
shape (docs/PRODUCT_ROADMAP.md §7 item 2, roadmap Phase 4.2).

This module deliberately DUPLICATES a minimal subset of clozn-client's own validation
(``clozn_client.manifests.InterventionManifest`` / ``clozn_client.models.AttentionKnockout``) rather
than importing the pip package: the gateway must not depend on the researcher-facing client, and the
manifest is untrusted wire input regardless of which side produced it. Every constraint below mirrors
clozn-client 1:1 -- field names, ranges, uniqueness rules, and the exact canonical-JSON shape used for
``sha256`` -- so a manifest built with the client library, submitted here, reproduces the SAME
``manifest_sha256`` a client-side ``.sha256`` would compute. A cross-check test
(``tests/test_intervention_manifest.py``) builds a manifest with the real client library and asserts
the two hashes match byte-for-byte.

Execution is duck-typed against an ``engine`` object exposing ``.score(prompt=, prompt_ids=,
continuation=, continuation_ids=, topk=, steer=, steer_vec=, attn_knockout=) -> dict`` -- the exact
shape ``engine.client.clozn_engine.EngineClient.score`` now exposes (extended by this same roadmap item
to forward ``attn_knockout``; see that module's docstring). This is NEVER generation and NEVER
sampling beyond what the manifest's own ``steer``/``steer_vec`` arms ask for -- every arm is one
teacher-forced ``/score`` call, matching ``clozn.experiments.stats``'s own definition of a
forced-rescore. Replay-class labels reuse ``clozn.experiments.stats.REPLAY_CLASSES`` /
``replay_class_for_meta`` verbatim rather than inventing a parallel vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import math
import time

from clozn.experiments.stats import replay_class_for_meta

SCHEMA = "clozn.intervention_manifest.v1"
_NAME_MAX = 128


class ManifestError(ValueError):
    """A structural problem with a submitted manifest -- never a capability gap (see
    ``replay_manifest``'s ``performed: False`` capability-refusal shape for that)."""


# --------------------------------------------------------------------------------------- validation

def _require_object(value, label: str) -> dict:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    return value


def _require_name(value, label: str) -> str:
    if not isinstance(value, str):
        raise ManifestError(f"{label} must be a string")
    text = value.strip()
    if not text or len(text) > _NAME_MAX or any(ord(ch) < 32 for ch in text):
        raise ManifestError(f"{label} must be a non-empty printable string up to {_NAME_MAX} characters")
    return text


def _require_ids(value, label: str) -> list[int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, list):
        raise ManifestError(f"{label} must be an integer array")
    out = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ManifestError(f"{label} must contain non-negative integers")
        out.append(item)
    if not out:
        raise ManifestError(f"{label} must not be empty")
    return out


def _require_finite_floats(value, label: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, list):
        raise ManifestError(f"{label} must be a numeric array")
    out = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ManifestError(f"{label} must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ManifestError(f"{label} must contain finite numbers")
        out.append(number)
    if not out:
        raise ManifestError(f"{label} must not be empty")
    return out


def _validate_request(value) -> dict:
    obj = _require_object(value, "manifest.request")
    has_prompt, has_prompt_ids = "prompt" in obj, "prompt_ids" in obj
    if has_prompt == has_prompt_ids:
        raise ManifestError("manifest.request must provide exactly one of prompt or prompt_ids")
    has_cont, has_cont_ids = "continuation" in obj, "continuation_ids" in obj
    if has_cont == has_cont_ids:
        raise ManifestError("manifest.request must provide exactly one of continuation or continuation_ids")
    if has_prompt and not isinstance(obj["prompt"], str):
        raise ManifestError("manifest.request.prompt must be a string")
    if has_cont and not isinstance(obj["continuation"], str):
        raise ManifestError("manifest.request.continuation must be a string")
    prompt_ids = _require_ids(obj["prompt_ids"], "manifest.request.prompt_ids") if has_prompt_ids else None
    continuation_ids = (
        _require_ids(obj["continuation_ids"], "manifest.request.continuation_ids") if has_cont_ids else None
    )
    topk = obj.get("topk", 0)
    if not isinstance(topk, int) or isinstance(topk, bool) or topk < 0:
        raise ManifestError("manifest.request.topk must be a non-negative integer")
    return {
        "prompt": obj.get("prompt"), "prompt_ids": prompt_ids,
        "continuation": obj.get("continuation"), "continuation_ids": continuation_ids,
        "topk": topk,
    }


def _validate_knockout(value, label: str) -> dict:
    # Mirrors clozn_client.models.AttentionKnockout exactly -- note there is deliberately NO "head"
    # field: the client's v1 wire form never sends one (see hook_vocabulary's client_head_gap), so a
    # validated knockout here always targets every head at the layer, same as a client-built one.
    obj = _require_object(value, label)
    layer = obj.get("layer")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
        raise ManifestError(f"{label}.layer must be a non-negative integer")
    queries = _require_ids(obj.get("queries"), f"{label}.queries")
    keys = _require_ids(obj.get("keys"), f"{label}.keys")
    renormalize = obj.get("renormalize", True)
    if not isinstance(renormalize, bool):
        raise ManifestError(f"{label}.renormalize must be a bool")
    return {"layer": layer, "queries": sorted(set(queries)), "keys": sorted(set(keys)),
            "renormalize": renormalize}


def _validate_arm(value, index: int) -> dict:
    obj = _require_object(value, f"manifest.arms[{index}]")
    name = _require_name(obj.get("name"), f"manifest.arms[{index}].name")
    knockout_value = obj.get("attention_knockout", [])
    if isinstance(knockout_value, (str, bytes, bytearray)) or not isinstance(knockout_value, list):
        raise ManifestError(f"manifest.arms[{index}].attention_knockout must be an array")
    knockouts = [
        _validate_knockout(item, f"manifest.arms[{index}].attention_knockout[{k}]")
        for k, item in enumerate(knockout_value)
    ]
    steer = obj.get("steer")
    if steer is not None:
        steer = _require_object(steer, f"manifest.arms[{index}].steer")
        if not steer:
            raise ManifestError(f"manifest.arms[{index}].steer must be a non-empty object")
    steer_vec = (
        _require_finite_floats(obj["steer_vec"], f"manifest.arms[{index}].steer_vec")
        if obj.get("steer_vec") is not None else None
    )
    metadata = _require_object(obj.get("metadata", {}), f"manifest.arms[{index}].metadata")
    if not knockouts and steer is None and steer_vec is None:
        raise ManifestError(f"manifest.arms[{index}] must define a knockout, steer, or steer_vec")
    return {"name": name, "attention_knockout": knockouts, "steer": steer, "steer_vec": steer_vec,
            "metadata": metadata}


def validate_manifest(value: dict) -> dict:
    """Validate an untrusted ``clozn.intervention_manifest.v1`` JSON object; raise ``ManifestError`` on
    any structural problem. Mirrors ``clozn_client.manifests.InterventionManifest.from_json``'s rules
    exactly (name/request/arm shapes, XOR request fields, arm-name uniqueness, "at least one of
    knockout/steer/steer_vec"), vendored so the gateway never imports the researcher-facing pip
    package. Returns a normalized dict, never the raw input."""
    obj = _require_object(value, "manifest")
    if obj.get("schema") != SCHEMA:
        raise ManifestError(f"unsupported manifest schema {obj.get('schema')!r}; expected {SCHEMA!r}")
    name = _require_name(obj.get("name"), "manifest.name")
    request = _validate_request(obj.get("request"))
    arms_value = obj.get("arms")
    if isinstance(arms_value, (str, bytes, bytearray)) or not isinstance(arms_value, list) or not arms_value:
        raise ManifestError("manifest.arms must be a non-empty array")
    arms = [_validate_arm(item, index) for index, item in enumerate(arms_value)]
    names = [arm["name"] for arm in arms]
    if len(set(names)) != len(names):
        raise ManifestError("manifest arm names must be unique")
    expected_health = _require_object(obj.get("expected_health", {}), "manifest.expected_health")
    metadata = _require_object(obj.get("metadata", {}), "manifest.metadata")
    return {"schema": SCHEMA, "name": name, "request": request, "arms": arms,
            "expected_health": expected_health, "metadata": metadata}


# ------------------------------------------------------------------------------- canonical json / sha

def _request_json_object(request: dict) -> dict:
    value: dict = {"topk": request["topk"]}
    if request["prompt_ids"] is not None:
        value["prompt_ids"] = list(request["prompt_ids"])
    else:
        value["prompt"] = request["prompt"]
    if request["continuation_ids"] is not None:
        value["continuation_ids"] = list(request["continuation_ids"])
    else:
        value["continuation"] = request["continuation"]
    return value


def _knockout_json_object(knockout: dict) -> dict:
    return {"layer": knockout["layer"], "queries": list(knockout["queries"]), "keys": list(knockout["keys"]),
            "renormalize": knockout["renormalize"]}


def _arm_json_object(arm: dict) -> dict:
    value: dict = {"name": arm["name"]}
    if arm["attention_knockout"]:
        value["attention_knockout"] = [_knockout_json_object(k) for k in arm["attention_knockout"]]
    if arm["steer"] is not None:
        value["steer"] = dict(arm["steer"])
    if arm["steer_vec"] is not None:
        value["steer_vec"] = list(arm["steer_vec"])
    if arm["metadata"]:
        value["metadata"] = dict(arm["metadata"])
    return value


def canonical_json_object(validated: dict) -> dict:
    """The EXACT JSON shape ``clozn_client.manifests.InterventionManifest.to_json_object()`` would
    produce for the same content (conditional-key omission included) -- so ``manifest_sha256`` below
    matches a client-computed ``.sha256`` for a byte-identical manifest."""
    value: dict = {
        "schema": validated["schema"],
        "name": validated["name"],
        "request": _request_json_object(validated["request"]),
        "arms": [_arm_json_object(arm) for arm in validated["arms"]],
    }
    if validated["expected_health"]:
        value["expected_health"] = dict(validated["expected_health"])
    if validated["metadata"]:
        value["metadata"] = dict(validated["metadata"])
    return value


def manifest_sha256(validated: dict) -> str:
    encoded = json.dumps(canonical_json_object(validated), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------ capability gate

# Which arm field requires which GET /health.capabilities key. Mirrors clozn-client's
# OperationSpec.health_capability for InterventionOperation.ATTENTION_KNOCKOUT (contracts.py).
_KNOCKOUT_CAPABILITY = "attn_knockout"


def _required_capabilities(validated: dict) -> list[str]:
    if any(arm["attention_knockout"] for arm in validated["arms"]):
        return [_KNOCKOUT_CAPABILITY]
    return []


def _capability_report(validated: dict, health: dict) -> list[dict]:
    caps = health.get("capabilities") if isinstance(health, dict) else None
    caps = caps if isinstance(caps, dict) else {}
    report = []
    for name in _required_capabilities(validated):
        available = caps.get(name) is True
        report.append({
            "capability": name,
            "available": available,
            "reason": (f"capabilities.{name}=true" if available
                       else f"capabilities.{name}=true is required (engine health: {caps.get(name)!r})"),
        })
    return report


# ---------------------------------------------------------------------------------------- execution

def _arm_kwargs(request: dict, arm: dict | None) -> dict:
    kwargs: dict = {"topk": request["topk"]}
    if request["prompt_ids"] is not None:
        kwargs["prompt_ids"] = list(request["prompt_ids"])
    else:
        kwargs["prompt"] = request["prompt"]
    if request["continuation_ids"] is not None:
        kwargs["continuation_ids"] = list(request["continuation_ids"])
    else:
        kwargs["continuation"] = request["continuation"]
    if arm is not None:
        if arm["steer"] is not None:
            kwargs["steer"] = dict(arm["steer"])
        if arm["steer_vec"] is not None:
            kwargs["steer_vec"] = list(arm["steer_vec"])
        if arm["attention_knockout"]:
            kwargs["attn_knockout"] = [_knockout_json_object(k) for k in arm["attention_knockout"]]
    return kwargs


def _support_drop(baseline_sum_logprob: float, arm_sum_logprob: float) -> float:
    # Positive support_drop means the intervention HURT the continuation (mirrors clozn-client's
    # InterventionArmResult.support_drop docstring exactly).
    return float(baseline_sum_logprob) - float(arm_sum_logprob)


def replay_manifest(manifest: dict, engine, *, health: dict | None = None,
                    clock=time.perf_counter) -> dict:
    """Validate + execute one ``clozn.intervention_manifest.v1`` manifest against ``engine``
    (duck-typed: ``engine.score(prompt=, prompt_ids=, continuation=, continuation_ids=, topk=, steer=,
    steer_vec=, attn_knockout=) -> dict``, the exact shape ``EngineClient.score`` exposes).

    Raises ``ManifestError`` for a structurally invalid manifest (the caller should map this to a 400).
    Returns a ``clozn.intervention_replay.v1`` result dict; when a required capability is missing the
    result has ``"performed": False`` and a typed ``capabilities`` report listing exactly what's
    missing -- the engine is never called in that case (the caller should map this to a 409/422, not
    a 500 or a silently-ignored intervention).

    Arms execute in manifest order, baseline first, one ``engine.score`` call per arm -- never
    generation, never sampling beyond what an arm's own ``steer``/``steer_vec`` asks for. Every call is
    a teacher-forced re-score of the SAME fixed continuation, so every result is labeled
    ``re_prefilled`` via ``clozn.experiments.stats.replay_class_for_meta`` (reused, not reinvented).
    """
    validated = validate_manifest(manifest)
    sha = manifest_sha256(validated)
    health = health if isinstance(health, dict) else {}
    capabilities = _capability_report(validated, health)
    missing = [item for item in capabilities if not item["available"]]
    started = clock()
    if missing:
        return {
            "schema": "clozn.intervention_replay.v1",
            "performed": False,
            "manifest_sha256": sha,
            "manifest_name": validated["name"],
            "error": {
                "code": "capability_unavailable",
                "message": "the engine does not satisfy a capability this manifest requires: "
                          + "; ".join(item["reason"] for item in missing),
            },
            "capabilities": capabilities,
            "timing": {"total_ms": round(max(0.0, (clock() - started) * 1000.0), 3)},
        }

    replay_class = replay_class_for_meta({"forced_rescore": True})
    request = validated["request"]

    baseline_started = clock()
    baseline_raw = engine.score(**_arm_kwargs(request, None))
    baseline_ms = max(0.0, (clock() - baseline_started) * 1000.0)
    baseline_sum = float(baseline_raw.get("sum_logprob", 0.0))

    arm_results = []
    for arm in validated["arms"]:
        arm_started = clock()
        arm_raw = engine.score(**_arm_kwargs(request, arm))
        arm_ms = max(0.0, (clock() - arm_started) * 1000.0)
        arm_results.append({
            "name": arm["name"],
            "result": arm_raw,
            "support_drop": _support_drop(baseline_sum, float(arm_raw.get("sum_logprob", 0.0))),
            "replay_class": replay_class,
            "metadata": dict(arm["metadata"]),
            "timing": {"score_ms": round(arm_ms, 3)},
        })

    identity = {
        "model": health.get("model"),
        "model_sha256": health.get("model_sha256"),
        "architecture": health.get("architecture"),
        "n_layer": health.get("n_layer"),
        "n_embd": health.get("n_embd"),
        "protocol_version": health.get("protocol_version"),
        "mode": health.get("mode"),
    }
    total_ms = max(0.0, (clock() - started) * 1000.0)
    return {
        "schema": "clozn.intervention_replay.v1",
        "performed": True,
        "manifest_sha256": sha,
        "manifest_name": validated["name"],
        "identity": identity,
        "capabilities": capabilities,
        "replay_class": replay_class,
        "baseline": baseline_raw,
        "arms": arm_results,
        "timing": {
            "baseline_ms": round(baseline_ms, 3),
            "arms_total_ms": round(sum(item["timing"]["score_ms"] for item in arm_results), 3),
            "total_ms": round(total_ms, 3),
            "score_calls": 1 + len(arm_results),
        },
    }
