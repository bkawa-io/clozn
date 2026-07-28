"""Model-free triage steps: identity diff (spec step 1) and context/rendered-prompt diff (spec step 2).

Both operate on two already-recorded run dicts (as returned by ``clozn.runs.store.get_run`` or embedded
in a ``clozn.experiment.result.v0`` cell) and never touch a model, an engine, or the network. Every
returned step is shaped for ``clozn.schemas.defs.clozn.triage.v1``'s ``Step`` definition.

WHAT THESE STEPS PROVE AND WHAT THEY DO NOT
---------------------------------------------
A comparison step's status is a RAW fact -- ``matched`` or ``mismatched`` -- never a causal claim.
``clozn.triage.status.hypothesis_for`` (used by the rule engine, not here) is the mechanical promotion of
that raw fact to a per-hypothesis verdict (``eliminated``/``observed``). Nothing in this module decides
whether a mismatch CAUSED anything -- that requires a controlled intervention (spec step 3), which this
build does not implement (see ``clozn.triage.artifact``'s explicit ``not_run`` placeholders).

CONTENT IS NEVER EMBEDDED VERBATIM
-----------------------------------
Rendered prompts and assembled messages may be long and may carry private content. Every context-diff
step compares SHA-256 digests (plus lengths, for a human skimming a mismatch), never raw text -- this
mirrors the "hashes only" style already used by ``clozn.runs.identity`` and keeps a stored triage artifact
small and redaction-safe by construction, not by an opt-in flag someone has to remember.
"""
from __future__ import annotations

import hashlib
import json


def _step(kind: str, status: str, observations: list, *, inputs=None, artifact_refs=None,
          model_runs: int = 0, caveats=None, reason: str | None = None) -> dict:
    step = {
        "kind": kind,
        "status": status,
        "inputs": dict(inputs or {}),
        "observations": list(observations),
        "artifact_refs": list(artifact_refs or []),
        "cost": {"model_runs": model_runs},
    }
    if caveats:
        step["caveats"] = list(caveats)
    if reason:
        step["reason"] = reason
    return step


# ============================================================================================= step 1 ===

_IDENTITY_SCALAR_FIELDS = (
    ("model", "model_sha256"),
    ("template", "template_fingerprint"),
    ("engine_build", "engine_build"),
    ("clozn_version", "clozn_version"),
)

_TEMPLATE_TOKENIZER_CAVEAT = (
    "template_fingerprint hashes both template rendering AND tokenizer behavior together (clozn.runs."
    "identity.template_fingerprint renders one canonical conversation and hashes the result) -- a "
    "mismatch here cannot be attributed to template vs tokenizer alone. See identity_diff:tokenizer."
)

_TOKENIZER_NOT_RUN_REASON = (
    "tokenizer identity is not separately measurable in this build; template_fingerprint conflates it "
    "with template rendering (see identity_diff:template's caveat)"
)


def _identity(run) -> dict:
    identity = run.get("identity") if isinstance(run, dict) else None
    return identity if isinstance(identity, dict) else {}


def _comparison_step(kind: str, path: str, baseline_value, candidate_value) -> dict:
    """A plain equality comparison of two already-extracted values. `matched`/`mismatched` when both
    sides are present; `inconclusive` when exactly one is missing; `not_run` when both are missing (the
    dimension simply was not recorded for either run -- there was nothing to compare)."""
    if baseline_value is None and candidate_value is None:
        return _step(kind, "not_run", [{"path": path, "note": "not recorded on either run"}],
                     reason=f"{path} was not recorded on either run")
    if baseline_value is None or candidate_value is None:
        missing_side = "baseline" if baseline_value is None else "candidate"
        return _step(
            kind, "inconclusive",
            [{"path": path, "baseline": baseline_value, "candidate": candidate_value}],
            reason=f"{path} is missing on the {missing_side} run; comparison could not be attempted")
    raw = "matched" if baseline_value == candidate_value else "mismatched"
    return _step(kind, raw, [{"path": path, "baseline": baseline_value, "candidate": candidate_value}])


def identity_diff_steps(baseline_run: dict, candidate_run: dict) -> list[dict]:
    """Spec step 1 (identity diff): model, tokenizer, template, engine, version, and any registered
    identity-extension namespace (``identity.ext.*`` -- adapter, engine_artifact, machine, ... per
    ``clozn.runs.identity_ext``). Generic over ``ext``, so a future facet (e.g. feature 03's adapter
    identity) is picked up with zero code changes here: whatever namespaces exist on either run's
    ``identity.ext`` get their own diff step."""
    baseline_identity = _identity(baseline_run)
    candidate_identity = _identity(candidate_run)

    steps = []
    for kind_suffix, field in _IDENTITY_SCALAR_FIELDS:
        step = _comparison_step(f"identity_diff:{kind_suffix}", f"identity.{field}",
                                baseline_identity.get(field), candidate_identity.get(field))
        if kind_suffix == "template":
            step["caveats"] = [_TEMPLATE_TOKENIZER_CAVEAT]
        steps.append(step)

    # No standalone tokenizer identity hash exists today -- always an explicit not_run, never silently
    # absent (spec: "Unsupported steps are visible").
    steps.append(_step("identity_diff:tokenizer", "not_run",
                       [{"note": _TOKENIZER_NOT_RUN_REASON}], reason=_TOKENIZER_NOT_RUN_REASON))

    baseline_ext = baseline_identity.get("ext")
    candidate_ext = candidate_identity.get("ext")
    baseline_ext = baseline_ext if isinstance(baseline_ext, dict) else {}
    candidate_ext = candidate_ext if isinstance(candidate_ext, dict) else {}
    for namespace in sorted(set(baseline_ext) | set(candidate_ext)):
        steps.append(_comparison_step(
            f"identity_diff:ext.{namespace}", f"identity.ext.{namespace}",
            baseline_ext.get(namespace), candidate_ext.get(namespace)))
    return steps


# ============================================================================================= step 2 ===

def _context_receipt(run) -> dict:
    receipt = run.get("context_receipt") if isinstance(run, dict) else None
    return receipt if isinstance(receipt, dict) else {}


def _delivered_messages(run):
    delivered = _context_receipt(run).get("delivered")
    if isinstance(delivered, list):
        # v1 metadata-only segments.  Full content remains on run.messages,
        # but the stable IDs/hashes/order are the stronger comparison input.
        return delivered
    messages = delivered.get("messages") if isinstance(delivered, dict) else None
    return messages if isinstance(messages, list) else None


def _assembled_messages(run):
    assembled = _context_receipt(run).get("assembled")
    if isinstance(assembled, list):
        return assembled
    survived = _context_receipt(run).get("survived")
    messages = survived.get("assembled_messages") if isinstance(survived, dict) else None
    return messages if isinstance(messages, list) else None


def _final_prompt(run):
    rendered = _context_receipt(run).get("rendered")
    sha256 = rendered.get("sha256") if isinstance(rendered, dict) else None
    if isinstance(sha256, str) and len(sha256) == 64:
        return {"stored_sha256": sha256}
    survived = _context_receipt(run).get("survived")
    prompt = survived.get("final_prompt") if isinstance(survived, dict) else None
    return prompt if isinstance(prompt, str) else None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_messages(messages: list) -> str:
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_json(value) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _length_of(value) -> int | None:
    if isinstance(value, (str, list)):
        return len(value)
    return None


def _sha256_rendered(value) -> str:
    if isinstance(value, dict) and isinstance(value.get("stored_sha256"), str):
        return value["stored_sha256"]
    return _sha256_text(value)


def _omissions(run):
    value = _context_receipt(run).get("omissions")
    return value if isinstance(value, list) else None


def _special_tokens(run):
    value = _context_receipt(run).get("rendered")
    value = value.get("special_tokens") if isinstance(value, dict) else None
    return value if isinstance(value, list) else None


def _digest_comparison_step(kind: str, path: str, baseline_value, candidate_value, digest_fn) -> dict:
    """Same shape as `_comparison_step`, but compares a SHA-256 digest of each side rather than the raw
    value -- baseline/candidate content is never embedded in the artifact (see module docstring)."""
    if baseline_value is None and candidate_value is None:
        return _step(kind, "not_run", [{"path": path, "note": "not captured on either run"}],
                     reason=f"{path} was not captured on either run")
    if baseline_value is None or candidate_value is None:
        missing_side = "baseline" if baseline_value is None else "candidate"
        return _step(
            kind, "inconclusive",
            [{"path": path, "baseline_captured": baseline_value is not None,
              "candidate_captured": candidate_value is not None}],
            reason=f"{path} is missing on the {missing_side} run; comparison could not be attempted")
    baseline_digest = digest_fn(baseline_value)
    candidate_digest = digest_fn(candidate_value)
    raw = "matched" if baseline_digest == candidate_digest else "mismatched"
    return _step(kind, raw, [{
        "path": path,
        "baseline_sha256": baseline_digest, "candidate_sha256": candidate_digest,
        "baseline_length": _length_of(baseline_value), "candidate_length": _length_of(candidate_value),
    }])


def context_diff_steps(baseline_run: dict, candidate_run: dict) -> list[dict]:
    """Spec step 2 over both v1 segment receipts and historical receipt shapes."""
    return [
        _digest_comparison_step(
            "context_diff:delivered_messages", "context_receipt.delivered",
            _delivered_messages(baseline_run), _delivered_messages(candidate_run), _sha256_messages),
        _digest_comparison_step(
            "context_diff:assembled_messages", "context_receipt.assembled",
            _assembled_messages(baseline_run), _assembled_messages(candidate_run), _sha256_messages),
        _digest_comparison_step(
            "context_diff:rendered_prompt", "context_receipt.rendered.sha256",
            _final_prompt(baseline_run), _final_prompt(candidate_run), _sha256_rendered),
        _digest_comparison_step(
            "context_diff:omissions", "context_receipt.omissions",
            _omissions(baseline_run), _omissions(candidate_run), _sha256_messages),
        _digest_comparison_step(
            "context_diff:special_tokens", "context_receipt.rendered.special_tokens",
            _special_tokens(baseline_run), _special_tokens(candidate_run), _sha256_messages),
    ]


# ====================================================================================== sampling ===

_SAMPLING_FIELDS = (
    "sampling", "sampler_mode", "temperature", "top_p", "top_k",
    "repetition_penalty", "repeat_penalty", "seed", "max_tokens", "stop",
)


def sampling_diff_steps(baseline_run: dict, candidate_run: dict) -> list[dict]:
    baseline_meta = baseline_run.get("meta") if isinstance(baseline_run, dict) else None
    candidate_meta = candidate_run.get("meta") if isinstance(candidate_run, dict) else None
    baseline_meta = baseline_meta if isinstance(baseline_meta, dict) else {}
    candidate_meta = candidate_meta if isinstance(candidate_meta, dict) else {}
    return [
        _comparison_step(
            f"sampling_diff:{field}", f"meta.{field}",
            baseline_meta.get(field), candidate_meta.get(field),
        )
        for field in _SAMPLING_FIELDS
    ]


# ================================================================================= tool contracts ===

def _output_contract(run) -> dict:
    value = run.get("output_contract") if isinstance(run, dict) else None
    return value if isinstance(value, dict) else {}


def _contract_request(run):
    value = _output_contract(run).get("request")
    return value if isinstance(value, dict) and value else None


def _parser_runtime(run):
    contract = _output_contract(run)
    native = contract.get("native")
    native = native if isinstance(native, dict) else {}
    qualification = contract.get("qualification")
    qualification = qualification if isinstance(qualification, dict) else {}
    pipeline = native.get("pipeline") or qualification.get("pipeline")
    parser = contract.get("parser")
    if isinstance(pipeline, dict) and pipeline:
        return pipeline
    if isinstance(parser, dict) and parser:
        # v1 parser evidence has no one fixed version key.  Hash the complete
        # metadata object rather than guessing which field is authoritative.
        return parser
    return None


def _raw_output(run):
    value = _output_contract(run).get("raw_model_output")
    return value if isinstance(value, str) else None


def _outcome(run):
    value = _output_contract(run).get("outcome")
    if not isinstance(value, dict) or not value:
        return None
    # No raw generated text or free-form error message in triage artifacts.
    return {key: value[key] for key in ("status", "code", "kind", "tool_name") if key in value}


def tool_contract_diff_steps(baseline_run: dict, candidate_run: dict) -> list[dict]:
    steps = [
        _digest_comparison_step(
            "tool_contract_diff:requested_schema", "output_contract.request",
            _contract_request(baseline_run), _contract_request(candidate_run), _sha256_json),
        _digest_comparison_step(
            "tool_contract_diff:parser_runtime", "output_contract.parser_runtime",
            _parser_runtime(baseline_run), _parser_runtime(candidate_run), _sha256_json),
        _digest_comparison_step(
            "tool_contract_diff:raw_model_output", "output_contract.raw_model_output",
            _raw_output(baseline_run), _raw_output(candidate_run), _sha256_text),
        _digest_comparison_step(
            "tool_contract_diff:outcome", "output_contract.outcome",
            _outcome(baseline_run), _outcome(candidate_run), _sha256_json),
    ]
    candidate_outcome = _outcome(candidate_run) or {}
    request_step = next(s for s in steps if s["kind"].endswith("requested_schema"))
    parser_step = next(s for s in steps if s["kind"].endswith("parser_runtime"))
    if candidate_outcome.get("status") == "error":
        code = candidate_outcome.get("code")
        if request_step["status"] == "mismatched":
            reason = "requested tool/schema contract changed; parser/model attribution remains unisolated"
        elif parser_step["status"] == "mismatched":
            reason = "parser/runtime identity changed; model malformed-output attribution remains unisolated"
        elif code:
            reason = (
                f"candidate structured output failed with recorded code {code}; request/parser metadata "
                "matched where captured, but no controlled parser swap ran"
            )
        else:
            reason = "candidate structured output failed without a recorded error code"
        steps.append(_step(
            "tool_contract_diff:failure_class", "observed",
            [{"candidate_status": "error", **({"candidate_code": code} if code else {})}],
            reason=reason,
        ))
    return steps


# =================================================================================== quant/export ===

def _quant_identity(run):
    meta = run.get("meta") if isinstance(run, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    return {
        key: meta[key] for key in ("quant", "quantization", "model_file")
        if key in meta
    } or None


def _adapter_identity(run):
    identity = _identity(run)
    ext = identity.get("ext")
    ext = ext if isinstance(ext, dict) else {}
    value = ext.get("adapter")
    return value if isinstance(value, dict) and value else None


def quant_export_diff_steps(baseline_run: dict, candidate_run: dict) -> list[dict]:
    return [
        _digest_comparison_step(
            "quant_export_diff:quantization", "meta.quantization",
            _quant_identity(baseline_run), _quant_identity(candidate_run), _sha256_json),
        _digest_comparison_step(
            "quant_export_diff:adapter", "identity.ext.adapter",
            _adapter_identity(baseline_run), _adapter_identity(candidate_run), _sha256_json),
    ]


# The two step families this build actually executes. Keyed by the `--steps` filter name the CLI accepts.
STEP_FAMILIES = {
    "identity": identity_diff_steps,
    "context": context_diff_steps,
    "sampling": sampling_diff_steps,
    "quant_export": quant_export_diff_steps,
    "tool_contract": tool_contract_diff_steps,
}
