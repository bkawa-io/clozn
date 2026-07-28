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
    messages = delivered.get("messages") if isinstance(delivered, dict) else None
    return messages if isinstance(messages, list) else None


def _assembled_messages(run):
    survived = _context_receipt(run).get("survived")
    messages = survived.get("assembled_messages") if isinstance(survived, dict) else None
    return messages if isinstance(messages, list) else None


def _final_prompt(run):
    survived = _context_receipt(run).get("survived")
    prompt = survived.get("final_prompt") if isinstance(survived, dict) else None
    return prompt if isinstance(prompt, str) else None


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_messages(messages: list) -> str:
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _length_of(value) -> int | None:
    if isinstance(value, (str, list)):
        return len(value)
    return None


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


_CONTEXT_NOT_RUN = (
    ("context_diff:omissions",
     "segment-level omission reason codes require feature 06's context-receipt v1; this build's "
     "context_receipt precursor (clozn.runs.context_receipt) has no segment identity"),
    ("context_diff:special_tokens",
     "special-token insertion tracking requires feature 06's context-receipt v1; not captured by this "
     "build's context_receipt precursor"),
)


def context_diff_steps(baseline_run: dict, candidate_run: dict) -> list[dict]:
    """Spec step 2 (rendered prompt diff), against today's context_receipt precursor: delivered
    messages, assembled messages, and the exact rendered prompt, each compared by digest. Omitted/
    truncated-segment and special-token detail need feature 06's segment-identified artifact and are
    always reported as explicit `not_run` placeholders here, never silently skipped."""
    steps = [
        _digest_comparison_step(
            "context_diff:delivered_messages", "context_receipt.delivered.messages",
            _delivered_messages(baseline_run), _delivered_messages(candidate_run), _sha256_messages),
        _digest_comparison_step(
            "context_diff:assembled_messages", "context_receipt.survived.assembled_messages",
            _assembled_messages(baseline_run), _assembled_messages(candidate_run), _sha256_messages),
        _digest_comparison_step(
            "context_diff:rendered_prompt", "context_receipt.survived.final_prompt",
            _final_prompt(baseline_run), _final_prompt(candidate_run), _sha256_text),
    ]
    for kind, reason in _CONTEXT_NOT_RUN:
        steps.append(_step(kind, "not_run", [{"note": reason}], reason=reason))
    return steps


# The two step families this build actually executes. Keyed by the `--steps` filter name the CLI accepts.
STEP_FAMILIES = {
    "identity": identity_diff_steps,
    "context": context_diff_steps,
}
