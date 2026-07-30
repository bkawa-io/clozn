"""analysis/mech_target.py -- MECH-CASE-00: the behavioral-delta TARGET resolver.

THE RULE THIS MODULE ENFORCES
------------------------------
Cross-model mechanistic analysis (clozn.analysis.mechanistic_diff, causal_bisect, transplant) must never
scan a compatible model pair hunting for "interesting" activation differences -- an unguided scan over
every layer/position of two multi-billion-parameter models will always find SOMETHING that looks
different, and a difference found that way carries no evidence about what it means. The only defensible
starting point is a concrete, already-OBSERVED behavioral delta: a specific place where the two models'
outputs disagree, discovered by running them (an experiment, a teacher-forced ladder), not by inspecting
their internals. This module's whole job is turning one such observed delta into a versioned, file-
addressable `clozn.mechanistic-target.v1` document -- WHERE to look, never an explanation of WHY.

TWO ACCEPTED SOURCES (see `origin.kind` on the emitted document)
-------------------------------------------------------------------
  * `experiment_cell`  (`resolve_from_experiment_cell`) -- one FAILED cell of a
    `clozn.experiment.result.v0` (clozn/experiments/suite.py), compared against its own reference
    (baseline) variant's cell at the same (suite, case, seed) coordinate. `status == "fail"` is required:
    "fail" carries a real run_id/run AND a labeled disagreement (an assertion that did not pass);
    "error" (the generation itself threw) has no run to analyze; "pass"/"unscored" carry no delta to
    target. See clozn/experiments/suite.py's own docstring for the full cell-status contract -- this
    module reads it, never guesses it.
  * `diff_model_position`  (`resolve_from_diff_model_position`) -- one per-token disagreement position
    from `clozn.receipts.quant_receipts.diff_quant_scores()`'s `positions` list: a place where the
    reference and candidate models' OWN argmax choices disagreed on the identical teacher-forced
    continuation (`status == "flipped"`). `status == "preserved"` (the two models agreed) or `"unknown"`
    (no topk on at least one arm, so the flip status itself was never established) are refused -- neither
    is a concrete delta this module can honestly target.

VALIDATION BEFORE EMITTING
----------------------------
Every public resolver here takes an ALREADY-COMPUTED `clozn.pair-compatibility.v1` document (this module
never boots an engine or reads a GGUF itself -- see clozn/analysis/pair_compatibility.py, composed
read-only) and refuses, with a typed `reason` (see `REFUSAL_REASONS`), unless BOTH of the following hold:

  1. `pair_compat` permits per-token comparison (`pair_compatibility.may_per_token_compare`) -- a
     mismatched tokenizer makes a token id/position meaningless across the two models (see that module's
     own "THE TRAP" discussion), and this module's whole output is a token id + position.
  2. the run(s) supplying the observed delta carry a `identity.model_sha256` that matches EXACTLY the
     `pair_compat` side (`model_a`/`model_b`) they are being attributed to. This is a DIFFERENT check
     from (1): it is not asking whether the two models are compatible with each other, but whether the
     evidence handed to this resolver (a run recorded by generating against SOME model file) actually
     came from the SAME model file `pair_compat` was computed for. A caller who accidentally hands a
     pair-compatibility report for the wrong two files would otherwise get a target that looks valid but
     describes evidence from a model neither `reference_model` nor `candidate_model` actually is -- worse
     than no target at all, per docs/SEAMS.md rule 3 (no silent fallback).

A refusal is always `{"ok": False, "reason": <one of REFUSAL_REASONS>, "error": <message>}` -- never a
raised exception (mirrors clozn.analysis.pair_compatibility/mechanistic_diff/run_diff's own "pure,
never raise" discipline) and never a partially-built document.

NO CAUSAL VOCABULARY
----------------------
This module identifies a TARGET; it does not explain anything. Every string it can emit (docstrings,
comments, and every message this module builds) must stay free of causal vocabulary ("caused", "because",
"responsible for", "localized") -- that vocabulary belongs to a later slice that runs controlled
interventions with controls and can actually earn it (docs/SEAMS.md rule 4: evidence before narration;
mirrors clozn/analysis/mechanistic_diff.py's own discipline verbatim). tests/test_mech_target.py scans
this file's own source for exactly those words, the same guard tests/test_mechanistic_diff.py applies to
that module.

STDLIB ONLY
-----------
No numpy, not even lazily -- this module only ever compares small dicts and ints; there is no tensor math
here (that lives in clozn.analysis.mechanistic_diff, once a target from this module feeds it).
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Mapping

from clozn import schemas
from clozn.analysis import model_diff, pair_compatibility
from clozn.experiments import suite as experiment_suite

SCHEMA_VERSION = "clozn.mechanistic-target.v1"

REFUSAL_REASONS = frozenset({
    "invalid_result", "invalid_suite", "invalid_pair_compat", "invalid_run", "invalid_position",
    "invalid_anchor", "cell_not_found", "reference_cell_not_found", "candidate_not_failing",
    "candidate_has_no_run", "reference_has_no_run", "candidate_equals_reference",
    "per_token_comparison_not_permitted", "identity_unknown", "identity_mismatch",
    "position_not_flipped", "position_flip_unknown", "internal_error",
})


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_target_id() -> str:
    return "mechtarget_" + uuid.uuid4().hex[:16]


def _refuse(reason: str, message: str) -> dict:
    assert reason in REFUSAL_REASONS, f"unknown mech_target refusal reason {reason!r}"   # authoring guard
    return {"ok": False, "reason": reason, "error": message}


def _token_ref(token_id, piece) -> dict | None:
    """A `token_ref`-shaped dict carrying only the field(s) honestly known, or None if neither is --
    roadmap rule 2 (omit, never null-pad) applied to a single token observation."""
    out: dict = {}
    if isinstance(token_id, int) and not isinstance(token_id, bool):
        out["id"] = token_id
    if isinstance(piece, str):
        out["piece"] = piece
    return out or None


def _match_identity(pair_compat: Mapping, key: str, run: Mapping, label: str) -> dict | None:
    """None if `run`'s own recorded `identity.model_sha256` matches `pair_compat[key].sha256` exactly,
    else a typed refusal dict. Never guesses: missing on either side is `identity_unknown` (cannot
    confirm a match, so this refuses rather than assuming one), a genuine mismatch is `identity_mismatch`
    with both digests named."""
    model_ref = pair_compat.get(key) if isinstance(pair_compat.get(key), dict) else {}
    identity = run.get("identity") if isinstance(run.get("identity"), dict) else {}
    run_sha = identity.get("model_sha256")
    pair_sha = model_ref.get("sha256")
    if not run_sha or not pair_sha:
        return _refuse("identity_unknown", f"cannot confirm the {label} run's model identity matches "
                       f"pair_compatibility.{key} -- model_sha256 is missing on at least one side "
                       "(refusing rather than assuming a match)")
    if run_sha != pair_sha:
        return _refuse("identity_mismatch", f"the {label} run's recorded model_sha256 ({run_sha}) does "
                       f"not match pair_compatibility.{key}.sha256 ({pair_sha}) -- this target would be "
                       "built on evidence from a model file other than the one pair_compatibility assessed")
    return None


# ============================================================================= source #1: experiment cell

def _answer_position_from_response_diff(response_diff) -> tuple[dict, "str | None", "str | None"]:
    """The `answer_position` + (reference_piece, candidate_piece) derived from an already-computed
    `clozn.analysis.model_diff.diff_runs()` result -- the existing, tested, non-causal tool for "where do
    two recorded runs first disagree" (composed read-only, never reimplemented here). Falls back to
    `{"kind": "final_response", "note": ...}` whenever no single token position honestly carries the
    delta: the diff itself failed, no per-token trace was captured on one or both runs, or the two
    responses are token-identical (the candidate cell still failed -- e.g. a confidence/prove assertion
    -- but not on a text divergence this function can point at)."""
    if not isinstance(response_diff, dict) or not response_diff.get("ok"):
        return ({"kind": "final_response",
                "note": "the reference/candidate response comparison could not be computed"}, None, None)
    if not response_diff.get("trace_available"):
        return ({"kind": "final_response",
                "note": "no per-token trace is available on at least one of the two runs -- the "
                        "behavioral delta is reported at the level of the full response only"}, None, None)
    first_divergence = response_diff.get("first_divergence")
    if first_divergence is None:
        return ({"kind": "final_response",
                "note": "the reference and candidate responses are token-identical -- the behavioral "
                        "delta is not a text divergence"}, None, None)
    index = first_divergence.get("index")
    if not isinstance(index, int) or isinstance(index, bool):
        return ({"kind": "final_response",
                "note": "the response diff reported a divergence with no usable position index"}, None, None)
    return ({"kind": "token_index", "index": index},
            first_divergence.get("a_piece"), first_divergence.get("b_piece"))


def _trace_token_id(run: Mapping, answer_position: Mapping):
    """`run.trace.token_ids[answer_position.index]`, or None if unavailable -- an experiment-cell run is
    an ordinary recorded run (clozn.runs.trace), not a teacher-forced /score call, so its per-token ids
    are only present when the run's own capture tier recorded them; absence here is ordinary, not an
    error."""
    if not isinstance(answer_position, dict) or answer_position.get("kind") != "token_index":
        return None
    index = answer_position.get("index")
    trace = run.get("trace") if isinstance(run, dict) else None
    ids = trace.get("token_ids") if isinstance(trace, dict) else None
    if not isinstance(ids, list) or not isinstance(index, int) or not (0 <= index < len(ids)):
        return None
    value = ids[index]
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _experiment_cell_summary(*, suite: str, case: str, variant: str, seed: int, reference_variant: str,
                             failed_assertions: list, response_diff) -> str:
    checks = sorted({str(a.get("check")) for a in failed_assertions if a.get("check")})
    checks_text = ", ".join(checks) if checks else "an unnamed check"
    parts = [f"{len(failed_assertions)} assertion(s) did not pass on {suite}/{case} "
            f"(variant={variant!r}, seed={seed}), evaluated against the {reference_variant!r} "
            f"reference variant: {checks_text}."]
    if isinstance(response_diff, dict) and response_diff.get("ok"):
        summary = response_diff.get("summary") or {}
        first_divergence = response_diff.get("first_divergence")
        if summary.get("identical"):
            parts.append("The two responses are token-identical.")
        elif first_divergence is not None:
            parts.append(f"Responses first diverge at token {first_divergence.get('index')}: "
                        f"{reference_variant!r} said {first_divergence.get('a_piece')!r}, "
                        f"{variant!r} said {first_divergence.get('b_piece')!r}.")
    return " ".join(parts)


def resolve_from_experiment_cell(result: Mapping, *, case: str, variant: str, seed: int,
                                  pair_compat: Mapping, suite: str = "target",
                                  reference_variant: "str | None" = None,
                                  generated_at: "str | None" = None, target_id: "str | None" = None,
                                  validate: bool = True) -> dict:
    """Resolve a `clozn.mechanistic-target.v1` document from one FAILED cell of `result` (a
    `clozn.experiment.result.v0` document -- read clozn/experiments/suite.py, do not guess its shape),
    compared against its own reference variant's cell at the identical (suite, case, seed) coordinate.

    `pair_compat` is an already-computed `clozn.pair-compatibility.v1` document for the two model FILES
    this target's evidence is being attributed to (`model_a` = reference, `model_b` = candidate) -- this
    function never computes one itself (no GGUF read, no engine boot; see module docstring). Its two
    sides' `sha256` must match the reference/candidate cells' OWN recorded `run.identity.model_sha256`
    exactly, or this refuses (`identity_unknown`/`identity_mismatch`) rather than emit a target whose
    evidence and whose compatibility verdict may not describe the same model files.

    Returns `{"ok": True, "target": {...}}` on success, or a typed refusal (`{"ok": False, "reason",
    "error"}`, `reason` in `REFUSAL_REASONS`). Never raises: any unexpected internal failure is caught and
    reported as `internal_error` rather than propagated, matching every other clozn.analysis module's
    never-raise contract."""
    try:
        return _resolve_from_experiment_cell(
            result, case=case, variant=variant, seed=seed, pair_compat=pair_compat, suite=suite,
            reference_variant=reference_variant, generated_at=generated_at, target_id=target_id,
            validate=validate)
    except Exception as exc:      # noqa: BLE001 -- never raise; see docstring
        return _refuse("internal_error", f"{type(exc).__name__}: {exc}")


def _resolve_from_experiment_cell(result: Mapping, *, case: str, variant: str, seed: int,
                                  pair_compat: Mapping, suite: str, reference_variant, generated_at,
                                  target_id, validate: bool) -> dict:
    if not isinstance(result, dict) or not isinstance(result.get("cells"), list):
        return _refuse("invalid_result",
                       "result must be a clozn.experiment.result.v0 document with a 'cells' list")
    manifest = result.get("manifest")
    if not isinstance(manifest, dict):
        return _refuse("invalid_result", "result.manifest is missing or malformed")
    if suite not in ("target", "guard"):
        return _refuse("invalid_suite", f"suite must be 'target' or 'guard', got {suite!r}")
    if not isinstance(pair_compat, dict):
        return _refuse("invalid_pair_compat",
                       "pair_compat must be a clozn.pair-compatibility.v1 document (dict)")

    resolved_reference_variant = reference_variant or manifest.get("baseline_variant")
    if not isinstance(resolved_reference_variant, str) or not resolved_reference_variant:
        return _refuse("invalid_result", "could not determine a reference variant -- pass "
                       "reference_variant= explicitly, or fix manifest.baseline_variant")
    if variant == resolved_reference_variant:
        return _refuse("candidate_equals_reference",
                       f"variant {variant!r} is the same as the reference variant "
                       f"{resolved_reference_variant!r} -- there is nothing to target")

    candidate_cells = experiment_suite.select_cells(result, suite=suite, case=case, variant=variant,
                                                     seed=seed)
    if len(candidate_cells) != 1:
        return _refuse("cell_not_found", f"expected exactly one cell at (suite={suite!r}, case={case!r}, "
                       f"variant={variant!r}, seed={seed!r}); found {len(candidate_cells)}")
    candidate_cell = candidate_cells[0]
    candidate_status = candidate_cell.get("status")
    if candidate_status == "error":
        # error means the generation itself threw -- clozn.experiments.suite guarantees this cell has NO
        # run at all (see that module's docstring), so this gets its own precise reason rather than being
        # folded into the generic "not failing" bucket below.
        return _refuse("candidate_has_no_run",
                       f"cell (suite={suite}, case={case}, variant={variant}, seed={seed}) has status "
                       "'error' -- the candidate's generation itself failed and carries no run to analyze")
    if candidate_status != "fail":
        return _refuse("candidate_not_failing",
                       f"cell (suite={suite}, case={case}, variant={variant}, seed={seed}) has status "
                       f"{candidate_status!r}, not 'fail' -- a mechanistic target needs a concrete, "
                       "already-observed behavioral delta, and only a failed cell (assertions ran and did "
                       "not pass) carries both a run and a labeled disagreement")
    candidate_run = candidate_cell.get("run")
    if not isinstance(candidate_run, dict) or not candidate_run:
        return _refuse("candidate_has_no_run",
                       f"cell (suite={suite}, case={case}, variant={variant}, seed={seed}) carries no run "
                       "record to analyze")

    reference_cells = experiment_suite.select_cells(result, suite=suite, case=case,
                                                     variant=resolved_reference_variant, seed=seed)
    if len(reference_cells) != 1:
        return _refuse("reference_cell_not_found",
                       f"expected exactly one reference cell at (suite={suite!r}, case={case!r}, "
                       f"variant={resolved_reference_variant!r}, seed={seed!r}); found "
                       f"{len(reference_cells)}")
    reference_cell = reference_cells[0]
    reference_run = reference_cell.get("run")
    if not isinstance(reference_run, dict) or not reference_run:
        return _refuse("reference_has_no_run",
                       f"reference cell (suite={suite}, case={case}, variant={resolved_reference_variant}, "
                       f"seed={seed}) carries no run record to analyze")

    if not pair_compatibility.may_per_token_compare(pair_compat):
        reason = (pair_compat.get("verdict", {}).get("operations", {})
                 .get("per_token_comparison", {}).get("reason") or "per-token comparison is not permitted")
        return _refuse("per_token_comparison_not_permitted", reason)

    for key, run, label in (("model_a", reference_run, "reference"), ("model_b", candidate_run, "candidate")):
        refusal = _match_identity(pair_compat, key, run, label)
        if refusal is not None:
            return refusal

    failed_assertions = [dict(a) for a in (candidate_cell.get("assertions") or [])
                         if isinstance(a, dict) and a.get("status") in ("fail", "error")]
    response_diff = model_diff.diff_runs(reference_run, candidate_run)
    answer_position, reference_piece, candidate_piece = _answer_position_from_response_diff(response_diff)
    reference_token_id = _trace_token_id(reference_run, answer_position)
    candidate_token_id = _trace_token_id(candidate_run, answer_position)

    summary = _experiment_cell_summary(suite=suite, case=case, variant=variant, seed=seed,
                                       reference_variant=resolved_reference_variant,
                                       failed_assertions=failed_assertions, response_diff=response_diff)
    behavioral_delta: dict = {"summary": summary}
    if failed_assertions:
        behavioral_delta["failed_assertions"] = failed_assertions
    behavioral_delta["response_diff"] = response_diff

    target: dict = {
        "schema_version": SCHEMA_VERSION,
        "target_id": target_id or _new_target_id(),
        "generated_at": generated_at if generated_at is not None else _now_iso(),
        "origin": {
            "kind": "experiment_cell",
            "experiment_id": result.get("experiment_id") or "",
            "suite": suite, "case": case, "seed": seed,
            "reference_variant": resolved_reference_variant, "candidate_variant": variant,
            "reference_run_id": reference_run.get("id") or "",
            "candidate_run_id": candidate_run.get("id") or "",
        },
        "behavioral_delta": behavioral_delta,
        "answer_position": answer_position,
        "reference_model": dict(pair_compat.get("model_a") or {}),
        "candidate_model": dict(pair_compat.get("model_b") or {}),
        "pair_compatibility": dict(pair_compat),
    }
    reference_token = _token_ref(reference_token_id, reference_piece)
    if reference_token is not None:
        target["reference_token"] = reference_token
    candidate_token = _token_ref(candidate_token_id, candidate_piece)
    if candidate_token is not None:
        target["candidate_token"] = candidate_token

    if validate:
        schemas.validate(target)
    return {"ok": True, "target": target}


# ======================================================================= source #2: diff-model position

def resolve_from_diff_model_position(*, run: Mapping, position: Mapping, pair_compat: Mapping,
                                      anchor: str = "reference", label_a: str = "reference",
                                      label_b: str = "candidate", generated_at: "str | None" = None,
                                      target_id: "str | None" = None, validate: bool = True) -> dict:
    """Resolve a `clozn.mechanistic-target.v1` document from one per-token disagreement `position`
    (a `clozn.receipts.quant_receipts.diff_quant_scores()` `positions[i]` entry) -- a place where the
    reference and candidate models' OWN argmax choices disagreed on the identical teacher-forced
    continuation. `run` is the recorded run whose continuation was teacher-forced (the one `generate_*`
    call that produced it, under whichever side `anchor` names); `pair_compat` is an already-computed
    `clozn.pair-compatibility.v1` document, `model_a` = the `label_a` side, `model_b` = the `label_b`
    side. Composes `pair_compatibility`/`quant_receipts` shapes read-only; never computes either itself.

    Refuses (typed reason, never raises) unless `position["status"] == "flipped"` -- "preserved" (the two
    models agreed) and "unknown" (no topk on at least one arm, so flip status was never established)
    are both refused, since neither is a concrete delta this module can honestly target -- AND unless
    `pair_compat` permits per-token comparison AND `run`'s own recorded `identity.model_sha256` matches
    the `anchor` side of `pair_compat` exactly (see module docstring's VALIDATION BEFORE EMITTING).

    Returns `{"ok": True, "target": {...}}` or a typed refusal, exactly like
    `resolve_from_experiment_cell`."""
    try:
        return _resolve_from_diff_model_position(
            run=run, position=position, pair_compat=pair_compat, anchor=anchor, label_a=label_a,
            label_b=label_b, generated_at=generated_at, target_id=target_id, validate=validate)
    except Exception as exc:      # noqa: BLE001 -- never raise; see resolve_from_experiment_cell
        return _refuse("internal_error", f"{type(exc).__name__}: {exc}")


def _resolve_from_diff_model_position(*, run: Mapping, position: Mapping, pair_compat: Mapping,
                                      anchor: str, label_a: str, label_b: str, generated_at, target_id,
                                      validate: bool) -> dict:
    if not isinstance(pair_compat, dict):
        return _refuse("invalid_pair_compat",
                       "pair_compat must be a clozn.pair-compatibility.v1 document (dict)")
    if anchor not in ("reference", "candidate"):
        return _refuse("invalid_anchor", f"anchor must be 'reference' or 'candidate', got {anchor!r}")
    if not isinstance(run, dict) or not run.get("id"):
        return _refuse("invalid_run", "run must be a run record carrying an 'id'")
    if not isinstance(position, dict):
        return _refuse("invalid_position",
                       "position must be a diff_quant_scores() positions[] entry (dict)")

    if not pair_compatibility.may_per_token_compare(pair_compat):
        reason = (pair_compat.get("verdict", {}).get("operations", {})
                 .get("per_token_comparison", {}).get("reason") or "per-token comparison is not permitted")
        return _refuse("per_token_comparison_not_permitted", reason)

    anchor_key = "model_a" if anchor == "reference" else "model_b"
    refusal = _match_identity(pair_compat, anchor_key, run, "anchor")
    if refusal is not None:
        return refusal

    status = position.get("status")
    if status == "unknown":
        return _refuse("position_flip_unknown",
                       f"position {position.get('index')} has an unknown flip status (no topk on at "
                       "least one arm) -- there is no established disagreement here to target")
    if status != "flipped":
        return _refuse("position_not_flipped",
                       f"position {position.get('index')} has status {status!r}, not 'flipped' -- the "
                       "two models' own top choice agreed here, so there is no behavioral delta to target")

    index = position.get("index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        return _refuse("invalid_position", "position.index must be a non-negative integer")

    token_id, piece = position.get("token_id"), position.get("piece")
    argmax_a_id, argmax_a_piece = position.get("argmax_a_id"), position.get("argmax_a_piece")
    argmax_b_id, argmax_b_piece = position.get("argmax_b_id"), position.get("argmax_b_piece")

    summary = (f"at teacher-forced position {index} (forced token {piece!r}, id={token_id}), "
              f"{label_a}'s own top choice was {argmax_a_piece!r} (id={argmax_a_id}) and {label_b}'s own "
              f"top choice was {argmax_b_piece!r} (id={argmax_b_id}) -- the two models' argmax disagreed "
              "at this position.")

    target: dict = {
        "schema_version": SCHEMA_VERSION,
        "target_id": target_id or _new_target_id(),
        "generated_at": generated_at if generated_at is not None else _now_iso(),
        "origin": {
            "kind": "diff_model_position", "run_id": run.get("id"), "anchor": anchor,
            "position_index": index, "label_a": label_a, "label_b": label_b,
        },
        "behavioral_delta": {"summary": summary, "position_evidence": dict(position)},
        "answer_position": {"kind": "token_index", "index": index},
        "reference_model": dict(pair_compat.get("model_a") or {}),
        "candidate_model": dict(pair_compat.get("model_b") or {}),
        "pair_compatibility": dict(pair_compat),
    }
    reference_token = _token_ref(argmax_a_id, argmax_a_piece)
    if reference_token is not None:
        target["reference_token"] = reference_token
    candidate_token = _token_ref(argmax_b_id, argmax_b_piece)
    if candidate_token is not None:
        target["candidate_token"] = candidate_token

    if validate:
        schemas.validate(target)
    return {"ok": True, "target": target}


# ==================================================================================== file-addressable IO
# MECH-CLI-01 keeps every target file-addressable (write an artifact, print its path) -- no server
# storage, no job registry, in this slice. Mirrors clozn.experiments.suite.results_directory()/
# default_result_path()'s own pattern exactly; the CLI layer (clozn/cli/commands/mechanistic.py) is what
# actually calls clozn._io.atomic_write_json -- this module only computes the path.

def targets_directory() -> str:
    """Where `clozn diff-model --mechanistic`/`clozn experiment explain-cell` write target artifacts by
    default -- one constant so writer and any future reader cannot silently disagree on the path."""
    return os.path.expanduser("~/.clozn/mechanistic-targets")


def default_target_path(target: Mapping, directory: "str | None" = None) -> str:
    directory = directory or targets_directory()
    target_id = target.get("target_id") if isinstance(target, dict) else None
    return os.path.join(directory, str(target_id or _new_target_id()) + ".json")
