"""commands.mechanistic -- MECH-CLI-01: the two wired entry points onto `clozn.analysis.mech_target`
(MECH-CASE-00's behavioral-target resolver):

    clozn diff-model A.gguf B.gguf --mechanistic --case result.json:suite/case
    clozn experiment explain-cell RESULT --case CASE --variant candidate --seed 0

Both resolve one failed `clozn.experiment.result.v0` cell against its own reference (baseline) variant's
cell, write the resulting `clozn.mechanistic-target.v1` artifact to
`clozn.analysis.mech_target.targets_directory()` (or `--out`), and print its path -- file-addressable,
per MECH-CLI-01's explicit scope: no server storage, no job registry, in this slice.

WHERE THE TWO COMMANDS DIFFER
-------------------------------
`experiment explain-cell` needs no model file arguments at all: the reference/candidate model FILES come
straight from the two cells' own recorded `run.identity.model_path` (clozn.runs.identity -- the immutable
reproduction identity every run already carries), and this module reads both GGUF headers itself
(`pair_compatibility.assess_gguf_pair`, no engine boot) to build the `clozn.pair-compatibility.v1`
document `mech_target` requires.

`diff-model --mechanistic` instead takes the two model files EXPLICITLY on the command line (the same
`reference`/`candidate` positional arguments the ordinary per-token-ladder `diff-model` path uses) --
`--case` only picks WHICH failed case anchors the target, never which models. The pair-compatibility
document is built from those two explicit files, and `mech_target`'s own identity check then confirms the
picked case's cells were actually run against (at least one of) those same two files -- refusing,
never guessing, if the `--case` you named does not correspond to the `A.gguf`/`B.gguf` you are pointing
the mechanistic path at.

`--case`'s wire format is `RESULT.json:SUITE/CASE` (e.g. `result.json:target/refusal_case`); `--variant`
(default `candidate`) and `--reference-variant` (default: the manifest's own `baseline_variant`) refine
which two cells at that coordinate to compare, mirroring `explain-cell`'s own flags.

Neither command boots an engine or a GPU -- resolving a target is GGUF-header + JSON-file work only
(mirrors `clozn.analysis.pair_compatibility.assess_gguf_pair`'s own "no engine boot required" scope).
Model-free / unit-tested throughout (tests/test_mechanistic_cli.py): `pair_compatibility.assess_gguf_pair`
and `clozn.cli.commands.models.resolve_model` are the only two calls in this module that touch the
filesystem for real, and both are monkeypatched out in tests exactly like tests/test_pair_compatibility.py
monkeypatches `gguf_identity` -- no real GGUF, no real engine, ever exercised by this module's own suite.
"""
from __future__ import annotations

import json
import os

from clozn._io import atomic_write_json
from clozn.analysis import mech_target, pair_compatibility
from clozn.cli.commands.models import resolve_model
from clozn.experiments import suite as experiment_suite


# ============================================================================================ rendering

def format_target_report(target: dict) -> str:
    """Pure JSON(target) -> text render -- no I/O, mirrors every other clozn CLI command's `format_*`
    convention (docs/SEAMS.md: both --json and human-readable output are required)."""
    origin = target.get("origin") or {}
    delta = target.get("behavioral_delta") or {}
    position = target.get("answer_position") or {}
    ref_model = target.get("reference_model") or {}
    cand_model = target.get("candidate_model") or {}
    verdict = (target.get("pair_compatibility") or {}).get("verdict") or {}

    lines = [f"clozn mechanistic target {target.get('target_id')}  ({origin.get('kind')})"]
    if origin.get("kind") == "experiment_cell":
        lines.append(f"  {origin.get('suite')}/{origin.get('case')}  "
                     f"{origin.get('reference_variant')!r} (reference) vs "
                     f"{origin.get('candidate_variant')!r} (candidate)  seed={origin.get('seed')}")
    elif origin.get("kind") == "diff_model_position":
        lines.append(f"  run={origin.get('run_id')}  anchor={origin.get('anchor')}  "
                     f"position={origin.get('position_index')}")
    lines.append(f"  reference: {ref_model.get('filename') or ref_model.get('sha256') or '?'}")
    lines.append(f"  candidate: {cand_model.get('filename') or cand_model.get('sha256') or '?'}")
    lines.append(f"  pair compatibility: {verdict.get('overall', '?')}")
    lines.append("")
    lines.append(delta.get("summary", "(no summary)"))
    lines.append("")
    if position.get("kind") == "token_index":
        lines.append(f"answer position: token index {position.get('index')}")
    else:
        lines.append(f"answer position: final response ({position.get('note', 'no token position')})")
    ref_tok, cand_tok = target.get("reference_token"), target.get("candidate_token")
    if ref_tok or cand_tok:
        lines.append(f"  reference token: {ref_tok or '(unavailable)'}")
        lines.append(f"  candidate token: {cand_tok or '(unavailable)'}")
    return "\n".join(lines)


# =========================================================================================== the writer

def _emit_target(outcome: dict, args) -> int:
    from clozn.cli import main as ctx
    if not outcome.get("ok"):
        raise ctx.CloznError(f"mechanistic target refused ({outcome.get('reason')}): {outcome.get('error')}")
    target = outcome["target"]
    out_path = getattr(args, "out", None) or mech_target.default_target_path(target)
    atomic_write_json(out_path, target, indent=2, ensure_ascii=False)
    if args.json:
        print(json.dumps(target, indent=2, ensure_ascii=False))
    else:
        print(format_target_report(target))
        print(f"  target: {out_path}")
    return 0


# ==================================================================================== explain-cell (CLI)
# Nested under the EXISTING `clozn experiment` subparser tree (clozn/cli/commands/experiment_suite.py) --
# there is no separate top-level registration here, and no edit to clozn/cli/main.py.

def cmd_explain_cell(args):
    """`clozn experiment explain-cell RESULT --case CASE --variant V [--reference-variant V]
    [--suite target|guard] [--seed N] [--out PATH] [--json]`."""
    from clozn.cli import main as ctx
    try:
        result = experiment_suite.load_result(args.result)
    except experiment_suite.ManifestError as exc:
        raise ctx.CloznError(f"could not read experiment result: {exc}") from exc

    manifest = result.get("manifest") or {}
    reference_variant = args.reference_variant or manifest.get("baseline_variant")
    if not isinstance(reference_variant, str) or not reference_variant:
        raise ctx.CloznError("could not determine a reference variant -- pass --reference-variant "
                             "explicitly, or fix the manifest's baseline_variant")

    candidate_cells = experiment_suite.select_cells(result, suite=args.suite, case=args.case,
                                                     variant=args.variant, seed=args.seed)
    if len(candidate_cells) != 1:
        raise ctx.CloznError(f"expected exactly one cell at (suite={args.suite!r}, case={args.case!r}, "
                             f"variant={args.variant!r}, seed={args.seed!r}); found {len(candidate_cells)}")
    candidate_run = candidate_cells[0].get("run")
    if not isinstance(candidate_run, dict) or not candidate_run:
        raise ctx.CloznError("the candidate cell carries no run record -- nothing to analyze")

    reference_cells = experiment_suite.select_cells(result, suite=args.suite, case=args.case,
                                                     variant=reference_variant, seed=args.seed)
    if len(reference_cells) != 1:
        raise ctx.CloznError(f"expected exactly one reference cell at (suite={args.suite!r}, "
                             f"case={args.case!r}, variant={reference_variant!r}, seed={args.seed!r}); "
                             f"found {len(reference_cells)}")
    reference_run = reference_cells[0].get("run")
    if not isinstance(reference_run, dict) or not reference_run:
        raise ctx.CloznError("the reference cell carries no run record -- nothing to analyze")

    reference_path = (reference_run.get("identity") or {}).get("model_path")
    candidate_path = (candidate_run.get("identity") or {}).get("model_path")
    if not reference_path or not candidate_path:
        raise ctx.CloznError("cannot resolve a mechanistic target: at least one cell's run carries no "
                             "identity.model_path -- this needs a run recorded with reproduction identity "
                             "(clozn.runs.identity), which older runs may not have")

    pair_compat = pair_compatibility.assess_gguf_pair(reference_path, candidate_path,
                                                       label_a=reference_variant, label_b=args.variant)
    outcome = mech_target.resolve_from_experiment_cell(
        result, suite=args.suite, case=args.case, variant=args.variant, seed=args.seed,
        reference_variant=reference_variant, pair_compat=pair_compat)
    return _emit_target(outcome, args)


# ============================================================================= diff-model --mechanistic
# Called from clozn/cli/commands/diff_model.py's cmd_diff_model when --mechanistic is set (that module
# owns the `--mechanistic`/`--case`/`--variant`/`--reference-variant`/`--seed`/`--out` argparse wiring on
# its own `diff-model` subparser; this function owns everything that happens once those args are parsed).

def _parse_case_arg(raw: "str | None") -> tuple[str, str, str]:
    """`RESULT.json:SUITE/CASE` -> (result_path, suite, case). Splits on the LAST colon, never the
    first: a Windows absolute path (`C:\\models\\result.json`) carries its own colon right after the
    drive letter, so `partition(":")` would slice the path itself in two -- `rpartition(":")` is exact
    here because neither a suite name nor a case name may itself contain a colon (clozn.experiments.suite
    case/variant names are plain identifiers)."""
    from clozn.cli import main as ctx
    if not raw:
        raise ctx.CloznError("--mechanistic requires --case RESULT.json:SUITE/CASE")
    result_path, sep, coordinate = raw.rpartition(":")
    suite_name, sep2, case_name = coordinate.partition("/")
    if not sep or not sep2 or not result_path or not suite_name or not case_name:
        raise ctx.CloznError("--case must be RESULT.json:SUITE/CASE, e.g. result.json:target/refusal_case "
                             f"(got {raw!r})")
    return result_path, suite_name, case_name


def cmd_diff_model_mechanistic(args):
    """`clozn diff-model A.gguf B.gguf --mechanistic --case RESULT.json:SUITE/CASE [--variant V]
    [--reference-variant V] [--seed N] [--out PATH] [--json]` -- resolves a target instead of running the
    ordinary per-token ladder. Needs no engine/GPU: `resolve_model` only resolves file paths, and
    `pair_compatibility.assess_gguf_pair` only reads GGUF headers."""
    from clozn.cli import main as ctx
    result_path, suite_name, case_name = _parse_case_arg(getattr(args, "case", None))
    try:
        result = experiment_suite.load_result(result_path)
    except experiment_suite.ManifestError as exc:
        raise ctx.CloznError(f"could not read experiment result: {exc}") from exc

    model_a = resolve_model(args.reference)
    model_b = resolve_model(args.candidate)
    label_a = os.path.splitext(os.path.basename(model_a))[0]
    label_b = os.path.splitext(os.path.basename(model_b))[0]
    pair_compat = pair_compatibility.assess_gguf_pair(model_a, model_b, label_a=label_a, label_b=label_b)

    outcome = mech_target.resolve_from_experiment_cell(
        result, suite=suite_name, case=case_name, variant=getattr(args, "variant", "candidate") or "candidate",
        seed=getattr(args, "seed", 0) or 0, reference_variant=getattr(args, "reference_variant", None),
        pair_compat=pair_compat)
    return _emit_target(outcome, args)
