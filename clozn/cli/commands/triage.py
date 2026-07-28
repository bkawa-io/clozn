"""`clozn triage` -- evidence-ladder automatic regression triage (roadmap feature 05).

    clozn triage --baseline-run RUN --candidate-run RUN [--deep] [--steps a,b] [--json] [--out PATH]
    clozn triage RESULT.json --case CASE [--variant NAME] [--seed N] [--deep] [--json] [--out PATH]

NAMING, NOT THE SPEC'S VERB
-----------------------------
notes/agent_roadmap/05-automatic-regression-triage.md names this command `clozn diagnose`. That verb is
already an existing, unrelated, hand-wired command (`clozn/cli/commands/diagnose.py` +
`clozn/runs/diagnosis.py`: evidence-only single-run latency/cutoff explanation, registered directly in
`clozn/cli/main.py`, not autoloaded). `clozn triage` is used here instead -- reviewed and approved as a
substitution for the spec's literal verb, not a silent rename.

`RESULT.json` is a `clozn.experiment.result.v0` artifact -- feature 04's schema. This module consumes it
by name (`clozn.experiments.suite.load_result`/`select_cells`) and does not define or validate its shape
beyond what that module already does.

SCOPE OF THIS BUILD
----------------------
Only the model-free steps (identity diff, context diff) actually execute. Every other step the spec's
triage order names (controlled replay, quant/export, tool contract, internal localization) is reported as
an explicit, reasoned `not_run` -- `--deep` does not change that in this build; it is accepted and
threaded into the placeholder so the request is visible, not silently ignored (roadmap rule 3: no silent
fallback).
"""
from __future__ import annotations

import json

from clozn.triage.artifact import ALL_FAMILIES, build_triage_artifact

CLOZN_AUTOLOAD = True


_BUCKET_HEADINGS = (
    ("causally_supported", "CAUSALLY SUPPORTED"),
    ("eliminated", "ELIMINATED"),
    ("observed", "OBSERVED (not causally isolated)"),
    ("correlated", "CORRELATED (not causally isolated)"),
    ("inconclusive", "INCONCLUSIVE"),
    ("not_run", "NOT RUN"),
    ("unsupported", "UNSUPPORTED"),
)


def format_triage(document: dict) -> str:
    """Render the evidence ladder. Never prints anything not traceable to `document['steps']` -- the
    headline names a classification only when `summary.classification` already does (see
    `clozn.triage.rules.classify`'s "only causally_supported names a classification" rule)."""
    summary = document.get("summary") or {}
    classification = summary.get("classification", "undetermined")
    lines = []
    if classification != "undetermined":
        headline = classification.upper().replace("_", " ").replace(":", " ")
        lines.append(f"LIKELY {headline}")
    else:
        lines.append("UNDETERMINED")
    lines.append(f"  confidence basis: {summary.get('confidence_basis')}")
    lines.append("")

    steps_by_kind = {s.get("kind"): s for s in document.get("steps") or []}
    for bucket_key, heading in _BUCKET_HEADINGS:
        kinds = summary.get(bucket_key) or []
        if not kinds:
            continue
        lines.append(heading)
        for kind in kinds:
            step = steps_by_kind.get(kind) or {}
            reason = step.get("reason")
            suffix = f" -- {reason}" if reason else ""
            lines.append(f"  - {kind}{suffix}")
        lines.append("")

    if summary.get("caveats"):
        lines.append("CAVEATS")
        for caveat in summary["caveats"]:
            lines.append(f"  - {caveat}")
        lines.append("")

    lines.append(f"baseline_run_id: {document.get('baseline_run_id')}")
    lines.append(f"candidate_run_id: {document.get('candidate_run_id')}")
    if document.get("source_experiment_id"):
        lines.append(f"source_experiment_id: {document['source_experiment_id']}  "
                     f"case_id: {document.get('case_id')}")
    return "\n".join(lines).rstrip("\n") + "\n"


# =========================================================================================== resolution ==

def _validate_args(args) -> None:
    from clozn.cli import main as ctx
    has_experiment = bool(args.experiment_result)
    has_run_pair = bool(args.baseline_run or args.candidate_run)
    if has_experiment and has_run_pair:
        raise ctx.CloznError(
            "pass either an experiment result (with --case) or --baseline-run/--candidate-run, not both")
    if not has_experiment and not has_run_pair:
        raise ctx.CloznError(
            "need either an experiment result positional argument with --case, or "
            "--baseline-run and --candidate-run")
    if has_experiment:
        if not args.case:
            raise ctx.CloznError("--case is required when triaging from an experiment result")
    else:
        if not args.baseline_run or not args.candidate_run:
            raise ctx.CloznError("both --baseline-run and --candidate-run are required")
        if args.case or args.variant or args.seed is not None:
            raise ctx.CloznError(
                "--case/--variant/--seed apply only when triaging from an experiment result")


def _resolve_run_pair_from_ids(args):
    from clozn.cli import main as ctx
    import clozn.runs.store as runlog

    baseline = runlog.get_run(args.baseline_run)
    if baseline is None:
        raise ctx.CloznError(f"baseline run not found: {args.baseline_run}")
    candidate = runlog.get_run(args.candidate_run)
    if candidate is None:
        raise ctx.CloznError(f"candidate run not found: {args.candidate_run}")
    return baseline, candidate, None, None


def _pick_candidate_variant(manifest: dict, requested: str | None) -> str:
    from clozn.cli import main as ctx
    baseline_variant = manifest["baseline_variant"]
    candidates = [v["name"] for v in manifest["variants"] if v["name"] != baseline_variant]
    if requested:
        if requested not in candidates:
            raise ctx.CloznError(
                f"--variant {requested!r} is not a candidate variant in this experiment "
                f"(baseline is {baseline_variant!r}; candidates: {candidates})")
        return requested
    if len(candidates) == 1:
        return candidates[0]
    raise ctx.CloznError(
        f"this experiment has {len(candidates)} candidate variants {candidates}; pass --variant to pick one")


def _pick_seed(baseline_cells: list, candidate_cells: list, requested: int | None) -> int:
    """Prefer a REGRESSION seed (baseline passed, candidate did not) -- that is the pair worth triaging.
    Falls back to the smallest seed common to both sides when nothing regressed (e.g. triaging a
    target_gain or an unscored comparison out of curiosity). Never picks a seed missing on either side."""
    from clozn.cli import main as ctx
    by_seed_baseline = {c["seed"]: c for c in baseline_cells}
    by_seed_candidate = {c["seed"]: c for c in candidate_cells}
    common = sorted(set(by_seed_baseline) & set(by_seed_candidate))
    if not common:
        raise ctx.CloznError("no seed has both a baseline and a candidate cell for this case/variant")
    if requested is not None:
        if requested not in common:
            raise ctx.CloznError(
                f"seed {requested} has no matching baseline/candidate cell pair; available: {common}")
        return requested
    for seed in common:
        base_cell, cand_cell = by_seed_baseline[seed], by_seed_candidate[seed]
        if base_cell.get("status") == "pass" and cand_cell.get("status") != "pass":
            return seed
    return common[0]


def _resolve_run_pair_from_experiment(args):
    from clozn.cli import main as ctx
    from clozn.experiments import suite as experiment_suite

    try:
        result = experiment_suite.load_result(args.experiment_result)
    except experiment_suite.ManifestError as exc:
        raise ctx.CloznError(f"could not load experiment result: {exc}") from None

    manifest = result["manifest"]
    candidate_variant = _pick_candidate_variant(manifest, args.variant)
    baseline_variant = manifest["baseline_variant"]

    baseline_cells = experiment_suite.select_cells(result, case=args.case, variant=baseline_variant)
    candidate_cells = experiment_suite.select_cells(result, case=args.case, variant=candidate_variant)
    if not baseline_cells or not candidate_cells:
        raise ctx.CloznError(
            f"case {args.case!r} has no cells for variant {baseline_variant!r} "
            f"and/or {candidate_variant!r} -- check the case name")

    seed = _pick_seed(baseline_cells, candidate_cells, args.seed)
    baseline_cell = next(c for c in baseline_cells if c["seed"] == seed)
    candidate_cell = next(c for c in candidate_cells if c["seed"] == seed)

    baseline_run = baseline_cell.get("run")
    candidate_run = candidate_cell.get("run")
    if not isinstance(baseline_run, dict) or not isinstance(candidate_run, dict):
        raise ctx.CloznError(
            f"case {args.case!r} seed {seed} is missing embedded run evidence to triage "
            f"(baseline status={baseline_cell.get('status')!r}, "
            f"candidate status={candidate_cell.get('status')!r})")
    return baseline_run, candidate_run, result.get("experiment_id"), args.case


# =================================================================================================== CLI ==

def _write_artifact(path: str, document: dict, *, force: bool) -> None:
    from pathlib import Path
    from clozn.cli import main as ctx
    from clozn._io import atomic_write_json

    target = Path(path).expanduser().resolve()
    if target.exists() and not force:
        raise ctx.CloznError(f"refusing to overwrite {target}; pass --force")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(target), document, indent=2, ensure_ascii=False)


def cmd_triage(args) -> int:
    from clozn.cli import main as ctx

    _validate_args(args)
    if args.experiment_result:
        baseline_run, candidate_run, source_experiment_id, case_id = _resolve_run_pair_from_experiment(args)
    else:
        baseline_run, candidate_run, source_experiment_id, case_id = _resolve_run_pair_from_ids(args)

    families = None
    if args.steps:
        families = [item.strip() for item in args.steps.split(",") if item.strip()]
        unknown = sorted(set(families) - set(ALL_FAMILIES))
        if unknown:
            raise ctx.CloznError(f"unknown --steps value(s) {unknown}; know: {list(ALL_FAMILIES)}")

    try:
        document = build_triage_artifact(
            baseline_run=baseline_run, candidate_run=candidate_run,
            source_experiment_id=source_experiment_id, case_id=case_id,
            families=families, deep=args.deep, max_runs=args.max_runs, max_seconds=args.max_seconds,
        )
    except ValueError as exc:
        raise ctx.CloznError(str(exc)) from None

    if args.out:
        _write_artifact(args.out, document, force=bool(args.force))

    if args.json:
        print(json.dumps(document, indent=2, ensure_ascii=False))
    else:
        print(format_triage(document), end="")
        if args.out:
            print(f"triage artifact written - {args.out}")
    return 0


def add_subparser(sub):
    parser = sub.add_parser(
        "triage", help="evidence-ladder regression triage over a baseline/candidate run pair -- never "
                       "names a cause without a controlled intervention proving it (roadmap feature 05)")
    parser.add_argument(
        "experiment_result", nargs="?", default=None, metavar="RESULT.json",
        help="a clozn.experiment.result.v0 artifact (feature 04's schema) to resolve --case from, "
             "instead of --baseline-run/--candidate-run")
    parser.add_argument("--case", default=None,
                        help="experiment case name (requires the RESULT.json positional)")
    parser.add_argument("--variant", default=None,
                        help="candidate variant name, needed only if the experiment has more than one")
    parser.add_argument("--seed", type=int, default=None,
                        help="exact seed to triage; default prefers a seed where the candidate regressed")
    parser.add_argument("--baseline-run", default=None,
                        help="exact baseline run id (alternative to an experiment result)")
    parser.add_argument("--candidate-run", default=None, help="exact candidate run id")
    parser.add_argument(
        "--steps", default=None, metavar="FAMILY[,FAMILY...]",
        help="comma-separated step families to run/report (default: all -- " + ", ".join(ALL_FAMILIES) + ")")
    parser.add_argument(
        "--deep", action="store_true",
        help="request GPU-touching steps (controlled replay, quant/export, tool contract, internal "
             "localization); not implemented in this build -- still reported as an explicit, reasoned "
             "not_run rather than silently ignored")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="budget: maximum model runs a deep step may spend (not enforced by any "
                             "step this build executes; recorded for forward compatibility)")
    parser.add_argument("--max-seconds", type=float, default=None,
                        help="budget: maximum wall-clock seconds a deep step may spend (same caveat)")
    parser.add_argument("--out", default=None, help="write the clozn.triage.v1 artifact JSON to this path")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it already exists")
    parser.add_argument("--json", action="store_true",
                        help="print the artifact JSON instead of the evidence-ladder text")
    parser.set_defaults(fn=cmd_triage)
    return parser
