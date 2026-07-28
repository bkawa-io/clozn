"""CLI for the versioned model-developer experiment object."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from clozn._io import atomic_write_json
from clozn.cli.main import CloznError
from clozn.experiments import stats, suite


def add_subparser(sub):
    parser = sub.add_parser("experiment", help="run or inspect a case x variant x seed experiment manifest")
    commands = parser.add_subparsers(dest="experiment_cmd")
    parser.set_defaults(fn=_no_command)

    run = commands.add_parser("run", help="run target and guard suites across every variant and seed")
    run.add_argument("manifest", help="path to a clozn.experiment.v0 JSON manifest")
    run.add_argument("--url", default=suite.DEFAULT_URL, help="default Clozn gateway URL (default :8080)")
    run.add_argument("--seeds", type=int, default=None, help="override the manifest with seeds 0..N-1")
    run.add_argument("--out", default=None, help="result path (default ~/.clozn/experiments/<id>.json)")
    run.add_argument("--vcs-repository", help="explicit repository identity to record; never inferred")
    run.add_argument("--vcs-commit", help="explicit commit/revision to record; never inferred")
    run.add_argument("--vcs-branch", help="explicit branch to record; never inferred")
    run.add_argument("--workflow-url", help="explicit workflow URL to record")
    run.add_argument("--artifact-url", help="explicit artifact URL to record")
    run.add_argument("--artifact-expires-at", help="explicit artifact-link expiry timestamp")
    run.add_argument("--local-open-command", help="explicit local command for opening the stored artifact")
    run.add_argument("--json", action="store_true", help="print the full result JSON")
    run.set_defaults(fn=cmd_run)

    show = commands.add_parser("show", help="inspect summary or matching per-case evidence")
    show.add_argument("result", help="experiment result JSON")
    show.add_argument("--suite", choices=["target", "guard"], default=None)
    show.add_argument("--case", default=None)
    show.add_argument("--variant", default=None)
    show.add_argument("--seed", type=int, default=None)
    show.add_argument("--json", action="store_true")
    show.set_defaults(fn=cmd_show)

    stats_p = commands.add_parser(
        "stats", help="paired bootstrap CIs, seed aggregation, and multiple-comparison honesty over a result")
    stats_p.add_argument("result", help="clozn.experiment.result.v0 JSON artifact")
    stats_p.add_argument("--alpha", type=float, default=stats.DEFAULT_ALPHA,
                         help=f"raw per-comparison alpha before Bonferroni adjustment (default {stats.DEFAULT_ALPHA})")
    stats_p.add_argument("--resamples", type=int, default=stats.DEFAULT_RESAMPLES, dest="resamples",
                         help=f"bootstrap resample count (default {stats.DEFAULT_RESAMPLES})")
    stats_p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed, for an exactly "
                         "reproducible report (default 0)")
    stats_p.add_argument("--json", action="store_true", help="print the full stats report as JSON")
    stats_p.set_defaults(fn=cmd_stats)

    export = commands.add_parser("export", help="export a completed experiment to EEE/HF or an eval runner")
    export.add_argument("result", help="clozn.experiment.result.v0 JSON artifact")
    export.add_argument("--format", choices=("eee", "hf-community", "promptfoo", "inspect", "lighteval"),
                        default="eee", help="destination format (default: eee)")
    export.add_argument("--out", required=True, help="new local output directory")
    export.add_argument("--organization", help="organization/person reporting the eval (required for EEE/HF)")
    export.add_argument("--relationship", choices=("first_party", "third_party", "collaborative", "other"),
                        help="evaluator's relationship to the model (required for EEE/HF)")
    export.add_argument("--model-id", action="append", default=[], metavar="VARIANT=OWNER/REPO",
                        help="explicit public model identity; repeat for multiple variants")
    export.add_argument("--hf-dataset", metavar="OWNER/DATASET",
                        help="registered HF Benchmark dataset (required for hf-community)")
    export.add_argument("--target-task", help="HF Benchmark task_id for the target suite")
    export.add_argument("--guard-task", help="HF Benchmark task_id for the guard suite")
    export.add_argument("--provider", help="Promptfoo provider id; omitted configs carry a blocking issue")
    export.add_argument("--dataset-uri", help="LightEval dataset URI to place in the task skeleton")
    export.add_argument("--retrieved-timestamp", type=float,
                        help="override EEE record creation time with a Unix epoch")
    export.add_argument("--force", action="store_true", help="replace the exact existing output path")
    export.add_argument("--json", action="store_true", help="print the machine-readable export receipt")
    export.set_defaults(fn=cmd_export)
    return parser


def _no_command(_args):
    print("clozn experiment: use `clozn experiment run`, `show`, or `export`")
    return 2


def cmd_run(args):
    try:
        manifest = suite.load_manifest(args.manifest)
        vcs = {
            field: value for field, value in (
                ("repository", getattr(args, "vcs_repository", None)),
                ("commit", getattr(args, "vcs_commit", None)),
                ("branch", getattr(args, "vcs_branch", None)),
            ) if value
        } or None
        provenance = {
            field: value for field, value in (
                ("workflow_url", getattr(args, "workflow_url", None)),
                ("artifact_url", getattr(args, "artifact_url", None)),
                ("expires_at", getattr(args, "artifact_expires_at", None)),
                ("local_open_command", getattr(args, "local_open_command", None)),
            ) if value
        } or None
        result = suite.run_manifest(
            manifest, default_url=args.url, seeds_override=args.seeds,
            vcs=vcs, artifact_provenance=provenance,
        )
    except suite.ManifestError as exc:
        raise CloznError(str(exc)) from exc
    path = args.out or suite.default_result_path(result)
    atomic_write_json(path, result, indent=2, ensure_ascii=False)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(suite.format_summary(result))
        print(f"  result: {path}")
    return 1 if any(c.get("status") == "error" for c in result["cells"]) else 0


def cmd_show(args):
    try:
        result = suite.load_result(args.result)
    except suite.ManifestError as exc:
        raise CloznError(f"could not read experiment result: {exc}") from exc
    filtered = any(v is not None for v in (args.suite, args.case, args.variant, args.seed))
    if args.json:
        payload = suite.select_cells(result, suite=args.suite, case=args.case, variant=args.variant, seed=args.seed) if filtered else result
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif filtered:
        print(suite.format_cells(suite.select_cells(result, suite=args.suite, case=args.case,
                                                    variant=args.variant, seed=args.seed)))
    else:
        print(suite.format_summary(result))
    return 0


def cmd_stats(args):
    try:
        result = suite.load_result(args.result)
    except suite.ManifestError as exc:
        raise CloznError(f"could not read experiment result: {exc}") from exc
    if not math.isfinite(args.alpha) or not (0.0 < args.alpha < 1.0):
        raise CloznError("--alpha must be a finite number strictly between 0 and 1")
    if args.resamples < 100:
        raise CloznError("--resamples must be at least 100 for a stable bootstrap")
    report = stats.stats_report(result, alpha=args.alpha, n_resamples=args.resamples, seed=args.seed)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(stats.format_stats_report(report))
    return 0


def _model_id_map(values: list[str]) -> dict[str, str]:
    result = {}
    for raw in values:
        variant, separator, model_id = raw.partition("=")
        variant, model_id = variant.strip(), model_id.strip()
        if not separator or not variant or not model_id:
            raise CloznError("--model-id must be VARIANT=OWNER/REPO")
        if variant in result:
            raise CloznError(f"duplicate --model-id for variant {variant!r}")
        result[variant] = model_id
    return result


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _write_export_files(stage: Path, export_format: str, payload: dict) -> list[Path]:
    files = []
    if export_format in {"eee", "hf-community"}:
        bundle_path = stage / "clozn-evidence-bundle.json"
        _write_json(bundle_path, payload); files.append(bundle_path)
        for aggregate in payload["aggregates"]:
            path = stage / "eee" / aggregate["filename"]
            _write_json(path, aggregate["record"]); files.append(path)
        from clozn.eval.exports import canonical_jsonl
        for filename, records in payload["instances"].items():
            path = stage / "eee" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(canonical_jsonl(records), encoding="utf-8"); files.append(path)
        for preview in payload["hf_community"]:
            model_dir = preview["model_id"].replace("/", "--")
            path = stage / "hf-community" / model_dir / preview["filename"]
            # JSON is valid YAML 1.2. This keeps Clozn stdlib-only while producing the exact list/object
            # shape accepted from `.eval_results/*.yaml`; it is a preview and is never pushed here.
            _write_json(path, preview["records"]); files.append(path)
    else:
        main_name = "promptfooconfig.json" if export_format == "promptfoo" else f"{export_format}-adapter.json"
        main = stage / main_name
        _write_json(main, payload); files.append(main)
        records = payload.get("records")
        if isinstance(records, list):
            from clozn.eval.exports import canonical_jsonl
            rows = stage / f"{export_format}-records.jsonl"
            rows.write_text(canonical_jsonl(records), encoding="utf-8"); files.append(rows)
    return files


def _safe_replace_directory(stage: Path, target: Path, *, force: bool) -> None:
    target = target.resolve()
    root = Path(target.anchor).resolve()
    if target == root or target == Path.home().resolve():
        raise CloznError("refusing to use a filesystem root or home directory as export output")
    if target.exists() and not force:
        raise CloznError(f"export output already exists: {target} (pass --force to replace that exact path)")
    backup = None
    if target.exists():
        backup = target.parent / f".{target.name}.clozn-backup-{uuid.uuid4().hex}"
        if backup.resolve().parent != target.parent.resolve():
            raise CloznError("could not verify the export backup path")
        os.replace(target, backup)
    try:
        os.replace(stage, target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def _materialize_export(target_value: str, export_format: str, payload: dict, *, force: bool) -> dict:
    target = Path(target_value).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{target.name}.clozn-stage-", dir=target.parent))
    try:
        files = _write_export_files(stage, export_format, payload)
        receipt_files = []
        for path in sorted(files):
            data = path.read_bytes()
            receipt_files.append({"path": path.relative_to(stage).as_posix(), "bytes": len(data),
                                  "sha256": hashlib.sha256(data).hexdigest()})
        receipt = {"schema_version": "clozn.eval_export.receipt.v1", "format": export_format,
                   "source": payload.get("source"), "files": receipt_files,
                   "issues": payload.get("issues") or []}
        _write_json(stage / "export-receipt.json", receipt)
        _safe_replace_directory(stage, target, force=force)
        receipt["output"] = str(target)
        return receipt
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def cmd_export(args):
    from clozn.eval import exports

    try:
        result = suite.load_result(args.result)
        if args.format in {"eee", "hf-community"}:
            if not args.organization or not args.relationship:
                raise exports.EvalExportError("--organization and --relationship are required for EEE/HF exports")
            hf_benchmark = None
            if args.format == "hf-community":
                missing = [name for name, value in (("--hf-dataset", args.hf_dataset),
                                                     ("--target-task", args.target_task),
                                                     ("--guard-task", args.guard_task)) if not value]
                if missing:
                    raise exports.EvalExportError("hf-community requires " + ", ".join(missing))
                hf_benchmark = {"dataset_id": args.hf_dataset, "target_task_id": args.target_task,
                                "guard_task_id": args.guard_task}
            payload = exports.export_community_bundle(
                result, source_organization=args.organization, evaluator_relationship=args.relationship,
                model_ids=_model_id_map(args.model_id), hf_benchmark=hf_benchmark,
                retrieved_timestamp=args.retrieved_timestamp,
            )
            if args.format == "hf-community" and not payload["hf_community"]:
                raise exports.EvalExportError("no Community Eval previews were exportable; provide OWNER/REPO model ids")
        else:
            payload = exports.export_adapter(result, args.format, provider=args.provider,
                                             dataset_uri=args.dataset_uri)
        receipt = _materialize_export(args.out, args.format, payload, force=args.force)
    except (suite.ManifestError, exports.EvalExportError, OSError, ValueError) as exc:
        raise CloznError(f"experiment export failed: {exc}") from exc
    if args.json:
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
    else:
        print(f"exported {args.format} -> {receipt['output']}")
        print(f"  {len(receipt['files'])} data file(s); {len(receipt['issues'])} review issue(s)")
        if receipt["issues"]:
            print("  inspect export-receipt.json before running or publishing")
    return 0
