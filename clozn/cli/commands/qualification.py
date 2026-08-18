"""Qualification planner and the explicit Q3 core runner entry point.

``--plan`` remains model-free.  ``--run`` is opt-in: it starts the normal product runtime for one
portable core smoke and writes a receipt.  It does not install artifacts or claim Q5-Q8 lab support.
"""
from __future__ import annotations

import json

from clozn._io import atomic_write_json
from clozn.qualification import planner

CLOZN_AUTOLOAD = True


def _error(message: str):
    from clozn.cli.main import CloznError
    raise CloznError(message)


def cmd_qualify(args) -> int:
    if getattr(args, "run", False):
        from clozn.qualification import pipeline
        output = args.out or pipeline.default_run_path(args.model)
        try:
            report = pipeline.run_core(
                args.model,
                output=output,
                live=True,
                smoke_timeout=args.smoke_timeout,
            )
        except Exception as exc:
            _error(f"could not run core qualification: {type(exc).__name__}: {exc}")
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(f"clozn core qualification: {report['model']['input']}")
            print(f"  status: {report['claims']['qualification_status']}")
            for step in report["steps"]:
                suffix = f" -- {step['reason']}" if step.get("reason") else ""
                print(f"  [{step['boundary']}] {step['id']}: {step['status']}{suffix}")
            print("  Q5 J-lens, Q6 batteries, and Q7 installation remain separate steps")
            print(f"  receipt: {output}")
        return 0 if report["claims"]["qualification_status"] == "core_passed" else 1
    if not args.plan:
        _error("qualification execution is not available without `--run`; use `clozn qualify MODEL --plan` for a "
               "model-free readiness plan")
    try:
        report = planner.plan_from_model(
            args.model,
            vram_gb=args.vram,
            context=args.context,
            timeout=args.timeout,
        )
    except Exception as exc:
        _error(f"could not build qualification plan: {type(exc).__name__}: {exc}")
    output = args.out or planner.default_plan_path(report)
    atomic_write_json(output, report, indent=2, ensure_ascii=False)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        model = report["model"]
        summary = model.get("summary") or {}
        print(f"clozn qualification plan: {model['input']}")
        if summary:
            print("  " + "  ".join(f"{key}={value}" for key, value in summary.items()))
        resources = report["resources"]
        fit = resources.get("vram_fits_estimate")
        print(f"  estimated resources: disk={resources.get('disk_gb') or '?'} GB, "
              f"VRAM={resources.get('estimated_vram_gb') or '?'} GB "
              f"(budget {resources.get('vram_budget_gb')} GB, fits={fit})")
        for step in report["steps"]:
            suffix = f" -- {step['reason']}" if step.get("reason") else ""
            print(f"  [{step['boundary']}] {step['id']}: {step['status']}{suffix}")
        print("  qualification: NOT QUALIFIED (planning only)")
        print(f"  plan: {output}")
    return 0


def add_subparser(sub):
    parser = sub.add_parser("qualify", help="plan or run model qualification")
    parser.add_argument("model", help="local GGUF path, known model name, or GGUF URL")
    parser.add_argument("--plan", action="store_true", help="emit the model-free qualification plan")
    parser.add_argument("--run", action="store_true", help="run the Q3 core smoke through the product gateway")
    parser.add_argument("--vram", type=float, default=16.0, help="VRAM budget in GB (default 16)")
    parser.add_argument("--context", type=int, default=8192,
                        help="context size for the resource estimate (default 8192)")
    parser.add_argument("--timeout", type=float, default=30.0, help="header URL timeout in seconds")
    parser.add_argument("--smoke-timeout", type=float, default=180.0,
                        help="worker/request timeout for --run (default 180 seconds)")
    parser.add_argument("--out", default=None, help="plan JSON path (default ~/.clozn/qualification-plans)")
    parser.add_argument("--json", action="store_true", help="also print the full plan JSON")
    parser.set_defaults(fn=cmd_qualify)


__all__ = ["add_subparser", "cmd_qualify"]
