"""``clozn setup`` -- install and verify a matching native engine (roadmap feature 01), plus
``setup status``/``setup upgrade``/``setup rollback``.

Registered via CLOZN_AUTOLOAD (docs/SEAMS.md Seam 1) -- main.py needs no edit for this file to exist.
This module is the ONLY thing translating clozn/setup's plain SetupError hierarchy into
clozn.cli.main.CloznError (one clean line, no traceback) and rendering its result dicts as human text or
`--json`; every actual install/upgrade/rollback/status operation lives in clozn/setup/install.py.
"""
from __future__ import annotations

import json as json_mod

from clozn.setup import install as setup_install
from clozn.setup import manifest as setup_manifest
from clozn.setup.errors import SetupError

CLOZN_AUTOLOAD = True


def add_subparser(sub):
    p = sub.add_parser("setup", help="install and verify a matching native engine")
    p.add_argument("--backend", choices=setup_manifest.BACKEND_CHOICES, default="auto",
                  help="which backend to install (default: auto-detect from this machine)")
    p.add_argument("--version", default=None,
                  help="require this exact engine version (fails if the manifest publishes a different one)")
    p.add_argument("--dry-run", action="store_true", help="show the plan; change nothing on disk")
    p.add_argument("--force", action="store_true",
                  help="reinstall even if this version+platform is already present")
    p.add_argument("--json", action="store_true", help="print machine-readable output")
    p.set_defaults(fn=cmd_setup, setup_cmd=None)

    setup_sub = p.add_subparsers(dest="setup_cmd")

    status = setup_sub.add_parser("status", help="report installed/active managed-engine state")
    status.add_argument("--json", action="store_true")

    upgrade = setup_sub.add_parser(
        "upgrade", help="install a newer engine, keeping the current one for rollback")
    upgrade.add_argument("--backend", choices=setup_manifest.BACKEND_CHOICES, default="auto")
    upgrade.add_argument("--version", default=None)
    upgrade.add_argument("--dry-run", action="store_true")
    upgrade.add_argument("--json", action="store_true")

    rollback = setup_sub.add_parser("rollback", help="restore the previously active engine")
    rollback.add_argument("--json", action="store_true")


def cmd_setup(args):
    from clozn.cli import main as ctx
    sub = getattr(args, "setup_cmd", None)
    try:
        if sub == "status":
            return _cmd_status(args, ctx)
        if sub == "upgrade":
            return _cmd_install_or_upgrade(args, ctx, label="upgrade")
        if sub == "rollback":
            return _cmd_rollback(args, ctx)
        return _cmd_install_or_upgrade(args, ctx, label="setup")
    except SetupError as error:
        raise ctx.CloznError(str(error)) from None


# ------------------------------------------------------------------------------------------ setup / upgrade

def _cmd_install_or_upgrade(args, ctx, *, label: str):
    backend = getattr(args, "backend", "auto")
    version = getattr(args, "version", None)
    if getattr(args, "dry_run", False):
        plan = setup_install.plan_install(backend_pref=backend, version=version, home=ctx.HOME)
        return _emit_plan(plan, args.json, label=label)
    result = setup_install.run_install(
        backend_pref=backend, version=version, home=ctx.HOME, force=getattr(args, "force", False))
    return _emit_install_result(result, args.json, label=label)


def _emit_plan(plan: dict, as_json: bool, *, label: str) -> int:
    if as_json:
        print(json_mod.dumps({"action": "plan", **plan}, indent=2, sort_keys=True, default=str))
        return 0
    artifact = plan["artifact"]
    cuda_suffix = str(artifact["cuda_major"]) if artifact.get("cuda_major") else ""
    print(f"clozn {label} (--dry-run): would install engine {plan['engine_version']} "
         f"({artifact['os']}/{artifact['arch']}/{artifact['backend']}{cuda_suffix})")
    print(f"  manifest:          {plan['manifest_url']}")
    print(f"  install key:       {plan['install_key']}")
    print(f"  target directory:  {plan['target_dir']}")
    print(f"  already installed: {plan['already_installed']}")
    print(f"  currently active:  {plan['currently_active'] or '(none)'}")
    print(f"  would change active: {plan['would_change_active']}")
    print("  no disk state was changed (--dry-run)")
    return 0


def _emit_install_result(result: dict, as_json: bool, *, label: str) -> int:
    record = result.get("record") or {}
    states = setup_install.four_state_report(
        engine_exe=record.get("entrypoint"),
        discovery_source="managed",   # every run_install() result is, by definition, a managed install
        backend=record.get("backend"),
        qualification=record.get("qualification"),
    )
    if as_json:
        print(json_mod.dumps({**result, "states": states}, indent=2, sort_keys=True, default=str))
        return 0
    action = result["action"]
    if action == "noop_already_active":
        print(f"clozn {label}: {result['install_key']} is already installed and active. Nothing to do.")
    elif action == "activated_existing_install":
        print(f"clozn {label}: {result['install_key']} was already installed; activated it.")
    else:
        print(f"clozn {label}: installed and activated {result['install_key']}")
    engine_state = states["compatible_engine_installed"]
    print(f"  compatible engine installed: {engine_state['status']}")
    if engine_state["status"] in ("found_but_not_launchable", "found_but_not_qualified"):
        print(f"    WARNING: {engine_state.get('build_identity_check', {}).get('error')}")
    print(f"  core inference qualification: {states['core_inference_qualification']['status']} "
         f"({states['core_inference_qualification']['reason']})")
    print(f"  white-box qualification: {states['white_box_qualification']['status']} "
         f"({states['white_box_qualification']['reason']})")
    return 0


# --------------------------------------------------------------------------------------------------- status

def _cmd_status(args, ctx):
    status = setup_install.read_status(home=ctx.HOME)
    if args.json:
        print(json_mod.dumps(status, indent=2, sort_keys=True, default=str))
        return 0
    if status["active_key"] is None:
        print("clozn setup status: no managed engine is installed. Run `clozn setup` to install one.")
        return 0
    active = status["active"] or {}
    print(f"active:   {status['active_key']}  (backend={active.get('backend')}, "
         f"protocol={active.get('protocol_version', '?')})")
    if status["previous_key"]:
        print(f"previous: {status['previous_key']}  (available via `clozn setup rollback`)")
    else:
        print("previous: (none)")
    print(f"installed ({len(status['installed'])}):")
    for entry in status["installed"]:
        marker = "*" if entry["key"] == status["active_key"] else " "
        print(f"  {marker} {entry['key']}")
    return 0


# ------------------------------------------------------------------------------------------------ rollback

def _cmd_rollback(args, ctx):
    result = setup_install.run_rollback(home=ctx.HOME)
    if args.json:
        print(json_mod.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    print(f"clozn setup rollback: {result['rolled_back_from']} -> {result['active']}")
    return 0
