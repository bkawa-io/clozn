"""`clozn adopt ollama` -- try one model from an existing Ollama install without deleting Ollama,
redownloading multi-gigabyte weights, or permanently reconfiguring applications
(notes/agent_roadmap/11-adopt-ollama.md).

WHAT THIS IS NOT: this is not clozn/server/routes/ollama.py's story. That module makes Clozn *answer*
Ollama's wire protocol so an Ollama-speaking client can point at Clozn (already built, CI-tested against
the real `ollama` Python/JS SDKs). This module goes the other direction -- Clozn as a read-only *client*
of a real Ollama install -- and is what "adopt" actually means.

HARD SAFETY RULE, ENFORCED THROUGHOUT: nothing here ever writes, deletes, renames, or re-tags anything
under Ollama's own storage or reachable through Ollama's own mutating API. `clozn.adopt.ollama_discovery`
only reads `/api/version`, `/api/tags`, `/api/show`, and the on-disk storage layout. This module only
ever CREATES a new entry under Clozn's own model directory (`~/.clozn/models/`, the same directory
clozn.models.inventory.model_dirs() already searches -- an adopted model is discoverable by `clozn
models`/`clozn run` with zero changes to that code) and a transaction record under `~/.clozn/adopt/`.

REUSE STRATEGY -- WHAT THIS RELEASE IMPLEMENTS OF THE SPEC'S 4-ITEM LADDER
----------------------------------------------------------------------------------------------------
    1. Register an external immutable blob path as read-only, no new file at all.
    2. Hard link when supported and on the same volume.                              <- default here
    3. Symlink only when platform permissions and lifecycle are safe.
    4. Copy only with explicit --copy and a disk-space warning.                        <- --copy here

Only #2 and #4 are implemented. #1 would require clozn.models.inventory/resolve_model to understand a
new "externally registered, no local file" entry -- real new surface in shared model-resolution code
this feature's plan deliberately did not scope in, so it is not done here. #3 (symlink) is skipped on
purpose: Windows symlink creation commonly needs elevated privileges or Developer Mode, and a strategy
that silently fails half the time on the platform this was written on is worse than not offering it --
when a hard link cannot be created, the error names --copy as the explicit next step rather than falling
back to a symlink attempt that might just as easily fail the same way.

THE "GC CASE" (an Ollama-side blob deleted or replaced after adoption)
----------------------------------------------------------------------------------------------------
For `mode == "hard_link"`, Clozn's file and Ollama's file are the SAME inode: as long as at least one
directory entry references it, the bytes survive. `_undo`'s safety check is therefore not "does the
recorded source path still exist" (a guess about Ollama's layout, which this feature's plan already
flags as unverified) -- it is `os.stat(target).st_nlink`, the filesystem's own, always-correct answer to
"is Clozn's copy the last reference to these bytes." `st_nlink <= 1` means Ollama's own reference is
already gone (removed, GC'd, or Ollama itself uninstalled) and undo refuses outright rather than delete
the last surviving copy. `mode == "copy"` never needs this check: a copy is a fully independent set of
bytes, so removing it can never affect anything Ollama-side.

CLOZN_AUTOLOAD = True
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from clozn import schemas
from clozn._io import atomic_write_json
from clozn.adopt import ollama_discovery as discovery

CLOZN_AUTOLOAD = True

_SCHEMA = "clozn.adopt-ollama.v1"
_TEMPLATE_NOTE = (
    "template/parameter translation is not implemented in this release; Ollama's exact template, "
    "system prompt, stop sequences, and sampling parameters are not reproduced")
_FROM_LINE = re.compile(r"^\s*FROM\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(name: str) -> str:
    out = [ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(name)]
    return "".join(out) or "model"


def _adopt_dir() -> Path:
    from clozn.cli import main as ctx
    return Path(ctx.HOME) / "adopt"


def _transaction_path(registered_name: str) -> Path:
    return _adopt_dir() / f"{_slug(registered_name)}.json"


def _clozn_models_dir() -> Path:
    from clozn.cli import main as ctx
    return Path(ctx.HOME) / "models"


def _find_model(models: list[dict], wanted: str) -> dict:
    """Never a silent pick among multiple matches -- mirrors clozn.cli.commands.models.resolve_model's
    own discipline (see that module's docstring) rather than reinventing a looser convention here."""
    def _name_of(entry):
        return str(entry.get("name") or entry.get("model") or "")

    exact = [m for m in models if _name_of(m) == wanted]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(f"'{wanted}' matched more than one entry in Ollama's own /api/tags response; "
                         f"this should not happen -- please report it")
    fuzzy = [m for m in models if wanted.casefold() in _name_of(m).casefold()]
    if len(fuzzy) == 1:
        return fuzzy[0]
    names = ", ".join(sorted({_name_of(m) for m in models})) or "(none)"
    if len(fuzzy) > 1:
        raise ValueError(f"'{wanted}' is ambiguous among Ollama's models: {names}. Be more specific.")
    raise ValueError(f"'{wanted}' not found among Ollama's models: {names}")


def _resolve_source_blob_path(modelfile_text) -> str | None:
    """Best-effort parse of the FIRST `FROM <path>` line in Ollama's own /api/show 'modelfile' text --
    the one documented field that can name a local blob (see this module's own docstring: this is a
    best-effort text parse of one structured API field, not a scrape of an undocumented format). Returns
    None (never raises) when there is no FROM line, the target isn't an absolute path (a bare `FROM
    llama3:latest` names another Ollama tag, not a blob), or the path doesn't exist locally -- all three
    are the honest 'no local blob' case, not an error."""
    if not modelfile_text:
        return None
    match = _FROM_LINE.search(str(modelfile_text))
    if not match:
        return None
    candidate = match.group(1).strip().strip('"')
    if not os.path.isabs(candidate) or not os.path.isfile(candidate):
        return None
    return os.path.abspath(candidate)


def _read_gguf_header(blob_path: str):
    from clozn.cli import fit_planner
    try:
        return fit_planner.gguf_header_from_path(blob_path), None
    except Exception as exc:
        return None, f"{blob_path} could not be parsed as a GGUF file: {exc}"


def _engine_capability(header: dict | None) -> dict | None:
    if header is None:
        return None
    out = {"level": "discovered"}
    if header.get("arch"):
        out["architecture"] = header["arch"]
    return out


# ------------------------------------------------------------------------------------------ discovery-only

def _describe_setup(args) -> dict:
    """`clozn adopt ollama` with no --model: a preview -- what's found, nothing adopted. Mirrors the
    spec's own worked example output ('Found your Ollama setup / Models / Applications')."""
    from clozn.cli.commands._connector import AiderConnector

    disco = discovery.discover(host_override=args.host)
    report = {"status": "described", "discovery": disco, "models": [], "applications": []}
    if disco["found"] and disco["source"] in ("env", "endpoint"):
        try:
            raw = discovery.list_models(disco["host"])
        except Exception as exc:
            report["models_error"] = f"found Ollama's API but listing models failed: {exc}"
            raw = []
        for entry in raw:
            details = entry.get("details") or {}
            report["models"].append({
                "name": entry.get("name") or entry.get("model"),
                "size_bytes": entry.get("size"),
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
            })
    aider = AiderConnector().detect()
    report["applications"].append({
        "app": "aider", "detected": aider.installed,
        "connector_available": aider.installed, "note": aider.note})
    return report


# ------------------------------------------------------------------------------------------------ dry-run/apply

def _build_plan(args) -> dict:
    """Discover Ollama, locate the named model, and assemble everything a dry-run/apply needs. Never
    writes anything -- pure read (network + local filesystem reads only)."""
    disco = discovery.discover(host_override=args.host)
    if not disco["found"]:
        raise ValueError("no Ollama installation found. " +
                         ("; ".join(disco["warnings"]) if disco["warnings"] else
                          "checked OLLAMA_HOST, the 'ollama' executable, the default local API, and "
                          "known storage locations"))
    if disco["source"] not in ("env", "endpoint"):
        raise ValueError(
            f"Ollama was found via {disco['source']} but its API is not reachable, so models can't be "
            f"listed or read. Start Ollama (e.g. `ollama serve`) and try again."
            + ("  " + "; ".join(disco["warnings"]) if disco["warnings"] else ""))

    host = disco["host"]
    models = discovery.list_models(host)
    entry = _find_model(models, args.model)
    model_name = entry.get("name") or entry.get("model") or args.model
    shown = discovery.show_model(host, model_name)

    from clozn.cli.commands._connector import sha256_path
    blob_path = _resolve_source_blob_path(shown.get("modelfile"))
    blocked = None
    source_sha256 = None
    header = None
    if blob_path is None:
        blocked = ("no local GGUF blob could be resolved for this model (it may be a cloud-only/remote "
                  "model, or its modelfile does not FROM an absolute local path). Clozn can only adopt "
                  "weights it can read locally.")
    else:
        source_sha256 = sha256_path(Path(blob_path))
        header, header_error = _read_gguf_header(blob_path)
        if header is None:
            blocked = f"unsupported model: {header_error}"

    slug = _slug(model_name)
    return {
        "disco": disco, "host": host, "model_name": model_name, "shown": shown,
        "registered_name": f"ollama/{model_name}",
        "blob_path": blob_path, "source_sha256": source_sha256, "header": header,
        "reported_digest": entry.get("digest"),
        "clozn_path": str(_clozn_models_dir() / f"ollama__{slug}.gguf"),
        "mode": "copy" if args.copy else "hard_link",
        "blocked": blocked,
    }


def _dry_run_report(plan: dict) -> dict:
    extra_disk = None
    if plan["blob_path"] and not plan["blocked"]:
        extra_disk = 0 if plan["mode"] == "hard_link" else os.path.getsize(plan["blob_path"])
    return {
        "status": "dry_run",
        "app": "ollama",
        "discovery_source": plan["disco"]["source"],
        "model_name": plan["model_name"],
        "registered_name": plan["registered_name"],
        "reuse_method": None if plan["blocked"] else plan["mode"],
        "source_blob_path": plan["blob_path"],
        "source_blob_sha256": plan["source_sha256"],
        "reported_digest": plan["reported_digest"],
        "clozn_path": plan["clozn_path"],
        "extra_disk_bytes": extra_disk,
        "engine_capability": _engine_capability(plan["header"]),
        "template": {"source": "ollama_modelfile", "exactly_reproduced": False, "warnings": [_TEMPLATE_NOTE]},
        "blocked": plan["blocked"],
        "undo_plan": (
            "hard link: --undo removes only Clozn's own directory entry; the Ollama-side blob is never "
            "touched, and undo refuses automatically if Ollama's own reference to the same data is "
            "already gone (last-copy protection via the filesystem's own hard-link count)."
            if plan["mode"] == "hard_link" else
            "copy: --undo deletes only the independent copy Clozn made; Ollama's own file is never "
            "touched or affected."),
    }


def _apply_connect(args, document: dict) -> dict:
    """Best-effort: never raises, so a --connect failure cannot roll back an already-successful,
    already-recorded model adoption -- the two are independently undoable (see this module's schema
    doc). Returns a status dict; cmd_adopt uses it to decide the process exit code."""
    from clozn.cli import main as ctx
    from clozn.cli.commands._connector import AiderConnector

    if args.connect != "aider":
        return {"status": "failed", "app": args.connect,
                "error": f"connector {args.connect!r} is not implemented yet; supported: aider"}
    connector = AiderConnector()
    detection = connector.detect()
    if not detection.installed:
        return {"status": "failed", "app": "aider",
                "error": f"{detection.note}. Install Aider first, or run `clozn connect aider` "
                        f"manually once it is."}
    state_path = Path(ctx.HOME) / "connect" / "aider.json"
    try:
        transaction = connector.apply(base_url=args.url, model=args.client_model_label,
                                      api_key=args.api_key, state_path=state_path)
    except (OSError, ValueError) as exc:
        return {"status": "failed", "app": "aider", "error": str(exc)}
    document["client_transactions"].append({
        "app": "aider", "status": transaction.report.get("status"),
        "target": transaction.report.get("path"), "backup": transaction.report.get("backup"),
        "state_path": str(state_path)})
    return {"status": "connected", "app": "aider", "target": transaction.report.get("path")}


def _apply(args, plan: dict) -> dict:
    if plan["blocked"]:
        raise ValueError(f"cannot adopt {plan['model_name']!r}: {plan['blocked']}")

    from clozn.cli.commands._connector import atomic_copy_file, sha256_path

    target = Path(plan["clozn_path"])
    tx_path = _transaction_path(plan["registered_name"])

    if tx_path.is_file() and target.is_file():
        try:
            prior = json.loads(tx_path.read_text(encoding="utf-8"))
        except Exception:
            prior = None
        if (isinstance(prior, dict)
                and prior.get("ollama", {}).get("source_blob_sha256") == plan["source_sha256"]
                and sha256_path(target) == prior.get("clozn", {}).get("model_sha256")):
            return {"status": "unchanged", "app": "ollama", "model_name": plan["model_name"],
                   "registered_name": plan["registered_name"], "clozn_path": str(target),
                   "transaction_path": str(tx_path), "connect": None}

    if target.exists() and not tx_path.is_file():
        raise ValueError(f"refusing to write {target}: it already exists and was not created by a "
                         f"previous `clozn adopt` transaction")
    if target.exists() and tx_path.is_file():
        raise ValueError(f"{plan['registered_name']} is already adopted at {target} with different "
                         f"content than what's currently in Ollama; run `clozn adopt ollama --model "
                         f"{plan['model_name']} --undo` first, then adopt again")

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if plan["mode"] == "hard_link":
            try:
                os.link(plan["blob_path"], str(target))
            except OSError as exc:
                size_gb = os.path.getsize(plan["blob_path"]) / 1e9
                raise ValueError(
                    f"could not create a hard link at {target} ({exc}). This usually means the source "
                    f"and Clozn's model directory are on different volumes or filesystems that don't "
                    f"support hard links here. Retry with --copy to make a full copy instead (uses "
                    f"~{size_gb:.1f} GB more disk).") from None
        else:
            atomic_copy_file(Path(plan["blob_path"]), target)

        model_sha256 = sha256_path(target)
        if model_sha256 != plan["source_sha256"]:
            raise ValueError(f"verification failed: {target} does not match the source blob's hash "
                             f"after {plan['mode']}; nothing was left registered")
    except BaseException:
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        raise

    document = {
        "schema_version": _SCHEMA,
        "created_at": _now_iso(),
        "discovery": {k: v for k, v in {
            "source": plan["disco"]["source"], "host": plan["disco"].get("host"),
            "version": plan["disco"].get("version")}.items() if v is not None},
        "ollama": {k: v for k, v in {
            "model_name": plan["model_name"],
            "manifest_digest": plan.get("reported_digest"),
            "source_blob_path": plan["blob_path"],
            "source_blob_sha256": plan["source_sha256"]}.items() if v is not None},
        "clozn": {
            "registered_name": plan["registered_name"], "path": str(target),
            "mode": plan["mode"], "model_sha256": model_sha256},
        "template": {"source": "ollama_modelfile", "exactly_reproduced": False, "warnings": [_TEMPLATE_NOTE]},
        "client_transactions": [],
    }
    capability = _engine_capability(plan["header"])
    if capability is not None:
        document["engine_capability"] = capability

    schemas.validate(document, _SCHEMA)

    try:
        _adopt_dir().mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(tx_path), document, ensure_ascii=False, indent=2, sort_keys=True)
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise

    connect_report = None
    if args.connect:
        connect_report = _apply_connect(args, document)
        if document["client_transactions"]:
            # Best-effort re-save so the receipt reflects the connect step too; a failure here does not
            # unwind the already-successful, already-independently-recorded model adoption or connect
            # transaction (see _apply_connect's own docstring).
            try:
                atomic_write_json(str(tx_path), document, ensure_ascii=False, indent=2, sort_keys=True)
            except OSError:
                pass

    return {"status": "adopted", "app": "ollama", "model_name": plan["model_name"],
           "registered_name": plan["registered_name"], "clozn_path": str(target), "mode": plan["mode"],
           "transaction_path": str(tx_path), "connect": connect_report}


# ---------------------------------------------------------------------------------------------------- undo

def _undo(args) -> dict:
    from clozn.cli.commands._connector import sha256_path

    if args.model:
        tx_path = _transaction_path(f"ollama/{args.model}")
        if not tx_path.is_file():
            raise ValueError(f"no adopt transaction recorded for 'ollama/{args.model}' at {tx_path}")
    else:
        directory = _adopt_dir()
        candidates = sorted(directory.glob("*.json")) if directory.is_dir() else []
        if not candidates:
            raise ValueError("no `clozn adopt ollama` transaction found to undo")
        if len(candidates) > 1:
            names = ", ".join(p.stem for p in candidates)
            raise ValueError(f"more than one adopted model is registered ({names}); pass --model to "
                             f"pick which one to undo")
        tx_path = candidates[0]

    try:
        document = json.loads(tx_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"adopt transaction at {tx_path} is unreadable: {exc}") from None
    schemas.validate(document, _SCHEMA)

    target = Path(document["clozn"]["path"])
    mode = document["clozn"]["mode"]
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"the adopted file no longer exists as a regular file at {target}; refusing "
                         f"to guess -- remove the stale transaction record manually if you are sure: "
                         f"{tx_path}")

    current_sha256 = sha256_path(target)
    if current_sha256 != document["clozn"]["model_sha256"]:
        raise ValueError(f"{target} has changed since it was adopted; refusing to remove a file that "
                         f"may no longer be what this transaction created")

    if mode == "hard_link":
        # The one binding safety check for this mode: st_nlink is the filesystem's own ground truth for
        # "am I the last reference to these bytes," independent of any assumption about where Ollama's
        # own directory entry lives (see this module's docstring). <= 1 means Ollama's side is already
        # gone -- undo must refuse rather than delete the last surviving copy.
        st_nlink = os.stat(target).st_nlink
        if st_nlink <= 1:
            raise ValueError(
                f"refusing to undo: {target} now has only one remaining hard link. Ollama's own "
                f"reference to this data is gone (removed, garbage collected, or Ollama itself "
                f"uninstalled since this was adopted) -- Clozn's copy is the LAST copy of these bytes. "
                f"Delete it yourself if you are sure you want it gone: {target}")

    target.unlink()
    tx_path.unlink()
    return {"status": "undone", "app": "ollama",
           "registered_name": document["clozn"]["registered_name"], "clozn_path": str(target),
           "mode": mode}


# --------------------------------------------------------------------------------------------------- CLI

def add_subparser(sub):
    parser = sub.add_parser(
        "adopt", help="safely try one model from an existing Ollama install, without touching Ollama")
    parser.add_argument("app", choices=("ollama",))
    parser.add_argument("--model", default=None,
                        help="Ollama model name to adopt (e.g. qwen2.5:7b-instruct); omit to list what "
                             "Ollama has")
    parser.add_argument("--host", default=None,
                        help="explicit Ollama API base URL (default: OLLAMA_HOST or "
                             "http://127.0.0.1:11434)")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--link", action="store_true", help="hard-link into Clozn's model dir (default)")
    reuse.add_argument("--copy", action="store_true",
                       help="copy instead of hard-linking (uses more disk; works across volumes)")
    parser.add_argument("--connect", default=None, metavar="APP", choices=("aider",),
                        help="also point APP at Clozn after adopting (currently: aider)")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1",
                        help="Clozn OpenAI base URL for --connect (default http://127.0.0.1:8080/v1)")
    parser.add_argument("--api-key", default="local-clozn",
                        help="local placeholder API key for --connect (default local-clozn)")
    parser.add_argument("--client-model-label", default="clozn-local",
                        help="model label sent to a --connect'd app's config (default clozn-local; "
                             "deliberately NOT the Ollama model name -- see adopt.py's module docstring "
                             "for why an 'ollama/...' label would misroute Aider's own provider "
                             "resolution)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="show the plan without writing anything")
    action.add_argument("--undo", action="store_true", help="remove a previously adopted model")
    parser.add_argument("--json", action="store_true", help="print a machine-readable result")
    parser.set_defaults(fn=cmd_adopt)
    return parser


def _print_human(report: dict) -> None:
    status = report.get("status")
    if status == "described":
        disco = report["discovery"]
        if not disco["found"]:
            print("No Ollama installation found.")
            for warning in disco["warnings"]:
                print(f"  - {warning}")
            return
        print(f"Found your Ollama setup ({disco['source']}"
             + (f", version {disco['version']}" if disco.get("version") else "") + ")")
        if report.get("models_error"):
            print(f"  {report['models_error']}")
        elif report["models"]:
            print("\nModels")
            for m in report["models"]:
                extra = " ".join(x for x in (m.get("parameter_size"), m.get("quantization_level")) if x)
                print(f"  {m['name']}" + (f"  ({extra})" if extra else ""))
        else:
            print("\nNo models found via Ollama's API.")
        print("\nApplications")
        for app in report["applications"]:
            mark = "✓" if app["connector_available"] else "•"
            note = "" if app["connector_available"] else f" -- {app['note']}"
            print(f"  {mark} {app['app']}{note}")
        print("\nNothing was adopted. Pass --model NAME to adopt one (add --dry-run to preview first).")
        return

    if status == "dry_run":
        print(f"would adopt: {report['model_name']} -> {report['registered_name']}")
        if report["blocked"]:
            print(f"  BLOCKED: {report['blocked']}")
            return
        print(f"  source blob: {report['source_blob_path']}")
        print(f"  source sha256: {report['source_blob_sha256']}")
        print(f"  reuse method: {report['reuse_method']}")
        print(f"  clozn path: {report['clozn_path']}")
        if report["extra_disk_bytes"] is not None:
            print(f"  extra disk usage: {report['extra_disk_bytes'] / 1e9:.2f} GB")
        cap = report["engine_capability"]
        print(f"  engine capability: {cap['level'] if cap else 'unknown'}"
             + (f" ({cap.get('architecture')})" if cap and cap.get("architecture") else ""))
        for warning in report["template"]["warnings"]:
            print(f"  template: {warning}")
        print(f"  undo plan: {report['undo_plan']}")
        return

    if status == "unchanged":
        print(f"already adopted: {report['registered_name']} -> {report['clozn_path']}")
        return

    if status == "adopted":
        print(f"adopted: {report['registered_name']} -> {report['clozn_path']} ({report['mode']})")
        print(f"transaction: {report['transaction_path']}")
        connect = report.get("connect")
        if connect:
            if connect["status"] == "connected":
                print(f"connected {connect['app']}: {connect['target']}")
            else:
                print(f"--connect {connect['app']} FAILED: {connect['error']}")
        print(f"next: clozn run {report['model_name'].split(':')[0]} \"hello\"")
        return

    if status == "undone":
        print(f"undone: {report['registered_name']} ({report['clozn_path']} removed, {report['mode']})")
        return


def cmd_adopt(args):
    from clozn.cli import main as ctx
    try:
        if args.undo:
            report = _undo(args)
        elif args.model:
            plan = _build_plan(args)
            report = _dry_run_report(plan) if args.dry_run else _apply(args, plan)
        else:
            report = _describe_setup(args)
    except (OSError, ValueError) as exc:
        raise ctx.CloznError(f"could not adopt: {exc}") from None

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_human(report)

    if report.get("status") == "adopted" and (report.get("connect") or {}).get("status") == "failed":
        return 1
    return 0
