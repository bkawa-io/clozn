"""`clozn adopt ollama` -- try one model from an existing Ollama install without deleting Ollama,
redownloading multi-gigabyte weights, or configuring any other application. Clozn owns the model
entry it creates under its own directory; it does not own or edit another application's config --
see docs/CAPABILITIES.md for that boundary.

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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from clozn import schemas
from clozn._io import atomic_write_json
from clozn.adopt import ollama_discovery as discovery
from clozn.adopt import ollama_resolver as resolver

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
    return resolver.absolute_from_path(modelfile_text)


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
    spec's own worked example output ('Found your Ollama setup / Models')."""
    disco = discovery.discover(host_override=args.host)
    report = {"status": "described", "discovery": disco, "models": []}
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

    from clozn.cli.commands._fileops import sha256_path
    storage_roots = []
    if disco.get("storage_path"):
        storage_roots.append(disco["storage_path"])
    storage_roots.extend(
        path for path in discovery.known_storage_paths() if path not in storage_roots
    )
    resolution = resolver.resolve_model_blob(
        model_name,
        shown,
        storage_roots=storage_roots,
        explicit_blob=getattr(args, "blob", None),
    )
    blob_path = resolution["path"] if resolution else None
    blocked = None
    source_sha256 = None
    header = None
    if blob_path is None:
        blocked = ("no local GGUF blob could be resolved for this model (it may be a cloud-only/remote "
                  "model, or its modelfile does not FROM an absolute local path). Clozn can only adopt "
                  "weights it can read locally.")
    else:
        source_sha256 = sha256_path(Path(blob_path))
        expected_size = resolution.get("expected_size")
        if expected_size is not None and os.path.getsize(blob_path) != expected_size:
            blocked = (
                f"resolved Ollama blob size mismatch: manifest says {expected_size} bytes, "
                f"but {blob_path} is {os.path.getsize(blob_path)} bytes"
            )
        expected_digest = resolution.get("blob_digest")
        if expected_digest and expected_digest != f"sha256:{source_sha256}":
            blocked = (
                "resolved Ollama blob digest mismatch: manifest names "
                f"{expected_digest}, but the local file hashes to sha256:{source_sha256}"
            )
        header, header_error = _read_gguf_header(blob_path)
        if header is None and blocked is None:
            blocked = f"unsupported model: {header_error}"

    slug = _slug(model_name)
    plan = {
        "disco": disco, "host": host, "model_name": model_name, "shown": shown,
        "registered_name": f"ollama/{model_name}",
        "blob_path": blob_path, "source_sha256": source_sha256, "header": header,
        "resolution": resolution,
        "translation": resolver.translate_definition(shown),
        "reported_digest": entry.get("digest"),
        "clozn_path": str(_clozn_models_dir() / f"ollama__{slug}.gguf"),
        "mode": "copy" if args.copy else "hard_link",
        "blocked": blocked,
    }
    return plan


def _dry_run_report(plan: dict) -> dict:
    extra_disk = None
    if plan["blob_path"] and not plan["blocked"]:
        extra_disk = 0 if plan["mode"] == "hard_link" else os.path.getsize(plan["blob_path"])
    report = {
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
        "template": plan["translation"],
        "blob_resolution": plan["resolution"],
        "blocked": plan["blocked"],
        "undo_plan": (
            "hard link: --undo removes only Clozn's own directory entry; the Ollama-side blob is never "
            "touched, and undo refuses automatically if Ollama's own reference to the same data is "
            "already gone (last-copy protection via the filesystem's own hard-link count)."
            if plan["mode"] == "hard_link" else
            "copy: --undo deletes only the independent copy Clozn made; Ollama's own file is never "
            "touched or affected."),
    }
    report["requires_confirmation"] = False
    return report


def _run_qualification(plan: dict) -> dict:
    """Start the real product runtime, make one tiny deterministic request, and
    prove that its run has a Context Receipt. Always stops both child processes.
    """
    from clozn.cli.engine_process import _free_port
    from clozn.cli.runtime_process import RuntimeConfig, spawn_runtime

    stack = None
    try:
        port = _free_port()
        stack = spawn_runtime(RuntimeConfig(
            model=plan["clozn_path"],
            public_port=port,
            flags={"ctx": 1024},
            prefer_gpu=False,
            gateway_boot_timeout=45,
            worker_boot_timeout=180,
        ))
        payload = json.dumps({
            "model": "clozn-local",
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "temperature": 0,
            "max_tokens": 8,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            generated = json.loads(response.read().decode("utf-8"))
        run_id = generated.get("clozn_run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("qualification response did not include a Clozn run ID")
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/runs/{run_id}/context-receipt", timeout=10
        ) as response:
            receipt = json.loads(response.read().decode("utf-8"))
        if receipt.get("shape") not in {"new", "legacy"}:
            raise ValueError("qualification run did not produce a readable Context Receipt")
        report = {
            "status": "passed",
            "capability": "core",
            "run_id": run_id,
            "receipt_shape": receipt["shape"],
            "warnings": (
                ["qualification used CPU for a deterministic portable smoke; GPU remains unqualified"]
                if not stack.gpu else []
            ),
        }
        if stack.worker_health.get("protocol_version") is not None:
            report["worker_protocol_version"] = stack.worker_health["protocol_version"]
        if stack.worker_health.get("backend") is not None:
            report["worker_backend"] = stack.worker_health["backend"]
        return report
    except Exception as exc:
        return {
            "status": "failed",
            "capability": "discovered",
            "error": f"{type(exc).__name__}: {exc}",
            "warnings": ["the adopted model was preserved; qualification failure did not roll it back"],
            "undo_command": f"clozn adopt ollama --model {plan['model_name']} --undo",
        }
    finally:
        if stack is not None:
            stack.stop()


def _apply(args, plan: dict) -> dict:
    if plan["blocked"]:
        raise ValueError(f"cannot adopt {plan['model_name']!r}: {plan['blocked']}")

    from clozn.cli.commands._fileops import atomic_copy_file, sha256_path

    target = Path(plan["clozn_path"])
    tx_path = _transaction_path(plan["registered_name"])

    if tx_path.is_file() and target.is_file():
        try:
            prior = json.loads(tx_path.read_text(encoding="utf-8"))
        except Exception:
            prior = None
        if (isinstance(prior, dict)
                and prior.get("ollama", {}).get("source_blob_sha256") == plan["source_sha256"]
                and prior.get("ollama", {}).get("manifest_digest") == plan.get("reported_digest")
                and prior.get("ollama", {}).get("blob_digest")
                    == (plan.get("resolution") or {}).get("blob_digest")
                and sha256_path(target) == prior.get("clozn", {}).get("model_sha256")):
            qualification_report = None
            changed = False
            if getattr(args, "qualify", False):
                qualification_report = _run_qualification(plan)
                prior["qualification"] = qualification_report
                changed = True
            if changed:
                schemas.validate(prior, _SCHEMA)
                atomic_write_json(
                    str(tx_path), prior, ensure_ascii=False, indent=2, sort_keys=True
                )
            return {"status": "unchanged", "app": "ollama", "model_name": plan["model_name"],
                   "registered_name": plan["registered_name"], "clozn_path": str(target),
                   "transaction_path": str(tx_path), "qualification": qualification_report,
                   "undo_commands": {
                       "adoption": f"clozn adopt ollama --model {plan['model_name']} --undo",
                   }}

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
            "blob_digest": (plan.get("resolution") or {}).get("blob_digest"),
            "resolution_method": (plan.get("resolution") or {}).get("method"),
            "manifest_path": (plan.get("resolution") or {}).get("manifest_path"),
            "source_blob_path": plan["blob_path"],
            "source_blob_sha256": plan["source_sha256"]}.items() if v is not None},
        "clozn": {
            "registered_name": plan["registered_name"], "path": str(target),
            "mode": plan["mode"], "model_sha256": model_sha256},
        "template": plan["translation"],
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

    qualification_report = None
    if getattr(args, "qualify", False):
        qualification_report = _run_qualification(plan)
        document["qualification"] = qualification_report
        schemas.validate(document, _SCHEMA)
        try:
            atomic_write_json(str(tx_path), document, ensure_ascii=False, indent=2, sort_keys=True)
        except OSError:
            pass

    return {"status": "adopted", "app": "ollama", "model_name": plan["model_name"],
           "registered_name": plan["registered_name"], "clozn_path": str(target), "mode": plan["mode"],
           "transaction_path": str(tx_path), "qualification": qualification_report,
           "undo_commands": {
               "adoption": f"clozn adopt ollama --model {plan['model_name']} --undo",
           }}


# ---------------------------------------------------------------------------------------------------- undo

def _undo(args) -> dict:
    from clozn.cli.commands._fileops import sha256_path

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
    parser.add_argument(
        "--blob",
        default=None,
        help="explicit absolute GGUF blob fallback when Ollama metadata cannot resolve one",
    )
    parser.add_argument(
        "--qualify",
        action="store_true",
        help="after adoption, start Clozn on CPU, run a tiny deterministic prompt, verify its receipt, and stop",
    )
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--link", action="store_true", help="hard-link into Clozn's model dir (default)")
    reuse.add_argument("--copy", action="store_true",
                       help="copy instead of hard-linking (uses more disk; works across volumes)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="show the plan without writing anything")
    action.add_argument("--undo", action="store_true", help="remove a previously adopted model")
    action.add_argument(
        "--try",
        dest="try_flow",
        action="store_true",
        help="guided preview/apply flow; add --yes to apply the preview noninteractively",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="explicitly confirm --try in noninteractive use",
    )
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
        print("\nNothing was adopted. Pass --model NAME to adopt one (add --dry-run to preview first).")
        return

    if status == "dry_run":
        print(f"would adopt: {report['model_name']} -> {report['registered_name']}")
        if report["blocked"]:
            print(f"  BLOCKED: {report['blocked']}")
            return
        print(f"  source blob: {report['source_blob_path']}")
        print(f"  source sha256: {report['source_blob_sha256']}")
        resolution = report.get("blob_resolution") or {}
        print(f"  resolution: {resolution.get('method', 'unresolved')}")
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
        if report.get("requires_confirmation"):
            print(f"  confirmation required: {report['confirm_command']}")
        return

    if status == "unchanged":
        print(f"already adopted: {report['registered_name']} -> {report['clozn_path']}")
        if (report.get("qualification") or {}).get("status") == "passed":
            print(f"qualification: core (run {report['qualification']['run_id']})")
        return

    if status == "adopted":
        print(f"adopted: {report['registered_name']} -> {report['clozn_path']} ({report['mode']})")
        print(f"transaction: {report['transaction_path']}")
        qualification = report.get("qualification")
        if qualification:
            if qualification["status"] == "passed":
                print(
                    f"qualification: core (run {qualification['run_id']}, "
                    f"receipt {qualification['receipt_shape']})"
                )
            else:
                print(f"qualification FAILED: {qualification['error']}")
        for kind, command in report.get("undo_commands", {}).items():
            print(f"undo {kind}: {command}")
        print(f"next: clozn run {report['model_name'].split(':')[0]} \"hello\"")
        print("to point an existing app at Clozn: clozn serve, then set its OpenAI-compatible base "
              "URL to http://127.0.0.1:8080/v1 (see docs/CLIENT_CONFORMANCE.md for Aider/Open WebUI/"
              "SDK examples)")
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
            if args.dry_run or (getattr(args, "try_flow", False) and not getattr(args, "yes", False)):
                report = _dry_run_report(plan)
                if getattr(args, "try_flow", False):
                    report["requires_confirmation"] = True
                    report["confirm_command"] = (
                        f"clozn adopt ollama --model {args.model} --try --yes"
                        + (" --qualify" if getattr(args, "qualify", False) else "")
                    )
            else:
                report = _apply(args, plan)
        else:
            report = _describe_setup(args)
    except (OSError, ValueError) as exc:
        raise ctx.CloznError(f"could not adopt: {exc}") from None

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_human(report)

    if report.get("status") in {"adopted", "unchanged"} and (
        report.get("qualification") or {}
    ).get("status") == "failed":
        return 1
    return 0
