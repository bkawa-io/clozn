"""``clozn doctor`` -- a diagnostic sweep for "is this install actually usable", for both a source
checkout and a `pip install`ed release.

Design intent (see docs/BACKLOG.md Sec.2 "CI lanes + release artifact"): a pip user builds the C++ engine
separately (it isn't packaged -- see engine/core/CMakeLists.txt), so a missing engine binary is completely
normal on a fresh `pip install clozn` and must never fail this command; it prints build instructions and
moves on. The only things that fail `doctor` outright are installs that are *actually* broken in a way the
user can't route around by building something later -- today that's just "Python is older than clozn
supports". Every other check is informational (OK) or a WARN with an actionable next step.

Each check is independent and defensively wrapped: one check's unexpected exception becomes a WARN line
for that check, not a crash that hides the rest of the report.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

CLOZN_MIN_PYTHON = (3, 11)   # mirrors pyproject.toml's requires-python -- keep the two in sync by hand

_OK, _WARN, _FAIL = "OK", "WARN", "FAIL"

# Commands that reach a non-loopback host BY DESIGN (HuggingFace downloads, remote GGUF header reads).
# `--verify-offline` must never read as a blanket "this install is fully offline" -- it verifies that
# nothing initiates outbound traffic *unexpectedly*; these two remain able to, on purpose, when invoked.
_OUTBOUND_CAPABLE_COMMANDS = (
    "clozn pull <model> -- downloads a GGUF from HuggingFace",
    "clozn plan <owner/repo/file> -- reads a remote GGUF header over HTTP Range before any download",
)


def _check(label, status, detail=""):
    return {"label": label, "status": status, "detail": detail}


def _check_python() -> dict:
    v = sys.version_info
    ok = (v.major, v.minor) >= CLOZN_MIN_PYTHON
    detail = f"{v.major}.{v.minor}.{v.micro}"
    if not ok:
        detail += f" (clozn requires >= {'.'.join(map(str, CLOZN_MIN_PYTHON))})"
    return _check("python version", _OK if ok else _FAIL, detail)


def _check_protocol() -> dict:
    try:
        from clozn.protocol import PROTOCOL_VERSION
        return _check("protocol version", _OK, PROTOCOL_VERSION)
    except Exception as error:                    # pragma: no cover -- protocol.py is core; this is a canary
        return _check("protocol version", _FAIL, f"could not import clozn.protocol: {error}")


def _check_studio() -> dict:
    try:
        from clozn.server.config import DEMO
        from clozn.server.static import APP_INDEX
        # DERIVE the checked path from APP_INDEX rather than spelling it out. The intent was always
        # "check the app the gateway actually redirects to", but a hardcoded "app/index.html" quietly
        # became wrong the moment the served app moved -- doctor would have reported OK for a path
        # nothing serves, or WARNed about a directory that was correctly gone. Deriving it means the
        # redirect target and this check cannot disagree.
        rel = APP_INDEX.lstrip("/").replace("/", os.sep)
        index = os.path.join(DEMO, rel)
        if os.path.isfile(index):
            return _check("studio assets", _OK, f"{DEMO} (serving {APP_INDEX})")
        return _check("studio assets", _WARN,
                      f"{DEMO} exists but {APP_INDEX.lstrip('/')} is missing under it")
    except Exception as error:
        return _check("studio assets", _WARN, f"could not resolve studio assets: {error}")


def _check_models() -> dict:
    try:
        from clozn.cli.commands.models import _model_dirs, _scan_models
        dirs = _model_dirs()
        ggufs = _scan_models()
        if ggufs:
            return _check("models", _OK, f"{len(ggufs)} GGUF(s) across {len(dirs)} dir(s)")
        if dirs:
            return _check("models", _WARN,
                          f"no GGUFs found in {len(dirs)} dir(s) searched ({', '.join(dirs)}); "
                          "`clozn pull <model>` to fetch one")
        return _check("models", _WARN,
                      "no model dirs exist yet; put .gguf files in ~/.clozn/models or set CLOZN_MODELS=<dir>")
    except Exception as error:
        return _check("models", _WARN, f"could not scan for models: {error}")


def _check_registry() -> dict:
    from clozn.cli import main as ctx
    path = os.path.join(ctx.HOME, "daemons.json")
    if not os.path.isfile(path):
        return _check("registry", _OK, f"{path} (not created yet -- no `clozn serve` has run)")
    try:
        with open(path, encoding="utf-8") as handle:
            raw = handle.read()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return _check("registry", _WARN, f"{path} does not contain a JSON object; self-heals on next write")
    except Exception as error:
        return _check("registry", _WARN, f"{path} is not valid JSON ({error}); self-heals on next `clozn serve`/`clozn stop`")
    try:
        from clozn.cli.engine_process import _pid_alive
        stale = [port for port, entry in data.items() if not _pid_alive(entry.get("pid"))]
    except Exception:
        stale = None
    if not data:
        return _check("registry", _OK, f"{path} (empty)")
    if stale:
        return _check("registry", _WARN,
                      f"{path}: {len(stale)}/{len(data)} entries stale (process not running: port(s) "
                      f"{', '.join(stale)}); self-heals on next `clozn ps`/`clozn stop`")
    return _check("registry", _OK, f"{path}: {len(data)} entries, all live")


def _bootstrap_llama_pin(repo: str) -> "tuple[str, str] | None":
    """Parse the pinned llama.cpp TAG/COMMIT straight out of bootstrap_llama.py's source text (a static
    regex read, not an import/exec -- this only needs two string literals, and never wants to risk running
    that script's module-level code). Only present in a source checkout; a pip release doesn't ship
    engine/ at all, so returning None there is the expected, unremarkable case."""
    path = os.path.join(repo, "engine", "core", "third_party", "bootstrap_llama.py")
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None
    tag = re.search(r'^TAG\s*=\s*"([^"]+)"', text, re.MULTILINE)
    commit = re.search(r'^COMMIT\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if tag and commit:
        return tag.group(1), commit.group(1)
    return None


def _check_engine() -> dict:
    from clozn.cli.engine_process import find_engine, REPO
    from clozn.cli.main import CloznError
    try:
        exe, _dll_dirs, gpu = find_engine(prefer_gpu=True)
    except CloznError:
        return _check("engine binary", _WARN,
                      "no clozn-server build found. Pip installs the Python supervisor only -- build the "
                      "C++ worker separately: see docs/DEVELOPMENT.md, or set CLOZN_ENGINE_BIN to a "
                      "prebuilt clozn-server(.exe).")
    detail = f"{exe} ({'GPU' if gpu else 'CPU'} build)"
    pin = _bootstrap_llama_pin(REPO)
    if pin:
        tag, commit = pin
        # Static provenance only: the binary itself embeds no build-flags/commit record today (no
        # CMake step emits one -- see this task's report for a proposed server_main.cpp/CMakeLists.txt
        # addition). What we CAN say honestly is what source commit this checkout is pinned to build
        # from; whether the binary at `exe` was actually built from exactly that checkout is unverified.
        detail += f"; llama.cpp pinned @ {tag} ({commit[:12]}) -- unverified against the built binary"
    else:
        detail += "; llama.cpp pin unavailable (engine/core/third_party/bootstrap_llama.py not found)"
    return _check("engine binary", _OK, detail)


def _check_offline() -> dict:
    """Explicit trust check: local-only must be active and its guarded ledger window clean.

    This never claims the install is blanket "offline" -- a handful of commands reach the network BY
    DESIGN (model downloads), and their names are always listed alongside the verdict so a passing check
    is never mistaken for "nothing here can ever make a request."
    """
    known_outbound = list(_OUTBOUND_CAPABLE_COMMANDS)
    try:
        from clozn import network_policy
        report = network_policy.verify_offline()
    except Exception as error:
        result = _check("offline enforcement", _FAIL, f"verification could not run: {error}")
        result["known_outbound_capable_commands"] = known_outbound
        return result
    commands_note = "; unaffected by design: " + ", ".join(
        cmd.split(" -- ")[0].strip() for cmd in known_outbound)
    if report.get("verified"):
        blocked = int(report.get("blocked_external_attempt_count") or 0)
        detail = f"active; {blocked} external attempt(s) blocked"
        since = report.get("since")
        if since:
            detail += f" since {since}"
        result = _check("offline enforcement", _OK, detail + commands_note)
    else:
        reasons = []
        if report.get("local_only") is False:
            reasons.append("local-only is off")
        if report.get("guard_installed") is False:
            reasons.append("urllib guard is not installed")
        if report.get("probe_blocked") is False:
            reasons.append("external probe was not blocked before transport")
        elif report.get("probe_recorded") is False:
            reasons.append("blocked probe was not durably recorded")
        violations = report.get("violations") or []
        if violations:
            reasons.append(f"{len(violations)} unblocked external attempt(s) in the ledger window")
        result = _check("offline enforcement", _FAIL,
                        ("; ".join(reasons) or str(report.get("reason") or "verification failed"))
                        + commands_note)
    # Machine-readable doctor output retains the exact evidence without exposing request content
    # (network_policy's ledger contract stores destination metadata only).
    result["evidence"] = report
    result["known_outbound_capable_commands"] = known_outbound
    return result


def _check_bind_loopback() -> dict:
    """Static, code-shape verification (not a live network probe) that this CLI's own commands cannot bind
    the public gateway or the private engine worker to anything but loopback.

    Mirrors `_check_engine`'s static-source-read approach: reading `RuntimeConfig`'s default and `clozn
    serve`'s registered argparse flags, plus a literal-text scan of `_launch_args`'s subprocess argv
    construction (the function `spawn_engine` calls to build the private worker's command line), is a
    fact about the CODE that ships, not a claim about whatever process happens to be running right now --
    hand-editing these files, or launching the built engine binary directly outside `clozn serve`, is out
    of scope and cannot be verified from here. Never FAILs: an install that can't be statically confirmed
    still isn't necessarily broken, so this is WARN-or-OK like the rest of `doctor`.
    """
    try:
        import dataclasses
        import inspect

        from clozn.cli.engine_process import _launch_args
        from clozn.cli.main import build_parser
        from clozn.cli.runtime_process import RuntimeConfig

        gateway_default = next(
            f.default for f in dataclasses.fields(RuntimeConfig) if f.name == "host")

        parser = build_parser()
        subparsers_action = next(
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
        serve_parser = subparsers_action.choices.get("serve")
        serve_has_host_override = bool(serve_parser) and any(
            "--host" in action.option_strings for action in serve_parser._actions)

        engine_hardcodes_loopback = '"--host", "127.0.0.1"' in inspect.getsource(_launch_args)
    except Exception as error:
        return _check("gateway/engine bind", _WARN,
                      f"could not statically verify loopback-only bind: {error}")

    if gateway_default == "127.0.0.1" and not serve_has_host_override and engine_hardcodes_loopback:
        return _check("gateway/engine bind", _OK,
                      "loopback-only by construction: `clozn serve` exposes no --host override "
                      "(RuntimeConfig.host defaults to 127.0.0.1) and the private engine worker's argv "
                      "hardcodes --host 127.0.0.1 -- a code fact, not a live-process probe")
    return _check("gateway/engine bind", _WARN,
                  f"could not confirm loopback-only by construction (gateway default host="
                  f"{gateway_default!r}, `clozn serve --host` override present={serve_has_host_override}, "
                  f"engine argv hardcodes loopback={engine_hardcodes_loopback})")


def _run_all(*, verify_offline: bool = False) -> list:
    checks = [
        _check_python(),
        _check_protocol(),
        _check_studio(),
        _check_models(),
        _check_registry(),
        _check_engine(),
    ]
    if verify_offline:
        checks.append(_check_bind_loopback())
        checks.append(_check_offline())
    return checks


def cmd_doctor(args) -> int:
    results = _run_all(verify_offline=bool(getattr(args, "verify_offline", False)))
    as_json = getattr(args, "json", False)
    if as_json:
        worst = _FAIL if any(r["status"] == _FAIL for r in results) else \
                _WARN if any(r["status"] == _WARN for r in results) else _OK
        print(json.dumps({"status": worst, "checks": results}, indent=2))
    else:
        width = max(len(r["label"]) for r in results)
        for r in results:
            tag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[r["status"]]
            print(f"{tag} {r['label']:<{width}}  {r['detail']}")
    return 1 if any(r["status"] == _FAIL for r in results) else 0
