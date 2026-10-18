"""Clean-room install proof: build clozn's wheel, install it into a brand-new venv, then prove the
console entry point, `python -m clozn`, and `import clozn` all work OUTSIDE this repo checkout -- no
CLOZN_* env vars pointing back at it, no repo-relative sys.path hacks, cwd is a scratch temp dir. This is
the release bar: pip install into a clean environment must yield a working Clozn with no repo-path
hacks. A passing run here is what "packaging works" means, not just "the wheel built".

    python scripts/release/clean_room_install_test.py
    python scripts/release/clean_room_install_test.py --keep      # leave the venv+dist dirs for inspection

Deliberately stdlib-only (venv, subprocess, tempfile) -- this has to run before anything else is proven to
work, so it can't lean on the package it's testing. Requires the `build` package be importable by THIS
interpreter (the wheel is built with `sys.executable`, not the target venv) -- CI installs it as a step;
locally: `pip install build`.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _venv_python(venv_dir: str) -> str:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return os.path.join(venv_dir, bin_dir, exe)


def _venv_console_script(venv_dir: str, name: str) -> str:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    exe = f"{name}.exe" if os.name == "nt" else name
    return os.path.join(venv_dir, bin_dir, exe)


def _run(cmd, **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    return result


def _verify_wheel_manifest(wheel_path: str) -> None:
    """Reject missing v1 assets and any known removed product surface before installation."""
    required = {
        "clozn/__init__.py",
        "clozn/server/app.py",
        "clozn_engine/__init__.py",
        "clozn/studio/app/index.html",
        "clozn/schemas/defs/clozn.context-investigation-effects-job-result.v1.json",
        "clozn/schemas/defs/clozn.minimal-context-search-result.v2.json",
    }
    forbidden = (
        "clozn_client/",
        "clozn/runs/telemetry.py",
        "clozn/runs/sections.py",
        "clozn/runs/lineage.py",
        "clozn/receipts/forced.py",
        "clozn/experiments/state_ref.py",
        "clozn/models/lockfile.py",
        "clozn/cli/throughput_predictor.py",
        "clozn/studio/casting/",
        "clozn/studio/ext/",
        "clozn/studio/atlas_emergent.json",
        "clozn/studio/workspace_lens_trace.jsonl",
        "clozn/schemas/defs/clozn.context-answer-influence-export.v1.json",
        "clozn/schemas/defs/clozn.receipt-privacy-transaction.v1.json",
        "clozn/schemas/defs/clozn.model-lock.v1.json",
    )
    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        missing = sorted(required - names)
        leaked = sorted(
            name for name in names
            if any(fragment in name for fragment in forbidden)
            or (name.endswith(".py") and name.rsplit("/", 1)[-1].startswith("test_"))
        )
        javascript = b"\n".join(
            archive.read(name)
            for name in names
            if name.startswith("clozn/studio/app/assets/") and name.endswith(".js")
        )
    if missing:
        raise RuntimeError(f"wheel is missing required v1 entries: {missing}")
    if leaked:
        raise RuntimeError(f"wheel contains removed product entries: {leaked}")
    if b"clozn.influence-map-job.v1" in javascript:
        raise RuntimeError("packaged Studio contains the removed influence-map job contract")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true", help="don't delete the scratch venv/dist dirs afterward")
    args = ap.parse_args(argv)

    scratch = tempfile.mkdtemp(prefix="clozn-clean-room-")
    dist_dir = os.path.join(scratch, "dist")
    venv_dir = os.path.join(scratch, "venv")
    outside_cwd = os.path.join(scratch, "elsewhere")   # proves nothing depends on REPO being cwd
    os.makedirs(outside_cwd, exist_ok=True)
    print(f"scratch dir: {scratch}")

    try:
        # 1) Build the wheel with THIS interpreter's build tooling -- the target venv only ever needs pip.
        build_env = dict(os.environ)
        build_env["CLOZN_BUILD_BASE"] = os.path.join(scratch, "build")
        build = _run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", dist_dir],
            cwd=REPO,
            env=build_env,
        )
        if build.returncode != 0:
            print("FAIL: wheel build failed", file=sys.stderr)
            return 1
        wheels = [f for f in os.listdir(dist_dir) if f.endswith(".whl")]
        if not wheels:
            print("FAIL: no wheel produced", file=sys.stderr)
            return 1
        wheel_path = os.path.join(dist_dir, wheels[0])
        print(f"built {wheel_path}")
        try:
            _verify_wheel_manifest(wheel_path)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            print(f"FAIL: wheel manifest is invalid: {exc}", file=sys.stderr)
            return 1
        print("PASS: exact v1 wheel manifest")

        # 2) A brand-new venv -- not the interpreter running this script, so nothing it already has
        #    installed (or an editable-install path hack) can paper over a packaging bug.
        venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
        vpy = _venv_python(venv_dir)

        # 3) Install ONLY the wheel -- no `-e`, no repo on PYTHONPATH, no CLOZN_* overrides.
        install = _run([vpy, "-m", "pip", "install", "--no-input", wheel_path])
        if install.returncode != 0:
            print("FAIL: pip install of the wheel failed", file=sys.stderr)
            return 1

        checks = []

        def check(name, cmd):
            result = _run(cmd, cwd=outside_cwd)
            ok = result.returncode == 0
            checks.append((name, ok))
            print(f"{'PASS' if ok else 'FAIL'}: {name}")
            return result

        # `import clozn` must work with NO repo on sys.path and NOT running from inside the repo.
        check('python -c "import clozn"', [vpy, "-c", "import clozn; print('clozn at', clozn.__file__)"])

        # The installed product gateway must carry its private native-worker client. A source checkout
        # finds this client through engine/client, so importing only `clozn` cannot catch a wheel that
        # accidentally omitted it.
        engine_client_probe = """
import clozn.server.app as app
assert app.EngineClient is not None
assert "torch" not in __import__("sys").modules
print("private engine client", app.EngineClient.__module__)
"""
        check("packaged private engine client", [vpy, "-c", engine_client_probe])

        # The installed wheel must contain the application the server redirects to. Checking only
        # `clozn doctor` is insufficient because missing optional assets are reported as a warning.
        studio_probe = """
from importlib.resources import files
import re

package = files("clozn.studio")
directories = {item.name for item in package.iterdir() if item.is_dir()}
assert directories <= {"app", "__pycache__"}, directories
root = package.joinpath("app")
index = root.joinpath("index.html")
assert index.is_file(), index
html = index.read_text(encoding="utf-8")
refs = re.findall(r'(?:src|href)="([^"]+)"', html)
local = [ref for ref in refs if not ref.startswith(("http://", "https://", "//", "data:"))]
assert local, refs
for ref in local:
    assert not ref.startswith("/"), ref
    target = root.joinpath(*ref.removeprefix("./").split("/"))
    assert target.is_file(), target
print("Studio at", root)
"""
        check("packaged Studio application", [vpy, "-c", studio_probe])

        # `clozn version` / `clozn doctor` via `python -m clozn` (the documented entry point clozn.cmd /
        # clozn.sh both use) -- doctor must not crash even though nothing is installed/running yet.
        check("python -m clozn version", [vpy, "-m", "clozn", "version"])
        check("python -m clozn doctor", [vpy, "-m", "clozn", "doctor"])

        # The actual `clozn` console script pip put on the venv's PATH (not `python -m`) -- this is the
        # literal `[project.scripts]` entry point pyproject.toml declares.
        console = _venv_console_script(venv_dir, "clozn")
        if os.path.isfile(console):
            check("console script `clozn version`", [console, "version"])
            check("console script `clozn doctor`", [console, "doctor"])
        else:
            print(f"FAIL: console entry point not found at {console}", file=sys.stderr)
            checks.append(("console script exists", False))

        failed = [name for name, ok in checks if not ok]
        print()
        if failed:
            print(f"CLEAN-ROOM INSTALL TEST: FAILED ({len(failed)}/{len(checks)}): {', '.join(failed)}")
            return 1
        print(f"CLEAN-ROOM INSTALL TEST: PASSED ({len(checks)}/{len(checks)})")
        return 0
    finally:
        if args.keep:
            print(f"--keep: leaving {scratch} in place")
        else:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
