"""Server config -- path resolution + process-wide startup constants for clozn.server.app.

Imported FIRST by app.py so REPO_ROOT/DEMO exist and engine/client is on sys.path by the time app.py
goes on to `from clozn_engine import ...`. These side effects are deliberately module-load-time.

TORCH-FREE BY CONSTRUCTION: this runs on the PRODUCT import path, so it must not pull in anything only
the lab needs. The `engine/lab` sys.path entry (for the Dream substrate's clozn_lab) and the HF hub
symlink workaround used to live here; they moved to the `clozn lab` entry point (clozn/lab/app.py). A
product `clozn serve` process needs neither, and keeping them off this path is part of what makes the
product package physically unable to reach the Torch lab code.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))     # clozn/server/config.py -> repo root is two levels up
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))  # the engine white-box SDK (product; torch-free)


def _resolve_studio_dir() -> str:
    """Where Studio's static assets live, in priority order: 1) CLOZN_STUDIO_DIR (explicit override --
    unchanged from before), 2) the repo layout (studio/ next to clozn/ -- true in a source checkout or
    `pip install -e .`, and the only case this function used to handle), 3) the packaged copy shipped as
    the `clozn.studio` subpackage (true for a `pip install clozn` release, where studio/ is NOT a sibling
    of the installed clozn/ package -- see the root pyproject.toml/setup.py's package_dir remap). Falls
    back to the repo-layout path even when it doesn't exist so a broken install still points at a real,
    diagnosable location instead of raising here at import time.
    """
    override = os.environ.get("CLOZN_STUDIO_DIR")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    repo_layout = os.path.join(REPO_ROOT, "studio")
    if os.path.isdir(repo_layout):
        return repo_layout
    try:
        import importlib.resources as _ir
        packaged = _ir.files("clozn.studio")
        if packaged.is_dir():
            return str(packaged)
    except Exception:
        pass   # not a packaged install, or the package_data didn't land -- fall through to the honest default
    return repo_layout


DEMO = _resolve_studio_dir()

CLOZN_DIR = os.path.join(os.path.expanduser("~"), ".clozn")     # studio memory + personality persist here
