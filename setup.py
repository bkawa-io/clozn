"""Packaging glue pyproject.toml's declarative ``[tool.setuptools]`` tables can't express.

Everything else about the build (project metadata, the dynamic version, the ``clozn`` console entry
point) lives in pyproject.toml, which setuptools reads first; this file only supplies the ``packages``
list.

Why this can't be pure-TOML: ``[tool.setuptools.packages.find]`` discovers packages by walking real
directory structure under ``where`` and matching an ``include``/``exclude`` glob against the dotted names
it finds *that way* -- it cannot invent a package. Studio's static assets (``studio/``) and the protocol
contract (``protocol/``) need to ship inside the wheel so a `pip install`ed clozn can find them without a
repo checkout (see ``clozn/server/config.py``'s packaged-mode fallback), but both directories live at the
*repo root*, next to ``clozn/`` rather than inside it. ``find()`` never looks at them, so there is no
directory-structure route to a name like "clozn.studio". `find_packages()` supplies the real ``clozn.*``
tree; `_asset_subpackages()` below plus the matching ``package_dir`` remap are what actually pull
studio/ and protocol/ in under the ``clozn`` namespace, unmoved on disk.

Named ``clozn.protocol_fixtures`` rather than ``clozn.protocol`` to avoid colliding with the existing
``clozn/protocol.py`` MODULE (the ``PROTOCOL_VERSION`` constant) -- a distribution can't have both a
``clozn.protocol`` module and a ``clozn.protocol`` package.

Both source directories carry a trivial empty ``__init__.py`` (studio/__init__.py, protocol/__init__.py)
so the installed ``clozn.studio`` / ``clozn.protocol_fixtures`` are REGULAR packages, not implicit PEP 420
namespace packages -- `importlib.resources.files()` on a namespace package returns a `MultiplexedPath`
whose `str()` is not a usable filesystem path, which silently broke the packaged-mode Studio asset lookup
until this was caught by scripts/release/clean_room_install_test.py (see that file's docstring, and
studio/__init__.py's).
"""
import os

from setuptools import find_packages, setup


def _asset_subpackages(dist_root: str, disk_dir: str) -> dict:
    """Every directory under `disk_dir` (including itself) mapped to a dotted package name rooted at
    `dist_root` -> its real disk path. These are DATA packages (HTML/CSS/JS/JSON, no .py code beyond the
    root __init__.py marker), but setuptools' package-discovery sanity check still flags any nested
    directory that "looks importable" (per PEP 420, any directory is) yet isn't in the explicit `packages`
    list, as a likely-accidental omission. Declaring every nested dir explicitly -- rather than silencing
    the check -- keeps `pip install .` quiet without hiding a REAL future misconfiguration (e.g. a typo'd
    package_dir entry that actually does drop files)."""
    mapping = {dist_root: disk_dir}
    for current, subdirs, _files in os.walk(disk_dir):
        subdirs[:] = sorted(d for d in subdirs if not d.startswith("."))
        for d in subdirs:
            rel = os.path.relpath(os.path.join(current, d), disk_dir).replace(os.sep, ".")
            mapping[f"{dist_root}.{rel}"] = os.path.join(current, d)
    return mapping


_studio_map = _asset_subpackages("clozn.studio", "studio")
_protocol_map = _asset_subpackages("clozn.protocol_fixtures", "protocol")

setup(
    packages=find_packages(include=["clozn", "clozn.*"]) + list(_studio_map) + list(_protocol_map),
    package_dir={**_studio_map, **_protocol_map},
    package_data={
        # Artifact schemas ship with the wheel: clozn.schemas.validate() is on the path that reads and
        # writes stored artifacts, so a pip-installed clozn with no defs/ would fail every validation
        # (SchemaError "no schema named ...") rather than degrade. defs/ has no __init__.py -- it is
        # data, not code -- so it rides in as package_data on clozn.schemas rather than as a package.
        "clozn.schemas": ["defs/*.json"],
        "clozn.studio": ["*"],
        "clozn.protocol_fixtures": ["*"],
        **{name: ["*"] for name in _studio_map if name != "clozn.studio"},
        **{name: ["*"] for name in _protocol_map if name != "clozn.protocol_fixtures"},
    },
    include_package_data=True,
)
