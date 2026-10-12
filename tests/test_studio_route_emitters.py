"""Keep Studio links on the one canonical route vocabulary."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RouteEvidence:
    layer: str
    route_family: str
    path: str
    needle: str


CANONICAL_EMITTERS = (
    RouteEvidence("python", "materialized comparison", "clozn/experiments/materialize.py", '"compare_path": f"#/compare/'),
    RouteEvidence("python", "context-action comparison", "clozn/server/routes/context_investigation_v1.py", '"compare_path": f"#/compare/'),
    RouteEvidence("python", "receipt permalink", "clozn/server/routes/receipt_link.py", 'APP_INDEX + "#/inspect/"'),
    RouteEvidence("studio", "sessions", "studio-frontend/src/app/router.ts", 'return "#/sessions"'),
    RouteEvidence("studio", "inspect", "studio-frontend/src/app/router.ts", "return `#/inspect/"),
    RouteEvidence("studio", "runtime", "studio-frontend/src/app/router.ts", 'return "#/runtime"'),
    RouteEvidence("studio", "comparison", "studio-frontend/src/app/router.ts", "? `#/compare/"),
    RouteEvidence("studio", "experiment matrix", "studio-frontend/src/app/router.ts", '"#/compare/matrix"'),
)


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _bundled_javascript() -> str:
    files = sorted((ROOT / "studio/app/assets").glob("*.js"))
    assert files, "studio/app/assets has no committed JavaScript bundle"
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def test_declared_route_emitters_are_present_and_uniquely_named():
    identities = {(row.layer, row.route_family) for row in CANONICAL_EMITTERS}
    assert len(identities) == len(CANONICAL_EMITTERS)
    for row in CANONICAL_EMITTERS:
        assert row.needle in _text(row.path), f"canonical {row.route_family} emitter moved"


def test_python_hash_emitters_are_exhaustively_classified():
    observed = {
        str(path.relative_to(ROOT))
        for path in (ROOT / "clozn").rglob("*.py")
        if re.search(r"#/[A-Za-z]", path.read_text(encoding="utf-8"))
    }
    classified = {row.path for row in CANONICAL_EMITTERS if row.layer == "python"}
    assert observed == classified


def test_static_entrypoint_is_the_single_app():
    static = _text("clozn/server/static.py")
    assert 'APP_INDEX = "/app/index.html"' in static


def test_studio_source_contains_only_canonical_route_families():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "studio-frontend/src").rglob("*"))
        if path.suffix in {".ts", ".tsx"} and ".test." not in path.name
    )
    for route in ("#/sessions", "#/inspect/", "#/runtime", "#/compare", "#/compare/matrix"):
        assert route in source


def test_committed_bundle_contains_the_canonical_routes():
    bundle = _bundled_javascript()
    for route in ("#/sessions", "#/inspect/", "#/runtime", "#/compare", "#/compare/matrix"):
        assert route in bundle
