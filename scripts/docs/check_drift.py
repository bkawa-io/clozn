#!/usr/bin/env python3
"""High-confidence documentation drift checks.

The checker intentionally covers only claims that can be decided without a model, network, or release
API. Historical research and dated planning documents are allowlisted: they preserve what was true at
the time and carry their own archive banners.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from urllib.parse import unquote


ARCHIVAL_PREFIXES = (
    "docs/design/",
    "docs/research/",
)

ARCHIVAL_FILES = frozenset(
    {
        "docs/BACKLOG.md",
        "docs/DESIGN.md",
        "docs/EXPLAIN_THIS_ANSWER_SPEC.md",
        "docs/PRODUCT_ROADMAP.md",
        "docs/PRODUCT_STRATEGY_USER_NEEDS_2026-07-20.md",
        "docs/RESEARCH_ROADMAP.md",
        "docs/ROADMAP.md",
        "docs/STUDIO_NEXT_HANDOFF_2026-07-26.md",
        "docs/TECHNICAL.md",
    }
)

REMOVED_COMMAND_RE = re.compile(
    r"\bclozn\s+("
    r"lab|memory|preferences|privacy|qualify-whitebox|"
    r"qualify-chat-io|migrate(?:-runs)?"
    r")\b",
    re.IGNORECASE,
)
GITHUB_CLOZN_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/\s)>]+)/clozn(?:[/?#)\s>`]|$)",
    re.IGNORECASE,
)
INLINE_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\((?P<target>[^)\n]+)\)")
ADAPTER_NEGATION_RE = re.compile(
    r"(?:"
    r"\bclozn\s+diff-adapter\b.{0,100}\b(?:not\s+(?:yet\s+)?built|unbuilt|unsupported)"
    r"|"
    r"\b(?:no\s+LoRA\s+support|LoRA\s+support\s+is\s+not\s+(?:built|supported))\b"
    r")",
    re.IGNORECASE,
)
HEADING_RE = re.compile(r"^\s*#{1,6}\s+(?P<title>.+?)\s*$")


@dataclass(frozen=True, order=True)
class Violation:
    path: str
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.code}: {self.message}"


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_archival(relative_path: str) -> bool:
    return relative_path in ARCHIVAL_FILES or relative_path.startswith(ARCHIVAL_PREFIXES)


def iter_current_docs(root: Path):
    readme = root / "README.md"
    if readme.is_file():
        yield readme
    docs = root / "docs"
    if not docs.is_dir():
        return
    for path in sorted(docs.rglob("*.md")):
        relative = _relative(path, root)
        if not is_archival(relative):
            yield path


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _link_target(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        return value[1:value.index(">")].strip()
    # Markdown permits an optional title after whitespace. Repository paths in this project do not
    # contain unescaped spaces.
    return value.split(maxsplit=1)[0] if value else ""


def _is_external_or_route(target: str) -> bool:
    if not target or target.startswith(("#", "/", "//")):
        return True
    return re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target) is not None


def _check_links(path: Path, root: Path, text: str) -> list[Violation]:
    relative = _relative(path, root)
    violations: list[Violation] = []
    for match in INLINE_LINK_RE.finditer(text):
        target = _link_target(match.group("target"))
        if _is_external_or_route(target):
            continue
        local = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if not local:
            continue
        resolved = (path.parent / local).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError:
            exists = False
        else:
            exists = resolved.exists()
        if not exists:
            violations.append(
                Violation(
                    relative,
                    _line_number(text, match.start()),
                    "broken-link",
                    f"local Markdown target does not exist: {target}",
                )
            )
    return violations


def _check_claims(path: Path, root: Path, text: str) -> list[Violation]:
    relative = _relative(path, root)
    violations: list[Violation] = []

    for match in GITHUB_CLOZN_RE.finditer(text):
        owner = match.group("owner")
        if owner.casefold() != "bkawa-io":
            violations.append(
                Violation(
                    relative,
                    _line_number(text, match.start()),
                    "obsolete-owner",
                    f"Clozn repository URL uses {owner!r}; expected 'bkawa-io'",
                )
            )

    for match in REMOVED_COMMAND_RE.finditer(text):
        violations.append(
            Violation(
                relative,
                _line_number(text, match.start()),
                "removed-command",
                f"current-user documentation invokes removed command: {match.group(0)}",
            )
        )

    section = ""
    for number, line in enumerate(text.splitlines(), start=1):
        heading = HEADING_RE.match(line)
        if heading:
            section = heading.group("title").strip().casefold()
        if ADAPTER_NEGATION_RE.search(line):
            violations.append(
                Violation(
                    relative,
                    number,
                    "adapter-contradiction",
                    "LoRA/diff-adapter is merged but this line says it is unavailable",
                )
            )
        elif "clozn diff-adapter" in line.casefold() and section in {
            "not built",
            "not yet built",
            "unbuilt",
            "unsupported",
        }:
            violations.append(
                Violation(
                    relative,
                    number,
                    "adapter-contradiction",
                    f"diff-adapter appears under contradictory section {section!r}",
                )
            )
    return violations


def check_docs(root: Path) -> list[Violation]:
    root = root.resolve()
    violations: list[Violation] = []
    for path in iter_current_docs(root):
        text = path.read_text(encoding="utf-8")
        violations.extend(_check_claims(path, root, text))
        violations.extend(_check_links(path, root, text))
    return sorted(violations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="check current-user Markdown for high-confidence Clozn documentation drift"
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root (default: inferred from this script)",
    )
    args = parser.parse_args(argv)
    violations = check_docs(args.root)
    for violation in violations:
        print(violation)
    if violations:
        print(f"documentation drift: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print("documentation drift: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
