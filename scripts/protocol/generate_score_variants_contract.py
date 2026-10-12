#!/usr/bin/env python3
"""Generate the Python and C++ score-variants constants from one contract fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "protocol" / "fixtures" / "score_variants.json"
PYTHON_OUTPUT = ROOT / "engine" / "client" / "clozn_engine" / "_score_variants_contract.py"
CPP_OUTPUT = ROOT / "engine" / "core" / "include" / "clozn" / "score_variants_contract.generated.hpp"


def _python_literal(value: object) -> str:
    if value is False:
        return "False"
    if value is True:
        return "True"
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return "(" + ", ".join(_python_literal(item) for item in value) + ",)"
    raise TypeError(f"unsupported generated value: {value!r}")


def render_python(contract: dict) -> str:
    limits = contract["limits"]
    execution = contract["execution"]
    values = {
        "SCORE_VARIANTS_ENDPOINT": contract["endpoint"],
        "SCORE_VARIANTS_CAPABILITY": contract["capability"],
        "SCORE_VARIANTS_SCHEMA": contract["schema_version"],
        "SCORE_VARIANTS_MAX_VARIANTS": limits["max_variants"],
        "SCORE_VARIANTS_MAX_VARIANT_ID_BYTES": limits["max_variant_id_bytes"],
        "SCORE_VARIANTS_MAX_PROMPT_ROWS_PER_CONTEXT": limits["max_prompt_rows_per_context"],
        "SCORE_VARIANTS_MAX_CHECKPOINT_ID_BYTES": limits["max_checkpoint_id_bytes"],
        "SCORE_VARIANTS_MAX_WORKER_GENERATION_ID_BYTES": limits["max_worker_generation_id_bytes"],
        "SCORE_VARIANTS_REQUEST_KEYS": contract["request_keys"],
        "SCORE_VARIANT_KEYS": contract["variant_keys"],
        "SCORE_VARIANTS_GENERATION_STATE_KEYS": contract["generation_state_keys"],
        "SCORE_VARIANTS_TOP_STATUSES": contract["top_statuses"],
        "SCORE_VARIANTS_ROW_STATUSES": contract["row_statuses"],
        "SCORE_VARIANTS_STRATEGY_PREFIX": execution["strategy_prefix"],
        "SCORE_VARIANTS_NUMERICAL_REGIME": execution["numerical_regime"],
        "SCORE_VARIANTS_EVIDENCE_USE": execution["evidence_use"],
        "SCORE_VARIANTS_QUALIFIED": execution["qualified"],
    }
    lines = [
        '"""Generated from protocol/fixtures/score_variants.json. Do not edit by hand."""',
        "",
    ]
    lines.extend(f"{name} = {_python_literal(value)}" for name, value in values.items())
    lines.append("")
    return "\n".join(lines)


def _cpp_string(value: str) -> str:
    return json.dumps(value)


def _cpp_array(name: str, values: list[str]) -> str:
    members = ", ".join(_cpp_string(value) for value in values)
    return (
        f"inline constexpr std::array<std::string_view, {len(values)}> {name} = "
        f"{{{members}}};"
    )


def render_cpp(contract: dict) -> str:
    limits = contract["limits"]
    execution = contract["execution"]
    return "\n".join([
        "// Generated from protocol/fixtures/score_variants.json. Do not edit by hand.",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <string_view>",
        "",
        "namespace clozn::score_variants_contract {",
        "",
        f"inline constexpr std::string_view ENDPOINT = {_cpp_string(contract['endpoint'])};",
        f"inline constexpr std::string_view CAPABILITY = {_cpp_string(contract['capability'])};",
        f"inline constexpr std::string_view SCHEMA_VERSION = {_cpp_string(contract['schema_version'])};",
        f"inline constexpr std::size_t MAX_VARIANTS = {limits['max_variants']};",
        f"inline constexpr std::size_t MAX_VARIANT_ID_BYTES = {limits['max_variant_id_bytes']};",
        f"inline constexpr long long MAX_PROMPT_ROWS_PER_CONTEXT = {limits['max_prompt_rows_per_context']};",
        f"inline constexpr std::size_t MAX_CHECKPOINT_ID_BYTES = {limits['max_checkpoint_id_bytes']};",
        f"inline constexpr std::size_t MAX_WORKER_GENERATION_ID_BYTES = {limits['max_worker_generation_id_bytes']};",
        _cpp_array("REQUEST_KEYS", contract["request_keys"]),
        _cpp_array("VARIANT_KEYS", contract["variant_keys"]),
        _cpp_array("GENERATION_STATE_KEYS", contract["generation_state_keys"]),
        _cpp_array("TOP_STATUSES", contract["top_statuses"]),
        _cpp_array("ROW_STATUSES", contract["row_statuses"]),
        f"inline constexpr std::string_view STRATEGY_PREFIX = {_cpp_string(execution['strategy_prefix'])};",
        f"inline constexpr std::string_view NUMERICAL_REGIME = {_cpp_string(execution['numerical_regime'])};",
        f"inline constexpr std::string_view EVIDENCE_USE = {_cpp_string(execution['evidence_use'])};",
        f"inline constexpr bool QUALIFIED = {'true' if execution['qualified'] else 'false'};",
        "",
        "}  // namespace clozn::score_variants_contract",
        "",
    ])


def _write_or_check(path: Path, expected: str, *, check: bool) -> bool:
    if check:
        return path.is_file() and path.read_text(encoding="utf-8") == expected
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    document = json.loads(SOURCE.read_text(encoding="utf-8"))
    contract = document["contract"]
    outputs = {
        PYTHON_OUTPUT: render_python(contract),
        CPP_OUTPUT: render_cpp(contract),
    }
    stale = [path for path, content in outputs.items()
             if not _write_or_check(path, content, check=args.check)]
    if stale:
        for path in stale:
            print(f"stale generated contract: {path.relative_to(ROOT)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
