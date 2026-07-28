"""Static, stdlib-only LoRA GGUF validation and pinned conversion guidance."""
from __future__ import annotations

import json
import os

from clozn.cli.engine_process import REPO
from clozn.cli.commands.models import resolve_model
from clozn.cli.commands.validate_export import inspect_adapter, inspect_model

CLOZN_AUTOLOAD = True

LLAMA_CPP_COMMIT = "88a39274ecf88ba11686acd357b59685b1cbf03d"
_RELATIVE_CONVERTER = os.path.join(
    "engine", "core", "third_party", "llama.cpp", "convert_lora_to_gguf.py")


def conversion_instructions(peft_dir: str = "PEFT_DIR", *,
                            base_dir: str = "HF_BASE_DIR",
                            out_path: str = "tune.gguf") -> dict:
    """Exact external command; executing it remains opt-in and outside product dependencies."""
    local_script = os.path.join(REPO, _RELATIVE_CONVERTER)
    if os.path.isfile(local_script):
        setup = []
        script = local_script
    else:
        setup = [
            "git clone https://github.com/ggml-org/llama.cpp.git",
            f"git -C llama.cpp checkout {LLAMA_CPP_COMMIT}",
        ]
        script = os.path.join("llama.cpp", "convert_lora_to_gguf.py")
    command = (
        f'python "{script}" --base "{base_dir}" --outfile "{out_path}" "{peft_dir}"'
    )
    return {
        "tool": "ggml-org/llama.cpp convert_lora_to_gguf.py",
        "tool_commit": LLAMA_CPP_COMMIT,
        "setup": setup,
        "command": command,
        "verify": f'clozn adapter validate "{out_path}" --base BASE.gguf',
        "dependency_note": (
            "run the converter in a separate environment containing its optional Torch, "
            "Transformers, and Safetensors dependencies; Clozn does not import or install them"
        ),
    }


def validate_adapter(path: str, base_identity: dict | None = None) -> dict:
    adapter = inspect_adapter(path)
    checks = [
        {
            "name": "lora_gguf",
            "passed": adapter.get("format") == "lora_gguf",
            "detail": f"detected {adapter.get('format', 'unknown')}",
        },
        {
            "name": "architecture_declared",
            "passed": bool(adapter.get("architecture")),
            "detail": f"adapter architecture={adapter.get('architecture', 'missing')!r}",
        },
    ]
    if base_identity is not None:
        checks.append({
            "name": "base_architecture_match",
            "passed": (
                bool(adapter.get("architecture"))
                and adapter.get("architecture") == base_identity.get("architecture")
            ),
            "detail": (
                f"base={base_identity.get('architecture')!r}; "
                f"adapter={adapter.get('architecture')!r}"
            ),
        })
    return {
        "adapter": adapter,
        **({"base": base_identity} if base_identity is not None else {}),
        "checks": checks,
        "valid": all(check["passed"] for check in checks),
        "conversion": conversion_instructions(),
    }


def _format(report: dict) -> str:
    lines = [
        f"adapter validate: {'PASS' if report['valid'] else 'FAIL'}",
        f"  path: {report['adapter']['path']}",
        f"  sha256: {report['adapter']['sha256']}",
    ]
    for check in report["checks"]:
        lines.append(
            f"  [{'PASS' if check['passed'] else 'FAIL'}] "
            f"{check['name']}: {check['detail']}")
    conversion = report["conversion"]
    lines.extend([
        "",
        "Pinned PEFT/Safetensors conversion (external optional environment):",
        f"  llama.cpp commit: {conversion['tool_commit']}",
    ])
    lines.extend(f"  {command}" for command in conversion["setup"])
    lines.extend([
        f"  {conversion['command']}",
        f"  {conversion['verify']}",
        f"  {conversion['dependency_note']}",
    ])
    return "\n".join(lines)


def add_subparser(sub):
    adapter = sub.add_parser(
        "adapter",
        help="validate a LoRA GGUF and show pinned PEFT/Safetensors conversion instructions")
    actions = adapter.add_subparsers(dest="adapter_command", required=True)
    validate = actions.add_parser(
        "validate",
        help="inspect LoRA GGUF metadata without importing optional ML dependencies")
    validate.add_argument("path", help="LoRA GGUF produced by convert_lora_to_gguf.py")
    validate.add_argument("--base", help="optional base GGUF for architecture compatibility")
    validate.add_argument("--json", action="store_true", help="print the validation report as JSON")
    validate.set_defaults(fn=cmd_adapter_validate)
    return adapter


def cmd_adapter_validate(args):
    from clozn.cli import main as ctx

    if os.path.isdir(os.path.abspath(os.path.expanduser(args.path))):
        instructions = conversion_instructions(args.path)
        message = (
            "input is a directory, not a LoRA GGUF; convert it first with: "
            + " ; ".join(instructions["setup"] + [instructions["command"]])
        )
        raise ctx.CloznError(message)
    try:
        base = inspect_model(resolve_model(args.base)) if args.base else None
        report = validate_adapter(args.path, base)
    except Exception as error:
        instructions = conversion_instructions(args.path)
        raise ctx.CloznError(
            f"could not validate adapter: {error}. Pinned converter "
            f"{LLAMA_CPP_COMMIT}: {instructions['command']}") from None
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_format(report))
    return 0 if report["valid"] else 1
