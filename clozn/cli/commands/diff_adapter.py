"""`clozn diff-adapter MODEL ADAPTER` -- what did my fine-tune adapter actually change?

The fine-tune author's one command. Boots ONE base model twice -- once plain, once with the LoRA
attached -- and runs the same teacher-forced diff ladder `clozn diff-model` uses, so the answer is
per-token behavior receipts rather than a vibe.

WHY THIS IS A CLEANER COMPARISON THAN diff-model
------------------------------------------------
`clozn diff-model base.gguf finetune.gguf` compares two separate files, and has to defend against a
whole class of confounds: the two may not share a tokenizer (it refuses outright when they don't), they
may carry different chat templates (there is a whole --own-templates policy for that), and any observed
difference could be attributable to the tokenizer, the template, the quantization, or the weights.

Here, both arms are the SAME model file, the SAME tokenizer, the SAME chat template, the SAME
quantization, and the same engine build. Every one of those confounds is held constant BY CONSTRUCTION
rather than by preflight check. The only difference between the arms is the adapter's weight delta, so
a behavioral difference is attributable to it and nothing else. That is a materially stronger claim than
diff-model can make, and it is the reason this is a separate command rather than a flag on that one.

The tokenizer/template preflights still run -- they cost nothing and would catch a genuinely broken
engine -- but they are expected to pass trivially, and the report says so rather than presenting a
tautology as evidence.

ON THE BASELINE ARM
-------------------
Arm A serves the base model with no adapter at all, rather than the same adapter at --adapter-scale 0.
Those two are equivalent -- the engine work that added adapter support proved the scale-0 arm is
byte-identical to the no-adapter arm -- and the no-adapter arm is the simpler thing to explain in a
report. Use `clozn serve --adapter X --adapter-scale 0` directly if you want the loaded-but-inert
control for some other purpose.

Like `diff-model`, this is THE LIVE PATH: it needs a free GPU and two engine processes, so it is not
exercised by this module's own model-free tests, which cover argument wiring and the flag contract.
"""
from __future__ import annotations

import json
import os
import sys

from clozn.cli import formatting as fmt
from clozn.cli import main as ctx
from clozn.cli.commands import quant_check as qc
from clozn.cli.commands.diff_model import format_diff_model_report, run_diff_model
from clozn.cli.commands.models import _flags_for, resolve_model
from clozn.cli.engine_process import _free_port, spawn_engine

CLOZN_AUTOLOAD = True


def add_subparser(sub):
    pd = sub.add_parser("diff-adapter",
                        help="base vs base+LoRA per-token behavior receipts -- what did your fine-tune "
                             "adapter actually change? (tokenizer, template and quantization are held "
                             "constant by construction, so the difference is the adapter's weights)")
    pd.add_argument("model", help="the BASE model -- a known short name, local GGUF path, or fuzzy "
                                  "filename fragment, resolved like `clozn run`'s model arg")
    pd.add_argument("adapter", help="the LoRA adapter GGUF under test (as produced by llama.cpp's "
                                    "convert_lora_to_gguf.py)")
    pd.add_argument("--adapter-scale", type=float, default=None, metavar="F",
                    help="multiplier on the adapter's weight delta for the candidate arm (default 1.0)")
    pd.add_argument("--runs", type=int, default=8,
                    help="how many runs to diff: fresh greedy prompts generated under the base model "
                         "(default), or the N most recent run-journal entries with --from-log")
    pd.add_argument("--from-log", action="store_true",
                    help="diff the N most recent runs from your own run journal instead of fresh prompts")
    pd.add_argument("--topk", type=int, default=8,
                    help="topk requested on every /score call -- rank 0 IS that arm's argmax, needed for "
                         "flip detection (default 8)")
    pd.add_argument("--max-tokens", type=int, default=200,
                    help="max tokens for a FRESH generation under the base model (ignored with --from-log)")
    pd.add_argument("--port-a", type=int, default=0, help="port for the base engine (default: a free port)")
    pd.add_argument("--port-b", type=int, default=0,
                    help="port for the adapter engine (default: a free port)")
    pd.add_argument("--cpu", action="store_true", help="force the CPU build for both engines")
    pd.add_argument("--json", action="store_true",
                    help="print the raw result as JSON instead of the text report")
    pd.add_argument("--both", action="store_true",
                    help="also run the adapter-anchored reverse ladder (generate under base+adapter, "
                         "teacher-force under both) -- the target-gain view, in addition to the default "
                         "base-anchored forgetting/no-op view")
    # Deliberately NOT exposing --own-templates: both arms are the same model file and therefore the same
    # chat template, so the policy it selects between does not exist here. Offering it would imply a
    # degree of freedom this comparison does not have.
    pd.set_defaults(fn=cmd_diff_adapter)
    return pd


def cmd_diff_adapter(args):
    """Boot base and base+adapter, then delegate to diff_model's ladder.

    Always tears down both engines it spawned, even on error -- including a tokenizer-preflight refusal,
    which raises through the `finally` exactly as cmd_diff_model's does.
    """
    EngineClient = qc._import_engine_client()

    model = resolve_model(args.model)
    adapter = os.path.abspath(os.path.expanduser(args.adapter))
    if not os.path.isfile(adapter):
        # Fail before booting anything: a missing adapter is a typo, and two model loads is an expensive
        # way to discover one.
        raise ctx.CloznError(f"adapter not found: {adapter}")

    scale = 1.0 if args.adapter_scale is None else args.adapter_scale
    base_label = os.path.splitext(os.path.basename(model))[0]
    adapter_label = os.path.splitext(os.path.basename(adapter))[0]
    label_a = f"{base_label} (base)"
    label_b = f"{base_label} + {adapter_label}"
    if scale != 1.0:
        label_b += f" @{scale:g}"

    port_a = args.port_a or _free_port()
    port_b = args.port_b or _free_port()
    prefer_gpu = not args.cpu

    # Same flags on both arms except the adapter -- that identity is what makes the comparison clean, so
    # it is built from one dict rather than two independently-derived ones.
    flags_a = _flags_for(model)
    flags_b = dict(flags_a)
    flags_b["adapter"] = adapter
    flags_b["adapter_scale"] = scale

    proc_a = proc_b = None
    try:
        print(f"{fmt.DIM}- booting {base_label} (base) on port {port_a}...{fmt.RST}", file=sys.stderr)
        proc_a, _health_a, _gpu_a = spawn_engine(model, port_a, flags_a, prefer_gpu=prefer_gpu)
        print(f"{fmt.DIM}- booting {base_label} + adapter (scale {scale:g}) on port {port_b}...{fmt.RST}",
              file=sys.stderr)
        # The engine REFUSES to start when an adapter will not attach rather than serving the base model,
        # so a mismatched adapter surfaces here as a boot failure -- never as a diff that silently
        # compared the base model against itself and reported "no detectable difference".
        proc_b, _health_b, _gpu_b = spawn_engine(model, port_b, flags_b, prefer_gpu=prefer_gpu)

        eng_a = EngineClient(port=port_a)
        eng_b = EngineClient(port=port_b)
        result = run_diff_model(eng_a, eng_b, args, label_a=label_a, label_b=label_b)
    finally:
        for proc in (proc_a, proc_b):
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    # Record what this comparison controlled for. The tokenizer/template preflights pass trivially here
    # (same model file on both arms), and reporting that as though it were evidence would dress up a
    # tautology; naming it as held-constant-by-construction is the honest version, and it is also the
    # stronger claim.
    result["comparison"] = {
        "kind": "base_vs_adapter",
        "adapter_path": adapter,
        "adapter_scale": scale,
        "held_constant_by_construction": ["model_file", "tokenizer", "chat_template", "quantization",
                                          "engine_build"],
        "sole_difference": "adapter_weight_delta",
    }

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_diff_model_report(result))
        print(f"\n{fmt.DIM}base vs base+adapter: model file, tokenizer, chat template, quantization and "
              f"engine build are identical across both arms by construction, so any difference above is "
              f"the adapter's weight delta.{fmt.RST}")
    return 0
