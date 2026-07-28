#!/usr/bin/env python3
"""clozn -- a boring, reliable front door to the local model engine.

The fast runtime is the C++ engine (clozn-server.exe). This wraps it so the daily path is one command:

    clozn run   <model> "<prompt>"     one-shot, streams tokens to the terminal
    clozn serve <model> [--port 8080]  bring up the OpenAI-compatible endpoint, print the base URL
    clozn models                       discover local GGUFs and the backend that would run them

The CLI/supervisor path uses no Torch. It finds the engine build (GPU preferred), puts the right DLLs on
PATH, picks per-model flags (diffusion
mask tokens, etc.), reports honestly what it's running on, and fails with one actionable line instead of a
stack trace. Model dirs: $CLOZN_MODELS, ~/.clozn/models, <repo>/models, ~/.clozn/config.json["model_dirs"].

This module is the argparse root: the full command tree (build_parser), dispatch (main), and the shared
constants (HOME, CloznError) every other clozn.cli.* module reads at call time via
`from clozn.cli import main as ctx` (never `from clozn.cli.main import HOME`, which would bind a stale
copy immune to a test's monkeypatch or a later change). Each actual `cmd_X` implementation lives in
clozn/cli/commands/*.py, grouped by family; this file only imports and wires them.
"""
from __future__ import annotations

import argparse
import os
import sys

HOME = os.path.expanduser("~/.clozn")


class CloznError(Exception):
    """A clean, user-facing failure -- printed as one line, no traceback."""


def _positive_context(value: str) -> int:
    """argparse type for a real, non-zero worker context window."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("context size must be a positive integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("context size must be a positive integer")
    return parsed


# Imported after HOME/CloznError are defined: every module below reaches back into this one (`from
# clozn.cli import main as ctx` for HOME, `from clozn.cli.main import CloznError`), so this file must
# finish defining both before triggering those imports -- see engine_process.py's module docstring for the
# full circular-import trace this depends on. Order also matters *between* these: commands.serve/run/explain
# import names directly off commands.models (and commands.run), so models/run must load first.
#
# Several of these (_free_port, format_explain, _SPARK, ...) exist here purely as stable
# re-exports: CLI tests written against the pre-split flat module call `clozn.cli.main.<name>` directly, and
# since none of them are mutated globals (they're functions/constants, not DIM/BOLD/RST/COLOR), a plain
# import is safe -- a function always reads its OWN defining module's globals, never a stale copy of them.
from clozn.cli import formatting as fmt                                                       # noqa: E402
from clozn.cli.engine_process import _free_port                                               # noqa: E402
from clozn.cli.formatting import _C_BLUE, _C_HOT, _C_PALE, _SPARK, _conf_rgb, _heatmap_lines   # noqa: E402,F401
from clozn.cli.formatting import _paint, _paint_sparkline, _sparkline, _stream_token           # noqa: E402,F401
from clozn.cli.commands.models import cmd_models, cmd_pull, cmd_plan, format_plan              # noqa: E402
from clozn.cli.commands.models import format_throughput                                       # noqa: E402,F401
from clozn.cli.commands.run import cmd_run                                                    # noqa: E402
from clozn.cli.commands.serve import cmd_serve, cmd_ps, cmd_stop                              # noqa: E402
from clozn.cli.commands.studio import cmd_studio                                              # noqa: E402
from clozn.cli.commands.explain import (cmd_explain, cmd_inspect, cmd_trace, cmd_branch, cmd_prove,  # noqa: E402
                                        format_explain, format_narrate, format_prove, _fetch_explain,
                                        _fetch_narrate, _fetch_prove, _last_run_id, _verified_tag)
from clozn.cli.commands.test import cmd_test                                                  # noqa: E402
from clozn.cli.commands.quant_check import cmd_quant_check, add_subparser as _add_quant_check  # noqa: E402,F401
from clozn.cli.commands.diff_model import cmd_diff_model, add_subparser as _add_diff_model      # noqa: E402,F401
from clozn.cli.commands.eval import cmd_eval, add_subparser as _add_eval                       # noqa: E402,F401
from clozn.cli.commands.test_model import cmd_test_model, add_subparser as _add_test_model      # noqa: E402,F401
from clozn.cli.commands.trace_circuit import cmd_trace_circuit, add_subparser as _add_causal_trace  # noqa: E402
from clozn.cli.commands.ci_check import cmd_ci_baseline, cmd_ci_check, add_subparser as _add_ci_check  # noqa: E402,F401
from clozn.cli.commands.experiment_suite import add_subparser as _add_experiment_suite                  # noqa: E402
from clozn.cli.commands.smoke import cmd_smoke                                                  # noqa: E402
from clozn.cli.commands.version import cmd_version                                              # noqa: E402
from clozn.cli.commands.doctor import cmd_doctor                                                # noqa: E402
from clozn.cli.commands.context import add_subparser as _add_context                             # noqa: E402
from clozn.cli.commands.watch import add_subparser as _add_watch                                 # noqa: E402
from clozn.cli.commands.connect import add_subparser as _add_connect                             # noqa: E402
from clozn.cli.commands.retry import add_subparser as _add_retry                                 # noqa: E402
from clozn.cli.commands.diagnose import add_subparser as _add_diagnose                           # noqa: E402
from clozn.cli.commands.regression_suite import add_subparser as _add_regression_suite            # noqa: E402
from clozn.cli.commands.runs_privacy import add_subparser as _add_runs_privacy                    # noqa: E402
from clozn.cli.commands.provenance import add_subparser as _add_provenance                        # noqa: E402
from clozn.cli.commands import _autoload                                                          # noqa: E402


def build_parser():
    """The full argparse tree, factored out of main() so tests can introspect flags without dispatching."""
    p = argparse.ArgumentParser(prog="clozn", description="a reliable front door to the local model engine")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="one-shot: stream a completion to the terminal")
    pr.add_argument("model"); pr.add_argument("prompt", nargs="?", default=None)
    pr.add_argument("--max", type=int, default=256, help="max new tokens (default 256)")
    pr.add_argument("--ctx", type=_positive_context, default=None,
                    help="worker context window in tokens (default 4096; reduce on tight unified memory)")
    pr.add_argument("--cpu", action="store_true", help="force the CPU build")
    pr.add_argument("--port", type=int, default=0); pr.add_argument("--mask", type=int, default=None)
    pr.add_argument("--eos", type=int, default=None)
    pr.add_argument("--heat", action="store_true", help="paint each token as it streams by the model's "
                    "confidence (warm = wavered, cool = sure) -- the denoise heatmap, live (AR models)")
    pr.set_defaults(fn=cmd_run)

    ps = sub.add_parser("serve", help="bring up the OpenAI-compatible endpoint")
    ps.add_argument("model"); ps.add_argument("--port", type=int, default=0)
    ps.add_argument("--ctx", type=_positive_context, default=None,
                    help="worker context window in tokens (default 4096; reduce on tight unified memory)")
    ps.add_argument("--cpu", action="store_true"); ps.add_argument("--mask", type=int, default=None)
    ps.add_argument("--eos", type=int, default=None)
    ps.add_argument("--sae", default=None, help="on-device SAE readout dir (dims must match the model; "
                    "server refuses politely on mismatch)")
    ps.add_argument("--sae-k", type=int, default=None, help="SAE features kept per position (default 16)")
    # NB: this flag does NOT gate the Studio Sources lens, despite what this help text claimed until
    # 2026-07-27. That lens calls GET /runs/<id>/influence-map, which routes to
    # receipts/context_answer_influence.py's forced-scoring path and never touches attn_knockout --
    # verified by the absence of any attn_knockout reference in server/routes/influence_map.py. Only
    # `clozn provenance` (analysis/provenance.py's attention-edge knockout) needs it. The old wording
    # told users to pay a decode-speed cost for a lens that already worked without it.
    ps.add_argument("--no-flash-attn", action="store_true",
                    help="materialize attention weights instead of fusing them, so /score's "
                         "attn_knockout works and `clozn provenance` becomes available "
                         "(GET /health.capabilities.attn_knockout flips true). Off by "
                         "default; costs decode speed. Not needed for the Studio Sources lens.")
    ps.set_defaults(fn=cmd_serve)

    sub.add_parser("models", help="list local models + the engine backend").set_defaults(fn=cmd_models)
    pp = sub.add_parser("pull", help="download a model GGUF (by name, or owner/repo/file.gguf)")
    pp.add_argument("model"); pp.set_defaults(fn=cmd_pull)
    ppl = sub.add_parser("plan", help="will it fit? read a GGUF's header (no download, no load, no GPU) "
                         "before you commit to a multi-GB pull")
    ppl.add_argument("model", help="a known model name, a local .gguf path, or a HF resolve/... .gguf URL")
    ppl.add_argument("--vram", type=float, default=None,
                     help="VRAM budget in GB (default: detect via nvidia-smi, else 16)")
    ppl.add_argument("--bandwidth-gb-s", type=float, default=None,
                     help="assumed effective memory bandwidth in GB/s for the decode-throughput roofline "
                          "predictor (default: 900 GB/s, RTX-5080-class -- a model-free estimate, stated "
                          "explicitly since it drives the whole prediction; see `clozn plan`'s output)")
    ppl.add_argument("--calibrate", action="store_true",
                     help="DEFERRED: would boot the engine and measure ACTUAL tok/s to correct the "
                          "bandwidth assumption -- not implemented yet (prints a stub explaining why), "
                          "never boots anything")
    ppl.set_defaults(fn=cmd_plan)
    pst = sub.add_parser("studio", help="attach to the Studio served by a running Clozn runtime")
    pst.add_argument("--port", type=int, default=0); pst.add_argument("--open", action="store_true", help="open the UI in your browser")
    pst.set_defaults(fn=cmd_studio)
    psmoke = sub.add_parser("smoke", help="live acceptance test for the one-gateway product runtime")
    psmoke.add_argument("model", nargs="?", help="model name/path; omitted when attaching with --url")
    psmoke.add_argument("--url", default=None, help="attach to an existing gateway instead of launching one")
    psmoke.add_argument("--port", type=int, default=0, help="managed gateway port (default: choose a free port)")
    psmoke.add_argument("--cpu", action="store_true", help="force the CPU worker build")
    psmoke.add_argument("--preflight", action="store_true", help="only audit model/build/asset prerequisites")
    psmoke.add_argument("--deep", action="store_true", help="also exercise forced receipts and replay")
    restart = psmoke.add_mutually_exclusive_group()
    restart.add_argument("--restart-worker", dest="restart_worker", action="store_true",
                         help="kill and recover the registered private worker")
    restart.add_argument("--no-restart-worker", dest="restart_worker", action="store_false",
                         help="skip the worker recovery check")
    psmoke.add_argument("--timeout", type=float, default=120.0, help="per-request timeout in seconds")
    psmoke.add_argument("--startup-timeout", type=float, default=240.0,
                        help="startup/restart timeout in seconds")
    psmoke.add_argument("--json", action="store_true", help="print a machine-readable report")
    psmoke.set_defaults(fn=cmd_smoke, restart_worker=None)
    sub.add_parser("ps", help="list running product runtimes").set_defaults(fn=cmd_ps)
    pstop = sub.add_parser("stop", help="stop a product runtime (by model name, port, or 'all')")
    pstop.add_argument("which"); pstop.set_defaults(fn=cmd_stop)
    pt = sub.add_parser("trace", help="inspect the last run journal entry's confidence timeline")
    pt.add_argument("--list", action="store_true", help="list recent run journal entries instead of showing the last")
    pt.set_defaults(fn=cmd_trace)
    pb = sub.add_parser("branch", help="re-run from an uncertain point on the alternative (the road not taken)")
    pb.add_argument("--at", type=int, default=None, help="token index to fork at (default: the most uncertain)")
    pb.add_argument("--pick", type=int, default=0, help="which alternative to take (0 = the runner-up)")
    pb.add_argument("--max", type=int, default=80); pb.add_argument("--cpu", action="store_true")
    pb.set_defaults(fn=cmd_branch)
    pe = sub.add_parser("explain", help="explain a run: hesitations, active influences, concepts "
                        "(needs a running Clozn gateway)")
    pe.add_argument("run_id", nargs="?", default=None, help="run id, as shown in the Studio's Runs list")
    pe.add_argument("--last", action="store_true", help="use the most recently recorded run")
    pe.add_argument("--port", type=int, default=0, help="Clozn gateway port (default 8080)")
    pe.add_argument("--why", action="store_true", help="also generate the accountable-self narration (M4): "
                    "a receipt-constrained \"why\", diffed against an independent judge and flagged wherever "
                    "it overclaims. Opt-in -- unlike the rest of `explain`, this GENERATES (two model calls; "
                    "needs a running Clozn gateway)")
    pe.set_defaults(fn=cmd_explain)
    pv = sub.add_parser("prove", help="leave-one-out causal receipts for a run's active influences, "
                        "+ opt-in coalition/Shapley credit (needs a running Clozn gateway)")
    pv.add_argument("run_id", nargs="?", default=None, help="run id, as shown in the Studio's Runs list")
    pv.add_argument("--last", action="store_true", help="use the most recently recorded run")
    pv.add_argument("--port", type=int, default=0, help="Clozn gateway port (default 8080)")
    pv.add_argument("--mode", default="regen", choices=["regen", "forced", "both"],
                    help="receipt mode (default regen -- both-arms-greedy regeneration)")
    pv.add_argument("--coalitions", action="store_true",
                    help="opt-in (docs/PRODUCT_ROADMAP.md §8 tail): pairwise coalition deltas, a Shapley "
                         "approximation, and the interaction gap, on top of the default leave-one-out "
                         "receipts above. Never changes the default output when omitted.")
    pv.add_argument("--coalitions-batch", dest="coalitions_batch", default="auto",
                    choices=["auto", "off", "approximate"],
                    help="batching policy for coalition arms through a substrate's optional /v1/branch-"
                         "backed batching hook (default auto: trust only a self-certified-exact substrate, "
                         "cross-check an uncertified one against sequential, else run sequential)")
    pv.add_argument("--json", action="store_true", help="print the raw receipts JSON")
    pv.set_defaults(fn=cmd_prove)
    pi = sub.add_parser("inspect", help="inspect a returned clozn_run_id (local journal first; no generation)")
    pi.add_argument("run_id", nargs="?", default=None, help="clozn_run_id from an API response/header")
    pi.add_argument("--last", action="store_true", help="inspect the most recent organic run")
    pi.add_argument("--port", type=int, default=0,
                    help="gateway fallback when the id is not in the local journal (default 8080)")
    pi.add_argument("--json", action="store_true", help="print the exact explanation object as JSON")
    pi.set_defaults(fn=cmd_inspect)
    pte = sub.add_parser("test", help="run tiny-test assertions against a stored run (the receipt/replay seams)")
    pte.add_argument("file", help="path to a JSON tiny-test spec (see clozn/testkit/runner.py's module docstring)")
    pte.add_argument("--json", action="store_true",
                     help="print the machine-readable suite result instead of the report")
    pte.add_argument("--attach", action="store_true",
                     help="write results into each touched run's tiny_tests field (rides the receipt_bundle export)")
    pte.add_argument("--live", action="store_true",
                     help="permit causal (leans_on) assertions to run against a live product gateway; "
                          "without it they're honestly skipped ('needs --live'), never silently passed")
    pte.add_argument("--port", type=int, default=0, help="Clozn gateway port for --live (default 8080)")
    pte.set_defaults(fn=cmd_test)
    _add_quant_check(sub)   # `clozn quant-check <A> <B>` — quant-ladder receipts (Tier-1)
    _add_diff_model(sub)    # `clozn diff-model <ref> <candidate>` — base-vs-fine-tune/merge receipts (§4.1)
    _add_eval(sub)          # `clozn eval` — outcome-grounded calibration (Brier/ECE/risk-coverage)
    _add_test_model(sub)    # `clozn test-model` — the model's own CI: pinned probes vs. a golden fixture
    _add_causal_trace(sub)  # `clozn causal-trace` — intervention-validated causal tracing
    _add_ci_check(sub)      # `clozn ci baseline`/`clozn ci check` — headless CI gate (§4.4)
    _add_experiment_suite(sub)  # `clozn experiment run/show` — versioned case x variant x seed object (§4.2)
    _add_context(sub)           # `clozn context last` — delivered vs survived prompt receipt (Phase 2.4)
    _add_watch(sub)             # `clozn watch` — insertion-ordered run tail/correlation (Phase 2.6)
    _add_connect(sub)           # `clozn connect aider` — safe third-party app config with backup
    _add_retry(sub)             # `clozn retry last` — prompt-first corrective compare + scoped undo
    _add_diagnose(sub)          # `clozn diagnose last` — evidence-only latency/cutoff diagnosis
    _add_regression_suite(sub)  # `clozn suite create` — promote captured app runs into Model CI cases
    _add_runs_privacy(sub)      # `clozn runs` — local journal privacy controls and telemetry export
    _add_provenance(sub)        # `clozn provenance` — attention-knockout context-vs-parametric receipt
    # Every command module that sets CLOZN_AUTOLOAD = True, registered without another line here. Runs
    # LAST so an autoloaded module reaching back into this one finds it fully defined, and so the
    # hand-wired order above (models before serve/run/explain) is already settled. See _autoload.py.
    _autoload.register_all(sub)
    sub.add_parser("version", help="print the installed clozn version (+ git commit if available)"
                   ).set_defaults(fn=cmd_version)
    pdoc = sub.add_parser("doctor", help="diagnose this install; --verify-offline adds a strict "
                          "local-only enforcement gate")
    pdoc.add_argument("--json", action="store_true", help="print a machine-readable report")
    pdoc.add_argument("--verify-offline", action="store_true",
                      help="fail unless local-only enforcement is active and its ledger window is clean")
    pdoc.add_argument("--deep", action="store_true",
                      help="also run a model-free process-start check on the discovered engine binary "
                           "(roadmap feature 01: one extra subprocess launch, still no model/network)")
    pdoc.set_defaults(fn=cmd_doctor)
    return p


def main(argv=None):
    fmt._setup_console()
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help(); return 2
    try:
        rc = args.fn(args)
        return rc if isinstance(rc, int) else 0
    except CloznError as e:
        print(f"{fmt.BOLD}clozn:{fmt.RST} {e}", file=sys.stderr); return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
