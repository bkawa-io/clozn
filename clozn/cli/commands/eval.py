"""commands.eval -- `clozn eval [--set easy|hard|arith|both|all|extended]`: outcome-grounded calibration on
a live endpoint.

The thin CLI shell around clozn.eval.bench: run the built-in factual probe set through a RUNNING Clozn
gateway, grade each answer against gold (eval.outcome), read each reply's answer-span confidence from its
logged run trace, and print Brier / ECE-vs-truth / a risk-coverage curve for selective generation. This
is the TRUTH tier actuary.py flags as missing -- calibration against correctness, not the acceptance proxy.

Needs a running Clozn gateway (default http://127.0.0.1:8080); it reads per-token confidence from the run
journal, which the OpenAI wire format omits. Small built-in n -- a directional demonstration that the
model's own confidence separates its right answers from its wrong ones, not a benchmark score.

Registration in clozn/cli/main.py mirrors quant-check exactly: import `cmd_eval, add_subparser as
_add_eval` alongside the other commands.* imports, and call `_add_eval(sub)` in build_parser() before
`return p`.
"""
from __future__ import annotations

import json
import math
import unicodedata


_PROBE_SETS = ("easy", "hard", "arith", "both", "all", "extended")
_SCORE_AGGREGATES = ("min", "mean")


def add_subparser(sub):
    """Register `clozn eval` on an argparse subparsers object (own function so its wiring is testable
    without dispatching; mirrors commands.quant_check.add_subparser)."""
    pe = sub.add_parser("eval", help="outcome-grounded calibration: run a factual probe set through a live "
                        "gateway and score Brier/ECE/risk-coverage against TRUTH (needs a running endpoint)")
    pe.add_argument("--url", default="http://127.0.0.1:8080", help="Clozn gateway base URL (default :8080)")
    pe.add_argument("--set", dest="which", default="arith",
                    choices=_PROBE_SETS,
                    help="which built-in probe set (default: arith -- programmatic, guaranteed golds, "
                         "graded errors; 'easy'/'hard' are curated factual sets; 'extended' is the v2 set "
                         "-- reasoning puzzles, common misconceptions, and trick questions; 'all' folds "
                         "every set together)")
    pe.add_argument("--score", default="min", choices=["min", "mean"],
                    help="answer-span aggregate used as the abstention signal (default: min token conf)")
    pe.add_argument("--target-error", type=float, default=0.05, dest="target_error",
                    help="the selective-generation error budget the recommended policy is tuned to (0.05)")
    pe.add_argument("--json", action="store_true", help="print the raw calibration report as JSON")
    pe.add_argument("--save", action="store_true", help="persist this report so the studio can serve it "
                    "as the TRUTH-tier curve at GET /journal/calibration (beside the proxy at /journal/actuary)")
    pe.add_argument("--task", default="general",
                    help="task label for a model/task-specific calibration profile (default: general)")
    pe.add_argument("--wizard", action="store_true",
                    help="guided calibration: confirm task, probes, score aggregate, error budget, and saving")
    pe.add_argument("--list-profiles", action="store_true",
                    help="list installed model/task calibration profiles without running live probes")
    pe.set_defaults(fn=cmd_eval)

    eval_sub = pe.add_subparsers(dest="eval_cmd")
    policy = eval_sub.add_parser("policy", help="inspect the calibrated selective-generation policy")
    policy.set_defaults(fn=_no_policy_command)
    policy_sub = policy.add_subparsers(dest="policy_cmd")
    show = policy_sub.add_parser("show", help="print the active policy for a model/task + its scope caveats")
    show.add_argument("--model", default=None,
                      help="exact model identity (default: auto-detect from a live gateway at --url)")
    show.add_argument("--task", default=None,
                      help="task label (default: that model's newest saved profile)")
    show.add_argument("--url", default="http://127.0.0.1:8080",
                      help="Clozn gateway base URL used only to auto-detect the live model (default :8080)")
    show.add_argument("--json", action="store_true", help="print the raw policy report as JSON")
    show.set_defaults(fn=_cmd_policy_show)
    return pe


def _task_label(value) -> str:
    raw = str(value or "")
    if any(unicodedata.category(char) == "Cc" for char in raw):
        raise ValueError("task must be a non-empty label of at most 80 characters without control characters")
    task = " ".join(raw.strip().split()).lower()
    if not task or len(task) > 80:
        raise ValueError("task must be a non-empty label of at most 80 characters without control characters")
    return task


def _target_error(value) -> float:
    try:
        target = float(value)
    except (TypeError, ValueError):
        raise ValueError("target error must be a number from 0 to 1") from None
    if not math.isfinite(target) or target < 0 or target > 1:
        raise ValueError("target error must be a finite number from 0 to 1")
    return target


def _read_answer(prompt: str) -> str | None:
    """One wizard read. Redirected/closed stdin means "accept safe defaults", never a crash."""
    try:
        return input(prompt).strip()
    except (EOFError, OSError, StopIteration):
        return None


def _choose(prompt: str, default, parse, choices: tuple[str, ...] | None = None):
    """Prompt at most twice, then retain the already-validated/default value."""
    for _ in range(2):
        raw = _read_answer(prompt)
        if raw is None or raw == "":
            return default
        try:
            value = parse(raw)
            if choices is not None and value not in choices:
                raise ValueError(f"choose one of {', '.join(choices)}")
            return value
        except ValueError as exc:
            print(f"  {exc}; press Enter to keep {default!s}")
    return default


def _choose_save(default: bool) -> bool:
    raw = _read_answer(f"Save this model/task profile? [{'Y/n' if default else 'y/N'}]: ")
    if raw is None or raw == "":
        return default
    answer = raw.lower()
    if answer in {"y", "yes"}:
        return True
    if answer in {"n", "no"}:
        return False
    print(f"  unrecognized answer; using {'yes' if default else 'no'}")
    return default


def _wizard(args) -> tuple[str, str, str, float, bool]:
    task = _choose(f"Task label [{args.task}]: ", args.task, _task_label)
    which = _choose(f"Probe set ({'/'.join(_PROBE_SETS)}) [{args.which}]: ",
                    args.which, str, _PROBE_SETS)
    score = _choose(f"Token score aggregate ({'/'.join(_SCORE_AGGREGATES)}) [{args.score}]: ",
                    args.score, str, _SCORE_AGGREGATES)
    target = _choose(f"Target answered-error [{args.target_error}]: ",
                     args.target_error, _target_error)
    save = _choose_save(bool(getattr(args, "save", False)))
    # Validate defaults too: EOF must never let a malformed --task/--target-error reach model work.
    task, target = _task_label(task), _target_error(target)
    print(f"\nCalibration plan: task={task}  set={which}  score={score}  "
          f"target_error={target:g}  save={'yes' if save else 'no'}")
    return task, which, score, target, save


def _saved_paths(saved) -> str:
    if isinstance(saved, dict):
        if saved.get("model") and saved.get("task"):
            return f"{saved['model']} / {saved['task']}"
        return "model/task profile saved"
    return str(saved)


# CRITICAL HONESTY LABEL: the deployed selective-generation signal was proven bit-identical to
# exp(min(logprob)), exactly what any
# OpenAI-compatible black-box API already returns). Every surface this wizard/policy-show prints must say
# so plainly, and must print the one band limitation that autopsy also found: the signal degrades sharply
# on hard-tail / out-of-distribution inputs (dense multi-digit arithmetic in the measured case) -- never a
# universal guarantee.
_SIGNAL_NOTE = ("Signal: token-probability based (answer-span token confidence, min or mean over content "
               "tokens) -- this is NOT an internal/white-box signal; it is bit-identical to the logprobs "
               "any OpenAI-compatible API already returns.")
_HARD_TAIL_NOTE = ("Band limitation: known to fail on hard-tail / out-of-distribution inputs (e.g. dense "
                   "multi-digit arithmetic) -- a fitted band is directional outside the probed "
                   "distribution, never a guarantee.")


def _print_wizard_scope(out: dict, task: str, which: str, score: str,
                        target_error: float, recommendation: dict) -> None:
    summary = recommendation.get("summary") or {}
    model = out.get("model") or "unavailable"
    print(f"\nCalibration scope: {out.get('n', 0)} gradeable sample(s), "
          f"{out.get('unmatched', 0)} unmatched; model={model}  task={task}  probes={which}.")
    print(f"  Policy tradeoff @ target_error={target_error:g}: {score}-token aggregate; "
          f"answer={summary.get('n_answer', 0)}, ask={summary.get('n_ask', 0)}, "
          f"abstain={summary.get('n_abstain', 0)}, correct_withheld={summary.get('correct_withheld', 0)}.")
    print("  Distribution limit: this built-in probe mix may not represent the task's live inputs; "
          "treat a small sample as directional evidence.")
    print(f"  {_SIGNAL_NOTE}")
    print(f"  {_HARD_TAIL_NOTE}")
    print("  Caveat: confidence comes from recorded token probabilities; it is not a live fact-check "
          "and does not verify a new answer's claims.")


def _list_saved_profiles(*, json_output: bool = False) -> int:
    from clozn.eval import store as eval_store

    profiles = eval_store.list_profiles()
    if json_output:
        print(json.dumps({"profiles": profiles}, indent=2, default=str))
        return 0
    if not profiles:
        print("No model/task calibration profiles installed. Run `clozn eval --wizard` to fit one.")
        return 0
    print("Installed model/task calibration profiles (newest first):")
    for profile in profiles:
        policy_summary = profile.get("policy") or {}
        print(f"  {profile.get('model') or '?'}  task={profile.get('task') or '?'}  "
              f"probes={profile.get('set') or '?'}  n={profile.get('n', '?')}  "
              f"score={profile.get('score') or '?'}  answer_at={policy_summary.get('answer_at', '?')}  "
              f"ask_at={policy_summary.get('ask_at', '—')}")
    print("Thresholds are outcome-grounded for the named model/task distribution; they are not a live fact-check.")
    return 0


def _no_policy_command(_args) -> int:
    print("clozn eval policy: use `clozn eval policy show`")
    return 2


def _detect_live_model(url: str, timeout: float = 2.0) -> str | None:
    """Best-effort live model identity via GET {url}/readyz -- the same worker-health route `clozn doctor`
    and the readiness probe already use. Returns None on ANY failure (unreachable, timeout, bad JSON, a
    missing/blank model field) -- never raises, and `clozn eval policy show` never guesses a model identity
    when this comes back empty; it asks for --model or reports that it can't tell instead."""
    import json as _json
    import urllib.request as _rq
    try:
        with _rq.urlopen(f"{url.rstrip('/')}/readyz", timeout=timeout) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    model = payload.get("model") if isinstance(payload, dict) else None
    return model.strip() if isinstance(model, str) and model.strip() else None


def _fmt_ts(ts) -> str:
    try:
        import time as _time
        return _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(float(ts)))
    except (TypeError, ValueError, OSError, OverflowError):
        return "unknown"


def _cmd_policy_show(args) -> int:
    """`clozn eval policy show [--model] [--task] [--url] [--json]` -- print the policy a live reply would
    ACTUALLY get right now (clozn.server.generation_gateway.policy_signal's own resolution: exact model,
    then exact-or-newest task), without needing to run any probes or generate a reply. Read-only inspection
    over clozn.eval.store; never fabricates a model identity or a profile match."""
    from clozn.eval import store as eval_store

    model = (getattr(args, "model", None) or "").strip() or None
    task = getattr(args, "task", None)
    if task is not None:
        try:
            task = _task_label(task)
        except ValueError as exc:
            print(f"clozn eval policy show: {exc}")
            return 2

    as_json = bool(getattr(args, "json", False))
    live_note = None
    if not model:
        model = _detect_live_model(args.url)
        if model:
            live_note = f"model detected live at {args.url}"

    saved = None
    if model:
        saved = eval_store.load_profile(model, task)
    else:
        # No --model and no live gateway answered: only fall back when there is exactly ONE saved
        # profile total -- never guess among several, and never silently pick "the newest" across models.
        profiles = eval_store.list_profiles()
        if len(profiles) == 1:
            saved = profiles[0]
            model = saved.get("model")
            live_note = f"no live model detected at {args.url} -- showing the one saved profile"
        elif len(profiles) > 1:
            reason = (f"no live model detected at {args.url} and {len(profiles)} saved profiles exist -- "
                      "pass --model to pick one (see `clozn eval --list-profiles`)")
            if as_json:
                print(json.dumps({"available": False, "reason": reason}, indent=2))
            else:
                print(f"clozn eval policy show: {reason}")
            return 1

    if not isinstance(saved, dict) or not isinstance(saved.get("policy"), dict):
        if model:
            reason = f"no calibration profile saved for model={model!r}" + (f" task={task!r}" if task else "")
        else:
            reason = f"no live model detected at {args.url} and no saved profiles exist"
        if as_json:
            print(json.dumps({"available": False, "reason": reason}, indent=2))
        else:
            print(f"clozn eval policy show: {reason} -- run `clozn eval --wizard --save` first")
        return 1

    pol = saved.get("policy") or {}
    fit_summary = pol.get("summary") or {}
    out = {
        "available": True, "model": saved.get("model"), "task": saved.get("task") or saved.get("set"),
        "probe_set": saved.get("set"), "score_aggregate": saved.get("score"),
        "n": saved.get("n"), "unmatched": saved.get("unmatched"), "saved_ts": saved.get("saved_ts"),
        "answer_at": pol.get("answer_at"), "ask_at": pol.get("ask_at"), "achievable": pol.get("achievable"),
        "target_error": pol.get("target_error") or saved.get("target_error"),
        "fit_summary": fit_summary, "live_note": live_note,
    }
    if as_json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    print(f"Active policy: model={out['model']}  task={out['task']}")
    if live_note:
        print(f"  ({live_note})")
    print(f"  source: probes={out['probe_set']}  score={out['score_aggregate']}  n={out['n']} "
          f"(unmatched={out['unmatched']})  saved={_fmt_ts(out['saved_ts'])}")
    print(f"  thresholds: answer_at={out['answer_at']}  ask_at={out['ask_at']}  achievable={out['achievable']}")
    print(f"  at fit time: answer {fit_summary.get('n_answer', '?')} / ask {fit_summary.get('n_ask', '?')} / "
          f"abstain {fit_summary.get('n_abstain', '?')}  coverage={fit_summary.get('coverage', '?')}  "
          f"answered_error={fit_summary.get('answered_error', '?')}")
    print(f"  {_SIGNAL_NOTE}")
    print(f"  {_HARD_TAIL_NOTE}")
    print("  Caveat: not a live fact-check; a live reply only carries this verdict when its model and "
          "score aggregate match this profile EXACTLY (clozn.eval.policy.classify_run).")
    return 0


def cmd_eval(args):
    """`clozn eval [--set ...] [--score ...] [--json]` -- LIVE: needs a running Clozn gateway (see module
    docstring). Delegates to clozn.eval.bench; prints the human report, or the raw report JSON with --json."""
    if getattr(args, "list_profiles", False):
        return _list_saved_profiles(json_output=bool(getattr(args, "json", False)))
    try:
        if getattr(args, "wizard", False):
            task, which, score, te, save = _wizard(args)
        else:
            task, which, score = _task_label(getattr(args, "task", "general")), args.which, args.score
            te, save = _target_error(getattr(args, "target_error", 0.05)), bool(getattr(args, "save", False))
    except ValueError as exc:
        print(f"clozn eval: {exc}")
        return 2

    from clozn.eval import bench
    from clozn.eval import policy

    out = bench.bench(args.url, which, score)
    rec = policy.recommend(out.get("pairs", []), target_error=te)
    if getattr(args, "json", False):
        print(json.dumps({"model": out.get("model"), "task": task, "set": which, "score": score,
                          "target_error": te, "n": out["n"], "unmatched": out["unmatched"],
                          "report": out["report"], "policy": rec, "rows": out["rows"]},
                         indent=2, default=str))
    else:
        bench._print(out, which, score, te)
    if getattr(args, "wizard", False):
        _print_wizard_scope(out, task, which, score, te, rec)
    if save:
        from clozn.eval import store as eval_store
        if not isinstance(out.get("model"), str) or not out["model"].strip():
            print("\n  calibration was not saved: no active model identity was captured by the probe run")
            return 1
        payload = {"set": which, "score": score, "target_error": te, "model": out.get("model"),
                   "task": task,
                   "n": out["n"], "unmatched": out["unmatched"], "report": out["report"],
                   "policy": rec, "rows": out["rows"]}
        try:
            saved = eval_store.save_profile(payload, task=task)
        except (OSError, ValueError) as exc:
            print(f"\n  calibration was not saved: {exc}")
            return 1
        print(f"\n  saved TRUTH-tier profile -> {_saved_paths(saved)}"
              "  (served at GET /journal/calibration)")
    return 0
