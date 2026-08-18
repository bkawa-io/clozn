"""testkit/runner.py -- the tiny-test harness: user-authored, run-level assertions over the run record's already-
legible seams (trace / memory / meta / response / causal receipts). Like unit tests, but for model runs.

Spec format is plain JSON (stdlib `json` -- no yaml, no pytest, no pip installs; clozn is a no-install
product and this harness rides its own runner):

    { "tests": [
        { "name": "capital is paris",
          "run": "latest",                        // a run_id, or "latest" (the newest recorded run)
          "assert": [
            {"check": "contains", "value": "Paris"},
            {"check": "finish_reason", "value": "stop"},
            {"check": "min_confidence", "value": 0.5},
            {"check": "leans_on", "card": "france-facts", "min_effect": 0.0}
          ] } ] }

Two assertion classes -- kept clearly separated below, and in every result's `check` label:

  STATIC (model-free; evaluated against the stored run record alone -- see STATIC_DISPATCH):
    contains / not_contains / matches (regex) / equals            -- on the response text
    finish_reason                                                  -- run["finish_reason"]
    <any REPRO_META_KEYS name> (seed, quant, n_ctx, device, sampler_mode, ...) -- run["meta"][name]
    min_confidence / max_confidence (overall, or `"at": <token index>`)        -- trace["confidence"]
    max_entropy (overall, or `"at": <token index>`)                            -- trace["topk_entropy"]
    alternative_present {"value": piece, "at": optional index}     -- trace["alternatives"]

  CAUSAL (opt-in; needs a live substrate -- runs receipts.py's leave-one-out ablation seam):
    leans_on {"card": id-or-text}, "min_effect": float (default 0.0)
        -> receipts.receipt(run, {"card_id": ...}, sub); passes only when the ablation actually verified
           (`causal_verified` truthy) AND showed an effect (`has_effect` True) at least `min_effect` (a
           0..1 fraction of receipt_metrics()'s word-type "changed" delta).

HONESTY RULE (non-negotiable): a causal assertion that cannot be verified -- no substrate/fetcher supplied,
the receipt seam itself returned nothing, or `causal_verified` came back False (internalized memory mode,
or nothing to ablate) -- is `status: "skipped"` with the reason, NEVER a silent pass or a false fail. Running
this module's `evaluate`/`run_suite` with no `sub` and no `fetch_receipt` (the default) skips every causal
assertion; only an explicit live hookup (the CLI's `clozn test --live`) can pass or fail one.

Result shape (one dict per ASSERTION, not per test -- this is exactly the shape `run["tiny_tests"]` wants,
so `--attach` results ride straight into receipt_bundle.build(run)["tiny_tests"] with no reshaping):
    {"name": <test name>, "check": <check id>, "target": <what was inspected>,
     "expected": <<the asserted value>>, "actual": <<the observed value>>,
     "status": "pass" | "fail" | "skip" | "error", "note": <str|None>}

A test's overall status is the worst of its own assertions' statuses (rank error > fail > skip > pass).
This module is pure: it never prints (the CLI owns rendering) and never raises out of `evaluate`/`run_suite`
(a malformed assertion or test degrades to one "error"-status result, never crashes the run).
"""
from __future__ import annotations

import re

from clozn import receipts as _receipts
import clozn.runs.store as _runlog

STATUSES = ("pass", "fail", "skip", "error")
_STATUS_RANK = {"pass": 0, "skip": 1, "fail": 2, "error": 3}

STATIC_CHECKS = ("contains", "not_contains", "matches", "equals", "finish_reason",
                 "min_confidence", "max_confidence", "max_entropy", "alternative_present")
CAUSAL_CHECKS = ("leans_on",)

# The exact repro-meta field names a spec may check by name (mirrors receipt_bundle.REPRO_META_KEYS so the
# two can never silently drift -- a run's `meta` dict is the single source of truth for both).
REPRO_META_KEYS = (
    "model_id", "model_file", "quant", "mode", "sampler_mode", "sampling", "temperature", "top_p",
    "repetition_penalty", "no_repeat_ngram_size", "max_tokens", "seed", "n_ctx", "device", "gpu_layers",
    "build_git_commit", "finish_reason_source", "finish_reason_fallback", "capture_tier",
)


# ------------------------------------------------------------------------------------------------ resolving
def default_get_run(ref: str | None) -> dict | None:
    """The default `get_run` for run_suite(): a literal run id -> runlog.get_run(id); "latest"/None -> the
    newest recorded run (runlog.list_runs(limit=1)). Returns None when nothing resolves -- never raises."""
    try:
        if not ref or ref == "latest":
            rows = _runlog.list_runs(limit=1)
            if not rows:
                return None
            ref = rows[0].get("id")
            if not ref:
                return None
        return _runlog.get_run(ref)
    except Exception:
        return None


# --------------------------------------------------------------------------------------------------- utils
def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _result(name, check, target, expected, actual, status, note=None) -> dict:
    return {"name": name, "check": check, "target": target, "expected": expected,
            "actual": actual, "status": status, "note": note}


def _worst(statuses) -> str:
    worst, rank = "error", -1
    seen = False
    for s in statuses:
        seen = True
        r = _STATUS_RANK.get(s, _STATUS_RANK["error"])
        if r > rank:
            rank, worst = r, s
    return worst if seen else "error"


# ============================================================================================ static checks
# Each takes (run, assertion_dict, test_name) -> one assertion result. Model-free: reads only the stored
# run record. "error" status means the run's data couldn't even be evaluated against (a missing/malformed
# field, a bad spec, an out-of-range index) -- distinct from "fail" (the data was there; the bar wasn't met).

def _eval_contains(run, a, name):
    value = a.get("value")
    if value is None:
        return _result(name, "contains", "response", value, None, "error", "contains needs a 'value'")
    response = str(run.get("response") or "")
    ok = str(value) in response
    return _result(name, "contains", "response", value, response, "pass" if ok else "fail")


def _eval_not_contains(run, a, name):
    value = a.get("value")
    if value is None:
        return _result(name, "not_contains", "response", value, None, "error",
                       "not_contains needs a 'value'")
    response = str(run.get("response") or "")
    ok = str(value) not in response
    return _result(name, "not_contains", "response", value, response, "pass" if ok else "fail")


def _eval_matches(run, a, name):
    pattern = a.get("value")
    if pattern is None:
        return _result(name, "matches", "response", pattern, None, "error", "matches needs a 'value' (regex)")
    response = str(run.get("response") or "")
    try:
        ok = re.search(pattern, response) is not None
    except re.error as e:
        return _result(name, "matches", "response", pattern, response, "error", f"invalid regex: {e}")
    return _result(name, "matches", "response", pattern, response, "pass" if ok else "fail")


def _eval_equals(run, a, name):
    value = a.get("value")
    response = str(run.get("response") or "")
    ok = response == value
    return _result(name, "equals", "response", value, response, "pass" if ok else "fail")


def _eval_finish_reason(run, a, name):
    value = a.get("value")
    actual = run.get("finish_reason")
    ok = actual == value
    return _result(name, "finish_reason", "finish_reason", value, actual, "pass" if ok else "fail")


def _eval_meta(run, a, name):
    field = a.get("check")          # the check name itself IS the meta key (any REPRO_META_KEYS name)
    value = a.get("value")
    meta = _as_dict(run.get("meta"))
    actual = meta.get(field)
    ok = actual == value
    return _result(name, field, f"meta.{field}", value, actual, "pass" if ok else "fail")


def _confidence_list(run):
    return _as_list(_as_dict(run.get("trace")).get("confidence"))


def _eval_confidence_bound(run, a, name, *, check, cmp, agg):
    """Shared body for min_confidence / max_confidence: `cmp(actual, value)` decides pass; `agg` reduces the
    whole trace when no `at` index is given."""
    value = _as_float(a.get("value"))
    if value is None:
        return _result(name, check, "confidence", a.get("value"), None, "error",
                       f"{check} needs a numeric 'value'")
    conf = _confidence_list(run)
    at = a.get("at")
    if at is not None:
        idx = _as_int(at)
        if idx is None or idx < 0 or idx >= len(conf):
            return _result(name, check, f"confidence[{at}]", value, None, "error",
                           f"token index {at!r} out of range (trace has {len(conf)} tokens)")
        actual = _as_float(conf[idx])
        target = f"confidence[{idx}]"
    else:
        if not conf:
            return _result(name, check, "confidence", value, None, "error",
                           "no confidence trace recorded on this run")
        actual = agg(_as_float(c) or 0.0 for c in conf)
        target = f"{agg.__name__}(confidence)"
    ok = actual is not None and cmp(actual, value)
    return _result(name, check, target, value, actual, "pass" if ok else "fail")


def _eval_min_confidence(run, a, name):
    return _eval_confidence_bound(run, a, name, check="min_confidence",
                                  cmp=lambda actual, value: actual >= value, agg=min)


def _eval_max_confidence(run, a, name):
    return _eval_confidence_bound(run, a, name, check="max_confidence",
                                  cmp=lambda actual, value: actual <= value, agg=max)


def _eval_max_entropy(run, a, name):
    value = _as_float(a.get("value"))
    if value is None:
        return _result(name, "max_entropy", "topk_entropy", a.get("value"), None, "error",
                       "max_entropy needs a numeric 'value'")
    ent = _as_list(_as_dict(run.get("trace")).get("topk_entropy"))
    at = a.get("at")
    if at is not None:
        idx = _as_int(at)
        if idx is None or idx < 0 or idx >= len(ent):
            return _result(name, "max_entropy", f"topk_entropy[{at}]", value, None, "error",
                           f"token index {at!r} out of range (trace has {len(ent)} entries)")
        raw = ent[idx]
        if raw is None:
            return _result(name, "max_entropy", f"topk_entropy[{idx}]", value, None, "error",
                           "no topk_entropy recorded at that index (no top-k captured there)")
        actual, target = float(raw), f"topk_entropy[{idx}]"
    else:
        vals = [float(x) for x in ent if x is not None]
        if not vals:
            return _result(name, "max_entropy", "topk_entropy", value, None, "error",
                           "no topk_entropy data recorded on this run")
        actual, target = max(vals), "max(topk_entropy)"
    ok = actual <= value
    return _result(name, "max_entropy", target, value, actual, "pass" if ok else "fail")


def _alt_pieces(alts) -> list[str]:
    out = []
    for a in alts or []:
        if isinstance(a, dict):
            out.append(str(a.get("piece", a.get("text", ""))))
    return out


def _eval_alternative_present(run, a, name):
    value = a.get("value")
    if value is None:
        return _result(name, "alternative_present", "alternatives", value, None, "error",
                       "alternative_present needs a 'value'")
    alts_arr = _as_list(_as_dict(run.get("trace")).get("alternatives"))
    if not alts_arr:
        return _result(name, "alternative_present", "alternatives", value, None, "error",
                       "no alternatives recorded on this run")
    at = a.get("at")
    if at is not None:
        idx = _as_int(at)
        if idx is None or idx < 0 or idx >= len(alts_arr):
            return _result(name, "alternative_present", f"alternatives[{at}]", value, None, "error",
                           f"token index {at!r} out of range (trace has {len(alts_arr)} positions)")
        pieces = _alt_pieces(alts_arr[idx])
        target = f"alternatives[{idx}]"
    else:
        pieces = [p for step_alts in alts_arr for p in _alt_pieces(step_alts)]
        target = "alternatives"
    ok = str(value) in pieces
    return _result(name, "alternative_present", target, value, pieces, "pass" if ok else "fail")


STATIC_DISPATCH = {
    "contains": _eval_contains,
    "not_contains": _eval_not_contains,
    "matches": _eval_matches,
    "equals": _eval_equals,
    "finish_reason": _eval_finish_reason,
    "min_confidence": _eval_min_confidence,
    "max_confidence": _eval_max_confidence,
    "max_entropy": _eval_max_entropy,
    "alternative_present": _eval_alternative_present,
}


# ============================================================================================ causal checks
# Opt-in: runs receipts.py's leave-one-out ablation seam. HONESTY RULE lives here -- see module docstring.
# `fetch_receipt(run, influence) -> dict | None` is the injectable seam: defaults to calling
# `receipts.receipt(run, influence, sub)` against the in-process, duck-typed `sub` (exactly like
# tests/test_receipts.py's FakeSub); the CLI's `--live` swaps in an HTTP-backed fetcher instead (talking to
# a running `clozn studio`'s already-shipped POST /runs/<id>/receipt) without this module ever knowing the
# difference.

def judge_receipt(rec: dict | None, min_effect: float) -> tuple[str, object, str | None]:
    """Pure judgement of an already-computed receipt dict -> (status, actual, note). Shared by the in-
    process `sub` path and the CLI's live-HTTP path, so both apply IDENTICAL pass/fail/skip logic to a
    receipt regardless of how it was obtained."""
    if isinstance(rec, dict) and rec.get("_fetch_error") == "engine_unreachable":
        # the live-HTTP path's distinguishing signal (see cli.commands.test._fetch_live_receipt): the
        # gateway itself answered 502/503 "engine not reachable", not an ambiguous None a bad influence
        # spec would ALSO produce -- say so, instead of the generic "bad spec or couldn't generate" note.
        return "skip", None, f"causal assertion skipped: {rec.get('_fetch_detail') or 'engine not reachable'}"
    if rec is None:
        return "skip", None, ("causal assertion skipped: the receipt could not be computed "
                              "(bad influence spec, or the substrate could not generate)")
    if not rec.get("causal_verified"):
        note = rec.get("ablation_note") or rec.get("note") or "the ablation could not be verified on this run"
        return "skip", {"has_effect": rec.get("has_effect")}, note
    delta = rec.get("delta") or {}
    changed = delta.get("changed")
    effect = (float(changed) / 100.0) if isinstance(changed, (int, float)) else 0.0
    has_effect = bool(rec.get("has_effect"))
    actual = {"has_effect": has_effect, "effect": round(effect, 4)}
    ok = has_effect and effect >= min_effect
    return ("pass" if ok else "fail"), actual, rec.get("note")


def _eval_causal(run, a, name, sub, fetch_receipt):
    check = a.get("check")
    card = a.get("card")
    min_effect = _as_float(a.get("min_effect"))
    if min_effect is None:
        min_effect = 0.0
    if card:
        influence, target = {"card_id": card}, f"card:{card}"
    else:
        return _result(name, check, "leans_on", a.get("min_effect"), None, "error",
                       "leans_on needs a 'card'")

    if fetch_receipt is None and sub is None:
        return _result(name, check, target, min_effect, None, "skip",
                       "causal assertion skipped: no substrate supplied (needs --live)")
    resolver = fetch_receipt if fetch_receipt is not None else (lambda r, inf: _receipts.receipt(r, inf, sub))
    try:
        rec = resolver(run, influence)
    except Exception as e:
        return _result(name, check, target, min_effect, None, "skip",
                       f"causal assertion skipped: receipt fetch failed ({type(e).__name__}: {e})")
    status, actual, note = judge_receipt(rec, min_effect)
    return _result(name, check, target, min_effect, actual, status, note)


# ================================================================================================= evaluate
def evaluate(run: dict | None, spec_test: dict, sub=None, *, fetch_receipt=None) -> dict:
    """Evaluate one spec test's assertions against an already-resolved `run` dict. Pure, never raises: a
    missing run or a malformed assertion degrades to an "error"-status result for just that piece.

    `sub`           -- an in-process, receipts.py-duck-typed substrate for causal checks (None = skip them).
    `fetch_receipt` -- an injectable `(run, influence) -> receipt|None` used instead of calling
                       receipts.receipt(..., sub) directly (the CLI's --live HTTP path plugs in here).
    """
    name = spec_test.get("name") if isinstance(spec_test, dict) else None
    name = name or "(unnamed test)"
    if not isinstance(run, dict) or not run:
        note = f"run not found: {spec_test.get('run')!r}" if isinstance(spec_test, dict) else "run not found"
        return {"name": name, "run_id": None, "status": "error",
                "assertions": [_result(name, None, "run", spec_test.get("run") if isinstance(spec_test, dict)
                                       else None, None, "error", note)]}

    assertions_spec = spec_test.get("assert") if isinstance(spec_test, dict) else None
    if not isinstance(assertions_spec, list) or not assertions_spec:
        return {"name": name, "run_id": run.get("id"), "status": "error",
                "assertions": [_result(name, None, "assert", None, None, "error",
                                       "test has no non-empty 'assert' list")]}

    results = []
    for a in assertions_spec:
        if not isinstance(a, dict):
            results.append(_result(name, None, "assert", None, a, "error", "assertion is not an object"))
            continue
        check = a.get("check")
        if check in CAUSAL_CHECKS:
            results.append(_eval_causal(run, a, name, sub, fetch_receipt))
        elif check in STATIC_DISPATCH:
            results.append(STATIC_DISPATCH[check](run, a, name))
        elif check in REPRO_META_KEYS:
            results.append(_eval_meta(run, a, name))
        else:
            results.append(_result(name, check, check, a.get("value"), None, "error",
                                   f"unknown check: {check!r}"))

    return {"name": name, "run_id": run.get("id"), "status": _worst(r["status"] for r in results),
            "assertions": results}


def run_suite(spec: dict, *, get_run=default_get_run, sub=None, fetch_receipt=None) -> dict:
    """Evaluate every test in `spec` (a JSON dict with a "tests" list). Pure: never prints, never raises --
    a malformed test entry becomes an "error"-status test, not a crash. Returns:
        {"tests": [test_result, ...], "status": <worst over all tests>,
         "tiny_tests": <flat list of every assertion result -- exactly run["tiny_tests"]'s shape>}
    """
    tests_spec = spec.get("tests") if isinstance(spec, dict) else None
    tests_spec = tests_spec if isinstance(tests_spec, list) else []

    results = []
    for t in tests_spec:
        if not isinstance(t, dict):
            bad_name = "(malformed test)"
            results.append({"name": bad_name, "run_id": None, "status": "error",
                            "assertions": [_result(bad_name, None, "test", None, t, "error",
                                                   "test entry is not an object")]})
            continue
        run_ref = t.get("run") or "latest"
        try:
            run = get_run(run_ref)
        except Exception:
            run = None
        results.append(evaluate(run, t, sub, fetch_receipt=fetch_receipt))

    status = _worst(r["status"] for r in results) if results else "error"
    tiny_tests = [a for r in results for a in r["assertions"]]
    counts = {s: 0 for s in STATUSES}
    for a in tiny_tests:
        counts[a["status"]] = counts.get(a["status"], 0) + 1
    return {"tests": results, "status": status, "tiny_tests": tiny_tests, "counts": counts}


def results_by_run(suite: dict) -> dict:
    """Group a run_suite() result's assertions by the run_id they were evaluated against -- what `clozn
    test --attach` needs, since one spec file can exercise more than one run. A test whose run never
    resolved (run_id None) is excluded -- nothing to attach it to."""
    out: dict = {}
    for t in suite.get("tests") or []:
        rid = t.get("run_id")
        if not rid:
            continue
        out.setdefault(rid, []).extend(t.get("assertions") or [])
    return out
