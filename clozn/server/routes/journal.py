"""The actuarial journal over HTTP: the full ActuaryReport (GET /journal/actuary), a past-only failure
assessment for one run (POST /runs/<id>/actuary), the OUTCOME-grounded calibration report (GET
/journal/calibration), and calibrated trust spans (POST /runs/<id>/trust_spans).

The default trust-spans read is model-free. Passing ``{"support": true}`` explicitly opts into the
independent local NLI cross-encoder: it checks each span against stored causal-receipt premises when present,
or a plainly labeled active-influence manifest otherwise, and may load that optional model. It never
generates with the audited model, and an unavailable NLI checkpoint stays labeled unavailable rather than
falling back to lexical overlap under the same name.

Two calibration tiers sit side by side, each labelled: /journal/actuary is the PROXY curve (acceptance,
always available, computed on-demand); /journal/calibration is the TRUTH curve (correctness on a labeled
probe set + a selective-generation policy) -- it needs live generation, so it is PERSISTED by
`clozn eval --save` (clozn.eval.store) and served from disk, or available:false until one is saved.

Everything served here inherits actuary.py's honesty stance verbatim: "trusted" is a behavioral PROXY
(the run was kept -- not errored/truncated/test-failed/re-rolled), never a verified-correctness figure.
The dataclass `note` fields ride the wire unchanged, and trust_spans carries calibrated_trust.NOTE at
the top level, so no consumer can read these numbers without the proxy language attached. A journal
with no scored organic runs answers available:false -- absence over an invented curve.

The route module is registered in both product router tables in ``clozn.server.app``. The per-run
failure assessment refits on strictly earlier organic records when timestamps are available, and never
trains on the run id it scores.

The report is cached in-process for 60s (recomputing over a few hundred journal files is cheap but not
free on every poll); every response states its age as "computed_ago_s" so a consumer knows how stale
the curve it was served is.
"""
import threading
import time

_CACHE_TTL_S = 60.0
_LOCK = threading.Lock()
_REPORT = None          # the cached actuary.ActuaryReport
_COMPUTED_AT = 0.0      # time.time() when _REPORT was computed


def _report():
    """The (possibly cached) ActuaryReport + its age in seconds. Recomputes when older than
    _CACHE_TTL_S. Serialized under _LOCK: the server is threaded, and two concurrent recomputes over
    the same journal files would be wasted work (the read itself is safe, just not free)."""
    global _REPORT, _COMPUTED_AT
    with _LOCK:
        now = time.time()
        if _REPORT is None or (now - _COMPUTED_AT) >= _CACHE_TTL_S:
            from clozn.runs import actuary
            _REPORT = actuary.load_and_analyze()
            _COMPUTED_AT = now
        return _REPORT, max(0.0, time.time() - _COMPUTED_AT)


def try_get(h, p):
    if p == "/journal/actuary":   # the full actuarial report -- calibration/drift/failure model, all proxy-labelled
        import dataclasses
        report, age = _report()
        out = dataclasses.asdict(report)      # nested dataclasses -> plain dicts; every `note` rides verbatim
        out["computed_ago_s"] = round(age, 1)
        h._json(200, out)
        return True
    if p == "/journal/calibration":   # the TRUTH tier: correctness on a labeled probe set + selective policy
        import dataclasses
        import time as _time
        from clozn.eval import store as eval_store
        rep = eval_store.load_latest_profile()
        if not rep:
            h._json(200, {"available": False,
                          "note": "no outcome-grounded calibration saved yet -- run `clozn eval --save` "
                                  "(needs a live studio) to populate this TRUTH-tier curve. It measures "
                                  "correctness on a labeled probe set, not the acceptance proxy."})
            return True
        out = dict(rep)
        out["available"] = True
        out["saved_ago_s"] = round(max(0.0, _time.time() - float(out.get("saved_ts", _time.time()))), 1)
        rows = out.get("rows") or []
        if rows:
            from clozn.eval import calibration as cal
            pairs = [(r["score"], r["correct"]) for r in rows
                     if isinstance(r, dict) and r.get("score") is not None and r.get("correct") is not None]
            ece_result = cal.ece(pairs)
            out["reliability_bins"] = [dataclasses.asdict(b) for b in ece_result["bins"]]
            out["risk_coverage"] = [dataclasses.asdict(pt) for pt in cal.risk_coverage(pairs)]
        h._json(200, out)
        return True
    return False


def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/actuary"):
        rid = p[len("/runs/"):-len("/actuary")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.runs import actuary
        out = actuary.assess_failure(run, actuary.load_runs())
        out["run_id"] = rid
        h._json(200, out)
        return True
    if p.startswith("/runs/") and p.endswith("/trust_spans"):   # confidence + proxy + truth; support is opt-in
        rid = p[len("/runs/"):-len("/trust_spans")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.runs import calibrated_trust, confidence_spans
        report, age = _report()
        cal = report.calibration
        sp = confidence_spans.spans(run)                        # REUSE the existing segmentation, unchanged
        proxy_available = calibrated_trust.has_curve(cal)
        mapped = calibrated_trust.attach(sp, cal) if proxy_available else [dict(x) for x in sp]
        proxy = ({"available": True, "n_scored": cal.n_scored, "computed_ago_s": round(age, 1),
                  "note": calibrated_trust.NOTE} if proxy_available else
                 {"available": False,
                  "reason": "the journal has no scored organic runs — no acceptance curve was invented",
                  "note": calibrated_trust.NOTE})

        from clozn.eval import store as eval_store
        mapped, truth = calibrated_trust.attach_truth(
            mapped, eval_store.load_profile(run.get("model")), run.get("model")
        )

        support = {"requested": False, "available": False,
                   "reason": "not computed — pass support:true to run the optional independent NLI check",
                   "note": ("support is separate from confidence and correctness calibration; it asks whether "
                            "an active recorded influence entails a span")}
        if body.get("support") is True:
            from clozn.receipts import explain, semantic_matcher
            from clozn.runs import span_support
            manifest = explain.explain(run)                     # pure journal reshape; no generation
            stored = run.get("receipts")
            if isinstance(stored, list):
                stored = {"receipts": stored}
            if not isinstance(stored, dict) and isinstance(run.get("receipt"), dict):
                stored = {"receipts": [run["receipt"]]}
            if isinstance(stored, dict):
                # A stored prove-all result lets support use only receipt-verified, load-bearing premises.
                # causal_explanation receives the precomputed object, so it cannot generate here.
                from clozn.receipts import self_report_reliability
                explanation = self_report_reliability.causal_explanation(
                    run, None, manifest=manifest, prove=stored)
                evidence_tier = "causal_receipts"
                evidence_note = ("NLI premises are restricted to influences whose stored leave-one-out "
                                 "receipt has_effect=true on this run")
            else:
                explanation = manifest
                evidence_tier = "active_manifest"
                evidence_note = ("no stored causal receipt was available; NLI premises are active recorded "
                                 "influences, which proves presence, not causal effect")
            mapped, support = span_support.attach(mapped, explanation,
                                                  semantic_matcher.nli_support_matcher)
            support["evidence_tier"] = evidence_tier
            support["evidence_note"] = evidence_note

        any_calibration = proxy_available or truth.get("available") is True
        out = {"available": any_calibration, "run_id": rid, "spans": mapped,
               "summary": confidence_spans.summarize(sp), "proxy": proxy, "truth": truth,
               "support": support, "n_scored": (cal.n_scored if proxy_available else 0),
               "computed_ago_s": round(age, 1), "note": calibrated_trust.NOTE}
        if not any_calibration:
            out["reason"] = ("neither an acceptance-proxy curve nor a matching outcome-grounded "
                             "temperature fit is available; raw spans are returned without a trust estimate")
        h._json(200, out)
        return True
    return False
