"""testkit/ci.py -- the model CI runner: a nightly/on-demand regression suite over a whole battery of
tiny-test cases, run against a live `clozn` server, with cross-run diffing so a regression is a diff, not
a vibe.

Where this sits relative to `runner.py`: `runner.py` is the ASSERTION VOCABULARY -- one test, one already-
resolved run record, a list of checks. This module is the ORCHESTRATION layer on top of it: a suite is a
batch of (prompt -> expected checks) cases, each of which (1) actually calls the server to get a fresh run,
(2) evaluates it with `runner.evaluate()` (imported, not reimplemented -- the static-check semantics must
never drift between `clozn test <file.json>` and the CI runner), and (3) optionally proves causal claims
via the receipts endpoints. `diff_suites()` then compares two `SuiteResult`s (e.g. last night's vs.
tonight's) and reports what changed.

HOUSE RULE (non-negotiable, mirrors runner.py's HONESTY RULE): this runner never asks the model to grade
itself. Every check here is a MEASUREMENT over an already-recorded run -- string containment, a recorded
finish_reason, a recorded confidence trace, a server-computed causal receipt (leave-one-out ablation, or
forced teacher-forced scoring). There is no "ask the model if its answer was good" check anywhere in this
module, and there never should be.

REPRODUCIBILITY CAVEAT (S5 sampling): a CI suite is only a meaningful diff if the two runs being compared
were decoded the same way. If the server's active decode mode is sampling (`meta.sampler_mode == "sample"`,
see HEAVN_API_CONTRACTS.md §2/§18), token picks -- and therefore `contains`/`min_confidence`/receipts --
will vary run-to-run for reasons that have nothing to do with a regression; a `diff_suites()` report against
sampled runs will be contaminated by sampling noise, not signal. Pin the server to greedy decode (or a fixed
seed, where the substrate supports one -- see `meta.decode.seed` in the contracts doc) before trusting a CI
diff. This module does not enforce that (it has no way to configure the live server), but callers should.

Suite manifest shape (plain JSON, stdlib `json` only):

    {
      "model_note": "Qwen2.5-7B-Instruct via engine substrate, greedy decode pinned for CI",
      "cases": [
        {
          "name": "capital-of-france",
          "prompt": "What is the capital of France?",
          "expect": {
            "contains": ["Paris"],
            "not_contains": ["Berlin"],
            "finish_reason": "stop",
            "min_confidence": 0.5,
            "max_tokens": 256
          },
          "prove": false
        }
      ]
    }

`expect` keys map onto runner.py's tiny-test vocabulary (STATIC_DISPATCH / REPRO_META_KEYS) --
`contains`/`not_contains` take a list of strings (or a single string, auto-wrapped) and expand to one
assertion per item; `finish_reason`/`min_confidence` map
1:1; `max_tokens` is an EQUALITY check against the run's recorded `meta.max_tokens` (the requested cap that
was actually in effect for this run -- a reproducibility check, exactly like runner.py's other
REPRO_META_KEYS checks -- NOT an upper bound on how many tokens were generated). An unrecognized `expect`
key is not silently dropped: it is passed through to `runner.evaluate()` verbatim, which turns any check
name it doesn't recognize into a clean "error"-status assertion (never a crash, never a silent pass) --
this module adds no separate validation layer on top of that, by design, so there is exactly one place
("unknown check") that decides what's a valid assertion.

`prove: true` additionally calls the receipts prove-all endpoint (`POST /runs/<id>/receipts`, contracts
§7) after the static checks, and records `{influence, has_effect, causal_verified}` for every influence
that actually fired (leave-one-out over the run's fired prompt sections -- the only ablatable influence
kind left in the product since memory cards and tone dials were cut). This is a MEASUREMENT, not a gate:
a prove-only case does not fail just because some influence turned out not to be load-bearing (that's
exactly the kind of finding `diff_suites()` exists to surface across runs, not something a single suite run
should flunk on its own). The only way `prove: true` fails a case is if the measurement itself could not be
taken (the receipts call raised, or came back empty) -- silently skipping a causal claim is exactly what
the honesty rule forbids.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field

from clozn.testkit import runner

__all__ = [
    "CIError", "ClientError", "ClientHTTPError", "Client",
    "CaseResult", "SuiteResult", "Diff",
    "run_case", "run_suite", "diff_suites", "save_result", "load_result",
    "DEFAULT_CI_DIR",
]

STATUSES = ("pass", "fail", "skip", "error")
_RANK = {"pass": 0, "skip": 1, "fail": 2, "error": 3}

DEFAULT_CI_DIR = os.path.join(os.path.expanduser("~"), ".clozn", "ci")


# ================================================================================================== errors
class CIError(Exception):
    """Base for every error this module raises itself (as opposed to letting an underlying urllib error
    propagate uncaught -- ClientError below always wraps those)."""


class ClientError(CIError):
    """A `Client` request failed at the transport level (connection refused, timeout, DNS, ...) -- as
    opposed to `ClientHTTPError`, which means the server answered but with a non-2xx status."""


class ClientHTTPError(ClientError):
    """The server answered with a non-2xx HTTP status. `status` is the numeric code; `body` is the
    parsed-JSON error body if the response was valid JSON, else `{"error": <raw text or str(exception)>}`."""

    def __init__(self, status: int, body: dict):
        self.status = status
        self.body = body if isinstance(body, dict) else {"error": str(body)}
        super().__init__(f"HTTP {status}: {self.body.get('error', self.body)}")


# ================================================================================================== client
class Client:
    """A minimal, 100% mockable HTTP seam over a running `clozn` server. Every public method funnels
    through `_request(method, path, body)` -- that single method is the one tests monkeypatch (bind a
    replacement function/bound-method to `client._request`) to run this whole module with no network and
    no live server, exactly the way `runner.py`'s own causal checks accept an injectable `fetch_receipt`.

    Uses only `urllib`/`json` from the standard library -- no `requests`, no third-party HTTP client.
    """

    def __init__(self, base_url: str, *, timeout: float = 30.0,
                 headers: dict[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = {str(key): str(value) for key, value in (headers or {}).items()}

    # ---- the one seam ------------------------------------------------------------------------------
    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        """POST/GET `path` against `base_url`; returns the parsed JSON body. Raises `ClientHTTPError` on a
        non-2xx response (with `.status`/`.body` populated from the server's own error JSON where
        possible) and `ClientError` on a transport-level failure (server unreachable, timeout, ...)."""
        url = self.base_url + path
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = dict(self.headers)
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8") if e.fp else ""
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"error": raw or str(e)}
            raise ClientHTTPError(e.code, parsed) from e
        except urllib.error.URLError as e:
            raise ClientError(f"request failed: {method} {path}: {e}") from e

    # ---- public API ---------------------------------------------------------------------------------
    def chat(self, prompt: str, *, max_tokens: int = 256, model: str = "clozn-qwen",
             extra_messages: list[dict] | None = None, temperature: float | None = None,
             top_p: float | None = None, seed: int | None = None) -> dict:
        """POST /v1/chat/completions (non-stream), then resolve and return the FULL run record (contracts
        §2 shape) that call produced -- not the OpenAI-shaped completion envelope itself.

        Association strategy (contracts §18): the non-stream response carries the run id directly as a
        top-level `clozn_run_id` field (also mirrored on the `X-Clozn-Run-Id` header, which a plain
        `urllib` response object exposes too, but the body field is sufficient and doesn't require reading
        headers). If that field is present, it is used directly -- no polling needed. If it is ABSENT
        (the field is documented as cleanly omitted, never a literal "null", whenever the server's
        internal `_log_run` failed) this falls back to `GET /runs` and takes the newest entry, exactly
        the strategy a streaming caller is forced to use unconditionally (§18: the run id is NEVER
        communicated in-stream, by design -- no custom SSE event, no trailing frame, nothing). This
        fallback is inherently racy against concurrent traffic on the same server; it is a best-effort
        recovery path, not the primary mechanism.
        """
        messages = list(extra_messages or []) + [{"role": "user", "content": prompt}]
        body = {"messages": messages, "max_tokens": max_tokens, "model": model, "stream": False}
        for key, value in (("temperature", temperature), ("top_p", top_p), ("seed", seed)):
            if value is not None:
                body[key] = value
        resp = self._request("POST", "/v1/chat/completions", body)
        run_id = resp.get("clozn_run_id") if isinstance(resp, dict) else None
        if not run_id:
            rows = self._request("GET", "/runs")
            runs_list = rows.get("runs") if isinstance(rows, dict) else None
            run_id = runs_list[0].get("id") if runs_list else None
        if not run_id:
            raise CIError(f"chat(): could not associate this call with any run (prompt={prompt!r})")
        run = self.get_run(run_id)
        if run is None:
            raise CIError(f"chat(): run {run_id!r} was not found after the chat call completed")
        return run

    def get_run(self, run_id: str) -> dict | None:
        """GET /runs/<id> (contracts §2) -> the bare run record, or None on a 404."""
        try:
            return self._request("GET", f"/runs/{run_id}")
        except ClientHTTPError as e:
            if e.status == 404:
                return None
            raise

    def receipts(self, run_id: str, mode: str = "regen") -> dict | None:
        """POST /runs/<id>/receipts (contracts §7, prove-all) -> the raw response dict, or None on a 404."""
        try:
            return self._request("POST", f"/runs/{run_id}/receipts", {"mode": mode})
        except ClientHTTPError as e:
            if e.status == 404:
                return None
            raise


# ============================================================================================== dataclasses
@dataclass
class CaseResult:
    name: str
    run_id: str | None
    status: str                      # "pass" | "fail" | "error" (never "skip" -- see module docstring)
    assertions: list = field(default_factory=list)     # runner.py assertion-result dicts
    min_confidence: float | None = None
    receipts: list | None = None      # None = not proved; [] = proved, nothing fired
    receipts_skipped: list = field(default_factory=list)
    prove_error: str | None = None
    error: str | None = None          # a case-level failure that never even reached assertion evaluation

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CaseResult":
        return cls(
            name=d.get("name", "(unnamed case)"), run_id=d.get("run_id"), status=d.get("status", "error"),
            assertions=list(d.get("assertions") or []), min_confidence=d.get("min_confidence"),
            receipts=d.get("receipts"), receipts_skipped=list(d.get("receipts_skipped") or []),
            prove_error=d.get("prove_error"), error=d.get("error"),
        )


@dataclass
class SuiteResult:
    model_note: str
    timestamp: str
    cases: list = field(default_factory=list)   # list[CaseResult]
    status: str = "error"
    counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_note": self.model_note, "timestamp": self.timestamp, "status": self.status,
            "counts": dict(self.counts), "cases": [c.to_dict() for c in self.cases],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SuiteResult":
        return cls(
            model_note=d.get("model_note", ""), timestamp=d.get("timestamp", ""),
            status=d.get("status", "error"), counts=dict(d.get("counts") or {}),
            cases=[CaseResult.from_dict(c) for c in (d.get("cases") or [])],
        )

    def case(self, name: str) -> CaseResult | None:
        return next((c for c in self.cases if c.name == name), None)


@dataclass
class Diff:
    regressions: list = field(default_factory=list)
    fixed: list = field(default_factory=list)
    drift: list = field(default_factory=list)
    receipt_changes: list = field(default_factory=list)
    new_cases: list = field(default_factory=list)
    removed_cases: list = field(default_factory=list)
    prev_timestamp: str | None = None
    curr_timestamp: str | None = None

    def render(self) -> str:
        lines = [f"CI diff: {self.prev_timestamp or '?'}  ->  {self.curr_timestamp or '?'}"]
        if self.regressions:
            lines.append(f"\nREGRESSIONS ({len(self.regressions)}):")
            for r in self.regressions:
                extra = f" [{r['influence']}]" if r.get("influence") else ""
                lines.append(f"  - {r['case']}{extra}: {r['kind']} "
                              f"({r.get('prev_status')} -> {r.get('curr_status')})")
        else:
            lines.append("\nREGRESSIONS: none")
        if self.fixed:
            lines.append(f"\nFIXED ({len(self.fixed)}):")
            for f_ in self.fixed:
                lines.append(f"  - {f_['case']}: {f_['prev_status']} -> {f_['curr_status']}")
        if self.drift:
            lines.append(f"\nCONFIDENCE DRIFT ({len(self.drift)}):")
            for d in self.drift:
                lines.append(f"  - {d['case']}: min_confidence {d['prev_min_confidence']:.4f} -> "
                              f"{d['curr_min_confidence']:.4f} (delta {d['delta']:+.4f})")
        if self.receipt_changes:
            lines.append(f"\nRECEIPT CHANGES ({len(self.receipt_changes)}):")
            for rc in self.receipt_changes:
                lines.append(f"  - {rc['case']} [{rc['influence']}]: {rc['kind']}")
        if self.new_cases:
            lines.append(f"\nNEW CASES: {', '.join(self.new_cases)}")
        if self.removed_cases:
            lines.append(f"\nREMOVED CASES: {', '.join(self.removed_cases)}")
        return "\n".join(lines)


# ==================================================================================================== utils
def _worst(*statuses: str) -> str:
    worst, rank = "pass", -1
    for s in statuses:
        r = _RANK.get(s, _RANK["error"])
        if r > rank:
            rank, worst = r, s
    return worst


def _min_confidence(run: dict) -> float | None:
    trace = run.get("trace") if isinstance(run, dict) else None
    conf = trace.get("confidence") if isinstance(trace, dict) else None
    vals = []
    for c in conf or []:
        try:
            vals.append(float(c))
        except (TypeError, ValueError):
            continue
    return min(vals) if vals else None


def _translate_expect(expect) -> list:
    """`expect` (suite-manifest shape) -> a list of runner.py assertion dicts. Never raises; a malformed
    shape (not a dict, or a bad value for a known key) is passed through as-is so `runner.evaluate()`'s
    own never-crash, error-status handling is the single source of truth for "this assertion is bad"."""
    if not isinstance(expect, dict):
        return [{"check": None, "value": expect}]
    assertions = []
    for key, value in expect.items():
        if key in ("contains", "not_contains"):
            values = value if isinstance(value, list) else [value]
            for v in values:
                assertions.append({"check": key, "value": v})
        else:
            # finish_reason / min_confidence / max_tokens all map 1:1 onto runner.py's own vocabulary
            # (max_tokens via REPRO_META_KEYS); anything runner.py doesn't recognize becomes a clean
            # "unknown check" error result there -- never a crash here.
            assertions.append({"check": key, "value": value})
    return assertions


def _extract_receipts(prove_result: dict | None) -> tuple[list, list]:
    """A raw `POST /runs/<id>/receipts` response (contracts §7, any of regen/forced/both mode) -> (fired,
    skipped). `fired` is a flat list of `{"influence", "has_effect", "causal_verified"}` covering every
    entry in both `receipts` (regen) and `forced_receipts` (forced) -- both keys are read so `mode: "both"`
    is fully covered; whichever key is absent for the requested mode simply contributes nothing."""
    if not isinstance(prove_result, dict):
        return [], []
    fired = []
    for entry in (prove_result.get("receipts") or []) + (prove_result.get("forced_receipts") or []):
        if not isinstance(entry, dict):
            continue
        fired.append({
            "influence": entry.get("influence"),
            "has_effect": entry.get("has_effect"),
            "causal_verified": entry.get("causal_verified"),
        })
    skipped = list(prove_result.get("skipped") or [])
    return fired, skipped


def _influence_key(inf) -> str:
    if not isinstance(inf, dict):
        return str(inf)
    return "&".join(f"{k}={inf[k]}" for k in sorted(inf))


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


# =============================================================================================== the runner
def run_case(case, client) -> CaseResult:
    """Run one suite case end-to-end: call `client.chat(prompt)`, evaluate its `expect` block via
    `runner.evaluate()`, and (if `prove: true`) call `client.receipts()` to record causal measurements.
    Never raises: a malformed case (not a dict, no prompt, chat()/receipts() raising, ...) degrades to an
    "error"-status CaseResult with an explanatory `.error`/`.prove_error`, exactly mirroring runner.py's own
    contract for a malformed test entry."""
    if not isinstance(case, dict):
        return CaseResult(name="(malformed case)", run_id=None, status="error",
                           error=f"case is not an object: {case!r}")

    name = case.get("name")
    name = name if isinstance(name, str) and name else "(unnamed case)"

    prompt = case.get("prompt")
    messages = case.get("messages")
    extra_messages = None
    if messages is not None:
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(message, dict)
                       or message.get("role") not in ("system", "developer", "user", "assistant")
                       or not isinstance(message.get("content"), str) for message in messages)):
            return CaseResult(name=name, run_id=None, status="error",
                              error="case messages must be supported text message objects")
        if isinstance(prompt, str):
            return CaseResult(name=name, run_id=None, status="error",
                              error="case must define prompt or messages, not both")
        last_user = next((index for index in range(len(messages) - 1, -1, -1)
                          if messages[index].get("role") == "user"), None)
        if last_user is None or last_user != len(messages) - 1:
            return CaseResult(name=name, run_id=None, status="error",
                              error="case messages must end with a user message")
        prompt = messages[last_user]["content"]
        extra_messages = [dict(message) for message in messages[:last_user]]
    if not isinstance(prompt, str) or not prompt.strip():
        return CaseResult(name=name, run_id=None, status="error",
                          error="case has no non-empty prompt or messages")

    try:
        call_kw = {}
        if extra_messages:
            call_kw["extra_messages"] = extra_messages
        if isinstance(case.get("max_tokens"), int) and not isinstance(case.get("max_tokens"), bool):
            call_kw["max_tokens"] = case["max_tokens"]
        if isinstance(case.get("model"), str) and case["model"]:
            call_kw["model"] = case["model"]
        sampling = case.get("sampling")
        if isinstance(sampling, dict):
            for key in ("temperature", "top_p", "seed"):
                if sampling.get(key) is not None:
                    call_kw[key] = sampling[key]
        run = client.chat(prompt, **call_kw)
    except Exception as e:
        return CaseResult(name=name, run_id=None, status="error",
                           error=f"chat() failed: {type(e).__name__}: {e}")
    if not isinstance(run, dict) or not run:
        return CaseResult(name=name, run_id=None, status="error",
                           error="chat() returned no usable run record")

    run_id = run.get("id")
    min_conf = _min_confidence(run)
    prove = bool(case.get("prove", False))
    expect = case.get("expect")
    assertions_spec = _translate_expect(expect) if expect not in (None, {}) else []

    if not assertions_spec and not prove:
        return CaseResult(name=name, run_id=run_id, status="error", min_confidence=min_conf,
                           error="case has no checks (empty/absent 'expect' and 'prove' not set)")

    if assertions_spec:
        evaluated = runner.evaluate(run, {"name": name, "assert": assertions_spec})
        assertions, status = evaluated["assertions"], evaluated["status"]
    else:
        assertions, status = [], "pass"

    receipts_out, receipts_skipped, prove_error = None, [], None
    if prove:
        mode = case.get("prove_mode", "regen")
        try:
            prove_result = client.receipts(run_id, mode)
        except Exception as e:
            prove_result = None
            prove_error = f"receipts() failed: {type(e).__name__}: {e}"
        if prove_result is None and prove_error is None:
            prove_error = "receipts() returned no result (run not found, or the call failed)"
        if prove_error:
            status = _worst(status, "error")
            receipts_out = []
        else:
            receipts_out, receipts_skipped = _extract_receipts(prove_result)

    return CaseResult(name=name, run_id=run_id, status=status, assertions=assertions,
                       min_confidence=min_conf, receipts=receipts_out,
                       receipts_skipped=receipts_skipped, prove_error=prove_error)


def run_suite(suite: dict, client) -> SuiteResult:
    """Run every case in `suite` (the manifest shape documented at module top) against `client` (a `Client`
    or anything duck-typed the same way -- `.chat(prompt)`, `.get_run(id)`, `.receipts(id, mode)`). Pure
    given `client`'s behavior: never raises, never prints. Returns a `SuiteResult` with one `CaseResult`
    per case plus an overall `status` (the worst of all case statuses) and a `counts` tally."""
    model_note = suite.get("model_note", "") if isinstance(suite, dict) else ""
    cases_spec = suite.get("cases") if isinstance(suite, dict) else None
    cases_spec = cases_spec if isinstance(cases_spec, list) else []

    cases = [run_case(c, client) for c in cases_spec]
    status = _worst(*(c.status for c in cases)) if cases else "error"
    counts = {s: 0 for s in ("pass", "fail", "error")}
    for c in cases:
        counts[c.status] = counts.get(c.status, 0) + 1

    return SuiteResult(model_note=str(model_note or ""), timestamp=_now_iso(), cases=cases,
                        status=status, counts=counts)


# ================================================================================================ diffing
def diff_suites(prev: SuiteResult, curr: SuiteResult, *, confidence_drift_threshold: float = 0.1) -> Diff:
    """Compare two `SuiteResult`s (matched by case `name`) and report what changed. This is the regression
    heart of the module:

      - a case that went pass -> fail is a REGRESSION
      - a case that went fail -> pass is FIXED
      - a case that went (pass|fail) -> error is also a REGRESSION (kind "new_error") -- the measurement
        itself broke, which is at least as bad as a failed assertion
      - a case whose recorded min-confidence moved by at least `confidence_drift_threshold` (either
        direction) is flagged as DRIFT, independent of its pass/fail status (a still-passing case can be
        quietly losing confidence)
      - for `prove: true` cases, an influence keyed by its `{section}` spec that was `causal_verified: true`
        in `prev` and is NOT in `curr` is a receipt regression (something that used to be causally
        load-bearing no longer verifiably is) -- also folded into `.regressions`
      - cases present in only one of the two suites are reported separately (`new_cases`/`removed_cases`),
        never silently ignored and never scored as a regression on their own
    """
    prev_by_name = {c.name: c for c in prev.cases}
    curr_by_name = {c.name: c for c in curr.cases}
    common = sorted(set(prev_by_name) & set(curr_by_name))

    regressions, fixed, drift, receipt_changes = [], [], [], []

    for name in common:
        p, c = prev_by_name[name], curr_by_name[name]

        if p.status == "pass" and c.status == "fail":
            regressions.append({"case": name, "kind": "status_regression", "influence": None,
                                 "prev_status": p.status, "curr_status": c.status,
                                 "prev_run_id": p.run_id, "curr_run_id": c.run_id})
        elif p.status == "fail" and c.status == "pass":
            fixed.append({"case": name, "prev_status": p.status, "curr_status": c.status,
                          "prev_run_id": p.run_id, "curr_run_id": c.run_id})
        elif p.status != "error" and c.status == "error":
            regressions.append({"case": name, "kind": "new_error", "influence": None,
                                 "prev_status": p.status, "curr_status": c.status,
                                 "prev_run_id": p.run_id, "curr_run_id": c.run_id})

        if p.min_confidence is not None and c.min_confidence is not None:
            delta = c.min_confidence - p.min_confidence
            if abs(delta) >= confidence_drift_threshold:
                drift.append({"case": name, "prev_min_confidence": p.min_confidence,
                              "curr_min_confidence": c.min_confidence, "delta": delta})

        p_receipts = {_influence_key(r.get("influence")): r for r in (p.receipts or [])}
        c_receipts = {_influence_key(r.get("influence")): r for r in (c.receipts or [])}
        for key in sorted(set(p_receipts) & set(c_receipts)):
            pr, cr = p_receipts[key], c_receipts[key]
            p_verified, c_verified = bool(pr.get("causal_verified")), bool(cr.get("causal_verified"))
            if p_verified and not c_verified:
                receipt_changes.append({"case": name, "influence": key, "kind": "causal_verified_lost",
                                        "prev_causal_verified": p_verified, "curr_causal_verified": c_verified})
                regressions.append({"case": name, "kind": "receipt_regression", "influence": key,
                                    "prev_status": p.status, "curr_status": c.status})
            elif c_verified and not p_verified:
                receipt_changes.append({"case": name, "influence": key, "kind": "causal_verified_gained",
                                        "prev_causal_verified": p_verified, "curr_causal_verified": c_verified})
            if bool(pr.get("has_effect")) != bool(cr.get("has_effect")):
                receipt_changes.append({"case": name, "influence": key, "kind": "has_effect_changed",
                                        "prev_has_effect": pr.get("has_effect"),
                                        "curr_has_effect": cr.get("has_effect")})

    new_cases = sorted(set(curr_by_name) - set(prev_by_name))
    removed_cases = sorted(set(prev_by_name) - set(curr_by_name))

    return Diff(regressions=regressions, fixed=fixed, drift=drift, receipt_changes=receipt_changes,
                new_cases=new_cases, removed_cases=removed_cases,
                prev_timestamp=prev.timestamp, curr_timestamp=curr.timestamp)


# ============================================================================================ persistence
def save_result(result: SuiteResult, *, directory: str | None = None) -> str:
    """Persist `result` as JSON under `directory` (default `~/.clozn/ci/`, injectable for tests) and return
    the path written. Filename is timestamp + a short random suffix, so back-to-back saves in the same
    process (as in a test) never collide."""
    directory = directory or DEFAULT_CI_DIR
    os.makedirs(directory, exist_ok=True)
    fname = f"ci_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(directory, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def load_result(path: str) -> SuiteResult:
    """Load a `SuiteResult` previously written by `save_result`."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return SuiteResult.from_dict(data)
