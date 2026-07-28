"""clozn/cli/ci_report_render.py -- pure renderers over a `clozn ci check` report (`clozn.ci-report.v1`,
see clozn/schemas/defs/clozn.ci-report.v1.json).

Both `render_job_summary` and `render_junit_xml` take ONLY a report dict -- the same shape `clozn ci
check --json`/`--report` writes -- and return a string. Neither does any I/O, opens a socket, or imports
anything model/engine-related. That is the whole point: feature 02's "verify mode" (notes/agent_roadmap/
02-github-action-model-gate.md) must be safe to run on a free CPU GitHub-hosted runner with no model, no
GPU, and no network access, and these renderers are exactly the "main new product code" that mode needs --
turning a report clozn already produces into a GitHub check/job summary and a JUnit export. Purity here is
not a style preference, it is what makes that safety claim checkable: neither function touches a file, a
socket, or `sys`.

WHY THESE TWO LIVE TOGETHER, AND WHY NEITHER DUPLICATES `format_ci_report`
---------------------------------------------------------------------------
Both walk the same `report["checks"]` shape, so splitting "how do I read a check result" into two modules
would just mean maintaining it twice. `clozn.cli.commands.ci_check.format_ci_report` already renders a
human terminal report from the same dict; this module does not re-implement that -- it renders the two
GitHub-specific formats the feature spec calls out (a job/check-summary table and a JUnit XML export),
which terminal output has no use for.

THE "EVIDENCE" COLUMN IS HONEST, NOT COMPLETE
------------------------------------------------
The spec's job-summary table has an "Evidence" column (a receipt/run path). `run_id` -- the only field
that could resolve to one -- is present on `_check_diff`'s and `_check_tiny`'s worst_offenders, but is
NOT present on `_check_golden`'s ("wrong" probes carry no run_id) or on any of `gate_experiment_result`'s
four experiment-mode budget checks' worst_offenders (their `label` dict is case/seed/status only -- see
`ci_check.py`'s `_variant_budget_check`). Because these renderers must stay pure over the report alone
(no re-reading the original experiment artifact to backfill it), the table below renders "-" for Evidence
wherever `run_id` is genuinely absent from the report, rather than fabricating a path. That gap is real,
tracked product debt, not a bug in this renderer -- see the feature 02 plan's "receipts.zip" deferral.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

_EXPERIMENT_BUDGET_FLAGS = {
    "max_execution_errors": "--max-execution-errors",
    "max_target_regressions": "--max-target-regressions",
    "max_guard_regressions": "--max-guard-regressions",
    "min_target_gains": "--min-target-gains",
}


def _fmt_compact(value) -> str:
    """A short, single-line rendering of a budget/observed sub-dict for a Markdown table cell."""
    if value is None:
        return "-"
    if isinstance(value, dict):
        if not value:
            return "-"
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _remediation_command(report: dict) -> str:
    """Reconstruct the exact `clozn ci check` invocation that reproduces this report's own budgets."""
    if report.get("mode") == "experiment":
        parts = [f"clozn ci check --experiment {report.get('experiment_path') or '<result.json>'}"]
        checks = report.get("checks") or {}
        budgets: dict = {}
        for name in ("execution_errors", "target_regressions", "guard_regressions", "target_gains"):
            for key, value in ((checks.get(name) or {}).get("budget") or {}).items():
                if key != "scope":
                    budgets[key] = value
        for key, flag in _EXPERIMENT_BUDGET_FLAGS.items():
            if key in budgets:
                parts.append(f"{flag} {budgets[key]}")
        return " ".join(parts)

    parts = [f"clozn ci check --baseline {report.get('baseline_path') or '<baseline.json>'} "
             f"{report.get('model') or '<model>'}"]
    policy = report.get("identity_policy") or {}
    if policy.get("pin_model") and not policy.get("match"):
        parts.append("--allow-model-change")
    return " ".join(parts)


def _identity_drift_lines(report: dict) -> list[str]:
    """Markdown bullet lines describing any identity drift this report recorded. Empty list if there is
    nothing to warn about (never a fabricated "all clear" line -- absence of a bullet IS the all-clear)."""
    lines: list[str] = []
    if report.get("mode") == "experiment":
        integrity = (report.get("checks") or {}).get("artifact_integrity") or {}
        for finding in integrity.get("worst_offenders") or []:
            kind = finding.get("kind")
            if kind == "variant_identity_changed":
                lines.append(
                    f"- variant `{finding.get('variant')}` was measured under more than one model "
                    f"sha256: {', '.join(finding.get('model_sha256_values') or [])}")
            elif kind == "missing_model_sha256":
                lines.append(
                    f"- run `{finding.get('run_id')}` (case `{finding.get('case')}`, variant "
                    f"`{finding.get('variant')}`) has no recorded model_sha256")
            elif kind == "manifest_sha256_mismatch":
                lines.append(
                    f"- manifest_sha256 mismatch: expected `{finding.get('expected')}`, "
                    f"observed `{finding.get('observed')}`")
        return lines

    policy = report.get("identity_policy") or {}
    if policy.get("pin_model") is not None:
        detail = f"  -- {policy['reason']}" if policy.get("reason") else ""
        lines.append(f"- baseline pin_model={policy.get('pin_model')}  match={policy.get('match')}{detail}")
    live = report.get("live_identity") or {}
    state = live.get("state")
    if state == "mismatch":
        lines.append(
            f"- *** the model that answered (sha256 {live.get('live_sha256')}) is NOT the model this "
            f"report certifies (sha256 {live.get('certified_sha256')}) ***")
    elif state == "unverified":
        lines.append("- identity not verified against a live engine (no live probe ran to compare against)")
    return lines


def _experiment_rows(report: dict) -> list[tuple]:
    """(case, role, seed, result, baseline_status, candidate_status, evidence) rows for the job-summary
    table, one per worst_offender across target_regressions/guard_regressions/target_gains. Each source
    check already caps itself at 20 offenders (see ci_check.py's `_variant_budget_check`)."""
    rows = []
    checks = report.get("checks") or {}
    for check_name, role, result_label in (
        ("target_regressions", "target", "Regression"),
        ("guard_regressions", "guard", "Regression"),
        ("target_gains", "target", "Gain"),
    ):
        for item in (checks.get(check_name) or {}).get("worst_offenders") or []:
            rows.append((
                item.get("case", "-"), role, item.get("seed", "-"), result_label,
                item.get("baseline_status", "-"), item.get("candidate_status", "-"),
                item.get("run_id") or "-",
            ))
    return rows


def render_job_summary(report: dict) -> str:
    """Render `report` (a `clozn ci check` report, `clozn.ci-report.v1`) as GitHub-flavored Markdown
    suitable for `$GITHUB_STEP_SUMMARY`. Pure: takes only `report`, does no I/O, imports nothing beyond
    the stdlib.
    """
    overall = str(report.get("overall", "?")).upper()
    lines = [f"# clozn ci check -- {overall}"]
    if report.get("reason"):
        lines.append(f"\n{report['reason']}")

    mode = report.get("mode")
    if mode == "experiment":
        artifact = report.get("artifact") or {}
        lines.append(
            f"\nExperiment `{artifact.get('experiment_id', '-')}` -- "
            f"baseline=`{artifact.get('baseline_variant', '-')}`  "
            f"candidates={artifact.get('candidate_variants', [])}  cells={artifact.get('cells', '-')}")
        rows = _experiment_rows(report)
        if rows:
            lines.append("\n| Case | Role | Seed | Result | Baseline | Candidate | Evidence |")
            lines.append("|---|---|---:|---|---:|---:|---|")
            for case, role, seed, result, base_status, cand_status, evidence in rows:
                lines.append(
                    f"| {case} | {role} | {seed} | {result} | {base_status} | {cand_status} "
                    f"| {evidence} |")
        else:
            lines.append("\nNo target/guard regressions or target gains to report.")
    else:
        identity = report.get("identity") or {}
        lines.append(
            f"\nModel `{report.get('model', identity.get('model_path', '-'))}` "
            f"(sha256 {identity.get('model_sha256', '-')})")

    lines.append("\n## Checks")
    lines.append("| Check | Result | Reason | Budget | Observed |")
    lines.append("|---|---|---|---|---|")
    for name, check in (report.get("checks") or {}).items():
        mark = "PASS" if check.get("passed") else "FAIL"
        lines.append(
            f"| {name} | {mark} | {check.get('reason') or '-'} | "
            f"{_fmt_compact(check.get('budget'))} | {_fmt_compact(check.get('observed'))} |")

    drift = _identity_drift_lines(report)
    if drift:
        lines.append("\n## Identity")
        lines.extend(drift)

    lines.append("\n## Reproduce locally")
    lines.append(f"```\n{_remediation_command(report)}\n```")

    return "\n".join(lines) + "\n"


def render_junit_xml(report: dict) -> str:
    """Render `report` as JUnit XML: one `<testsuite>` for the whole report, one `<testcase>` per budget
    check in `report["checks"]` (golden/tiny/diff for mode=baseline; artifact_integrity/execution_errors/
    target_regressions/guard_regressions/target_gains for mode=experiment). A failed check gets a
    `<failure>` child carrying its `reason`/`budget`/`observed`. stdlib `xml.etree.ElementTree` only --
    `pyproject.toml` declares `dependencies = []` on purpose, and this must never be the thing that adds
    the first one. Pure: takes only `report`, does no I/O.
    """
    checks = report.get("checks") or {}
    mode = report.get("mode", "?")
    failures = sum(1 for c in checks.values() if not c.get("passed"))

    testsuites = ET.Element("testsuites")
    suite = ET.SubElement(testsuites, "testsuite", {
        "name": f"clozn ci check ({mode})",
        "tests": str(len(checks)),
        "failures": str(failures),
        "errors": "0",
        "time": "0",
    })
    for name, check in checks.items():
        case = ET.SubElement(suite, "testcase", {
            "classname": "clozn.ci_check", "name": name, "time": "0",
        })
        if not check.get("passed"):
            failure = ET.SubElement(case, "failure", {
                "message": check.get("reason") or f"{name} failed its budget",
            })
            failure.text = (f"budget: {_fmt_compact(check.get('budget'))}\n"
                            f"observed: {_fmt_compact(check.get('observed'))}")

    return ET.tostring(testsuites, encoding="utf-8", xml_declaration=True).decode("utf-8")


__all__ = ["render_job_summary", "render_junit_xml"]
