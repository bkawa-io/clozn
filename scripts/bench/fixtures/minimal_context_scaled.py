"""Six deterministic, long-form realistic traces for evaluation v1.

These fixtures deliberately contain no answer key or source-selection oracle.
They are ordinary chat messages whose structural spans are discovered by the
same Context Units and bounded-search-universe code used by recorded runs.
"""
from __future__ import annotations

from typing import Iterable

from .minimal_context_realistic import RealisticScenario, build_fixture_run
from clozn.runs.context_search_universe import plan_context_search_universe


SCALED_SCENARIO_IDS = (
    "long_rag_single",
    "long_rag_distributed",
    "long_rag_redundant",
    "long_multi_turn",
    "long_broad_synthesis",
    "code_context",
)


_DOMAINS = (
    "travel operations", "release governance", "customer support", "data stewardship",
    "vendor management", "workplace safety", "finance controls", "identity operations",
    "service reliability", "records administration", "privacy review", "communications",
)


def _policy_document(title: str, topics: Iterable[tuple[str, str, str]]) -> str:
    parts = [
        f"# {title}",
        "This controlled reference packet is a realistic internal operating manual. "
        "Headings identify independent sections, while the two subsections preserve "
        "the distinction between a rule and the evidence used to apply it.",
    ]
    for index, (heading, rule, evidence) in enumerate(topics, start=1):
        domain = _DOMAINS[(index - 1) % len(_DOMAINS)]
        parts.extend((
            f"## {heading}",
            f"The {domain} owner uses this section when reviewing a request. {rule} "
            "The owner records the decision in the ordinary work item, identifies "
            "the responsible team, and avoids treating a draft or an informal chat "
            "message as an approval. If circumstances differ, the exception path "
            "below takes precedence only when its evidence is present.",
            "### Operating rule",
            f"{rule} The normal workflow is to check the named owner, confirm the "
            "effective date, and state any assumption in the decision note. "
            "Requests that combine several sections should preserve each applicable "
            "constraint instead of silently choosing the most convenient one.",
            "### Evidence and exception",
            f"{evidence} Evidence may be a dated ticket, an approved calendar entry, "
            "or a signed record from the responsible function. A reviewer should "
            "not infer an exception merely because a similar request was approved "
            "in an earlier quarter.",
        ))
    return "\n\n".join(parts)


def _topics(count: int, *, focus: dict[int, tuple[str, str, str]] | None = None) -> list[tuple[str, str, str]]:
    focus = focus or {}
    topics: list[tuple[str, str, str]] = []
    for index in range(1, count + 1):
        if index in focus:
            topics.append(focus[index])
            continue
        domain = _DOMAINS[(index - 1) % len(_DOMAINS)]
        topics.append((
            f"{domain.title()} reference {index:02d}",
            f"Routine {domain} requests use the standard review queue, retain the "
            f"request identifier, and follow the {domain} service calendar.",
            f"The {domain} coordinator records the review date and the named "
            f"approver in the case file before closing the request.",
        ))
    return topics


def _long_rag_single() -> RealisticScenario:
    focus = {
        22: (
            "Client-site meal allowance",
            "For approved client-site travel, the daily meal allowance is 68 US dollars per traveler per day. "
            "The allowance covers meals during the client visit and is not a separate bonus or a hotel limit.",
            "The expense record must identify the client site, travel dates, and traveler. A receipt is still kept "
            "when available, but the policy amount is the controlling daily cap for this question.",
        ),
    }
    return RealisticScenario(
        "long_rag_single",
        "A long travel and operations manual with one localized allowance section and many unrelated policy sections.",
        ("scaled_realistic", "document", "single-source", "localized-evidence"),
        (
            {"role": "system", "content": _policy_document("Northstar field operations manual", _topics(26, focus=focus))},
            {"role": "user", "content": "What is the daily meal allowance for approved client-site travel? Answer briefly and include the unit."},
        ),
    )


def _long_rag_distributed() -> RealisticScenario:
    focus = {
        7: (
            "Standard review window",
            "A normal release request is reviewed within five business days after the complete request reaches the release queue.",
            "The queue timestamp is the first complete submission, not the time a draft was discussed in a meeting.",
        ),
        17: (
            "Published freeze date",
            "The quarterly production change freeze begins on 18 September at 17:00 Pacific time.",
            "The operations calendar is the authoritative source for the freeze date and time; local copies can be stale.",
        ),
        25: (
            "Business-day calendar",
            "Company holidays and weekends do not count toward the five-business-day review window.",
            "The release coordinator checks the company holiday calendar before calculating the due date.",
        ),
    }
    return RealisticScenario(
        "long_rag_distributed",
        "A long release handbook with separated timing, freeze-calendar, and business-day evidence regions.",
        ("scaled_realistic", "document", "distributed-evidence", "nonadjacent"),
        (
            {"role": "system", "content": _policy_document("Northstar release and change handbook", _topics(26, focus=focus))},
            {"role": "user", "content": "For a complete normal request submitted on 15 September before the freeze, what review rule and calendar facts determine its deadline? Keep the answer concise."},
        ),
    )


def _long_rag_redundant() -> RealisticScenario:
    focus = {
        6: (
            "Visitor badge departure rule",
            "Every visitor badge must be collected and returned to reception before the visitor leaves the building.",
            "Reception logs the badge number at departure and reports an unreturned badge to the security desk immediately.",
        ),
        18: (
            "Security exit control",
            "Security staff independently require all temporary visitor badges to be surrendered at the exit checkpoint.",
            "The exit log and the reception log are separate records, so either record can establish that the badge was returned.",
        ),
        24: (
            "Badge incident response",
            "A missing visitor badge is treated as an access incident and must be reported before the visitor departs or as soon as the loss is discovered.",
            "The sponsor supplies the visitor name and last known location; this incident process does not replace the return requirement.",
        ),
    }
    return RealisticScenario(
        "long_rag_redundant",
        "A long facilities manual with independent badge-return statements separated by operational distractors.",
        ("scaled_realistic", "document", "redundant-support", "independent-regions", "distractors"),
        (
            {"role": "system", "content": _policy_document("Northstar visitor and facilities manual", _topics(26, focus=focus))},
            {"role": "user", "content": "What must happen to a visitor badge when the visitor leaves, and what should staff do if it is missing? Answer in two short clauses."},
        ),
    )


def _long_broad_synthesis() -> RealisticScenario:
    focus = {
        4: (
            "Production data exports",
            "Production exports require encryption at rest, a named data owner, and a documented destination before transfer.",
            "The export ticket records the dataset classification, destination, owner, and deletion date.",
        ),
        12: (
            "Access review cadence",
            "Privileged access is reviewed quarterly and removed promptly when the business role ends.",
            "The identity owner retains the review sign-off and the deprovisioning event identifier.",
        ),
        20: (
            "Production change control",
            "A production change requires a tracked ticket, an approver, a rollback plan, and a verification step.",
            "Emergency changes use the incident record and receive a retrospective review after service stability returns.",
        ),
        25: (
            "Incident communications",
            "A customer-impacting severity-one incident receives an initial status update and recurring updates at least every hour until recovery.",
            "The incident commander owns the update cadence and records material changes in the incident timeline.",
        ),
    }
    return RealisticScenario(
        "long_broad_synthesis",
        "A broad operating standard where a compact answer must synthesize constraints from many sections.",
        ("scaled_realistic", "document", "broad-synthesis", "control"),
        (
            {"role": "system", "content": _policy_document("Northstar operating and assurance standard", _topics(26, focus=focus))},
            {"role": "user", "content": "Summarize the key controls for production data, privileged access, production changes, and severe customer incidents in one compact sentence."},
        ),
    )


def _conversation_message(role: str, title: str, body: str) -> dict[str, str]:
    return {"role": role, "content": f"# {title}\n\n{body}\n\n"
            "This update belongs to the same release record as the surrounding turns. "
            "The owner checks the dated decision, preserves the reason for a change, "
            "and does not promote an unapproved proposal into the current state. "
            "Operational details remain useful only when they are consistent with the "
            "latest scope and approval note. The handoff should quote the current value "
            "and leave older values identifiable as history.\n\n"
            "## Decision metadata\n\n"
            "The entry is reviewed against the release record, the approval state, and "
            "the current scope before it is repeated in a handoff. A later turn may "
            "correct a date, owner, or boundary, so the reader must retain enough "
            "context to tell which value is current and which value is historical. "
            "This metadata describes the record-keeping behavior rather than adding a "
            "new deployment decision."}


def _long_multi_turn() -> RealisticScenario:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "# Conversation policy\n\nUse the latest explicit correction when summarizing a decision. Preserve current scope and distinguish a proposal from an approved final state."},
    ]
    turns = [
        ("user", "Project intake", "We are preparing the Northstar billing migration. Track decisions in the release record and call out changes explicitly."),
        ("assistant", "Intake note", "I will use the release record as the decision source and will separate planning ideas from approved changes."),
        ("user", "Initial deployment plan", "The first plan places the deployment on Tuesday morning after the database rehearsal."),
        ("assistant", "Initial schedule", "I recorded Tuesday morning as the proposed deployment window, conditional on the rehearsal."),
        ("user", "Staffing detail", "The primary on-call is Priya and the backup is Mateo for the rehearsal week."),
        ("assistant", "Roster note", "The rehearsal roster contains Priya as primary and Mateo as backup."),
        ("user", "Scope clarification", "The billing migration covers invoices and credits, but not the customer portal redesign."),
        ("assistant", "Scope note", "I will keep the portal redesign outside the migration scope."),
        ("user", "Risk discussion", "The largest risk is a delayed ledger backfill; the rollback should leave the existing invoice path available."),
        ("assistant", "Risk note", "The rollback discussion should preserve the existing invoice path if the ledger backfill is delayed."),
        ("user", "Calendar update", "The database rehearsal moved to Wednesday afternoon because the test data refresh was late."),
        ("assistant", "Rehearsal update", "The rehearsal is now Wednesday afternoon; the deployment proposal remains Tuesday until the release owner decides otherwise."),
        ("user", "Approval status", "The release owner has not approved the deployment window yet, so Tuesday is still only a proposal."),
        ("assistant", "Approval note", "Tuesday should not be described as final because release-owner approval is still pending."),
        ("user", "Correction", "Correction: the deployment window is Thursday afternoon, not Tuesday morning. The release owner approved Thursday."),
        ("assistant", "Corrected schedule", "I updated the approved deployment window to Thursday afternoon and marked Tuesday as stale."),
        ("user", "Operational detail", "Keep Priya primary and Mateo backup, and run the ledger backfill verification before the change begins."),
        ("assistant", "Runbook update", "The runbook retains Priya and Mateo and places ledger-backfill verification before deployment."),
        ("user", "Final scope correction", "The current request is only for the approved deployment window; do not restate the portal redesign or old Tuesday plan."),
        ("assistant", "Current-state note", "The current question is limited to the approved deployment window; older plans are historical."),
        ("user", "Handoff note", "The release handoff should point to the approved release record and preserve the Thursday decision."),
        ("assistant", "Handoff note", "The handoff points to the approved release record and preserves Thursday afternoon as the current decision."),
    ]
    messages.extend(_conversation_message(role, title, body) for role, title, body in turns)
    messages.append(_conversation_message("user", "Current request", "What is the final approved deployment window? Answer with the current state only."))
    return RealisticScenario(
        "long_multi_turn",
        "A long recorded conversation with prior topic shifts, stale proposals, explicit corrections, and a final current-state question.",
        ("scaled_realistic", "conversation", "revision", "topic-shifts", "stale-history"),
        tuple(messages),
    )


_CODE_FILES = (
    ("services/billing/models.py", (
        ("Invoice model", "from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass Invoice:\n    invoice_id: str\n    customer_id: str\n    amount_cents: int\n    currency: str\n    status: str\n"),
        ("Credit validation", "def validate_credit(invoice: Invoice, credit_cents: int) -> bool:\n    if credit_cents < 0 or credit_cents > invoice.amount_cents:\n        return False\n    return invoice.status in {\"open\", \"overdue\"}\n"),
        ("Ledger key", "def ledger_key(invoice: Invoice) -> tuple[str, str]:\n    return (invoice.customer_id, invoice.currency)\n"),
    )),
    ("services/billing/ledger.py", (
        ("Ledger repository", "class LedgerRepository:\n    def __init__(self, store):\n        self.store = store\n\n    def append(self, invoice_id: str, delta_cents: int) -> None:\n        self.store.insert(\"ledger_entries\", {\"invoice_id\": invoice_id, \"delta_cents\": delta_cents})\n"),
        ("Idempotency lookup", "    def already_applied(self, operation_id: str) -> bool:\n        return self.store.exists(\"ledger_operations\", operation_id)\n"),
        ("Backfill guard", "    def begin_backfill(self, batch_id: str) -> None:\n        if self.store.exists(\"backfills\", batch_id):\n            raise ValueError(\"backfill already started\")\n        self.store.insert(\"backfills\", {\"batch_id\": batch_id, \"state\": \"running\"})\n"),
    )),
    ("services/billing/credits.py", (
        ("Credit command", "class ApplyCredit:\n    def __init__(self, ledger, clock):\n        self.ledger = ledger\n        self.clock = clock\n"),
        ("Apply operation", "    def execute(self, invoice: Invoice, credit_cents: int, operation_id: str) -> None:\n        if not validate_credit(invoice, credit_cents):\n            raise ValueError(\"invalid credit\")\n        if self.ledger.already_applied(operation_id):\n            return\n        self.ledger.append(invoice.invoice_id, -credit_cents)\n"),
        ("Audit record", "        self.ledger.store.insert(\"credit_audit\", {\"operation_id\": operation_id, \"at\": self.clock.now()})\n"),
    )),
    ("api/routes.py", (
        ("Request schema", "class CreditRequest(BaseModel):\n    invoice_id: str\n    amount_cents: int\n    operation_id: str\n"),
        ("Credit route", "@router.post(\"/invoices/{invoice_id}/credits\")\ndef create_credit(invoice_id: str, request: CreditRequest, service: ApplyCredit = Depends(get_service)):\n    invoice = service.load_invoice(invoice_id)\n    service.execute(invoice, request.amount_cents, request.operation_id)\n    return {\"status\": \"accepted\"}\n"),
        ("Error mapping", "@router.exception_handler(ValueError)\ndef validation_error(request, exc):\n    return JSONResponse(status_code=422, content={\"error\": str(exc)})\n"),
    )),
    ("workers/backfill.py", (
        ("Batch reader", "def read_batches(source, size: int = 250):\n    batch = []\n    for row in source:\n        batch.append(row)\n        if len(batch) == size:\n            yield batch\n            batch = []\n    if batch:\n        yield batch\n"),
        ("Backfill worker", "def run_backfill(repository, source, batch_id: str) -> int:\n    repository.begin_backfill(batch_id)\n    count = 0\n    for batch in read_batches(source):\n        for row in batch:\n            repository.append(row.invoice_id, row.delta_cents)\n            count += 1\n    return count\n"),
        ("Retry policy", "def retryable(error: Exception) -> bool:\n    return isinstance(error, (TimeoutError, ConnectionError))\n"),
    )),
    ("tests/test_credits.py", (
        ("Valid credit", "def test_valid_credit_writes_negative_delta(fake_invoice, command, ledger):\n    command.execute(fake_invoice, 250, \"op-1\")\n    ledger.append.assert_called_once_with(fake_invoice.invoice_id, -250)\n"),
        ("Duplicate operation", "def test_duplicate_operation_is_noop(fake_invoice, command, ledger):\n    ledger.already_applied.return_value = True\n    command.execute(fake_invoice, 250, \"op-1\")\n    ledger.append.assert_not_called()\n"),
        ("Invalid amount", "def test_credit_above_invoice_is_rejected(fake_invoice, command):\n    with pytest.raises(ValueError, match=\"invalid credit\"):\n        command.execute(fake_invoice, 999999, \"op-2\")\n"),
    )),
    ("config/feature_flags.py", (
        ("Flag model", "@dataclass(frozen=True)\nclass FeatureFlags:\n    ledger_backfill: bool = False\n    strict_credit_audit: bool = True\n"),
        ("Flag loading", "def load_flags(environment: str, source) -> FeatureFlags:\n    values = source.for_environment(environment)\n    return FeatureFlags(ledger_backfill=bool(values.get(\"ledger_backfill\")), strict_credit_audit=values.get(\"strict_credit_audit\", True))\n"),
        ("Safe default", "def can_backfill(flags: FeatureFlags) -> bool:\n    return flags.ledger_backfill\n"),
    )),
    ("docs/credit-flow.md", (
        ("Command contract", "The credit command validates the invoice, checks idempotency, writes the ledger delta, and records an audit event. The API returns accepted only after the command completes."),
        ("Rollback behavior", "A failed audit write must be visible to the caller. Operators use the incident procedure and do not silently retry a credit whose ledger operation may already exist."),
        ("Question focus", "A compact behavioral question should be answerable by following the route, command, ledger, and test files together rather than by relying on a single comment."),
    )),
)


def _code_context() -> RealisticScenario:
    parts = [
        "# Repository snapshot",
        "The following files are from a small billing service. Treat file names, imports, tests, and comments as one code context.",
    ]
    for path, sections in _CODE_FILES:
        parts.extend((f"## File: {path}", f"The file {path} participates in the billing credit path."))
        for heading, body in sections:
            parts.extend((
                f"### {heading}",
                f"The {heading.lower()} portion of {path} is relevant to tracing behavior. "
                "Read its inputs, side effects, and failure behavior together with the "
                "neighboring files; a function name alone is not a guarantee. The test "
                "suite and the repository notes are part of the intended contract.\n\n"
                "During review, follow the value from the caller into the command, "
                "then check the repository operation and the observable response. "
                "A retry path may be safe only when its idempotency condition is "
                "checked before a write. Keep validation errors distinct from storage "
                "failures, and do not infer a transaction boundary that the code does "
                "not implement.\n\n"
                f"```python\n{body}\n```",
            ))
    return RealisticScenario(
        "code_context",
        "A realistic multi-file billing repository with a compact behavioral question and a short expected-style response.",
        ("scaled_realistic", "code", "multi-file", "behavioral-question"),
        (
            {"role": "system", "content": "# Code review context\n\nRead the supplied repository snapshot as source context. Explain behavior from the code and tests, and distinguish a guaranteed path from a likely operational consequence.\n\n" + "\n\n".join(parts)},
            {"role": "user", "content": "If the same credit operation is submitted twice, what happens on the second submission? Answer in two short sentences and name the key guard."},
        ),
    )


_BUILDERS = {
    "long_rag_single": _long_rag_single,
    "long_rag_distributed": _long_rag_distributed,
    "long_rag_redundant": _long_rag_redundant,
    "long_multi_turn": _long_multi_turn,
    "long_broad_synthesis": _long_broad_synthesis,
    "code_context": _code_context,
}


def built_in_scaled_scenarios() -> tuple[RealisticScenario, ...]:
    return tuple(_BUILDERS[case_id]() for case_id in SCALED_SCENARIO_IDS)


def get_scaled_scenario(case_id: str) -> RealisticScenario:
    if case_id not in _BUILDERS:
        raise KeyError(f"unknown scaled realistic scenario: {case_id}")
    return _BUILDERS[case_id]()


def validate_scaled_scenario(scenario: RealisticScenario, *, max_units: int = 50) -> dict:
    """Validate shape and structural scale without asserting model token counts."""
    if not isinstance(scenario, RealisticScenario) or scenario.case_id not in SCALED_SCENARIO_IDS:
        raise ValueError("scenario is not a registered scaled scenario")
    if not scenario.description or not scenario.tags or not scenario.messages:
        raise ValueError(f"scaled scenario {scenario.case_id} is incomplete")
    if not any(message.get("role") == "user" for message in scenario.messages):
        raise ValueError(f"scaled scenario {scenario.case_id} needs a user message")
    for message in scenario.messages:
        if message.get("role") not in {"system", "user", "assistant"} or not message.get("content", "").strip():
            raise ValueError(f"scaled scenario {scenario.case_id} contains an invalid message")
    run = build_fixture_run(scenario)
    manifest = run["context_units"]
    universe = plan_context_search_universe(run, manifest, max_units=max_units)
    if universe.get("status") != "planned":
        raise ValueError(f"scaled scenario {scenario.case_id} cannot be bounded: {universe.get('condition')}")
    total_chars = sum(len(message["content"]) for message in scenario.messages)
    if total_chars < 9000:
        raise ValueError(f"scaled scenario {scenario.case_id} is not structurally large enough")
    if len(manifest.get("units") or []) < 20:
        raise ValueError(f"scaled scenario {scenario.case_id} has too few raw Context Units")
    return {
        "case_id": scenario.case_id,
        "description": scenario.description,
        "case_tags": list(scenario.tags),
        "message_count": len(scenario.messages),
        "total_message_chars": total_chars,
        "raw_context_unit_count": len(manifest.get("units") or []),
        "bounded_search_universe_count": universe.get("source_count"),
        "removable_message_indices": sorted({
            int(unit["message_index"]) for unit in manifest.get("units") or []
            if isinstance(unit.get("message_index"), int)
        }),
        "protected_message_indices": list(manifest.get("protected_message_indices") or []),
        "universe_id": universe.get("universe_id"),
    }


def validate_scaled_registry(*, max_units: int = 50) -> list[dict]:
    scenarios = built_in_scaled_scenarios()
    if tuple(item.case_id for item in scenarios) != SCALED_SCENARIO_IDS:
        raise ValueError("scaled scenario registry order changed")
    if len({item.case_id for item in scenarios}) != len(scenarios):
        raise ValueError("scaled scenario IDs are not unique")
    return [validate_scaled_scenario(item, max_units=max_units) for item in scenarios]


__all__ = [
    "SCALED_SCENARIO_IDS",
    "built_in_scaled_scenarios",
    "get_scaled_scenario",
    "validate_scaled_registry",
    "validate_scaled_scenario",
]
