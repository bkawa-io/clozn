"""Five model-free, synthetic-realistic recorded-run scenarios.

The scenarios contain only ordinary chat messages and descriptive tags.  They
deliberately do not contain an answer field or a source-selection oracle.  A
caller can turn one into the same receipt/unit shell used by the run store for
fixture validation; live evaluation records the actual response separately.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any

from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.runs.context_units import build_context_unit_manifest


SCENARIO_IDS = (
    "single_relevant",
    "distributed",
    "redundant",
    "multi_turn",
    "broad_control",
)


@dataclass(frozen=True)
class RealisticScenario:
    case_id: str
    description: str
    tags: tuple[str, ...]
    messages: tuple[dict[str, str], ...]


def _document(title: str, sections: list[tuple[str, str]]) -> str:
    parts = [f"# {title}", "This reference packet contains operational guidance for the current request."]
    for heading, body in sections:
        parts.extend((f"## {heading}", body))
    return "\n\n".join(parts)


def _single_relevant() -> RealisticScenario:
    sections = [
        ("Travel booking", "Use the approved booking portal for domestic and international travel."),
        ("Hotel limits", "Hotel receipts should identify the traveler, dates, and nightly rate."),
        ("Ground transport", "Local transit may be reimbursed when it is connected to an approved trip."),
        ("Security review", "Devices used abroad must complete the security checklist before departure."),
        ("Expense timing", "Submit ordinary travel expenses within ten business days of returning."),
        ("Regional exception", "A regional office may require a local tax form for some reimbursements."),
        ("Approval routing", "Requests above a team threshold go to the department cost owner."),
        ("Personal extensions", "Personal vacation days are not part of the reimbursable itinerary."),
        ("International airfare", "International fares need an itinerary and the business purpose."),
        ("Lost receipts", "Use the missing-receipt declaration only after checking the expense archive."),
        ("Remote work", "Remote-work equipment follows the equipment purchasing policy."),
        ("Business-class exception", "Flights longer than eight hours may use business class with director approval."),
        ("Traveler safety", "Contact the travel desk when a destination changes after ticketing."),
        ("Reimbursement rule", "For client-site travel, the daily meal allowance is 68 dollars per day."),
        ("Currency conversion", "Use the card statement exchange rate when no local receipt rate is available."),
    ]
    return RealisticScenario(
        "single_relevant",
        "A long travel policy packet with one section that answers the current allowance question.",
        ("document", "single-source", "distractors"),
        (
            {"role": "system", "content": _document("Northstar travel policy", sections)},
            {"role": "user", "content": "What is the daily meal allowance for client-site travel?"},
        ),
    )


def _distributed() -> RealisticScenario:
    sections = [
        ("Release calendar", "The standard review window is five business days."),
        ("Change requests", "A request submitted after the freeze needs a release-manager exception."),
        ("Freeze window", "The quarterly change freeze begins on 18 September."),
        ("Service ownership", "The service owner confirms whether a change is customer-facing."),
        ("Holiday adjustment", "Company holidays do not count as business days in the review window."),
        ("Exception authority", "The release manager may approve a late request when the incident is active."),
        ("Incident response", "An active severity-one incident permits an emergency change path."),
        ("Documentation", "Every approved change needs a rollback note and an owner."),
        ("Maintenance notices", "Customer notices should be posted before a planned maintenance event."),
        ("Audit retention", "Release evidence remains available for one year."),
        ("Calendar ownership", "The operations calendar is the source for the published freeze date."),
        ("Deployment stages", "A normal change moves through review, staging, and production."),
    ]
    return RealisticScenario(
        "distributed",
        "A release request whose timing combines a review rule, a freeze date, and a holiday exception.",
        ("document", "distributed-evidence", "nonadjacent"),
        (
            {"role": "system", "content": _document("Northstar release handbook", sections)},
            {"role": "user", "content": "When is the normal review deadline for a request submitted on 15 September, before the freeze?"},
        ),
    )


def _redundant() -> RealisticScenario:
    sections = [
        ("General badge rule", "A visitor badge must be returned before the visitor leaves the building."),
        ("Reception checklist", "Reception records the badge number and takes the visitor badge back at departure."),
        ("Facilities notes", "Desk reservations are released at the end of the scheduled visit."),
        ("Security handbook", "All temporary visitor badges are collected when the visitor exits the building."),
        ("Contractor access", "Contractors need a sponsor and a valid work order before entry."),
        ("Parking", "Visitor parking is available in the signed short-stay area."),
        ("After-hours entry", "After-hours guests must use the staffed security entrance."),
        ("Incident reporting", "Report a missing badge to the security desk immediately."),
        ("Meeting rooms", "Meeting hosts should release unused rooms in the scheduling system."),
        ("Escorted areas", "Visitors entering a restricted lab remain with their sponsor."),
        ("Records", "Reception retains the visitor log according to the facilities schedule."),
        ("Temporary credentials", "Temporary credentials are not transferable between visitors."),
    ]
    return RealisticScenario(
        "redundant",
        "A facilities packet with two independent statements supporting the same short visitor-exit rule.",
        ("document", "redundant-support", "distractors"),
        (
            {"role": "system", "content": _document("Northstar visitor operations", sections)},
            {"role": "user", "content": "What must happen to a visitor badge when the visitor leaves?"},
        ),
    )


def _turn(title: str, body: str) -> dict[str, str]:
    return {"role": "user", "content": f"# {title}\n\n{body}"}


def _multi_turn() -> RealisticScenario:
    return RealisticScenario(
        "multi_turn",
        "A recorded conversation where a later correction replaces an earlier deployment-window value.",
        ("conversation", "revision", "stale-history"),
        (
            {"role": "system", "content": "# Conversation policy\n\nUse the latest explicit correction when summarizing a decision."},
            _turn("Earlier deployment plan", "The deployment window was planned for Tuesday morning."),
            {"role": "assistant", "content": "# Planning note\n\nI recorded Tuesday morning as the deployment window."},
            _turn("Unrelated staffing note", "The on-call handoff includes Priya and Mateo."),
            {"role": "assistant", "content": "# Staffing note\n\nThe handoff roster is saved for the release team."},
            _turn("Deployment correction", "Correction: the deployment window is Thursday afternoon, not Tuesday morning."),
            {"role": "assistant", "content": "# Corrected plan\n\nI updated the deployment window to Thursday afternoon."},
            _turn("Current request", "What is the final deployment window?"),
        ),
    )


def _broad_control() -> RealisticScenario:
    sections = [
        ("Data handling", "Production exports require encryption at rest and a named data owner."),
        ("Access review", "Access is reviewed quarterly and removed when a role ends."),
        ("Change control", "Production changes require a ticket, an approver, and a rollback plan."),
        ("Backups", "Critical stores are backed up daily and restoration is tested monthly."),
        ("Incident response", "Severity-one incidents page the primary and secondary responders."),
        ("Availability", "The service target is 99.9 percent monthly availability."),
        ("Logging", "Security logs are retained for twelve months and access is audited."),
        ("Vendor review", "Vendors handling customer data complete an annual security review."),
        ("Secrets", "Credentials are stored in the managed vault and rotated every ninety days."),
        ("Network boundary", "Administrative access is limited to the private operations network."),
        ("Recovery", "The recovery exercise records the recovery point and recovery time results."),
        ("Support", "Customer-impacting incidents receive a status update at least every hour."),
    ]
    return RealisticScenario(
        "broad_control",
        "A compact synthesis request that draws on many independent operational constraints.",
        ("document", "broad-synthesis", "control"),
        (
            {"role": "system", "content": _document("Northstar operating standard", sections)},
            {"role": "user", "content": "Summarize the key operating constraints from this standard in one compact sentence."},
        ),
    )


_BUILDERS = {
    "single_relevant": _single_relevant,
    "distributed": _distributed,
    "redundant": _redundant,
    "multi_turn": _multi_turn,
    "broad_control": _broad_control,
}


def built_in_scenarios() -> tuple[RealisticScenario, ...]:
    """Return the stable built-in registry in benchmark order."""
    return tuple(_BUILDERS[case_id]() for case_id in SCENARIO_IDS)


def get_scenario(case_id: str) -> RealisticScenario:
    if case_id not in _BUILDERS:
        raise KeyError(f"unknown realistic scenario: {case_id}")
    return _BUILDERS[case_id]()


def build_fixture_run(scenario: RealisticScenario, *, run_id: str | None = None) -> dict[str, Any]:
    """Build an ordinary run-shaped receipt/unit shell without model evidence."""
    rid = run_id or f"fixture_{scenario.case_id}"
    messages = [dict(message) for message in scenario.messages]
    receipt = build_context_receipt(messages=messages, run_id=rid, privacy="full")
    run: dict[str, Any] = {"id": rid, "messages": deepcopy(messages), "context_receipt": receipt}
    run["context_units"] = build_context_unit_manifest(run)
    return run


def validate_scenario(scenario: RealisticScenario, *, max_units: int = 50) -> dict[str, Any]:
    """Validate fixture shape and the real Context Units/search-universe seams."""
    if not isinstance(scenario, RealisticScenario):
        raise ValueError("scenario must be a RealisticScenario")
    if not scenario.case_id or scenario.case_id not in SCENARIO_IDS:
        raise ValueError("scenario has an unknown case ID")
    if not scenario.description or not scenario.tags:
        raise ValueError(f"scenario {scenario.case_id} needs a description and tags")
    if re.search(r"(?:/Users/|/home/|[A-Za-z]:[\\/])", scenario.description):
        raise ValueError(f"scenario {scenario.case_id} description contains a local path")
    if not isinstance(scenario.messages, tuple) or not scenario.messages:
        raise ValueError(f"scenario {scenario.case_id} needs non-empty messages")
    if not any(message.get("role") == "user" for message in scenario.messages):
        raise ValueError(f"scenario {scenario.case_id} needs a user message")
    for message in scenario.messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError(f"scenario {scenario.case_id} contains an invalid chat message")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"scenario {scenario.case_id} contains empty message content")
        if re.search(r"(?:/Users/|/home/|[A-Za-z]:[\\/])", message["content"]):
            raise ValueError(f"scenario {scenario.case_id} contains a local path")

    run = build_fixture_run(scenario)
    manifest = run["context_units"]
    removable = manifest.get("default_source_ids") if isinstance(manifest, dict) else None
    if not isinstance(removable, list) or len(removable) < 2:
        raise ValueError(f"scenario {scenario.case_id} has no removable context units")
    universe = plan_context_search_universe(run, manifest, max_units=max_units)
    if universe.get("status") != "planned" or not universe.get("source_ids"):
        raise ValueError(f"scenario {scenario.case_id} has no planned search universe")
    if universe["source_count"] != len(universe["source_ids"]):
        raise ValueError(f"scenario {scenario.case_id} search universe count is inconsistent")
    return {
        "case_id": scenario.case_id,
        "description": scenario.description,
        "case_tags": list(scenario.tags),
        "removable_unit_count": len(universe["source_ids"]),
        "protected_message_indices": list(manifest["protected_message_indices"]),
        "universe_id": universe["universe_id"],
        "source_count": universe["source_count"],
    }


def validate_registry(*, max_units: int = 50) -> list[dict[str, Any]]:
    scenarios = built_in_scenarios()
    if tuple(scenario.case_id for scenario in scenarios) != SCENARIO_IDS:
        raise ValueError("built-in scenario registry order changed")
    if len({scenario.case_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("built-in scenario IDs are not unique")
    return [validate_scenario(scenario, max_units=max_units) for scenario in scenarios]


__all__ = [
    "RealisticScenario",
    "SCENARIO_IDS",
    "build_fixture_run",
    "built_in_scenarios",
    "get_scenario",
    "validate_registry",
    "validate_scenario",
]
