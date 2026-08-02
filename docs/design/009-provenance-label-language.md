# ADR 009 — provenance label language

Status: accepted
Date: 2026-08-02

## Decision

Keep the existing provenance receipt enum (`CONTEXT_CARRIED`, `MIXED`, `PARAMETRIC`,
`INCONCLUSIVE`) unchanged for wire and stored-artifact compatibility. Human-facing surfaces render a
second, conservative label:

| Receipt enum | Human-facing label |
| --- | --- |
| `CONTEXT_CARRIED` | high measured context dependence |
| `MIXED` | mixed measured context dependence |
| `PARAMETRIC` | low measured context dependence |
| `INCONCLUSIVE` | inconclusive context-dependence measurement |

The label names the method's observation. It does not say that an answer literally came from a
document, that the document was the sole cause, or that the result generalizes outside the recorded
prompt, model, attention-knockout control, and measurement floor.

## Rationale

Attention knockout with matched controls is useful evidence about whether the recorded answer changed
when a context region was cut. It is not a general causal attribution system. Replacing the stable
enum would unnecessarily break clients and fixtures; adding a presentation label preserves that
contract while making the product language harder to overread.

## Scope and follow-up

This decision applies to CLI human output and future Studio chips. JSON receipts continue to expose
the stable enum, method, control ratio, dependence, and capability-unavailable reason. A future UI may
show both the enum and label, but must retain the method caveat near the label.
