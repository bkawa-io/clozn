/**
 * C4 -- "Ask another question" investigation entry point. A structured directory of the investigations
 * clozn can actually run against a recorded run, never a free-form chatbot pretending to know evidence
 * (notes/EPIC_ROADMAP_A-F_Q_M.md's own C4 section, verbatim: "Structured action routing, NOT a
 * free-form chatbot pretending to know evidence").
 *
 * THE HONESTY CORE
 * -----------------
 * `INVESTIGATION_QUESTIONS` is a STATIC registry, verified against the actual backend code (not the
 * dispatch brief's own table, which this file's authoring checked rather than trusted), not a per-run
 * probe:
 *
 *   - why                -> GET /runs/<id>/diagnosis-findings (D1/D2)         EXISTS
 *   - what_mattered      -> the cross-linked prompt/answer influence heatmap (C2)     EXISTS
 *   - what_received      -> GET /runs/<id>/investigation (B2/C1)              EXISTS
 *   - claims_supported   -> GET /runs/<id>/claim-support (E1/E2/E3)           EXISTS
 *   - retry_correction   -> corrective-flow preview/confirm (D3/D5)           EXISTS
 *   - second_opinion     -> a live second model's own answer, run to compare against   NOT BUILT --
 *                           clozn/runs/token_workbench_actions.py's mechanistic-diff action always
 *                           returns 422 "cross_model_execution_not_wired" over HTTP (see
 *                           clozn/server/routes/token_workbench_actions.py's `_mechanistic_diff_action`);
 *                           only `clozn diff-model` at the CLI can load two GGUFs today.
 *   - without_passage    -> arbitrary-span ablation (C3)                      PARTIAL --
 *                           clozn/server/routes/section_influence.py ablates NAMED prompt sections (a
 *                           run's own `sections` manifest) with no regeneration; there is no
 *                           arbitrary-span counterpart. notes/EPIC_ROADMAP_A-F_Q_M.md's own C3 audit,
 *                           verbatim: "you can ablate 'the RAG context section'; you cannot ablate
 *                           'these two sentences I highlighted'." The coverage that DOES exist already
 *                           renders in the "What mattered?" panel this question routes to.
 *
 * Per-run absence (a run with nothing measured yet) is deliberately NOT this registry's job to detect --
 * every destination panel (WhatMattered.tsx, DiagnosisRepair.tsx, ReceivedContext.tsx,
 * ClaimVerification.tsx) already renders its own honest "not measured" / "unavailable" state once the
 * user gets there. This file answers one question only: does clozn have this KIND of investigation at
 * all -- a property of the product, not of the run currently open.
 *
 * Every target below is an in-page anchor, never a route change. This whole feature is mounted as a
 * `lens.evidence` slot panel (see `src/slots/lens.evidence/ask-another-question.tsx`), so every
 * destination it can honestly point to is already rendered as a SIBLING on the very same page (see
 * `src/components/SlotHost.tsx`) -- there is nowhere to navigate to, only somewhere to scroll.
 */

export type InvestigationQuestionId =
  | "why"
  | "what_mattered"
  | "what_received"
  | "claims_supported"
  | "retry_correction"
  | "second_opinion"
  | "without_passage";

export type InvestigationCapability = "available" | "partial" | "unavailable";

export interface InvestigationTarget {
  /** DOM id of the destination heading, already rendered by a sibling slot panel -- scrolled into
   * view, never a hash/route change (see this module's own doc comment). */
  anchorId: string;
  /** One line naming the real surface being opened. Shown on the chip and recorded verbatim into
   * investigation history, so a later "revisit" always describes where it actually went. */
  description: string;
}

interface InvestigationQuestionCommon {
  id: InvestigationQuestionId;
  label: string;
  /** The real capability backing this question, named precisely enough that the claim above is itself
   * checkable -- never a vague "coming soon". */
  backedBy: string;
  /** Free-text intent keywords, lowercase, matched as substrings against the typed query (see
   * `matchInvestigationIntent` below). Deliberately short, common English phrasing -- this is a router,
   * not an NLU model. */
  keywords: string[];
}

/** Mirrors this codebase's own discipline for never collapsing distinct states into one shape (see
 * e.g. `RepairFinding` in data/diagnosisRepair.ts): only "available" and "partial" carry a `target`
 * (there is somewhere real to send the user); only "partial" and "unavailable" carry a `reason` (there
 * is something the user needs told). An "available" question with a missing target, or an
 * "unavailable" one with no reason, fails the TypeScript build instead of silently rendering blank. */
export type InvestigationQuestion =
  | (InvestigationQuestionCommon & { capability: "available"; target: InvestigationTarget })
  | (InvestigationQuestionCommon & { capability: "partial"; reason: string; target: InvestigationTarget })
  | (InvestigationQuestionCommon & { capability: "unavailable"; reason: string });

export const INVESTIGATION_QUESTIONS: InvestigationQuestion[] = [
  {
    id: "why",
    label: "Why?",
    capability: "available",
    backedBy: "GET /runs/<id>/diagnosis-findings (D1/D2)",
    target: { anchorId: "diagnosis-repair-title", description: "the rule-engine findings and plain-language narrative below" },
    keywords: ["why", "reason", "cause", "explain"],
  },
  {
    id: "what_mattered",
    label: "What mattered?",
    capability: "available",
    backedBy: "cross-linked prompt/answer influence heatmap (C2)",
    target: { anchorId: "what-mattered-title", description: "the cross-linked prompt/answer heatmap below" },
    keywords: ["matter", "mattered", "influence", "important", "affected", "impact"],
  },
  {
    id: "what_received",
    label: "What did the model receive?",
    capability: "available",
    backedBy: "GET /runs/<id>/investigation (B2/C1)",
    target: { anchorId: "received-context-title", description: "the delivered-context receipt below" },
    keywords: ["receive", "received", "context", "prompt", "input", "see", "saw", "sent"],
  },
  {
    id: "claims_supported",
    label: "Which claims are supported?",
    capability: "available",
    backedBy: "GET /runs/<id>/claim-support (E1/E2/E3)",
    target: { anchorId: "claim-verify-title", description: "the claim-by-claim verification below" },
    keywords: ["claim", "claims", "support", "supported", "true", "accurate", "verify", "fact"],
  },
  {
    id: "retry_correction",
    label: "Retry with a correction",
    capability: "available",
    backedBy: "corrective-flow preview/confirm (D3/D5)",
    target: { anchorId: "diagnosis-repair-retries-title", description: "corrective retries below" },
    keywords: ["retry", "correct", "correction", "fix", "redo", "again"],
  },
  {
    id: "second_opinion",
    label: "Would another model disagree?",
    capability: "unavailable",
    backedBy: "cross-model answer comparison (E4)",
    reason: "model second opinion is not implemented yet -- clozn can compare two models' internals "
      + "once both are loaded (`clozn diff-model` at the CLI), but Studio has no route yet to "
      + "run a second model's own answer for comparison.",
    keywords: ["another model", "other model", "different model", "second opinion", "disagree", "second model"],
  },
  {
    id: "without_passage",
    label: "What happens without this passage?",
    capability: "partial",
    backedBy: "named-section ablation, no regeneration (C3, partial)",
    reason: "covers context sources already identified for this run, not a span you pick freely -- "
      + "arbitrary-span ablation is not built yet.",
    target: {
      anchorId: "what-mattered-title",
      description: "the measured per-source effect below (named sources only, not an arbitrary span)",
    },
    keywords: ["without", "remove", "removed", "delete", "omit", "passage", "if i removed", "what if"],
  },
];

export function findInvestigationQuestion(id: InvestigationQuestionId): InvestigationQuestion {
  const found = INVESTIGATION_QUESTIONS.find((question) => question.id === id);
  if (!found) throw new Error(`unknown investigation question id: ${id}`);
  return found;
}

/**
 * ROUTING ONLY. This function's entire contract is picking one of the seven fixed questions above, or
 * none -- it must never synthesize an explanation, a summary, or any text derived from `query` beyond
 * the caller echoing it back verbatim. See this module's own doc comment and C4's acceptance criterion:
 * "no fabricated explanations when measurements are unavailable."
 *
 * Scoring: one point per keyword that appears in the lowercased query as a plain substring -- these are
 * short English phrases, not tokens that need real word-boundary matching, and a false-positive
 * substring match only ever costs a slightly-too-eager route, never a fabricated answer. The question
 * with the most hits wins; a tie keeps whichever was declared first in `INVESTIGATION_QUESTIONS` (the
 * loop only replaces `best` on a STRICTLY greater score). Deterministic on purpose -- no
 * `Math.random`, no hash -- so identical input always routes identically, both for tests and for a user
 * re-reading their own history. Zero hits anywhere returns `null`.
 */
export function matchInvestigationIntent(query: string): InvestigationQuestionId | null {
  const needle = query.trim().toLowerCase();
  if (!needle) return null;
  let best: { id: InvestigationQuestionId; score: number } | null = null;
  for (const question of INVESTIGATION_QUESTIONS) {
    const score = question.keywords.reduce((count, keyword) => count + (needle.includes(keyword) ? 1 : 0), 0);
    if (score > 0 && (!best || score > best.score)) best = { id: question.id, score };
  }
  return best?.id ?? null;
}

export function capabilityLabel(capability: InvestigationCapability): string {
  switch (capability) {
    case "available": return "OPENS REAL EVIDENCE";
    case "partial": return "PARTIAL COVERAGE";
    case "unavailable": return "NOT BUILT YET";
    default: {
      const exhaustive: never = capability;
      return exhaustive;
    }
  }
}
