/**
 * Captures every Studio surface state for the static visual preview, then writes the assembled,
 * self-contained HTML file. Run via `node scripts/generate-preview.mjs` (which points vitest at
 * `vitest.preview.config.ts` so this file, deliberately outside `src/`, is the only thing that config
 * discovers -- see that config's own doc comment for why it is not folded into the main one).
 *
 * WHY THIS IS A VITEST FILE AND NOT A HAND-ROLLED JSDOM SCRIPT
 * -----------------------------------------------------------------
 * `scripts/smoke-render.mjs` proves the module graph resolves and every panel mounts once via
 * `react-dom/server`'s `renderToString` -- which never runs effects, so a fetching component only ever
 * shows its initial/loading branch that way. Every dense state this preview needs to show (the four-cell
 * WhatMattered grid, the five-status DiagnosisRepair findings, the PARTIAL TRACE banner) only exists
 * AFTER a fetch resolves and React re-renders, i.e. after an effect has actually run. `renderToString`
 * cannot get there. `@testing-library/react`'s `render()` (client rendering into real jsdom, the exact
 * mechanism every one of these components' own `*.test.tsx` files already use) can, so this file stubs
 * `fetch` "the way the tests do" -- literally reusing `src/test/render.tsx`, `src/test/fetch.ts`, and
 * `src/test/setup.ts` -- and captures the resulting real DOM once each state settles.
 *
 * WHAT THIS DOES NOT PROVE
 * ----------------------------
 * Every render below is driven by hand-built fixture JSON (see `fixtures.ts`), not a real backend. It
 * proves what these components render for THIS exact fixture shape; it says nothing about correctness
 * against real data, scale (a 4,000-turn session, a 300-row claim list), or any interaction beyond the
 * one scripted here per state (a click to expand a turn, open a drawer, or start a preview). See the
 * generated page's own banner, which states this in the artifact itself, not just here.
 *
 * SUBCOMPONENTS ARE NOT EXPORTED -- A FINDING, NOT WORKED AROUND
 * --------------------------------------------------------------------
 * The originating brief for this generator expected several presentational subcomponents (e.g.
 * ConversationInvestigation.tsx's `TurnRow`, `FindingCard`, `StatusCountsRow`) to be directly importable
 * so dense states could be built from props with no fetch at all. In the actual source, none of those
 * are `export`ed -- only each file's top-level, fetching component is. Adding `export` to make them
 * importable would be exactly the kind of "contort product code for a dev tool" trade the brief rules
 * out, so this file never does that; every non-trivial state below goes through the real top-level
 * component and a stubbed `fetch` instead, which reaches the identical rendered markup by construction
 * (it IS the component actually fetching, just against a fake network).
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import userEvent from "@testing-library/user-event";
import { afterAll, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../src/test/fetch";
import { render, screen, waitFor, within } from "../../src/test/render";

import { AskAnotherQuestion } from "../../src/features/lens/AskAnotherQuestion";
import { DiagnosisRepair } from "../../src/features/lens/DiagnosisRepair";
import { ClaimVerification } from "../../src/features/lens/ClaimVerification";
import { WhatMattered } from "../../src/features/lens/WhatMattered";
import { ConversationInvestigation } from "../../src/features/investigation/ConversationInvestigation";
import { SessionPicker } from "../../src/features/investigation/SessionPicker";
import { ForkOutcomePanel } from "../../src/features/observatory/ForkOutcomePanel";
import { App } from "../../src/app/App";

import * as fx from "./fixtures";
import { assemblePreviewHtml, type CapturedState, type FragmentWrap } from "./assemble";

// ------------------------------------------------------------------------------------------------- plumbing

const here = path.dirname(fileURLToPath(import.meta.url));
const studioFrontendRoot = path.resolve(here, "../..");
const repoRoot = path.resolve(studioFrontendRoot, "..");
const cssAssetsDir = path.join(repoRoot, "studio", "next", "assets");
const outDir = path.join(studioFrontendRoot, ".preview");
const outFile = path.join(outDir, "surfaces.html");

const cssFiles = fs.existsSync(cssAssetsDir)
  ? fs.readdirSync(cssAssetsDir).filter((f) => f.endsWith(".css")).sort()
  : [];
const cssText = cssFiles.map((f) => fs.readFileSync(path.join(cssAssetsDir, f), "utf-8")).join("\n");

const captured: CapturedState[] = [];

function record(input: {
  surfaceId: string;
  surfaceTitle: string;
  surfaceSource: string;
  stateId: string;
  stateTitle: string;
  note?: string;
  html: string;
  wrap: FragmentWrap;
  heightPx?: number;
}) {
  captured.push({
    heightPx: 560,
    note: "",
    ...input,
  });
}

function pathOf(request: PendingFetch): string {
  return typeof request.input === "string"
    ? request.input
    : request.input instanceof URL
      ? request.input.pathname + request.input.search
      : request.input.url;
}

function requestFor(requests: PendingFetch[], suffix: string): PendingFetch {
  const request = requests.find((item) => pathOf(item).endsWith(suffix));
  if (!request) throw new Error(`missing request ending "${suffix}" (have: ${requests.map(pathOf).join(", ") || "none"})`);
  return request;
}

async function waitForSuffix(controller: ReturnType<typeof createFetchController>, suffix: string) {
  return waitFor(() => requestFor(controller.requests, suffix));
}

// =================================================================================================================
// AskAnotherQuestion (C4) -- src/features/lens/AskAnotherQuestion.tsx
// Fires zero requests by design; every question's capability comes from static data
// (src/data/askAnotherQuestion.ts), so both states below stub fetch to THROW, matching the component's
// own test file's "must never fetch" discipline.
// =================================================================================================================

const ASK_ANOTHER_SOURCE = "src/features/lens/AskAnotherQuestion.tsx";

test("AskAnotherQuestion / directory", async () => {
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("AskAnotherQuestion must never fetch"); }));
  const view = render(<AskAnotherQuestion runId="run-demo-001" />);
  await screen.findByText("6 / 7 RUNNABLE");

  record({
    surfaceId: "ask-another-question",
    surfaceTitle: "Ask another question (C4)",
    surfaceSource: ASK_ANOTHER_SOURCE,
    stateId: "directory",
    stateTitle: "Directory — 6/7 runnable, 1 disabled with its own reason, 1 partial-coverage",
    note: "The \"Would another model disagree?\" question is disabled with a specific, visible reason "
      + "(never hidden). \"What happens without this passage?\" is PARTIAL COVERAGE and names its own "
      + "limit plus a second real surface. Both are the C4 honesty requirements this panel exists for.",
    html: view.container.innerHTML,
    wrap: "instrument",
    heightPx: 640,
  });
  view.unmount();
});

test("AskAnotherQuestion / with recorded history", async () => {
  vi.stubGlobal("fetch", vi.fn(() => { throw new Error("AskAnotherQuestion must never fetch"); }));
  const user = userEvent.setup();
  const view = render(<AskAnotherQuestion runId="run-demo-001" />);
  await screen.findByText("6 / 7 RUNNABLE");
  await user.click(screen.getByRole("button", { name: /^Why\?/ }));
  const historyRegion = await screen.findByRole("region", { name: "Past investigations for this run" });
  await within(historyRegion).findByText("Why?");

  record({
    surfaceId: "ask-another-question",
    surfaceTitle: "Ask another question (C4)",
    surfaceSource: ASK_ANOTHER_SOURCE,
    stateId: "history",
    stateTitle: "After asking one question — the dated, structured investigation history log",
    note: "A real click on the \"Why?\" chip, recorded via the component's own localStorage-backed "
      + "history (data/investigationHistory.ts). Not a chat transcript on purpose — see the file's own "
      + "doc comment on staying visually distinct from a chat turn.",
    html: view.container.innerHTML,
    wrap: "instrument",
    heightPx: 700,
  });
  view.unmount();
});

// =================================================================================================================
// DiagnosisRepair (D5) -- src/features/lens/DiagnosisRepair.tsx
// =================================================================================================================

const DIAGNOSIS_REPAIR_SOURCE = "src/features/lens/DiagnosisRepair.tsx";

test("DiagnosisRepair / five-status findings", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const runId = "run-demo-diagnosis";
  const view = render(<DiagnosisRepair runId={runId} />);
  const findingsReq = await waitForSuffix(controller, "/diagnosis-findings");
  const registryReq = await waitForSuffix(controller, "/corrective-actions");
  controller.respondJson(findingsReq, fx.diagnosisFindingsBody(runId));
  controller.respondJson(registryReq, fx.correctiveRegistryBody(runId));
  await screen.findAllByText("FINDING");

  record({
    surfaceId: "diagnosis-repair",
    surfaceTitle: "Diagnosis & repair (D5)",
    surfaceSource: DIAGNOSIS_REPAIR_SOURCE,
    stateId: "five-status",
    stateTitle: "All five finding statuses distinct: finding / not_observed / unavailable / pending / suppressed",
    note: "D1's five-value evidence vocabulary, each with its own visible label and CSS hook — never "
      + "collapsed into a binary pass/fail. Registers (measured effects / observed changes / "
      + "plausible-unproven) are empty here (no comparison run supplied), which is itself an honest, "
      + "explicitly-labelled state, not a blank space.",
    html: view.container.innerHTML,
    wrap: "instrument",
    heightPx: 1050,
  });
  view.unmount();
});

test("DiagnosisRepair / corrective retry preview", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const user = userEvent.setup();
  const runId = "run-demo-diagnosis-preview";
  const view = render(<DiagnosisRepair runId={runId} />);
  const findingsReq = await waitForSuffix(controller, "/diagnosis-findings");
  const registryReq = await waitForSuffix(controller, "/corrective-actions");
  controller.respondJson(findingsReq, fx.diagnosisFindingsBody(runId));
  controller.respondJson(registryReq, fx.correctiveRegistryBody(runId));
  const actionButton = await screen.findByRole("button", { name: /More concise/ });
  await user.click(actionButton);
  const previewReq = await waitForSuffix(controller, "/corrective-actions/preview");
  controller.respondJson(previewReq, fx.previewBody(), { status: 201 });
  await screen.findByText("WILL INJECT");

  record({
    surfaceId: "diagnosis-repair",
    surfaceTitle: "Diagnosis & repair (D5)",
    surfaceSource: DIAGNOSIS_REPAIR_SOURCE,
    stateId: "retry-preview",
    stateTitle: "Corrective retry: preview shown before execution (D3 preview → confirm mechanics)",
    note: "Preview only — CONFIRM has not been clicked, nothing has executed. Demonstrates the "
      + "preview/confirm gate the panel shares with Behavior.tsx's \"Fix this answer\" flow.",
    html: view.container.innerHTML,
    wrap: "instrument",
    heightPx: 1050,
  });
  view.unmount();
});

// =================================================================================================================
// WhatMattered -- src/features/lens/WhatMattered.tsx
// =================================================================================================================

const WHAT_MATTERED_SOURCE = "src/features/lens/WhatMattered.tsx";

test("WhatMattered / four cell states", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const runId = "run-demo-mattered";
  const view = render(<WhatMattered runId={runId} />);
  const invReq = await waitForSuffix(controller, "/investigation");
  const spanReq = await waitForSuffix(controller, "/span-addresses");
  controller.respondJson(invReq, fx.investigation(runId));
  controller.respondJson(spanReq, fx.spanDocument(runId));
  await screen.findByText("F+");

  record({
    surfaceId: "what-mattered",
    surfaceTitle: "What mattered?",
    surfaceSource: WHAT_MATTERED_SOURCE,
    stateId: "four-cell-grid",
    stateTitle: "Cross-linked grid — all four cell states: measured / below floor / omitted / not measured",
    note: "The one property this feature exists for: a cold cell is never a single collapsed color — "
      + "\"never reached the model\", \"reached it but never scored\", and \"scored, cleared nothing\" "
      + "render as three distinct, labelled states, never merged.",
    html: view.container.innerHTML,
    wrap: "instrument",
    heightPx: 620,
  });
  view.unmount();
});

test("WhatMattered / not yet measured", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const runId = "run-demo-mattered-idle";
  const view = render(<WhatMattered runId={runId} />);
  const invReq = await waitForSuffix(controller, "/investigation");
  const spanReq = await waitForSuffix(controller, "/span-addresses");
  controller.respondJson(invReq, fx.unmeasuredInvestigation(runId, "ready"));
  controller.respondJson(spanReq, fx.spanDocument(runId));
  await screen.findByText("NOT MEASURED");

  record({
    surfaceId: "what-mattered",
    surfaceTitle: "What mattered?",
    surfaceSource: WHAT_MATTERED_SOURCE,
    stateId: "not-measured",
    stateTitle: "Not yet measured — action available, nothing has run automatically",
    note: "Viewing this panel never starts a measurement on its own; the button is enabled and idle.",
    html: view.container.innerHTML,
    wrap: "instrument",
    heightPx: 420,
  });
  view.unmount();
});

test("WhatMattered / measurement unavailable with reason", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const runId = "run-demo-mattered-unavailable";
  const view = render(<WhatMattered runId={runId} />);
  const invReq = await waitForSuffix(controller, "/investigation");
  const spanReq = await waitForSuffix(controller, "/span-addresses");
  controller.respondJson(
    invReq,
    fx.unmeasuredInvestigation(runId, "unavailable", "the active worker does not expose token scoring"),
  );
  controller.respondJson(spanReq, fx.spanDocument(runId));
  await screen.findByText(/does not expose token scoring/);

  record({
    surfaceId: "what-mattered",
    surfaceTitle: "What mattered?",
    surfaceSource: WHAT_MATTERED_SOURCE,
    stateId: "unavailable",
    stateTitle: "Measurement unavailable — disabled with its own specific reason",
    note: "The MEASURE button is disabled; the reason names the real boundary (worker capability), "
      + "never a generic \"unavailable\" with no explanation.",
    html: view.container.innerHTML,
    wrap: "instrument",
    heightPx: 420,
  });
  view.unmount();
});

// =================================================================================================================
// ClaimVerification (E3) -- src/features/lens/ClaimVerification.tsx
// =================================================================================================================

const CLAIM_VERIFY_SOURCE = "src/features/lens/ClaimVerification.tsx";

test("ClaimVerification / six statuses + source drawer", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const user = userEvent.setup();
  const runId = "run-demo-claims";
  const fixture = fx.buildClaimFixture(runId, fx.SIX_STATUS_CLAIM_SPECS);
  const view = render(<ClaimVerification runId={runId} />);
  const claimSupportReq = await waitForSuffix(controller, "/claim-support");
  const runReq = await waitForSuffix(controller, `/runs/${runId}`);
  const registryReq = await waitForSuffix(controller, "/corrective-actions");
  controller.respondJson(claimSupportReq, fx.claimSupportBody(runId, fixture));
  controller.respondJson(runReq, fx.claimRunBody(runId, fixture.text));
  controller.respondJson(registryReq, fx.claimRegistryBody(runId, ["use-context"]));
  await screen.findByText("6 CLAIMS");
  const contradictedRow = document.querySelector(
    '[data-claim-status="contradicted"].claim-verify-row',
  ) as HTMLElement;
  await user.click(contradictedRow);
  await screen.findByRole("complementary", { name: "Source drawer" });

  record({
    surfaceId: "claim-verification",
    surfaceTitle: "Are the claims supported? (E3)",
    surfaceSource: CLAIM_VERIFY_SOURCE,
    stateId: "six-statuses",
    stateTitle: "All six claim statuses, inline markers + list, source drawer open on a contradicted claim",
    note: "Every status renders with its own glyph and label — the panel's language discipline "
      + "(\"not supported by supplied materials\", never \"false\"/\"wrong\"/\"incorrect\") is a "
      + "structural property tested against this exact rendered text, not just a copy convention.",
    html: view.container.innerHTML,
    wrap: "instrument",
    heightPx: 1150,
  });
  view.unmount();
});

test("ClaimVerification / no claims extracted", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const runId = "run-demo-claims-empty";
  const fixture = fx.buildClaimFixture(runId, []);
  const view = render(<ClaimVerification runId={runId} />);
  const claimSupportReq = await waitForSuffix(controller, "/claim-support");
  const runReq = await waitForSuffix(controller, `/runs/${runId}`);
  const registryReq = await waitForSuffix(controller, "/corrective-actions");
  controller.respondJson(claimSupportReq, fx.claimSupportBody(runId, fixture));
  controller.respondJson(runReq, fx.claimRunBody(runId, fixture.text || " "));
  controller.respondJson(registryReq, fx.claimRegistryBody(runId, []));
  await screen.findByText("No claims were extracted from this answer.");

  record({
    surfaceId: "claim-verification",
    surfaceTitle: "Are the claims supported? (E3)",
    surfaceSource: CLAIM_VERIFY_SOURCE,
    stateId: "no-claims",
    stateTitle: "No claims extracted from this answer — honest empty state",
    html: view.container.innerHTML,
    wrap: "instrument",
    heightPx: 420,
  });
  view.unmount();
});

// =================================================================================================================
// ConversationInvestigation (F3) -- src/features/investigation/ConversationInvestigation.tsx
// =================================================================================================================

const CONVERSATION_INVESTIGATION_SOURCE = "src/features/investigation/ConversationInvestigation.tsx";

test("ConversationInvestigation / partial trace, a finding, a branch", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const user = userEvent.setup();
  const sessionId = "session-demo-1";
  const view = render(<ConversationInvestigation sessionId={sessionId} />);
  const req = await waitForSuffix(controller, "/trace?limit=50");
  controller.respondJson(req, fx.pageFixture(sessionId, [
    fx.turnFixture("run-a1", {
      diagnostic_highlights: {
        findings: [{
          rule_id: "R01", rule_name: "input_omitted_or_rejected", status: "finding",
          severity: "medium", confidence: "exact", summary: "1 input segment was omitted.",
          evidence: [], limitations: [],
        }],
        status_counts: { finding: 1, not_observed: 9, unavailable: 0, pending: 2, suppressed: 0 },
      },
    }),
    fx.turnFixture("run-a2"),
  ], {
    page: { cursor: null, next_cursor: "CURSOR_NEXT", limit: 50, count: 2 },
    branches: [{
      parent_run_id: "run-a1",
      children: [{ id: "branch-child-1", source: "fork", prompt_summary: "retry with a shorter prompt" }],
    }],
    first_went_wrong_candidates: [
      {
        kind: "first_finding", run_id: "run-a1", recorded_ts: 1700000000,
        summary: "diagnostic rule(s) reported a finding on this turn: R01", rule_ids: ["R01"],
      },
    ],
  }));
  await screen.findByText("prompt for run-a1");
  expect(screen.getByText("PARTIAL TRACE")).toBeInTheDocument();
  const toggle = screen.getAllByRole("button", { expanded: false })[0];
  await user.click(toggle);
  await screen.findByText("input omitted or rejected");

  record({
    surfaceId: "conversation-investigation",
    surfaceTitle: "Conversation investigation (F3)",
    surfaceSource: CONVERSATION_INVESTIGATION_SOURCE,
    stateId: "partial-finding-branch",
    stateTitle: "PARTIAL TRACE banner + an expanded turn with a finding + a branch, first suspicious turn found",
    note: "next_cursor is non-null, so the timeline states plainly it is PARTIAL, never silently "
      + "presented as the whole session. The candidate aside shows a real backend-supplied "
      + "\"first finding\" candidate.",
    html: view.container.innerHTML,
    wrap: "bare",
    heightPx: 1500,
  });
  view.unmount();
});

test("ConversationInvestigation / no suspicious turn found, all-zero status row", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const user = userEvent.setup();
  const sessionId = "session-demo-2";
  const view = render(<ConversationInvestigation sessionId={sessionId} />);
  const req = await waitForSuffix(controller, "/trace?limit=50");
  controller.respondJson(req, fx.pageFixture(sessionId, [
    fx.turnFixture("run-b1", {
      diagnostic_highlights: {
        findings: [],
        status_counts: { finding: 0, not_observed: 0, unavailable: 0, pending: 0, suppressed: 0 },
      },
    }),
  ]));
  // Default page fixture: next_cursor null (complete trace) and first_went_wrong_candidates: [].
  await screen.findByText("prompt for run-b1");
  expect(screen.getByText(/this is the full session/)).toBeInTheDocument();
  const toggle = screen.getAllByRole("button", { expanded: false })[0];
  await user.click(toggle);
  await screen.findByText("0 FINDING");

  record({
    surfaceId: "conversation-investigation",
    surfaceTitle: "Conversation investigation (F3)",
    surfaceSource: CONVERSATION_INVESTIGATION_SOURCE,
    stateId: "no-candidate-all-zero",
    stateTitle: "No first-suspicious-turn candidate of any kind + D1's five-status row at all-zero",
    note: "All three candidate kinds (first finding / first settings drift / first failed run) say "
      + "plainly that none was found — never a blank space, never a fabricated arrow. The expanded "
      + "turn's status-count row shows all five D1 values at zero, still individually labelled.",
    html: view.container.innerHTML,
    wrap: "bare",
    heightPx: 1500,
  });
  view.unmount();
});

test("ConversationInvestigation / session not found", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const sessionId = "ghost-session";
  const view = render(<ConversationInvestigation sessionId={sessionId} />);
  const req = await waitForSuffix(controller, "/trace?limit=50");
  controller.respondJson(req, { error: "session not found" }, { status: 404 });
  await screen.findByText(/was not found/);

  record({
    surfaceId: "conversation-investigation",
    surfaceTitle: "Conversation investigation (F3)",
    surfaceSource: CONVERSATION_INVESTIGATION_SOURCE,
    stateId: "not-found",
    stateTitle: "Session genuinely does not exist — a specific 404, not a generic failure sentence",
    html: view.container.innerHTML,
    wrap: "bare",
    heightPx: 320,
  });
  view.unmount();
});

// =================================================================================================================
// SessionPicker (F3 entry point) -- src/features/investigation/SessionPicker.tsx
// =================================================================================================================

const SESSION_PICKER_SOURCE = "src/features/investigation/SessionPicker.tsx";

test("SessionPicker / populated", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const view = render(<SessionPicker />);
  const req = await controller.nextRequest();
  controller.respondJson(req, {
    sessions: [
      fx.sessionFixture("session_abc", { title: "Debugging the retry loop", run_count: 6, last_activity_ts: 1700000900 }),
      fx.sessionFixture("session_def", { title: "Long-context summarization test", run_count: 21, last_activity_ts: 1700050000 }),
      fx.sessionFixture("session_hidden", {
        title: "Internal QA sweep", run_count: 3, last_activity_ts: 1700020000,
        privacy: { visibility: "hidden" },
      }),
    ],
  });
  await screen.findByText("Debugging the retry loop");

  record({
    surfaceId: "session-picker",
    surfaceTitle: "Conversation sessions (F3 entry point)",
    surfaceSource: SESSION_PICKER_SOURCE,
    stateId: "populated",
    stateTitle: "Populated session list, including a hidden-visibility session flagged as such",
    html: view.container.innerHTML,
    wrap: "bare",
    heightPx: 520,
  });
  view.unmount();
});

test("SessionPicker / empty install", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const view = render(<SessionPicker />);
  const req = await controller.nextRequest();
  controller.respondJson(req, { sessions: [] });
  await screen.findByText("No sessions recorded yet.");

  record({
    surfaceId: "session-picker",
    surfaceTitle: "Conversation sessions (F3 entry point)",
    surfaceSource: SESSION_PICKER_SOURCE,
    stateId: "empty",
    stateTitle: "No sessions recorded yet — an honest empty state, not a blank list",
    html: view.container.innerHTML,
    wrap: "bare",
    heightPx: 320,
  });
  view.unmount();
});

test("SessionPicker / failed request", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const view = render(<SessionPicker />);
  const req = await controller.nextRequest();
  controller.respondJson(req, { error: "list_sessions timed out after 5s" }, { status: 500 });
  await screen.findByText(/timed out/);

  record({
    surfaceId: "session-picker",
    surfaceTitle: "Conversation sessions (F3 entry point)",
    surfaceSource: SESSION_PICKER_SOURCE,
    stateId: "failed",
    stateTitle: "The session list request failed — reported, never silently swallowed",
    html: view.container.innerHTML,
    wrap: "bare",
    heightPx: 320,
  });
  view.unmount();
});

// =================================================================================================================
// ForkOutcomePanel -- src/features/observatory/ForkOutcomePanel.tsx
// Pure, prop-driven, never fetches -- these three states are copied directly from its own test file.
// =================================================================================================================

const FORK_OUTCOME_SOURCE = "src/features/observatory/ForkOutcomePanel.tsx";

test("ForkOutcomePanel / exact execution fork", async () => {
  const view = render(<ForkOutcomePanel {...fx.FORK_OUTCOME_EXACT} />);
  await screen.findByText("EXACT EXECUTION FORK");

  record({
    surfaceId: "fork-outcome-panel",
    surfaceTitle: "Fork outcome panel",
    surfaceSource: FORK_OUTCOME_SOURCE,
    stateId: "exact",
    stateTitle: "Exact execution fork — the strong result, full exactness facts",
    html: view.container.innerHTML,
    wrap: "bare",
    heightPx: 420,
  });
  view.unmount();
});

test("ForkOutcomePanel / reconstructed replay", async () => {
  const view = render(<ForkOutcomePanel {...fx.FORK_OUTCOME_RECONSTRUCTED} />);
  await screen.findByText("RECONSTRUCTED REPLAY");

  record({
    surfaceId: "fork-outcome-panel",
    surfaceTitle: "Fork outcome panel",
    surfaceSource: FORK_OUTCOME_SOURCE,
    stateId: "reconstructed",
    stateTitle: "Reconstructed replay — visibly weaker, names the retokenization risk outright",
    html: view.container.innerHTML,
    wrap: "bare",
    heightPx: 420,
  });
  view.unmount();
});

test("ForkOutcomePanel / unavailable", async () => {
  const view = render(<ForkOutcomePanel {...fx.FORK_OUTCOME_UNAVAILABLE} />);
  await screen.findByText("FORK UNAVAILABLE");

  record({
    surfaceId: "fork-outcome-panel",
    surfaceTitle: "Fork outcome panel",
    surfaceSource: FORK_OUTCOME_SOURCE,
    stateId: "unavailable",
    stateTitle: "Unavailable — the gateway's own typed reason, never a generic failure sentence",
    html: view.container.innerHTML,
    wrap: "bare",
    heightPx: 320,
  });
  view.unmount();
});

// =================================================================================================================
// The workbench shell -- src/app/App.tsx
// Chrome only: rail nav, topbar (runtime badge, theme toggle), workspace grid. NOT a full panel-content
// preview -- see the note recorded alongside this state and the generator's own report for why.
// =================================================================================================================

test("App shell / rail nav + topbar chrome", async () => {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  window.localStorage.clear();
  location.hash = "";
  const view = render(<App />);
  const healthReq = await waitForSuffix(controller, "/healthz");
  const runsReq = await waitForSuffix(controller, "/runs");
  const engineReq = await waitForSuffix(controller, "/engine/health");
  controller.respondJson(healthReq, fx.healthzBody());
  controller.respondJson(runsReq, fx.runtimeRunsBody());
  controller.respondJson(engineReq, fx.engineHealthBody());
  await screen.findByText("CONNECTED");

  record({
    surfaceId: "workbench-shell",
    surfaceTitle: "The workbench shell (rail nav / topbar / theme)",
    surfaceSource: "src/app/App.tsx",
    stateId: "chrome",
    stateTitle: "Chrome only — runtime CONNECTED, 2 fixture runs, Runs panel routed but its own body not fetched",
    note: "This state exists to check the shell itself (nav rail, brand lockup, runtime badge, theme "
      + "toggle, topbar layout) — not any one panel's evidence. The Runs panel underneath is left in its "
      + "native loading look; every other panel (Lens, Scope, Behavior, Model, Experiments, Compare) is "
      + "pre-existing multi-week surface area, not part of the seven surfaces this generator targets, "
      + "and is not captured here at all.",
    html: view.container.innerHTML,
    wrap: "shell",
    heightPx: 760,
  });
  view.unmount();
});

// ------------------------------------------------------------------------------------------------- assembly

const SKIPPED_NOTE = [
  "SCOPE OF THIS GENERATOR (from scripts/preview/capture-surfaces.spec.tsx):",
  "",
  "Covered: AskAnotherQuestion, DiagnosisRepair, WhatMattered, ClaimVerification,",
  "ConversationInvestigation, SessionPicker, and ForkOutcomePanel -- the seven surfaces this generator",
  "was built for -- plus one chrome-only state of the App shell (rail nav / topbar / theme).",
  "",
  "Deliberately NOT covered:",
  "  - Lens.tsx, Observatory.tsx/Scope, Behavior.tsx, Model.tsx, Experiments.tsx, Compare.tsx -- large,",
  "    pre-existing multi-week surfaces, not part of the recently-shipped batch this tool targets.",
  "  - Any panel's own fetched body content inside the App-shell state above (only the shell chrome",
  "    itself was the target there).",
  "  - DiagnosisRepair's post-CONFIRM \"kept\"/\"undone\" result state, and WhatMattered's live",
  "    \"measuring…\" progress banner -- both reachable, but each needs another endpoint stubbed",
  "    (confirm/keep, and the influence-map job poll) beyond what this pass covers.",
  "  - Any component subtree gated behind a real, non-fixture backend call this generator has no",
  "    fixture for.",
].join("\n");

afterAll(() => {
  fs.mkdirSync(outDir, { recursive: true });
  const html = assemblePreviewHtml({
    states: captured,
    cssText,
    cssFiles,
    generatedAtIso: new Date().toISOString(),
    skippedNote: cssFiles.length
      ? SKIPPED_NOTE
      : `WARNING: no compiled CSS was found under ${cssAssetsDir} -- run "npm run build" first. `
        + `This page's surfaces will render UNSTYLED.\n\n${SKIPPED_NOTE}`,
  });
  fs.writeFileSync(outFile, html, "utf-8");
  const kb = (Buffer.byteLength(html, "utf-8") / 1024).toFixed(1);
  // eslint-disable-next-line no-console
  console.log(`[preview] wrote ${outFile} (${kb} KB), ${captured.length} states across `
    + `${new Set(captured.map((s) => s.surfaceId)).size} surfaces.`);
});
