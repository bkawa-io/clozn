import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../test/fetch";
import { render, screen, waitFor, within } from "../../test/render";
import {
  buildInlineSegments,
  CLAIM_STATUS_ORDER,
  ClaimVerification,
  claimStatusMeta,
  toCodePoints,
  type Selection,
} from "./ClaimVerification";
import type { ClaimSupportStatus } from "../../data/claimSupport";

/**
 * E3 coverage. Every status/claim/support fixture below is HAND-BUILT JSON matching the wire shapes
 * `clozn/server/routes/claim_support.py` actually serves (clozn.answer-claims.v1 / clozn.claim-support.v1)
 * -- not run through the real Python builders, so offsets are computed locally by `buildFixture` from
 * plain ASCII claim texts (one code point per JS string index, so no BMP/surrogate subtlety here; that
 * case gets its own dedicated pure-function test below instead).
 */

function pathOf(request: PendingFetch) {
  return typeof request.input === "string"
    ? request.input
    : request.input instanceof URL
      ? request.input.pathname
      : request.input.url;
}

/** Matches by EXACT path, never a bare suffix -- across a run change, the array holds both generations'
 * requests together, and "/claim-support" alone would happily (and wrongly) match the OLD run's request
 * first. Every caller passes a full, run-scoped path for exactly this reason. */
function requestFor(requests: PendingFetch[], path: string) {
  const request = requests.find((item) => pathOf(item) === path);
  if (!request) throw new Error(`missing request for ${path}`);
  return request;
}

async function initialRequests(controller: ReturnType<typeof createFetchController>, runId: string) {
  const encoded = encodeURIComponent(runId);
  const paths = [`/runs/${encoded}/claim-support`, `/runs/${encoded}`, `/runs/${encoded}/corrective-actions`];
  await waitFor(() => {
    for (const path of paths) expect(controller.requests.some((item) => pathOf(item) === path)).toBe(true);
  });
  return {
    claimSupport: requestFor(controller.requests, paths[0]),
    run: requestFor(controller.requests, paths[1]),
    registry: requestFor(controller.requests, paths[2]),
  };
}

interface ClaimSpec {
  text: string;
  category: string;
  status: string;
  method: Record<string, unknown>;
  sourceSpanIds?: string[];
}

function buildFixture(runId: string, specs: ClaimSpec[]) {
  let cursor = 0;
  const parts: string[] = [];
  const claims: Record<string, unknown>[] = [];
  const results: Record<string, unknown>[] = [];
  specs.forEach((spec, index) => {
    if (index > 0) {
      parts.push(" ");
      cursor += 1;
    }
    const start = cursor;
    parts.push(spec.text);
    cursor += spec.text.length;
    const end = cursor;
    const addressId = `span_claim${String(index).padStart(3, "0")}`;
    claims.push({
      index,
      category: spec.category,
      category_reason: "factual_declarative",
      text_span: {
        address_id: addressId,
        run_id: runId,
        kind: "claim",
        relation_key: `rel_claim${index}`,
        native_ref: { artifact_schema: "clozn.answer-claims.v1", collection: "derived.claims", id: `claim-${index}` },
        resolution: {
          state: "metadata_only",
          canonical: {
            basis: "recorded_answer", unit: "unicode_code_points", interval: "half_open",
            start, end, basis_sha256: "a".repeat(64), span_sha256: "b".repeat(64),
          },
        },
      },
    });
    results.push({
      claim_index: index,
      claim_address_id: addressId,
      status: spec.status,
      method: spec.method,
      ...(spec.sourceSpanIds ? { source_span_ids: spec.sourceSpanIds } : {}),
    });
  });
  return { text: parts.join(""), claims, results };
}

const OFFSET_CONTRACT = {
  unit: "unicode_code_points", interval: "half_open", hash_algorithm: "sha256",
  canonicalization: "exact_string_utf8_v1",
};

function claimSupportBody(runId: string, fixture: ReturnType<typeof buildFixture>, gate = "ok") {
  return {
    claims: {
      schema_version: "clozn.answer-claims.v1",
      run_id: runId,
      privacy: "metadata_only",
      offset_contract: OFFSET_CONTRACT,
      segmentation: { state: "ok", claim_count: fixture.claims.length },
      answer_source: {
        basis: "recorded_answer", basis_sha256: "c".repeat(64),
        basis_code_points: fixture.text.length, basis_utf8_bytes: fixture.text.length,
      },
      claims: fixture.claims,
    },
    support: {
      schema_version: "clozn.claim-support.v1",
      run_id: runId,
      privacy: "metadata_only",
      offset_contract: OFFSET_CONTRACT,
      source: { claims_schema_version: "clozn.answer-claims.v1", influence_map: { gate } },
      results: fixture.results,
    },
  };
}

function runBody(runId: string, responseText: string) {
  return { id: runId, response: responseText };
}

function registryBody(runId: string, actionIds: string[]) {
  return {
    schema_version: "clozn.action-registry.v1",
    version: "1",
    run_id: runId,
    actions: actionIds.map((id) => ({
      id,
      label: id,
      description: `Corrective description for ${id}.`,
      conflicts: [],
      scopes: ["once"],
      eligibility: { eligible: true },
      evaluation_metrics: [],
      backends: [{ type: "prompt_policy", available: true }],
      scope_eligibility: [{ scope: "once", available: true, prior_hash: "hash-once" }],
    })),
  };
}

// Six specs, one per status. Claim 0 (supported) and claim 2 (contradicted) deliberately cite the SAME
// source id -- the fixture the cross-linking test needs.
// Deliberately long enough that `.slice(-10)` (the drawer's own button-label truncation) never collides
// with the leading "span_" prefix -- the exact label a test asserts against is computed from this same
// constant below, never hand-counted.
const SHARED_SOURCE_ID = "span_source0000000000001aaaa";
const OTHER_SOURCE_ID = "span_source0000000000002bbbb";

const SIX_STATUS_SPECS: ClaimSpec[] = [
  {
    text: "Paris is the capital of France.", category: "factual_claim", status: "supported",
    method: { name: "forced_score_intervention", max_abs_delta_nats: 1.2345 },
    sourceSpanIds: [SHARED_SOURCE_ID],
  },
  {
    text: "The bridge was likely finished around then.", category: "factual_claim", status: "weakly_supported",
    method: { name: "textual_overlap", overlap_fraction: 0.5 }, sourceSpanIds: [OTHER_SOURCE_ID],
  },
  {
    text: "The event happened in 1950.", category: "factual_claim", status: "contradicted",
    method: { name: "numeric_or_date_mismatch" }, sourceSpanIds: [SHARED_SOURCE_ID],
  },
  {
    text: "The moon is made of cheese.", category: "factual_claim", status: "unsupported_by_supplied_materials",
    method: { name: "measured_comparison_no_match" },
  },
  {
    text: "Water boils at exactly 100 degrees.", category: "factual_claim", status: "measurement_unavailable",
    method: { name: "no_influence_map" },
  },
  {
    text: "You should try the new approach.", category: "recommendation", status: "unverifiable_from_available_evidence",
    method: { name: "category_rule" },
  },
];

async function renderSixStatuses(runId = "run-six", registryActionIds = ["use-context"]) {
  const controller = createFetchController();
  vi.stubGlobal("fetch", controller.fetch);
  const fixture = buildFixture(runId, SIX_STATUS_SPECS);
  render(<ClaimVerification runId={runId} />);
  const requests = await initialRequests(controller, runId);
  controller.respondJson(requests.claimSupport, claimSupportBody(runId, fixture));
  controller.respondJson(requests.run, runBody(runId, fixture.text));
  controller.respondJson(requests.registry, registryBody(runId, registryActionIds));
  await screen.findByText("6 CLAIMS");
  return { controller, fixture };
}

describe("ClaimVerification pure functions", () => {
  test("claimStatusMeta never uses false/wrong/incorrect, and reads distinctly per status", () => {
    for (const status of CLAIM_STATUS_ORDER) {
      const meta = claimStatusMeta(status);
      const combined = `${meta.label} ${meta.description}`.toLowerCase();
      expect(combined).not.toMatch(/\bfalse\b/);
      expect(combined).not.toMatch(/\bwrong\b/);
      expect(combined).not.toMatch(/\bincorrect\b/);
    }
    const labels = new Set(CLAIM_STATUS_ORDER.map((status) => claimStatusMeta(status).label));
    expect(labels.size).toBe(6);
    const glyphs = new Set(CLAIM_STATUS_ORDER.map((status) => claimStatusMeta(status).glyph));
    expect(glyphs.size).toBe(6);

    expect(claimStatusMeta("unsupported_by_supplied_materials").label).toBe(
      "NOT SUPPORTED BY SUPPLIED MATERIALS",
    );
    expect(claimStatusMeta("measurement_unavailable").label).toBe("WE COULD NOT MEASURE THIS");
    // The two must never share wording -- "we could not measure" must never read as "not supported".
    expect(claimStatusMeta("measurement_unavailable").label.toLowerCase()).not.toContain("not supported");
    expect(claimStatusMeta("measurement_unavailable").description.toLowerCase()).not.toContain("not supported by");
  });

  test("toCodePoints/buildInlineSegments slice by Unicode code point, never by UTF-16 code unit", () => {
    // An emoji outside the BMP is TWO UTF-16 code units but ONE code point. A naive text.slice(0,1)
    // would cut it in half; toCodePoints + buildInlineSegments must not.
    const text = "🙂 hello";
    const points = toCodePoints(text);
    expect(points.length).toBe(text.length - 1); // one fewer than the raw UTF-16 length
    expect(points[0]).toBe("🙂");

    const row = {
      claim: {
        index: 0, category: "factual_claim" as const, categoryReason: "factual_declarative" as const,
        textSpan: {} as never,
      },
      result: undefined,
      start: 0,
      end: 1,
    };
    const segments = buildInlineSegments([row], points);
    expect(segments[0].kind).toBe("claim");
    expect(segments[0].text).toBe("🙂");
    expect(segments[1].kind).toBe("plain");
    expect(segments[1].text).toBe(" hello");
  });
});

describe("ClaimVerification panel", () => {
  test("all six statuses render distinctly, each with a visible glyph, never colour alone", async () => {
    await renderSixStatuses();

    // The legend spells out all six labels as literal text -- never color-only.
    const legend = screen.getByRole("list", { name: "Claim status filter" });
    for (const status of CLAIM_STATUS_ORDER) {
      expect(within(legend).getByText(claimStatusMeta(status).label)).toBeInTheDocument();
    }

    const marks = Array.from(document.querySelectorAll(".claim-mark"));
    expect(marks).toHaveLength(6);
    const markStatuses = new Set(marks.map((mark) => mark.getAttribute("data-claim-status")));
    expect(markStatuses.size).toBe(6);
    const markClasses = new Set(marks.map((mark) => mark.className));
    expect(markClasses.size).toBe(6); // six distinct CSS hooks, not just six colours

    // Every mark carries a VISIBLE glyph badge -- not a hover-only tooltip.
    for (const mark of marks) {
      const glyph = mark.querySelector(".claim-mark-glyph");
      expect(glyph).not.toBeNull();
      expect(glyph!.textContent).not.toBe("");
    }
    const glyphTexts = new Set(marks.map((mark) => mark.querySelector(".claim-mark-glyph")!.textContent));
    expect(glyphTexts.size).toBe(6);
  });

  test("never renders the banned false/wrong/incorrect vocabulary anywhere in the panel", async () => {
    const user = userEvent.setup();
    await renderSixStatuses();

    // Open the drawer for the contradicted claim too, so its contradiction-basis text is included in
    // the scan -- the surface most likely to accidentally reach for "wrong"/"incorrect".
    const contradictedRow = document.querySelector('[data-claim-status="contradicted"].claim-verify-row') as HTMLElement;
    await user.click(contradictedRow);
    await screen.findByRole("complementary", { name: "Source drawer" });

    const panel = document.querySelector(".claim-verify") as HTMLElement;
    const text = panel.textContent!.toLowerCase();
    expect(text).not.toMatch(/\bfalse\b/);
    expect(text).not.toMatch(/\bwrong\b/);
    expect(text).not.toMatch(/\bincorrect\b/);
    // The one phrase this whole feature exists to render, verbatim.
    expect(panel.textContent).toContain("NOT SUPPORTED BY SUPPLIED MATERIALS");
  });

  test("measurement_unavailable is visually and textually distinct from unsupported_by_supplied_materials", async () => {
    await renderSixStatuses();

    const muRow = document.querySelector('.claim-verify-row[data-claim-status="measurement_unavailable"]') as HTMLElement;
    const unsupRow = document.querySelector('.claim-verify-row[data-claim-status="unsupported_by_supplied_materials"]') as HTMLElement;
    expect(muRow).not.toBeNull();
    expect(unsupRow).not.toBeNull();
    expect(muRow.className).not.toBe(unsupRow.className);

    expect(muRow.textContent).toContain("WE COULD NOT MEASURE THIS");
    expect(muRow.textContent).not.toContain("NOT SUPPORTED");
    expect(unsupRow.textContent).toContain("NOT SUPPORTED BY SUPPLIED MATERIALS");
    expect(unsupRow.textContent).not.toContain("WE COULD NOT MEASURE");

    const muMark = document.querySelector('.claim-mark[data-claim-status="measurement_unavailable"]')!;
    const unsupMark = document.querySelector('.claim-mark[data-claim-status="unsupported_by_supplied_materials"]')!;
    expect(muMark.className).not.toBe(unsupMark.className);
  });

  test("selecting a claim shows its sources; selecting a source highlights every claim citing it", async () => {
    const user = userEvent.setup();
    await renderSixStatuses();

    // Claim 0 (supported) and claim 2 (contradicted) both cite SHARED_SOURCE_ID.
    const row0 = document.querySelector('.claim-verify-row[data-claim-index="0"]') as HTMLElement;
    await user.click(row0);

    const drawer = await screen.findByRole("complementary", { name: "Source drawer" });
    const sourceButton = within(drawer).getByRole("button", { name: SHARED_SOURCE_ID.slice(-10) });
    await user.click(sourceButton);

    // The drawer pivots to source mode, listing every claim citing this source.
    await within(drawer).findByText(/CLAIMS CITING THIS SOURCE/);
    const citingButtons = within(drawer).getAllByRole("button").filter((btn) =>
      btn.textContent?.includes("Paris") || btn.textContent?.includes("1950"));
    expect(citingButtons).toHaveLength(2);

    // Cross-highlighting reaches the main list too, for BOTH claims, not just the one that opened the drawer.
    const row0After = document.querySelector('.claim-verify-row[data-claim-index="0"]')!;
    const row2After = document.querySelector('.claim-verify-row[data-claim-index="2"]')!;
    const row1After = document.querySelector('.claim-verify-row[data-claim-index="1"]')!;
    expect(row0After.className).toMatch(/is-linked/);
    expect(row2After.className).toMatch(/is-linked/);
    expect(row1After.className).not.toMatch(/is-linked/);
  });

  test("a stale response for a previous run is never shown once the run changes", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);

    const view = render(<ClaimVerification runId="run-old" />);
    const oldRequests = await initialRequests(controller, "run-old");

    view.rerender(<ClaimVerification runId="run-new" />);
    await waitFor(() => expect(controller.requests.length).toBeGreaterThanOrEqual(6));
    expect(oldRequests.claimSupport.signal?.aborted).toBe(true);
    expect(oldRequests.run.signal?.aborted).toBe(true);
    expect(oldRequests.registry.signal?.aborted).toBe(true);

    const newRequests = await initialRequests(controller, "run-new");
    const newFixture = buildFixture("run-new", [SIX_STATUS_SPECS[0]]);
    controller.respondJson(newRequests.claimSupport, claimSupportBody("run-new", newFixture));
    controller.respondJson(newRequests.run, runBody("run-new", newFixture.text));
    controller.respondJson(newRequests.registry, registryBody("run-new", ["use-context"]));
    await screen.findByText("1 CLAIM");

    // The old run's (fully populated, distinctly-worded) response arrives late. It must never overwrite
    // the run the panel now shows.
    const oldFixture = buildFixture("run-old", SIX_STATUS_SPECS);
    controller.respondJson(oldRequests.claimSupport, claimSupportBody("run-old", oldFixture));
    controller.respondJson(oldRequests.run, runBody("run-old", oldFixture.text));
    controller.respondJson(oldRequests.registry, registryBody("run-old", ["use-context"]));
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(screen.getByText("1 CLAIM")).toBeInTheDocument();
    expect(screen.queryByText("6 CLAIMS")).not.toBeInTheDocument();
    // A claim text that exists ONLY in the old run's six-claim fixture, never the new one's single claim.
    expect(screen.queryByText(/moon is made of cheese/)).not.toBeInTheDocument();
  });

  test("retry button is disabled with a visible reason when no corrective action maps", async () => {
    // The registry exists and loaded fine, but does not include "use-context" -- the honest
    // disabled-with-reason state, not a silently missing button.
    await renderSixStatuses("run-no-mapping", ["less-verbose"]);

    const button = screen.getByRole("button", { name: "RETRY UNSUPPORTED CLAIMS" });
    expect(button).toBeDisabled();
    expect(screen.getByText(/no corrective action maps/i)).toBeInTheDocument();
  });

  test("retry button is disabled with a reason when there are no unsupported claims, even if the action exists", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const runId = "run-none-unsupported";
    const fixture = buildFixture(runId, [SIX_STATUS_SPECS[0]]); // supported only, no unsupported claim
    render(<ClaimVerification runId={runId} />);
    const requests = await initialRequests(controller, runId);
    controller.respondJson(requests.claimSupport, claimSupportBody(runId, fixture));
    controller.respondJson(requests.run, runBody(runId, fixture.text));
    controller.respondJson(requests.registry, registryBody(runId, ["use-context"]));

    const button = await screen.findByRole("button", { name: "RETRY UNSUPPORTED CLAIMS" });
    expect(button).toBeDisabled();
    expect(screen.getByText(/no unsupported claims/i)).toBeInTheDocument();
  });

  test("retry button is enabled and wired through the existing preview flow when a mapped action exists", async () => {
    const user = userEvent.setup();
    const { controller } = await renderSixStatuses("run-retry");
    const button = screen.getByRole("button", { name: "RETRY UNSUPPORTED CLAIMS" });
    expect(button).toBeEnabled();

    await user.click(button);
    const previewRequest = await waitFor(
      () => requestFor(controller.requests, "/runs/run-retry/corrective-actions/preview"),
    );
    expect(previewRequest.init?.method).toBe("POST");
    const body = JSON.parse(String(previewRequest.init?.body));
    expect(body.action_id).toBe("use-context");
  });
});

// Type-only sanity: Selection must exhaustively cover claim/source/null, exercised implicitly above --
// this keeps the exported union visible to a reader of this test file too.
const _selectionShapeCheck: Selection = null;
void _selectionShapeCheck;
void (undefined as unknown as ClaimSupportStatus);
