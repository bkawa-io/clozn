import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { InspectRoute } from "../../app/router";
import { studioApi } from "../../data/client";
import type {
  AnswerClaim,
  AnswerSegmentation,
  ClaimSupportEvidence,
  ClaimSupportResult,
} from "../../data/claimSupport";
import { ContractError } from "../../data/contracts";
import { ClaimSupportSurface } from "./ClaimSupport";
import { InspectSurface } from "./Inspect";

const runId = "run_claims";
const response = "🌕 Alpha. Beta 2020. Gamma. Delta. Maybe choose Epsilon. Zeta.";
const claimTexts = ["🌕 Alpha.", "Beta 2020.", "Gamma.", "Delta.", "Maybe choose Epsilon.", "Zeta."] as const;

function route(selectedRunId: string, sessionId?: string): InspectRoute {
  return {
    surface: "inspect",
    runId: selectedRunId,
    view: "claims",
    ...(sessionId ? { sessionId } : {}),
  };
}

function ranges(value: string, needles: readonly string[]): readonly [number, number][] {
  const points = Array.from(value);
  let cursor = 0;
  return needles.map((needle) => {
    const selected = Array.from(needle);
    const start = points.findIndex((_, index) => (
      index >= cursor && selected.every((point, offset) => points[index + offset] === point)
    ));
    cursor = start + selected.length;
    return [start, cursor];
  });
}

function claim(index: number, text: string, start: number, end: number): AnswerClaim {
  const recommendation = index === 4;
  return {
    index,
    category: recommendation ? "recommendation" : "factual_claim",
    categoryReason: recommendation ? "recommendation_marker" : "factual_declarative",
    textSpan: {
      addressId: `span_${String(index + 1).repeat(24)}`,
      runId,
      kind: "claim",
      relationKey: `relation-${index}`,
      nativeRef: {
        artifactSchema: "clozn.answer-claims.v1",
        collection: "derived.claims",
        id: `claim-${index}`,
      },
      resolution: {
        state: "metadata_only",
        canonical: {
          basis: "recorded_answer",
          start,
          end,
          basisSha256: "a".repeat(64),
          spanSha256: String(index + 1).repeat(64),
          basisCodePoints: Array.from(response).length,
          spanCodePoints: Array.from(text).length,
          basisUtf8Bytes: new TextEncoder().encode(response).byteLength,
          spanUtf8Bytes: new TextEncoder().encode(text).byteLength,
        },
      },
    },
  };
}

function fullEvidence(selectedRunId = runId): ClaimSupportEvidence {
  const claims = ranges(response, claimTexts).map(([start, end], index) => {
    const value = claim(index, claimTexts[index], start, end);
    return selectedRunId === runId
      ? value
      : { ...value, textSpan: { ...value.textSpan, runId: selectedRunId } };
  });
  const results: readonly ClaimSupportResult[] = [
    {
      claimIndex: 0,
      claimAddressId: claims[0].textSpan.addressId,
      status: "supported",
      method: { name: "forced_score_intervention", maxAbsDeltaNats: 0.625 },
      sourceSpanIds: ["span_aaaaaaaaaaaaaaaaaaaaaaaa"],
    },
    {
      claimIndex: 1,
      claimAddressId: claims[1].textSpan.addressId,
      status: "weakly_supported",
      method: { name: "textual_overlap", overlapFraction: 0.75 },
      sourceSpanIds: ["span_bbbbbbbbbbbbbbbbbbbbbbbb"],
    },
    {
      claimIndex: 2,
      claimAddressId: claims[2].textSpan.addressId,
      status: "contradicted",
      method: { name: "numeric_or_date_mismatch" },
      sourceSpanIds: ["span_cccccccccccccccccccccccc"],
    },
    {
      claimIndex: 3,
      claimAddressId: claims[3].textSpan.addressId,
      status: "unsupported_by_supplied_materials",
      method: { name: "measured_comparison_no_match" },
    },
    {
      claimIndex: 4,
      claimAddressId: claims[4].textSpan.addressId,
      status: "unverifiable_from_available_evidence",
      method: { name: "category_rule" },
    },
    {
      claimIndex: 5,
      claimAddressId: claims[5].textSpan.addressId,
      status: "measurement_unavailable",
      method: { name: "no_resolvable_answer_spans" },
    },
  ];
  return {
    claims: {
      schemaVersion: "clozn.answer-claims.v1",
      runId: selectedRunId,
      privacy: "metadata_only",
      segmentation: { state: "ok", claimCount: claims.length },
      answerSource: {
        basis: "recorded_answer",
        basisSha256: "a".repeat(64),
        basisCodePoints: Array.from(response).length,
        basisUtf8Bytes: new TextEncoder().encode(response).byteLength,
      },
      claims,
    },
    support: {
      schemaVersion: "clozn.claim-support.v1",
      runId: selectedRunId,
      privacy: "metadata_only",
      claimsSchemaVersion: "clozn.answer-claims.v1",
      influenceGate: "ok",
      results,
    },
    recordedResponse: response,
    claimTexts,
  };
}

function noClaimsEvidence(segmentation: AnswerSegmentation): ClaimSupportEvidence {
  const recordedResponse = segmentation.state === "unavailable" ? null : segmentation.state === "empty" ? "" : "漢字";
  return {
    claims: {
      schemaVersion: "clozn.answer-claims.v1",
      runId,
      privacy: "metadata_only",
      segmentation,
      answerSource: segmentation.state === "unavailable" ? { basis: "recorded_answer" } : {
        basis: "recorded_answer",
        basisSha256: "a".repeat(64),
        basisCodePoints: Array.from(recordedResponse ?? "").length,
        basisUtf8Bytes: new TextEncoder().encode(recordedResponse ?? "").byteLength,
      },
      claims: [],
    },
    support: {
      schemaVersion: "clozn.claim-support.v1",
      runId,
      privacy: "metadata_only",
      claimsSchemaVersion: "clozn.answer-claims.v1",
      influenceGate: "no_influence_map",
      results: [],
    },
    recordedResponse,
    claimTexts: [],
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => { resolve = resolvePromise; });
  return { promise, resolve };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Claim Support Inspect instrument", () => {
  it("renders all six honest statuses, Unicode-safe ranges, methods, filters, and stable source links", async () => {
    const read = vi.spyOn(studioApi, "claimSupportEvidence").mockResolvedValue(fullEvidence());
    const rawClaimRead = vi.spyOn(studioApi, "claimSupport");
    const rawRunRead = vi.spyOn(studioApi, "recordedRunResponse");
    const contextRead = vi.spyOn(studioApi, "runInvestigation");

    const { container } = render(<ClaimSupportSurface route={route(runId, "session_claims")} />);

    expect(await screen.findByRole("heading", { name: "Claim support" })).toBeInTheDocument();
    expect(screen.getByText(/unsupported is not the same as false/i)).toBeInTheDocument();
    expect(screen.getByText(/unavailable is not the same as unsupported/i)).toBeInTheDocument();
    expect(screen.getByText("Influence evidence available")).toBeInTheDocument();

    const marks = Array.from(container.querySelectorAll("mark.claim-answer-mark"));
    expect(marks).toHaveLength(6);
    expect(marks.map((mark) => mark.textContent)).toEqual(claimTexts);
    expect(marks[0]).toHaveTextContent("🌕 Alpha.");

    for (const label of [
      "Supported",
      "Weakly supported",
      "Contradicted",
      "Unsupported by supplied materials",
      "Unverifiable from available evidence",
      "Measurement unavailable",
    ]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    const filterBar = screen.getByRole("navigation", { name: "Filter claims by support status" });
    expect(Array.from(filterBar.querySelectorAll("i")).map((glyph) => glyph.textContent))
      .toEqual(["✓", "≈", "!", "∅", "?", "—"]);
    expect(screen.getByText(/does not mean the claim is false/i)).toBeInTheDocument();
    expect(screen.getByText(/this is not an unsupported finding/i)).toBeInTheDocument();
    expect(screen.getByText("Forced-score intervention · 0.625 max |Δ| nats")).toBeInTheDocument();
    expect(screen.getByText("Textual overlap · 75.0%")).toBeInTheDocument();
    expect(screen.getByText("Recommendation")).toBeInTheDocument();
    expect(screen.getByText("Recommendation marker")).toBeInTheDocument();

    const sourceLink = screen.getByRole("link", {
      name: "Stable source span span_aaaaaaaaaaaaaaaaaaaaaaaa",
    });
    expect(sourceLink).toHaveAttribute(
      "href",
      "/runs/run_claims/span-addresses#span_aaaaaaaaaaaaaaaaaaaaaaaa",
    );
    expect(screen.getByRole("link", { name: "Open Received Context" }))
      .toHaveAttribute("href", "#/inspect/run_claims?view=context&sessionId=session_claims");
    expect(screen.getByRole("link", { name: "Open corrective actions" }))
      .toHaveAttribute("href", "#/inspect/run_claims?view=actions&sessionId=session_claims");

    fireEvent.click(screen.getByRole("button", { name: /contradicted 1/i }));
    const results = container.querySelector<HTMLElement>(".claim-support-results");
    expect(results).not.toBeNull();
    expect(within(results!).getByText("Gamma.")).toBeInTheDocument();
    expect(within(results!).queryByText("Delta.")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /^supported 1$/i }));
    expect(within(results!).getByText("🌕 Alpha.")).toBeInTheDocument();

    expect(read).toHaveBeenCalledTimes(1);
    expect(read.mock.calls[0]?.[0]).toBe(runId);
    expect(read.mock.calls[0]?.[1]).toBeInstanceOf(AbortSignal);
    expect(rawClaimRead).not.toHaveBeenCalled();
    expect(rawRunRead).not.toHaveBeenCalled();
    expect(contextRead).not.toHaveBeenCalled();
  });

  it("wires the claims coordinate through the main Inspect surface", async () => {
    vi.spyOn(studioApi, "claimSupportEvidence").mockResolvedValue(fullEvidence());

    render(<InspectSurface route={route(runId)} />);

    expect(await screen.findByRole("heading", { name: "Claim support" })).toBeInTheDocument();
  });

  it.each([
    [{ state: "empty", reason: "answer_text_empty" } as const, "Recorded answer is empty", "exact empty answer"],
    [{ state: "segmentation_limited", reason: "unsupported_script_density" } as const, "Claim segmentation limited", "unsupported-script density"],
    [{ state: "unavailable", reason: "answer_text_redacted" } as const, "Claim segmentation unavailable", "recorded answer was redacted"],
    [{ state: "unavailable", reason: "no_answer_text" } as const, "Claim segmentation unavailable", "No recorded answer text"],
  ])("keeps the typed %s no-evidence state", async (segmentation, title, detail) => {
    vi.spyOn(studioApi, "claimSupportEvidence").mockResolvedValue(noClaimsEvidence(segmentation));

    render(<ClaimSupportSurface route={route(runId)} />);

    expect(await screen.findByText(title)).toBeInTheDocument();
    expect(screen.getByText(new RegExp(detail, "i"))).toBeInTheDocument();
    expect(screen.getByText(/factual claims are measurement unavailable, not unsupported/i)).toBeInTheDocument();
    expect(screen.queryByRole("navigation", { name: /filter claims/i })).not.toBeInTheDocument();
  });

  it("distinguishes unavailable transport from a malformed evidence contract", async () => {
    const read = vi.spyOn(studioApi, "claimSupportEvidence")
      .mockRejectedValueOnce(new Error("gateway offline"))
      .mockRejectedValueOnce(new ContractError("/runs/run_claims/claim-support", "$.support invalid"));

    const first = render(<ClaimSupportSurface route={route(runId)} />);
    expect(await screen.findByText("Claim evidence unavailable.")).toBeInTheDocument();
    expect(screen.getByText(/gateway offline/i)).toBeInTheDocument();
    first.unmount();

    render(<ClaimSupportSurface route={route(runId)} />);
    expect(await screen.findByText("Claim evidence response malformed.")).toBeInTheDocument();
    expect(screen.getByText(/support invalid/i)).toBeInTheDocument();
    expect(read).toHaveBeenCalledTimes(2);
  });

  it("aborts stale Run reads and never lets their evidence replace the current Run", async () => {
    const oldRead = deferred<ClaimSupportEvidence>();
    const fresh = fullEvidence("run_fresh");
    const read = vi.spyOn(studioApi, "claimSupportEvidence")
      .mockImplementationOnce(() => oldRead.promise)
      .mockResolvedValueOnce(fresh);

    const { rerender } = render(<ClaimSupportSurface route={route(runId)} />);
    const oldSignal = read.mock.calls[0]?.[1];
    rerender(<ClaimSupportSurface route={route("run_fresh")} />);

    expect(await screen.findByText("run_fresh")).toBeInTheDocument();
    expect(oldSignal?.aborted).toBe(true);
    oldRead.resolve(fullEvidence());
    await waitFor(() => expect(screen.queryByText(runId)).not.toBeInTheDocument());
    expect(read).toHaveBeenCalledTimes(2);
  });
});
