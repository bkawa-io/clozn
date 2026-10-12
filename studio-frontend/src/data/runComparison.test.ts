import { afterEach, describe, expect, it, vi } from "vitest";
import { studioApi } from "./client";
import { ContractError } from "./contracts";
import { decodeRunComparison } from "./runComparison";

export function comparisonDocument(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: "clozn.run-diff.v1",
    ok: true,
    run_a: "run_a",
    run_b: "run_b",
    comparison_selection: {
      mode: "explicit",
      reference_run_id: "run_a",
      candidate_run_id: "run_b",
      reason: "both run ids were supplied explicitly",
    },
    generated_at: "2026-10-09T12:00:00+00:00",
    privacy_limited: false,
    ranking: {
      order: ["identity.model_sha256", "generation", "context", "output"],
      note: "presentation-only sort key; never an evidence-strength claim",
    },
    summary_axes: {
      model: { status: "unchanged" },
      adapter: { status: "unavailable", note: "Adapter identity was not recorded." },
      template: { status: "unchanged" },
      context: { status: "changed" },
      sampling: { status: "unchanged" },
      engine: { status: "unchanged" },
      tool_parse: { status: "unavailable", note: "No output contract was recorded." },
      output: { status: "changed" },
    },
    differences: [{
      dimension: "context.delivered.messages.count",
      kind: "changed",
      value_a: 2,
      value_b: 1,
      rank: 30,
      evidence: [],
    }],
    findings: [{
      classification: "context_omission",
      status: "observed",
      summary: "One fewer message was delivered in run_b than in run_a.",
      dimensions: ["context.delivered.messages.count"],
    }],
    ...overrides,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("run comparison read contract", () => {
  it("decodes every rendered evidence state without upgrading its meaning", () => {
    const decoded = decodeRunComparison(comparisonDocument(), "/runs/compare", "run_a", "run_b");
    expect(decoded.axes).toHaveLength(8);
    expect(decoded.axes.find((axis) => axis.id === "adapter")).toEqual({
      id: "adapter", status: "unavailable", note: "Adapter identity was not recorded.",
    });
    expect(decoded.differences[0]).toEqual(expect.objectContaining({
      dimension: "context.delivered.messages.count", kind: "changed", valueA: 2, valueB: 1,
    }));
    expect(decoded.findings[0].status).toBe("observed");
  });

  it.each([
    ["extra top-level field", { invented: true }],
    ["unsupported finding status", { findings: [{ classification: "x", status: "proven", summary: "x", dimensions: [] }] }],
    ["unsupported difference kind", { differences: [{ dimension: "x", kind: "same", rank: 0, evidence: [] }] }],
    ["missing required evidence", { differences: [{ dimension: "x", kind: "changed", rank: 0, value_a: 1 }] }],
  ])("fails closed on %s", (_label, overrides) => {
    expect(() => decodeRunComparison(comparisonDocument(overrides))).toThrow(ContractError);
  });

  it("accepts evidence-only output changes emitted by the canonical differ", () => {
    const decoded = decodeRunComparison(comparisonDocument({
      differences: [{ dimension: "output.text", kind: "changed", rank: 40, evidence: [{ summary: { identical: false } }] }],
    }));
    expect(decoded.differences[0]).toEqual(expect.objectContaining({ dimension: "output.text", kind: "changed" }));
    expect(decoded.differences[0]).not.toHaveProperty("valueA");
    expect(decoded.differences[0]).not.toHaveProperty("valueB");
  });

  it("rejects stale or crossed response identities", () => {
    expect(() => decodeRunComparison(comparisonDocument(), "/runs/compare", "run_old", "run_b"))
      .toThrow(/run_a does not match the request/);
    expect(() => decodeRunComparison(comparisonDocument({
      comparison_selection: {
        mode: "explicit", reference_run_id: "run_b", candidate_run_id: "run_a", reason: "crossed",
      },
    }))).toThrow(/reference_run_id does not match run_a/);
  });

  it("calls the canonical GET route with encoded ids and decodes before returning", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify(comparisonDocument({
      run_a: "run / α",
      run_b: "run / β",
      comparison_selection: {
        mode: "explicit", reference_run_id: "run / α", candidate_run_id: "run / β", reason: "explicit",
      },
    })), { status: 200 }));
    vi.stubGlobal("fetch", fetch);

    const result = await studioApi.runComparison("run / α", "run / β");

    expect(result.runA).toBe("run / α");
    expect(fetch).toHaveBeenCalledWith(
      "/runs/compare?a=run+%2F+%CE%B1&b=run+%2F+%CE%B2",
      expect.objectContaining({ method: "GET", headers: { Accept: "application/json" } }),
    );
  });
});
