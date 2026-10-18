import { afterEach, describe, expect, it, vi } from "vitest";
import { studioApi } from "./client";
import { ContractError } from "./contracts";
import { decodeRunComparison } from "./runComparison";

export function comparisonDocument(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: "clozn.reduction-comparison.v1",
    ok: true,
    parent_run_id: "run_parent",
    child_run_id: "run_child",
    context: {
      parent_source_ids: ["src_a", "src_b", "src_c"],
      retained_source_ids: ["src_a"],
      removed_source_ids: ["src_b", "src_c"],
    },
    rendered_prompt_tokens: { parent: 20, child: 11, saved: 9, improved: true },
    preservation: { answer_exact_match: true, finish_reason_match: true, exact: true },
    generation_identity: {
      parent: { model_sha256: "a".repeat(64), engine_build: "engine-v1" },
      child: { model_sha256: "a".repeat(64), engine_build: "engine-v1" },
      match: true,
    },
    ...overrides,
  };
}

afterEach(() => vi.unstubAllGlobals());

describe("saved reduction comparison contract", () => {
  it("decodes the narrow parent/child evidence", () => {
    const decoded = decodeRunComparison(
      comparisonDocument(), "/runs/compare", "run_parent", "run_child",
    );

    expect(decoded.context.removedSourceIds).toEqual(["src_b", "src_c"]);
    expect(decoded.renderedPromptTokens).toEqual({ parent: 20, child: 11, saved: 9, improved: true });
    expect(decoded.preservation).toEqual({
      answerExactMatch: true, finishReasonMatch: true, exact: true,
    });
    expect(decoded.generationIdentity.match).toBe(true);
  });

  it.each([
    ["wrong schema", { schema_version: "clozn.run-diff.v1" }],
    ["missing context", { context: undefined }],
    ["invalid token count", { rendered_prompt_tokens: { parent: 20, child: 11, saved: 9.5, improved: true } }],
    ["invalid preservation", { preservation: { answer_exact_match: true, finish_reason_match: true, exact: "yes" } }],
  ])("fails closed on %s", (_label, overrides) => {
    expect(() => decodeRunComparison(comparisonDocument(overrides))).toThrow(ContractError);
  });

  it("rejects stale or crossed response identities", () => {
    expect(() => decodeRunComparison(
      comparisonDocument(), "/runs/compare", "run_old", "run_child",
    )).toThrow(/parent_run_id does not match the request/);
    expect(() => decodeRunComparison(
      comparisonDocument(), "/runs/compare", "run_parent", "run_old",
    )).toThrow(/child_run_id does not match the request/);
  });

  it("calls the canonical GET route with encoded ids and decodes before returning", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify(comparisonDocument({
      parent_run_id: "run / α",
      child_run_id: "run / β",
    })), { status: 200 }));
    vi.stubGlobal("fetch", fetch);

    const result = await studioApi.runComparison("run / α", "run / β");

    expect(result.parentRunId).toBe("run / α");
    expect(fetch).toHaveBeenCalledWith(
      "/runs/compare?a=run+%2F+%CE%B1&b=run+%2F+%CE%B2",
      expect.objectContaining({ method: "GET", headers: { Accept: "application/json" } }),
    );
  });
});
