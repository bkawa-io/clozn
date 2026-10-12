import { afterEach, describe, expect, it, vi } from "vitest";
import fullFixture from "../../../tests/fixtures/schemas/clozn.context-receipt.v1/valid__full.json";
import minimalFixture from "../../../tests/fixtures/schemas/clozn.context-receipt.v1/valid__minimal.json";
import { studioApi } from "./client";
import { ContractError } from "./contracts";
import { decodeContextReceipt } from "./contextReceipt";

function recordedEnvelope(contextReceipt: unknown, runId = fullFixture.run_id) {
  return { run_id: runId, shape: "recorded", context_receipt: contextReceipt };
}

describe("Context Receipt read contract", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("decodes the real full v1 fixture and its controlled evidence vocabulary", () => {
    const view = decodeContextReceipt(recordedEnvelope({
      ...fullFixture,
      rendered: { ...fullFixture.rendered, tokens: 61 },
    }), "/receipt", fullFixture.run_id);
    expect(view.shape).toBe("recorded");
    if (view.shape !== "recorded" || view.receipt.kind !== "v1") throw new Error("expected a v1 receipt");

    expect(view.receipt).toMatchObject({
      schemaVersion: "clozn.context-receipt.v1",
      runId: fullFixture.run_id,
      privacy: "full",
      termination: { reason: "max_tokens", reasonRaw: "length", generatedTokens: 512 },
      rendered: { tokens: 61, estimated: false },
    });
    expect(view.receipt.delivered).toHaveLength(2);
    expect(view.receipt.transformations?.[0]).toMatchObject({
      reason: "template_transformed",
      segmentIds: fullFixture.transformations[0].segment_ids,
    });
  });

  it("preserves missing versus explicitly empty evidence in the real minimal fixture", () => {
    const missing = decodeContextReceipt(
      recordedEnvelope(minimalFixture, minimalFixture.run_id),
      "/receipt",
      minimalFixture.run_id,
    );
    if (missing.shape !== "recorded" || missing.receipt.kind !== "v1") throw new Error("expected a v1 receipt");
    expect(missing.receipt.delivered).toBeUndefined();
    expect(missing.receipt.omissions).toBeUndefined();
    expect(missing.receipt.termination).toBeUndefined();

    const empty = decodeContextReceipt(recordedEnvelope({
      ...minimalFixture,
      delivered: [],
      omissions: [],
      transformations: [],
    }, minimalFixture.run_id), "/receipt", minimalFixture.run_id);
    if (empty.shape !== "recorded" || empty.receipt.kind !== "v1") throw new Error("expected a v1 receipt");
    expect(empty.receipt.delivered).toEqual([]);
    expect(empty.receipt.omissions).toEqual([]);
    expect(empty.receipt.transformations).toEqual([]);
  });

  it("rejects removed aliases and unknown v1 fields", () => {
    expect(() => decodeContextReceipt(recordedEnvelope({
      ...fullFixture,
      producer_extension: { revision: 2 },
    }))).toThrow(/producer_extension/);
    expect(() => decodeContextReceipt(recordedEnvelope({
      ...fullFixture,
      rendered: { ...fullFixture.rendered, token_count: 61 },
    }))).toThrow(/token_count/);
    expect(() => decodeContextReceipt(recordedEnvelope({
      ...fullFixture,
      limits: { prompt_tokens: 61 },
    }))).toThrow(/limits/);
  });

  it("keeps absent, unrecognized, and unsupported-future receipts distinct", () => {
    expect(decodeContextReceipt({
      run_id: "run_absent",
      shape: "absent",
      context_receipt: {},
      note: "no context receipt was recorded for this run",
    })).toEqual({
      shape: "absent",
      runId: "run_absent",
      note: "no context receipt was recorded for this run",
    });

    expect(decodeContextReceipt({
      run_id: "run_odd",
      shape: "unrecognized",
      context_receipt: { schema: "other.receipt.v7", opaque: true },
    })).toEqual({
      shape: "unrecognized",
      runId: "run_odd",
      schemaVersionRaw: "other.receipt.v7",
      rawKeys: ["schema", "opaque"],
    });

    const future = decodeContextReceipt({
      run_id: "run_future",
      shape: "recorded",
      context_receipt: {
        schema_version: "clozn.context-receipt.v2",
        run_id: "run_future",
        next_generation_fact: true,
      },
    });
    expect(future).toEqual({
      shape: "recorded",
      runId: "run_future",
      receipt: {
        kind: "unsupported",
        schemaVersion: "clozn.context-receipt.v2",
        runId: "run_future",
        rawKeys: ["schema_version", "run_id", "next_generation_fact"],
      },
    });
  });

  it("fails closed on envelope drift, identity drift, unknown enums, and malformed nested facts", () => {
    expect(() => decodeContextReceipt({
      ...recordedEnvelope(minimalFixture, minimalFixture.run_id),
      invented: true,
    })).toThrow(/invented/);
    expect(() => decodeContextReceipt(recordedEnvelope(minimalFixture, minimalFixture.run_id), "/receipt", "run_other"))
      .toThrow(/does not match the request/);
    expect(() => decodeContextReceipt(recordedEnvelope({ ...minimalFixture, run_id: "run_other" }, minimalFixture.run_id)))
      .toThrow(/does not match the response run_id/);
    expect(() => decodeContextReceipt(recordedEnvelope({ ...fullFixture, privacy: "sometimes" })))
      .toThrow(/must be one of full/);
    expect(() => decodeContextReceipt(recordedEnvelope({
      ...minimalFixture,
      delivered: [{ segment_id: "not-a-segment", source_type: "message", original_order: 0 }],
    }, minimalFixture.run_id))).toThrow(/segment id/);
    expect(() => decodeContextReceipt(recordedEnvelope({
      ...minimalFixture,
      delivered: [{
        segment_id: "seg_aaaaaaaaaaaaaaaa",
        source_type: "message",
        original_order: 0,
        sources: [{
          source_id: "src_bbbbbbbbbbbbbbbbbbbbbbbb",
          segment_id: "seg_aaaaaaaaaaaaaaaa",
          message_index: 0,
          unicode_range: [0, 1],
          byte_range: [0, 1],
          content_sha256: "c".repeat(64),
          provenance_kind: "message",
          invented: true,
        }],
      }],
    }, minimalFixture.run_id))).toThrow(/invented/);
  });

  it("reads the canonical receipt resource and validates the selected Run", async () => {
    const runId = "run / α";
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => new Response(JSON.stringify({
      run_id: runId,
      shape: "absent",
      context_receipt: {},
      note: "no context receipt was recorded for this run",
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    await studioApi.contextReceipt(runId);

    expect(String(fetchMock.mock.calls[0][0])).toBe("/runs/run%20%2F%20%CE%B1/context-receipt");
    await expect(studioApi.contextReceipt("   ")).rejects.toBeInstanceOf(ContractError);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
