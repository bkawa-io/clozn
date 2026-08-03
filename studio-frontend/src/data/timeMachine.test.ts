import { afterEach, describe, expect, test, vi } from "vitest";
import {
  continueTimeMachine,
  TimeMachineContinuationReceiptError,
} from "./timeMachine";

function completedReceipt() {
  return {
    schema_version: "clozn.time-machine-continuation.v1",
    continuation_id: "tmc_1234567890abcdef1234",
    created_ts: 1,
    finished_ts: 2,
    requested_run_id: "run-parent",
    source_turn: 3,
    status: "completed",
    request: {
      request_id: "request-1",
      turn: 3,
      append_kind: "new_user_turn",
      user_content_sha256: "a".repeat(64),
      user_content_bytes: 12,
      max_tokens: 91,
      generation_config_sha256: "b".repeat(64),
    },
    source: {
      status: "resolved",
      source_run_id: "run-source",
      source_turn: 3,
      resolution: "exact_organic_session_prefix",
    },
    source_checkpoint: {
      status: "available",
      provenance: "durable_pin_import",
      restart_safe: true,
      source_run_id: "run-source",
      checkpoint_reference_id: "checkpoint-ref",
      checkpoint_id: "checkpoint-id",
      pin_id: "pin_12345678901234567890",
    },
    identity: { status: "matched" },
    append: { status: "validated" },
    sampler: { status: "preserved" },
    exactness: {
      status: "confirmed",
      claim: "exact_historical_state_append",
      append_only_execution: true,
      historical_prefix_recomputed: false,
      historical_prefix_retokenized_for_execution: false,
      structural_fallback_used: false,
    },
    worker: { status: "completed" },
    child_lineage: {
      status: "created",
      requested_parent_run_id: "run-parent",
      source_checkpoint_run_id: "run-source",
      child_run_id: "run-child",
      relation: "exact_continuation",
      parent_immutable: true,
      source_immutable: true,
      receipt_persisted: true,
    },
    unavoidable_differences: ["new_append_tokens", "new_generated_suffix"],
    reasons: [{ code: "continuation_completed", message: "child persisted" }],
    failure: null,
  };
}

afterEach(() => vi.restoreAllMocks());

describe("continueTimeMachine", () => {
  test("sends only the closed v1 request and parses a proven append-only child", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(completedReceipt()), { status: 201 }),
    );

    const result = await continueTimeMachine("run-parent", {
      turn: 3,
      user: { content: "What happened next?" },
      maxTokens: 91,
    });

    expect(result.status).toBe("completed");
    if (result.status === "completed") {
      expect(result.childLineage).toEqual({
        requestedParentRunId: "run-parent",
        sourceCheckpointRunId: "run-source",
        childRunId: "run-child",
      });
      expect(result.exactness.structuralFallbackUsed).toBe(false);
    }
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0][0])).toBe("/runs/run-parent/time-machine/continue");
    expect(JSON.parse(String((fetchMock.mock.calls[0][1] as RequestInit).body))).toEqual({
      turn: 3,
      user: { content: "What happened next?" },
      max_tokens: 91,
    });
  });

  test("fails closed when a purported completed receipt does not prove append-only exactness", async () => {
    const malformed = completedReceipt();
    malformed.exactness.structural_fallback_used = true;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(malformed), { status: 201 }),
    );

    await expect(continueTimeMachine("run-parent", {
      turn: 3,
      user: { content: "What happened next?" },
      maxTokens: 91,
    })).rejects.toBeInstanceOf(TimeMachineContinuationReceiptError);
  });
});
