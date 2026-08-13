import { describe, expect, it } from "vitest";
import { toJournalRun } from "./fromContracts";

describe("run journal adapter", () => {
  it("keeps session and lineage separate while preserving recorded signals", () => {
    expect(toJournalRun({
      id: "run-123456789",
      createdTs: 1_700_000_000,
      promptSummary: "Why this answer?",
      responseSummary: "Because of the source.",
      sessionKey: "session_1",
      parentRunId: "run-parent",
      flags: ["truncated"],
      warningCount: 2,
    })).toMatchObject({
      createdAt: "2023-11-14T22:13:20.000Z",
      sessionKey: "session_1",
      parentRunId: "run-parent",
      flags: ["truncated"],
      warningCount: 2,
    });
  });
});
