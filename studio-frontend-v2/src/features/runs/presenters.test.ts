import { describe, expect, test } from "vitest";
import { filterJournal, groupRunsByDay, runState, type JournalRun } from "./presenters";

const run = (over: Partial<JournalRun>): JournalRun => ({ id: "run-one", flags: [], warningCount: 0, ...over });

describe("Runs journal presenters", () => {
  test("keeps failure and truncation distinct", () => {
    expect(runState(run({ flags: ["error"] }))).toBe("failed");
    expect(runState(run({ finishReason: "length" }))).toBe("truncated");
    expect(runState(run({ finishReason: "stop" }))).toBe("complete");
  });

  test("groups only adjacent chronologically ordered days", () => {
    const groups = groupRunsByDay([
      run({ id: "a", createdAt: "2026-08-10T10:00:00Z" }),
      run({ id: "b", createdAt: "2026-08-10T09:00:00Z" }),
      run({ id: "c", createdAt: "2026-08-09T09:00:00Z" }),
    ]);
    expect(groups.map((group) => group.runs.map((item) => item.id))).toEqual([["a", "b"], ["c"]]);
  });

  test("search includes readable prompt and response content", () => {
    const runs = [run({ id: "a", prompt: "Refund policy", response: "Eligible" }), run({ id: "b", prompt: "Build logs" })];
    expect(filterJournal(runs, "eligible").map((item) => item.id)).toEqual(["a"]);
  });
});
