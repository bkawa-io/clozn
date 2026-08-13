import { describe, expect, test } from "vitest";
import { readRoute, routeHref } from "./router";

describe("Studio v2 routes", () => {
  test("maps one selected run to the inspect surface", () => {
    expect(readRoute("#/runs/run%2Fone")).toEqual({ surface: "inspect", runId: "run/one" });
  });

  test("keeps comparison identity in the URL", () => {
    const route = { surface: "compare", runA: "parent", runB: "child" } as const;
    expect(routeHref(route)).toBe("#/compare/parent/child");
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  test("retains exact comparison and local reader coordinates across routes", () => {
    const selection = {
      runAId: "parent/a",
      runBId: "child b",
      differenceId: "recorded-difference-4",
      a: { start: 7, end: 11 },
      b: { start: 9, end: 14 },
    };
    const compare = { surface: "compare", runA: selection.runAId, runB: selection.runBId, selectedDifference: selection } as const;
    expect(readRoute(routeHref(compare))).toEqual(compare);

    const testThis = {
      surface: "time-travel",
      runId: "child b",
      answerLocus: { id: "answer-5", start: 3, end: 8 },
      sourceLocus: { id: "source-2", start: 18, end: 25 },
      comparison: selection,
    } as const;
    expect(readRoute(routeHref(testThis))).toEqual(testThis);
  });

  test("fails malformed coordinate query values closed without discarding a valid run route", () => {
    expect(readRoute("#/time-travel/run-1?answer=span&answerStart=1&answerEnd=1&source=ctx&sourceStart=2&sourceEnd=5"))
      .toEqual({ surface: "time-travel", runId: "run-1", sourceLocus: { id: "ctx", start: 2, end: 5 } });
  });

  test("fails unknown paths closed to the runs journal", () => {
    expect(readRoute("#/internals")).toEqual({ surface: "runs" });
  });

  test("round-trips a recorded decision boundary and rival token", () => {
    const route = { surface: "time-travel", runId: "run-1", mode: "token", tokenPosition: 7, breakpointId: "breakpoint-7", rivalTokenId: 525 } as const;
    expect(readRoute(routeHref(route))).toEqual(route);
  });
});
