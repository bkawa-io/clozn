import { describe, expect, it } from "vitest";
import { INSPECT_VIEWS, readRoute, routeHref, type InspectRoute, type StudioRoute } from "./router";

describe("Studio hash routes", () => {
  it("uses Runs as the only neutral destination", () => {
    expect(readRoute("#/")) .toEqual({ surface: "runs" });
    expect(readRoute("#/runs")).toEqual({ surface: "runs" });
    expect(readRoute("#/conversations")).toEqual({ surface: "runs" });
    expect(readRoute("#/sessions")).toEqual({ surface: "runs" });
    expect(readRoute("#/runtime")).toEqual({ surface: "runs" });
    expect(readRoute("#/compare/a/b")).toEqual({ surface: "runs" });
    expect(routeHref({ surface: "runs" })).toBe("#/runs");
  });

  it("round-trips a selected conversation", () => {
    const route = { surface: "runs", conversationId: "session / alpha" } as const;
    expect(routeHref(route)).toBe("#/conversations/session%20%2F%20alpha");
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  it.each(INSPECT_VIEWS)("round-trips the %s workspace mode", (view) => {
    const route = { surface: "inspect", runId: "run / alpha", view } as InspectRoute;
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  it("round-trips a Context Effects answer selection", () => {
    const route = {
      surface: "inspect", runId: "run / one", view: "context-effects",
      conversationId: "conversation / alpha",
      answerStart: 12, answerEnd: 29, sourceId: "source / three",
    } as const;
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  it("round-trips Context Effects granularity and arm budget", () => {
    const route = {
      surface: "inspect", runId: "run_alpha", granularity: "adaptive",
      armBudget: 12, scoringMode: "exact", answerStart: 1, answerEnd: 4,
    } as const;
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  it.each([
    "#/inspect/run_alpha?view=receipt",
    "#/inspect/run_alpha?view=minimal-context",
    "#/inspect/run_alpha?view=lean-context&answerStart=1&answerEnd=2",
    "#/inspect/run_alpha?answerStart=2",
    "#/inspect/run_alpha?answerStart=-1&answerEnd=7",
    "#/inspect/run_alpha?answerStart=7&answerEnd=7",
    "#/inspect/run_alpha?sourceId=source_a",
    "#/inspect/run_alpha?granularity=adaptive",
    "#/inspect/run_alpha?granularity=semantic&armBudget=12",
    "#/inspect/run_alpha?granularity=message&armBudget=0",
    "#/inspect/run_alpha?view=lean-context&granularity=message&armBudget=2",
    "#/inspect/run_alpha?scoringMode=other",
    "#/inspect/run_alpha?view=lean-context&scoringMode=fast",
    "#/inspect/run_alpha?conversationId=%20",
    "#/inspect/run_alpha?sessionId=session_a",
    "#/conversations/%20", "#/conversations/%E0%A4%A", "#/conversations/run_a/extra",
    "#/conversations/run_a?view=chat",
    "#/inspect", "#/inspect/%20", "#/inspect/%E0%A4%A", "#/inspect/run_alpha/extra",
  ])("lands on Runs for malformed workspace route %s", (href) => {
    expect(readRoute(href)).toEqual({ surface: "runs" });
  });

  it("does not emit invalid workspace coordinates", () => {
    expect(routeHref({ surface: "inspect", runId: "run_a", sourceId: "source_a" } as StudioRoute))
      .toBe("#/inspect/run_a");
    expect(routeHref({ surface: "inspect", runId: "" })).toBe("#/runs");
  });
});
