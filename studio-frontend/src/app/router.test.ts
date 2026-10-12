import { describe, expect, it } from "vitest";
import { INSPECT_VIEWS, readRoute, routeHref, type InspectRoute, type StudioRoute } from "./router";

describe("Studio hash routes", () => {
  it("uses Sessions as the neutral destination", () => {
    expect(readRoute("#/")).toEqual({ surface: "sessions" });
    expect(readRoute("#/sessions")).toEqual({ surface: "sessions" });
    expect(readRoute("#/unknown")).toEqual({ surface: "sessions" });
    expect(routeHref({ surface: "sessions" })).toBe("#/sessions");
  });

  it("round-trips a Session and its selected Run", () => {
    const route = { surface: "session", sessionId: "session / one", runId: "run / two" } as const;
    expect(routeHref(route)).toBe("#/sessions/session%20%2F%20one?run=run%20%2F%20two");
    expect(readRoute(routeHref(route))).toEqual(route);
    expect(readRoute("#/sessions/session_alpha")).toEqual({ surface: "session", sessionId: "session_alpha" });
  });

  it.each([
    "#/sessions/%20",
    "#/sessions/%E0%A4%A",
    "#/sessions/session_alpha?run=%20",
    "#/sessions/session_alpha?other=run_a",
    "#/sessions/session_alpha/extra",
  ])("lands neutrally for malformed Session route %s", (href) => {
    expect(readRoute(href)).toEqual({ surface: "sessions" });
  });

  it("owns the canonical Runtime route", () => {
    expect(readRoute("#/runtime")).toEqual({ surface: "runtime" });
    expect(routeHref({ surface: "runtime" })).toBe("#/runtime");
  });

  it("round-trips canonical Compare pair routes", () => {
    expect(readRoute("#/compare")).toEqual({ surface: "compare" });
    const route = { surface: "compare", runA: "run / α", runB: "run / β" } as const;
    expect(routeHref(route)).toBe("#/compare/run%20%2F%20%CE%B1/run%20%2F%20%CE%B2");
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  it("round-trips canonical Compare matrix routes", () => {
    expect(readRoute("#/compare/matrix")).toEqual({ surface: "compare", mode: "matrix" });
    const route = { surface: "compare", mode: "matrix", experimentId: "experiment / α" } as const;
    expect(routeHref(route)).toBe("#/compare/matrix/experiment%20%2F%20%CE%B1");
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  it.each([
    "#/compare/run_a", "#/compare/run_a/run_b/extra", "#/compare/run_a/run_a",
    "#/compare/%20/run_b", "#/compare/%E0%A4%A/run_b", "#/compare?b=run_b",
    "#/compare/run_a/run_b?replay=1", "#/compare/matrix/experiment/extra",
    "#/compare/matrix/%20", "#/compare/matrix?filter=fail",
  ])("lands neutrally for malformed Compare route %s", (href) => {
    expect(readRoute(href)).toEqual({ surface: "sessions" });
  });

  it("does not emit a partial or same-Run Compare route", () => {
    expect(routeHref({ surface: "compare", runA: "run_a" } as unknown as StudioRoute)).toBe("#/compare");
    expect(routeHref({ surface: "compare", runA: "run_a", runB: "run_a" })).toBe("#/compare");
  });

  it("round-trips an Inspect Run and locked answer coordinates", () => {
    const route = {
      surface: "inspect", runId: "run / one", sessionId: "session / two",
      answerStart: 12, answerEnd: 29, sourceId: "source / three",
    } as const;
    expect(routeHref(route)).toBe(
      "#/inspect/run%20%2F%20one?sessionId=session%20%2F%20two&answerStart=12&answerEnd=29&sourceId=source%20%2F%20three",
    );
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  it.each(INSPECT_VIEWS)("round-trips the shipped Inspect view %s", (view) => {
    const route = { surface: "inspect", runId: "run / alpha", sessionId: "session / beta", view } as InspectRoute;
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  it.each([
    "#/inspect/run_alpha?view=unknown",
    "#/inspect/run_alpha?view=actions&action=remove-source",
    "#/inspect/run_alpha?view=read&token=2",
    "#/inspect/run_alpha?view=receipt&sourceId=source_a",
    "#/inspect/run_alpha?answerStart=2",
    "#/inspect/run_alpha?answerEnd=7",
    "#/inspect/run_alpha?answerStart=-1&answerEnd=7",
    "#/inspect/run_alpha?answerStart=7&answerEnd=7",
    "#/inspect/run_alpha?sourceId=source_a",
    "#/inspect/run_alpha?session=session_a",
    "#/inspect/run_alpha?sessionId=%20",
    "#/inspect", "#/inspect/%20", "#/inspect/%E0%A4%A", "#/inspect/run_alpha/extra",
  ])("lands neutrally for malformed Inspect route %s", (href) => {
    expect(readRoute(href)).toEqual({ surface: "sessions" });
  });

  it("does not emit invalid optional Inspect coordinates", () => {
    expect(routeHref({ surface: "inspect", runId: "run_a", sourceId: "source_a" } as unknown as StudioRoute))
      .toBe("#/inspect/run_a");
    expect(routeHref({ surface: "inspect", runId: "" })).toBe("#/sessions");
  });

});
