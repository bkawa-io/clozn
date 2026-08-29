import { describe, expect, it } from "vitest";
import { readRoute, routeHref } from "./router";

describe("v3 hash routes", () => {
  it("represents a selected run in the session URL", () => {
    const route = { surface: "session", sessionId: "session / one", runId: "run / two" } as const;
    expect(routeHref(route)).toBe("#/sessions/session%20%2F%20one?run=run%20%2F%20two");
    expect(readRoute(routeHref(route))).toEqual(route);
  });

  it("keeps the session index as the default surface", () => {
    expect(readRoute("#/unknown")).toEqual({ surface: "sessions" });
    expect(routeHref({ surface: "sessions" })).toBe("#/sessions");
  });
});
