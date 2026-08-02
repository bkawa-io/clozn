import { describe, expect, test } from "vitest";
import panel from "./investigation";

describe("investigation panel routing", () => {
  test("claims the canonical rail route and preserves the sessions deep-link alias", () => {
    expect(panel.match("#/investigation")).toEqual({});
    expect(panel.match("#/investigation/")).toEqual({});
    expect(panel.match("#/sessions")).toEqual({});
    expect(panel.match("#/sessions/")).toEqual({});
  });

  test("keeps session investigation routes addressable", () => {
    expect(panel.match("#/sessions/session_abc/investigate")).toEqual({ sessionId: "session_abc" });
    expect(panel.match("#/sessions/session%2Fabc/investigate/")).toEqual({ sessionId: "session/abc" });
    expect(panel.match("#/lens")).toBeNull();
  });
});
