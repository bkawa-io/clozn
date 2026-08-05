import { describe, expect, test } from "vitest";
import panel from "./model";

describe("Runtime panel routing", () => {
  test("uses Runtime naming while preserving the historical model deep link", () => {
    expect(panel.navLabel).toBe("Runtime");
    expect(panel.routeName({})).toBe("RUNTIME");
    expect(panel.match("#/runtime")).toEqual({});
    expect(panel.match("#/runtime/")).toEqual({});
    expect(panel.match("#/model")).toEqual({});
    expect(panel.match("#/model/")).toEqual({});
  });

  test("does not claim the inventory endpoint path or unrelated routes", () => {
    expect(panel.match("#/models")).toBeNull();
    expect(panel.match("#/behavior")).toBeNull();
  });
});
