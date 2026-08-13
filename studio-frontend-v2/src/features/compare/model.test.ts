import { describe, expect, it } from "vitest";
import { differenceTextParts, validRange } from "./model";

describe("Compare model", () => {
  it("partitions only valid recorded regions and preserves all surrounding prose", () => {
    expect(differenceTextParts("alpha beta gamma", [{ id: "first", label: "first", kind: "changed", a: { start: 6, end: 10 }, alignment: "recorded" }], "a"))
      .toEqual([{ text: "alpha " }, { text: "beta", differenceId: "first" }, { text: " gamma" }]);
  });

  it("drops overlapping coordinates rather than duplicating text or pretending a precedence", () => {
    expect(differenceTextParts("abcdef", [
      { id: "one", label: "one", kind: "changed", a: { start: 1, end: 4 }, alignment: "recorded" },
      { id: "two", label: "two", kind: "changed", a: { start: 3, end: 5 }, alignment: "recorded" },
    ], "a")).toEqual([{ text: "a" }, { text: "bcd", differenceId: "one" }, { text: "ef" }]);
    expect(validRange("abc", { start: 0, end: 4 })).toBeUndefined();
  });
});
