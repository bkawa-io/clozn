import { describe, expect, test } from "vitest";
import { textFragments } from "./model";

describe("linked reader text coordinates", () => {
  test("preserves every character around stable half-open loci", () => {
    const text = "The refund is approved.";
    const fragments = textFragments(text, [{ id: "answer-1", start: 4, end: 10 }]);
    expect(fragments.map((part) => part.text).join("")).toBe(text);
    expect(fragments[1]).toMatchObject({ text: "refund", locus: { id: "answer-1" } });
  });

  test("drops malformed and overlapping coordinates rather than duplicating prose", () => {
    const text = "abcdef";
    const fragments = textFragments(text, [
      { id: "first", start: 1, end: 4 },
      { id: "overlap", start: 2, end: 5 },
      { id: "outside", start: 9, end: 10 },
    ]);
    expect(fragments.map((part) => part.text).join("")).toBe(text);
    expect(fragments.filter((part) => part.locus).map((part) => part.locus?.id)).toEqual(["first"]);
  });
});
