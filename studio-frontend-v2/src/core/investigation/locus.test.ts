import { describe, expect, it } from "vitest";
import { locusKey, sameLocus, withSelection, type InvestigationLocus } from "./locus";

const answerSpan: InvestigationLocus = {
  kind: "answer-span", runId: "run/a", answerId: "answer:1", startChar: 4, endChar: 9,
};

describe("investigation loci", () => {
  it("derives a stable identity from backend coordinates, not a display label", () => {
    expect(locusKey(answerSpan)).toBe("answer-span:run%2Fa:answer%3A1:4:9");
    expect(sameLocus(answerSpan, { ...answerSpan })).toBe(true);
    expect(sameLocus(answerSpan, { ...answerSpan, endChar: 10 })).toBe(false);
  });

  it("retains the source surface and lock state with a selection", () => {
    expect(withSelection(answerSpan, "inspect", true)).toEqual({ locus: answerSpan, origin: "inspect", locked: true });
  });
});
