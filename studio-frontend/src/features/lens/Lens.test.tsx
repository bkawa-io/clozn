import { describe, expect, test } from "vitest";
import type { TokenReading } from "../../data/types";
import { tokenLensClasses, type ReaderLensId } from "./Lens";

const linkedToken: TokenReading = {
  text: "therefore",
  entropy: 1.3,
  confidence: 0.42,
  band: "shaky",
  sources: [{
    sourceId: "source-a",
    label: "Retrieved note",
    effect: "supports",
    deltaNats: 0.24,
    evidenceState: "causally_supported",
  }],
};

describe("Lens reader annotations", () => {
  test("combines independent per-token decoration without replacing the prose", () => {
    const active = new Set<ReaderLensId>(["shakiness", "concepts"]);
    expect(tokenLensClasses(linkedToken, active).split(" ")).toEqual(expect.arrayContaining([
      "lens-reader-token",
      "is-shaky",
      "concept-lens-active",
    ]));
  });

  test("source links are a claim/span-level concern, never a per-token class", () => {
    const active = new Set<ReaderLensId>(["shakiness", "sources", "concepts"]);
    expect(tokenLensClasses(linkedToken, active).split(" ")).not.toEqual(expect.arrayContaining([
      "has-source-index",
      "source-tone-2",
      "has-provenance",
    ]));
  });

  test("keeps the clean reader free of evidence styling", () => {
    expect(tokenLensClasses(linkedToken, new Set())).toBe("lens-reader-token");
  });
});
