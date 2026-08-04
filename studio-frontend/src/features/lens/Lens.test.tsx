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
  test("combines independent text lenses without replacing the prose", () => {
    const active = new Set<ReaderLensId>(["shakiness", "sources", "provenance", "concepts"]);
    expect(tokenLensClasses(linkedToken, active, 2).split(" ")).toEqual(expect.arrayContaining([
      "lens-reader-token",
      "is-shaky",
      "has-source-index",
      "source-tone-2",
      "has-provenance",
      "concept-lens-active",
    ]));
  });

  test("keeps the clean reader free of evidence styling", () => {
    expect(tokenLensClasses(linkedToken, new Set(), 2)).toBe("lens-reader-token");
  });
});
