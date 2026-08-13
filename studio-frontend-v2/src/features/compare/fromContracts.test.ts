import { describe, expect, it } from "vitest";
import { projectComparisonSpecimen, recordedStructureFromComparison } from "./fromContracts";

describe("projectComparisonSpecimen", () => {
  it("does not turn absent output or raw run pairing into invented comparison evidence", () => {
    const specimen = projectComparisonSpecimen(
      { id: "a", response: null, promptSummary: "First" },
      { id: "b", response: "answer", promptSummary: "Second" },
      { relationship: { kind: "arbitrary" }, differences: [] },
    );
    expect(specimen.a.outputState).toBe("unavailable");
    expect(specimen.a.output).toBeUndefined();
    expect(specimen.relationship).toEqual({ kind: "arbitrary" });
    expect(specimen.differences).toEqual([]);
  });
});

it("uses the server's exact first divergence and keeps evidence status independent", () => {
  const comparison = {
    runA: "a",
    runB: "b",
    privacyLimited: false,
    axes: { output: { status: "changed" as const } },
    differences: [{
      dimension: "output.text",
      kind: "changed" as const,
      rank: 1,
      evidence: [{ first_divergence_view: { recorded_answer_location: { a: { state: "exact", start: 1, end: 2 }, b: { state: "exact", start: 1, end: 2 } } } }],
    }],
    findings: [{ classification: "output_changed", status: "observed" as const, summary: "The outputs diverged.", dimensions: ["output.text"] }],
  };
  const structure = recordedStructureFromComparison(comparison, { id: "a", response: "A🙂B" }, { id: "b", response: "A🙃B", parentRunId: "a" });
  expect(structure.relationship.kind).toBe("related");
  expect(structure.differences[0]).toMatchObject({ a: { start: 1, end: 3 }, b: { start: 1, end: 3 }, alignment: "recorded" });
  expect(structure.evidenceByDifferenceId?.["recorded-difference-0"]).toMatchObject({ state: "available", observations: [{ label: "observed" }] });
});

it("keeps non-output condition changes out of the prose navigation spine", () => {
  const comparison = {
    runA: "a", runB: "b", privacyLimited: false,
    axes: { sampling: { status: "changed" as const }, output: { status: "changed" as const } },
    differences: [
      { dimension: "generation.temperature", kind: "changed" as const, rank: 0, evidence: [] },
      { dimension: "output.response_length_words", kind: "changed" as const, rank: 1, evidence: [] },
      { dimension: "output.text", kind: "changed" as const, rank: 2, evidence: [] },
    ],
    findings: [],
  };
  const structure = recordedStructureFromComparison(comparison, { id: "a", response: "A" }, { id: "b", response: "B" });
  expect(structure.differences.map((difference) => difference.label)).toEqual(["output / text", "output / response_length_words"]);
  expect(structure.relationship).toMatchObject({ kind: "arbitrary", changedConditions: ["sampling", "output"] });
});
