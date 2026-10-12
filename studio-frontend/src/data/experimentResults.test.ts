import { describe, expect, it } from "vitest";
import fixture from "../../../tests/fixtures/schemas/clozn.experiment.result.v0/valid__minimal_unscored_smoke.json";
import { ContractError } from "./contracts";
import { decodeExperimentCells, decodeExperimentDetail, decodeExperimentList } from "./experimentResults";

const THIN_KEYS = [
  "suite", "case", "variant", "variant_kind", "seed", "status", "run_id",
  "assertions", "min_confidence", "error",
] as const;

function thinDetail() {
  return {
    ...structuredClone(fixture),
    cells: fixture.cells.map((cell) => Object.fromEntries(THIN_KEYS.map((key) => [key, cell[key]]))),
  };
}

function listDocument() {
  const variants = fixture.manifest.variants.map((variant) => variant.name);
  return {
    experiments: [{
      experiment_id: fixture.experiment_id,
      name: fixture.name,
      created_at: fixture.created_at,
      baseline_variant: fixture.summary.baseline_variant,
      variants,
      seeds: fixture.seeds,
      cell_count: fixture.cells.length,
      aggregates: fixture.summary.aggregates,
      suite_fingerprint: { algorithm: "canonical-json-v1", sha256: "a".repeat(64) },
    }],
    total: 1,
    limit: 50,
    offset: 0,
    broken: [{ path: "/results/broken.json", error: "invalid JSON" }],
  };
}

describe("experiment-result read contracts", () => {
  it("decodes a list projection derived from the canonical result fixture", () => {
    const document = decodeExperimentList(listDocument());
    expect(document.experiments[0]).toMatchObject({
      experimentId: fixture.experiment_id,
      baselineVariant: fixture.summary.baseline_variant,
      cellCount: fixture.cells.length,
    });
    expect(document.broken).toEqual([{ path: "/results/broken.json", error: "invalid JSON" }]);
  });

  it("decodes the real result fixture after applying the server's thin-cell projection", () => {
    const detail = decodeExperimentDetail(thinDetail(), "/experiment-results/fixture", fixture.experiment_id);
    expect(detail.schemaVersion).toBe("clozn.experiment.result.v0");
    expect(detail.cells).toHaveLength(fixture.cells.length);
    expect(detail.cells.every((cell) => cell.status === "unscored")).toBe(true);
    expect(detail.variants.map((variant) => variant.name)).toEqual(["base", "candidate"]);
  });

  it("decodes one full on-demand cell and preserves its immutable Run identity", () => {
    const cell = fixture.cells[0];
    const document = decodeExperimentCells({ experiment_id: fixture.experiment_id, cells: [cell] }, "/cells", fixture.experiment_id);
    expect(document.cells[0]).toMatchObject({
      runId: cell.run_id,
      response: cell.response,
      status: "unscored",
    });
  });

  it("fails closed on drift, identity mismatch, and duplicate coordinates", () => {
    expect(() => decodeExperimentList({ ...listDocument(), invented: true })).toThrow(ContractError);
    expect(() => decodeExperimentDetail(thinDetail(), "/detail", "another-id")).toThrow(/does not match/);
    const duplicate = { experiment_id: fixture.experiment_id, cells: [fixture.cells[0], fixture.cells[0]] };
    expect(() => decodeExperimentCells(duplicate)).toThrow(/duplicate coordinates/);
  });
});
