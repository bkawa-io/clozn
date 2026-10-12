import userEvent from "@testing-library/user-event";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import fixture from "../../../../tests/fixtures/schemas/clozn.experiment.result.v0/valid__minimal_unscored_smoke.json";
import { studioApi } from "../../data/client";
import { ContractError } from "../../data/contracts";
import { decodeExperimentCells, decodeExperimentDetail, decodeExperimentList } from "../../data/experimentResults";
import { CompareSurface } from "./Compare";

const THIN_KEYS = [
  "suite", "case", "variant", "variant_kind", "seed", "status", "run_id",
  "assertions", "min_confidence", "error",
] as const;

function detail() {
  return decodeExperimentDetail({
    ...structuredClone(fixture),
    cells: fixture.cells.map((cell) => Object.fromEntries(THIN_KEYS.map((key) => [key, cell[key]]))),
  });
}

function indexDocument() {
  const variants = fixture.manifest.variants.map((variant) => variant.name);
  return decodeExperimentList({
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
  });
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Experiment matrix surface", () => {
  it("lists recorded experiments and visibly retains broken artifacts", async () => {
    vi.spyOn(studioApi, "experimentResults").mockResolvedValue(indexDocument());
    render(<CompareSurface route={{ surface: "compare", mode: "matrix" }} />);

    expect(await screen.findByText(fixture.name)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open matrix" })).toHaveAttribute(
      "href", `#/compare/matrix/${fixture.experiment_id}`,
    );
    expect(screen.getByRole("heading", { name: "Broken result artifacts" })).toBeInTheDocument();
    expect(screen.getByText("invalid JSON")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("renders the complete thin matrix and fetches full cell evidence only on click", async () => {
    const decodedDetail = detail();
    const cell = fixture.cells[0];
    vi.spyOn(studioApi, "experimentResult").mockResolvedValue(decodedDetail);
    const fullRead = vi.spyOn(studioApi, "experimentCells").mockResolvedValue(
      decodeExperimentCells({ experiment_id: fixture.experiment_id, cells: [cell] }),
    );
    const user = userEvent.setup();
    render(<CompareSurface route={{ surface: "compare", mode: "matrix", experimentId: fixture.experiment_id }} />);

    const table = await screen.findByRole("table");
    expect(within(table).getAllByText("unscored")).toHaveLength(fixture.cells.length);
    expect(fullRead).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "View base evidence for target t1 seed 0" }));
    expect(await screen.findByRole("complementary", { name: "Selected cell evidence" })).toHaveTextContent(cell.response);
    expect(fullRead).toHaveBeenCalledWith(fixture.experiment_id, {
      suite: "target", caseName: "t1", variant: "base", seed: 0,
    }, expect.any(AbortSignal));
    expect(screen.getByRole("link", { name: "Open immutable Run evidence" })).toHaveAttribute("href", `#/inspect/${cell.run_id}`);
    expect(screen.getByText(/does not rerun or promote/)).toBeInTheDocument();
  });

  it("keeps malformed list responses distinct from unavailable requests", async () => {
    const read = vi.spyOn(studioApi, "experimentResults");
    read.mockRejectedValueOnce(new ContractError("/experiment-results", "$.experiments drifted"));
    const view = render(<CompareSurface route={{ surface: "compare", mode: "matrix" }} />);
    expect(await screen.findByText("Malformed experiment-result contract.")).toBeInTheDocument();

    read.mockRejectedValueOnce(new Error("gateway offline"));
    view.rerender(<CompareSurface route={{ surface: "compare", mode: "matrix" }} />);
    // Remount to model a new navigation after the previous request failed.
    view.unmount();
    render(<CompareSurface route={{ surface: "compare", mode: "matrix" }} />);
    expect(await screen.findByText("Experiment results unavailable.")).toBeInTheDocument();
  });
});
