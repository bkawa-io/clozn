import { describe, expect, test, vi } from "vitest";
import { render, screen, within } from "../../test/render";
import { DEFAULT_FILTERS } from "./urlState";
import { Matrix } from "./Matrix";
import type { ExperimentDetail } from "./types";

const detail: ExperimentDetail = {
  experimentId: "experiment-1",
  name: "Comparison matrix fixture",
  createdAt: null,
  manifest: {
    name: "Comparison matrix fixture",
    baselineVariant: "baseline",
    seeds: [1],
    defaults: {},
    variants: [
      { name: "baseline", kind: "base" },
      { name: "candidate", kind: "prompt" },
    ],
    suites: {
      target: { cases: [{ name: "missing-coordinate" }, { name: "failed-assertion" }] },
      guard: { cases: [] },
    },
  },
  manifestSha256: null,
  suiteFingerprint: null,
  vcs: null,
  artifactProvenance: null,
  seeds: [1],
  summary: {
    baselineVariant: "baseline",
    aggregates: {},
    comparisons: [],
  },
  cells: [
    {
      suite: "target",
      case: "missing-coordinate",
      variant: "baseline",
      variantKind: "base",
      seed: 1,
      status: "pass",
      runId: "baseline-missing",
      assertions: [],
      minConfidence: null,
      error: null,
    },
    {
      suite: "target",
      case: "failed-assertion",
      variant: "baseline",
      variantKind: "base",
      seed: 1,
      status: "pass",
      runId: "baseline-fail",
      assertions: [],
      minConfidence: null,
      error: null,
    },
    {
      suite: "target",
      case: "failed-assertion",
      variant: "candidate",
      variantKind: "prompt",
      seed: 1,
      status: "fail",
      runId: "candidate-fail",
      assertions: [],
      minConfidence: null,
      error: null,
    },
  ],
};

describe("experiment comparison matrix", () => {
  test("renders an unavailable coordinate as explicit evidence absence, distinct from a failed assertion", () => {
    render(
      <Matrix
        detail={detail}
        filters={DEFAULT_FILTERS}
        onFiltersChange={vi.fn()}
        selection={null}
        onSelectCell={vi.fn()}
      />,
    );

    const unavailable = screen.getByText("UNAVAILABLE").closest(".experiments-cell") as HTMLElement;
    const failed = screen.getByText("FAIL").closest(".experiments-cell") as HTMLElement;

    expect(unavailable).toHaveClass("is-unavailable");
    expect(within(unavailable).getByRole("img", { name: /Unavailable -- No result cell was recorded/ })).toBeInTheDocument();
    expect(failed).toHaveClass("is-fail");
    expect(failed).not.toHaveClass("is-unavailable");
    expect(within(failed).getByText("×")).toBeInTheDocument();
  });
});
