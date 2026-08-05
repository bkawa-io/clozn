import { beforeEach, describe, expect, test, vi } from "vitest";
import { loadRunInspection, loadRunPerformance } from "../../data/api";
import { render, screen, waitFor } from "../../test/render";
import { RunTimingInstrument } from "./RunDiagnostics";

vi.mock("../../data/api", () => ({
  loadRunInspection: vi.fn(),
  loadRunPerformance: vi.fn(),
}));

const performance = vi.mocked(loadRunPerformance);

beforeEach(() => {
  vi.mocked(loadRunInspection).mockReset();
  performance.mockReset();
});

describe("RunTimingInstrument", () => {
  test("keeps a failed performance request local and explained", async () => {
    performance.mockRejectedValue(new Error("trace unavailable"));
    render(<RunTimingInstrument runId="run-timing" />);

    expect(screen.getByRole("heading", { name: "Recorded performance" })).toBeInTheDocument();
    expect(await screen.findByRole("img", {
      name: "Performance artifact unavailable -- The recorded performance trace request failed for this run.",
    })).toBeInTheDocument();
    expect(performance).toHaveBeenCalledWith("run-timing", expect.any(AbortSignal));
  });

  test("uses the clock-aware waterfall when a timing trace is present", async () => {
    performance.mockResolvedValue({
      totalDuration: { value: 120, source: "trace" },
      rules: {
        schemaVersion: "clozn.performance-trace.v1",
        phases: [{
          name: "prefill",
          durationNs: 30_000_000,
          clockOwner: "worker",
          clockDomain: "worker.monotonic",
          measurement: "measured",
          aggregation: "exclusive",
          includes: [],
        }],
        metrics: {},
        aggregation: {
          knownDurationNs: 30_000_000,
          unaccountedDurationNs: 90_000_000,
          wallClockTotalNs: 120_000_000,
          measurementCoverage: 0.25,
          consistency: "consistent",
        },
        diagnoses: [],
      },
    });

    render(<RunTimingInstrument runId="run-timing" />);
    expect(await screen.findByRole("heading", { name: "Timing breakdown" })).toBeInTheDocument();
    expect(screen.getByText("Clock owner")).toBeInTheDocument();
    expect(screen.getByText("Unaccounted gap")).toBeInTheDocument();
    await waitFor(() => expect(performance).toHaveBeenCalledTimes(1));
  });
});
