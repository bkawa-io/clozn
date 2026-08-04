import { describe, expect, test } from "vitest";
import { render, screen, within } from "../test/render";
import {
  formatDurationNs,
  SpanWaterfall,
  type SpanWaterfallPhase,
} from "./SpanWaterfall";

const PHASES: SpanWaterfallPhase[] = [
  {
    id: "gateway-queue",
    name: "gateway_queue",
    durationNs: 125_000_000,
    startNs: 0,
    clockOwner: "clozn_gateway",
    clockDomain: "clozn_gateway:monotonic",
    measurement: "measured",
    aggregation: "exclusive",
  },
  {
    id: "worker-prefill",
    name: "prefill",
    durationNs: 1_250_000_000,
    startNs: 50_000_000,
    clockOwner: "clozn_worker",
    clockDomain: "clozn_worker:steady_clock",
    measurement: "measured",
    aggregation: "exclusive",
  },
];

describe("SpanWaterfall", () => {
  test("groups spans into one clearly labelled lane per clock owner", () => {
    render(<SpanWaterfall phases={PHASES} />);

    const gatewayLane = screen.getByRole("region", { name: "Clock owner: clozn_gateway" });
    const workerLane = screen.getByRole("region", { name: "Clock owner: clozn_worker" });

    // `within` matters here because phase names and clock-domain labels can recur in other lanes; the
    // assertion is about ownership, not merely whether some similarly named span rendered somewhere.
    expect(within(gatewayLane).getByText("gateway queue")).toBeInTheDocument();
    expect(within(workerLane).getByText("prefill")).toBeInTheDocument();
    expect(screen.getByText(/Separate clock-owner lanes are not mutually aligned/)).toBeInTheDocument();
  });

  test("renders unaccounted duration as a labelled visible hatch gap", () => {
    const { container } = render(
      <SpanWaterfall
        phases={PHASES}
        aggregation={{ knownDurationNs: 1_375_000_000, unaccountedDurationNs: 625_000_000, wallClockTotalNs: 2_000_000_000 }}
      />,
    );

    const accounting = screen.getByRole("region", { name: "Request accounting" });
    expect(within(accounting).getByText("Unaccounted gap")).toBeInTheDocument();
    expect(within(accounting).getByText("625 ms")).toBeInTheDocument();
    expect(container.querySelector(".span-waterfall-accounting-gap")).not.toBeNull();
  });

  test("uses EvidenceMark for an absent duration instead of drawing a zero-width bar", () => {
    const { container } = render(
      <SpanWaterfall phases={[{
        id: "missing",
        name: "decode",
        clockOwner: "clozn_worker",
        clockDomain: "clozn_worker:steady_clock",
      }]} />,
    );

    expect(screen.getByRole("img", { name: /Duration not recorded -- This span has no recorded duration/ })).toBeInTheDocument();
    expect(container.querySelector(".span-waterfall-bar")).toBeNull();
  });

  test("formats nanoseconds as milliseconds below one second and seconds at or above one second", () => {
    expect(formatDurationNs(125_000_000)).toBe("125 ms");
    expect(formatDurationNs(1_250_000_000)).toBe("1.25 s");
  });
});
