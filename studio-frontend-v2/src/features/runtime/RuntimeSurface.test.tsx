import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { composeRuntimeState, type RuntimeSnapshot } from "./model";
import { RuntimeSurface } from "./RuntimeSurface";

afterEach(cleanup);

const snapshot: RuntimeSnapshot = {
  service: "live", readiness: "ready", protocol: "compatible", queueCount: 0,
  maxLoadedModels: 2, configuredCount: 3, residentCount: 2,
  models: [
    { modelId: "default-7b", state: "ready", isDefault: true, preloaded: true, runtimeKeyFingerprint: "a9c3", workerGeneration: 12 },
    { modelId: "secondary-7b", state: "loading", workerGeneration: 13 },
    { modelId: "archive-13b", state: "unloaded", preloaded: false },
  ],
};

describe("RuntimeSurface", () => {
  it("separates service liveness, inference readiness, residency, and queue facts", () => {
    render(<RuntimeSurface snapshot={snapshot} />);
    expect(screen.getByLabelText(/^LIVE: Gateway service liveness$/)).toBeVisible();
    expect(screen.getByText("2 / 2 slots")).toBeVisible();
    expect(screen.getByText("0 waiting")).toBeVisible();
    expect(screen.getAllByText("LOADING").length).toBeGreaterThan(0);
    expect(screen.getByText("UNLOADED")).toBeVisible();
  });

  it("makes default failure with a secondary ready worker degraded without calling it unavailable", () => {
    render(<RuntimeSurface snapshot={{ ...snapshot, models: [{ ...snapshot.models![0], state: "failed", failureCode: "worker_identity_mismatch" }, { ...snapshot.models![1], state: "ready" }] }} />);
    expect(screen.getByLabelText(/^DEGRADED: Composed Studio runtime state$/)).toBeVisible();
    expect(screen.getByText(/A secondary ready worker may still serve requests/)).toBeVisible();
  });

  it("supports accessible model selection and does not manufacture unreported identity or telemetry", async () => {
    const user = userEvent.setup();
    render(<RuntimeSurface snapshot={snapshot} />);
    const secondary = screen.getByRole("option", { name: /secondary-7b/i });
    await user.click(secondary);
    expect(secondary).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("secondary-7b", { selector: ".selected-instrument-bay p" })).toBeVisible();
    expect(screen.getAllByText("Not reported by current runtime contract").length).toBeGreaterThan(0);
    expect(screen.getByText("Resource telemetry not reported by this runtime.")).toBeVisible();
    expect(screen.queryByText(/0 MB|0 GB|0%/)).not.toBeInTheDocument();
  });

  it("keeps unsupported telemetry neutral and explicit", () => {
    render(<RuntimeSurface snapshot={{ ...snapshot, telemetry: { availability: "unsupported", detail: "CPU runtime has no GPU telemetry." } }} />);
    expect(screen.getByText("UNSUPPORTED")).toBeVisible();
    expect(screen.getByText("CPU runtime has no GPU telemetry.")).toBeVisible();
  });

  it("renders reported legacy queue and wire facts without calling the protocol compatible", () => {
    render(<RuntimeSurface snapshot={{
      service: "live",
      readiness: "ready",
      protocolVersion: "1.1",
      queue: { active: 1, waiting: 2, capacity: 32 },
      maxLoadedModels: 1,
      configuredCount: 1,
      residentCount: 1,
      models: [{ modelId: "models/current.gguf", state: "ready", isDefault: true, workerGenerationId: "opaque-generation" }],
    }} />);
    expect(screen.getByText("2 waiting")).toBeVisible();
    expect(screen.getByText("1 active · 32 capacity")).toBeVisible();
    expect(screen.getByLabelText(/^REPORTED 1\.1: Reported gateway-to-worker wire version/)).toBeVisible();
    expect(screen.queryByText("COMPATIBLE")).not.toBeInTheDocument();
    expect(screen.getByText("opaque-generation")).toBeVisible();
  });

  it("selects the first truthful model when an async runtime snapshot arrives", () => {
    const view = render(<RuntimeSurface phase="loading" />);
    view.rerender(<RuntimeSurface phase="ready" snapshot={snapshot} />);
    expect(screen.getByRole("option", { name: /default-7b/i })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("default-7b", { selector: ".selected-instrument-bay p" })).toBeVisible();
  });
});

describe("composeRuntimeState", () => {
  it("does not conflate an unreachable gateway and an inference-not-ready live service", () => {
    expect(composeRuntimeState({ service: "unreachable" })).toBe("UNREACHABLE");
    expect(composeRuntimeState({ service: "live", readiness: "not_ready" })).toBe("NOT READY");
  });
});
