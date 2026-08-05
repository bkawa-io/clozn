import { beforeEach, describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../test/fetch";
import { render, screen, within } from "../../test/render";
import { Model } from "./Model";

function pathOf(request: PendingFetch): string {
  return typeof request.input === "string" ? request.input : request.input.toString();
}

async function respondWithWorkspace(controller: ReturnType<typeof createFetchController>) {
  const health = await controller.nextRequest();
  const axes = await controller.nextRequest();
  const inventory = await controller.nextRequest();
  const profiles = await controller.nextRequest();

  expect([pathOf(health), pathOf(axes), pathOf(inventory), pathOf(profiles)]).toEqual([
    "/engine/health",
    "/steer/axes",
    "/models/local",
    "/profiles/list",
  ]);

  controller.respondJson(health, {
    engine: {
      model: "/models/qwen2.Q4_K_M.gguf",
      architecture: "qwen2",
      n_ctx: 4096,
      n_layer: 32,
      gpu_layers: 20,
      capabilities: {
        attn_knockout: false,
        jlens: true,
      },
    },
  });
  controller.respondJson(axes, {
    axes: [{ name: "concise", value: 0.25, calibrated: true }],
  });
  controller.respondJson(inventory, {
    models: [{
      path: "/models/qwen2.Q4_K_M.gguf",
      filename: "qwen2.Q4_K_M.gguf",
      quant: "Q4_K_M",
      size_bytes: 4_000_000_000,
      sha256: "0123456789abcdef0123456789abcdef",
    }],
  });
  controller.respondJson(profiles, { active: "local-default" });
}

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("Runtime Model surface", () => {
  test("reframes the surface as Runtime while preserving the existing scoped fetches", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const { container } = render(<Model inspectorOpen={false} />);

    await respondWithWorkspace(controller);

    const runtimeSurface = await screen.findByRole("region", { name: "Runtime" });
    expect(within(runtimeSurface).getByRole("heading", { name: "Runtime" })).toBeInTheDocument();
    expect(within(runtimeSurface).getByRole("heading", { name: "Serving model record" })).toBeInTheDocument();
    const engineRecord = screen.getByRole("region", { name: "Serving model record" });
    expect(within(engineRecord).getByText("qwen2.Q4_K_M.gguf")).toBeInTheDocument();
    expect(container.querySelector(".state-board-hard-cap")).toBeNull();
    expect(container.querySelector('[role="progressbar"]')).toBeNull();
    expect(screen.getByRole("img", { name: /Resident capacity unavailable -- The current inventory route lists model files/i })).toBeInTheDocument();
  });

  test("renders an explicit reason for a disabled capability instead of treating it as supported", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const { container } = render(<Model inspectorOpen={false} />);

    await respondWithWorkspace(controller);
    await screen.findByRole("heading", { name: "Capability flags" });

    const unavailableCapability = container.querySelector('[data-capability-id="attn_knockout"]');
    expect(unavailableCapability).not.toBeNull();
    expect(within(unavailableCapability as HTMLElement).getByText("Attention knockout")).toBeInTheDocument();
    expect(within(unavailableCapability as HTMLElement).getByText("Reported unavailable by /engine/health.")).toBeInTheDocument();
    expect(within(unavailableCapability as HTMLElement).getByRole("img", {
      name: /Unavailable -- Attention knockout is reported unavailable by \/engine\/health/i,
    })).toBeInTheDocument();
  });

  test("keeps unsupported installation controls visibly blocked with their reasons", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<Model inspectorOpen={false} />);

    await respondWithWorkspace(controller);

    const snapshotOffer = await screen.findByRole("region", { name: "Snapshot storage" });
    expect(within(snapshotOffer).getByRole("button", { name: "Snapshot controls unavailable" })).toBeDisabled();
    expect(within(snapshotOffer).getByText(/Studio has no snapshot action descriptor or storage evidence to invoke\./)).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Ollama adoption" })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "Privacy controls" })).toBeInTheDocument();
  });
});
