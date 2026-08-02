import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../test/fetch";
import { render, screen } from "../test/render";
import type { RuntimeState } from "../data/types";
import { SnapshotsPanel } from "./snapshots";

function pathOf(request: PendingFetch): string {
  return typeof request.input === "string" ? request.input : request.input.toString();
}

const runtime: RuntimeState = {
  status: "connected",
  runs: [{
    id: "run-alpha", label: "Explain alpha · native · alpha", prompt: "Explain alpha", response: "Alpha",
    createdAt: "2026-08-01T00:00:00Z", source: "openai_api", client: "local", model: "qwen",
    substrate: "cpu", duration: "1 s", flags: [], warningCount: 0, activeDialCount: 0, memoryCardCount: 0,
  }],
};

const manifest = {
  schema_version: "clozn.pinned-checkpoint.v1",
  pin_id: "pin_12345678901234567890",
  run_id: "run-alpha",
  pinned_at: "2026-08-01T00:00:00Z",
  identity: { architecture: "qwen2", n_ctx: 4096 },
  state: { n_past: 128, n_tokens: 128, prompt_tokens: 96, causal: true, has_sampler: false, has_steer: false },
  blob: { kv_bytes: 2048, envelope_bytes: 3072 },
};

beforeEach(() => {
  vi.restoreAllMocks();
  location.hash = "";
});

describe("Snapshots panel", () => {
  test("lists pins, previews storage before pinning, then explicitly persists", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<SnapshotsPanel runtime={runtime} inspectorOpen={false} params={{}} />);

    const list = await controller.nextRequest();
    expect(pathOf(list)).toBe("/snapshots");
    controller.respondJson(list, { schema_version: "clozn.pinned-checkpoint-list.v1", snapshots: [manifest] });
    expect(await screen.findByText("run-alpha")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "PREVIEW PIN" }));
    const preview = await controller.nextRequest();
    expect(pathOf(preview)).toContain("/runs/run-alpha/snapshot/pin");
    expect(JSON.parse(String(preview.init?.body))).toMatchObject({ preview: true });
    controller.respondJson(preview, { ok: true, preview: true, run_id: "run-alpha", envelope_bytes: 3072, size_bytes: 2048 });
    expect(await screen.findByText(/This will write 3.0 KB/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "PIN 3.0 KB" }));
    const pin = await controller.nextRequest();
    expect(JSON.parse(String(pin.init?.body))).toMatchObject({ preview: false });
    controller.respondJson(pin, { ok: true, manifest });
    const refresh = await controller.nextRequest();
    expect(pathOf(refresh)).toBe("/snapshots");
    controller.respondJson(refresh, { schema_version: "clozn.pinned-checkpoint-list.v1", snapshots: [manifest] });
    expect(await screen.findByText(/Checkpoint pinned/)).toBeInTheDocument();
  });

  test("selection alone never pins or unpins a checkpoint", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<SnapshotsPanel runtime={runtime} inspectorOpen={false} params={{}} />);
    const list = await controller.nextRequest();
    controller.respondJson(list, { schema_version: "clozn.pinned-checkpoint-list.v1", snapshots: [] });
    await screen.findByText("No durable snapshots are pinned yet.");
    await user.selectOptions(screen.getByLabelText("RUN"), "run-alpha");
    expect(controller.requests.filter((request) => pathOf(request).includes("snapshot/pin") || pathOf(request).includes("/unpin")).length).toBe(0);
  });

  test("makes cascade unpin an explicit second confirmation", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<SnapshotsPanel runtime={runtime} inspectorOpen={false} params={{}} />);
    const list = await controller.nextRequest();
    controller.respondJson(list, { schema_version: "clozn.pinned-checkpoint-list.v1", snapshots: [manifest] });
    await screen.findByRole("article", { name: "Pinned snapshot run-alpha" });
    await user.click(screen.getByRole("button", { name: "UNPIN", exact: true }));
    await user.click(screen.getByRole("button", { name: "CASCADE UNPIN", exact: true }));
    const request = await controller.nextRequest();
    expect(pathOf(request)).toBe("/snapshots/run-alpha/unpin");
    expect(JSON.parse(String(request.init?.body))).toEqual({ cascade: true });
    controller.respondJson(request, { ok: true, action: "unpin", run_id: "run-alpha", cascade: true });
    const refresh = await controller.nextRequest();
    controller.respondJson(refresh, { schema_version: "clozn.pinned-checkpoint-list.v1", snapshots: [] });
    expect(await screen.findByText("No durable snapshots are pinned yet.")).toBeInTheDocument();
  });
});
