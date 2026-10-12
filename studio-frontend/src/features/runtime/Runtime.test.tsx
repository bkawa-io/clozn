import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { RuntimeSurface } from "./Runtime";
import { App } from "../../app/App";

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    statusText: status === 200 ? "OK" : "Unavailable",
    headers: { "content-type": "application/json" },
  });
}

const workerHealth = {
  engine: {
    status: "ok",
    model: "/models/qwen.Q4_K_M.gguf",
    mode: "autoregressive",
    architecture: "qwen2",
    backend: "metal",
    device: "metal",
    protocol_version: "4",
    n_ctx: 4096,
    gpu_layers: 22,
    capabilities: { score_variants: true, attn_knockout: false },
  },
};

const inventory = {
  models: [{
    path: "/models/qwen.Q4_K_M.gguf",
    filename: "qwen.Q4_K_M.gguf",
    size_bytes: 500_000_000,
    quant: "Q4_K_M",
    sha256: null,
  }],
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.location.hash = "";
});

describe("Runtime surface", () => {
  it("is reachable from the primary navigation at the canonical Runtime route", async () => {
    window.location.hash = "#/runtime";
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => (
      Promise.resolve(json(String(input) === "/engine/health" ? workerHealth : inventory))
    )));
    render(<App />);

    expect(screen.getByRole("link", { name: "Runtime" })).toHaveAttribute("aria-current", "page");
    expect(screen.getByRole("link", { name: "Runtime" })).toHaveAttribute("href", "#/runtime");
    expect(screen.getByRole("link", { name: "CLOZN Runtime" })).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: "Serving runtime" })).toBeInTheDocument();
    expect(document.title).toBe("CLOZN · Runtime");
  });

  it("renders backed worker health, capability flags, and model inventory without controls", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const path = String(input);
      return Promise.resolve(json(path === "/engine/health" ? workerHealth : inventory));
    }));
    render(<RuntimeSurface />);

    expect(await screen.findAllByText("qwen.Q4_K_M.gguf")).toHaveLength(2);
    expect(screen.getByText("qwen2")).toBeInTheDocument();
    expect(screen.getByText("score_variants")).toBeInTheDocument();
    expect(screen.getByText("attn_knockout")).toBeInTheDocument();
    expect(screen.getByText("Serving")).toBeInTheDocument();
    expect(screen.getByText("Reported")).toBeInTheDocument();
    expect(screen.getByText(/does not load, switch, download, or remove models/)).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("keeps a malformed health response distinct while retaining valid inventory", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const path = String(input);
      if (path === "/engine/health") {
        return Promise.resolve(json({
          ...workerHealth,
          engine: { ...workerHealth.engine, n_ctx: "4096" },
        }));
      }
      return Promise.resolve(json(inventory));
    }));
    render(<RuntimeSurface />);

    expect(await screen.findByRole("alert")).toHaveTextContent("Malformed engine health contract.");
    expect(screen.getByText("qwen.Q4_K_M.gguf")).toBeInTheDocument();
    expect(screen.getByText("Partial")).toBeInTheDocument();
  });

  it("renders the managed router projection without inferring worker ports", async () => {
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      if (String(input) === "/models/local") return Promise.resolve(json({ models: [] }));
      return Promise.resolve(json({ engine: {
        managed: true,
        default_model_id: "alpha",
        preload_model_ids: ["alpha"],
        max_loaded_models: 2,
        configured_count: 1,
        resident_count: 1,
        models: [{
          model_id: "alpha",
          state: "ready",
          default: true,
          preloaded: true,
          runtime_key_sha256: "a".repeat(64),
          worker_generation: 1,
          worker_id: "alpha-worker",
          failure_code: null,
        }],
      } }));
    }));
    render(<RuntimeSurface />);

    const panel = await screen.findByRole("region", { name: "Serving runtime" });
    expect(within(panel).getAllByText("alpha")).toHaveLength(2);
    expect(within(panel).getByText("1 of 1")).toBeInTheDocument();
    expect(within(panel).getByText("ready")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("No local GGUF files were reported.")).toBeInTheDocument());
  });
});
