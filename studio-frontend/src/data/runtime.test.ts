import { describe, expect, it } from "vitest";
import { ContractError } from "./contracts";
import { decodeLocalModels, decodeRuntimeHealth } from "./runtime";

describe("v3 Runtime read contracts", () => {
  it("decodes the worker health fields the surface consumes and ignores protocol extensions", () => {
    expect(decodeRuntimeHealth({
      engine: {
        status: "ok",
        model: "/models/qwen.Q4_K_M.gguf",
        mode: "autoregressive",
        architecture: "qwen2",
        n_ctx: 4096,
        gpu_layers: 12,
        capabilities: {
          score_variants: true,
          attn_knockout: false,
          execution_fork_regimes: ["live_kv_truncated", "reprefill"],
        },
        native_chat_io: { available: true },
      },
    })).toEqual({
      kind: "worker",
      status: "ok",
      model: "/models/qwen.Q4_K_M.gguf",
      mode: "autoregressive",
      architecture: "qwen2",
      contextTokens: 4096,
      gpuLayers: 12,
      capabilities: { score_variants: true, attn_knockout: false },
    });
  });

  it("decodes the closed managed-runtime projection", () => {
    expect(decodeRuntimeHealth({ engine: {
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
        worker_generation: 2,
        worker_id: "alpha-worker",
        failure_code: null,
      }],
    } })).toMatchObject({
      kind: "managed",
      defaultModelId: "alpha",
      residentCount: 1,
      models: [{ modelId: "alpha", state: "ready", workerGeneration: 2 }],
    });
  });

  it("fails closed on malformed health fields and managed projection drift", () => {
    expect(() => decodeRuntimeHealth({
      engine: { status: "ok", model: "x.gguf", mode: "autoregressive", architecture: "x", n_ctx: "4096", capabilities: {} },
    })).toThrow(ContractError);
    expect(() => decodeRuntimeHealth({
      engine: {
        status: "ok", model: "x.gguf", mode: "autoregressive", architecture: "x",
        capabilities: { invented_descriptor: ["not", "a", "flag"] },
      },
    })).toThrow(/invented_descriptor/);
    expect(() => decodeRuntimeHealth({ engine: {
      managed: true,
      default_model_id: "alpha",
      preload_model_ids: [],
      max_loaded_models: 1,
      configured_count: 0,
      resident_count: 0,
      models: [],
      private_worker_port: 8123,
    } })).toThrow(/private_worker_port/);
    expect(() => decodeRuntimeHealth({ engine: {
      managed: true,
      default_model_id: "alpha",
      preload_model_ids: [],
      max_loaded_models: 1,
      configured_count: 2,
      resident_count: 0,
      models: [],
    } })).toThrow(/must match models.length/);
  });

  it("decodes only the canonical five-field local model record", () => {
    expect(decodeLocalModels({ models: [{
      path: "/models/qwen.Q4_K_M.gguf",
      filename: "qwen.Q4_K_M.gguf",
      size_bytes: 500_000_000,
      quant: "Q4_K_M",
      sha256: null,
    }] })).toEqual([{
      path: "/models/qwen.Q4_K_M.gguf",
      filename: "qwen.Q4_K_M.gguf",
      sizeBytes: 500_000_000,
      quant: "Q4_K_M",
      sha256: null,
    }]);
    expect(() => decodeLocalModels({ models: [{
      path: "/models/x.gguf", filename: "x.gguf", size_bytes: 1, quant: null, sha256: null, loaded: true,
    }] })).toThrow(/loaded/);
  });
});
