import { expect, it } from "vitest";
import { decodeReadiness } from "../../data/contracts";
import { toRuntimeSnapshot } from "./fromContracts";

it("projects only runtime facts reported by the current contracts", () => {
  const snapshot = toRuntimeSnapshot(
    { status: "ok", service: "clozn" },
    { status: "not_ready", service: "clozn", reason: "default worker loading" },
    {
      managed: true,
      defaultModelId: "model-a",
      preloadModelIds: ["model-a"],
      maxLoadedModels: 2,
      configuredCount: 1,
      residentCount: 1,
      models: [{ modelId: "model-a", state: "loading", isDefault: true, preloaded: true, runtimeKeySha256: "a".repeat(64), workerGeneration: 3, workerId: "worker-1", failureCode: null }],
    },
  );
  expect(snapshot).toMatchObject({ service: "live", readiness: "not_ready", readinessDetail: "default worker loading", residentCount: 1 });
  expect(snapshot.models?.[0]).toMatchObject({ modelId: "model-a", runtimeKeyFingerprint: "a".repeat(64), workerIdentity: "worker-1" });
  expect(snapshot.protocol).toBeUndefined();
  expect(snapshot.queueCount).toBeUndefined();
  expect(snapshot.telemetry).toBeUndefined();
});

it("uses legacy /readyz worker, queue, and wire facts when the registry truthfully has no rows", () => {
  const readiness = decodeReadiness({
    status: "ok",
    service: "clozn",
    protocol_version: "1.1",
    model: "models/current.gguf",
    queue: { active: 1, waiting: 2, capacity: 32, wait_timeout_seconds: 600 },
    capabilities: { jlens: true, sae: false, attn_knockout: true, execution_fork: true, execution_fork_regimes: ["reprefill"] },
    worker: {
      model: "models/current.gguf",
      model_sha256: "b".repeat(64),
      device: "cpu",
      n_ctx: 4096,
      build_id: "build-17",
      worker_generation_id: "opaque-generation",
      capabilities: { jlens: true },
    },
  });
  const snapshot = toRuntimeSnapshot(
    { status: "ok", service: "clozn" },
    readiness,
    { managed: false, defaultModelId: null, preloadModelIds: [], maxLoadedModels: 1, configuredCount: 1, residentCount: 1, models: [] },
  );

  expect(snapshot.models).toEqual([expect.objectContaining({ modelId: "models/current.gguf", state: "ready", isDefault: true, workerGenerationId: "opaque-generation" })]);
  expect(snapshot.protocolVersion).toBe("1.1");
  expect(snapshot.queue).toEqual({ active: 1, waiting: 2, capacity: 32 });
  expect(snapshot.identityByModelId?.["models/current.gguf"]).toMatchObject({ artifactSha: "b".repeat(64), backendDevice: "cpu", contextSize: 4096, engineBuild: "build-17" });
  expect(snapshot.capabilitiesByModelId?.["models/current.gguf"]).toEqual(expect.arrayContaining([
    expect.objectContaining({ label: "J-lens", state: "available now" }),
    expect.objectContaining({ label: "SAE", state: "runtime unsupported" }),
  ]));
  expect(snapshot.protocol).toBeUndefined();
});

it("does not invent a legacy configured row from a non-ready probe", () => {
  const snapshot = toRuntimeSnapshot(
    { status: "ok", service: "clozn" },
    { status: "not_ready", service: "clozn", model: "models/previous.gguf" },
    { managed: false, defaultModelId: null, preloadModelIds: [], maxLoadedModels: 1, configuredCount: 0, residentCount: 0, models: [] },
  );
  expect(snapshot.models).toEqual([]);
});
