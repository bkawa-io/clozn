import type { HealthStatus, JsonObject, RuntimeModels } from "../../data/contracts";
import type { RuntimeCapability, RuntimeIdentity, RuntimeModelRecord, RuntimeSnapshot } from "./model";

function reportedBoolean(document: JsonObject | undefined, key: string): boolean | undefined {
  const value = document?.[key];
  return typeof value === "boolean" ? value : undefined;
}

function capability(label: string, value: boolean | undefined): RuntimeCapability | undefined {
  if (value === undefined) return undefined;
  return { label, state: value ? "available now" : "runtime unsupported" };
}

function legacyCapabilities(readiness: HealthStatus): readonly RuntimeCapability[] | undefined {
  const reported = readiness.capabilities ?? readiness.worker?.capabilities;
  const values = [
    capability("Token rewind", reportedBoolean(reported, "execution_fork")),
    capability("J-lens", reportedBoolean(reported, "jlens")),
    capability("SAE", reportedBoolean(reported, "sae")),
    capability("Attention knockout / causal trace", reportedBoolean(reported, "attn_knockout")),
  ].filter((entry): entry is RuntimeCapability => entry !== undefined);
  return values.length ? values : undefined;
}

/**
 * /runtime/models is intentionally only a managed-registry projection. In legacy single-model mode it
 * truthfully contains no registry rows, while /readyz carries the live worker, wire protocol, and queue.
 * Keep those two sources distinct and construct one display row only when /readyz directly reports a
 * ready legacy worker identity.
 */
function legacyModel(readiness: HealthStatus, inventory: RuntimeModels): RuntimeModelRecord | undefined {
  if (inventory.managed || readiness.status !== "ok") return undefined;
  const modelId = readiness.model ?? readiness.worker?.model;
  if (!modelId) return undefined;
  return {
    modelId,
    state: "ready",
    isDefault: true,
    runtimeKeyFingerprint: readiness.worker?.modelSha256,
    workerGenerationId: readiness.worker?.workerGenerationId,
  };
}

function legacyIdentity(readiness: HealthStatus): RuntimeIdentity | undefined {
  const worker = readiness.worker;
  if (!worker) return undefined;
  const backendDevice = [...new Set([worker.backend, worker.device].filter((value): value is string => Boolean(value)))].join(" / ") || undefined;
  const identity: RuntimeIdentity = {
    artifactSha: worker.modelSha256,
    backendDevice,
    contextSize: worker.contextSize,
    engineBuild: worker.buildId ?? worker.engineVersion,
  };
  return Object.values(identity).some((value) => value !== undefined) ? identity : undefined;
}

export function toRuntimeSnapshot(
  health: HealthStatus,
  readiness: HealthStatus,
  inventory: RuntimeModels,
): RuntimeSnapshot {
  const legacy = legacyModel(readiness, inventory);
  const legacyModelId = legacy?.modelId;
  const legacyIdentityFacts = legacyIdentity(readiness);
  const capabilities = legacyCapabilities(readiness);
  return {
    service: health.status === "ok" ? "live" : "unreachable",
    readiness: readiness.status === "ok" ? "ready" : "not_ready",
    readinessDetail: readiness.reason,
    protocolVersion: readiness.protocolVersion,
    queue: readiness.queue,
    maxLoadedModels: inventory.maxLoadedModels,
    configuredCount: inventory.configuredCount,
    residentCount: inventory.residentCount,
    models: inventory.managed
      ? inventory.models.map((model) => ({
        modelId: model.modelId,
        state: model.state,
        isDefault: model.isDefault,
        preloaded: model.preloaded,
        runtimeKeyFingerprint: model.runtimeKeySha256,
        workerGeneration: model.workerGeneration,
        workerIdentity: model.workerId,
        failureCode: model.failureCode,
      }))
      : legacy ? [legacy] : [],
    identityByModelId: legacyModelId && legacyIdentityFacts
      ? { [legacyModelId]: legacyIdentityFacts }
      : undefined,
    capabilitiesByModelId: legacyModelId && capabilities
      ? { [legacyModelId]: capabilities }
      : undefined,
  };
}
