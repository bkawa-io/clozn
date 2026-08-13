export type RuntimeSurfacePhase = "loading" | "ready" | "unavailable" | "error" | "stale";
export type RuntimeLifecycle = "unloaded" | "ready" | "loading" | "evicting" | "failed" | (string & {});
export type CapabilityState = "available now" | "runtime unsupported" | "artifact unavailable" | "not qualified" | "not reported";
export type TelemetryAvailability = "available" | "unsupported" | "unavailable" | "error";

export interface RuntimeModelRecord {
  modelId: string;
  state: RuntimeLifecycle;
  isDefault?: boolean;
  preloaded?: boolean;
  runtimeKeyFingerprint?: string;
  workerGeneration?: number | null;
  /** Opaque process generation reported by the legacy worker; it is not a numeric registry generation. */
  workerGenerationId?: string;
  workerIdentity?: string | null;
  failureCode?: string | null;
}

export interface RuntimeIdentity {
  artifactFormat?: string;
  artifactSha?: string;
  quantization?: string;
  backendDevice?: string;
  contextSize?: string | number;
  engineBuild?: string;
  templateFingerprint?: string;
  adapterIdentity?: string;
}

export interface RuntimeCapability {
  label: string;
  state: CapabilityState;
  detail?: string;
}

export interface RuntimeTelemetryMetric {
  label: string;
  value: string;
}

export interface RuntimeTelemetry {
  availability: TelemetryAvailability;
  detail?: string;
  provider?: string;
  observedAt?: string;
  device?: string;
  metrics?: readonly RuntimeTelemetryMetric[];
}

/** Directly observed gateway queue counters from /readyz. */
export interface RuntimeQueueSnapshot {
  active?: number;
  waiting?: number;
  capacity?: number;
}

export interface RuntimeSnapshot {
  service: "live" | "unreachable";
  readiness?: "ready" | "not_ready";
  readinessDetail?: string;
  /** Reported gateway-to-worker wire version. This does not claim compatibility. */
  protocolVersion?: string;
  /** A compatibility assessment is only shown when a caller explicitly supplies one. */
  protocol?: "compatible" | "incompatible" | "not_reported";
  queue?: RuntimeQueueSnapshot;
  /** @deprecated Use the complete queue snapshot when the gateway reports it. */
  queueCount?: number | null;
  maxLoadedModels?: number;
  configuredCount?: number;
  residentCount?: number;
  models?: readonly RuntimeModelRecord[];
  identityByModelId?: Readonly<Record<string, RuntimeIdentity | undefined>>;
  capabilitiesByModelId?: Readonly<Record<string, readonly RuntimeCapability[] | undefined>>;
  telemetry?: RuntimeTelemetry;
}

export type RuntimeComposedState = "READY" | "DEGRADED" | "NOT READY" | "UNREACHABLE" | "CHECKING";

export function lifecycleLabel(state: RuntimeLifecycle): string {
  return state.replaceAll("_", " ").toUpperCase();
}

export function lifecycleTone(state: RuntimeLifecycle): "ready" | "transition" | "failed" | "neutral" {
  if (state === "ready") return "ready";
  if (state === "loading" || state === "evicting") return "transition";
  if (state === "failed") return "failed";
  return "neutral";
}

export function isOccupyingSlot(state: RuntimeLifecycle): boolean {
  return state === "ready" || state === "loading" || state === "evicting" || state === "failed";
}

export function composeRuntimeState(snapshot?: RuntimeSnapshot): RuntimeComposedState {
  if (!snapshot) return "CHECKING";
  if (snapshot.service === "unreachable") return "UNREACHABLE";
  if (snapshot.readiness !== "ready") return "NOT READY";
  const defaultModel = snapshot.models?.find((model) => model.isDefault);
  const hasReadySecondary = snapshot.models?.some((model) => !model.isDefault && model.state === "ready");
  return defaultModel?.state === "failed" && hasReadySecondary ? "DEGRADED" : "READY";
}

export function capabilitiesFor(model?: RuntimeModelRecord, supplied?: readonly RuntimeCapability[]): readonly RuntimeCapability[] {
  const suppliedByLabel = new Map(supplied?.map((capability) => [capability.label, capability]));
  return [
    { label: "Generation", state: model?.state === "ready" ? "available now" : "not reported" },
    { label: "Token rewind", state: "not reported" },
    { label: "Context influence", state: "not reported" },
    { label: "J-lens", state: "not reported" },
    { label: "SAE", state: "not reported" },
    { label: "Attention knockout / causal trace", state: "not reported" },
  ].map((capability) => suppliedByLabel.get(capability.label) ?? capability as RuntimeCapability);
}
