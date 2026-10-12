import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

function fail(endpoint: string, path: string, message: string): never {
  throw new ContractError(endpoint, `${path} ${message}`);
}

function object(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(endpoint, path, "must be an object");
  }
  return value as JsonRecord;
}

function required(row: JsonRecord, key: string, endpoint: string, path: string): unknown {
  if (!(key in row)) fail(endpoint, path, `is missing required field ${key}`);
  return row[key];
}

function stringValue(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string") fail(endpoint, path, "must be a string");
  return value;
}

function nonEmptyString(value: unknown, endpoint: string, path: string): string {
  const parsed = stringValue(value, endpoint, path);
  if (!parsed.trim()) fail(endpoint, path, "must not be empty");
  return parsed;
}

function nonNegativeInteger(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    fail(endpoint, path, "must be a non-negative integer");
  }
  return value;
}

function booleanValue(value: unknown, endpoint: string, path: string): boolean {
  if (typeof value !== "boolean") fail(endpoint, path, "must be a boolean");
  return value;
}

function nullableString(value: unknown, endpoint: string, path: string): string | null {
  return value === null ? null : stringValue(value, endpoint, path);
}

function sha256(value: unknown, endpoint: string, path: string): string {
  const parsed = stringValue(value, endpoint, path);
  if (!/^[0-9a-f]{64}$/.test(parsed)) fail(endpoint, path, "must be a lowercase SHA256 digest");
  return parsed;
}

function nullableSha256(value: unknown, endpoint: string, path: string): string | null {
  return value === null ? null : sha256(value, endpoint, path);
}

function optionalString(row: JsonRecord, key: string, endpoint: string, path: string): string | undefined {
  return key in row ? stringValue(row[key], endpoint, `${path}.${key}`) : undefined;
}

function closed(row: JsonRecord, keys: readonly string[], endpoint: string, path: string): void {
  const allowed = new Set(keys);
  for (const key of Object.keys(row)) {
    if (!allowed.has(key)) fail(endpoint, `${path}.${key}`, "is not part of the contract");
  }
}

export interface LocalModelRecord {
  path: string;
  filename: string;
  sizeBytes: number;
  quant: string | null;
  sha256: string | null;
}

export interface WorkerRuntime {
  kind: "worker";
  status: "ok";
  model: string;
  mode: string;
  architecture: string;
  device?: string;
  backend?: string;
  protocolVersion?: string;
  engineVersion?: string;
  buildId?: string;
  modelSha256?: string;
  contextTokens?: number;
  gpuLayers?: number;
  capabilities: Readonly<Record<string, boolean>>;
}

export type ManagedModelState = "unloaded" | "loading" | "ready" | "evicting" | "failed";

export interface ManagedRuntimeModel {
  modelId: string;
  state: ManagedModelState;
  default: boolean;
  preloaded: boolean;
  runtimeKeySha256: string;
  workerGeneration: number | null;
  workerId: string | null;
  failureCode: string | null;
}

export interface ManagedRuntime {
  kind: "managed";
  defaultModelId: string;
  preloadModelIds: readonly string[];
  maxLoadedModels: number;
  configuredCount: number;
  residentCount: number;
  models: readonly ManagedRuntimeModel[];
}

export type RuntimeHealth = WorkerRuntime | ManagedRuntime;

export function decodeLocalModels(value: unknown, endpoint = "/models/local"): readonly LocalModelRecord[] {
  const document = object(value, endpoint, "$");
  closed(document, ["models"], endpoint, "$");
  const models = required(document, "models", endpoint, "$");
  if (!Array.isArray(models)) fail(endpoint, "$.models", "must be an array");
  return models.map((value, index) => {
    const path = `$.models[${index}]`;
    const row = object(value, endpoint, path);
    closed(row, ["path", "filename", "size_bytes", "quant", "sha256"], endpoint, path);
    return {
      path: nonEmptyString(required(row, "path", endpoint, path), endpoint, `${path}.path`),
      filename: nonEmptyString(required(row, "filename", endpoint, path), endpoint, `${path}.filename`),
      sizeBytes: nonNegativeInteger(required(row, "size_bytes", endpoint, path), endpoint, `${path}.size_bytes`),
      quant: nullableString(required(row, "quant", endpoint, path), endpoint, `${path}.quant`),
      sha256: nullableSha256(required(row, "sha256", endpoint, path), endpoint, `${path}.sha256`),
    };
  });
}

function decodeManagedModel(value: unknown, endpoint: string, path: string): ManagedRuntimeModel {
  const row = object(value, endpoint, path);
  closed(row, [
    "model_id", "state", "default", "preloaded", "runtime_key_sha256",
    "worker_generation", "worker_id", "failure_code",
  ], endpoint, path);
  const state = required(row, "state", endpoint, path);
  const states: readonly ManagedModelState[] = ["unloaded", "loading", "ready", "evicting", "failed"];
  if (typeof state !== "string" || !states.includes(state as ManagedModelState)) {
    fail(endpoint, `${path}.state`, `must be one of ${states.join(", ")}`);
  }
  return {
    modelId: nonEmptyString(required(row, "model_id", endpoint, path), endpoint, `${path}.model_id`),
    state: state as ManagedModelState,
    default: booleanValue(required(row, "default", endpoint, path), endpoint, `${path}.default`),
    preloaded: booleanValue(required(row, "preloaded", endpoint, path), endpoint, `${path}.preloaded`),
    runtimeKeySha256: sha256(required(row, "runtime_key_sha256", endpoint, path), endpoint, `${path}.runtime_key_sha256`),
    workerGeneration: row.worker_generation === null
      ? null
      : nonNegativeInteger(required(row, "worker_generation", endpoint, path), endpoint, `${path}.worker_generation`),
    workerId: nullableString(required(row, "worker_id", endpoint, path), endpoint, `${path}.worker_id`),
    failureCode: nullableString(required(row, "failure_code", endpoint, path), endpoint, `${path}.failure_code`),
  };
}

function decodeManagedRuntime(engine: JsonRecord, endpoint: string): ManagedRuntime {
  closed(engine, [
    "managed", "default_model_id", "preload_model_ids", "max_loaded_models",
    "configured_count", "resident_count", "models",
  ], endpoint, "$.engine");
  const preloadIds = required(engine, "preload_model_ids", endpoint, "$.engine");
  const models = required(engine, "models", endpoint, "$.engine");
  if (!Array.isArray(preloadIds)) fail(endpoint, "$.engine.preload_model_ids", "must be an array");
  if (!Array.isArray(models)) fail(endpoint, "$.engine.models", "must be an array");
  const parsedModels = models.map((value, index) => decodeManagedModel(value, endpoint, `$.engine.models[${index}]`));
  const configuredCount = nonNegativeInteger(required(engine, "configured_count", endpoint, "$.engine"), endpoint, "$.engine.configured_count");
  const residentCount = nonNegativeInteger(required(engine, "resident_count", endpoint, "$.engine"), endpoint, "$.engine.resident_count");
  if (configuredCount !== parsedModels.length) fail(endpoint, "$.engine.configured_count", "must match models.length");
  if (residentCount > configuredCount) fail(endpoint, "$.engine.resident_count", "must not exceed configured_count");
  if (residentCount !== parsedModels.filter((model) => model.state === "ready").length) {
    fail(endpoint, "$.engine.resident_count", "must match the number of ready models");
  }
  const defaultModelId = nonEmptyString(required(engine, "default_model_id", endpoint, "$.engine"), endpoint, "$.engine.default_model_id");
  if (parsedModels.filter((model) => model.default).length !== 1 || !parsedModels.some((model) => model.default && model.modelId === defaultModelId)) {
    fail(endpoint, "$.engine.models", "must mark default_model_id as the single default model");
  }
  return {
    kind: "managed",
    defaultModelId,
    preloadModelIds: preloadIds.map((value, index) => nonEmptyString(value, endpoint, `$.engine.preload_model_ids[${index}]`)),
    maxLoadedModels: nonNegativeInteger(required(engine, "max_loaded_models", endpoint, "$.engine"), endpoint, "$.engine.max_loaded_models"),
    configuredCount,
    residentCount,
    models: parsedModels,
  };
}

function decodeWorkerRuntime(engine: JsonRecord, endpoint: string): WorkerRuntime {
  // Worker health is an additive engine protocol. Validate every field this surface
  // consumes, while leaving extension fields owned by that protocol untouched.
  if (required(engine, "status", endpoint, "$.engine") !== "ok") {
    fail(endpoint, "$.engine.status", "must be ok");
  }
  const rawCapabilities = object(required(engine, "capabilities", endpoint, "$.engine"), endpoint, "$.engine.capabilities");
  const capabilities: Record<string, boolean> = {};
  for (const [name, value] of Object.entries(rawCapabilities)) {
    if (typeof value === "boolean") {
      capabilities[name] = value;
      continue;
    }
    if (name === "execution_fork_regimes" && Array.isArray(value)) {
      value.forEach((regime, index) => nonEmptyString(regime, endpoint, `$.engine.capabilities.execution_fork_regimes[${index}]`));
      continue;
    }
    fail(endpoint, `$.engine.capabilities.${name}`, "must be a boolean capability flag");
  }
  const optionalInteger = (key: string): number | undefined => key in engine
    ? nonNegativeInteger(engine[key], endpoint, `$.engine.${key}`)
    : undefined;
  return {
    kind: "worker",
    status: "ok",
    model: nonEmptyString(required(engine, "model", endpoint, "$.engine"), endpoint, "$.engine.model"),
    mode: nonEmptyString(required(engine, "mode", endpoint, "$.engine"), endpoint, "$.engine.mode"),
    architecture: nonEmptyString(required(engine, "architecture", endpoint, "$.engine"), endpoint, "$.engine.architecture"),
    device: optionalString(engine, "device", endpoint, "$.engine"),
    backend: optionalString(engine, "backend", endpoint, "$.engine"),
    protocolVersion: optionalString(engine, "protocol_version", endpoint, "$.engine"),
    engineVersion: optionalString(engine, "engine_version", endpoint, "$.engine"),
    buildId: optionalString(engine, "build_id", endpoint, "$.engine"),
    modelSha256: "model_sha256" in engine ? sha256(engine.model_sha256, endpoint, "$.engine.model_sha256") : undefined,
    contextTokens: optionalInteger("n_ctx"),
    gpuLayers: optionalInteger("gpu_layers"),
    capabilities,
  };
}

export function decodeRuntimeHealth(value: unknown, endpoint = "/engine/health"): RuntimeHealth {
  const document = object(value, endpoint, "$");
  closed(document, ["engine"], endpoint, "$");
  const engine = object(required(document, "engine", endpoint, "$"), endpoint, "$.engine");
  return engine.managed === true ? decodeManagedRuntime(engine, endpoint) : decodeWorkerRuntime(engine, endpoint);
}
