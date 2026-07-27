export interface EngineModel {
  architecture: string;
  model: string;
  modelName: string;
  sha256?: string;
  quant?: string;
  device?: string;
  mode?: string;
  protocolVersion?: string;
  context?: number;
  embedding?: number;
  layers?: number;
  vocabulary?: number;
  gpuLayers?: number;
  capabilities: Record<string, boolean>;
}

export interface ModelAxis {
  name: string;
  value: number;
  calibrated: boolean;
}

export interface LocalModel {
  path: string;
  filename: string;
  sizeBytes?: number;
  quant?: string;
  sha256?: string;
}

export interface ModelWorkspaceData {
  engine?: EngineModel;
  axes: ModelAxis[];
  localModels?: LocalModel[];
  activeProfile?: string;
  errors: Partial<Record<"engine" | "axes" | "inventory" | "profiles", string>>;
}

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function basename(value: unknown) {
  return String(value || "").split(/[\\/]/).pop() || "—";
}

function quantFromName(value: unknown) {
  const match = String(value || "").match(/(IQ\d+[A-Z0-9_]*|Q\d+(?:_[A-Z0-9]+)+|Q\d+|BF16|F16|F32)/i);
  return match?.[1].toUpperCase();
}

function errorText(body: JsonRecord, status: number) {
  if (typeof body.error === "string") return body.error;
  const nested = record(body.error);
  if (typeof nested.message === "string") return nested.message;
  return `Request failed (${status})`;
}

async function request(url: string, options: RequestInit = {}): Promise<JsonRecord> {
  const response = await fetch(url, options);
  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // The HTTP status remains authoritative when the route returns no JSON.
  }
  if (!response.ok || body.error) throw new Error(errorText(body, response.status));
  return body;
}

function get(url: string, signal?: AbortSignal) {
  return request(url, { signal });
}

function post(url: string, body: JsonRecord, signal?: AbortSignal) {
  return request(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

function engineFromBody(body: JsonRecord): EngineModel | undefined {
  const engine = record(body.engine);
  if (!Object.keys(engine).length) return undefined;
  const capabilities = Object.fromEntries(
    Object.entries(record(engine.capabilities)).map(([name, value]) => [name, value === true]),
  );
  const finite = (value: unknown) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  };
  return {
    architecture: String(engine.architecture || "—"),
    model: String(engine.model || ""),
    modelName: basename(engine.model),
    sha256: typeof engine.model_sha256 === "string" ? engine.model_sha256 : undefined,
    quant: quantFromName(engine.model),
    device: typeof engine.device === "string" ? engine.device : undefined,
    mode: typeof engine.mode === "string" ? engine.mode : undefined,
    protocolVersion: typeof engine.protocol_version === "string" ? engine.protocol_version : undefined,
    context: finite(engine.n_ctx),
    embedding: finite(engine.n_embd),
    layers: finite(engine.n_layer),
    vocabulary: finite(engine.vocab_size),
    gpuLayers: finite(engine.gpu_layers),
    capabilities,
  };
}

function localModelsFromBody(body: JsonRecord): LocalModel[] {
  return records(body.models).map((model) => {
    const size = Number(model.size_bytes);
    return {
      path: String(model.path || ""),
      filename: String(model.filename || basename(model.path)),
      sizeBytes: Number.isFinite(size) ? size : undefined,
      quant: typeof model.quant === "string" ? model.quant : quantFromName(model.filename),
      sha256: typeof model.sha256 === "string" ? model.sha256 : undefined,
    };
  }).filter((model) => model.path || model.filename);
}

function message(reason: unknown, fallback: string) {
  return reason instanceof Error ? reason.message : fallback;
}

export async function loadModelWorkspace(signal?: AbortSignal): Promise<ModelWorkspaceData> {
  const results = await Promise.allSettled([
    get("/engine/health", signal),
    post("/steer/axes", {}, signal),
    get("/models/local", signal),
    get("/profiles/list", signal),
  ]);
  const [engine, axes, inventory, profiles] = results;
  const errors: ModelWorkspaceData["errors"] = {};
  if (engine.status === "rejected") errors.engine = message(engine.reason, "Engine health unavailable");
  if (axes.status === "rejected") errors.axes = message(axes.reason, "Steering axes unavailable");
  if (inventory.status === "rejected") errors.inventory = message(inventory.reason, "Model inventory unavailable");
  if (profiles.status === "rejected") errors.profiles = message(profiles.reason, "Profiles unavailable");

  const profileBody = profiles.status === "fulfilled" ? profiles.value : {};
  return {
    engine: engine.status === "fulfilled" ? engineFromBody(engine.value) : undefined,
    axes: axes.status === "fulfilled"
      ? records(axes.value.axes).map((axis) => ({
          name: String(axis.name || ""),
          value: Number(axis.value) || 0,
          calibrated: axis.calibrated === true,
        })).filter((axis) => axis.name)
      : [],
    localModels: inventory.status === "fulfilled" ? localModelsFromBody(inventory.value) : undefined,
    activeProfile: typeof profileBody.active === "string" ? profileBody.active : undefined,
    errors,
  };
}
