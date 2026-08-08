export interface BehaviorAxis {
  name: string;
  poles: [string, string];
  value: number;
  max: number;
  calibrated: boolean;
  library?: boolean;
  custom?: boolean;
}

export interface SamplingSettings {
  sampling: boolean;
  sample_temperature: number;
  sample_top_p: number;
  sample_top_k: number;
  sample_repeat_penalty: number;
}

export interface GuardSettings {
  enabled: boolean;
  guard: {
    concepts?: string[];
    threshold?: number;
    counter_strength?: number;
    max_fires?: number;
    layer?: number;
  } | null;
}

export interface BehaviorWorkspaceData {
  axes: BehaviorAxis[];
  sampling?: SamplingSettings;
  guard?: GuardSettings;
  errors: Partial<Record<"axes" | "sampling" | "guard", string>>;
}

export interface AxisPreview {
  prompt: string;
  axis: string;
  value: number;
  baseline: string;
  steered: string;
  warning?: string;
}

export interface ConceptPreview {
  prompt: string;
  concept: string;
  strength: number;
  baseline: string;
  steered?: string;
  note?: string;
}

// The clozn.corrective-flow.v1 client (types + preview/confirm/keep/undo/source-use functions) moved to
// src/data/correctiveFlow.ts for D5 (the Lens guided-repair panel needs the identical client; this
// codebase's convention is that cross-feature-reusable API clients live under src/data/, never imported
// feature-to-feature -- see that module's own doc comment). Re-exported here verbatim so every existing
// import in this file (and in Behavior.tsx) keeps resolving unchanged.
export * from "../../data/correctiveFlow";

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function errorText(body: JsonRecord, status: number) {
  const error = body.error;
  if (typeof error === "string") return error;
  const nested = record(error);
  if (typeof nested.message === "string") return nested.message;
  if (typeof body.blocked === "string") return body.blocked;
  if (typeof body.reason === "string") return body.reason;
  if (typeof body.note === "string") return body.note;
  return `Request failed (${status})`;
}

async function request(
  url: string,
  options: RequestInit = {},
): Promise<JsonRecord> {
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

async function get(url: string, signal?: AbortSignal) {
  return request(url, { signal });
}

async function post(url: string, body: JsonRecord, signal?: AbortSignal) {
  return request(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

function axisFromBody(axis: JsonRecord): BehaviorAxis | null {
  const name = String(axis.name || "");
  const rawPoles = Array.isArray(axis.poles) ? axis.poles.map(String) : [];
  if (!name || rawPoles.length !== 2) return null;
  const max = Number(axis.max);
  const value = Number(axis.value);
  return {
    name,
    poles: [rawPoles[0], rawPoles[1]],
    max: Number.isFinite(max) && max > 0 ? max : 1.5,
    value: Number.isFinite(value) ? value : 0,
    calibrated: axis.calibrated === true,
    library: axis.library === true,
    custom: axis.custom === true,
  };
}

function samplingFromBody(body: JsonRecord): SamplingSettings {
  return {
    sampling: body.sampling === true,
    sample_temperature: Number(body.sample_temperature) || 0,
    sample_top_p: Number(body.sample_top_p) || 0,
    sample_top_k: Number(body.sample_top_k) || 0,
    sample_repeat_penalty: Number(body.sample_repeat_penalty) || 0,
  };
}

function guardFromBody(body: JsonRecord): GuardSettings {
  const source = record(body.guard);
  const concepts = Array.isArray(source.concepts)
    ? source.concepts.map(String).filter(Boolean)
    : undefined;
  const numberOrUndefined = (value: unknown) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  };
  return {
    enabled: body.enabled === true,
    guard: body.guard && typeof body.guard === "object"
      ? {
          concepts,
          threshold: numberOrUndefined(source.threshold),
          counter_strength: numberOrUndefined(source.counter_strength),
          max_fires: numberOrUndefined(source.max_fires),
          layer: numberOrUndefined(source.layer),
        }
      : null,
  };
}

export async function loadBehaviorWorkspace(signal?: AbortSignal): Promise<BehaviorWorkspaceData> {
  const results = await Promise.allSettled([
    post("/steer/axes", {}, signal),
    get("/sampling/mode", signal),
    get("/guard/mode", signal),
  ]);
  const [axes, sampling, guard] = results;
  const errors: BehaviorWorkspaceData["errors"] = {};
  if (axes.status === "rejected") errors.axes = axes.reason instanceof Error ? axes.reason.message : "Axes unavailable";
  if (sampling.status === "rejected") errors.sampling = sampling.reason instanceof Error ? sampling.reason.message : "Sampling unavailable";
  if (guard.status === "rejected") errors.guard = guard.reason instanceof Error ? guard.reason.message : "Guard unavailable";

  return {
    axes: axes.status === "fulfilled"
      ? records(axes.value.axes).map(axisFromBody).filter((axis): axis is BehaviorAxis => axis !== null)
      : [],
    sampling: sampling.status === "fulfilled" ? samplingFromBody(sampling.value) : undefined,
    guard: guard.status === "fulfilled" ? guardFromBody(guard.value) : undefined,
    errors,
  };
}

export async function applyAxis(name: string, value: number) {
  const body = await post("/steer/set", { name, value });
  return {
    active: Object.fromEntries(
      Object.entries(record(body.active))
        .map(([axis, next]) => [axis, Number(next)])
        .filter((entry) => Number.isFinite(entry[1])),
    ),
    warning: typeof body.warning === "string" ? body.warning : undefined,
  };
}

export async function previewAxis(name: string, value: number, prompt: string): Promise<AxisPreview> {
  const body = await post("/steer/check", { name, value, prompt });
  return {
    prompt: String(body.prompt || prompt),
    axis: String(body.axis || name),
    value: Number(body.value) || value,
    baseline: String(body.baseline || ""),
    steered: String(body.steered || ""),
    warning: typeof body.warning === "string" ? body.warning : undefined,
  };
}

export async function applyConcept(concept: string, strength: number) {
  const body = await post("/steer/concept/set", { concept, strength });
  if (body.ok !== true) throw new Error(errorText(body, 200));
  return Object.fromEntries(
    Object.entries(record(body.active))
      .map(([name, value]) => [name, Number(value)])
      .filter((entry) => Number.isFinite(entry[1])),
  );
}

export async function previewConcept(
  concept: string,
  strength: number,
  prompt: string,
): Promise<ConceptPreview> {
  const body = await post("/steer/concept/check", { concept, strength, prompt });
  if (body.steered == null) throw new Error(errorText(body, 200));
  return {
    prompt: String(body.prompt || prompt),
    concept: String(body.concept || concept),
    strength: Number(body.strength) || strength,
    baseline: String(body.baseline || ""),
    steered: String(body.steered || ""),
    note: typeof body.note === "string" ? body.note : undefined,
  };
}

export async function saveSampling(settings: SamplingSettings): Promise<SamplingSettings> {
  return samplingFromBody(await post("/sampling/mode", settings as unknown as JsonRecord));
}

export async function saveGuard(settings: GuardSettings): Promise<GuardSettings> {
  const body = await post("/guard/mode", {
    enabled: settings.enabled,
    concepts: settings.guard?.concepts ?? [],
    ...(settings.guard?.threshold == null ? {} : { threshold: settings.guard.threshold }),
    ...(settings.guard?.counter_strength == null
      ? {}
      : { counter_strength: settings.guard.counter_strength }),
    ...(settings.guard?.max_fires == null ? {} : { max_fires: settings.guard.max_fires }),
    ...(settings.guard?.layer == null ? {} : { layer: settings.guard.layer }),
  });
  return guardFromBody(body);
}

