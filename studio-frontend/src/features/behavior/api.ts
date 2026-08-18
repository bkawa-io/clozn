export interface SamplingSettings {
  sampling: boolean;
  sample_temperature: number;
  sample_top_p: number;
  sample_top_k: number;
  sample_repeat_penalty: number;
}

export interface BehaviorWorkspaceData {
  sampling?: SamplingSettings;
  errors: Partial<Record<"sampling", string>>;
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

function samplingFromBody(body: JsonRecord): SamplingSettings {
  return {
    sampling: body.sampling === true,
    sample_temperature: Number(body.sample_temperature) || 0,
    sample_top_p: Number(body.sample_top_p) || 0,
    sample_top_k: Number(body.sample_top_k) || 0,
    sample_repeat_penalty: Number(body.sample_repeat_penalty) || 0,
  };
}

export async function loadBehaviorWorkspace(signal?: AbortSignal): Promise<BehaviorWorkspaceData> {
  try {
    return { sampling: samplingFromBody(await get("/sampling/mode", signal)), errors: {} };
  } catch (error) {
    return {
      sampling: undefined,
      errors: { sampling: error instanceof Error ? error.message : "Sampling unavailable" },
    };
  }
}

export async function saveSampling(settings: SamplingSettings): Promise<SamplingSettings> {
  return samplingFromBody(await post("/sampling/mode", settings as unknown as JsonRecord));
}


