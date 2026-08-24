import {
  ContractError,
  decodeContextTension,
  decodeJobSnapshot,
  decodeInfluenceQuery,
  decodeJsonObject,
  decodeLiveness,
  decodeMinimalContextResult,
  decodeMinimalContextResultsDocument,
  decodeReadiness,
  decodeRewindFidelity,
  decodeRunComparison,
  decodeRunDocument,
  decodeRunsDocument,
  decodeRuntimeModels,
  decodeSpanAddressDocument,
  decodeSuggestedBreakpoints,
  type ContextTension,
  type JobSnapshot,
  type HealthStatus,
  type InfluenceQuery,
  type JsonObject,
  type MinimalContextResult,
  type MinimalContextResultSummary,
  type RunRecord,
  type RunComparison,
  type RewindFidelity,
  type RuntimeModels,
  type SpanAddressDocument,
  type SuggestedBreakpoints,
} from "./contracts";

export class HttpError extends Error {
  readonly name = "HttpError";
  constructor(readonly endpoint: string, readonly status: number, readonly statusText: string) {
    super(`Request to ${endpoint} failed (${status} ${statusText})`);
  }
}

async function getJson(endpoint: string, signal?: AbortSignal, acceptedStatuses: readonly number[] = [200]): Promise<unknown> {
  const response = await fetch(endpoint, { method: "GET", signal, headers: { Accept: "application/json" } });
  if (!acceptedStatuses.includes(response.status)) throw new HttpError(endpoint, response.status, response.statusText);
  try {
    return await response.json();
  } catch {
    throw new ContractError(endpoint, "response is not valid JSON");
  }
}

async function postJson(endpoint: string, body: JsonObject, signal?: AbortSignal, acceptedStatuses: readonly number[] = [200]): Promise<unknown> {
  const response = await fetch(endpoint, { method: "POST", signal, headers: { Accept: "application/json", "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!acceptedStatuses.includes(response.status)) {
    let statusText = response.statusText;
    try {
      const failure = await response.json() as { error?: unknown; code?: unknown };
      if (typeof failure.error === "string") statusText = `${statusText}: ${failure.error}${typeof failure.code === "string" ? ` (${failure.code})` : ""}`;
    } catch { /* Preserve the ordinary HTTP status when an error body is not JSON. */ }
    throw new HttpError(endpoint, response.status, statusText);
  }
  try {
    return await response.json();
  } catch {
    throw new ContractError(endpoint, "response is not valid JSON");
  }
}

function runPath(runId: string, suffix = ""): string {
  if (!runId.trim()) throw new ContractError("client", "run id must not be blank");
  return `/runs/${encodeURIComponent(runId)}${suffix}`;
}

function unicodeRange(start: number, end: number): URLSearchParams {
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start) {
    throw new ContractError("client", "range must be non-negative integer Unicode code-point offsets with end > start");
  }
  return new URLSearchParams({ start: String(start), end: String(end) });
}

export const studioApi = {
  async health(signal?: AbortSignal): Promise<HealthStatus> { return decodeLiveness(await getJson("/healthz", signal)); },
  // Readiness deliberately returns a typed 503 `not_ready` document while liveness remains healthy.
  async readiness(signal?: AbortSignal): Promise<HealthStatus> { return decodeReadiness(await getJson("/readyz", signal, [200, 503])); },
  async runtimeModels(signal?: AbortSignal): Promise<RuntimeModels> { return decodeRuntimeModels(await getJson("/runtime/models", signal)); },
  async runs(signal?: AbortSignal): Promise<readonly RunRecord[]> { return decodeRunsDocument(await getJson("/runs", signal)); },
  async run(runId: string, signal?: AbortSignal): Promise<RunRecord> {
    const endpoint = runPath(runId);
    return decodeRunDocument(await getJson(endpoint, signal), endpoint);
  },
  async family(runId: string, signal?: AbortSignal): Promise<readonly RunRecord[]> {
    const endpoint = runPath(runId, "/family");
    return decodeRunsDocument(await getJson(endpoint, signal), endpoint);
  },
  async compare(runA: string, runB: string, signal?: AbortSignal): Promise<RunComparison> {
    if (!runA.trim() || !runB.trim()) throw new ContractError("client", "comparison requires two non-blank run ids");
    const query = new URLSearchParams({ a: runA, b: runB });
    const endpoint = `/runs/compare?${query}`;
    return decodeRunComparison(await getJson(endpoint, signal), endpoint);
  },
  /** start/end are Unicode code-point, half-open offsets — never UTF-16 string indexes. */
  async influenceQuery(runId: string, range: { start: number; end: number; limit?: number }, signal?: AbortSignal): Promise<InfluenceQuery> {
    if (range.limit !== undefined && (!Number.isInteger(range.limit) || range.limit < 1 || range.limit > 50)) throw new ContractError("client", "influence limit must be an integer from 1 to 50");
    const query = unicodeRange(range.start, range.end);
    if (range.limit !== undefined) query.set("limit", String(range.limit));
    const endpoint = `${runPath(runId, "/influence-query")}?${query}`;
    return decodeInfluenceQuery(await getJson(endpoint, signal), endpoint);
  },
  async contextTension(runId: string, options: { start?: number; end?: number; limit?: number } = {}, signal?: AbortSignal): Promise<ContextTension> {
    if ((options.start === undefined) !== (options.end === undefined)) throw new ContractError("client", "context tension range requires both start and end");
    if (options.limit !== undefined && (!Number.isInteger(options.limit) || options.limit < 1 || options.limit > 100)) throw new ContractError("client", "context tension limit must be an integer from 1 to 100");
    const query = options.start === undefined ? new URLSearchParams() : unicodeRange(options.start, options.end as number);
    if (options.limit !== undefined) query.set("limit", String(options.limit));
    const base = runPath(runId, "/context-tension");
    const endpoint = query.size ? `${base}?${query}` : base;
    return decodeContextTension(await getJson(endpoint, signal), endpoint);
  },
  async spanAddresses(runId: string, signal?: AbortSignal): Promise<SpanAddressDocument> {
    const endpoint = runPath(runId, "/span-addresses");
    return decodeSpanAddressDocument(await getJson(endpoint, signal), endpoint);
  },
  async rewindFidelity(runId: string, signal?: AbortSignal): Promise<RewindFidelity> {
    const endpoint = runPath(runId, "/rewind-fidelity");
    return decodeRewindFidelity(await getJson(endpoint, signal), endpoint);
  },
  async suggestedBreakpoints(runId: string, signal?: AbortSignal): Promise<SuggestedBreakpoints> {
    const endpoint = `${runPath(runId, "/suggested-breakpoints")}?limit=50`;
    return decodeSuggestedBreakpoints(await getJson(endpoint, signal), endpoint);
  },
  async minimalContextResults(runId: string, signal?: AbortSignal): Promise<readonly MinimalContextResultSummary[]> {
    const endpoint = runPath(runId, "/minimal-context/results");
    return decodeMinimalContextResultsDocument(await getJson(endpoint, signal), endpoint);
  },
  async minimalContextResult(runId: string, resultId: string, signal?: AbortSignal): Promise<MinimalContextResult> {
    if (!resultId.trim()) throw new ContractError("client", "Minimal Context result id must not be blank");
    const endpoint = runPath(runId, `/minimal-context/results/${encodeURIComponent(resultId)}`);
    return decodeMinimalContextResult(await getJson(endpoint, signal), endpoint);
  },
  async startMinimalContext(runId: string, signal?: AbortSignal): Promise<JobSnapshot<MinimalContextResult>> {
    const endpoint = runPath(runId, "/minimal-context/jobs");
    return decodeJobSnapshot(await postJson(endpoint, {}, signal, [200, 202]), endpoint, decodeMinimalContextResult);
  },
  async minimalContextJob(runId: string, jobId: string, signal?: AbortSignal): Promise<JobSnapshot<MinimalContextResult>> {
    if (!jobId.trim()) throw new ContractError("client", "Minimal Context job id must not be blank");
    const endpoint = runPath(runId, `/minimal-context/jobs/${encodeURIComponent(jobId)}`);
    return decodeJobSnapshot(await getJson(endpoint, signal), endpoint, decodeMinimalContextResult);
  },
  async cancelMinimalContext(runId: string, jobId: string, signal?: AbortSignal): Promise<JobSnapshot<MinimalContextResult>> {
    if (!jobId.trim()) throw new ContractError("client", "Minimal Context job id must not be blank");
    const endpoint = runPath(runId, `/minimal-context/jobs/${encodeURIComponent(jobId)}/cancel`);
    return decodeJobSnapshot(await postJson(endpoint, {}, signal), endpoint, decodeMinimalContextResult);
  },
  async testThis(runId: string, request: JsonObject, signal?: AbortSignal): Promise<JsonObject> {
    const endpoint = runPath(runId, "/test-this");
    return decodeJsonObject(await postJson(endpoint, request, signal, [201, 409, 422]), endpoint);
  },
};
