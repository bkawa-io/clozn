import { HttpError } from "./client";
import { ContractError } from "./contracts";
import {
  decodeContextInvestigationQuery,
  decodeContextInvestigationReader,
  type ContextInvestigationQuery,
  type ContextInvestigationReader,
} from "./contextInvestigation";
import {
  decodeContextEffectsJob,
  type ContextEffectsJob,
} from "./contextInvestigationJobs";

export interface ContextInvestigationQueryRequest {
  answerStart: number;
  answerEnd: number;
  floorNats?: number;
}

export interface ContextEffectsRequest {
  sourceIds?: readonly string[];
}

export class ContextInvestigationHttpError extends HttpError {
  constructor(
    endpoint: string,
    status: number,
    statusText: string,
    readonly code: string,
    readonly serverMessage: string,
  ) {
    super(endpoint, status, statusText);
    this.message = `${serverMessage} (${code}; ${status} ${statusText})`;
  }
}

function runPath(runId: string, suffix: string): string {
  if (!runId.trim()) throw new ContractError("client", "run id must not be blank");
  return `/runs/${encodeURIComponent(runId)}/context-investigation/${suffix}`;
}

async function responseJson(
  response: Response,
  endpoint: string,
  acceptedStatuses: readonly number[] = [200],
): Promise<unknown> {
  if (!acceptedStatuses.includes(response.status)) {
    try {
      const body: unknown = await response.json();
      if (body !== null && typeof body === "object" && !Array.isArray(body)) {
        const error = (body as Record<string, unknown>).error;
        const code = (body as Record<string, unknown>).code;
        if (typeof error === "string" && error.trim() && typeof code === "string" && code.trim()) {
          throw new ContextInvestigationHttpError(
            endpoint, response.status, response.statusText, code, error,
          );
        }
      }
    } catch (error) {
      if (error instanceof ContextInvestigationHttpError) throw error;
    }
    throw new HttpError(endpoint, response.status, response.statusText);
  }
  try {
    return await response.json();
  } catch {
    throw new ContractError(endpoint, "response is not valid JSON");
  }
}

function identifier(value: string, name: string): string {
  if (!value.trim()) throw new ContractError("client", `${name} must not be blank`);
  return value;
}

function effectsBody(request: ContextEffectsRequest): Record<string, unknown> {
  if (request.sourceIds === undefined) return {};
  const sourceIds = [...request.sourceIds];
  if (
    !sourceIds.length
    || sourceIds.some((sourceId) => !sourceId.trim())
    || new Set(sourceIds).size !== sourceIds.length
  ) {
    throw new ContractError("client", "sourceIds must contain unique non-empty source IDs");
  }
  return { source_ids: sourceIds };
}

function validateQueryRequest(request: ContextInvestigationQueryRequest): void {
  if (
    !Number.isSafeInteger(request.answerStart) || !Number.isSafeInteger(request.answerEnd)
    || request.answerStart < 0 || request.answerEnd <= request.answerStart
  ) throw new ContractError("client", "answerStart and answerEnd must be a non-empty half-open Unicode code-point range");
  if (request.floorNats !== undefined && (!Number.isFinite(request.floorNats) || request.floorNats < 0)) {
    throw new ContractError("client", "floorNats must be a finite non-negative number");
  }
}

export const contextInvestigationApi = {
  async reader(runId: string, signal?: AbortSignal): Promise<ContextInvestigationReader> {
    const base = runPath(runId, "reader");
    const endpoint = `${base}?floor_nats=0.1`;
    const response = await fetch(endpoint, {
      method: "GET", signal, headers: { Accept: "application/json" },
    });
    return decodeContextInvestigationReader(
      await responseJson(response, endpoint), endpoint, runId,
    );
  },

  async query(
    runId: string,
    request: ContextInvestigationQueryRequest,
    signal?: AbortSignal,
  ): Promise<ContextInvestigationQuery> {
    validateQueryRequest(request);
    const endpoint = runPath(runId, "query");
    const body: Record<string, number> = {
      answer_start: request.answerStart,
      answer_end: request.answerEnd,
    };
    if (request.floorNats !== undefined) body.floor_nats = request.floorNats;
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const decoded = decodeContextInvestigationQuery(
      await responseJson(response, endpoint), endpoint,
    );
    if (decoded.selection && (
      decoded.selection.unicodeRange[0] !== request.answerStart
      || decoded.selection.unicodeRange[1] !== request.answerEnd
    )) throw new ContractError(endpoint, "$.selection.unicode_range does not match the requested answer range");
    if (request.floorNats !== undefined && decoded.measurementFloorNats !== request.floorNats) {
      throw new ContractError(endpoint, "$.measurement_floor_nats does not match the requested measurement floor");
    }
    return decoded;
  },

  async startEffects(
    runId: string,
    request: ContextEffectsRequest = {},
    signal?: AbortSignal,
  ): Promise<ContextEffectsJob> {
    const endpoint = runPath(runId, "effects/jobs");
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(effectsBody(request)),
    });
    return decodeContextEffectsJob(
      await responseJson(response, endpoint, [202]), endpoint, runId,
    );
  },

  async effectsJob(
    runId: string,
    jobId: string,
    signal?: AbortSignal,
  ): Promise<ContextEffectsJob> {
    identifier(jobId, "job id");
    const endpoint = runPath(runId, `effects/jobs/${encodeURIComponent(jobId)}`);
    const response = await fetch(endpoint, {
      method: "GET", signal, headers: { Accept: "application/json" },
    });
    return decodeContextEffectsJob(
      await responseJson(response, endpoint), endpoint, runId,
    );
  },

  async cancelEffects(
    runId: string,
    jobId: string,
    signal?: AbortSignal,
  ): Promise<ContextEffectsJob> {
    identifier(jobId, "job id");
    const endpoint = runPath(runId, `effects/jobs/${encodeURIComponent(jobId)}/cancel`);
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: "{}",
    });
    return decodeContextEffectsJob(
      await responseJson(response, endpoint), endpoint, runId,
    );
  },
};
