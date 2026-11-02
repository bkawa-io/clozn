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
import {
  decodeContextInteractionsJob,
  type ContextInteractionsJob,
} from "./contextInteractions";
import { decodeLeanContextJob, type LeanContextJob } from "./leanContext";

export interface ContextInvestigationQueryRequest {
  answerStart: number;
  answerEnd: number;
  floorNats?: number;
  sourceIds?: readonly string[];
  granularity?: EffectsGranularity;
  maxArms?: number;
  scoringMode?: ScoringMode;
}

export type EffectsGranularity = "message" | "adaptive";
export type ScoringMode = "fast" | "exact";

export interface ContextEffectsRequest {
  sourceIds?: readonly string[];
  granularity?: EffectsGranularity;
  maxArms?: number;
  scoringMode?: ScoringMode;
}

export interface ContextInteractionsRequest {
  maxNewExecutions: number;
  maxPairs: number;
  seed: number;
}

export interface LeanContextRequest {
  maxNewExecutions: number;
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
  const body: Record<string, unknown> = {};
  if (request.scoringMode !== undefined) {
    if (request.scoringMode !== "fast" && request.scoringMode !== "exact") {
      throw new ContractError("client", "scoringMode must be fast or exact");
    }
    body.scoring_mode = request.scoringMode;
  }
  if (request.sourceIds !== undefined) {
    const sourceIds = [...request.sourceIds];
    if (
      !sourceIds.length
      || sourceIds.some((sourceId) => !sourceId.trim())
      || new Set(sourceIds).size !== sourceIds.length
    ) {
      throw new ContractError("client", "sourceIds must contain unique non-empty source IDs");
    }
    body.source_ids = sourceIds;
  }
  const configured = request.granularity !== undefined || request.maxArms !== undefined;
  if (configured) {
    if (
      (request.granularity !== "message" && request.granularity !== "adaptive")
      || !Number.isSafeInteger(request.maxArms)
      || request.maxArms === undefined
      || request.maxArms < 1
    ) throw new ContractError("client", "granularity and a positive integer maxArms are required together");
    if (request.sourceIds !== undefined) {
      throw new ContractError("client", "sourceIds cannot be combined with granularity controls");
    }
    body.granularity = request.granularity;
    body.max_arms = request.maxArms;
  }
  return body;
}

function interactionsBody(request: ContextInteractionsRequest): Record<string, number> {
  const values = [
    ["maxNewExecutions", request.maxNewExecutions, 0, 64],
    ["maxPairs", request.maxPairs, 1, 256],
    ["seed", request.seed, 0, 2 ** 31 - 1],
  ] as const;
  for (const [name, value, minimum, maximum] of values) {
    if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
      throw new ContractError("client", `${name} must be an integer between ${minimum} and ${maximum}`);
    }
  }
  return {
    max_new_executions: request.maxNewExecutions,
    max_pairs: request.maxPairs,
    seed: request.seed,
  };
}

function readerEndpoint(base: string, request?: ContextEffectsRequest): string {
  const query = new URLSearchParams({ floor_nats: "0.1" });
  if (request) {
    const body = effectsBody(request);
    if (Array.isArray(body.source_ids)) {
      for (const sourceId of body.source_ids) query.append("source_id", String(sourceId));
    }
    if (typeof body.granularity === "string") query.set("granularity", body.granularity);
    if (typeof body.max_arms === "number") query.set("max_arms", String(body.max_arms));
    if (typeof body.scoring_mode === "string") query.set("scoring_mode", body.scoring_mode);
  }
  return `${base}?${query.toString()}`;
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
  async reader(
    runId: string,
    signal?: AbortSignal,
    configuration?: ContextEffectsRequest,
  ): Promise<ContextInvestigationReader> {
    const base = runPath(runId, "reader");
    const endpoint = readerEndpoint(base, configuration);
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
    const body: Record<string, unknown> = {
      answer_start: request.answerStart,
      answer_end: request.answerEnd,
    };
    if (request.floorNats !== undefined) body.floor_nats = request.floorNats;
    if (
      request.sourceIds !== undefined
      || request.granularity !== undefined
      || request.maxArms !== undefined
      || request.scoringMode !== undefined
    ) {
      const configuration = effectsBody({
        sourceIds: request.sourceIds,
        granularity: request.granularity,
        maxArms: request.maxArms,
        scoringMode: request.scoringMode,
      });
      Object.assign(body, configuration);
    }
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

  async startLeanContext(
    runId: string,
    request: LeanContextRequest = { maxNewExecutions: 16 },
    signal?: AbortSignal,
  ): Promise<LeanContextJob> {
    if (!Number.isInteger(request.maxNewExecutions) || request.maxNewExecutions < 1 || request.maxNewExecutions > 64) {
      throw new ContractError("client", "maxNewExecutions must be an integer from 1 to 64");
    }
    const endpoint = runPath(runId, "lean-context/jobs");
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({ max_new_executions: request.maxNewExecutions }),
    });
    return decodeLeanContextJob(
      await responseJson(response, endpoint, [202]), endpoint, runId,
    );
  },

  async leanContextJob(
    runId: string,
    jobId: string,
    signal?: AbortSignal,
  ): Promise<LeanContextJob> {
    identifier(jobId, "job id");
    const endpoint = runPath(runId, `lean-context/jobs/${encodeURIComponent(jobId)}`);
    const response = await fetch(endpoint, {
      method: "GET", signal, headers: { Accept: "application/json" },
    });
    return decodeLeanContextJob(await responseJson(response, endpoint), endpoint, runId);
  },

  async cancelLeanContext(
    runId: string,
    jobId: string,
    signal?: AbortSignal,
  ): Promise<LeanContextJob> {
    identifier(jobId, "job id");
    const endpoint = runPath(runId, `lean-context/jobs/${encodeURIComponent(jobId)}/cancel`);
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: "{}",
    });
    return decodeLeanContextJob(await responseJson(response, endpoint), endpoint, runId);
  },

  async startInteractions(
    runId: string,
    request: ContextInteractionsRequest,
    signal?: AbortSignal,
  ): Promise<ContextInteractionsJob> {
    const endpoint = runPath(runId, "interactions/jobs");
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(interactionsBody(request)),
    });
    return decodeContextInteractionsJob(
      await responseJson(response, endpoint, [202]), endpoint, runId,
    );
  },

  async interactionsJob(
    runId: string,
    jobId: string,
    signal?: AbortSignal,
  ): Promise<ContextInteractionsJob> {
    identifier(jobId, "job id");
    const endpoint = runPath(runId, `interactions/jobs/${encodeURIComponent(jobId)}`);
    const response = await fetch(endpoint, {
      method: "GET", signal, headers: { Accept: "application/json" },
    });
    return decodeContextInteractionsJob(
      await responseJson(response, endpoint), endpoint, runId,
    );
  },

  async cancelInteractions(
    runId: string,
    jobId: string,
    signal?: AbortSignal,
  ): Promise<ContextInteractionsJob> {
    identifier(jobId, "job id");
    const endpoint = runPath(runId, `interactions/jobs/${encodeURIComponent(jobId)}/cancel`);
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: "{}",
    });
    return decodeContextInteractionsJob(
      await responseJson(response, endpoint), endpoint, runId,
    );
  },
};
