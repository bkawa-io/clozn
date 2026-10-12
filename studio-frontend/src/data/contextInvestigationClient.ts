import { HttpError } from "./client";
import { ContractError } from "./contracts";
import {
  decodeContextInvestigationQuery,
  decodeContextInvestigationReader,
  type ContextInvestigationQuery,
  type ContextInvestigationReader,
} from "./contextInvestigation";
import {
  decodeEffectsSearchJob,
  type EffectsSearchBudget,
  type EffectsSearchExecutionPolicy,
  type EffectsSearchJob,
} from "./contextInvestigationJobs";
import {
  decodeContextCounterfactualJob,
  decodeCounterfactualMaterializationResult,
  decodeRemoveSourceTestJob,
  type ContextCounterfactualJob,
  type CounterfactualMaterializationRequest,
  type CounterfactualMaterializationResult,
  type RemoveSourceTestJob,
} from "./contextInvestigationActions";

export interface ContextInvestigationQueryRequest {
  answerStart: number;
  answerEnd: number;
  floorNats?: number;
}

export interface EffectsSearchRequest {
  sourceIds?: readonly string[];
  coarseGroupSize?: number;
  refineThresholdNats?: number;
  drilldownSourceIds?: readonly string[];
  jointSourceSets?: readonly (readonly string[])[];
  budget: EffectsSearchBudget;
  executionPolicy: EffectsSearchExecutionPolicy;
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

function singleSourceBody(sourceId: string): string {
  return JSON.stringify({ source_id: identifier(sourceId, "source id") });
}

function materializationBody(request: CounterfactualMaterializationRequest): string {
  identifier(request.experimentId, "experiment id");
  identifier(request.armId, "arm id");
  identifier(request.observationId, "observation id");
  return JSON.stringify({
    experiment_id: request.experimentId,
    arm_id: request.armId,
    observation_id: request.observationId,
  });
}

function positiveInteger(value: number | undefined, name: string): void {
  if (value !== undefined && (!Number.isSafeInteger(value) || value < 1)) {
    throw new ContractError("client", `${name} must be a positive integer`);
  }
}

function sourceIds(values: readonly string[] | undefined, name: string, minimum = 0): string[] | undefined {
  if (values === undefined) return undefined;
  const result = [...values];
  if (result.length < minimum || result.some((value) => !value.trim()) || new Set(result).size !== result.length) {
    throw new ContractError("client", `${name} must contain ${minimum ? `at least ${minimum} ` : ""}unique non-empty source IDs`);
  }
  return result;
}

function effectsSearchBody(request: EffectsSearchRequest): Record<string, unknown> {
  positiveInteger(request.coarseGroupSize, "coarseGroupSize");
  if (request.refineThresholdNats !== undefined
      && (!Number.isFinite(request.refineThresholdNats) || request.refineThresholdNats < 0)) {
    throw new ContractError("client", "refineThresholdNats must be a finite non-negative number");
  }
  const budgetValues = [request.budget.maxCoarseArms, request.budget.maxFineArms, request.budget.maxJointArms];
  if (budgetValues.some((value) => !Number.isSafeInteger(value) || value < 0)) {
    throw new ContractError("client", "effects-search budgets must be non-negative integers");
  }
  const sources = sourceIds(request.sourceIds, "sourceIds", request.sourceIds === undefined ? 0 : 1);
  const drilldowns = sourceIds(request.drilldownSourceIds, "drilldownSourceIds");
  const joints = request.jointSourceSets?.map((group, index) =>
    sourceIds(group, `jointSourceSets[${index}]`, 2)!);
  return {
    ...(sources ? { source_ids: sources } : {}),
    ...(request.coarseGroupSize !== undefined ? { coarse_group_size: request.coarseGroupSize } : {}),
    ...(request.refineThresholdNats !== undefined ? { refine_threshold_nats: request.refineThresholdNats } : {}),
    ...(drilldowns ? { drilldown_source_ids: drilldowns } : {}),
    ...(joints ? { joint_source_sets: joints } : {}),
    budget: {
      max_coarse_arms: request.budget.maxCoarseArms,
      max_fine_arms: request.budget.maxFineArms,
      max_joint_arms: request.budget.maxJointArms,
    },
    execution_policy: request.executionPolicy,
  };
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
    const response = await fetch(endpoint, { method: "GET", signal, headers: { Accept: "application/json" } });
    return decodeContextInvestigationReader(await responseJson(response, endpoint), endpoint, runId);
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
    const decoded = decodeContextInvestigationQuery(await responseJson(response, endpoint), endpoint);
    if (decoded.selection && (
      decoded.selection.unicodeRange[0] !== request.answerStart
      || decoded.selection.unicodeRange[1] !== request.answerEnd
    )) throw new ContractError(endpoint, "$.selection.unicode_range does not match the requested answer range");
    if (request.floorNats !== undefined && decoded.measurementFloorNats !== request.floorNats) {
      throw new ContractError(endpoint, "$.measurement_floor_nats does not match the requested measurement floor");
    }
    return decoded;
  },

  async startEffectsSearch(
    runId: string,
    request: EffectsSearchRequest,
    signal?: AbortSignal,
  ): Promise<EffectsSearchJob> {
    const endpoint = runPath(runId, "effects-search/jobs");
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(effectsSearchBody(request)),
    });
    return decodeEffectsSearchJob(await responseJson(response, endpoint, [202]), endpoint, runId);
  },

  async effectsSearchJob(
    runId: string,
    jobId: string,
    signal?: AbortSignal,
  ): Promise<EffectsSearchJob> {
    if (!jobId.trim()) throw new ContractError("client", "job id must not be blank");
    const endpoint = runPath(runId, `effects-search/jobs/${encodeURIComponent(jobId)}`);
    const response = await fetch(endpoint, {
      method: "GET", signal, headers: { Accept: "application/json" },
    });
    return decodeEffectsSearchJob(await responseJson(response, endpoint), endpoint, runId);
  },

  async cancelEffectsSearch(
    runId: string,
    jobId: string,
    signal?: AbortSignal,
  ): Promise<EffectsSearchJob> {
    if (!jobId.trim()) throw new ContractError("client", "job id must not be blank");
    const endpoint = runPath(runId, `effects-search/jobs/${encodeURIComponent(jobId)}/cancel`);
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: "{}",
    });
    return decodeEffectsSearchJob(await responseJson(response, endpoint), endpoint, runId);
  },

  async startRemoveSourceTest(
    runId: string,
    sourceId: string,
    signal?: AbortSignal,
  ): Promise<RemoveSourceTestJob> {
    const endpoint = runPath(runId, "remove-test/jobs");
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: singleSourceBody(sourceId),
    });
    return decodeRemoveSourceTestJob(
      await responseJson(response, endpoint, [202]), endpoint, runId, sourceId,
    );
  },

  async removeSourceTestJob(
    runId: string,
    jobId: string,
    sourceId: string,
    signal?: AbortSignal,
  ): Promise<RemoveSourceTestJob> {
    identifier(jobId, "job id");
    identifier(sourceId, "source id");
    const endpoint = runPath(runId, `remove-test/jobs/${encodeURIComponent(jobId)}`);
    const response = await fetch(endpoint, {
      method: "GET", signal, headers: { Accept: "application/json" },
    });
    return decodeRemoveSourceTestJob(
      await responseJson(response, endpoint), endpoint, runId, sourceId, jobId,
    );
  },

  async cancelRemoveSourceTest(
    runId: string,
    jobId: string,
    sourceId: string,
    signal?: AbortSignal,
  ): Promise<RemoveSourceTestJob> {
    identifier(jobId, "job id");
    identifier(sourceId, "source id");
    const endpoint = runPath(runId, `remove-test/jobs/${encodeURIComponent(jobId)}/cancel`);
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: "{}",
    });
    return decodeRemoveSourceTestJob(
      await responseJson(response, endpoint), endpoint, runId, sourceId, jobId,
    );
  },

  async startContextCounterfactual(
    runId: string,
    sourceId: string,
    signal?: AbortSignal,
  ): Promise<ContextCounterfactualJob> {
    const endpoint = runPath(runId, "counterfactuals/jobs");
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: singleSourceBody(sourceId),
    });
    return decodeContextCounterfactualJob(
      await responseJson(response, endpoint, [202]), endpoint, runId, sourceId,
    );
  },

  async contextCounterfactualJob(
    runId: string,
    jobId: string,
    sourceId: string,
    signal?: AbortSignal,
  ): Promise<ContextCounterfactualJob> {
    identifier(jobId, "job id");
    identifier(sourceId, "source id");
    const endpoint = runPath(runId, `counterfactuals/jobs/${encodeURIComponent(jobId)}`);
    const response = await fetch(endpoint, {
      method: "GET", signal, headers: { Accept: "application/json" },
    });
    return decodeContextCounterfactualJob(
      await responseJson(response, endpoint), endpoint, runId, sourceId, jobId,
    );
  },

  async cancelContextCounterfactual(
    runId: string,
    jobId: string,
    sourceId: string,
    signal?: AbortSignal,
  ): Promise<ContextCounterfactualJob> {
    identifier(jobId, "job id");
    identifier(sourceId, "source id");
    const endpoint = runPath(runId, `counterfactuals/jobs/${encodeURIComponent(jobId)}/cancel`);
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: "{}",
    });
    return decodeContextCounterfactualJob(
      await responseJson(response, endpoint), endpoint, runId, sourceId, jobId,
    );
  },

  async materializeContextCounterfactual(
    runId: string,
    request: CounterfactualMaterializationRequest,
    signal?: AbortSignal,
  ): Promise<CounterfactualMaterializationResult> {
    const endpoint = runPath(runId, "counterfactuals/materialize");
    const response = await fetch(endpoint, {
      method: "POST",
      signal,
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: materializationBody(request),
    });
    return decodeCounterfactualMaterializationResult(
      await responseJson(response, endpoint, [201]), endpoint, runId, request,
    );
  },
};
