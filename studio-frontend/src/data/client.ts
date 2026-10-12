import {
  ContractError,
  decodeSession,
  decodeSessionListDocument,
  decodeSessionTrace,
  decodeStandaloneRunsDocument,
  type SessionRecord,
  type SessionTrace,
  type StandaloneRun,
} from "./contracts";
import {
  decodeLocalModels,
  decodeRuntimeHealth,
  type LocalModelRecord,
  type RuntimeHealth,
} from "./runtime";
import { decodeRunComparison, type RunComparison } from "./runComparison";
import {
  decodeExperimentCells,
  decodeExperimentDetail,
  decodeExperimentList,
  type ExperimentCellsDocument,
  type ExperimentDetail,
  type ExperimentListDocument,
  type ExperimentSuite,
} from "./experimentResults";
import {
  decodeContextReceipt,
  type ContextReceiptView,
} from "./contextReceipt";
import {
  decodeRunInvestigation,
  decodeSpanAddressDocument,
  type RunInvestigation,
  type SpanAddressDocument,
} from "./receivedContext";
import {
  bindRecordedResponse,
  decodeClaimSupportDocument,
  decodeRecordedRunResponse,
  type ClaimSupportDocument,
  type ClaimSupportEvidence,
  type RecordedRunResponse,
} from "./claimSupport";

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

function sessionPath(sessionId: string, suffix = ""): string {
  if (!sessionId.trim()) throw new ContractError("client", "session id must not be blank");
  return `/sessions/${encodeURIComponent(sessionId)}${suffix}`;
}

function comparisonRunId(value: string, label: string): string {
  if (!value.trim()) throw new ContractError("/runs/compare", `${label} must not be blank`);
  return value;
}

function experimentId(value: string): string {
  if (!value.trim()) throw new ContractError("/experiment-results", "experiment id must not be blank");
  return value;
}

function contextReceiptRunId(value: string): string {
  if (!value.trim()) throw new ContractError("/runs/<id>/context-receipt", "run id must not be blank");
  return value;
}

function receivedContextRunId(value: string, endpoint: string): string {
  if (!value.trim()) throw new ContractError(endpoint, "run id must not be blank");
  return value;
}

export const studioApi = {
  async sessions(signal?: AbortSignal): Promise<readonly SessionRecord[]> {
    const endpoint = "/sessions";
    return decodeSessionListDocument(await getJson(endpoint, signal), endpoint);
  },

  async session(sessionId: string, signal?: AbortSignal): Promise<SessionRecord> {
    const endpoint = sessionPath(sessionId);
    return decodeSession(await getJson(endpoint, signal), endpoint);
  },

  async sessionTracePage(sessionId: string, cursor: string | undefined, signal?: AbortSignal): Promise<SessionTrace> {
    const base = sessionPath(sessionId, "/trace");
    const query = new URLSearchParams({ limit: "100" });
    if (cursor) query.set("cursor", cursor);
    const endpoint = `${base}?${query}`;
    return decodeSessionTrace(await getJson(endpoint, signal), endpoint);
  },

  /** Read every persisted trace page in backend order. The decoder runs before each page is accumulated. */
  async sessionTrace(sessionId: string, signal?: AbortSignal): Promise<SessionTrace> {
    let cursor: string | undefined;
    let first: SessionTrace | undefined;
    let latest: SessionTrace | undefined;
    const turns: SessionTrace["turns"] = [];
    const branches: SessionTrace["branches"] = [];
    for (;;) {
      const page = await this.sessionTracePage(sessionId, cursor, signal);
      first ??= page;
      latest = page;
      turns.push(...page.turns);
      branches.push(...page.branches);
      if (!page.page.nextCursor) break;
      if (page.page.nextCursor === cursor) throw new ContractError("/sessions/<id>/trace", "pagination cursor did not advance");
      cursor = page.page.nextCursor;
    }
    if (!first || !latest) throw new ContractError("/sessions/<id>/trace", "no trace page was returned");
    return { ...latest, turns, branches, page: { ...latest.page, cursor: first.page.cursor, count: turns.length } };
  },

  async standaloneRuns(signal?: AbortSignal): Promise<readonly StandaloneRun[]> {
    const endpoint = "/runs";
    return decodeStandaloneRunsDocument(await getJson(endpoint, signal), endpoint);
  },

  async runComparison(runAValue: string, runBValue: string, signal?: AbortSignal): Promise<RunComparison> {
    const base = "/runs/compare";
    const runA = comparisonRunId(runAValue, "reference run id");
    const runB = comparisonRunId(runBValue, "candidate run id");
    if (runA === runB) throw new ContractError(base, "reference and candidate run ids must differ");
    const endpoint = `${base}?${new URLSearchParams({ a: runA, b: runB })}`;
    return decodeRunComparison(await getJson(endpoint, signal), endpoint, runA, runB);
  },

  async contextReceipt(
    runIdValue: string,
    signal?: AbortSignal,
  ): Promise<ContextReceiptView> {
    const runId = contextReceiptRunId(runIdValue);
    const endpoint = `/runs/${encodeURIComponent(runId)}/context-receipt`;
    return decodeContextReceipt(await getJson(endpoint, signal), endpoint, runId);
  },

  async runInvestigation(runIdValue: string, signal?: AbortSignal): Promise<RunInvestigation> {
    const base = "/runs/<id>/investigation";
    const runId = receivedContextRunId(runIdValue, base);
    const endpoint = `/runs/${encodeURIComponent(runId)}/investigation`;
    return decodeRunInvestigation(await getJson(endpoint, signal), endpoint, runId);
  },

  async spanAddresses(runIdValue: string, signal?: AbortSignal): Promise<SpanAddressDocument> {
    const base = "/runs/<id>/span-addresses";
    const runId = receivedContextRunId(runIdValue, base);
    const endpoint = `/runs/${encodeURIComponent(runId)}/span-addresses`;
    return decodeSpanAddressDocument(await getJson(endpoint, signal), endpoint, runId);
  },

  async claimSupport(runIdValue: string, signal?: AbortSignal): Promise<ClaimSupportDocument> {
    const base = "/runs/<id>/claim-support";
    const runId = receivedContextRunId(runIdValue, base);
    const endpoint = `/runs/${encodeURIComponent(runId)}/claim-support`;
    return decodeClaimSupportDocument(await getJson(endpoint, signal), endpoint, runId);
  },

  async recordedRunResponse(runIdValue: string, signal?: AbortSignal): Promise<RecordedRunResponse> {
    const base = "/runs/<id>";
    const runId = receivedContextRunId(runIdValue, base);
    const endpoint = `/runs/${encodeURIComponent(runId)}`;
    return decodeRecordedRunResponse(await getJson(endpoint, signal), endpoint, runId);
  },

  async claimSupportEvidence(runIdValue: string, signal?: AbortSignal): Promise<ClaimSupportEvidence> {
    const base = "/runs/<id>/claim-support";
    const runId = receivedContextRunId(runIdValue, base);
    const [document, recorded] = await Promise.all([
      this.claimSupport(runId, signal),
      this.recordedRunResponse(runId, signal),
    ]);
    return bindRecordedResponse(document, recorded, base);
  },

  async experimentResults(signal?: AbortSignal): Promise<ExperimentListDocument> {
    const endpoint = "/experiment-results";
    return decodeExperimentList(await getJson(endpoint, signal), endpoint);
  },

  async experimentResult(idValue: string, signal?: AbortSignal): Promise<ExperimentDetail> {
    const id = experimentId(idValue);
    const endpoint = `/experiment-results/${encodeURIComponent(id)}`;
    return decodeExperimentDetail(await getJson(endpoint, signal), endpoint, id);
  },

  async experimentCells(
    idValue: string,
    filter: { suite: ExperimentSuite; caseName: string; variant: string; seed: number },
    signal?: AbortSignal,
  ): Promise<ExperimentCellsDocument> {
    const id = experimentId(idValue);
    if (!filter.caseName.trim() || !filter.variant.trim() || !Number.isSafeInteger(filter.seed)) {
      throw new ContractError("/experiment-results/<id>/cells", "cell coordinates must be complete");
    }
    const query = new URLSearchParams({
      suite: filter.suite,
      case: filter.caseName,
      variant: filter.variant,
      seed: String(filter.seed),
    });
    const endpoint = `/experiment-results/${encodeURIComponent(id)}/cells?${query}`;
    const document = decodeExperimentCells(await getJson(endpoint, signal), endpoint, id);
    const cell = document.cells[0];
    if (
      document.cells.length !== 1
      || !cell
      || cell.suite !== filter.suite
      || cell.caseName !== filter.caseName
      || cell.variant !== filter.variant
      || cell.seed !== filter.seed
    ) {
      throw new ContractError(endpoint, "response does not contain exactly the requested cell");
    }
    return document;
  },

  async runtimeHealth(signal?: AbortSignal): Promise<RuntimeHealth> {
    const endpoint = "/engine/health";
    return decodeRuntimeHealth(await getJson(endpoint, signal), endpoint);
  },

  async localModels(signal?: AbortSignal): Promise<readonly LocalModelRecord[]> {
    const endpoint = "/models/local";
    return decodeLocalModels(await getJson(endpoint, signal), endpoint);
  },
};

export function isContractError(error: unknown): error is ContractError {
  return error instanceof ContractError;
}

export function describeError(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed.";
}
