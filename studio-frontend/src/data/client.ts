import {
  ContractError,
  decodeCreatedRun,
  decodeRunDetails,
  decodeRunsDocument,
  type CreatedRun,
  type RunDetails,
  type RunSummary,
} from "./contracts";
import { decodeRunComparison, type ReductionComparison } from "./runComparison";
import { decodeContextReceipt, type ContextReceiptView } from "./contextReceipt";

export class HttpError extends Error {
  readonly name = "HttpError";

  constructor(readonly endpoint: string, readonly status: number, readonly detail: string) {
    super(`Request to ${endpoint} failed (${status}): ${detail}`);
  }
}

async function responseJson(response: Response, endpoint: string): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    throw new ContractError(endpoint, "response is not valid JSON");
  }
}

async function getJson(endpoint: string, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(endpoint, { method: "GET", signal, headers: { Accept: "application/json" } });
  if (!response.ok) throw new HttpError(endpoint, response.status, response.statusText || "request failed");
  return responseJson(response, endpoint);
}

async function postJson(endpoint: string, body: unknown, signal?: AbortSignal): Promise<unknown> {
  const response = await fetch(endpoint, {
    method: "POST",
    signal,
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await responseJson(response, endpoint);
  if (!response.ok) {
    const row = payload && typeof payload === "object" && !Array.isArray(payload) ? payload as Record<string, unknown> : {};
    const error = row.error;
    const detail = typeof error === "string"
      ? error
      : error && typeof error === "object" && !Array.isArray(error) && typeof (error as Record<string, unknown>).message === "string"
        ? String((error as Record<string, unknown>).message)
        : response.statusText || "request failed";
    throw new HttpError(endpoint, response.status, detail);
  }
  return payload;
}

function runId(value: string, endpoint: string): string {
  if (!value.trim()) throw new ContractError(endpoint, "run id must not be blank");
  return value;
}

export interface RunDraft {
  instructions: string;
  contextBlocks: readonly string[];
  question: string;
}

function generationRequest(draft: RunDraft): Record<string, unknown> {
  const instructions = draft.instructions.trim();
  const question = draft.question.trim();
  const contextBlocks = draft.contextBlocks.map((block) => block.trim()).filter(Boolean);
  if (!question) throw new ContractError("composer", "question must not be blank");
  if (!contextBlocks.length) throw new ContractError("composer", "at least one context block is required");
  if (contextBlocks.length > 16) throw new ContractError("composer", "at most 16 context blocks are supported");

  const messages: Array<{ role: "system" | "user"; content: string }> = [];
  if (instructions) messages.push({ role: "system", content: instructions });
  const firstContextIndex = messages.length;
  for (const content of contextBlocks) messages.push({ role: "user", content });
  messages.push({ role: "user", content: question });

  return {
    messages,
    max_tokens: 128,
    temperature: 0,
    clozn_sources: contextBlocks.map((content, index) => ({
      message_index: firstContextIndex + index,
      source_id: `context-${index + 1}`,
      label: `Context ${index + 1}`,
      unicode_range: [0, [...content].length],
      provenance_kind: "client_supplied",
    })),
  };
}

export const studioApi = {
  async runs(signal?: AbortSignal): Promise<readonly RunSummary[]> {
    const endpoint = "/runs";
    return decodeRunsDocument(await getJson(endpoint, signal), endpoint);
  },

  async run(runIdValue: string, signal?: AbortSignal): Promise<RunDetails> {
    const id = runId(runIdValue, "/runs/<id>");
    const endpoint = `/runs/${encodeURIComponent(id)}`;
    return decodeRunDetails(await getJson(endpoint, signal), endpoint, id);
  },

  async createRun(draft: RunDraft, signal?: AbortSignal): Promise<CreatedRun> {
    const endpoint = "/v1/chat/completions";
    return decodeCreatedRun(await postJson(endpoint, generationRequest(draft), signal), endpoint);
  },

  async deleteRun(runIdValue: string, signal?: AbortSignal): Promise<void> {
    const id = runId(runIdValue, "/runs/<id>");
    const endpoint = `/runs/${encodeURIComponent(id)}`;
    const response = await fetch(endpoint, { method: "DELETE", signal, headers: { Accept: "application/json" } });
    const payload = await responseJson(response, endpoint);
    if (!response.ok) {
      const row = payload && typeof payload === "object" && !Array.isArray(payload) ? payload as Record<string, unknown> : {};
      throw new HttpError(endpoint, response.status, typeof row.error === "string" ? row.error : response.statusText || "request failed");
    }
  },

  async runComparison(runAValue: string, runBValue: string, signal?: AbortSignal): Promise<ReductionComparison> {
    const base = "/runs/compare";
    const runA = runId(runAValue, base);
    const runB = runId(runBValue, base);
    if (runA === runB) throw new ContractError(base, "original and reduced child run ids must differ");
    const endpoint = `${base}?${new URLSearchParams({ a: runA, b: runB })}`;
    return decodeRunComparison(await getJson(endpoint, signal), endpoint, runA, runB);
  },

  async contextReceipt(runIdValue: string, signal?: AbortSignal): Promise<ContextReceiptView> {
    const id = runId(runIdValue, "/runs/<id>/context-receipt");
    const endpoint = `/runs/${encodeURIComponent(id)}/context-receipt`;
    return decodeContextReceipt(await getJson(endpoint, signal), endpoint, id);
  },
};

export function isContractError(error: unknown): error is ContractError {
  return error instanceof ContractError;
}

export function describeError(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed.";
}
