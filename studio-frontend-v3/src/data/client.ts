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
};

export function isContractError(error: unknown): error is ContractError {
  return error instanceof ContractError;
}

export function describeError(error: unknown): string {
  return error instanceof Error ? error.message : "The request failed.";
}
