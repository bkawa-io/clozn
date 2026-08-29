import { describe, expect, it } from "vitest";
import { ContractError, decodeSessionListDocument, decodeSessionTrace } from "./contracts";

const sessionId = "session_0123456789abcdef01234567";

function trace(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: "clozn.session-trace.v1",
    generated_at: "2026-08-25T00:00:00Z",
    session_id: sessionId,
    session: { id: sessionId, privacy: { visibility: "visible" } },
    page: { cursor: null, next_cursor: null, limit: 50, count: 0 },
    turns: [],
    branches: [],
    totals_through_this_page: { turn_count: 0, duration_ms_total: 0, prompt_tokens_total: 0, generated_tokens_total: 0 },
    diagnostic_rule_registry: [],
    first_went_wrong_candidates: [],
    ...overrides,
  };
}

describe("v3 read contracts", () => {
  it("decodes session-list previews without turning preview prose into a title", () => {
    const [session] = decodeSessionListDocument({
      sessions: [{
        schema_version: "clozn.session.v1",
        id: sessionId,
        created_ts: 10,
        created_at: "2026-08-25T00:00:00",
        privacy: { visibility: "visible" },
        materialized_from: "explicit",
        run_count: 2,
        turn_count: 1,
        last_activity_ts: 20,
        preview: { run_id: "run_1", prompt_summary: "Question", response_summary: "Answer" },
      }],
    });
    expect(session.title).toBeUndefined();
    expect(session.runCount).toBe(2);
    expect(session.turnCount).toBe(1);
    expect(session.preview).toEqual({ runId: "run_1", promptSummary: "Question", responseSummary: "Answer" });
  });

  it("fails closed on an extra session field", () => {
    expect(() => decodeSessionListDocument({ sessions: [{
      schema_version: "clozn.session.v1", id: sessionId, created_ts: 1, created_at: "now",
      privacy: { visibility: "visible" }, materialized_from: "explicit", invented: true,
    }] })).toThrow(ContractError);
  });

  it("fails closed on malformed trace turns instead of dropping them", () => {
    expect(() => decodeSessionTrace(trace({
      page: { cursor: null, next_cursor: null, limit: 50, count: 1 },
      turns: [{ run_id: "run_1" }],
    }))).toThrow(ContractError);
  });

  it("keeps a valid empty trace distinct from a malformed trace", () => {
    const decoded = decodeSessionTrace(trace());
    expect(decoded.turns).toEqual([]);
    expect(decoded.page.count).toBe(0);
  });
});
