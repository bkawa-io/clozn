import { describe, expect, it } from "vitest";
import { ContractError, decodeCreatedRun, decodeRunDetails, decodeRunsDocument } from "./contracts";

describe("Studio run contracts", () => {
  it("decodes compact run rows without accepting a missing id", () => {
    expect(decodeRunsDocument({ runs: [{
      id: "run_a", created_at: "2026-10-15T12:00:00", created_ts: 1,
      model: "local", prompt_summary: "Question", response_summary: "Answer",
      parent_run_id: null, conversation_id: "run_root", conversation_turn: 2,
      conversation_size: 3, analysis_status: "partial", analysis_source_count: 8,
      analysis_completed_source_count: 3, analysis_scoring_mode: "exact", flags: ["truncated"], extra: "ignored",
    }] })[0]).toMatchObject({
      id: "run_a", model: "local", flags: ["truncated"],
      conversationId: "run_root", conversationTurn: 2, conversationSize: 3,
      analysisStatus: "partial", analysisSourceCount: 8, analysisCompletedSourceCount: 3, analysisScoringMode: "exact",
    });
    expect(() => decodeRunsDocument({ runs: [{}] })).toThrow(ContractError);
    expect(() => decodeRunsDocument({ runs: [{ id: "run_a", analysis_status: "guessing" }] }))
      .toThrow(/analysis status/);
    expect(() => decodeRunsDocument({ runs: [{ id: "run_a", analysis_scoring_mode: "guessing" }] }))
      .toThrow(/scoring mode/);
  });

  it("binds full run details to the requested run", () => {
    expect(decodeRunDetails({
      id: "run_a",
      model: "local",
      messages: [{ role: "user", content: "Hello" }],
      response: "Hi",
      identity: { engine_build: "v1" },
    }, "/runs/run_a", "run_a")).toMatchObject({
      messages: [{ role: "user", content: "Hello" }],
      response: "Hi",
      identity: { engine_build: "v1" },
    });
    expect(() => decodeRunDetails({ id: "run_b" }, "/runs/run_a", "run_a")).toThrow(/does not match/);
  });

  it("requires a durable id on a generated answer", () => {
    const response = { clozn_run_id: "run_new", choices: [{ message: { role: "assistant", content: "Answer" } }] };
    expect(decodeCreatedRun(response)).toEqual({ runId: "run_new", answer: "Answer" });
    expect(() => decodeCreatedRun({ choices: response.choices })).toThrow(/clozn_run_id/);
  });
});
