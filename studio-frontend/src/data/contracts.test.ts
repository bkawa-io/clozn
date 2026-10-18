import { describe, expect, it } from "vitest";
import { ContractError, decodeCreatedRun, decodeRunDetails, decodeRunsDocument } from "./contracts";

describe("Studio run contracts", () => {
  it("decodes compact run rows without accepting a missing id", () => {
    expect(decodeRunsDocument({ runs: [{
      id: "run_a", created_at: "2026-10-15T12:00:00", created_ts: 1,
      model: "local", prompt_summary: "Question", response_summary: "Answer",
      parent_run_id: null, flags: ["truncated"], extra: "ignored",
    }] })[0]).toMatchObject({ id: "run_a", model: "local", flags: ["truncated"] });
    expect(() => decodeRunsDocument({ runs: [{}] })).toThrow(ContractError);
  });

  it("binds full run details to the requested run", () => {
    expect(decodeRunDetails({ id: "run_a", model: "local", identity: { engine_build: "v1" } }, "/runs/run_a", "run_a").identity)
      .toEqual({ engine_build: "v1" });
    expect(() => decodeRunDetails({ id: "run_b" }, "/runs/run_a", "run_a")).toThrow(/does not match/);
  });

  it("requires a durable id on a generated answer", () => {
    const response = { clozn_run_id: "run_new", choices: [{ message: { role: "assistant", content: "Answer" } }] };
    expect(decodeCreatedRun(response)).toEqual({ runId: "run_new", answer: "Answer" });
    expect(() => decodeCreatedRun({ choices: response.choices })).toThrow(/clozn_run_id/);
  });
});
