import { afterEach, describe, expect, it, vi } from "vitest";
import { ContractError } from "./contracts";
import { studioApi } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("Studio API client", () => {
  it("builds the protected-instructions, removable-context, protected-question request", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      clozn_run_id: "run_new",
      choices: [{ message: { role: "assistant", content: "May 22." } }],
    }), { status: 200, headers: { "content-type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(studioApi.createRun({
      instructions: " Be concise. ",
      contextBlocks: [" Northstar launches May 22. ", "A😀B"],
      question: " When does Northstar launch? ",
    })).resolves.toEqual({ runId: "run_new", answer: "May 22." });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [endpoint, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(endpoint).toBe("/v1/chat/completions");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({
      messages: [
        { role: "system", content: "Be concise." },
        { role: "user", content: "Northstar launches May 22." },
        { role: "user", content: "A😀B" },
        { role: "user", content: "When does Northstar launch?" },
      ],
      max_tokens: 128,
      temperature: 0,
      clozn_sources: [
        {
          message_index: 1,
          source_id: "context-1",
          label: "Context 1",
          unicode_range: [0, 26],
          provenance_kind: "client_supplied",
        },
        {
          message_index: 2,
          source_id: "context-2",
          label: "Context 2",
          unicode_range: [0, 3],
          provenance_kind: "client_supplied",
        },
      ],
    });
  });

  it("rejects an empty or over-limit removable context before dispatch", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(studioApi.createRun({ instructions: "", contextBlocks: ["  "], question: "Question" }))
      .rejects.toBeInstanceOf(ContractError);
    await expect(studioApi.createRun({
      instructions: "",
      contextBlocks: Array.from({ length: 17 }, (_, index) => `Context ${index}`),
      question: "Question",
    })).rejects.toThrow(/at most 16/);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
