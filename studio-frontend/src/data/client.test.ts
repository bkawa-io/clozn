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
      context: " # Northstar\nNorthstar launches May 22.\n\n# Code\nA😀B ",
      question: " When does Northstar launch? ",
    })).resolves.toEqual({ runId: "run_new", answer: "May 22." });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [endpoint, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(endpoint).toBe("/v1/chat/completions");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({
      messages: [
        { role: "system", content: "Be concise." },
        { role: "user", content: "# Northstar\nNorthstar launches May 22.\n\n# Code\nA😀B" },
        { role: "user", content: "When does Northstar launch?" },
      ],
      max_tokens: 128,
      temperature: 0,
    });
  });

  it("rejects empty context before dispatch", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(studioApi.createRun({ instructions: "", context: "  ", question: "Question" }))
      .rejects.toBeInstanceOf(ContractError);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
