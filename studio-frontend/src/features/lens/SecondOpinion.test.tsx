import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../test/fetch";
import { render, screen, waitFor } from "../../test/render";
import { SecondOpinion } from "./SecondOpinion";

function pathOf(request: PendingFetch) {
  return typeof request.input === "string"
    ? request.input
    : request.input instanceof URL
      ? request.input.pathname
      : request.input.url;
}

function bodyFor(request: PendingFetch) {
  return JSON.parse(String(request.init?.body ?? "{}")) as Record<string, unknown>;
}

function candidatesBody() {
  return {
    managed: true,
    own_model_id: "anchor-model",
    candidates: [
      { model_id: "second-model", ready: true },
      { model_id: "cold-model", ready: false },
    ],
  };
}

function resultBody() {
  return {
    schema_version: "clozn.model-second-opinion.v1",
    generated_at: "2026-08-01T00:00:00Z",
    run_id: "run-one",
    delivered_input: { message_count: 2, sha256: "a".repeat(64), identical_across_arms: true },
    arm_a: {
      role: "anchor", run_id: "run-one", model_id: "anchor-model", status: "ok",
      response_text: "The anchor answer.", finish_reason: "stop", latency_ms: 12,
      worker_identity: { template_fingerprint: "tmpl-a", worker_id: "worker-a" },
    },
    arm_b: {
      role: "second_opinion", requested_model_id: "second-model", model_id: "second-model", status: "ok",
      response_text: "A different answer.", finish_reason: "stop", latency_ms: 21,
      worker_identity: { template_fingerprint: "tmpl-b", worker_id: "worker-b" },
    },
    compatibility: {
      chat_template: { state: "differs", method: "template_fingerprint_compare", caveat: "template caveat" },
      context_limit: { state: "within_estimate", method: "arm_a_recorded_prompt_tokens_vs_arm_b_context_window" },
      tools_or_schema: { state: "none_used" },
      qualified_evidence: { state: "anchor_only", note: "anchor support only" },
    },
    comparison: {
      agreement: { method: "lexical_overlap_heuristic", lexical_difference_percent: 42, caveat: "lexical only" },
      format_changed: false,
      length: { arm_a_words: 3, arm_b_words: 3 },
    },
  };
}

describe("Second opinion", () => {
  test("checks candidates without generating, then runs only after an explicit model choice and click", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();
    render(<SecondOpinion runId="run-one" />);

    const candidateRequest = await controller.nextRequest();
    expect(pathOf(candidateRequest)).toContain("/runs/run-one/second-opinion/candidates");
    expect(candidateRequest.init?.method).toBeUndefined();
    controller.respondJson(candidateRequest, candidatesBody());
    await screen.findByLabelText("RESIDENT MODEL");
    expect(controller.requests.filter((request) => pathOf(request).endsWith("/second-opinion")).length).toBe(0);

    await user.selectOptions(screen.getByLabelText("RESIDENT MODEL"), "second-model");
    await user.click(screen.getByRole("button", { name: "ASK SECOND OPINION" }));
    const runRequest = await controller.nextRequest();
    expect(pathOf(runRequest)).toContain("/runs/run-one/second-opinion");
    expect(runRequest.init?.method).toBe("POST");
    expect(bodyFor(runRequest)).toEqual({ model: "second-model" });
    controller.respondJson(runRequest, resultBody());

    expect(await screen.findByText("The anchor answer.")).toBeInTheDocument();
    expect(screen.getByText("A different answer.")).toBeInTheDocument();
    expect(screen.getByText("42% LEXICAL DIFFERENCE")).toBeInTheDocument();
    expect(screen.getByText("template caveat")).toBeInTheDocument();
    expect(screen.getByText(/no cross-model token probabilities are shown/)).toBeInTheDocument();
  });

  test("keeps an arm-level refusal visible without inventing a top-level success state", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    render(<SecondOpinion runId="run-one" />);
    const candidateRequest = await controller.nextRequest();
    controller.respondJson(candidateRequest, candidatesBody());
    await screen.findByLabelText("RESIDENT MODEL");

    const user = userEvent.setup();
    await user.selectOptions(screen.getByLabelText("RESIDENT MODEL"), "second-model");
    await user.click(screen.getByRole("button", { name: "ASK SECOND OPINION" }));
    const runRequest = await controller.nextRequest();
    const refused = resultBody();
    delete (refused as Record<string, unknown>).comparison;
    controller.respondJson(runRequest, {
      ...refused,
      arm_b: {
        role: "second_opinion", requested_model_id: "second-model", model_id: "second-model",
        status: "generation_error", refusal: { code: "generation_error", message: "worker timed out" },
      },
    });

    expect(await screen.findByText("worker timed out")).toBeInTheDocument();
    expect(screen.getByText("OK")).toBeInTheDocument();
    expect(screen.queryByText("42% LEXICAL DIFFERENCE")).not.toBeInTheDocument();
  });
});
