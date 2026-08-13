import { afterEach, describe, expect, it, vi } from "vitest";
import { ContractError } from "./contracts";
import { HttpError, studioApi } from "./client";

const hash = "a".repeat(64);
const answerSpan = "span_0123456789abcdef01234567";
const sourceSpan = "span_89abcdef0123456789abcdef";

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, statusText: status === 200 ? "OK" : "Not Found", headers: { "content-type": "application/json" } });
}

afterEach(() => vi.unstubAllGlobals());

describe("Studio v2 data clients", () => {
  it("requests an encoded run influence range with AbortSignal and preserves available empty evidence", async () => {
    const controller = new AbortController();
    const fetchMock = vi.fn().mockResolvedValue(json({
      schema_version: "clozn.influence-query.v1", run_id: "run / α", privacy: "metadata_only",
      target: { basis: "recorded_answer", unit: "unicode_code_points", interval: "half_open", start: 1, end: 3, basis_sha256: hash, answer_span_ids: [] },
      measurement: { state: "available", influence_schema: "clozn.context_answer_influence.v1", artifact_sha256: hash, method: {}, thresholds: {} },
      links: [],
      summary: { selected_answer_spans: 0, measured_links: 0, returned_links: 0, causally_supported_links: 0, observed_links: 0, supporting_links: 0, suppressing_links: 0, neutral_links: 0 },
    }));
    vi.stubGlobal("fetch", fetchMock);

    const result = await studioApi.influenceQuery("run / α", { start: 1, end: 3 }, controller.signal);

    expect(fetchMock).toHaveBeenCalledWith("/runs/run%20%2F%20%CE%B1/influence-query?start=1&end=3", expect.objectContaining({ signal: controller.signal }));
    expect(result.measurement).toEqual(expect.objectContaining({ state: "available" }));
    expect(result.measurement.reason).toBeUndefined();
    expect(result.links).toEqual([]);
  });

  it("preserves not_measured rather than recasting it as an empty available result", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({
      schema_version: "clozn.context-tension.v1", run_id: "run-1", privacy: "metadata_only",
      target: { scope: "whole_answer", basis: "recorded_answer", unit: "unicode_code_points", interval: "half_open" },
      measurement: { state: "not_measured", reason: "no_influence_map" },
      tensions: [],
      summary: { answer_spans_examined: 0, answer_spans_with_tension: 0, tension_pairs: 0, returned_tension_pairs: 0, distinct_source_spans: 0 },
    })));

    const result = await studioApi.contextTension("run-1");

    expect(result.measurement).toEqual({ state: "not_measured", reason: "no_influence_map" });
    expect(result.tensions).toEqual([]);
  });

  it("keeps readable run messages and assembled context only when retained", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({
      id: "run-1", response: "Recorded answer", messages: [
        { role: "user", content: "Question" },
        { role: "user", content: "Attached source", source_id: "brief-7", source_label: "Project brief" },
      ],
      assembled_messages: [{ role: "user", content: "Attached source", client_source_id: "brief-7", source_label: "Project brief" }],
      trace: { tokens: ["Recorded", " answer"], token_ids: [12, 13] },
    })));

    const run = await studioApi.run("run-1");

    expect(run.response).toBe("Recorded answer");
    expect(run.messages?.[1]).toEqual({ role: "user", content: "Attached source", sourceId: "brief-7", sourceLabel: "Project brief" });
    expect(run.assembledMessages).toHaveLength(1);
    expect(run.responseTokens).toEqual(["Recorded", " answer"]);
    expect(run.responseTokenIds).toEqual([12, 13]);
  });

  it("rejects malformed schema documents rather than dropping malformed links or inventing zeroes", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({
      schema_version: "clozn.influence-query.v1", run_id: "run-1", privacy: "metadata_only",
      target: { basis: "recorded_answer", unit: "unicode_code_points", interval: "half_open", start: 0, end: 1 },
      measurement: { state: "available" },
      links: [{ source_span_id: sourceSpan, answer_span_id: answerSpan, effect: "supports", delta_nats: -1, abs_delta_nats: 1, clears_floor: true }],
      summary: { selected_answer_spans: 1, measured_links: 1, returned_links: 1, causally_supported_links: 1, observed_links: 0, supporting_links: 1, suppressing_links: 0, neutral_links: 0 },
    })));

    await expect(studioApi.influenceQuery("run-1", { start: 0, end: 1 })).rejects.toBeInstanceOf(ContractError);
  });

  it("decodes stable, metadata-only span addresses with canonical Unicode offsets", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({
      schema_version: "clozn.text-span-addresses.v1", run_id: "run-1", privacy: "metadata_only",
      offset_contract: { unit: "unicode_code_points", interval: "half_open", hash_algorithm: "sha256", canonicalization: "exact_string_utf8_v1" },
      source_artifacts: [],
      addresses: [{ address_id: answerSpan, run_id: "run-1", kind: "answer_span", relation_key: "rel_0123456789abcdef01234567", native_ref: { artifact_schema: "clozn.context_answer_influence.v1", collection: "influence.answer_spans", id: "answer-0" }, resolution: { state: "metadata_only", canonical: { basis: "recorded_answer", unit: "unicode_code_points", interval: "half_open", start: 2, end: 5, basis_sha256: hash, span_sha256: hash } } }],
      lineage: { parent_run_id: null, mappings: [] },
    })));

    const result = await studioApi.spanAddresses("run-1");

    expect(result.addresses[0]).toEqual(expect.objectContaining({ addressId: answerSpan, kind: "answer_span", nativeRef: expect.objectContaining({ id: "answer-0" }), resolution: expect.objectContaining({ canonical: expect.objectContaining({ start: 2, end: 5 }) }) }));
  });

  it("surfaces HTTP failures instead of silently returning an empty document", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ error: "run not found" }, 404)));
    await expect(studioApi.run("missing")).rejects.toBeInstanceOf(HttpError);
  });

  it("keeps structural comparison changes separate from evidence-strength findings", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({
      schema_version: "clozn.run-diff.v1",
      ok: true,
      run_a: "run a",
      run_b: "run/b",
      privacy_limited: true,
      summary_axes: { model: { status: "changed" }, context: { status: "unavailable", note: "metadata absent" } },
      differences: [{ dimension: "identity.model_sha256", kind: "changed", rank: 0, value_a: "old", value_b: "new", evidence: [] }],
      findings: [{ classification: "model_changed", status: "observed", summary: "The recorded model identity changed.", dimensions: ["identity.model_sha256"] }],
    })));

    const result = await studioApi.compare("run a", "run/b");

    expect(fetch).toHaveBeenCalledWith("/runs/compare?a=run+a&b=run%2Fb", expect.any(Object));
    expect(result.axes.context.status).toBe("unavailable");
    expect(result.differences[0]).toMatchObject({ kind: "changed", rank: 0 });
    expect(result.findings[0]).toMatchObject({ status: "observed" });
  });

  it("keeps live exactness unchecked when projecting rewind fidelity", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({
      schema_version: "clozn.rewind-fidelity.v1",
      run_id: "run-1",
      privacy: "metadata_only",
      coordinates: { kind: "recorded_response_token_boundary", index_base: 0, start: 0, end_exclusive: 2, recorded_token_count: 2 },
      recorded_capability: {
        state: "available",
        reconstructed_replay: { state: "available", supported_change_types: ["force_token"], unavoidable_differences: ["kv_state_not_restored"] },
        exact_rewind: { state: "requires_live_plan", static_prerequisites: { recorded_token_pieces: "available", recorded_token_ids: "available", token_alignment: "available", runtime_identity: "available" }, supported_change_types_if_live_plan_succeeds: ["force_token"], live_requirements: ["unchanged_control"], authority: "execution_fork_plan" },
      },
      historical_proof: { state: "available", verified_boundaries: [] },
      live_execution: { state: "not_checked", reason: "read_only_projection", authority: "execution_fork_plan" },
    })));

    const result = await studioApi.rewindFidelity("run-1");

    expect(result.recordedTokenCount).toBe(2);
    expect(result.exactRewind.state).toBe("requires_live_plan");
    expect(result.liveExecution.state).toBe("not_checked");
  });

  it("preserves every backend close-call locator without assigning UI significance", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({
      schema_version: "clozn.suggested-breakpoints.v1", run_id: "run-1", privacy: "metadata_only",
      coordinates: {}, analysis: { state: "partially_available" }, evidence: {},
      breakpoints: [{
        breakpoint_id: "breakpoint_0123456789abcdef01234567", position: 7,
        placement: "exact_token_decision", rank_class: "close_call",
        token_interval: { start: 10, end: 14, unit: "unicode_code_points", interval: "half_open" },
        reasons: [{ type: "close_call", emitted_token_id: 10, rival_token_id: 11, emitted_probability: .44, rival_probability: .42, margin: .02, meaningful_heuristic: false }],
      }],
      summary: { candidate_state: "detected", suggested_breakpoints: 1, returned_breakpoints: 1, combined_breakpoints: 0, meaningful_close_call_breakpoints: 0, context_tension_breakpoints: 0, ordinary_close_call_breakpoints: 1 },
    })));

    const result = await studioApi.suggestedBreakpoints("run-1");

    expect(fetch).toHaveBeenCalledWith("/runs/run-1/suggested-breakpoints?limit=50", expect.any(Object));
    expect(result.breakpoints).toEqual([expect.objectContaining({ position: 7, tokenInterval: { start: 10, end: 14 }, closeCall: expect.objectContaining({ rivalTokenId: 11, meaningful: false }) })]);
  });
});
