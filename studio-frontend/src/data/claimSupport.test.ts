import { afterEach, describe, expect, it, vi } from "vitest";
import invalidAnswerUnknownField from "../../../tests/fixtures/schemas/clozn.answer-claims.v1/invalid__unknown_top_level_field.json";
import invalidAnswerNativeCollection from "../../../tests/fixtures/schemas/clozn.answer-claims.v1/invalid__wrong_native_ref_collection.json";
import emptyAnswer from "../../../tests/fixtures/schemas/clozn.answer-claims.v1/valid__empty_answer.json";
import limitedAnswer from "../../../tests/fixtures/schemas/clozn.answer-claims.v1/valid__segmentation_limited.json";
import metadataOnlyAnswer from "../../../tests/fixtures/schemas/clozn.answer-claims.v1/valid__typical_run_metadata_only.json";
import unavailableAnswer from "../../../tests/fixtures/schemas/clozn.answer-claims.v1/valid__unavailable_no_response.json";
import redactedAnswer from "../../../tests/fixtures/schemas/clozn.answer-claims.v1/valid__unavailable_redacted.json";
import mixedSupport from "../../../tests/fixtures/schemas/clozn.claim-support.v1/valid__mixed_statuses.json";
import { studioApi } from "./client";
import {
  bindRecordedResponse,
  decodeAnswerClaims,
  decodeClaimSupportDocument,
  decodeRecordedRunResponse,
  sliceUnicodeCodePoints,
} from "./claimSupport";
import { ContractError } from "./contracts";

type Row = Record<string, unknown>;

function row(value: unknown): Row {
  return value as Row;
}

function rows(value: unknown): Row[] {
  return value as Row[];
}

function pairedDocument(answerFixture: unknown = metadataOnlyAnswer): Row {
  const claims = structuredClone(answerFixture) as unknown as Row;
  const claimRows = rows(claims.claims);
  const supportFixture = structuredClone(mixedSupport) as unknown as Row;
  const firstSource = rows(supportFixture.results)[0].source_span_ids;
  supportFixture.run_id = claims.run_id;
  supportFixture.privacy = "metadata_only";
  row(supportFixture.source).claims_schema_version = "clozn.answer-claims.v1";
  row(row(supportFixture.source).influence_map).gate = claimRows.length === 0 ? "no_influence_map" : "ok";
  supportFixture.results = claimRows.map((claim, index) => {
    const category = claim.category;
    const claimAddressId = row(claim.text_span).address_id;
    return category === "factual_claim"
      ? {
          claim_index: index,
          claim_address_id: claimAddressId,
          status: "supported",
          method: { name: "forced_score_intervention", max_abs_delta_nats: 3.1 },
          source_span_ids: structuredClone(firstSource),
        }
      : {
          claim_index: index,
          claim_address_id: claimAddressId,
          status: "unverifiable_from_available_evidence",
          method: { name: "category_rule" },
        };
  });
  return { claims, support: supportFixture };
}

function singleClaimDocument(
  result: Row,
  gate = "ok",
  category: "factual_claim" | "recommendation" = "factual_claim",
): Row {
  const document = pairedDocument();
  const claims = row(document.claims);
  const claim = structuredClone(rows(claims.claims)[0]);
  claim.category = category;
  claim.category_reason = category === "factual_claim" ? "factual_declarative" : "recommendation_marker";
  claims.claims = [claim];
  row(claims.segmentation).claim_count = 1;
  const support = row(document.support);
  row(row(support.source).influence_map).gate = gate;
  support.results = [{
    claim_index: 0,
    claim_address_id: row(claim.text_span).address_id,
    ...result,
  }];
  return document;
}

function unicodeDocument(): Row {
  const basisSha256 = "7aadd8e96a0e5ac985118732639d514c32bc18a4af681f8b50c340740ce51db2";
  return {
    claims: {
      schema_version: "clozn.answer-claims.v1",
      run_id: "run-unicode",
      privacy: "metadata_only",
      offset_contract: {
        unit: "unicode_code_points",
        interval: "half_open",
        hash_algorithm: "sha256",
        canonicalization: "exact_string_utf8_v1",
      },
      segmentation: { state: "ok", claim_count: 1 },
      answer_source: {
        basis: "recorded_answer",
        basis_sha256: basisSha256,
        basis_code_points: 6,
        basis_utf8_bytes: 10,
      },
      claims: [{
        index: 0,
        category: "factual_claim",
        category_reason: "factual_declarative",
        text_span: {
          address_id: "span_111111111111111111111111",
          run_id: "run-unicode",
          kind: "claim",
          relation_key: "rel_111111111111111111111111",
          native_ref: {
            artifact_schema: "clozn.answer-claims.v1",
            collection: "derived.claims",
            id: "claim-0",
          },
          resolution: {
            state: "metadata_only",
            canonical: {
              basis: "recorded_answer",
              unit: "unicode_code_points",
              interval: "half_open",
              start: 1,
              end: 3,
              basis_sha256: basisSha256,
              span_sha256: "71b897d47640bd5e39412fb2d81c5d04b1195fc5524663a7704b6bfe2ec0db63",
              basis_code_points: 6,
              span_code_points: 2,
              basis_utf8_bytes: 10,
              span_utf8_bytes: 6,
            },
          },
        },
      }],
    },
    support: {
      schema_version: "clozn.claim-support.v1",
      run_id: "run-unicode",
      privacy: "metadata_only",
      offset_contract: {
        unit: "unicode_code_points",
        interval: "half_open",
        hash_algorithm: "sha256",
        canonicalization: "exact_string_utf8_v1",
      },
      source: {
        claims_schema_version: "clozn.answer-claims.v1",
        influence_map: { gate: "ok" },
      },
      results: [{
        claim_index: 0,
        claim_address_id: "span_111111111111111111111111",
        status: "unsupported_by_supplied_materials",
        method: { name: "measured_comparison_no_match" },
      }],
    },
  };
}

describe("claim-support data boundary", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("decodes the real metadata-only fixtures and binds every result to its ordered claim", () => {
    const decoded = decodeClaimSupportDocument(pairedDocument(), "/claim-support", "run-fixture-1");

    expect(decoded.claims).toMatchObject({
      schemaVersion: "clozn.answer-claims.v1",
      runId: "run-fixture-1",
      privacy: "metadata_only",
      segmentation: { state: "ok", claimCount: 5 },
    });
    expect(decoded.claims.claims.map((claim) => claim.textSpan.nativeRef.id))
      .toEqual(["claim-0", "claim-1", "claim-2", "claim-3", "claim-4"]);
    expect(decoded.support.results.map((result) => result.claimAddressId))
      .toEqual(decoded.claims.claims.map((claim) => claim.textSpan.addressId));
  });

  it.each([
    [emptyAnswer, "empty", "answer_text_empty"],
    [limitedAnswer, "segmentation_limited", "unsupported_script_density"],
    [unavailableAnswer, "unavailable", "no_answer_text"],
    [redactedAnswer, "unavailable", "answer_text_redacted"],
  ])("preserves the four-state segmentation contract for %s", (fixture, state, reason) => {
    const decoded = decodeClaimSupportDocument(pairedDocument(fixture));
    expect(decoded.claims.segmentation).toEqual({ state, reason });
    expect(decoded.claims.claims).toEqual([]);
    expect(decoded.support.results).toEqual([]);
  });

  it.each([
    ["supported", { status: "supported", method: { name: "forced_score_intervention", max_abs_delta_nats: 2 }, source_span_ids: ["span_222222222222222222222222"] }, "ok", "factual_claim"],
    ["weakly_supported", { status: "weakly_supported", method: { name: "textual_overlap", overlap_fraction: 0.75 }, source_span_ids: ["span_222222222222222222222222"] }, "ok", "factual_claim"],
    ["contradicted", { status: "contradicted", method: { name: "direct_negation" }, source_span_ids: ["span_222222222222222222222222"] }, "ok", "factual_claim"],
    ["unsupported_by_supplied_materials", { status: "unsupported_by_supplied_materials", method: { name: "measured_comparison_no_match" } }, "ok", "factual_claim"],
    ["unverifiable_from_available_evidence", { status: "unverifiable_from_available_evidence", method: { name: "category_rule" } }, "ok", "recommendation"],
    ["measurement_unavailable", { status: "measurement_unavailable", method: { name: "no_influence_map" } }, "no_influence_map", "factual_claim"],
  ] as const)("decodes the %s support state with its exact method/source shape", (status, result, gate, category) => {
    const decoded = decodeClaimSupportDocument(singleClaimDocument(structuredClone(result) as unknown as Row, gate, category));
    expect(decoded.support.results[0].status).toBe(status);
  });

  it("rejects real malformed answer fixtures and metadata-only text leakage", () => {
    expect(() => decodeAnswerClaims(invalidAnswerUnknownField)).toThrow(ContractError);
    expect(() => decodeAnswerClaims(invalidAnswerNativeCollection)).toThrow(ContractError);

    const leaked = pairedDocument();
    const canonical = row(row(row(rows(row(leaked.claims).claims)[0].text_span).resolution).canonical);
    canonical.text = "private claim text";
    expect(() => decodeClaimSupportDocument(leaked)).toThrow(/canonical.text/);
  });

  it("fails closed on schema, identity, category, native-address, count, order, and hash drift", () => {
    const mutations: Array<(document: Row) => void> = [
      (document) => { row(document.claims).schema_version = "clozn.answer-claims.v2"; },
      (document) => { row(document.support).run_id = "run-other"; },
      (document) => { row(document.claims).privacy = "full"; },
      (document) => { row(row(document.support).offset_contract).unit = "utf16_code_units"; },
      (document) => { rows(row(document.claims).claims)[0].category_reason = "recommendation_marker"; },
      (document) => { row(rows(row(document.claims).claims)[0].text_span).kind = "answer_span"; },
      (document) => { row(row(rows(row(document.claims).claims)[0].text_span).native_ref).id = "claim-99"; },
      (document) => { row(row(document.claims).segmentation).claim_count = 4; },
      (document) => { rows(row(document.support).results)[0].claim_address_id = "span_ffffffffffffffffffffffff"; },
      (document) => {
        row(row(row(rows(row(document.claims).claims)[0].text_span).resolution).canonical).basis_sha256 = "f".repeat(64);
      },
      (document) => { row(document.support).surprise = true; },
    ];
    for (const mutate of mutations) {
      const document = pairedDocument();
      mutate(document);
      expect(() => decodeClaimSupportDocument(document)).toThrow(ContractError);
    }
  });

  it("rejects status/method/gate/source-span mismatches and duplicate identifiers", () => {
    const badMethod = singleClaimDocument({
      status: "supported",
      method: { name: "forced_score_intervention", max_abs_delta_nats: 2, overlap_fraction: 1 },
      source_span_ids: ["span_222222222222222222222222"],
    });
    expect(() => decodeClaimSupportDocument(badMethod)).toThrow(/overlap_fraction/);

    const duplicateSources = singleClaimDocument({
      status: "supported",
      method: { name: "forced_score_intervention", max_abs_delta_nats: 2 },
      source_span_ids: ["span_222222222222222222222222", "span_222222222222222222222222"],
    });
    expect(() => decodeClaimSupportDocument(duplicateSources)).toThrow(/duplicates/);

    const claimAsSource = singleClaimDocument({
      status: "supported",
      method: { name: "forced_score_intervention", max_abs_delta_nats: 2 },
      source_span_ids: ["span_4cb9302c99c982fda3acd79b"],
    });
    expect(() => decodeClaimSupportDocument(claimAsSource)).toThrow(/not a claim span/);

    const wrongGate = singleClaimDocument({
      status: "measurement_unavailable",
      method: { name: "influence_measurement_error" },
    }, "no_influence_map");
    expect(() => decodeClaimSupportDocument(wrongGate)).toThrow(/failed influence gate/);

    const duplicateClaims = pairedDocument();
    rows(row(duplicateClaims.claims).claims)[1].text_span = rows(row(duplicateClaims.claims).claims)[0].text_span;
    expect(() => decodeClaimSupportDocument(duplicateClaims)).toThrow(ContractError);
  });

  it("binds hashes and Python-style Unicode code-point coordinates to the recorded response", async () => {
    const document = decodeClaimSupportDocument(unicodeDocument());
    const recorded = decodeRecordedRunResponse({ id: "run-unicode", response: "A😀é Z.", extra: "ignored" }, "/run", "run-unicode");
    const evidence = await bindRecordedResponse(document, recorded);

    expect(sliceUnicodeCodePoints("A😀é Z.", 1, 3)).toBe("😀é");
    expect(evidence.claimTexts).toEqual(["😀é"]);

    await expect(bindRecordedResponse(document, { runId: "run-unicode", response: "A😀ê Z." }))
      .rejects.toThrow(/SHA-256/);
    await expect(bindRecordedResponse(document, { runId: "run-other", response: "A😀é Z." }))
      .rejects.toThrow(/does not match/);
  });

  it("accepts an explicit null answer and rejects missing or malformed response values", async () => {
    const document = decodeClaimSupportDocument(pairedDocument(unavailableAnswer));
    const recorded = decodeRecordedRunResponse(
      { id: "run-fixture-4", response: null },
      "/runs/run-fixture-4",
      "run-fixture-4",
    );

    expect(recorded).toEqual({ runId: "run-fixture-4", response: null });
    await expect(bindRecordedResponse(document, recorded)).resolves.toMatchObject({
      recordedResponse: null,
      claimTexts: [],
      claims: { segmentation: { state: "unavailable", reason: "no_answer_text" } },
    });
    expect(() => decodeRecordedRunResponse({ id: "run-fixture-4", response: 42 }))
      .toThrow(/string or null/);
    expect(() => decodeRecordedRunResponse({ id: "run-fixture-4" }))
      .toThrow(/missing required field response/);
    expect(() => decodeRecordedRunResponse({ id: "run-fixture-4", response: undefined }))
      .toThrow(/string or null/);
  });

  it("rejects out-of-bounds Unicode ranges and a valid-looking but wrong span hash", async () => {
    const outside = unicodeDocument();
    row(row(row(rows(row(outside.claims).claims)[0].text_span).resolution).canonical).end = 7;
    expect(() => decodeClaimSupportDocument(outside)).toThrow(/within the recorded answer/);

    const wrongHash = unicodeDocument();
    row(row(row(rows(row(wrongHash.claims).claims)[0].text_span).resolution).canonical).span_sha256 = "f".repeat(64);
    const decoded = decodeClaimSupportDocument(wrongHash);
    await expect(bindRecordedResponse(decoded, { runId: "run-unicode", response: "A😀é Z." }))
      .rejects.toThrow(/span SHA-256/);
  });

  it("uses encoded GET-only HTTP reads and validates both response identities", async () => {
    const runId = "run / α";
    const wire = unicodeDocument();
    row(wire.claims).run_id = runId;
    row(wire.support).run_id = runId;
    rows(row(wire.claims).claims).forEach((claim) => { row(claim.text_span).run_id = runId; });
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const endpoint = String(input);
      const body = endpoint.endsWith("/claim-support") ? wire : { id: runId, response: "A😀é Z.", model: "ignored" };
      return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
    });
    vi.stubGlobal("fetch", fetchMock);

    const evidence = await studioApi.claimSupportEvidence(runId);

    expect(evidence.claimTexts).toEqual(["😀é"]);
    expect(fetchMock.mock.calls.map((call) => String(call[0])).sort()).toEqual([
      "/runs/run%20%2F%20%CE%B1",
      "/runs/run%20%2F%20%CE%B1/claim-support",
    ]);
    expect(fetchMock.mock.calls.every((call) => call[1]?.method === "GET")).toBe(true);
    await expect(studioApi.claimSupport("   ")).rejects.toBeInstanceOf(ContractError);
    await expect(studioApi.recordedRunResponse("   ")).rejects.toBeInstanceOf(ContractError);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
