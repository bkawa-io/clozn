import { afterEach, describe, expect, it, vi } from "vitest";
import invalidInvestigationState from "../../../tests/fixtures/schemas/clozn.run-investigation.v1/invalid__unknown_outer_state.json";
import minimalInvestigation from "../../../tests/fixtures/schemas/clozn.run-investigation.v1/valid__minimal.json";
import driftLeaksText from "../../../tests/fixtures/schemas/clozn.text-span-addresses.v1/invalid__drift_leaks_text.json";
import driftWithoutReason from "../../../tests/fixtures/schemas/clozn.text-span-addresses.v1/invalid__drift_without_reason.json";
import exactWithoutText from "../../../tests/fixtures/schemas/clozn.text-span-addresses.v1/invalid__exact_without_text.json";
import metadataLeaksText from "../../../tests/fixtures/schemas/clozn.text-span-addresses.v1/invalid__metadata_only_leaks_text.json";
import metadataOnly from "../../../tests/fixtures/schemas/clozn.text-span-addresses.v1/valid__metadata_only.json";
import redactedReference from "../../../tests/fixtures/schemas/clozn.text-span-addresses.v1/valid__redacted_reference.json";
import { studioApi } from "./client";
import { ContractError } from "./contracts";
import { decodeRunInvestigation, decodeSpanAddressDocument } from "./receivedContext";

describe("Received Context read contracts", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("decodes the fixture-backed investigation, typed actions, and unavailable measurements", () => {
    const document = decodeRunInvestigation(minimalInvestigation, "/investigation", "run_fixture");

    expect(document).toMatchObject({
      schemaVersion: "clozn.run-investigation.v1",
      runId: "run_fixture",
      receivedContext: {
        state: "delivered_not_measured",
        artifact: {
          schema: "clozn.context-receipt.v1",
          method: "gateway_context_receipt",
          nativeStatus: "new",
        },
      },
      actions: [{
        id: "measure_prompt_source_influence",
        kind: "measurement",
        method: "POST",
        availability: "unavailable",
      }],
      unavailableMeasurements: [{
        id: "prompt_source_influence",
        state: "delivered_not_measured",
        actionId: "measure_prompt_source_influence",
      }],
    });
  });

  it("fails closed on investigation state, identity, and closed action drift", () => {
    expect(() => decodeRunInvestigation(invalidInvestigationState)).toThrow(/received_context.state/);
    expect(() => decodeRunInvestigation(minimalInvestigation, "/investigation", "run_other"))
      .toThrow(/does not match the request/);
    expect(() => decodeRunInvestigation({
      ...minimalInvestigation,
      actions: [{ ...minimalInvestigation.actions[0], surprise: true }],
    })).toThrow(/surprise/);
    expect(() => decodeRunInvestigation({
      ...minimalInvestigation,
      unavailable_measurements: [{
        ...minimalInvestigation.unavailable_measurements[0],
        state: "measured_effect",
      }],
    })).toThrow(/unavailable_measurements\[0\].state/);
  });

  it("decodes metadata-only addresses, source methods, and lineage mappings", () => {
    const document = decodeSpanAddressDocument(metadataOnly, "/spans", "run_child");

    expect(document).toMatchObject({
      schemaVersion: "clozn.text-span-addresses.v1",
      runId: "run_child",
      privacy: "metadata_only",
      parentRunId: "run_parent",
      sourceArtifacts: [{
        schema: "clozn.context_answer_influence.v1",
        method: { name: "forced scoring", mode: "forced_score_intervention" },
      }],
      lineageMappings: [{
        relationKey: "rel_0123456789abcdef01234567",
        parentAddressId: "span_aaaaaaaaaaaaaaaaaaaaaaaa",
        childAddressId: "span_0123456789abcdef01234567",
        state: "inherited",
        reason: "exact_text_and_hashes_unchanged",
      }],
    });
    expect(document.addresses[0]).toMatchObject({
      addressId: "span_0123456789abcdef01234567",
      resolution: {
        state: "metadata_only",
        canonical: { start: 0, end: 5, spanCodePoints: 5 },
      },
    });
  });

  it("preserves a validated truncated recorded hash without promoting it to canonical evidence", () => {
    const document = decodeSpanAddressDocument(redactedReference, "/spans", "run_redacted");

    expect(document.addresses[0]).toMatchObject({
      nativeRef: {
        recordedHash: {
          algorithm: "sha256_truncated_64bit",
          value: "0123456789abcdef",
          scope: "canonical_basis",
        },
      },
      resolution: { state: "redacted", reason: "source_text_redacted" },
    });
    expect(document.addresses[0].resolution.canonical).toBeUndefined();
  });

  it("rejects the real invalid privacy and resolution fixtures", () => {
    for (const fixture of [driftLeaksText, driftWithoutReason, exactWithoutText, metadataLeaksText]) {
      expect(() => decodeSpanAddressDocument(fixture)).toThrow(ContractError);
    }
  });

  it("fails closed on nested identity drift, malformed hashes, lineage, and source artifacts", () => {
    expect(() => decodeSpanAddressDocument(metadataOnly, "/spans", "run_other"))
      .toThrow(/does not match the request/);
    expect(() => decodeSpanAddressDocument({
      ...metadataOnly,
      addresses: [{ ...metadataOnly.addresses[0], run_id: "run_other" }],
    })).toThrow(/does not match the document/);
    expect(() => decodeSpanAddressDocument({
      ...redactedReference,
      addresses: [{
        ...redactedReference.addresses[0],
        native_ref: {
          ...redactedReference.addresses[0].native_ref,
          recorded_hash: {
            ...redactedReference.addresses[0].native_ref.recorded_hash,
            value: "too-short",
          },
        },
      }],
    })).toThrow(/16-character lowercase hash/);
    expect(() => decodeSpanAddressDocument({
      ...metadataOnly,
      lineage: {
        ...metadataOnly.lineage,
        mappings: [{ ...metadataOnly.lineage.mappings[0], reason: "looks_close_enough" }],
      },
    })).toThrow(/lineage.mappings\[0\].reason/);
    expect(() => decodeSpanAddressDocument({
      ...metadataOnly,
      source_artifacts: [{ ...metadataOnly.source_artifacts[0], method: "not-an-object" }],
    })).toThrow(/method must be an object/);
  });

  it("uses only encoded read-only GET routes and validates the selected Run", async () => {
    const runId = "run / α";
    const investigation = { ...minimalInvestigation, run_id: runId };
    const spans = {
      ...metadataOnly,
      run_id: runId,
      addresses: metadataOnly.addresses.map((address) => ({ ...address, run_id: runId })),
    };
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const endpoint = String(input);
      return new Response(JSON.stringify(endpoint.endsWith("/investigation") ? investigation : spans), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    await studioApi.runInvestigation(runId);
    await studioApi.spanAddresses(runId);

    expect(String(fetchMock.mock.calls[0][0])).toBe("/runs/run%20%2F%20%CE%B1/investigation");
    expect(String(fetchMock.mock.calls[1][0])).toBe("/runs/run%20%2F%20%CE%B1/span-addresses");
    expect(fetchMock.mock.calls.every((call) => call[1]?.method === "GET")).toBe(true);
    await expect(studioApi.runInvestigation("   ")).rejects.toBeInstanceOf(ContractError);
    await expect(studioApi.spanAddresses("   ")).rejects.toBeInstanceOf(ContractError);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
