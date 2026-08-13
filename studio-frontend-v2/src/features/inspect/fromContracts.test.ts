import { describe, expect, it } from "vitest";
import type { InfluenceQuery, RunRecord, SpanAddressDocument } from "../../data/contracts";
import { codePointOffset, locusQueryRange, projectInfluenceSelection, projectLinkedReader } from "./fromContracts";

const hash = "a".repeat(64);
const answerId = "span_111111111111111111111111";
const sourceId = "span_222222222222222222222222";

const addresses: SpanAddressDocument = {
  runId: "run-one",
  privacy: "metadata_only",
  addresses: [
    {
      addressId: answerId,
      runId: "run-one",
      kind: "answer_span",
      relationKey: "rel_111111111111111111111111",
      nativeRef: { artifactSchema: "clozn.context_answer_influence.v1", collection: "influence.answer_spans", id: "a.0" },
      resolution: { state: "metadata_only", canonical: { basis: "scored_answer", start: 1, end: 2, basisSha256: hash, spanSha256: hash } },
    },
    {
      addressId: sourceId,
      runId: "run-one",
      kind: "attached_source_span",
      relationKey: "rel_222222222222222222222222",
      nativeRef: { artifactSchema: "clozn.context_answer_influence.v1", collection: "influence.prompt_spans", id: "p.0.0", clientSourceId: "doc-1" },
      resolution: { state: "metadata_only", canonical: { basis: "attached_source", start: 0, end: 6, basisSha256: hash, spanSha256: hash } },
    },
  ],
};

const run: RunRecord = {
  id: "run-one",
  response: "A🙂B",
  messages: [{ role: "user", sourceId: "doc-1", sourceLabel: "Policy", content: "Refund within 30 days." }],
};

it("converts Unicode code-point offsets without splitting surrogate pairs", () => {
  expect(codePointOffset("A🙂B", 2)).toBe(3);
  expect(locusQueryRange("A🙂B", { id: "x", start: 1, end: 3 })).toEqual({ start: 1, end: 2 });
});

describe("linked reader projection", () => {
  it("links real answer and source coordinates through stable span addresses", () => {
    const projection = projectLinkedReader(run, addresses);
    expect(projection.specimen.answerLoci[0]).toMatchObject({ id: answerId, start: 1, end: 3 });
    const query: InfluenceQuery = {
      runId: "run-one",
      target: { start: 1, end: 2 },
      measurement: { state: "available", method: { name: "forced_score" }, thresholds: {} },
      links: [{ sourceSpanId: sourceId, answerSpanId: answerId, effect: "supports", deltaNats: 0.8, absDeltaNats: 0.8, clearsFloor: true, evidenceState: "causally_supported" }],
      summary: { selectedAnswerSpans: 1, measuredLinks: 1, returnedLinks: 1, causallySupportedLinks: 1, observedLinks: 0, supportingLinks: 1, suppressingLinks: 0, neutralLinks: 0 },
    };
    expect(projectInfluenceSelection(query, run, projection).related[0]).toMatchObject({
      documentId: "source:doc-1:0",
      start: 0,
      end: 6,
      effect: "supports",
    });
  });

  it("does not invent measured links for unavailable evidence", () => {
    const projection = projectLinkedReader(run, addresses);
    const query = {
      runId: "run-one",
      target: { start: 1, end: 2 },
      measurement: { state: "not_measured", reason: "no_influence_map" },
      links: [],
      summary: { selectedAnswerSpans: 0, measuredLinks: 0, returnedLinks: 0, causallySupportedLinks: 0, observedLinks: 0, supportingLinks: 0, suppressingLinks: 0, neutralLinks: 0 },
    } satisfies InfluenceQuery;
    expect(projectInfluenceSelection(query, run, projection)).toEqual({ state: "not_measured", reason: "no_influence_map", related: [] });
  });

  it("collapses duplicate backend links at one source coordinate", () => {
    const projection = projectLinkedReader(run, addresses);
    const query: InfluenceQuery = {
      runId: "run-one",
      target: { start: 1, end: 2 },
      measurement: { state: "available", method: { name: "forced_score" }, thresholds: {} },
      links: [
        { sourceSpanId: sourceId, answerSpanId: answerId, effect: "supports", deltaNats: 0.2, absDeltaNats: 0.2, clearsFloor: true, evidenceState: "causally_supported" },
        { sourceSpanId: sourceId, answerSpanId: answerId, effect: "supports", deltaNats: 0.8, absDeltaNats: 0.8, clearsFloor: true, evidenceState: "causally_supported" },
      ],
      summary: { selectedAnswerSpans: 1, measuredLinks: 2, returnedLinks: 2, causallySupportedLinks: 2, observedLinks: 0, supportingLinks: 2, suppressingLinks: 0, neutralLinks: 0 },
    };
    expect(projectInfluenceSelection(query, run, projection).related).toHaveLength(1);
    expect(projectInfluenceSelection(query, run, projection).related[0].deltaNats).toBe(0.8);
  });
});
