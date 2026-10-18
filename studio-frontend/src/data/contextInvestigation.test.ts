import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "./client";
import { ContractError } from "./contracts";
import {
  decodeContextInvestigationQuery,
  decodeContextInvestigationReader,
} from "./contextInvestigation";
import { contextInvestigationApi } from "./contextInvestigationClient";
import { decodeContextEffectsJob } from "./contextInvestigationJobs";

const runId = "run / α";
const baselineObservationId = "obs_baseline";
type WireDocument = Record<string, any>;

function readerDocument(): WireDocument {
  return {
    schema_version: "clozn.context-investigation-reader.v1",
    run_id: runId,
    status: "completed",
    measurement: {
      status: "completed",
      experiment_id: "exp_reader",
      evaluator: "score_recorded_continuation",
      intervention: "delete_source",
      source_universe: ["source-a"],
      source_universe_basis: "explicit",
      completed_source_count: 1,
      unavailable_source_count: 0,
      failed_source_count: 0,
      not_measured_source_count: 0,
      baseline_observation_id: baselineObservationId,
      additive_measurement_detail: true,
    },
    context: {
      coordinate_basis: "context_display_unicode.v1",
      separator: "\n\n",
      blocks: [{
        message_index: 0,
        role: "user",
        text: "A😀B",
        segment_id: "source-a",
        display_unicode_range: [0, 3],
        source_ids: ["source-a"],
      }],
    },
    answer: {
      status: "available",
      text: "Hi 🌕!",
      sha256: "a".repeat(64),
      tokens: [
        { index: 0, token_id: 10, piece: "Hi ", unicode_range: [0, 3], logprob: -0.5 },
        { index: 1, token_id: 11, piece: "🌕", unicode_range: [3, 4], logprob: -1 },
        { index: 2, token_id: 12, piece: "!", unicode_range: [4, 5], logprob: -0.2 },
      ],
      fidelity: "baseline_token_score_observation",
      observation_id: baselineObservationId,
    },
    sources: [{
      source_id: "source-a",
      segment_id: "source-a",
      message_index: 0,
      label: "Source A",
      provenance_kind: "message",
      parent_source_id: null,
      unicode_range: [0, 3],
      byte_range: [0, 6],
      display_unicode_range: [0, 3],
      granularity: "whole_segment",
      measurement: {
        status: "measured",
        experiment_id: "exp_reader",
        arm_id: "arm_a",
        observation_id: "obs_a",
        intervention: {
          kind: "delete_source",
          target: { kind: "context_selection", source_ids: ["source-a"] },
        },
      },
      summary: {
        max_abs_token_delta_nats: 0.4,
        total_continuation_delta_nats: 0.4,
        above_floor_locus_count: 1,
      },
      loci: ["locus_a"],
    }],
    loci: [{
      locus_id: "locus_a",
      source_id: "source-a",
      direction: "support",
      classification: "measured_support",
      answer_token_range: [1, 2],
      answer_unicode_range: [3, 4],
      answer_text: "🌕",
      delta_nats: 0.4,
      absolute_delta_nats: 0.4,
      baseline_selected_logp: -1,
      intervened_selected_logp: -1.4,
      experiment_id: "exp_reader",
      arm_id: "arm_a",
      baseline_observation_id: baselineObservationId,
      intervention_observation_id: "obs_a",
      per_token_delta_nats: [0.4],
      additive_locus_detail: "retained on the wire, not consumed",
    }],
    presentation: { measurement_floor_nats: 0.1 },
    capabilities: {
      context_effects: "measurable",
      measurement: "completed",
      answer_selection_projection: "available",
    },
    additive_projection_detail: { version: 2 },
  };
}

function queryDocument(): WireDocument {
  return {
    schema_version: "clozn.context-investigation-query.v1",
    status: "completed",
    selection: {
      unicode_range: [3, 4],
      token_range: [1, 2],
      text: "🌕",
      token_ids: [11],
      token_pieces: ["🌕"],
    },
    effects: [{
      source_id: "source-a",
      delta_nats: 0.4,
      direction: "support",
      classification: "measured_support",
      measurement_status: "measured",
      experiment_id: "exp_reader",
      arm_id: "arm_a",
      baseline_observation_id: baselineObservationId,
      intervention_observation_id: "obs_a",
      additive_effect_detail: true,
    }],
    measurement_floor_nats: 0.1,
    additive_query_detail: true,
  };
}

function effectsJobDocument(state = "completed"): WireDocument {
  const document: WireDocument = {
    schema_version: "clozn.analysis-job.v1",
    job_id: "infjob_fixture",
    run_id: runId,
    kind: "context_investigation_effects",
    state,
    progress: {
      phase: state === "completed" ? "done" : "scoring_recorded_continuation",
      completed_units: state === "completed" ? 1 : 0,
      total_units: 1,
      percent: state === "completed" ? 100 : 0,
    },
    cancel_requested: false,
    cancellable: state !== "completed",
    cached: false,
    created_at: "2026-10-08T00:00:00+00:00",
    updated_at: "2026-10-08T00:00:01+00:00",
  };
  if (state === "completed") document.result = {
    schema_version: "clozn.context-investigation-effects-job-result.v1",
    experiment_id: "exp_reader",
    source_count: 2,
    completed_source_count: 1,
    unavailable_source_count: 1,
    failed_source_count: 0,
    not_measured_source_count: 0,
    state: "completed",
  };
  return document;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    statusText: status === 200 ? "OK" : "Not Found",
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("Context Investigation reader contract", () => {
  it("decodes canonical evidence in Unicode code-point coordinates and tolerates open additive fields", () => {
    const reader = decodeContextInvestigationReader(readerDocument(), "/reader", runId);

    expect(reader.context.blocks[0].displayUnicodeRange).toEqual([0, 3]);
    expect(reader.sources[0].byteRange).toEqual([0, 6]);
    expect(reader.answer.status).toBe("available");
    expect(reader.loci[0]).toEqual(expect.objectContaining({
      answerText: "🌕",
      answerUnicodeRange: [3, 4],
      direction: "support",
    }));
    expect(reader).not.toHaveProperty("additiveProjectionDetail");
  });

  it("keeps stale evidence distinct from not measured evidence", () => {
    const stale = readerDocument();
    stale.status = "stale";
    stale.reason = "persisted evidence no longer binds this Run";
    stale.reason_code = "context_investigation_stale";
    stale.measurement.status = "stale";
    stale.measurement.completed_source_count = 0;
    stale.measurement.not_measured_source_count = 1;
    stale.measurement.baseline_observation_id = null;
    stale.answer = { status: "unavailable" } as typeof stale.answer;
    stale.sources = [];
    stale.loci = [];
    stale.capabilities.measurement = "stale";
    stale.capabilities.answer_selection_projection = "unavailable";

    expect(decodeContextInvestigationReader(stale).status).toBe("stale");

    const notMeasured = readerDocument();
    notMeasured.status = "not_measured";
    notMeasured.measurement.status = "not_measured";
    notMeasured.measurement.completed_source_count = 0;
    notMeasured.measurement.not_measured_source_count = 1;
    notMeasured.measurement.baseline_observation_id = null;
    notMeasured.answer.fidelity = "recorded_run_trace";
    notMeasured.answer.observation_id = null;
    notMeasured.answer.tokens.forEach((token: WireDocument) => delete token.logprob);
    notMeasured.sources[0].measurement.status = "not_measured";
    notMeasured.sources[0].measurement.experiment_id = null;
    notMeasured.sources[0].measurement.arm_id = null;
    notMeasured.sources[0].measurement.observation_id = null;
    delete notMeasured.sources[0].measurement.intervention;
    notMeasured.sources[0].summary.max_abs_token_delta_nats = null;
    notMeasured.sources[0].summary.total_continuation_delta_nats = null;
    notMeasured.sources[0].summary.above_floor_locus_count = 0;
    notMeasured.sources[0].loci = [];
    notMeasured.loci = [];
    notMeasured.capabilities.measurement = "not_measured";

    expect(decodeContextInvestigationReader(notMeasured).status).toBe("not_measured");
  });

  it.each(["unavailable", "failed"] as const)("decodes partial reader coverage with an explicitly %s source", (sourceStatus) => {
    const partial = readerDocument();
    partial.status = "partial";
    partial.measurement.status = "partial";
    partial.measurement.completed_source_count = 0;
    partial.measurement.unavailable_source_count = sourceStatus === "unavailable" ? 1 : 0;
    partial.measurement.failed_source_count = sourceStatus === "failed" ? 1 : 0;
    partial.sources[0].measurement.status = sourceStatus;
    partial.sources[0].measurement.observation_id = null;
    partial.sources[0].summary.max_abs_token_delta_nats = null;
    partial.sources[0].summary.total_continuation_delta_nats = null;
    partial.sources[0].summary.above_floor_locus_count = 0;
    partial.sources[0].loci = [];
    partial.loci = [];
    partial.capabilities.measurement = "partial";

    const decoded = decodeContextInvestigationReader(partial);
    expect(decoded.status).toBe("partial");
    expect(decoded.sources[0].measurement.status).toBe(sourceStatus);
    expect(decoded.sources[0].summary.maxAbsTokenDeltaNats).toBeNull();
    expect(decoded.loci).toEqual([]);
  });

  it("decodes an unavailable reader reason without requiring fabricated source or answer evidence", () => {
    const unavailable = readerDocument();
    unavailable.status = "unavailable";
    unavailable.reason = "recorded answer score evidence is unavailable";
    unavailable.reason_code = "context_investigation_unavailable";
    unavailable.measurement.status = "unavailable";
    unavailable.measurement.completed_source_count = 0;
    unavailable.measurement.not_measured_source_count = 1;
    unavailable.measurement.baseline_observation_id = null;
    unavailable.answer = { status: "unavailable" } as typeof unavailable.answer;
    unavailable.sources = [];
    unavailable.loci = [];
    unavailable.capabilities.measurement = "unavailable";
    unavailable.capabilities.answer_selection_projection = "unavailable";

    const decoded = decodeContextInvestigationReader(unavailable);
    expect(decoded).toEqual(expect.objectContaining({
      status: "unavailable",
      reason: "recorded answer score evidence is unavailable",
      answer: { status: "unavailable" },
      sources: [],
      loci: [],
    }));
  });

  it("preserves canonical prose when the default measurement universe has no legal arm", () => {
    const unavailable = readerDocument();
    unavailable.status = "unavailable";
    unavailable.reason = "no canonical Context Receipt source is exactly removable";
    unavailable.reason_code = "context_investigation_no_exactly_removable_source";
    unavailable.measurement.status = "unavailable";
    unavailable.measurement.experiment_id = null;
    unavailable.measurement.source_universe = [];
    unavailable.measurement.completed_source_count = 0;
    unavailable.measurement.unavailable_source_count = 0;
    unavailable.measurement.failed_source_count = 0;
    unavailable.measurement.not_measured_source_count = 0;
    unavailable.measurement.baseline_observation_id = null;
    unavailable.sources[0].measurement = {
      status: "unavailable",
      experiment_id: null,
      arm_id: null,
      observation_id: null,
    };
    unavailable.sources[0].summary = {
      max_abs_token_delta_nats: null,
      total_continuation_delta_nats: null,
      above_floor_locus_count: 0,
    };
    unavailable.sources[0].loci = [];
    unavailable.loci = [];
    unavailable.capabilities.context_effects = "unavailable";
    unavailable.capabilities.measurement = "unavailable";

    const decoded = decodeContextInvestigationReader(unavailable);
    expect(decoded.context.blocks[0].text).toBe("A😀B");
    expect(decoded.answer.status).toBe("available");
    expect(decoded.measurement.experimentId).toBeNull();
    expect(decoded.sources[0].measurement.status).toBe("unavailable");

    unavailable.sources[0].measurement.arm_id = "arm_fabricated";
    expect(() => decodeContextInvestigationReader(unavailable)).toThrow(/must not claim experiment evidence/);
  });

  it("rejects a response for a different requested Run", () => {
    expect(() => decodeContextInvestigationReader(readerDocument(), "/reader", "run_other")).toThrow(/must match requested Run/);
  });

  it("rejects UTF-16-shaped context offsets and mismatched UTF-8 byte ranges", () => {
    const wrongUnicode = readerDocument();
    wrongUnicode.context.blocks[0].display_unicode_range = [0, 4];
    expect(() => decodeContextInvestigationReader(wrongUnicode)).toThrow(ContractError);

    const wrongBytes = readerDocument();
    wrongBytes.sources[0].byte_range = [0, 4];
    expect(() => decodeContextInvestigationReader(wrongBytes)).toThrow(/UTF-8 byte range/);
  });

  it("rejects answer tokens that do not reconstruct the answer in recorded order", () => {
    const malformed = readerDocument();
    malformed.answer.tokens[1].unicode_range = [3, 5];
    expect(() => decodeContextInvestigationReader(malformed)).toThrow(/token piece/);

    const reordered = readerDocument();
    reordered.answer.tokens[1].index = 2;
    expect(() => decodeContextInvestigationReader(reordered)).toThrow(/token order/);
  });

  it("rejects duplicate identities, broken cross references, and inconsistent counts", () => {
    const duplicate = readerDocument();
    duplicate.sources.push(structuredClone(duplicate.sources[0]));
    expect(() => decodeContextInvestigationReader(duplicate)).toThrow(/duplicate IDs/);

    const wrongEvidence = readerDocument();
    wrongEvidence.loci[0].intervention_observation_id = "obs_other";
    expect(() => decodeContextInvestigationReader(wrongEvidence)).toThrow(/source observation/);

    const wrongCount = readerDocument();
    wrongCount.measurement.completed_source_count = 0;
    wrongCount.measurement.not_measured_source_count = 1;
    expect(() => decodeContextInvestigationReader(wrongCount)).toThrow(/completed/);
  });

  it("rejects a locus whose token, Unicode, text, or signed evidence semantics disagree", () => {
    const wrongText = readerDocument();
    wrongText.loci[0].answer_text = "!";
    expect(() => decodeContextInvestigationReader(wrongText)).toThrow(/recorded answer coordinates/);

    const wrongSign = readerDocument();
    wrongSign.loci[0].direction = "suppression";
    wrongSign.loci[0].classification = "measured_suppression";
    expect(() => decodeContextInvestigationReader(wrongSign)).toThrow(/measurement floor/);

    const belowFloor = readerDocument();
    belowFloor.loci[0].per_token_delta_nats = [0.05];
    belowFloor.loci[0].delta_nats = 0.05;
    belowFloor.loci[0].absolute_delta_nats = 0.05;
    belowFloor.loci[0].intervened_selected_logp = -1.05;
    expect(() => decodeContextInvestigationReader(belowFloor)).toThrow(/measurement floor/);
  });
});

describe("Context Investigation query contract", () => {
  it("decodes measured support, suppression, below-floor, and explicit nonmeasurement effects", () => {
    const query = queryDocument();
    query.effects.push({
      ...query.effects[0],
      source_id: "source-b",
      arm_id: "arm_b",
      intervention_observation_id: "obs_b",
      delta_nats: -0.3,
      direction: "suppression",
      classification: "measured_suppression",
    });
    query.effects.push({
      ...query.effects[0],
      source_id: "source-c",
      arm_id: "arm_c",
      intervention_observation_id: "obs_c",
      delta_nats: 0.05,
      direction: "below_floor",
      classification: "below_floor",
    });
    const decoded = decodeContextInvestigationQuery(query);
    expect(decoded.effects.map((effect) => effect.direction)).toEqual(["support", "suppression", "below_floor"]);

    const partial = queryDocument();
    partial.status = "partial";
    partial.effects[0].measurement_status = "unavailable";
    partial.effects[0].direction = "unavailable";
    partial.effects[0].classification = null;
    partial.effects[0].delta_nats = null;
    partial.effects[0].intervention_observation_id = null;
    expect(decodeContextInvestigationQuery(partial).effects[0].measurementStatus).toBe("unavailable");
  });

  it("preserves a pure nonmeasurement query state", () => {
    const document = {
      schema_version: "clozn.context-investigation-query.v1",
      status: "not_measured",
      selection: null,
      effects: [],
      measurement_floor_nats: 0.1,
    };
    expect(decodeContextInvestigationQuery(document)).toEqual(expect.objectContaining({
      status: "not_measured",
      selection: null,
      effects: [],
    }));
  });

  it.each(["stale", "unavailable"] as const)("preserves a reasoned %s query without selection or effect evidence", (status) => {
    const document = {
      schema_version: "clozn.context-investigation-query.v1",
      status,
      reason: `${status} persisted score vectors`,
      reason_code: `context_investigation_${status}`,
      selection: null,
      effects: [],
      measurement_floor_nats: 0.1,
    };
    const decoded = decodeContextInvestigationQuery(document);
    expect(decoded.status).toBe(status);
    expect(decoded.reason).toBe(`${status} persisted score vectors`);
    expect(decoded.selection).toBeNull();
    expect(decoded.effects).toEqual([]);

    expect(() => decodeContextInvestigationQuery({
      ...document,
      selection: queryDocument().selection,
    })).toThrow(/must not carry a selection or effects/);
  });

  it("rejects query selections and effects that overstate the measured evidence", () => {
    const wrongLength = queryDocument();
    wrongLength.selection.unicode_range = [3, 5];
    expect(() => decodeContextInvestigationQuery(wrongLength)).toThrow(/length/);

    const duplicate = queryDocument();
    duplicate.effects.push({ ...duplicate.effects[0] });
    expect(() => decodeContextInvestigationQuery(duplicate)).toThrow(/duplicate IDs/);

    const wrongDirection = queryDocument();
    wrongDirection.effects[0].direction = "below_floor";
    wrongDirection.effects[0].classification = "below_floor";
    expect(() => decodeContextInvestigationQuery(wrongDirection)).toThrow(/configured floor/);

    const overstatesCompletion = queryDocument();
    overstatesCompletion.effects[0].measurement_status = "failed";
    overstatesCompletion.effects[0].direction = "failed";
    overstatesCompletion.effects[0].classification = null;
    overstatesCompletion.effects[0].delta_nats = null;
    overstatesCompletion.effects[0].intervention_observation_id = null;
    expect(() => decodeContextInvestigationQuery(overstatesCompletion)).toThrow(/whether every returned effect is measured/);
  });
});

describe("Context Effects job contract", () => {
  it("decodes progress and direct source-measurement counts", () => {
    const job = decodeContextEffectsJob(effectsJobDocument(), "/job", runId);
    expect(job.state).toBe("completed");
    expect(job.result).toEqual(expect.objectContaining({
      experimentId: "exp_reader",
      sourceCount: 2,
      completedSourceCount: 1,
      unavailableSourceCount: 1,
    }));
  });

  it("rejects inconsistent source status counts", () => {
    const snapshot = effectsJobDocument();
    snapshot.result.completed_source_count = 2;
    expect(() => decodeContextEffectsJob(snapshot, "/job", runId)).toThrow(/add up/);
  });

  it("rejects cross-run and completed-without-result snapshots", () => {
    expect(() => decodeContextEffectsJob(
      { ...effectsJobDocument(), run_id: "run_other" }, "/job", runId,
    )).toThrow(/requested Run/);
    expect(() => decodeContextEffectsJob(
      { ...effectsJobDocument("running"), state: "completed" }, "/job", runId,
    )).toThrow(/result/);
  });
});

describe("Context Investigation read client", () => {
  it("requests the fixed reader floor with encoded Run identity and AbortSignal", async () => {
    const controller = new AbortController();
    const fetchMock = vi.fn().mockResolvedValue(json(readerDocument()));
    vi.stubGlobal("fetch", fetchMock);

    const result = await contextInvestigationApi.reader(runId, controller.signal);

    expect(result.runId).toBe(runId);
    expect(fetchMock).toHaveBeenCalledWith(
      "/runs/run%20%2F%20%CE%B1/context-investigation/reader?floor_nats=0.1",
      { method: "GET", signal: controller.signal, headers: { Accept: "application/json" } },
    );
  });

  it("POSTs a pure-read JSON query and verifies the returned coordinate and floor", async () => {
    const response = queryDocument();
    response.measurement_floor_nats = 0.2;
    const controller = new AbortController();
    const fetchMock = vi.fn().mockResolvedValue(json(response));
    vi.stubGlobal("fetch", fetchMock);

    const result = await contextInvestigationApi.query(
      runId,
      { answerStart: 3, answerEnd: 4, floorNats: 0.2 },
      controller.signal,
    );

    expect(result.selection?.text).toBe("🌕");
    expect(fetchMock).toHaveBeenCalledWith(
      "/runs/run%20%2F%20%CE%B1/context-investigation/query",
      expect.objectContaining({
        method: "POST",
        signal: controller.signal,
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({ answer_start: 3, answer_end: 4, floor_nats: 0.2 }),
      }),
    );
  });

  it("rejects invalid requests, mismatched projections, and HTTP errors", async () => {
    vi.stubGlobal("fetch", vi.fn());
    await expect(contextInvestigationApi.query(runId, { answerStart: 2, answerEnd: 2 })).rejects.toBeInstanceOf(ContractError);

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ ...readerDocument(), run_id: "run_other" })));
    await expect(contextInvestigationApi.reader(runId)).rejects.toThrow(/must match requested Run/);

    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(json({ error: "missing" }, 404)));
    await expect(contextInvestigationApi.reader(runId)).rejects.toBeInstanceOf(HttpError);
  });
});

describe("Context Effects client", () => {
  it("starts one direct measurement job for canonical source IDs", async () => {
    const queued = effectsJobDocument("queued");
    const fetchMock = vi.fn().mockResolvedValue(json(queued, 202));
    vi.stubGlobal("fetch", fetchMock);

    const job = await contextInvestigationApi.startEffects(runId, {
      sourceIds: ["source-a"],
    });

    expect(job.state).toBe("queued");
    const [endpoint, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(endpoint).toBe("/runs/run%20%2F%20%CE%B1/context-investigation/effects/jobs");
    expect(init.method).toBe("POST");
    expect(JSON.parse(String(init.body))).toEqual({ source_ids: ["source-a"] });
  });

  it("polls and cancels the same encoded job resource", async () => {
    const running = effectsJobDocument("running");
    const cancelling = {
      ...running,
      state: "cancelling",
      cancel_requested: true,
      cancellable: true,
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(json(running))
      .mockResolvedValueOnce(json(cancelling));
    vi.stubGlobal("fetch", fetchMock);

    await contextInvestigationApi.effectsJob(runId, "job / α");
    await contextInvestigationApi.cancelEffects(runId, "job / α");

    expect(fetchMock.mock.calls[0][0]).toContain("effects/jobs/job%20%2F%20%CE%B1");
    expect(fetchMock.mock.calls[0][1]).toEqual(expect.objectContaining({ method: "GET" }));
    expect(fetchMock.mock.calls[1][0]).toContain("effects/jobs/job%20%2F%20%CE%B1/cancel");
    expect(fetchMock.mock.calls[1][1]).toEqual(expect.objectContaining({ method: "POST" }));
  });
});
