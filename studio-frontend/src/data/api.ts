import type {
  CandidateReading,
  ConceptCandidate,
  ContextCoverage,
  ForkExactness,
  ForkIntervention,
  ForkOutcome,
  ForkOutcomeKind,
  ForkReason,
  ForkUnchangedControl,
  InfluenceAbsence,
  InfluenceErrorCode,
  InfluenceEvidenceState,
  InfluenceMapJob,
  InfluenceMapJobState,
  InfluenceMethod,
  InfluenceThresholds,
  MeasureInfluenceResult,
  ObservatoryData,
  PerformanceEvidenceState,
  PerformancePhase,
  PerformanceRuleDiagnosis,
  PerformanceRuleEvidence,
  PerformanceRuleReport,
  PerformanceRuleStatus,
  RunConfiguration,
  RunConcepts,
  RunDiagnosis,
  RunDiagnosisFinding,
  RunFacts,
  RunPerformance,
  RunSummary,
  RuntimeState,
  SourceReading,
  TokenReading,
  TokenSourceReading,
  WorkspaceReadout,
} from "./types";

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function numberAt(value: unknown, index: number, fallback = 0): number {
  if (!Array.isArray(value)) return fallback;
  const item = Number(value[index]);
  return Number.isFinite(item) ? item : fallback;
}

function runLabel(run: JsonRecord): string {
  const value = run.prompt_summary ?? run.prompt ?? run.id ?? run.run_id;
  return String(value || "Untitled run").slice(0, 72);
}

function runPickerLabel(run: JsonRecord): string {
  const id = String(run.id ?? run.run_id ?? "");
  const source = String(run.source || "run");
  return `${runLabel(run).slice(0, 54)} · ${source} · ${id.slice(-6)}`;
}

function durationFromMilliseconds(value: unknown): { text: string; milliseconds?: number } {
  const duration = Number(value);
  if (!Number.isFinite(duration)) return { text: "—" };
  return {
    text: duration < 1000 ? `${Math.round(duration)} ms` : `${(duration / 1000).toFixed(2)} s`,
    milliseconds: duration,
  };
}

function runSummary(run: JsonRecord): RunSummary {
  const timing = record(run.timing);
  const memory = record(run.memory);
  const behavior = record(run.behavior);
  const duration = durationFromMilliseconds(timing.duration_ms ?? run.duration_ms);
  const activeDials = record(behavior.active_dials);
  const cards = Array.isArray(memory.cards_applied)
    ? memory.cards_applied
    : Array.isArray(memory.applied_ids)
      ? memory.applied_ids
      : [];
  const createdTs = Number(run.created_ts);
  return {
    id: String(run.id ?? run.run_id ?? ""),
    label: runPickerLabel(run),
    prompt: String(run.prompt_summary || ""),
    response: String(run.response_summary || ""),
    createdAt: String(run.created_at || "—"),
    createdTs: Number.isFinite(createdTs) ? createdTs : undefined,
    source: String(run.source || "—"),
    client: String(run.client || "—"),
    model: String(run.model || "—"),
    substrate: String(run.substrate || "—"),
    duration: duration.text,
    durationMs: duration.milliseconds,
    finishReason: typeof run.finish_reason === "string" ? run.finish_reason : undefined,
    parentRunId: typeof run.parent_run_id === "string" ? run.parent_run_id : undefined,
    sessionKey: typeof run.session_key === "string" && run.session_key ? run.session_key : undefined,
    flags: Array.isArray(run.flags) ? run.flags.map(String) : [],
    warningCount: Array.isArray(run.warnings) ? run.warnings.length : 0,
    activeDialCount: Object.keys(activeDials).length,
    memoryCardCount: cards.length,
  };
}

async function getJSON(url: string, signal?: AbortSignal): Promise<unknown | null> {
  try {
    const response = await fetch(url, { signal });
    if (!response.ok) return null;
    return await response.json();
  } catch {
    return null;
  }
}

function modelName(path: unknown): string {
  const raw = String(path || "");
  const tail = raw.split(/[\\/]/).pop();
  return tail || "—";
}

export async function loadRuntimeState(signal?: AbortSignal): Promise<RuntimeState> {
  const [healthBody, runsBody, engineBody] = await Promise.all([
    getJSON("/healthz", signal),
    getJSON("/runs", signal),
    getJSON("/engine/health", signal),
  ]);
  if (!healthBody) return { status: "offline", runs: [] };

  const rowsBody = record(runsBody);
  const rows = Array.isArray(runsBody) ? records(runsBody) : records(rowsBody.runs);
  const runs = rows.map(runSummary).filter((run) => run.id);

  const engine = record(record(engineBody).engine);
  const capabilities = record(engine.capabilities);
  const layerCount = Number(engine.n_layer);
  return {
    status: "connected",
    runs,
    engine: Object.keys(engine).length ? {
      model: modelName(engine.model),
      layerCount: Number.isFinite(layerCount) ? layerCount : undefined,
      jlens: capabilities.jlens === true,
      sae: capabilities.sae === true,
    } : undefined,
  };
}

export async function loadRunFamily(runId: string, signal?: AbortSignal): Promise<RunSummary[]> {
  const body = await getJSON(`/runs/${encodeURIComponent(runId)}/family`, signal);
  if (!body) return [];
  return records(record(body).runs).map(runSummary).filter((run) => run.id);
}

export async function loadRunFacts(runId: string, signal?: AbortSignal): Promise<RunFacts> {
  const body = await getJSON(`/runs/${encodeURIComponent(runId)}`, signal);
  if (!body) return { tokenCount: 0, traceAvailable: false };
  const trace = record(record(body).trace);
  const tokens = Array.isArray(trace.tokens) ? trace.tokens : [];
  const steps = Array.isArray(trace.steps) ? trace.steps : [];
  const tokenCount = Math.max(tokens.length, steps.length);
  return {
    tokenCount,
    traceAvailable: tokenCount > 0,
  };
}

function nonnegativeNumber(...values: unknown[]): number | undefined {
  for (const value of values) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) continue;
    return value;
  }
  return undefined;
}

function performanceValue(source: string, ...values: unknown[]) {
  const value = nonnegativeNumber(...values);
  return value == null ? undefined : { value, source };
}

function diagnosisFinding(value: unknown): RunDiagnosisFinding | undefined {
  const item = record(value);
  const status = item.status;
  if (
    typeof item.id !== "string"
    || typeof item.text !== "string"
    || (status !== "observed" && status !== "not_observed" && status !== "unavailable")
  ) {
    return undefined;
  }
  return {
    id: item.id,
    status,
    text: item.text,
    evidence: records(item.evidence).flatMap((entry) => {
      if (typeof entry.path !== "string") return [];
      return [{
        path: entry.path,
        value: entry.value,
        meaning: typeof entry.meaning === "string" ? entry.meaning : undefined,
      }];
    }),
  };
}

function runDiagnosis(value: unknown): RunDiagnosis | undefined {
  const body = record(value);
  const whySlow = record(body.why_slow);
  const findings = records(whySlow.findings).flatMap((item) => {
    const finding = diagnosisFinding(item);
    return finding ? [finding] : [];
  });
  if (!findings.length && typeof whySlow.summary !== "string") return undefined;
  return {
    schema: typeof body.schema === "string" ? body.schema : "—",
    summary: typeof whySlow.summary === "string" ? whySlow.summary : "",
    findings,
    cutoff: diagnosisFinding(record(body.why_cut_off).finding),
    auxiliary: diagnosisFinding(body.client_auxiliary_calls),
  };
}

const RULE_STATUSES: readonly PerformanceRuleStatus[] = ["fired", "not_fired", "unavailable"];
const EVIDENCE_STATES: readonly PerformanceEvidenceState[] = ["observed", "correlated", "causally_supported"];

function performanceRuleEvidence(value: unknown): PerformanceRuleEvidence[] {
  return records(value).flatMap((entry) => {
    if (typeof entry.path !== "string" || !("value" in entry)) return [];
    return [{ path: entry.path, value: entry.value }];
  });
}

function performanceRuleDiagnosis(value: unknown): PerformanceRuleDiagnosis | undefined {
  const item = record(value);
  const status = item.status;
  if (
    typeof item.rule !== "string"
    || typeof item.rule_version !== "string"
    || !RULE_STATUSES.includes(status as PerformanceRuleStatus)
  ) {
    return undefined;
  }
  const evidenceState = EVIDENCE_STATES.includes(item.evidence_state as PerformanceEvidenceState)
    ? (item.evidence_state as PerformanceEvidenceState)
    : undefined;
  return {
    rule: item.rule,
    ruleVersion: item.rule_version,
    status: status as PerformanceRuleStatus,
    reason: typeof item.reason === "string" ? item.reason : undefined,
    evidenceState,
    likelyCause: typeof item.likely_cause === "string" ? item.likely_cause : undefined,
    possibleFix: typeof item.possible_fix === "string" ? item.possible_fix : undefined,
    evidence: performanceRuleEvidence(item.evidence),
  };
}

function performancePhase(value: unknown): PerformancePhase | undefined {
  const item = record(value);
  if (typeof item.name !== "string" || !item.name) return undefined;
  return {
    name: item.name,
    owner: typeof item.owner === "string" ? item.owner : undefined,
    durationNs: typeof item.duration_ns === "number" && Number.isFinite(item.duration_ns)
      ? item.duration_ns
      : undefined,
    startNs: typeof item.start_ns === "number" && Number.isFinite(item.start_ns)
      ? item.start_ns
      : undefined,
    clockOwner: typeof item.clock_owner === "string" ? item.clock_owner : undefined,
    clockDomain: typeof item.clock_domain === "string" ? item.clock_domain : undefined,
    measurement: item.measurement === "measured" || item.measurement === "estimated"
      ? item.measurement
      : undefined,
    aggregation: item.aggregation === "exclusive"
      || item.aggregation === "overlapping"
      || item.aggregation === "context_only"
      ? item.aggregation
      : undefined,
    sourceSchema: typeof item.source_schema === "string" ? item.source_schema : undefined,
    scope: typeof item.scope === "string" ? item.scope : undefined,
    includes: Array.isArray(item.includes)
      ? item.includes.filter((entry): entry is string => typeof entry === "string")
      : [],
  };
}

function performanceMetrics(value: unknown): Record<string, number> {
  const body = record(value);
  const out: Record<string, number> = {};
  for (const [key, raw] of Object.entries(body)) {
    if (typeof raw === "number" && Number.isFinite(raw)) out[key] = raw;
  }
  return out;
}

// clozn.performance-trace.v1 -- see clozn/schemas/defs/clozn.performance-trace.v1.json and
// clozn/runs/perf_diagnosis.py. Kept as its own parser (not folded into runDiagnosis above) because it is
// a genuinely different artifact with a different status vocabulary -- see types.ts's PerformanceRuleReport
// doc comment for why the two are not merged into one shape.
function performanceRuleReport(value: unknown): PerformanceRuleReport | undefined {
  const body = record(value);
  if (typeof body.schema_version !== "string" || typeof body.run_id !== "string") return undefined;
  const diagnoses = records(body.diagnoses).flatMap((item) => {
    const parsed = performanceRuleDiagnosis(item);
    return parsed ? [parsed] : [];
  });
  const aggregation = record(body.aggregation);
  const regression = record(body.regression_attribution);
  return {
    schemaVersion: body.schema_version,
    phases: records(body.phases).flatMap((item) => {
      const phase = performancePhase(item);
      return phase ? [phase] : [];
    }),
    metrics: performanceMetrics(body.metrics),
    aggregation: Object.keys(aggregation).length > 0 ? {
      knownDurationNs: nonnegativeNumber(aggregation.known_duration_ns),
      unaccountedDurationNs: nonnegativeNumber(aggregation.unaccounted_duration_ns),
      wallClockTotalNs: nonnegativeNumber(aggregation.wall_clock_total_ns),
      measurementCoverage: nonnegativeNumber(aggregation.measurement_coverage),
      phaseCount: nonnegativeNumber(aggregation.phase_count),
      exclusivePhaseCount: nonnegativeNumber(aggregation.exclusive_phase_count),
      consistency: aggregation.consistency === "consistent"
        || aggregation.consistency === "known_exceeds_wall"
        ? aggregation.consistency
        : undefined,
    } : undefined,
    regressionAttribution: typeof regression.status === "string" ? {
      status: regression.status,
      rules: Array.isArray(regression.rules)
        ? regression.rules.filter((entry): entry is string => typeof entry === "string")
        : [],
      evaluableRuleCount: nonnegativeNumber(regression.evaluable_rule_count),
      evidenceState: regression.evidence_state === "observed"
        || regression.evidence_state === "correlated"
        || regression.evidence_state === "causally_supported"
        ? regression.evidence_state
        : undefined,
    } : undefined,
    diagnoses,
  };
}

export async function loadRunPerformance(
  runId: string,
  signal?: AbortSignal,
): Promise<RunPerformance> {
  const encoded = encodeURIComponent(runId);
  const [runBody, diagnosisBody, ruleReportBody] = await Promise.all([
    getJSON(`/runs/${encoded}`, signal),
    getJSON(`/runs/${encoded}/diagnosis`, signal),
    getJSON(`/runs/${encoded}/performance`, signal),
  ]);
  if (!runBody) throw new Error("Run not found");

  const run = record(runBody);
  const timing = record(run.timing);
  const meta = record(run.meta);
  const trace = record(run.trace);
  const limits = record(record(run.context_receipt).limits);
  const totalDuration = performanceValue("timing.duration_ms", timing.duration_ms);
  const promptTokens = performanceValue(
    typeof limits.prompt_tokens === "number"
      ? "context_receipt.limits.prompt_tokens"
      : "meta.prompt_tokens",
    limits.prompt_tokens,
    meta.prompt_tokens,
  );
  const traceTokens = Array.isArray(trace.tokens) ? trace.tokens.length : undefined;
  const generatedTokens = performanceValue(
    typeof limits.generated_tokens === "number"
      ? "context_receipt.limits.generated_tokens"
      : typeof meta.generation_tokens === "number"
        ? "meta.generation_tokens"
        : "trace.tokens.length",
    limits.generated_tokens,
    meta.generation_tokens,
    traceTokens,
  );
  const generationDuration = performanceValue(
    typeof timing.generation_duration_ms === "number"
      ? "timing.generation_duration_ms"
      : typeof meta.generation_duration_ms === "number"
        ? "meta.generation_duration_ms"
        : typeof timing.eval_duration_ms === "number"
          ? "timing.eval_duration_ms"
          : "meta.eval_duration_ms",
    timing.generation_duration_ms,
    meta.generation_duration_ms,
    timing.eval_duration_ms,
    meta.eval_duration_ms,
  );
  const measuredThroughput = performanceValue(
    typeof timing.generation_tokens_per_second === "number"
      ? "timing.generation_tokens_per_second"
      : "meta.generation_tokens_per_second",
    timing.generation_tokens_per_second,
    meta.generation_tokens_per_second,
  );
  const derivedThroughput = !measuredThroughput
    && generatedTokens
    && totalDuration
    && totalDuration.value > 0
      ? {
          value: generatedTokens.value * 1000 / totalDuration.value,
          source: `${generatedTokens.source} / ${totalDuration.source}`,
        }
      : undefined;
  const gpuLayers = nonnegativeNumber(meta.gpu_layers);

  return {
    totalDuration,
    promptTokens,
    generatedTokens,
    generationDuration,
    contextWindowTokens: performanceValue(
      "context_receipt.limits.context_window_tokens",
      limits.context_window_tokens,
    ),
    throughput: measuredThroughput
      ? { ...measuredThroughput, kind: "measured_decode" }
      : derivedThroughput
        ? { ...derivedThroughput, kind: "derived_end_to_end" }
        : undefined,
    finishReason: typeof run.finish_reason === "string" ? run.finish_reason : undefined,
    device: typeof meta.device === "string" ? meta.device : undefined,
    gpuLayers,
    samplerMode: typeof meta.sampler_mode === "string"
      ? meta.sampler_mode
      : typeof meta.sampling === "string"
        ? meta.sampling
        : undefined,
    diagnosis: runDiagnosis(diagnosisBody),
    rules: performanceRuleReport(ruleReportBody),
  };
}

export async function loadRunConcepts(
  runId: string,
  layer?: number,
  signal?: AbortSignal,
): Promise<RunConcepts> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  const timeout = window.setTimeout(abort, 5000);
  try {
    const response = await fetch(`/runs/${encodeURIComponent(runId)}/jlens`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        topk: 5,
        ...(layer == null ? {} : { layer }),
      }),
      signal: controller.signal,
    });
    const body = record(await response.json());
    if (!response.ok || body.available !== true) {
      return {
        available: false,
        reason: String(body.reason ?? body.error ?? `J-lens unavailable (${response.status})`),
        availableLayers: [],
        tokens: [],
        readouts: [],
      };
    }
    const readouts = Array.isArray(body.readouts)
      ? body.readouts.map((row) => records(row).map((candidate): ConceptCandidate => ({
        piece: String(candidate.piece ?? candidate.text ?? "∅"),
        score: Number(candidate.score) || 0,
      })))
      : [];
    return {
      available: true,
      layer: Number.isInteger(Number(body.layer)) ? Number(body.layer) : undefined,
      availableLayers: Array.isArray(body.available_layers)
        ? body.available_layers.map(Number).filter(Number.isInteger)
        : [],
      tokens: Array.isArray(body.tokens) ? body.tokens.map(String) : [],
      readouts,
      textSource: typeof body.text_source === "string" ? body.text_source : undefined,
    };
  } catch (error) {
    if (signal?.aborted) throw error;
    return {
      available: false,
      reason: controller.signal.aborted
        ? "J-lens request timed out"
        : error instanceof Error ? error.message : "J-lens unavailable",
      availableLayers: [],
      tokens: [],
      readouts: [],
    };
  } finally {
    window.clearTimeout(timeout);
    signal?.removeEventListener("abort", abort);
  }
}

// The exact set clozn.receipts.context_answer_influence.ERROR_CODES emits, mirrored here so an
// unrecognized code (drift, or a future backend addition this frontend hasn't been taught about yet)
// falls back to the generic "server_error" bucket instead of being silently trusted as one of the named
// ones -- fail toward "unlabeled failure," never toward a specific claim the backend didn't make.
const INFLUENCE_ERROR_CODES = new Set<InfluenceErrorCode>([
  "invalid_run",
  "no_text_context",
  "no_recorded_continuation",
  "scoring_unavailable",
  "invalid_baseline_score",
  "intervention_score_failed",
  "influence_map_error",
]);

function isInfluenceErrorCode(value: unknown): value is InfluenceErrorCode {
  return typeof value === "string" && INFLUENCE_ERROR_CODES.has(value as InfluenceErrorCode);
}

const INFLUENCE_JOB_STATES = new Set<InfluenceMapJobState>([
  "queued",
  "running",
  "persisting",
  "cancelling",
  "completed",
  "failed",
  "cancelled",
]);

function parseInfluenceMapJob(value: unknown): InfluenceMapJob {
  const body = record(value);
  const progress = record(body.progress);
  const error = record(body.error);
  const state = body.state;
  if (
    body.schema_version !== "clozn.influence-map-job.v1"
    || typeof body.job_id !== "string"
    || typeof body.run_id !== "string"
    || typeof state !== "string"
    || !INFLUENCE_JOB_STATES.has(state as InfluenceMapJobState)
    || typeof progress.phase !== "string"
  ) {
    throw new Error("the server returned an invalid influence-map job");
  }
  return {
    schemaVersion: "clozn.influence-map-job.v1",
    jobId: body.job_id,
    runId: body.run_id,
    state: state as InfluenceMapJobState,
    progress: {
      phase: progress.phase,
      completedUnits: Number(progress.completed_units) || 0,
      totalUnits: Number(progress.total_units) || 0,
      percent: Number(progress.percent) || 0,
    },
    cancelRequested: body.cancel_requested === true,
    cancellable: body.cancellable === true,
    cached: body.cached === true,
    cancelAccepted: typeof body.cancel_accepted === "boolean" ? body.cancel_accepted : undefined,
    error: typeof error.message === "string"
      ? {
          code: typeof error.code === "string" ? error.code : undefined,
          message: error.message,
          artifactStatus: error.artifact_status === "error" ? "error"
            : error.artifact_status === "unavailable" ? "unavailable"
              : undefined,
        }
      : undefined,
  };
}

async function influenceMapJobRequest(
  runId: string,
  suffix: string,
  method: "GET" | "POST",
  signal?: AbortSignal,
): Promise<InfluenceMapJob> {
  const response = await fetch(
    `/runs/${encodeURIComponent(runId)}/influence-map/jobs${suffix}`,
    {
      method,
      headers: method === "POST" ? { "content-type": "application/json" } : undefined,
      body: method === "POST" ? "{}" : undefined,
      signal,
    },
  );
  let body: unknown = {};
  try {
    body = await response.json();
  } catch {
    // Preserve the HTTP status below if the response is not JSON.
  }
  if (!response.ok) {
    const flatError = record(body).error;
    throw new Error(
      typeof flatError === "string"
        ? flatError
        : `influence-map job request failed (${response.status})`,
    );
  }
  return parseInfluenceMapJob(body);
}

export function startRunInfluenceMapJob(
  runId: string,
  signal?: AbortSignal,
): Promise<InfluenceMapJob> {
  return influenceMapJobRequest(runId, "", "POST", signal);
}

export function loadRunInfluenceMapJob(
  runId: string,
  jobId: string,
  signal?: AbortSignal,
): Promise<InfluenceMapJob> {
  return influenceMapJobRequest(runId, `/${encodeURIComponent(jobId)}`, "GET", signal);
}

export function cancelRunInfluenceMapJob(
  runId: string,
  jobId: string,
  signal?: AbortSignal,
): Promise<InfluenceMapJob> {
  return influenceMapJobRequest(
    runId,
    `/${encodeURIComponent(jobId)}/cancel`,
    "POST",
    signal,
  );
}

/**
 * POST /runs/<id>/influence-map, preserving the backend's typed absence states instead of collapsing
 * every non-2xx response into one generic thrown Error. `clozn/server/routes/influence_map.py` returns:
 *   - 200 { available: true, ... }              -- a fresh or cached measurement.
 *   - 503 { error: "...no worker..." }          -- no worker attached at all (infra-level, this build).
 *   - 400 { error: "..." }                      -- a malformed request (bad max_context_spans).
 *   - 422 { available: false, status: "unavailable", error: { code, message } } -- a precondition on
 *     THIS run was never met (no text context, no recorded continuation, scorer unavailable).
 *   - 500 { available: false, status: "error", error: { code, message } }      -- an intervention that
 *     should have worked did not complete.
 *   - 500 { error: "..." }                      -- an infra failure with no typed code (schema
 *     validation, persistence) -- rare, but still surfaced distinctly rather than guessed at.
 */
export async function measureRunInfluenceMap(
  runId: string,
  signal?: AbortSignal,
): Promise<MeasureInfluenceResult> {
  let response: Response;
  try {
    response = await fetch(`/runs/${encodeURIComponent(runId)}/influence-map`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    return {
      ok: false,
      absence: {
        kind: "network_error",
        message: error instanceof Error ? error.message : "the measurement request failed to reach the server",
      },
    };
  }

  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // The status remains authoritative when the response has no JSON body.
  }

  if (response.ok && body.available === true) {
    const cacheHeader = response.headers.get("X-Clozn-Influence-Cache");
    return {
      ok: true,
      cache: cacheHeader === "hit" || cacheHeader === "miss" ? cacheHeader : "unknown",
    };
  }

  if (response.status === 503) {
    return { ok: false, absence: { kind: "no_worker" } };
  }

  const errorBody = record(body.error);
  const code = errorBody.code;
  const typedMessage = typeof errorBody.message === "string" ? errorBody.message : undefined;
  if ((response.status === 422 || response.status === 500) && isInfluenceErrorCode(code) && typedMessage) {
    return {
      ok: false,
      absence: {
        kind: "typed",
        code,
        status: body.status === "error" ? "error" : "unavailable",
        message: typedMessage,
      },
    };
  }

  const flatMessage = typeof body.error === "string" ? body.error : undefined;
  if (response.status === 400) {
    return {
      ok: false,
      absence: { kind: "invalid_request", message: flatMessage ?? "the measurement request was rejected" },
    };
  }
  return {
    ok: false,
    absence: {
      kind: "server_error",
      message: flatMessage ?? `source measurement failed (${response.status})`,
    },
  };
}

function spanBands(spansBody: unknown): Map<number, TokenReading["band"]> {
  const bands = new Map<number, TokenReading["band"]>();
  for (const span of records(record(spansBody).spans)) {
    const start = Number(span.start);
    const end = Number(span.end);
    const band = span.band;
    if (!Number.isInteger(start) || !Number.isInteger(end)) continue;
    if (band !== "strong" && band !== "okay" && band !== "shaky") continue;
    for (let index = start; index <= end; index += 1) bands.set(index, band);
  }
  return bands;
}

interface InfluenceIndex {
  sources: SourceReading[];
  contextSources: SourceReading[];
  coverage: ContextCoverage;
  tokenSources: Map<number, TokenSourceReading[]>;
  observedSources: Map<number, TokenSourceReading[]>;
  method?: InfluenceMethod;
  thresholds?: InfluenceThresholds;
  absence?: InfluenceAbsence;
}

// How many below-floor candidates to surface per answer token in the inspector's secondary list.
// Mirrors the backend's own top_context_span_ids cap (3) for the cleared list so neither list reads as
// artificially truncated relative to the other.
const MAX_OBSERVED_SOURCES_PER_TOKEN = 3;

function promptTokenCount(run: JsonRecord): number | undefined {
  const rawValue = record(record(run.context_receipt).limits).prompt_tokens;
  if (rawValue == null || rawValue === "") return undefined;
  const value = Number(rawValue);
  return Number.isFinite(value) && value >= 0 ? value : undefined;
}

function messageContextSources(run: JsonRecord): SourceReading[] {
  const receiptByOrder = new Map<number, JsonRecord>();
  for (const segment of records(record(run.context_receipt).delivered)) {
    const order = Number(segment.original_order);
    if (Number.isInteger(order)) receiptByOrder.set(order, segment);
  }
  return records(run.messages).flatMap((message, index) => {
    if (typeof message.content !== "string" || !message.content.trim()) return [];
    const role = String(message.role || "unknown");
    const name = typeof message.name === "string" ? message.name : "";
    const receipt = receiptByOrder.get(index) ?? {};
    const segmentId = typeof receipt.segment_id === "string" ? receipt.segment_id : undefined;
    const clientSourceId = typeof receipt.client_source_id === "string"
      ? receipt.client_source_id : undefined;
    const receiptLabel = typeof receipt.source_label === "string" ? receipt.source_label : undefined;
    return [{
      id: segmentId ?? `context.m${String(index).padStart(3, "0")}`,
      text: message.content,
      role,
      kind: role === "tool" ? "tool_result" : "message",
      label: (receiptLabel ?? clientSourceId ?? name) || undefined,
      segmentId,
      clientSourceId,
      groupId: segmentId ?? `context.m${String(index).padStart(3, "0")}`,
      messageIndex: index,
      measured: false,
      start: 0,
      end: message.content.length,
    }];
  });
}

function sourceReading(source: JsonRecord, measured: boolean, clearEffect?: boolean): SourceReading | null {
  const id = String(source.id || "");
  const text = typeof source.text === "string" ? source.text : "";
  if (!id || !text.trim()) return null;
  const messageIndex = Number(source.message_index);
  const start = Number(source.start);
  const end = Number(source.end);
  const byteStart = Number(source.byte_start);
  const byteEnd = Number(source.byte_end);
  const segmentId = typeof source.segment_id === "string" ? source.segment_id : undefined;
  const clientSourceId = typeof source.client_source_id === "string"
    ? source.client_source_id
    : typeof source.external_source_id === "string" ? source.external_source_id : undefined;
  const explicitLabel = source.source_label ?? source.name;
  const label = (
    typeof explicitLabel === "string" && explicitLabel
      ? explicitLabel
      : clientSourceId
        ?? (segmentId ? `segment ${segmentId.slice(0, 10)}` : undefined)
  );
  return {
    id,
    text,
    role: String(source.role || "unknown"),
    kind: String(source.source_kind || source.kind || "message"),
    label: typeof label === "string" && label ? label : undefined,
    segmentId,
    clientSourceId,
    groupId: String(source.parent_id || source.id || ""),
    messageIndex: Number.isInteger(messageIndex) ? messageIndex : undefined,
    measured,
    clearEffect,
    start: Number.isInteger(start) ? start : undefined,
    end: Number.isInteger(end) ? end : undefined,
    byteStart: Number.isInteger(byteStart) ? byteStart : undefined,
    byteEnd: Number.isInteger(byteEnd) ? byteEnd : undefined,
  };
}

// Fails toward the WEAKER claim: only the exact backend string counts as `causally_supported`. A
// missing, malformed, or future-added value must never be silently upgraded into the stronger claim --
// this is the structural half of "below the measurement floor must not read the same as a real effect."
function toEvidenceState(raw: unknown): InfluenceEvidenceState {
  return raw === "causally_supported" ? "causally_supported" : "observed";
}

function sanitizeInfluenceEffect(raw: unknown): TokenSourceReading["effect"] {
  return raw === "suppresses" || raw === "neutral" ? raw : "supports";
}

function influenceMethodFrom(body: JsonRecord): InfluenceMethod | undefined {
  const method = record(body.method);
  if (typeof method.mode !== "string" || typeof method.claim_limit !== "string"
      || typeof method.caveat !== "string") {
    return undefined;
  }
  return {
    name: typeof method.name === "string" ? method.name : undefined,
    mode: method.mode,
    measurement: typeof method.measurement === "string" ? method.measurement : undefined,
    sign: typeof method.sign === "string" ? method.sign : undefined,
    segmentation: typeof method.segmentation === "string" ? method.segmentation : undefined,
    redundancyCheck: typeof method.redundancy_check === "string" ? method.redundancy_check : undefined,
    claimLimit: method.claim_limit,
    caveat: method.caveat,
  };
}

function influenceThresholdsFrom(body: JsonRecord): InfluenceThresholds | undefined {
  const thresholds = record(body.thresholds);
  if (!Object.keys(thresholds).length) return undefined;
  const floor = Number(thresholds.cell_abs_delta_nats);
  return {
    cellAbsDeltaNats: Number.isFinite(floor) ? floor : undefined,
    sourceClearRule: typeof thresholds.source_clear_rule === "string" ? thresholds.source_clear_rule : undefined,
    calibration: typeof thresholds.calibration === "string" ? thresholds.calibration : undefined,
  };
}

function influenceIndex(body: unknown, run: JsonRecord): InfluenceIndex {
  const influence = record(body);
  const fallbackSources = messageContextSources(run);
  if (influence.available !== true) {
    return {
      sources: [],
      contextSources: fallbackSources,
      coverage: {
        totalSources: fallbackSources.length,
        measuredSources: 0,
        omittedSources: fallbackSources.length,
        measuredSpans: 0,
        complete: false,
        promptTokens: promptTokenCount(run),
      },
      tokenSources: new Map(),
      observedSources: new Map(),
      // GET /runs/<id>/influence-map only ever returns an already-persisted `available: true` artifact
      // or a 404 -- the server never persists a failed measurement (influence_map.py's try_post only
      // attaches the run when status == "ok"). So on the read path, "not available" always means
      // "never successfully computed," never a specific typed failure (those only surface transiently
      // from the POST call itself, see measureRunInfluenceMap / Lens.tsx's local sourceAbsence state).
      absence: { kind: "not_measured" },
    };
  }

  const promptSourceRows = records(influence.prompt_sources);
  const selection = record(influence.selection);
  const selectedSourceIds = new Set(
    Array.isArray(selection.selected_source_ids) ? selection.selected_source_ids.map(String) : [],
  );
  const omittedSourceIds = new Set(
    Array.isArray(selection.omitted_source_ids) ? selection.omitted_source_ids.map(String) : [],
  );

  const clearEffectById = new Map<string, boolean>();
  for (const entry of records(record(influence.summary).context_to_answer)) {
    const id = String(entry.context_span_id || "");
    if (id) clearEffectById.set(id, entry.clear_effect === true);
  }

  const sources = records(influence.prompt_spans).flatMap((span) => {
    const id = String(span.id || "");
    const reading = sourceReading(span, true, clearEffectById.get(id));
    return reading ? [reading] : [];
  });
  const measuredParents = new Set(sources.map((source) => source.groupId || source.id));
  const omittedSources = promptSourceRows.flatMap((source) => {
    const id = String(source.id || "");
    if (selectedSourceIds.has(id) || measuredParents.has(id)) return [];
    const reading = sourceReading(source, false);
    return reading ? [reading] : [];
  });
  const contextSources = [...sources, ...omittedSources].sort((a, b) => {
    const messageDelta = (a.messageIndex ?? -1) - (b.messageIndex ?? -1);
    if (messageDelta) return messageDelta;
    const startDelta = (a.start ?? 0) - (b.start ?? 0);
    if (startDelta) return startDelta;
    return a.id.localeCompare(b.id);
  });
  const sourceById = new Map(sources.map((source) => [source.id, source]));

  const tokenByAnswerId = new Map<string, number>();
  for (const span of records(influence.answer_spans)) {
    const index = Number(span.token_index);
    if (Number.isInteger(index)) tokenByAnswerId.set(String(span.id || ""), index);
  }

  const linkByPair = new Map<string, JsonRecord>();
  const linksByAnswerId = new Map<string, JsonRecord[]>();
  for (const link of records(influence.links)) {
    const answerId = String(link.answer_span_id || "");
    const key = `${answerId}:${String(link.context_span_id || "")}`;
    const previous = linkByPair.get(key);
    if (!previous || Number(link.abs_delta_nats) > Number(previous.abs_delta_nats)) linkByPair.set(key, link);
    const bucket = linksByAnswerId.get(answerId) ?? [];
    bucket.push(link);
    linksByAnswerId.set(answerId, bucket);
  }

  function toReading(answerId: string, sourceId: string, link: JsonRecord | undefined): TokenSourceReading | null {
    const source = sourceById.get(sourceId);
    if (!source) return null;
    return {
      sourceId,
      label: source.label ?? source.text,
      effect: sanitizeInfluenceEffect(link?.effect),
      deltaNats: Number(link?.delta_nats) || 0,
      evidenceState: toEvidenceState(link?.evidence_state),
    };
  }

  const tokenSources = new Map<number, TokenSourceReading[]>();
  const answerLinks = records(record(influence.summary).answer_to_context);
  for (const answer of answerLinks) {
    if (answer.clear_source !== true) continue;
    const answerId = String(answer.answer_span_id || "");
    const tokenIndex = tokenByAnswerId.get(answerId);
    if (tokenIndex == null) continue;
    const linked = (Array.isArray(answer.top_context_span_ids) ? answer.top_context_span_ids : [])
      .map((sourceIdValue) => toReading(answerId, String(sourceIdValue), linkByPair.get(`${answerId}:${sourceIdValue}`)))
      .filter((item): item is TokenSourceReading => item !== null);
    if (linked.length) tokenSources.set(tokenIndex, linked);
  }

  // The secondary, honestly-labeled list: every measured link that did NOT clear the floor, read
  // straight off the full `links` matrix (the backend's curated `top_context_span_ids` only ever
  // contains clearing links, so a below-floor candidate is only visible here). Ranked by magnitude so
  // the closest-to-clearing spans surface first; capped so a long context doesn't flood the inspector.
  const observedSources = new Map<number, TokenSourceReading[]>();
  for (const [answerId, tokenIndex] of tokenByAnswerId) {
    const candidates = (linksByAnswerId.get(answerId) ?? [])
      .filter((link) => toEvidenceState(link.evidence_state) === "observed")
      .sort((a, b) => Number(b.abs_delta_nats) - Number(a.abs_delta_nats))
      .slice(0, MAX_OBSERVED_SOURCES_PER_TOKEN)
      .map((link) => toReading(answerId, String(link.context_span_id || ""), link))
      .filter((item): item is TokenSourceReading => item !== null);
    if (candidates.length) observedSources.set(tokenIndex, candidates);
  }

  const totalSources = promptSourceRows.length || new Set(
    contextSources.map((source) => source.groupId || source.id),
  ).size;
  const measuredSources = selectedSourceIds.size || measuredParents.size;
  const omittedSourcesCount = omittedSourceIds.size || Math.max(0, totalSources - measuredSources);
  return {
    sources,
    contextSources,
    coverage: {
      totalSources,
      measuredSources,
      omittedSources: omittedSourcesCount,
      measuredSpans: sources.length,
      complete: omittedSourcesCount === 0
        && selection.complete_for_selected_spans !== false,
      strategy: typeof selection.strategy === "string" ? selection.strategy : undefined,
      promptTokens: promptTokenCount(run),
    },
    tokenSources,
    observedSources,
    method: influenceMethodFrom(influence),
    thresholds: influenceThresholdsFrom(influence),
  };
}

function candidateReadings(piece: string, confidence: number, rawAlternatives: unknown): CandidateReading[] {
  const chosen: CandidateReading = { token: piece || "∅", score: confidence, delta: 0 };
  const seen = new Set([chosen.token]);
  const alternatives = records(rawAlternatives).map((alternative) => {
    const score = Number(alternative.prob);
    const tokenId = Number(alternative.token_id ?? alternative.id);
    return {
      token: String(alternative.piece ?? alternative.text ?? "∅"),
      score: Number.isFinite(score) ? score : 0,
      delta: (Number.isFinite(score) ? score : 0) - confidence,
      tokenId: Number.isFinite(tokenId) ? tokenId : undefined,
    };
  }).filter((alternative) => {
    if (!alternative.token || seen.has(alternative.token)) return false;
    seen.add(alternative.token);
    return true;
  }).slice(0, 3);
  return [chosen, ...alternatives];
}

function promptText(run: JsonRecord): string {
  const messages = records(run.messages);
  const userMessages = messages.filter((message) => message.role === "user" && typeof message.content === "string");
  if (userMessages.length) return String(userMessages[userMessages.length - 1].content);
  return String(run.prompt_summary || "");
}

function durationText(run: JsonRecord): string {
  return durationFromMilliseconds(record(run.timing).duration_ms).text;
}

function adapterLabels(...values: unknown[]) {
  const labels = new Set<string>();
  const add = (value: unknown) => {
    if (typeof value === "string" && value.trim()) {
      labels.add(value.trim().split(/[\\/]/).pop() || value.trim());
      return;
    }
    if (Array.isArray(value)) {
      value.forEach(add);
      return;
    }
    const item = record(value);
    const direct = item.name ?? item.id ?? item.path ?? item.adapter;
    if (direct !== value) add(direct);
  };
  values.forEach(add);
  return [...labels];
}

function runConfiguration(run: JsonRecord): RunConfiguration {
  const behavior = record(run.behavior);
  const memory = record(run.memory);
  const meta = record(run.meta);
  const identity = record(run.identity);
  const changes = record(run.changes_applied);
  const activeDials = Object.fromEntries(
    Object.entries(record(behavior.active_dials))
      .map(([name, value]) => [name, Number(value)] as const)
      .filter(([, value]) => Number.isFinite(value)),
  );
  const rawCards = Array.isArray(memory.cards_applied)
    ? memory.cards_applied
    : Array.isArray(memory.applied_ids)
      ? memory.applied_ids
      : [];
  const memoryStrength = Number(memory.strength);
  return {
    activeDials,
    memoryCards: rawCards.map((card) => {
      const item = record(card);
      return String(item.id ?? item.text ?? card);
    }).filter(Boolean),
    memoryStrength: Number.isFinite(memoryStrength) ? memoryStrength : undefined,
    adapters: adapterLabels(
      run.adapters,
      run.adapter,
      run.lora,
      meta.adapters,
      meta.adapter,
      meta.lora,
      identity.adapters,
      identity.adapter,
      identity.lora,
      changes.adapters,
      changes.adapter,
      changes.lora,
    ),
    changes: Object.keys(changes),
  };
}

function workspaceReadouts(trace: JsonRecord): WorkspaceReadout[] {
  return records(trace.workspace_readouts).flatMap((item) => {
    if (typeof item.provider !== "string" || !item.provider) return [];
    const tokenIndex = Number(item.token_index);
    const layer = Number(item.layer);
    const position = Number(item.position);
    const topReadouts = records(item.top_readouts).flatMap((readout) => {
      const score = Number(readout.score);
      if (typeof readout.label !== "string" || !Number.isFinite(score)) return [];
      return [{ label: readout.label, score }];
    });
    return [{
      tokenIndex: Number.isInteger(tokenIndex) ? tokenIndex : undefined,
      tokenText: typeof item.token_text === "string" ? item.token_text : undefined,
      layer: Number.isInteger(layer) ? layer : undefined,
      position: Number.isInteger(position) ? position : undefined,
      provider: item.provider,
      providerType: typeof item.provider_type === "string" ? item.provider_type : undefined,
      readoutKind: typeof item.readout_kind === "string" ? item.readout_kind : undefined,
      topReadouts,
    }];
  });
}

export async function loadRunInspection(runId: string, signal?: AbortSignal): Promise<ObservatoryData> {
  const encoded = encodeURIComponent(runId);
  const [runBody, spansBody, influenceBody] = await Promise.all([
    getJSON(`/runs/${encoded}`, signal),
    getJSON(`/runs/${encoded}/spans`, signal),
    getJSON(`/runs/${encoded}/influence-map`, signal),
  ]);
  if (!runBody) throw new Error("Run not found");

  const run = record(runBody);
  const trace = record(run.trace);
  const stepRows = records(trace.steps);
  const tokenPieces = Array.isArray(trace.tokens)
    ? trace.tokens.map((token) => String(token))
    : stepRows.map((step) => String(step.piece ?? step.text ?? ""));
  const bands = spanBands(spansBody);
  const influence = influenceIndex(influenceBody, run);

  const tokens: TokenReading[] = tokenPieces.map((text, index) => {
    const confidence = numberAt(trace.confidence, index, Number(stepRows[index]?.confidence) || 0);
    const sources = influence.tokenSources.get(index) ?? [];
    const observedSources = influence.observedSources.get(index) ?? [];
    const alternatives = Array.isArray(trace.alternatives)
      ? trace.alternatives[index]
      : stepRows[index]?.alternatives;
    return {
      text,
      entropy: numberAt(trace.topk_entropy, index, Number(stepRows[index]?.topk_entropy) || 0),
      confidence,
      band: bands.get(index),
      source: sources[0]?.label,
      sources,
      observedSources,
      alternatives: candidateReadings(text, confidence, alternatives),
    };
  });

  const meta = record(run.meta);
  return {
    id: String(run.id || runId),
    label: runLabel(run),
    model: String(run.model || "—"),
    quant: String(meta.quant || "—"),
    createdAt: String(run.created_at || "—"),
    duration: durationText(run),
    mode: "run",
    prompt: promptText(run),
    response: String(run.response || run.response_summary || ""),
    parentRunId: typeof run.parent_run_id === "string" ? run.parent_run_id : undefined,
    flags: Array.isArray(run.flags) ? run.flags.map(String) : [],
    tokens,
    candidates: tokens[0]?.alternatives ?? [],
    sources: influence.sources,
    contextSources: influence.contextSources,
    contextCoverage: influence.coverage,
    influenceMethod: influence.method,
    influenceThresholds: influence.thresholds,
    influenceAbsence: influence.absence,
    workspaceReadouts: workspaceReadouts(trace),
    configuration: runConfiguration(run),
  };
}

const FORK_OUTCOME_KINDS: readonly ForkOutcomeKind[] =
  ["exact_execution_fork", "reconstructed_replay", "unavailable"];

function forkReasons(value: unknown): ForkReason[] {
  return records(value).flatMap((entry) => {
    if (typeof entry.code !== "string" || !entry.code) return [];
    return [{ code: entry.code, message: typeof entry.message === "string" ? entry.message : "" }];
  });
}

function forkExactness(value: unknown): ForkExactness | undefined {
  const item = record(value);
  if (!Object.keys(item).length) return undefined;
  const truncateTo = Number(item.truncate_to);
  return {
    regime: typeof item.regime === "string" ? item.regime : undefined,
    source: typeof item.source === "string" ? item.source : undefined,
    proofStatus: typeof item.proof_status === "string" ? item.proof_status : undefined,
    truncateTo: Number.isFinite(truncateTo) ? truncateTo : undefined,
  };
}

function forkUnchangedControl(value: unknown): ForkUnchangedControl | undefined {
  const item = record(value);
  if (typeof item.status !== "string" || !item.status) return undefined;
  const result = record(item.result);
  return {
    required: typeof item.required === "boolean" ? item.required : undefined,
    status: item.status,
    result: Object.keys(result).length ? {
      status: typeof result.status === "string" ? result.status : undefined,
      exactMatch: typeof result.exact_match === "boolean" ? result.exact_match : undefined,
      note: typeof result.note === "string" ? result.note : undefined,
    } : undefined,
  };
}

/** The intervention the worker actually confirmed, read from the child's own immutable
 * `execution_fork` receipt (never re-derived from the request the UI sent -- see ForkIntervention's
 * doc comment in data/types.ts). Only present on a completed exact fork. */
function forkIntervention(body: JsonRecord): ForkIntervention | undefined {
  const receipt = record(body.execution_fork);
  const executionChange = record(record(receipt.request).execution_change);
  const interventionReceipt = record(record(receipt.execution).intervention);
  const type = executionChange.type;
  if (typeof type !== "string" || !type) return undefined;
  const tokenId = Number(executionChange.token_id);
  return {
    type,
    tokenId: Number.isFinite(tokenId) ? tokenId : undefined,
    tokenPiece: typeof executionChange.token_piece === "string" ? executionChange.token_piece : undefined,
    restoreMode: typeof interventionReceipt.restore_mode === "string"
      ? interventionReceipt.restore_mode
      : undefined,
  };
}

/** Turn the compat_fork response body into ForkOutcome's closed union -- the ONLY place this parsing
 * happens, so every caller sees the same three shapes instead of re-deriving them from raw JSON. A
 * missing/unrecognized `outcome` value degrades to `unavailable` with a synthetic reason rather than
 * fabricating exactness the response never claimed. */
function forkOutcome(body: JsonRecord): ForkOutcome {
  const kind = FORK_OUTCOME_KINDS.includes(body.outcome as ForkOutcomeKind)
    ? (body.outcome as ForkOutcomeKind)
    : "unavailable";
  const reasons = forkReasons(body.reasons);
  switch (kind) {
    case "exact_execution_fork":
      return {
        kind,
        reasons,
        exactness: forkExactness(body.exactness) ?? {},
        unchangedControl: forkUnchangedControl(body.unchanged_control),
        intervention: forkIntervention(body),
        executionId: typeof body.execution_fork_execution_id === "string"
          ? body.execution_fork_execution_id
          : undefined,
      };
    case "reconstructed_replay":
      return {
        kind,
        reasons,
        exactness: forkExactness(body.exactness) ?? {},
        unavoidableDifferences: Array.isArray(body.unavoidable_differences)
          ? body.unavoidable_differences.map(String)
          : [],
        retokenized: body.retokenized !== false,
      };
    case "unavailable":
      return {
        kind,
        reasons: reasons.length ? reasons : [{
          code: "outcome_missing",
          message: "the server response did not include a recognized fork outcome",
        }],
        exactness: forkExactness(body.exactness),
        unchangedControl: forkUnchangedControl(body.unchanged_control),
        executionId: typeof body.execution_fork_execution_id === "string"
          ? body.execution_fork_execution_id
          : undefined,
      };
    default: {
      const exhaustive: never = kind;
      return exhaustive;
    }
  }
}

export interface ForkResult {
  outcome: ForkOutcome;
  /** Present exactly when a child run was created -- exact_execution_fork or reconstructed_replay.
   * Absent for `unavailable`: no child, nothing to navigate to or compare against. */
  child?: { id: string; parentId: string; note?: string };
}

/** Decode a fork response body -- shared by legacy `createFork` (POST /runs/<id>/fork) below and the
 * Milestone F token-workbench fork action (POST /runs/<id>/tokens/<index>/fork, see
 * data/tokenWorkbench.ts). Both endpoints ultimately return the SAME `compat_fork` shape (the outcome
 * fields at the top level, or nested on a created child run record) -- this is the one place that shape
 * is parsed, so neither caller can drift from the other. */
export type ForkArtifact = ForkResult;

export function parseForkArtifact(value: unknown): ForkArtifact {
  const body = record(value);
  const outcome = forkOutcome(body);
  if (outcome.kind === "unavailable") return { outcome };
  const id = String(body.id || "");
  if (!id) throw new Error("Fork response has no child run id");
  return {
    outcome,
    child: {
      id,
      parentId: String(body.parent_run_id || ""),
      note: typeof body.note === "string" ? body.note : undefined,
    },
  };
}

/** POST /runs/<id>/fork -- FORK-02's compatibility wrapper. Tries the exact execution-fork path first
 * whenever `tokenId` names a recorded alternative's numeric id (the only thing the exact wire can force
 * directly); the gateway itself still degrades to the text splice, or reports `unavailable`, per
 * clozn.replay.fork.compat_fork -- this function never chooses a path itself, only decodes the one the
 * gateway ran. A 422 `unavailable` response is NOT a thrown error: it is a normal, typed outcome with no
 * child, distinguished from a genuine request failure (400/500/503/network) which still throws.
 *
 * Kept for callers outside the token workbench (none remain in this Studio build -- the workbench's own
 * action tray now calls the Milestone F route instead, see data/tokenWorkbench.ts's `postForkAction` --
 * but the legacy route itself is still live server-side, so this wrapper stays available). */
export async function createFork(
  runId: string,
  position: number,
  token: string,
  tokenId?: number,
): Promise<ForkResult> {
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/fork`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(
      tokenId == null ? { position, token } : { position, token, token_id: tokenId },
    ),
  });
  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // The status below remains the authoritative failure signal.
  }
  const isTypedUnavailable = response.status === 422 && body.outcome === "unavailable";
  if (!response.ok && !isTypedUnavailable) {
    throw new Error(String(body.error || `Fork failed (${response.status})`));
  }
  const artifact = parseForkArtifact(body);
  if (artifact.child) {
    return { outcome: artifact.outcome, child: { ...artifact.child, parentId: String(body.parent_run_id || runId) } };
  }
  return artifact;
}
