export type Theme = "halo" | "cathedral";

export interface TokenReading {
  text: string;
  entropy: number;
  confidence?: number;
  band?: "strong" | "okay" | "shaky";
  source?: string;
  /** Links that cleared the method's measurement floor -- a real, controlled intervention effect. */
  sources?: TokenSourceReading[];
  /**
   * Links that were measured but did NOT clear the floor. Kept separate from `sources` on purpose:
   * "below the measurement floor" is an honest observed reading, never the same claim as a cleared
   * effect, and never the same as "this source is irrelevant" (see InfluenceEvidenceState).
   */
  observedSources?: TokenSourceReading[];
  alternatives?: CandidateReading[];
}

export interface CandidateReading {
  token: string;
  score: number;
  delta: number;
  /** The recorded alternative's numeric token id, when the backend reported one. Threaded through so a
   * ForceToken requests can name an exact token id instead of only a text piece; the canonical
   * Time Travel result reports whether the generated observation was exact or reconstructed. */
  tokenId?: number;
}

export interface SourceReading {
  id: string;
  text: string;
  role: string;
  kind?: string;
  label?: string;
  segmentId?: string;
  clientSourceId?: string;
  groupId?: string;
  messageIndex?: number;
  measured?: boolean;
  /**
   * Whether ANY link from this source cleared the measurement floor for any answer token.
   * `undefined` means "not applicable" -- the source was never measured at all (a genuinely different
   * state from `false`, which means it WAS measured and nothing cleared).
   */
  clearEffect?: boolean;
  start?: number;
  end?: number;
  byteStart?: number;
  byteEnd?: number;
}

export interface ContextCoverage {
  totalSources: number;
  measuredSources: number;
  omittedSources: number;
  measuredSpans: number;
  complete: boolean;
  strategy?: string;
  promptTokens?: number;
}

/**
 * The roadmap's explicit-states vocabulary for one measured link (evidence_state on
 * clozn.context_answer_influence.v1's Link). `causally_supported`: cleared the measurement floor -- a
 * real, controlled intervention effect. `observed`: the intervention ran and produced a delta, but it
 * did not clear the floor -- measured absence of a strong effect, never proof of irrelevance. This is a
 * closed union on purpose: a value that is not exactly `causally_supported` must render as the WEAKER
 * claim, never be silently upgraded (see `toEvidenceState` in data/api.ts).
 */
export type InfluenceEvidenceState = "causally_supported" | "observed";

export interface TokenSourceReading {
  sourceId: string;
  label: string;
  effect: "supports" | "suppresses" | "neutral";
  deltaNats: number;
  evidenceState: InfluenceEvidenceState;
}

/** `context_answer_influence.py`'s `_METHOD` dict, camelCased. Present on every computed shape the
 * backend returns -- success or failure -- so a consumer holding it is always one field away from the
 * sentence that bounds every number this feature shows. */
export interface InfluenceMethod {
  name?: string;
  mode: string;
  measurement?: string;
  sign?: string;
  segmentation?: string;
  redundancyCheck?: string;
  claimLimit: string;
  caveat: string;
}

export interface InfluenceThresholds {
  cellAbsDeltaNats?: number;
  sourceClearRule?: string;
  calibration?: string;
}

/** The closed set of `error.code` values `clozn.receipts.context_answer_influence.ERROR_CODES` emits. */
export type InfluenceErrorCode =
  | "invalid_run"
  | "no_text_context"
  | "no_recorded_continuation"
  | "scoring_unavailable"
  | "invalid_baseline_score"
  | "intervention_score_failed"
  | "influence_map_error";

/**
 * Why the source map is unavailable, typed instead of collapsed into one generic message. A user who
 * can't tell "scoring unavailable on this build" (no_worker / scoring_unavailable) from "this run has
 * no measurable context" (no_text_context / no_recorded_continuation) cannot act on either.
 */
export type InfluenceAbsence =
  | { kind: "not_measured" }
  | { kind: "no_worker" }
  | { kind: "typed"; code: InfluenceErrorCode; status: "unavailable" | "error"; message: string }
  | { kind: "invalid_request"; message: string }
  | { kind: "server_error"; message: string }
  | { kind: "network_error"; message: string };

export type MeasureInfluenceResult =
  | { ok: true; cache: "hit" | "miss" | "unknown" }
  | { ok: false; absence: InfluenceAbsence };

export type InfluenceMapJobState =
  | "queued"
  | "running"
  | "persisting"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled";

export interface InfluenceMapJob {
  schemaVersion: "clozn.influence-map-job.v1";
  jobId: string;
  runId: string;
  state: InfluenceMapJobState;
  progress: {
    phase: string;
    completedUnits: number;
    totalUnits: number;
    percent: number;
  };
  cancelRequested: boolean;
  cancellable: boolean;
  cached: boolean;
  cancelAccepted?: boolean;
  error?: {
    code?: string;
    message: string;
    artifactStatus?: "unavailable" | "error";
  };
}

export interface WorkspaceReadout {
  tokenIndex?: number;
  tokenText?: string;
  layer?: number;
  position?: number;
  provider: string;
  providerType?: string;
  readoutKind?: string;
  topReadouts: Array<{ label: string; score: number }>;
}

export interface RunConfiguration {
  activeDials: Record<string, number>;
  memoryCards: string[];
  memoryStrength?: number;
  adapters: string[];
  changes: string[];
}

export interface ObservatoryData {
  id: string;
  label: string;
  model: string;
  quant: string;
  createdAt: string;
  duration: string;
  mode: "demo" | "run";
  prompt?: string;
  response?: string;
  parentRunId?: string;
  flags?: string[];
  tokens: TokenReading[];
  candidates: CandidateReading[];
  sources: SourceReading[];
  contextSources?: SourceReading[];
  /**
   * The delivered messages themselves, each carrying its own full text -- the reading surface.
   * `contextSources` holds the measurement's spans, which subdivide these and (once a span is
   * refined) overlap each other; rendering those as siblings repeats the prompt back at the reader
   * at two granularities. Render messages, and place spans inside them by their offsets.
   */
  contextMessages?: SourceReading[];
  contextCoverage?: ContextCoverage;
  /** Present exactly when a source map has been computed and persisted for this run. */
  influenceMethod?: InfluenceMethod;
  influenceThresholds?: InfluenceThresholds;
  /** Present exactly when `sources` is empty -- explains why, instead of leaving the reader to guess. */
  influenceAbsence?: InfluenceAbsence;
  workspaceReadouts?: WorkspaceReadout[];
  configuration: RunConfiguration;
}

export interface RunSummary {
  id: string;
  label: string;
  prompt: string;
  response: string;
  createdAt: string;
  createdTs?: number;
  source: string;
  client: string;
  model: string;
  substrate: string;
  duration: string;
  durationMs?: number;
  finishReason?: string;
  parentRunId?: string;
  /** `runs.session_key` (F1) -- omitted for a run with no session, exactly as
   * `clozn.runs.sessions`'s own "sessionless runs" contract treats that as an ordinary state, never an
   * error. Present so `features/runs/Runs.tsx` can link a selected run into F3's conversation
   * investigation view without a second request. */
  sessionKey?: string;
  flags: string[];
  warningCount: number;
  activeDialCount: number;
  memoryCardCount: number;
  /* Confidence shape from the run index (clozn/runs/summaries.py::confidence_facts). ALL OPTIONAL,
     and absent together: a run with no recorded trace sends none of these keys rather than zeros,
     because 0 is a real and terrible confidence value while absence is not a value. Never default
     them to 0 at a call site -- that is the same lie in TypeScript. */
  tokenCount?: number;
  confidence?: number[];
  confidenceMin?: number;
  confidenceMean?: number;
  lowConfidenceCount?: number;
}

export interface RunFacts {
  tokenCount: number;
  traceAvailable: boolean;
}

export type DiagnosisStatus = "observed" | "not_observed" | "unavailable";

export interface DiagnosisEvidence {
  path: string;
  value: unknown;
  meaning?: string;
}

export interface RunDiagnosisFinding {
  id: string;
  status: DiagnosisStatus;
  text: string;
  evidence: DiagnosisEvidence[];
}

export interface RunDiagnosis {
  schema: string;
  summary: string;
  findings: RunDiagnosisFinding[];
  cutoff?: RunDiagnosisFinding;
  auxiliary?: RunDiagnosisFinding;
}

export interface RecordedPerformanceValue {
  value: number;
  source: string;
}

export interface RunThroughput extends RecordedPerformanceValue {
  kind: "measured_decode" | "derived_end_to_end";
}

export interface RunPerformance {
  totalDuration?: RecordedPerformanceValue;
  promptTokens?: RecordedPerformanceValue;
  generatedTokens?: RecordedPerformanceValue;
  generationDuration?: RecordedPerformanceValue;
  contextWindowTokens?: RecordedPerformanceValue;
  throughput?: RunThroughput;
  finishReason?: string;
  device?: string;
  gpuLayers?: number;
  samplerMode?: string;
  diagnosis?: RunDiagnosis;
  rules?: PerformanceRuleReport;
}

// -- clozn.performance-trace.v1 (GET /runs/<id>/performance) --------------------------------------
//
// A DIFFERENT vocabulary from RunDiagnosis above, on purpose: RunDiagnosis's observed/not_observed/
// unavailable is clozn.runs.diagnosis's per-phase evidence read. PerformanceRuleReport's fired/not_fired/
// unavailable is clozn.runs.perf_diagnosis's versioned rule engine -- it names a likely cause and possible
// fix, not just a phase's presence. Keeping the two status enums as distinct TS unions (rather than
// widening one to fit both) makes it a type error to render a rule-engine entry with the diagnosis
// component's status classes, or vice versa.
export type PerformanceRuleStatus = "fired" | "not_fired" | "unavailable";

// This artifact's rule engine never emits "causally_supported" -- no rule here re-runs a request with one
// variable changed, which is the only thing that would earn it (see clozn/runs/perf_diagnosis.py's module
// docstring). The type still allows the value for forward compatibility with the schema, which reserves it
// for a future replay-backed rule; the UI must not render "correlated" as if it meant "causally_supported".
export type PerformanceEvidenceState = "observed" | "correlated" | "causally_supported";

export interface PerformanceRuleEvidence {
  path: string;
  value: unknown;
}

export interface PerformanceRuleDiagnosis {
  rule: string;
  ruleVersion: string;
  status: PerformanceRuleStatus;
  /** Present when status is "unavailable" (what evidence is missing) or "not_fired" (what was checked). */
  reason?: string;
  /** Present when status is "fired". Never stronger than "correlated" from this module -- see above. */
  evidenceState?: PerformanceEvidenceState;
  likelyCause?: string;
  possibleFix?: string;
  evidence: PerformanceRuleEvidence[];
}

export interface PerformancePhase {
  name: string;
  owner?: string;
  durationNs?: number;
  startNs?: number;
  clockOwner?: string;
  clockDomain?: string;
  measurement?: "measured" | "estimated";
  aggregation?: "exclusive" | "overlapping" | "context_only";
  sourceSchema?: string;
  scope?: string;
  includes: string[];
}

export interface PerformanceAggregation {
  knownDurationNs?: number;
  unaccountedDurationNs?: number;
  wallClockTotalNs?: number;
  measurementCoverage?: number;
  phaseCount?: number;
  exclusivePhaseCount?: number;
  consistency?: "consistent" | "known_exceeds_wall";
}

export interface PerformanceRuleReport {
  schemaVersion: string;
  phases: PerformancePhase[];
  metrics: Record<string, number>;
  aggregation?: PerformanceAggregation;
  regressionAttribution?: {
    status: string;
    rules: string[];
    evaluableRuleCount?: number;
    evidenceState?: PerformanceEvidenceState;
  };
  /** Always one entry per rule the engine knows about (see the backend module docstring) when this run's
   * trace could be built at all -- absence of this whole object means the fetch itself failed or the run
   * could not produce a trace, not that every rule was clean. The UI must render those two cases visibly
   * differently. */
  diagnoses: PerformanceRuleDiagnosis[];
}

export interface ConceptCandidate {
  piece: string;
  score: number;
}

export interface RunConcepts {
  available: boolean;
  reason?: string;
  layer?: number;
  availableLayers: number[];
  tokens: string[];
  readouts: ConceptCandidate[][];
  textSource?: string;
}

export interface RuntimeState {
  status: "checking" | "connected" | "offline";
  runs: RunSummary[];
  engine?: {
    model: string;
    layerCount?: number;
    jlens: boolean;
    sae: boolean;
  };
}
