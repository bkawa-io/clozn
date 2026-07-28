export type Theme = "halo" | "cathedral";

export interface TokenReading {
  text: string;
  entropy: number;
  confidence?: number;
  band?: "strong" | "okay" | "shaky";
  source?: string;
  sources?: TokenSourceReading[];
  alternatives?: CandidateReading[];
}

export interface CandidateReading {
  token: string;
  score: number;
  delta: number;
}

export interface SourceReading {
  id: string;
  text: string;
  role: string;
  kind?: string;
  label?: string;
  groupId?: string;
  messageIndex?: number;
  measured?: boolean;
  start?: number;
  end?: number;
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

export interface TokenSourceReading {
  sourceId: string;
  label: string;
  effect: "supports" | "suppresses" | "neutral";
  deltaNats: number;
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
  contextCoverage?: ContextCoverage;
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
  flags: string[];
  warningCount: number;
  activeDialCount: number;
  memoryCardCount: number;
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
}

export interface PerformanceRuleReport {
  schemaVersion: string;
  phases: PerformancePhase[];
  metrics: Record<string, number>;
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

export type ForkState =
  | { status: "idle" }
  | { status: "loading"; parentId: string }
  | { status: "error"; parentId: string; message: string }
  | { status: "success"; parentId: string; childId: string; note?: string };
