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
