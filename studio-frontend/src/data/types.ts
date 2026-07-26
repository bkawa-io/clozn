export type Theme = "halo" | "cathedral";

export interface LayerReading {
  layer: number;
  stage: string;
  activation: number;
  energy: number;
  stability: number;
  features: number;
  hue: "cyan" | "mint" | "violet" | "pink" | "magenta" | "peach";
}

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
}

export interface TokenSourceReading {
  sourceId: string;
  label: string;
  effect: "supports" | "suppresses" | "neutral";
  deltaNats: number;
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
  layerEvidence: "demo" | "measured" | "unavailable";
  layerReason?: string;
  layers: LayerReading[];
  tokens: TokenReading[];
  candidates: CandidateReading[];
  sources: SourceReading[];
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
  };
}

export type ForkState =
  | { status: "idle" }
  | { status: "loading"; parentId: string }
  | { status: "error"; parentId: string; message: string }
  | { status: "success"; parentId: string; childId: string; note?: string };
