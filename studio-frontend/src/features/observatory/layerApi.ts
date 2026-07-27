type JsonRecord = Record<string, unknown>;

export interface ResidualLayerEvidence {
  available: boolean;
  reason?: string;
  tokens: string[];
  norms: number[][];
  layerMean: number[];
  nLayer: number;
  nTokens: number;
  textChars: number;
  truncated: boolean;
}

export interface JLensCandidate {
  piece: string;
  score: number;
}

export interface JLensLayerEvidence {
  layer: number;
  tokens: string[];
  readouts: JLensCandidate[][];
}

export interface JLensEvidence {
  available: boolean;
  reason?: string;
  layers: JLensLayerEvidence[];
  availableLayers: number[];
  textChars: number;
  truncated: boolean;
}

export interface LayerEvidence {
  residual: ResidualLayerEvidence;
  jlens: JLensEvidence;
}

export interface CausalNode {
  layer: number;
  pos: number;
  deltaFull: number;
  deltaDirection?: number;
  concept?: string;
}

export interface CausalTraceEvidence {
  ok: boolean;
  blocked?: string;
  error?: string;
  target?: {
    pos?: number;
    piece?: string;
  };
  nodes: CausalNode[];
  candidateCount: number;
  survivorCount?: number;
  verdict?: string;
  noiseFloor?: number;
  medianAbsoluteDelta?: number;
}

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function finiteNumber(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

async function postJSON(
  url: string,
  body: JsonRecord,
  signal?: AbortSignal,
): Promise<JsonRecord> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  let value: unknown = {};
  try {
    value = await response.json();
  } catch {
    // The HTTP status below remains authoritative.
  }
  const result = record(value);
  if (!response.ok) {
    throw new Error(String(result.error || `Request failed (${response.status})`));
  }
  return result;
}

function matrix(value: unknown): number[][] {
  if (!Array.isArray(value)) return [];
  return value.map((row) => (
    Array.isArray(row)
      ? row.map(Number).filter(Number.isFinite)
      : []
  ));
}

function evenlySample(values: number[], count: number): number[] {
  if (values.length <= count) return values;
  const chosen = new Set<number>();
  for (let index = 0; index < count; index += 1) {
    chosen.add(values[Math.round((index / (count - 1)) * (values.length - 1))]);
  }
  return [...chosen];
}

function parseJLensLayer(body: JsonRecord): JLensLayerEvidence | null {
  if (body.available !== true) return null;
  const layer = finiteNumber(body.layer);
  if (layer == null) return null;
  return {
    layer,
    tokens: Array.isArray(body.tokens) ? body.tokens.map(String) : [],
    readouts: Array.isArray(body.readouts)
      ? body.readouts.map((row) => records(row).flatMap((candidate) => {
          const score = finiteNumber(candidate.score);
          const piece = candidate.piece;
          return score == null || typeof piece !== "string" ? [] : [{ piece, score }];
        }))
      : [],
  };
}

export async function loadLayerEvidence(
  text: string,
  signal?: AbortSignal,
): Promise<LayerEvidence> {
  const residualText = text.slice(0, 300);
  const lensText = text.slice(0, 600);
  const residualPromise = postJSON("/engine/layers", { text: residualText }, signal)
    .then((body): ResidualLayerEvidence => {
      const norms = matrix(body.norms);
      const nLayer = finiteNumber(body.n_layer) ?? norms.length;
      const tokens = Array.isArray(body.tokens) ? body.tokens.map(String) : [];
      const nTokens = finiteNumber(body.n_tokens) ?? tokens.length;
      return {
        available: norms.length > 0 && norms.some((row) => row.length > 0),
        reason: norms.length ? undefined : "No residual layer summary returned",
        tokens,
        norms,
        layerMean: Array.isArray(body.layer_mean)
          ? body.layer_mean.map(Number).filter(Number.isFinite)
          : [],
        nLayer,
        nTokens,
        textChars: residualText.length,
        truncated: text.length > residualText.length,
      };
    })
    .catch((error): ResidualLayerEvidence => ({
      available: false,
      reason: error instanceof Error ? error.message : "Residual layer summary unavailable",
      tokens: [],
      norms: [],
      layerMean: [],
      nLayer: 0,
      nTokens: 0,
      textChars: residualText.length,
      truncated: text.length > residualText.length,
    }));

  const lensPromise = postJSON("/jlens", { text: lensText, topk: 5 }, signal)
    .then(async (probe): Promise<JLensEvidence> => {
      if (probe.available !== true) {
        return {
          available: false,
          reason: String(probe.reason || "J-lens unavailable"),
          layers: [],
          availableLayers: [],
          textChars: lensText.length,
          truncated: text.length > lensText.length,
        };
      }
      const availableLayers = Array.isArray(probe.available_layers)
        ? probe.available_layers.map(Number).filter(Number.isFinite)
        : [];
      const probeLayer = parseJLensLayer(probe);
      const sampledLayers = evenlySample(availableLayers, 6);
      const fetched = await Promise.all(sampledLayers.map(async (layer) => {
        if (probeLayer?.layer === layer) return probeLayer;
        const result = await postJSON("/jlens", { text: lensText, layer, topk: 5 }, signal);
        return parseJLensLayer(result);
      }));
      const layers = fetched
        .filter((layer): layer is JLensLayerEvidence => layer !== null)
        .sort((a, b) => a.layer - b.layer);
      if (!layers.length && probeLayer) layers.push(probeLayer);
      return {
        available: layers.length > 0,
        reason: layers.length ? undefined : "J-lens returned no fitted layer readouts",
        layers,
        availableLayers,
        textChars: lensText.length,
        truncated: text.length > lensText.length,
      };
    })
    .catch((error): JLensEvidence => ({
      available: false,
      reason: error instanceof Error ? error.message : "J-lens unavailable",
      layers: [],
      availableLayers: [],
      textChars: lensText.length,
      truncated: text.length > lensText.length,
    }));

  const [residual, jlens] = await Promise.all([residualPromise, lensPromise]);
  return { residual, jlens };
}

export async function loadCausalTrace(
  runId: string,
  position: number,
  signal?: AbortSignal,
): Promise<CausalTraceEvidence> {
  const body = await postJSON(
    `/runs/${encodeURIComponent(runId)}/causal-trace`,
    { position },
    signal,
  );
  const accounting = record(body.accounting);
  const controls = record(body.controls);
  const target = record(body.target);
  const nodes = records(body.nodes).flatMap((node) => {
    const layer = finiteNumber(node.layer);
    const pos = finiteNumber(node.pos);
    const deltaFull = finiteNumber(node.delta_full);
    if (layer == null || pos == null || deltaFull == null) return [];
    return [{
      layer,
      pos,
      deltaFull,
      deltaDirection: finiteNumber(node.delta_dir),
      concept: typeof node.concept === "string" ? node.concept : undefined,
    }];
  }).sort((a, b) => Math.abs(b.deltaFull) - Math.abs(a.deltaFull));
  const candidateCount = Array.isArray(body.all_candidates)
    ? body.all_candidates.length
    : finiteNumber(accounting.candidates) ?? 0;
  return {
    ok: body.ok === true,
    blocked: typeof body.blocked === "string" ? body.blocked : undefined,
    error: typeof body.error === "string" ? body.error : undefined,
    target: Object.keys(target).length ? {
      pos: finiteNumber(target.pos),
      piece: typeof target.piece === "string" ? target.piece : undefined,
    } : undefined,
    nodes,
    candidateCount,
    survivorCount: finiteNumber(accounting.survivors),
    verdict: typeof controls.verdict === "string" ? controls.verdict : undefined,
    noiseFloor: finiteNumber(controls.noise_floor),
    medianAbsoluteDelta: finiteNumber(controls.median_abs),
  };
}
