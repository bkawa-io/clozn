import type {
  CandidateReading,
  ConceptCandidate,
  ContextCoverage,
  ObservatoryData,
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

export async function loadRunPerformance(
  runId: string,
  signal?: AbortSignal,
): Promise<RunPerformance> {
  const encoded = encodeURIComponent(runId);
  const [runBody, diagnosisBody] = await Promise.all([
    getJSON(`/runs/${encoded}`, signal),
    getJSON(`/runs/${encoded}/diagnosis`, signal),
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

export async function measureRunInfluenceMap(
  runId: string,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/influence-map`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: "{}",
    signal,
  });
  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // The status remains authoritative when the response has no JSON body.
  }
  if (!response.ok || body.available !== true) {
    throw new Error(String(body.reason ?? body.error ?? `Source measurement failed (${response.status})`));
  }
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
}

function promptTokenCount(run: JsonRecord): number | undefined {
  const rawValue = record(record(run.context_receipt).limits).prompt_tokens;
  if (rawValue == null || rawValue === "") return undefined;
  const value = Number(rawValue);
  return Number.isFinite(value) && value >= 0 ? value : undefined;
}

function messageContextSources(run: JsonRecord): SourceReading[] {
  return records(run.messages).flatMap((message, index) => {
    if (typeof message.content !== "string" || !message.content.trim()) return [];
    const role = String(message.role || "unknown");
    const name = typeof message.name === "string" ? message.name : "";
    return [{
      id: `context.m${String(index).padStart(3, "0")}`,
      text: message.content,
      role,
      kind: role === "tool" ? "tool_result" : "message",
      label: name || undefined,
      groupId: `context.m${String(index).padStart(3, "0")}`,
      messageIndex: index,
      measured: false,
      start: 0,
      end: message.content.length,
    }];
  });
}

function sourceReading(source: JsonRecord, measured: boolean): SourceReading | null {
  const id = String(source.id || "");
  const text = typeof source.text === "string" ? source.text : "";
  if (!id || !text.trim()) return null;
  const messageIndex = Number(source.message_index);
  const start = Number(source.start);
  const end = Number(source.end);
  const label = source.name ?? source.external_source_id;
  return {
    id,
    text,
    role: String(source.role || "unknown"),
    kind: String(source.source_kind || source.kind || "message"),
    label: typeof label === "string" && label ? label : undefined,
    groupId: String(source.parent_id || source.id || ""),
    messageIndex: Number.isInteger(messageIndex) ? messageIndex : undefined,
    measured,
    start: Number.isInteger(start) ? start : undefined,
    end: Number.isInteger(end) ? end : undefined,
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
  const sources = records(influence.prompt_spans).flatMap((span) => {
    const reading = sourceReading(span, true);
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
  for (const link of records(influence.links)) {
    const key = `${String(link.answer_span_id || "")}:${String(link.context_span_id || "")}`;
    const previous = linkByPair.get(key);
    if (!previous || Number(link.abs_delta_nats) > Number(previous.abs_delta_nats)) linkByPair.set(key, link);
  }

  const tokenSources = new Map<number, TokenSourceReading[]>();
  const answerLinks = records(record(influence.summary).answer_to_context);
  for (const answer of answerLinks) {
    if (answer.clear_source !== true) continue;
    const answerId = String(answer.answer_span_id || "");
    const tokenIndex = tokenByAnswerId.get(answerId);
    if (tokenIndex == null) continue;
    const linked = (Array.isArray(answer.top_context_span_ids) ? answer.top_context_span_ids : [])
      .map((sourceIdValue): TokenSourceReading | null => {
        const sourceId = String(sourceIdValue);
        const source = sourceById.get(sourceId);
        if (!source) return null;
        const link = linkByPair.get(`${answerId}:${sourceId}`);
        const effect = link?.effect;
        return {
          sourceId,
          label: source.text,
          effect: effect === "suppresses" || effect === "neutral" ? effect : "supports",
          deltaNats: Number(link?.delta_nats) || 0,
        };
      })
      .filter((item): item is TokenSourceReading => item !== null);
    if (linked.length) tokenSources.set(tokenIndex, linked);
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
  };
}

function candidateReadings(piece: string, confidence: number, rawAlternatives: unknown): CandidateReading[] {
  const chosen: CandidateReading = { token: piece || "∅", score: confidence, delta: 0 };
  const seen = new Set([chosen.token]);
  const alternatives = records(rawAlternatives).map((alternative) => {
    const score = Number(alternative.prob);
    return {
      token: String(alternative.piece ?? alternative.text ?? "∅"),
      score: Number.isFinite(score) ? score : 0,
      delta: (Number.isFinite(score) ? score : 0) - confidence,
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
    workspaceReadouts: workspaceReadouts(trace),
    configuration: runConfiguration(run),
  };
}

export async function createFork(
  runId: string,
  position: number,
  token: string,
): Promise<{ id: string; parentId: string; note?: string }> {
  const response = await fetch(`/runs/${encodeURIComponent(runId)}/fork`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ position, token }),
  });
  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // The status below remains the authoritative failure signal.
  }
  if (!response.ok || body.error) {
    throw new Error(String(body.error || `Fork failed (${response.status})`));
  }
  const id = String(body.id || "");
  if (!id) throw new Error("Fork response has no child run id");
  return {
    id,
    parentId: String(body.parent_run_id || runId),
    note: typeof body.note === "string" ? body.note : undefined,
  };
}
