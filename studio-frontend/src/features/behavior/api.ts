export interface BehaviorAxis {
  name: string;
  poles: [string, string];
  value: number;
  max: number;
  calibrated: boolean;
  library?: boolean;
  custom?: boolean;
}

export interface SamplingSettings {
  sampling: boolean;
  sample_temperature: number;
  sample_top_p: number;
  sample_top_k: number;
  sample_repeat_penalty: number;
}

export interface GuardSettings {
  enabled: boolean;
  guard: {
    concepts?: string[];
    threshold?: number;
    counter_strength?: number;
    max_fires?: number;
    layer?: number;
  } | null;
}

export interface BehaviorProfile {
  name: string;
  description: string;
  dials: Record<string, number>;
}

export interface BehaviorWorkspaceData {
  axes: BehaviorAxis[];
  sampling?: SamplingSettings;
  guard?: GuardSettings;
  profiles: BehaviorProfile[];
  activeProfile?: string;
  errors: Partial<Record<"axes" | "sampling" | "guard" | "profiles", string>>;
}

export interface AxisPreview {
  prompt: string;
  axis: string;
  value: number;
  baseline: string;
  steered: string;
  warning?: string;
}

export interface ConceptPreview {
  prompt: string;
  concept: string;
  strength: number;
  baseline: string;
  steered?: string;
  note?: string;
}

export type CorrectiveScope = "once" | "session" | "profile";
export type CorrectiveBackend = "prompt_policy" | "control_vector";

export interface CorrectiveBackendEntry {
  type: string;
  requested_type?: string;
  available?: boolean;
  qualification?: string;
  qualification_id?: string;
  reason?: string;
  unavailability_reason?: string;
}

export interface CorrectiveScopeEligibility {
  scope: CorrectiveScope;
  available: boolean;
  target?: string;
  prior_hash?: string;
  before?: string[];
  after?: string[];
  note?: string;
  unavailability_reason?: string;
}

export interface CorrectiveAction {
  id: string;
  label: string;
  description: string;
  scopes: CorrectiveScope[];
  backends: CorrectiveBackendEntry[];
  scope_eligibility: CorrectiveScopeEligibility[];
}

export interface CorrectiveRegistry {
  schema_version: string;
  version: string;
  run_id: string;
  actions: CorrectiveAction[];
}

export interface CorrectivePreviewReceipt {
  preview_id: string;
  status: string;
  parent_run_id: string;
  action: Pick<CorrectiveAction, "id" | "label" | "description">;
  execution: {
    requested_backend: CorrectiveBackend;
    expected_executed_backend: CorrectiveBackend;
    expected_fallback: boolean;
    qualification?: string;
    qualification_id?: string;
    unavailability_reason?: string;
  };
  scope_eligibility: CorrectiveScopeEligibility[];
}

export interface CorrectiveChildOutcome {
  status: "success" | "error" | "not_run";
  run_id?: string;
  error?: { code: string; message: string };
}

export interface CorrectiveResult {
  schema_version: string;
  result_id: string;
  parent_run_id: string;
  preview_id: string;
  action: Pick<CorrectiveAction, "id" | "label" | "description">;
  user_intent: { action_id: string; kept_scope?: CorrectiveScope };
  execution: {
    requested_backend: CorrectiveBackend;
    executed_backend?: CorrectiveBackend | null;
    fallback: boolean;
    qualification?: string | null;
    qualification_id?: string | null;
  };
  children: {
    baseline: CorrectiveChildOutcome;
    corrected: CorrectiveChildOutcome;
  };
  comparison: {
    stored_original_reply: string;
    baseline_reply: string;
    corrected_reply: string;
    note: string;
    changed: boolean;
  };
  metrics: Record<string, unknown>;
  coherence?: { degenerate?: boolean; reasons?: string[] };
  intervention_observed?: boolean;
  scope_eligibility: CorrectiveScopeEligibility[];
  outcome: { status: "succeeded" | "execution_error" | "cancelled"; note?: string };
  transaction?: {
    id: string;
    scope: CorrectiveScope;
    target: string;
    undone_ts?: number | null;
  };
  source_use_comparison?: SourceUseComparison;
}

export interface SourceUseComparison {
  status: "available";
  baseline: {
    answer_span_count: number;
    answer_spans_with_clear_source: number;
    observed_source_dependence_ratio: number;
  };
  corrected: {
    answer_span_count: number;
    answer_spans_with_clear_source: number;
    observed_source_dependence_ratio: number;
  };
  delta_observed_source_dependence_ratio: number;
  caveat: string;
}

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? value as JsonRecord : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function errorText(body: JsonRecord, status: number) {
  const error = body.error;
  if (typeof error === "string") return error;
  const nested = record(error);
  if (typeof nested.message === "string") return nested.message;
  if (typeof body.blocked === "string") return body.blocked;
  if (typeof body.reason === "string") return body.reason;
  if (typeof body.note === "string") return body.note;
  return `Request failed (${status})`;
}

async function request(
  url: string,
  options: RequestInit = {},
): Promise<JsonRecord> {
  const response = await fetch(url, options);
  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // The HTTP status remains authoritative when the route returns no JSON.
  }
  if (!response.ok || body.error) throw new Error(errorText(body, response.status));
  return body;
}

async function get(url: string, signal?: AbortSignal) {
  return request(url, { signal });
}

async function post(url: string, body: JsonRecord, signal?: AbortSignal) {
  return request(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
}

function axisFromBody(axis: JsonRecord): BehaviorAxis | null {
  const name = String(axis.name || "");
  const rawPoles = Array.isArray(axis.poles) ? axis.poles.map(String) : [];
  if (!name || rawPoles.length !== 2) return null;
  const max = Number(axis.max);
  const value = Number(axis.value);
  return {
    name,
    poles: [rawPoles[0], rawPoles[1]],
    max: Number.isFinite(max) && max > 0 ? max : 1.5,
    value: Number.isFinite(value) ? value : 0,
    calibrated: axis.calibrated === true,
    library: axis.library === true,
    custom: axis.custom === true,
  };
}

function samplingFromBody(body: JsonRecord): SamplingSettings {
  return {
    sampling: body.sampling === true,
    sample_temperature: Number(body.sample_temperature) || 0,
    sample_top_p: Number(body.sample_top_p) || 0,
    sample_top_k: Number(body.sample_top_k) || 0,
    sample_repeat_penalty: Number(body.sample_repeat_penalty) || 0,
  };
}

function guardFromBody(body: JsonRecord): GuardSettings {
  const source = record(body.guard);
  const concepts = Array.isArray(source.concepts)
    ? source.concepts.map(String).filter(Boolean)
    : undefined;
  const numberOrUndefined = (value: unknown) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  };
  return {
    enabled: body.enabled === true,
    guard: body.guard && typeof body.guard === "object"
      ? {
          concepts,
          threshold: numberOrUndefined(source.threshold),
          counter_strength: numberOrUndefined(source.counter_strength),
          max_fires: numberOrUndefined(source.max_fires),
          layer: numberOrUndefined(source.layer),
        }
      : null,
  };
}

function profilesFromBody(body: JsonRecord): BehaviorProfile[] {
  return records(body.profiles).map((profile) => ({
    name: String(profile.name || ""),
    description: String(profile.description || ""),
    dials: Object.fromEntries(
      Object.entries(record(profile.dials))
        .map(([name, value]) => [name, Number(value)])
        .filter((entry) => Number.isFinite(entry[1])),
    ),
  })).filter((profile) => profile.name);
}

export async function loadBehaviorWorkspace(signal?: AbortSignal): Promise<BehaviorWorkspaceData> {
  const results = await Promise.allSettled([
    post("/steer/axes", {}, signal),
    get("/sampling/mode", signal),
    get("/guard/mode", signal),
    get("/profiles/list", signal),
  ]);
  const [axes, sampling, guard, profiles] = results;
  const errors: BehaviorWorkspaceData["errors"] = {};
  if (axes.status === "rejected") errors.axes = axes.reason instanceof Error ? axes.reason.message : "Axes unavailable";
  if (sampling.status === "rejected") errors.sampling = sampling.reason instanceof Error ? sampling.reason.message : "Sampling unavailable";
  if (guard.status === "rejected") errors.guard = guard.reason instanceof Error ? guard.reason.message : "Guard unavailable";
  if (profiles.status === "rejected") {
    errors.profiles = profiles.reason instanceof Error ? profiles.reason.message : "Profiles unavailable";
  }

  const profileBody = profiles.status === "fulfilled" ? profiles.value : {};
  return {
    axes: axes.status === "fulfilled"
      ? records(axes.value.axes).map(axisFromBody).filter((axis): axis is BehaviorAxis => axis !== null)
      : [],
    sampling: sampling.status === "fulfilled" ? samplingFromBody(sampling.value) : undefined,
    guard: guard.status === "fulfilled" ? guardFromBody(guard.value) : undefined,
    profiles: profilesFromBody(profileBody),
    activeProfile: typeof profileBody.active === "string" ? profileBody.active : undefined,
    errors,
  };
}

export async function applyAxis(name: string, value: number) {
  const body = await post("/steer/set", { name, value });
  return {
    active: Object.fromEntries(
      Object.entries(record(body.active))
        .map(([axis, next]) => [axis, Number(next)])
        .filter((entry) => Number.isFinite(entry[1])),
    ),
    warning: typeof body.warning === "string" ? body.warning : undefined,
  };
}

export async function previewAxis(name: string, value: number, prompt: string): Promise<AxisPreview> {
  const body = await post("/steer/check", { name, value, prompt });
  return {
    prompt: String(body.prompt || prompt),
    axis: String(body.axis || name),
    value: Number(body.value) || value,
    baseline: String(body.baseline || ""),
    steered: String(body.steered || ""),
    warning: typeof body.warning === "string" ? body.warning : undefined,
  };
}

export async function applyConcept(concept: string, strength: number) {
  const body = await post("/steer/concept/set", { concept, strength });
  if (body.ok !== true) throw new Error(errorText(body, 200));
  return Object.fromEntries(
    Object.entries(record(body.active))
      .map(([name, value]) => [name, Number(value)])
      .filter((entry) => Number.isFinite(entry[1])),
  );
}

export async function previewConcept(
  concept: string,
  strength: number,
  prompt: string,
): Promise<ConceptPreview> {
  const body = await post("/steer/concept/check", { concept, strength, prompt });
  if (body.steered == null) throw new Error(errorText(body, 200));
  return {
    prompt: String(body.prompt || prompt),
    concept: String(body.concept || concept),
    strength: Number(body.strength) || strength,
    baseline: String(body.baseline || ""),
    steered: String(body.steered || ""),
    note: typeof body.note === "string" ? body.note : undefined,
  };
}

export async function saveSampling(settings: SamplingSettings): Promise<SamplingSettings> {
  return samplingFromBody(await post("/sampling/mode", settings as unknown as JsonRecord));
}

export async function saveGuard(settings: GuardSettings): Promise<GuardSettings> {
  const body = await post("/guard/mode", {
    enabled: settings.enabled,
    concepts: settings.guard?.concepts ?? [],
    ...(settings.guard?.threshold == null ? {} : { threshold: settings.guard.threshold }),
    ...(settings.guard?.counter_strength == null
      ? {}
      : { counter_strength: settings.guard.counter_strength }),
    ...(settings.guard?.max_fires == null ? {} : { max_fires: settings.guard.max_fires }),
    ...(settings.guard?.layer == null ? {} : { layer: settings.guard.layer }),
  });
  return guardFromBody(body);
}

export async function saveProfile(
  name: string,
  description: string,
  axes: BehaviorAxis[],
) {
  const body = await post("/profiles/save", {
    version: 1,
    name,
    description,
    dials: Object.fromEntries(axes.map((axis) => [axis.name, axis.value])),
  });
  return record(body.profile);
}

export async function switchProfile(name: string) {
  return post("/profiles/switch", { name });
}

export async function loadCorrectiveActions(
  runId: string,
  signal?: AbortSignal,
): Promise<CorrectiveRegistry> {
  return await get(
    `/runs/${encodeURIComponent(runId)}/corrective-actions`,
    signal,
  ) as unknown as CorrectiveRegistry;
}

export async function previewCorrectiveAction(
  runId: string,
  actionId: string,
  requestedBackend: CorrectiveBackend,
): Promise<CorrectivePreviewReceipt> {
  return await post(
    `/runs/${encodeURIComponent(runId)}/corrective-actions/preview`,
    { action_id: actionId, requested_backend: requestedBackend },
  ) as unknown as CorrectivePreviewReceipt;
}

export async function confirmCorrectivePreview(
  previewId: string,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<CorrectiveResult> {
  return await post(
    `/corrective-previews/${encodeURIComponent(previewId)}/confirm`,
    { idempotency_key: idempotencyKey },
    signal,
  ) as unknown as CorrectiveResult;
}

export async function cancelCorrectivePreview(previewId: string) {
  return post(`/corrective-previews/${encodeURIComponent(previewId)}/cancel`, {});
}

export async function keepCorrectiveResult(
  resultId: string,
  scope: CorrectiveScope,
  expectedPriorHash: string,
  idempotencyKey: string,
): Promise<CorrectiveResult> {
  return await post(
    `/corrective-results/${encodeURIComponent(resultId)}/keep`,
    {
      scope,
      expected_prior_hash: expectedPriorHash,
      idempotency_key: idempotencyKey,
    },
  ) as unknown as CorrectiveResult;
}

export async function undoCorrectiveKeep(transactionId: string) {
  return post(`/corrective-actions/${encodeURIComponent(transactionId)}/undo`, {});
}

export async function measureCorrectiveSourceUse(
  result: CorrectiveResult,
): Promise<SourceUseComparison> {
  const baseline = result.children.baseline.run_id;
  const corrected = result.children.corrected.run_id;
  if (!baseline || !corrected) throw new Error("Both child runs are required.");
  // Explicit, separately-triggered expensive work. The result comparison route itself is a pure
  // read of maps produced here and refuses mismatched method/version/threshold contracts.
  await Promise.all([
    post(`/runs/${encodeURIComponent(baseline)}/influence-map`, {}),
    post(`/runs/${encodeURIComponent(corrected)}/influence-map`, {}),
  ]);
  return await post(
    `/corrective-results/${encodeURIComponent(result.result_id)}/source-use`,
    {},
  ) as unknown as SourceUseComparison;
}
