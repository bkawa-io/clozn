import type {
  Aggregates,
  BrokenResult,
  CaseDef,
  CellRun,
  ExperimentDetail,
  ExperimentList,
  ExperimentListEntry,
  ExperimentSummary,
  FullCell,
  Manifest,
  RunIdentity,
  StatusCounts,
  SuiteAggregate,
  ThinCell,
  VariantComparison,
  VariantDef,
} from "./types";

/**
 * Adapter over `GET /experiment-results[...]` (`clozn/server/routes/experiment_results.py`). Normalizes
 * the server's snake_case JSON into the camelCase shapes in `./types.ts`, tolerating missing/malformed
 * fields the same way `data/api.ts` and `features/model/api.ts` do -- an absent or wrongly-typed field
 * becomes `undefined`/`null`/an empty collection, never a fabricated default.
 *
 * ON NOT RECOMPUTING THE SUMMARY CLIENT-SIDE
 * -------------------------------------------
 * `clozn.experiments.suite.validate_result` already recomputes `summary` from `cells` on every server
 * read (`_summarize(cells, ...) == result["summary"]` is enforced before a result is ever served -- see
 * `clozn/experiments/suite.py:258-262`), so a stored summary reaching this adapter is already verified
 * against its own cells. Recomputing it again here would duplicate that check with a second, easier-to-
 * drift implementation for no additional honesty. This adapter therefore trusts `summary.aggregates` /
 * `summary.comparisons` verbatim and only reformats them for display (see `format.ts`) -- it does not
 * re-derive pass/fail counts from cells.
 */

type JsonRecord = Record<string, unknown>;

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : {};
}

function str(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

function num(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function list<T>(value: unknown, map: (item: unknown) => T): T[] {
  return Array.isArray(value) ? value.map(map) : [];
}

function errorText(body: JsonRecord, status: number): string {
  return typeof body.error === "string" ? body.error : `Request failed (${status})`;
}

async function get(url: string, signal?: AbortSignal): Promise<JsonRecord> {
  const response = await fetch(url, { signal });
  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // The HTTP status remains authoritative when the route returns no JSON body.
  }
  if (!response.ok) throw new Error(errorText(body, response.status));
  return body;
}

// --- normalizers -----------------------------------------------------------------------------

function statusCounts(value: unknown): StatusCounts {
  const c = record(value);
  return { pass: num(c.pass) ?? 0, fail: num(c.fail) ?? 0, error: num(c.error) ?? 0, unscored: num(c.unscored) ?? 0 };
}

function suiteAggregate(value: unknown): SuiteAggregate {
  const a = record(value);
  return { runs: num(a.runs) ?? 0, counts: statusCounts(a.counts), passRate: num(a.pass_rate) };
}

function aggregates(value: unknown): Aggregates {
  const out: Aggregates = {};
  for (const [variant, rawSuites] of Object.entries(record(value))) {
    const suites = record(rawSuites);
    const entry: Aggregates[string] = {};
    if (suites.target) entry.target = suiteAggregate(suites.target);
    if (suites.guard) entry.guard = suiteAggregate(suites.guard);
    out[variant] = entry;
  }
  return out;
}

function comparisonLabel(value: unknown): { case: string; seed: number } {
  const l = record(value);
  return { case: str(l.case) ?? "", seed: num(l.seed) ?? 0 };
}

function comparison(value: unknown): VariantComparison {
  const c = record(value);
  return {
    variant: str(c.variant) ?? "",
    baseline: str(c.baseline) ?? "",
    targetGains: list(c.target_gains, comparisonLabel),
    targetRegressions: list(c.target_regressions, comparisonLabel),
    guardRegressions: list(c.guard_regressions, comparisonLabel),
    guardFixes: list(c.guard_fixes, comparisonLabel),
    changedUnscored: list(c.changed_unscored, (item) => {
      const l = record(item);
      return { case: str(l.case) ?? "", seed: num(l.seed) ?? 0, suite: str(l.suite) ?? "" };
    }),
  };
}

function summary(value: unknown): ExperimentSummary {
  const s = record(value);
  return {
    baselineVariant: str(s.baseline_variant),
    aggregates: aggregates(s.aggregates),
    comparisons: list(s.comparisons, comparison),
  };
}

function listEntry(value: unknown): ExperimentListEntry {
  const e = record(value);
  return {
    experimentId: str(e.experiment_id) ?? "",
    name: str(e.name) ?? "(unnamed)",
    createdAt: str(e.created_at),
    baselineVariant: str(e.baseline_variant),
    variants: list(e.variants, (v) => String(v)),
    seeds: list(e.seeds, (v) => num(v) ?? 0),
    cellCount: num(e.cell_count) ?? 0,
    aggregates: e.aggregates ? aggregates(e.aggregates) : null,
  };
}

function assertion(value: unknown) {
  const a = record(value);
  return { check: str(a.check) ?? "", status: str(a.status) ?? "" };
}

function thinCell(value: unknown): ThinCell {
  const c = record(value);
  return {
    suite: str(c.suite) ?? "",
    case: str(c.case) ?? "",
    variant: str(c.variant) ?? "",
    variantKind: str(c.variant_kind),
    seed: num(c.seed) ?? 0,
    status: str(c.status) ?? "unscored",
    runId: str(c.run_id),
    assertions: list(c.assertions, assertion),
    minConfidence: num(c.min_confidence),
    error: str(c.error),
  };
}

function caseDef(value: unknown): CaseDef {
  const c = record(value);
  return {
    name: str(c.name) ?? "",
    prompt: typeof c.prompt === "string" ? c.prompt : undefined,
    messages: Array.isArray(c.messages)
      ? c.messages.map((m) => {
          const mm = record(m);
          return { role: str(mm.role) ?? "user", content: str(mm.content) ?? "" };
        })
      : undefined,
    expect: c.expect && typeof c.expect === "object" ? (c.expect as Record<string, unknown>) : undefined,
    prove: "prove" in c ? c.prove : undefined,
  };
}

function variantDef(value: unknown): VariantDef {
  const v = record(value);
  return {
    name: str(v.name) ?? "",
    kind: str(v.kind) ?? "base",
    dials: v.dials && typeof v.dials === "object" ? (v.dials as Record<string, number>) : undefined,
    model: typeof v.model === "string" ? v.model : undefined,
    baseUrl: typeof v.base_url === "string" ? v.base_url : undefined,
    systemPrompt: typeof v.system_prompt === "string" ? v.system_prompt : undefined,
    promptPrefix: typeof v.prompt_prefix === "string" ? v.prompt_prefix : undefined,
    promptSuffix: typeof v.prompt_suffix === "string" ? v.prompt_suffix : undefined,
  };
}

function manifest(value: unknown): Manifest {
  const m = record(value);
  const suites = record(m.suites);
  const suiteCases = (name: string) => ({ cases: list(record(suites[name]).cases, caseDef) });
  const primary = record(m.primary_metric);
  return {
    name: str(m.name) ?? "",
    baselineVariant: str(m.baseline_variant) ?? "",
    seeds: list(m.seeds, (v) => num(v) ?? 0),
    defaults: record(m.defaults),
    variants: list(m.variants, variantDef),
    suites: { target: suiteCases("target"), guard: suiteCases("guard") },
    primaryMetric: primary.suite ? { suite: str(primary.suite) ?? "", metric: str(primary.metric) ?? "" } : undefined,
  };
}

function identity(value: unknown): RunIdentity | undefined {
  if (!value || typeof value !== "object") return undefined;
  const i = record(value);
  const out: RunIdentity = {};
  const capturedAt = str(i.captured_at);
  if (capturedAt) out.capturedAt = capturedAt;
  const modelPath = str(i.model_path);
  if (modelPath) out.modelPath = modelPath;
  const modelSha256 = str(i.model_sha256);
  if (modelSha256) out.modelSha256 = modelSha256;
  const modelSizeBytes = num(i.model_size_bytes);
  if (modelSizeBytes != null) out.modelSizeBytes = modelSizeBytes;
  const templateFingerprint = str(i.template_fingerprint);
  if (templateFingerprint) out.templateFingerprint = templateFingerprint;
  const engineBuild = str(i.engine_build);
  if (engineBuild) out.engineBuild = engineBuild;
  const cloznVersion = str(i.clozn_version);
  if (cloznVersion) out.cloznVersion = cloznVersion;
  if (i.ext && typeof i.ext === "object") out.ext = i.ext as Record<string, unknown>;
  return Object.keys(out).length ? out : undefined;
}

function cellRun(value: unknown): CellRun | null {
  if (!value || typeof value !== "object") return null;
  const r = record(value);
  return {
    id: str(r.id),
    model: typeof r.model === "string" ? r.model : undefined,
    identity: identity(r.identity),
    meta: r.meta && typeof r.meta === "object" ? (r.meta as Record<string, unknown>) : undefined,
  };
}

function fullCell(value: unknown): FullCell {
  const c = record(value);
  return {
    ...thinCell(value),
    response: str(c.response),
    receipts: c.receipts && typeof c.receipts === "object" ? (c.receipts as Record<string, unknown>) : null,
    run: cellRun(c.run),
  };
}

function experimentDetail(value: unknown): ExperimentDetail {
  const d = record(value);
  return {
    experimentId: str(d.experiment_id) ?? "",
    name: str(d.name) ?? "(unnamed)",
    createdAt: str(d.created_at),
    manifest: manifest(d.manifest),
    manifestSha256: str(d.manifest_sha256),
    seeds: list(d.seeds, (v) => num(v) ?? 0),
    summary: summary(d.summary),
    cells: list(d.cells, thinCell),
  };
}

// --- public fetchers ---------------------------------------------------------------------------

export async function listExperiments(
  options: { limit?: number; offset?: number } = {},
  signal?: AbortSignal,
): Promise<ExperimentList> {
  const params = new URLSearchParams();
  if (options.limit != null) params.set("limit", String(options.limit));
  if (options.offset != null) params.set("offset", String(options.offset));
  const query = params.toString();
  const body = await get(`/experiment-results${query ? `?${query}` : ""}`, signal);
  return {
    experiments: list(body.experiments, listEntry),
    total: num(body.total) ?? 0,
    limit: num(body.limit) ?? 50,
    offset: num(body.offset) ?? 0,
    broken: list(body.broken, (item) => {
      const b = record(item);
      return { path: str(b.path) ?? "", error: str(b.error) ?? "" } satisfies BrokenResult;
    }),
  };
}

export async function loadExperimentDetail(id: string, signal?: AbortSignal): Promise<ExperimentDetail> {
  return experimentDetail(await get(`/experiment-results/${encodeURIComponent(id)}`, signal));
}

export interface CellFilter {
  suite?: string;
  case?: string;
  variant?: string;
  seed?: number;
}

export async function loadExperimentCells(
  id: string,
  filter: CellFilter,
  signal?: AbortSignal,
): Promise<FullCell[]> {
  const params = new URLSearchParams();
  if (filter.suite) params.set("suite", filter.suite);
  if (filter.case) params.set("case", filter.case);
  if (filter.variant) params.set("variant", filter.variant);
  if (filter.seed != null) params.set("seed", String(filter.seed));
  const query = params.toString();
  const body = await get(`/experiment-results/${encodeURIComponent(id)}/cells${query ? `?${query}` : ""}`, signal);
  return list(body.cells, fullCell);
}

/** The CLI equivalent of opening exactly this cell locally -- zero new backend work, per the roadmap
 * plan: `clozn experiment show` already accepts these four filters. `suite.default_result_path` names
 * results `<experiment_id>.json` under `~/.clozn/experiments/`; a result saved with a custom `--out`
 * would not match this guess, so the command is offered as the common-case reproduction path, not a
 * guaranteed-correct one -- the drawer labels it accordingly. */
export function reproductionCommand(
  experimentId: string,
  cell: { suite: string; case: string; variant: string; seed: number },
): string {
  return `clozn experiment show ~/.clozn/experiments/${experimentId}.json ` +
    `--suite ${cell.suite} --case ${cell.case} --variant ${cell.variant} --seed ${cell.seed}`;
}
