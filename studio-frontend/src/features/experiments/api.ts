import type {
  Aggregates,
  BrokenResult,
  CaseDef,
  CellRun,
  CiPreview,
  CompatibleTrends,
  ExperimentDetail,
  ExperimentList,
  ExperimentListEntry,
  ExperimentSummary,
  FullCell,
  Manifest,
  PromotionPreview,
  PromotionTransaction,
  RunIdentity,
  StatusCounts,
  SuiteAggregate,
  SuiteFingerprint,
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

async function post(url: string, payload: JsonRecord, signal?: AbortSignal): Promise<JsonRecord> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
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

function fingerprint(value: unknown): SuiteFingerprint | null {
  const f = record(value);
  const algorithm = str(f.algorithm);
  const sha256 = str(f.sha256);
  return algorithm && sha256 ? { algorithm, sha256 } : null;
}

function vcs(value: unknown) {
  if (!value || typeof value !== "object") return null;
  const v = record(value);
  return {
    repository: str(v.repository) ?? undefined,
    commit: str(v.commit) ?? undefined,
    branch: str(v.branch) ?? undefined,
  };
}

function provenance(value: unknown) {
  if (!value || typeof value !== "object") return null;
  const p = record(value);
  return {
    workflowUrl: str(p.workflow_url) ?? undefined,
    artifactUrl: str(p.artifact_url) ?? undefined,
    localOpenCommand: str(p.local_open_command) ?? undefined,
    expiresAt: str(p.expires_at) ?? undefined,
  };
}

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
    suiteFingerprint: fingerprint(e.suite_fingerprint),
    vcs: vcs(e.vcs),
    artifactProvenance: provenance(e.artifact_provenance),
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
    suiteFingerprint: fingerprint(d.suite_fingerprint),
    vcs: vcs(d.vcs),
    artifactProvenance: provenance(d.artifact_provenance),
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

export async function loadExperimentTrends(
  id: string,
  signal?: AbortSignal,
): Promise<CompatibleTrends> {
  const body = await get(`/experiment-results/${encodeURIComponent(id)}/trends`, signal);
  const fp = fingerprint(body.suite_fingerprint);
  if (!fp) throw new Error("Trend response omitted its suite fingerprint");
  return {
    suiteFingerprint: fp,
    points: list(body.points, (value) => {
      const point = record(value);
      const pointFingerprint = fingerprint(point.suite_fingerprint);
      if (!pointFingerprint) throw new Error("Trend point omitted its suite fingerprint");
      const instability = record(point.replicate_instability);
      return {
        experimentId: str(point.experiment_id) ?? "",
        name: str(point.name) ?? "(unnamed)",
        createdAt: str(point.created_at),
        suiteFingerprint: pointFingerprint,
        identity: Object.fromEntries(
          Object.entries(record(point.identity)).map(([key, values]) => [
            key, list(values, (item) => String(item)),
          ]),
        ),
        vcs: vcs(point.vcs),
        artifactProvenance: provenance(point.artifact_provenance),
        baselineVariant: str(point.baseline_variant),
        aggregates: aggregates(point.aggregates),
        comparisonCounts: Object.fromEntries(
          Object.entries(record(point.comparison_counts)).map(([variant, counts]) => [
            variant,
            Object.fromEntries(
              Object.entries(record(counts)).map(([name, count]) => [name, num(count) ?? 0]),
            ),
          ]),
        ),
        errorCells: num(point.error_cells) ?? 0,
        replicateInstability: {
          coordinateCount: num(instability.coordinate_count) ?? 0,
          coordinates: list(instability.coordinates, (item) => {
            const row = record(item);
            return {
              suite: str(row.suite) ?? "",
              case: str(row.case) ?? "",
              variant: str(row.variant) ?? "",
              statuses: list(row.statuses, (status) => String(status)),
            };
          }),
        },
      };
    }),
    broken: list(body.broken, (item) => {
      const broken = record(item);
      return { path: str(broken.path) ?? "", error: str(broken.error) ?? "" };
    }),
  };
}

export interface PromotionRequest {
  suite: string;
  case: string;
  variant: string;
  seed: number;
  destination: string;
  case_name?: string;
  suite_name?: string;
  replacements?: Record<string, string>;
  expected_destination_hash?: string;
  acknowledged_findings?: string[];
}

function promotionPreview(value: unknown): PromotionPreview {
  const p = record(value);
  const diff = record(p.destination_diff);
  return {
    sourceRunId: str(p.source_run_id) ?? "",
    role: (str(p.role) ?? "target") as "target" | "guard",
    candidateCase: record(p.candidate_case),
    destination: str(p.destination) ?? "",
    destinationDiff: {
      operation: str(diff.operation) ?? "",
      beforeCaseCount: num(diff.before_case_count) ?? 0,
      afterCaseCount: num(diff.after_case_count) ?? 0,
      addedCase: str(diff.added_case) ?? "",
    },
    expectedDestinationHash: str(p.expected_destination_hash) ?? "",
    proposedDestinationSha256: str(p.proposed_destination_sha256) ?? "",
    redactionFindings: list(p.redaction_findings, (item) => {
      const finding = record(item);
      return {
        id: str(finding.id) ?? "",
        kind: str(finding.kind) ?? "",
        path: str(finding.path) ?? "",
        start: num(finding.start) ?? 0,
        end: num(finding.end) ?? 0,
        preview: str(finding.preview) ?? "",
      };
    }),
    requiredAcknowledgements: list(p.required_acknowledgements, (item) => String(item)),
  };
}

export async function previewPromotion(
  id: string,
  request: PromotionRequest,
  signal?: AbortSignal,
): Promise<PromotionPreview> {
  return promotionPreview(await post(
    `/experiment-results/${encodeURIComponent(id)}/promotion-preview`,
    request as unknown as JsonRecord,
    signal,
  ));
}

export async function applyPromotion(
  id: string,
  request: PromotionRequest,
  signal?: AbortSignal,
): Promise<PromotionTransaction> {
  const body = await post(
    `/experiment-results/${encodeURIComponent(id)}/promotion-apply`,
    request as unknown as JsonRecord,
    signal,
  );
  return {
    transactionId: str(body.transaction_id) ?? "",
    transactionPath: str(body.transaction_path) ?? "",
    destination: str(body.destination) ?? "",
    destinationSha256: str(body.destination_sha256) ?? "",
    backup: str(body.backup),
    role: (str(body.role) ?? "target") as "target" | "guard",
  };
}

export async function generateCiPreview(
  id: string,
  inputs: JsonRecord,
  signal?: AbortSignal,
): Promise<CiPreview> {
  const body = await post(
    `/experiment-results/${encodeURIComponent(id)}/ci-preview`, inputs, signal);
  const fp = fingerprint(body.suite_fingerprint);
  if (!fp) throw new Error("CI preview omitted its suite fingerprint");
  return {
    suiteFingerprint: fp,
    cacheKey: str(body.cache_key) ?? "",
    workflowYaml: str(body.workflow_yaml) ?? "",
    inputs: record(body.inputs),
  };
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
