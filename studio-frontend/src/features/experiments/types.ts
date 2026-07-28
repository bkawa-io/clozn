/**
 * Frontend shape for `clozn.experiment.result.v0` artifacts, read through the read-only
 * `GET /experiment-results[...]` family (`clozn/server/routes/experiment_results.py`). Field names are
 * camelCased on the way in (`api.ts`); every optional field stays optional/undefined rather than being
 * defaulted to a plausible-looking value -- an omitted identity field, a null `pass_rate` (an unscored
 * suite), and a null `receipts` object are all real, distinct states this workspace must show honestly,
 * never paper over. See `docs/SURFACES.md` and the roadmap plan's "missing values stay missing" rule.
 */

export type CellStatus = "pass" | "fail" | "error" | "unscored";
export type SuiteName = "target" | "guard";

export interface Assertion {
  check: string;
  status: string;
}

export interface StatusCounts {
  pass: number;
  fail: number;
  error: number;
  unscored: number;
}

export interface SuiteAggregate {
  runs: number;
  counts: StatusCounts;
  /** null means every cell in this variant/suite was "unscored" -- there is no pass rate to show. */
  passRate: number | null;
}

/** variant name -> per-suite aggregate. A variant/suite pair is absent, not zeroed, if never run. */
export type Aggregates = Record<string, Partial<Record<SuiteName, SuiteAggregate>>>;

export interface ComparisonLabel {
  case: string;
  seed: number;
}

export interface VariantComparison {
  variant: string;
  baseline: string;
  targetGains: ComparisonLabel[];
  targetRegressions: ComparisonLabel[];
  guardRegressions: ComparisonLabel[];
  guardFixes: ComparisonLabel[];
  changedUnscored: (ComparisonLabel & { suite: string })[];
}

export interface ExperimentSummary {
  baselineVariant: string | null;
  aggregates: Aggregates;
  comparisons: VariantComparison[];
}

export interface ExperimentListEntry {
  experimentId: string;
  name: string;
  createdAt: string | null;
  baselineVariant: string | null;
  variants: string[];
  seeds: number[];
  cellCount: number;
  aggregates: Aggregates | null;
}

export interface BrokenResult {
  path: string;
  error: string;
}

export interface ExperimentList {
  experiments: ExperimentListEntry[];
  total: number;
  limit: number;
  offset: number;
  broken: BrokenResult[];
}

/** The list/detail-weight cell: coordinates, status, and small evaluation facts -- never response
 * text, receipts, or the embedded run record. See the server module's own "TWO SIZES OF RESPONSE"
 * note for why. */
export interface ThinCell {
  suite: string;
  case: string;
  variant: string;
  variantKind: string | null;
  seed: number;
  status: string;
  runId: string | null;
  assertions: Assertion[];
  minConfidence: number | null;
  error: string | null;
}

export interface CaseMessage {
  role: string;
  content: string;
}

export interface CaseDef {
  name: string;
  prompt?: string;
  messages?: CaseMessage[];
  expect?: Record<string, unknown>;
  prove?: unknown;
}

export interface VariantDef {
  name: string;
  kind: string;
  dials?: Record<string, number>;
  model?: string;
  baseUrl?: string;
  systemPrompt?: string;
  promptPrefix?: string;
  promptSuffix?: string;
}

export interface Manifest {
  name: string;
  baselineVariant: string;
  seeds: number[];
  defaults: Record<string, unknown>;
  variants: VariantDef[];
  suites: Partial<Record<SuiteName, { cases: CaseDef[] }>>;
  primaryMetric?: { suite: string; metric: string };
}

export interface ExperimentDetail {
  experimentId: string;
  name: string;
  createdAt: string | null;
  manifest: Manifest;
  manifestSha256: string | null;
  seeds: number[];
  summary: ExperimentSummary;
  /** Every cell in the case x variant x seed matrix, thinned -- see ThinCell. */
  cells: ThinCell[];
}

/** clozn/runs/identity.py's block, verbatim field-for-field. Every key is honestly omitted -- never
 * null-padded -- when a run does not carry it, which is why this interface has no required fields
 * beyond the object itself existing. `ext` holds namespaced facets from identity_providers/*.py. */
export interface RunIdentity {
  capturedAt?: string;
  modelPath?: string;
  modelSha256?: string;
  modelSizeBytes?: number;
  templateFingerprint?: string;
  engineBuild?: string;
  cloznVersion?: string;
  ext?: Record<string, unknown>;
}

export interface CellRun {
  id: string | null;
  model?: string;
  identity?: RunIdentity;
  meta?: Record<string, unknown>;
}

/** The "open this cell" payload -- response text, receipts, and the full embedded run record. */
export interface FullCell extends ThinCell {
  response: string | null;
  receipts: Record<string, unknown> | null;
  run: CellRun | null;
}
