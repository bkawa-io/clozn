import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type ExperimentCellStatus = "pass" | "fail" | "error" | "unscored";
export type ExperimentSuite = "target" | "guard";

export interface ExperimentBrokenResult {
  path: string;
  error: string;
}

export interface ExperimentAggregate {
  runs: number;
  counts: Record<ExperimentCellStatus, number>;
  passRate: number | null;
}

export interface ExperimentListEntry {
  experimentId: string;
  name: string;
  createdAt: string;
  baselineVariant: string;
  variants: readonly string[];
  seeds: readonly number[];
  cellCount: number;
  aggregates: Readonly<Record<string, Readonly<Record<ExperimentSuite, ExperimentAggregate>>>>;
  suiteFingerprint: { algorithm: "canonical-json-v1"; sha256: string };
}

export interface ExperimentListDocument {
  experiments: readonly ExperimentListEntry[];
  total: number;
  limit: number;
  offset: number;
  broken: readonly ExperimentBrokenResult[];
}

export interface ExperimentVariant {
  name: string;
  kind: "base" | "tuned" | "quant" | "prompt";
}

export interface ExperimentCase {
  name: string;
}

export interface ExperimentThinCell {
  suite: ExperimentSuite;
  caseName: string;
  variant: string;
  variantKind: ExperimentVariant["kind"];
  seed: number;
  status: ExperimentCellStatus;
  runId: string | null;
  assertions: readonly JsonRecord[];
  minConfidence: number | null;
  error: string | null;
}

export interface ExperimentDetail {
  schemaVersion: "clozn.experiment.result.v0";
  experimentId: string;
  name: string;
  createdAt: string;
  manifestSha256: string;
  suiteFingerprint?: { algorithm: "canonical-json-v1"; sha256: string };
  seeds: readonly number[];
  baselineVariant: string;
  variants: readonly ExperimentVariant[];
  suites: Readonly<Record<ExperimentSuite, readonly ExperimentCase[]>>;
  cells: readonly ExperimentThinCell[];
  aggregates: Readonly<Record<string, Readonly<Record<ExperimentSuite, ExperimentAggregate>>>>;
}

export interface ExperimentFullCell extends ExperimentThinCell {
  response: string | null;
  receipts: JsonRecord | null;
  run: JsonRecord | null;
}

export interface ExperimentCellsDocument {
  experimentId: string;
  cells: readonly ExperimentFullCell[];
}

const SUITES = ["target", "guard"] as const;
const STATUSES = ["pass", "fail", "error", "unscored"] as const;
const VARIANT_KINDS = ["base", "tuned", "quant", "prompt"] as const;
const SHA256 = /^[0-9a-f]{64}$/;

function fail(endpoint: string, path: string, message: string): never {
  throw new ContractError(endpoint, `${path} ${message}`);
}

function object(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(endpoint, path, "must be an object");
  return value as JsonRecord;
}

function array(value: unknown, endpoint: string, path: string): unknown[] {
  if (!Array.isArray(value)) fail(endpoint, path, "must be an array");
  return value;
}

function required(row: JsonRecord, key: string, endpoint: string, path: string): unknown {
  if (!(key in row)) fail(endpoint, path, `is missing required field ${key}`);
  return row[key];
}

function closed(row: JsonRecord, keys: readonly string[], endpoint: string, path: string): void {
  const allowed = new Set(keys);
  for (const key of Object.keys(row)) if (!allowed.has(key)) fail(endpoint, `${path}.${key}`, "is not part of the contract");
}

function text(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string" || !value.trim()) fail(endpoint, path, "must be a non-empty string");
  return value;
}

function nullableText(value: unknown, endpoint: string, path: string): string | null {
  if (value === null) return null;
  if (typeof value !== "string") fail(endpoint, path, "must be a string or null");
  return value;
}

function integer(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) fail(endpoint, path, "must be an integer");
  return value;
}

function nonNegativeInteger(value: unknown, endpoint: string, path: string): number {
  const parsed = integer(value, endpoint, path);
  if (parsed < 0) fail(endpoint, path, "must be non-negative");
  return parsed;
}

function nullableFinite(value: unknown, endpoint: string, path: string): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) fail(endpoint, path, "must be a finite number or null");
  return value;
}

function oneOf<const T extends readonly string[]>(value: unknown, values: T, endpoint: string, path: string): T[number] {
  if (typeof value !== "string" || !values.includes(value)) fail(endpoint, path, `must be one of ${values.join(", ")}`);
  return value as T[number];
}

function distinctIntegers(value: unknown, endpoint: string, path: string): number[] {
  const items = array(value, endpoint, path).map((item, index) => integer(item, endpoint, `${path}[${index}]`));
  if (!items.length) fail(endpoint, path, "must not be empty");
  if (new Set(items).size !== items.length) fail(endpoint, path, "must not contain duplicates");
  return items;
}

function distinctTexts(value: unknown, endpoint: string, path: string): string[] {
  const items = array(value, endpoint, path).map((item, index) => text(item, endpoint, `${path}[${index}]`));
  if (new Set(items).size !== items.length) fail(endpoint, path, "must not contain duplicates");
  return items;
}

function decodeFingerprint(value: unknown, endpoint: string, path: string) {
  const row = object(value, endpoint, path);
  closed(row, ["algorithm", "sha256"], endpoint, path);
  const algorithm = oneOf(required(row, "algorithm", endpoint, path), ["canonical-json-v1"] as const, endpoint, `${path}.algorithm`);
  const sha256 = text(required(row, "sha256", endpoint, path), endpoint, `${path}.sha256`);
  if (!SHA256.test(sha256)) fail(endpoint, `${path}.sha256`, "must be a lowercase SHA-256 digest");
  return { algorithm, sha256 };
}

function decodeCounts(value: unknown, endpoint: string, path: string): Record<ExperimentCellStatus, number> {
  const row = object(value, endpoint, path);
  closed(row, STATUSES, endpoint, path);
  return {
    pass: nonNegativeInteger(required(row, "pass", endpoint, path), endpoint, `${path}.pass`),
    fail: nonNegativeInteger(required(row, "fail", endpoint, path), endpoint, `${path}.fail`),
    error: nonNegativeInteger(required(row, "error", endpoint, path), endpoint, `${path}.error`),
    unscored: nonNegativeInteger(required(row, "unscored", endpoint, path), endpoint, `${path}.unscored`),
  };
}

function decodeAggregate(value: unknown, endpoint: string, path: string): ExperimentAggregate {
  const row = object(value, endpoint, path);
  closed(row, ["runs", "counts", "pass_rate"], endpoint, path);
  const counts = decodeCounts(required(row, "counts", endpoint, path), endpoint, `${path}.counts`);
  const runs = nonNegativeInteger(required(row, "runs", endpoint, path), endpoint, `${path}.runs`);
  if (Object.values(counts).reduce((sum, count) => sum + count, 0) !== runs) fail(endpoint, path, "counts do not sum to runs");
  const passRate = nullableFinite(required(row, "pass_rate", endpoint, path), endpoint, `${path}.pass_rate`);
  if (passRate !== null && (passRate < 0 || passRate > 1)) fail(endpoint, `${path}.pass_rate`, "must be between 0 and 1");
  return { runs, counts, passRate };
}

function decodeAggregates(value: unknown, variants: readonly string[], endpoint: string, path: string) {
  const row = object(value, endpoint, path);
  closed(row, variants, endpoint, path);
  const out: Record<string, Record<ExperimentSuite, ExperimentAggregate>> = {};
  for (const variant of variants) {
    const suites = object(required(row, variant, endpoint, path), endpoint, `${path}.${variant}`);
    closed(suites, SUITES, endpoint, `${path}.${variant}`);
    out[variant] = {
      target: decodeAggregate(required(suites, "target", endpoint, `${path}.${variant}`), endpoint, `${path}.${variant}.target`),
      guard: decodeAggregate(required(suites, "guard", endpoint, `${path}.${variant}`), endpoint, `${path}.${variant}.guard`),
    };
  }
  return out;
}

function validateOptionalMetadata(row: JsonRecord, endpoint: string, path: string): void {
  if ("vcs" in row) {
    const vcs = object(row.vcs, endpoint, `${path}.vcs`);
    closed(vcs, ["repository", "commit", "branch"], endpoint, `${path}.vcs`);
    for (const [key, value] of Object.entries(vcs)) text(value, endpoint, `${path}.vcs.${key}`);
  }
  if ("artifact_provenance" in row) {
    const provenance = object(row.artifact_provenance, endpoint, `${path}.artifact_provenance`);
    closed(provenance, ["workflow_url", "artifact_url", "local_open_command", "expires_at"], endpoint, `${path}.artifact_provenance`);
    for (const [key, value] of Object.entries(provenance)) text(value, endpoint, `${path}.artifact_provenance.${key}`);
  }
}

function decodeListEntry(value: unknown, endpoint: string, path: string): ExperimentListEntry {
  const row = object(value, endpoint, path);
  closed(row, [
    "experiment_id", "name", "created_at", "baseline_variant", "variants", "seeds", "cell_count",
    "aggregates", "suite_fingerprint", "vcs", "artifact_provenance",
  ], endpoint, path);
  const variants = distinctTexts(required(row, "variants", endpoint, path), endpoint, `${path}.variants`);
  if (variants.length < 2) fail(endpoint, `${path}.variants`, "must contain at least two variants");
  const baselineVariant = text(required(row, "baseline_variant", endpoint, path), endpoint, `${path}.baseline_variant`);
  if (!variants.includes(baselineVariant)) fail(endpoint, `${path}.baseline_variant`, "must name a listed variant");
  validateOptionalMetadata(row, endpoint, path);
  return {
    experimentId: text(required(row, "experiment_id", endpoint, path), endpoint, `${path}.experiment_id`),
    name: text(required(row, "name", endpoint, path), endpoint, `${path}.name`),
    createdAt: text(required(row, "created_at", endpoint, path), endpoint, `${path}.created_at`),
    baselineVariant,
    variants,
    seeds: distinctIntegers(required(row, "seeds", endpoint, path), endpoint, `${path}.seeds`),
    cellCount: nonNegativeInteger(required(row, "cell_count", endpoint, path), endpoint, `${path}.cell_count`),
    aggregates: decodeAggregates(required(row, "aggregates", endpoint, path), variants, endpoint, `${path}.aggregates`),
    suiteFingerprint: decodeFingerprint(required(row, "suite_fingerprint", endpoint, path), endpoint, `${path}.suite_fingerprint`),
  };
}

export function decodeExperimentList(value: unknown, endpoint = "/experiment-results"): ExperimentListDocument {
  const row = object(value, endpoint, "$");
  closed(row, ["experiments", "total", "limit", "offset", "broken"], endpoint, "$");
  const experiments = array(required(row, "experiments", endpoint, "$"), endpoint, "$.experiments")
    .map((item, index) => decodeListEntry(item, endpoint, `$.experiments[${index}]`));
  const broken = array(required(row, "broken", endpoint, "$"), endpoint, "$.broken").map((item, index) => {
    const path = `$.broken[${index}]`;
    const entry = object(item, endpoint, path);
    closed(entry, ["path", "error"], endpoint, path);
    return {
      path: text(required(entry, "path", endpoint, path), endpoint, `${path}.path`),
      error: text(required(entry, "error", endpoint, path), endpoint, `${path}.error`),
    };
  });
  const total = nonNegativeInteger(required(row, "total", endpoint, "$"), endpoint, "$.total");
  const limit = nonNegativeInteger(required(row, "limit", endpoint, "$"), endpoint, "$.limit");
  const offset = nonNegativeInteger(required(row, "offset", endpoint, "$"), endpoint, "$.offset");
  if (limit < 1) fail(endpoint, "$.limit", "must be positive");
  if (experiments.length > limit || offset + experiments.length > total) fail(endpoint, "$.experiments", "is inconsistent with pagination metadata");
  const ids = experiments.map((entry) => entry.experimentId);
  if (new Set(ids).size !== ids.length) fail(endpoint, "$.experiments", "must not contain duplicate experiment ids");
  return { experiments, total, limit, offset, broken };
}

function decodeVariant(value: unknown, endpoint: string, path: string): ExperimentVariant {
  const row = object(value, endpoint, path);
  closed(row, [
    "name", "kind", "base_url", "model", "system_prompt", "prompt_prefix", "prompt_suffix",
    "max_tokens", "temperature", "top_p", "top_k", "repeat_penalty",
  ], endpoint, path);
  for (const key of ["base_url", "model", "system_prompt", "prompt_prefix", "prompt_suffix"]) {
    if (key in row && typeof row[key] !== "string") fail(endpoint, `${path}.${key}`, "must be a string");
  }
  for (const key of ["max_tokens", "temperature", "top_p", "top_k", "repeat_penalty"]) {
    if (key in row && (typeof row[key] !== "number" || !Number.isFinite(row[key]))) fail(endpoint, `${path}.${key}`, "must be a finite number");
  }
  return {
    name: text(required(row, "name", endpoint, path), endpoint, `${path}.name`),
    kind: oneOf(required(row, "kind", endpoint, path), VARIANT_KINDS, endpoint, `${path}.kind`),
  };
}

function decodeCase(value: unknown, endpoint: string, path: string): ExperimentCase {
  const row = object(value, endpoint, path);
  closed(row, ["name", "prompt", "messages", "expect", "prove"], endpoint, path);
  const hasPrompt = "prompt" in row;
  const hasMessages = "messages" in row;
  if (hasPrompt === hasMessages) fail(endpoint, path, "must define exactly one of prompt or messages");
  if (hasPrompt) text(row.prompt, endpoint, `${path}.prompt`);
  if (hasMessages) {
    const messages = array(row.messages, endpoint, `${path}.messages`);
    if (!messages.length) fail(endpoint, `${path}.messages`, "must not be empty");
    messages.forEach((item, index) => {
      const messagePath = `${path}.messages[${index}]`;
      const message = object(item, endpoint, messagePath);
      closed(message, ["role", "content"], endpoint, messagePath);
      oneOf(required(message, "role", endpoint, messagePath), ["system", "developer", "user", "assistant"] as const, endpoint, `${messagePath}.role`);
      if (typeof required(message, "content", endpoint, messagePath) !== "string") fail(endpoint, `${messagePath}.content`, "must be a string");
    });
  }
  if ("expect" in row) object(row.expect, endpoint, `${path}.expect`);
  if ("prove" in row) object(row.prove, endpoint, `${path}.prove`);
  return { name: text(required(row, "name", endpoint, path), endpoint, `${path}.name`) };
}

function decodeAssertions(value: unknown, endpoint: string, path: string): JsonRecord[] {
  return array(value, endpoint, path).map((item, index) => object(item, endpoint, `${path}[${index}]`));
}

const THIN_CELL_KEYS = [
  "suite", "case", "variant", "variant_kind", "seed", "status", "run_id", "assertions", "min_confidence", "error",
] as const;

function decodeThinCell(value: unknown, endpoint: string, path: string, full = false): ExperimentThinCell {
  const row = object(value, endpoint, path);
  closed(row, full ? [...THIN_CELL_KEYS, "response", "receipts", "run"] : THIN_CELL_KEYS, endpoint, path);
  const status = oneOf(required(row, "status", endpoint, path), STATUSES, endpoint, `${path}.status`);
  const runId = nullableText(required(row, "run_id", endpoint, path), endpoint, `${path}.run_id`);
  if (status !== "error" && !runId?.trim()) fail(endpoint, `${path}.run_id`, "must identify run evidence for a non-error cell");
  return {
    suite: oneOf(required(row, "suite", endpoint, path), SUITES, endpoint, `${path}.suite`),
    caseName: text(required(row, "case", endpoint, path), endpoint, `${path}.case`),
    variant: text(required(row, "variant", endpoint, path), endpoint, `${path}.variant`),
    variantKind: oneOf(required(row, "variant_kind", endpoint, path), VARIANT_KINDS, endpoint, `${path}.variant_kind`),
    seed: integer(required(row, "seed", endpoint, path), endpoint, `${path}.seed`),
    status,
    runId,
    assertions: decodeAssertions(required(row, "assertions", endpoint, path), endpoint, `${path}.assertions`),
    minConfidence: nullableFinite(required(row, "min_confidence", endpoint, path), endpoint, `${path}.min_confidence`),
    error: nullableText(required(row, "error", endpoint, path), endpoint, `${path}.error`),
  };
}

function validateSummary(value: unknown, variants: readonly string[], baseline: string, endpoint: string): ReturnType<typeof decodeAggregates> {
  const path = "$.summary";
  const row = object(value, endpoint, path);
  closed(row, ["baseline_variant", "aggregates", "comparisons"], endpoint, path);
  if (text(required(row, "baseline_variant", endpoint, path), endpoint, `${path}.baseline_variant`) !== baseline) {
    fail(endpoint, `${path}.baseline_variant`, "does not match manifest.baseline_variant");
  }
  const aggregates = decodeAggregates(required(row, "aggregates", endpoint, path), variants, endpoint, `${path}.aggregates`);
  const comparisons = array(required(row, "comparisons", endpoint, path), endpoint, `${path}.comparisons`);
  if (comparisons.length !== variants.length - 1) fail(endpoint, `${path}.comparisons`, "must contain one entry per non-baseline variant");
  const seen = new Set<string>();
  comparisons.forEach((item, index) => {
    const comparisonPath = `${path}.comparisons[${index}]`;
    const comparison = object(item, endpoint, comparisonPath);
    closed(comparison, [
      "variant", "baseline", "target_gains", "target_regressions", "guard_regressions", "guard_fixes", "changed_unscored",
    ], endpoint, comparisonPath);
    const variant = text(required(comparison, "variant", endpoint, comparisonPath), endpoint, `${comparisonPath}.variant`);
    if (variant === baseline || !variants.includes(variant) || seen.has(variant)) fail(endpoint, `${comparisonPath}.variant`, "must uniquely name a non-baseline variant");
    seen.add(variant);
    if (text(required(comparison, "baseline", endpoint, comparisonPath), endpoint, `${comparisonPath}.baseline`) !== baseline) fail(endpoint, `${comparisonPath}.baseline`, "does not match the manifest baseline");
    for (const key of ["target_gains", "target_regressions", "guard_regressions", "guard_fixes", "changed_unscored"]) {
      array(required(comparison, key, endpoint, comparisonPath), endpoint, `${comparisonPath}.${key}`).forEach((label, labelIndex) => {
        const labelPath = `${comparisonPath}.${key}[${labelIndex}]`;
        const labelRow = object(label, endpoint, labelPath);
        closed(labelRow, key === "changed_unscored" ? ["suite", "case", "seed"] : ["case", "seed"], endpoint, labelPath);
        if (key === "changed_unscored") oneOf(required(labelRow, "suite", endpoint, labelPath), SUITES, endpoint, `${labelPath}.suite`);
        text(required(labelRow, "case", endpoint, labelPath), endpoint, `${labelPath}.case`);
        integer(required(labelRow, "seed", endpoint, labelPath), endpoint, `${labelPath}.seed`);
      });
    }
  });
  return aggregates;
}

export function decodeExperimentDetail(value: unknown, endpoint = "/experiment-results/<id>", expectedId?: string): ExperimentDetail {
  const row = object(value, endpoint, "$");
  closed(row, [
    "schema_version", "experiment_id", "name", "created_at", "manifest_sha256", "suite_fingerprint",
    "vcs", "artifact_provenance", "manifest", "seeds", "cells", "summary",
  ], endpoint, "$");
  const schemaVersion = oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.experiment.result.v0"] as const, endpoint, "$.schema_version");
  const experimentId = text(required(row, "experiment_id", endpoint, "$"), endpoint, "$.experiment_id");
  if (expectedId !== undefined && experimentId !== expectedId) fail(endpoint, "$.experiment_id", "does not match the request");
  const manifest = object(required(row, "manifest", endpoint, "$"), endpoint, "$.manifest");
  closed(manifest, ["schema_version", "name", "defaults", "seeds", "variants", "baseline_variant", "primary_metric", "suites"], endpoint, "$.manifest");
  oneOf(required(manifest, "schema_version", endpoint, "$.manifest"), ["clozn.experiment.v0"] as const, endpoint, "$.manifest.schema_version");
  text(required(manifest, "name", endpoint, "$.manifest"), endpoint, "$.manifest.name");
  object(required(manifest, "defaults", endpoint, "$.manifest"), endpoint, "$.manifest.defaults");
  distinctIntegers(required(manifest, "seeds", endpoint, "$.manifest"), endpoint, "$.manifest.seeds");
  const variants = array(required(manifest, "variants", endpoint, "$.manifest"), endpoint, "$.manifest.variants")
    .map((item, index) => decodeVariant(item, endpoint, `$.manifest.variants[${index}]`));
  if (variants.length < 2 || new Set(variants.map((variant) => variant.name)).size !== variants.length) fail(endpoint, "$.manifest.variants", "must contain at least two unique variants");
  const variantNames = variants.map((variant) => variant.name);
  const baselineVariant = text(required(manifest, "baseline_variant", endpoint, "$.manifest"), endpoint, "$.manifest.baseline_variant");
  if (!variantNames.includes(baselineVariant)) fail(endpoint, "$.manifest.baseline_variant", "must name a manifest variant");
  if ("primary_metric" in manifest && manifest.primary_metric !== null) {
    const metric = object(manifest.primary_metric, endpoint, "$.manifest.primary_metric");
    closed(metric, ["suite", "metric"], endpoint, "$.manifest.primary_metric");
    oneOf(required(metric, "suite", endpoint, "$.manifest.primary_metric"), SUITES, endpoint, "$.manifest.primary_metric.suite");
    oneOf(required(metric, "metric", endpoint, "$.manifest.primary_metric"), ["pass_rate"] as const, endpoint, "$.manifest.primary_metric.metric");
  }
  const suitesRow = object(required(manifest, "suites", endpoint, "$.manifest"), endpoint, "$.manifest.suites");
  closed(suitesRow, SUITES, endpoint, "$.manifest.suites");
  const suites = Object.fromEntries(SUITES.map((suite) => {
    const suitePath = `$.manifest.suites.${suite}`;
    const suiteRow = object(required(suitesRow, suite, endpoint, "$.manifest.suites"), endpoint, suitePath);
    closed(suiteRow, ["cases"], endpoint, suitePath);
    const cases = array(required(suiteRow, "cases", endpoint, suitePath), endpoint, `${suitePath}.cases`)
      .map((item, index) => decodeCase(item, endpoint, `${suitePath}.cases[${index}]`));
    if (!cases.length || new Set(cases.map((entry) => entry.name)).size !== cases.length) fail(endpoint, `${suitePath}.cases`, "must contain unique cases");
    return [suite, cases];
  })) as unknown as Record<ExperimentSuite, ExperimentCase[]>;
  const seeds = distinctIntegers(required(row, "seeds", endpoint, "$"), endpoint, "$.seeds");
  const cells = array(required(row, "cells", endpoint, "$"), endpoint, "$.cells")
    .map((item, index) => decodeThinCell(item, endpoint, `$.cells[${index}]`));
  const expectedCoordinates = new Set<string>();
  const variantKind = new Map(variants.map((variant) => [variant.name, variant.kind]));
  for (const suite of SUITES) for (const caseEntry of suites[suite]) for (const variant of variants) for (const seed of seeds) {
    expectedCoordinates.add(`${suite}\u0000${caseEntry.name}\u0000${variant.name}\u0000${seed}`);
  }
  const seenCoordinates = new Set<string>();
  for (const cell of cells) {
    const coordinate = `${cell.suite}\u0000${cell.caseName}\u0000${cell.variant}\u0000${cell.seed}`;
    if (!expectedCoordinates.has(coordinate)) fail(endpoint, "$.cells", "contains a coordinate outside the manifest matrix");
    if (seenCoordinates.has(coordinate)) fail(endpoint, "$.cells", "contains a duplicate coordinate");
    if (variantKind.get(cell.variant) !== cell.variantKind) fail(endpoint, "$.cells", "contains a variant kind that disagrees with the manifest");
    seenCoordinates.add(coordinate);
  }
  if (seenCoordinates.size !== expectedCoordinates.size) fail(endpoint, "$.cells", "is incomplete for the manifest matrix");
  const manifestSha256 = text(required(row, "manifest_sha256", endpoint, "$"), endpoint, "$.manifest_sha256");
  if (!SHA256.test(manifestSha256)) fail(endpoint, "$.manifest_sha256", "must be a lowercase SHA-256 digest");
  validateOptionalMetadata(row, endpoint, "$");
  const aggregates = validateSummary(required(row, "summary", endpoint, "$"), variantNames, baselineVariant, endpoint);
  return {
    schemaVersion,
    experimentId,
    name: text(required(row, "name", endpoint, "$"), endpoint, "$.name"),
    createdAt: text(required(row, "created_at", endpoint, "$"), endpoint, "$.created_at"),
    manifestSha256,
    ...(row.suite_fingerprint !== undefined ? { suiteFingerprint: decodeFingerprint(row.suite_fingerprint, endpoint, "$.suite_fingerprint") } : {}),
    seeds,
    baselineVariant,
    variants,
    suites,
    cells,
    aggregates,
  };
}

function decodeFullCell(value: unknown, endpoint: string, path: string): ExperimentFullCell {
  const row = object(value, endpoint, path);
  const thin = decodeThinCell(value, endpoint, path, true);
  const receiptsValue = required(row, "receipts", endpoint, path);
  const runValue = required(row, "run", endpoint, path);
  const receipts = receiptsValue === null ? null : object(receiptsValue, endpoint, `${path}.receipts`);
  const run = runValue === null ? null : object(runValue, endpoint, `${path}.run`);
  if (thin.status !== "error" && run === null) fail(endpoint, `${path}.run`, "must contain run evidence for a non-error cell");
  if (run && "id" in run && run.id !== thin.runId) fail(endpoint, `${path}.run.id`, "does not match run_id");
  return {
    ...thin,
    response: nullableText(required(row, "response", endpoint, path), endpoint, `${path}.response`),
    receipts,
    run,
  };
}

export function decodeExperimentCells(value: unknown, endpoint = "/experiment-results/<id>/cells", expectedId?: string): ExperimentCellsDocument {
  const row = object(value, endpoint, "$");
  closed(row, ["experiment_id", "cells"], endpoint, "$");
  const experimentId = text(required(row, "experiment_id", endpoint, "$"), endpoint, "$.experiment_id");
  if (expectedId !== undefined && experimentId !== expectedId) fail(endpoint, "$.experiment_id", "does not match the request");
  const cells = array(required(row, "cells", endpoint, "$"), endpoint, "$.cells")
    .map((item, index) => decodeFullCell(item, endpoint, `$.cells[${index}]`));
  const coordinates = cells.map((cell) => `${cell.suite}\u0000${cell.caseName}\u0000${cell.variant}\u0000${cell.seed}`);
  if (new Set(coordinates).size !== coordinates.length) fail(endpoint, "$.cells", "must not contain duplicate coordinates");
  return { experimentId, cells };
}
