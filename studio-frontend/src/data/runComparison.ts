import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export type ComparisonAxisStatus = "changed" | "unchanged" | "unavailable";
export type ComparisonDifferenceKind = "added" | "removed" | "changed" | "unavailable" | "diff_failed";
export type ComparisonFindingStatus = "observed" | "eliminated" | "reproduced" | "correlated" | "causally_supported";

export interface ComparisonAxis {
  id: "model" | "adapter" | "template" | "context" | "sampling" | "engine" | "tool_parse" | "output";
  status: ComparisonAxisStatus;
  note?: string;
}

export interface ComparisonDifference {
  dimension: string;
  kind: ComparisonDifferenceKind;
  rank: number;
  evidence: readonly unknown[];
  valueA?: unknown;
  valueB?: unknown;
  note?: string;
}

export interface ComparisonFinding {
  classification: string;
  status: ComparisonFindingStatus;
  summary: string;
  dimensions: readonly string[];
}

export interface RunComparison {
  schemaVersion: "clozn.run-diff.v1";
  runA: string;
  runB: string;
  generatedAt: string;
  privacyLimited: boolean;
  axes: readonly ComparisonAxis[];
  differences: readonly ComparisonDifference[];
  findings: readonly ComparisonFinding[];
}

const AXES = ["model", "adapter", "template", "context", "sampling", "engine", "tool_parse", "output"] as const;
const DIFFERENCE_KINDS = ["added", "removed", "changed", "unavailable", "diff_failed"] as const;
const FINDING_STATUSES = ["observed", "eliminated", "reproduced", "correlated", "causally_supported"] as const;

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

function optionalText(row: JsonRecord, key: string, endpoint: string, path: string): string | undefined {
  return key in row ? text(row[key], endpoint, `${path}.${key}`) : undefined;
}

function booleanValue(value: unknown, endpoint: string, path: string): boolean {
  if (typeof value !== "boolean") fail(endpoint, path, "must be a boolean");
  return value;
}

function nonNegativeInteger(value: unknown, endpoint: string, path: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) fail(endpoint, path, "must be a non-negative integer");
  return value;
}

function oneOf<const T extends readonly string[]>(value: unknown, values: T, endpoint: string, path: string): T[number] {
  if (typeof value !== "string" || !values.includes(value)) fail(endpoint, path, `must be one of ${values.join(", ")}`);
  return value as T[number];
}

function stringArray(value: unknown, endpoint: string, path: string): string[] {
  return array(value, endpoint, path).map((item, index) => text(item, endpoint, `${path}[${index}]`));
}

function decodeAxis(value: unknown, id: ComparisonAxis["id"], endpoint: string, path: string): ComparisonAxis {
  const row = object(value, endpoint, path);
  closed(row, ["status", "note"], endpoint, path);
  return {
    id,
    status: oneOf(required(row, "status", endpoint, path), ["changed", "unchanged", "unavailable"] as const, endpoint, `${path}.status`),
    note: optionalText(row, "note", endpoint, path),
  };
}

function decodeDifference(value: unknown, endpoint: string, path: string): ComparisonDifference {
  const row = object(value, endpoint, path);
  closed(row, ["dimension", "kind", "rank", "evidence", "value_a", "value_b", "note"], endpoint, path);
  const kind = oneOf(required(row, "kind", endpoint, path), DIFFERENCE_KINDS, endpoint, `${path}.kind`);
  const hasA = Object.prototype.hasOwnProperty.call(row, "value_a");
  const hasB = Object.prototype.hasOwnProperty.call(row, "value_b");
  return {
    dimension: text(required(row, "dimension", endpoint, path), endpoint, `${path}.dimension`),
    kind,
    rank: nonNegativeInteger(required(row, "rank", endpoint, path), endpoint, `${path}.rank`),
    evidence: array(required(row, "evidence", endpoint, path), endpoint, `${path}.evidence`),
    ...(hasA ? { valueA: row.value_a } : {}),
    ...(hasB ? { valueB: row.value_b } : {}),
    ...(row.note !== undefined ? { note: text(row.note, endpoint, `${path}.note`) } : {}),
  };
}

function decodeFinding(value: unknown, endpoint: string, path: string): ComparisonFinding {
  const row = object(value, endpoint, path);
  closed(row, ["classification", "status", "summary", "dimensions"], endpoint, path);
  return {
    classification: text(required(row, "classification", endpoint, path), endpoint, `${path}.classification`),
    status: oneOf(required(row, "status", endpoint, path), FINDING_STATUSES, endpoint, `${path}.status`),
    summary: text(required(row, "summary", endpoint, path), endpoint, `${path}.summary`),
    dimensions: stringArray(required(row, "dimensions", endpoint, path), endpoint, `${path}.dimensions`),
  };
}

function validateSelection(value: unknown, endpoint: string, runA: string, runB: string): void {
  const path = "$.comparison_selection";
  const row = object(value, endpoint, path);
  closed(row, ["mode", "reference_run_id", "candidate_run_id", "reason", "basis"], endpoint, path);
  oneOf(required(row, "mode", endpoint, path), ["explicit", "previous_compatible", "same_session", "same_client", "same_task", "pinned"] as const, endpoint, `${path}.mode`);
  if (text(required(row, "reference_run_id", endpoint, path), endpoint, `${path}.reference_run_id`) !== runA) fail(endpoint, `${path}.reference_run_id`, "does not match run_a");
  if (text(required(row, "candidate_run_id", endpoint, path), endpoint, `${path}.candidate_run_id`) !== runB) fail(endpoint, `${path}.candidate_run_id`, "does not match run_b");
  text(required(row, "reason", endpoint, path), endpoint, `${path}.reason`);
  if ("basis" in row) object(row.basis, endpoint, `${path}.basis`);
}

function validateRanking(value: unknown, endpoint: string): void {
  const path = "$.ranking";
  const row = object(value, endpoint, path);
  closed(row, ["order", "note"], endpoint, path);
  stringArray(required(row, "order", endpoint, path), endpoint, `${path}.order`);
  text(required(row, "note", endpoint, path), endpoint, `${path}.note`);
}

/** Decode the exact successful GET /runs/compare wire document. Unknown evidence vocabulary is rejected. */
export function decodeRunComparison(
  value: unknown,
  endpoint = "/runs/compare",
  expectedRunA?: string,
  expectedRunB?: string,
): RunComparison {
  const row = object(value, endpoint, "$");
  closed(row, [
    "schema_version", "ok", "run_a", "run_b", "comparison_selection", "generated_at",
    "privacy_limited", "ranking", "summary_axes", "differences", "findings",
  ], endpoint, "$");
  if (required(row, "ok", endpoint, "$") !== true) fail(endpoint, "$.ok", "must be true for a successful comparison");
  const schemaVersion = oneOf(required(row, "schema_version", endpoint, "$"), ["clozn.run-diff.v1"] as const, endpoint, "$.schema_version");
  const runA = text(required(row, "run_a", endpoint, "$"), endpoint, "$.run_a");
  const runB = text(required(row, "run_b", endpoint, "$"), endpoint, "$.run_b");
  if (expectedRunA !== undefined && runA !== expectedRunA) fail(endpoint, "$.run_a", "does not match the request");
  if (expectedRunB !== undefined && runB !== expectedRunB) fail(endpoint, "$.run_b", "does not match the request");
  validateSelection(required(row, "comparison_selection", endpoint, "$"), endpoint, runA, runB);
  validateRanking(required(row, "ranking", endpoint, "$"), endpoint);
  const axes = object(required(row, "summary_axes", endpoint, "$"), endpoint, "$.summary_axes");
  closed(axes, AXES, endpoint, "$.summary_axes");
  return {
    schemaVersion,
    runA,
    runB,
    generatedAt: text(required(row, "generated_at", endpoint, "$"), endpoint, "$.generated_at"),
    privacyLimited: booleanValue(required(row, "privacy_limited", endpoint, "$"), endpoint, "$.privacy_limited"),
    axes: AXES.map((id) => decodeAxis(required(axes, id, endpoint, "$.summary_axes"), id, endpoint, `$.summary_axes.${id}`)),
    differences: array(required(row, "differences", endpoint, "$"), endpoint, "$.differences")
      .map((item, index) => decodeDifference(item, endpoint, `$.differences[${index}]`)),
    findings: array(required(row, "findings", endpoint, "$"), endpoint, "$.findings")
      .map((item, index) => decodeFinding(item, endpoint, `$.findings[${index}]`)),
  };
}
