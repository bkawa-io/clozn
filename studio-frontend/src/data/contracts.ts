/** Fail-closed decoders for the small Studio boundary. */

export class ContractError extends Error {
  readonly name = "ContractError";

  constructor(readonly endpoint: string, readonly detail: string) {
    super(`Invalid response from ${endpoint}: ${detail}`);
  }
}

type JsonRecord = Record<string, unknown>;

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

function text(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string") fail(endpoint, path, "must be a string");
  return value;
}

function nonEmptyText(value: unknown, endpoint: string, path: string): string {
  const parsed = text(value, endpoint, path);
  if (!parsed.trim()) fail(endpoint, path, "must not be empty");
  return parsed;
}

function nullableText(value: unknown, endpoint: string, path: string): string | null {
  return value === null ? null : text(value, endpoint, path);
}

function nullableNumber(value: unknown, endpoint: string, path: string): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value)) fail(endpoint, path, "must be a finite number or null");
  return value;
}

function optional<T>(row: JsonRecord, key: string, parser: (value: unknown, path: string) => T, endpoint: string, path: string): T | undefined {
  return key in row ? parser(row[key], `${path}.${key}`) : undefined;
}

export interface RunSummary {
  id: string;
  createdAt?: string | null;
  createdTs?: number | null;
  model?: string | null;
  promptSummary?: string | null;
  responseSummary?: string | null;
  parentRunId?: string | null;
  flags: readonly string[];
}

export function decodeRunsDocument(value: unknown, endpoint = "/runs"): readonly RunSummary[] {
  const row = object(value, endpoint, "$");
  return array(required(row, "runs", endpoint, "$"), endpoint, "$.runs").map((entry, index) => {
    const path = `$.runs[${index}]`;
    const item = object(entry, endpoint, path);
    return {
      id: nonEmptyText(required(item, "id", endpoint, path), endpoint, `${path}.id`),
      createdAt: optional(item, "created_at", (v, p) => nullableText(v, endpoint, p), endpoint, path),
      createdTs: optional(item, "created_ts", (v, p) => nullableNumber(v, endpoint, p), endpoint, path),
      model: optional(item, "model", (v, p) => nullableText(v, endpoint, p), endpoint, path),
      promptSummary: optional(item, "prompt_summary", (v, p) => nullableText(v, endpoint, p), endpoint, path),
      responseSummary: optional(item, "response_summary", (v, p) => nullableText(v, endpoint, p), endpoint, path),
      parentRunId: optional(item, "parent_run_id", (v, p) => nullableText(v, endpoint, p), endpoint, path),
      flags: optional(item, "flags", (v, p) => array(v, endpoint, p).map((flag, flagIndex) => (
        nonEmptyText(flag, endpoint, `${p}[${flagIndex}]`)
      )), endpoint, path) ?? [],
    };
  });
}

export interface RunDetails {
  id: string;
  model?: string;
  substrate?: string;
  parentRunId?: string | null;
  identity: JsonRecord;
}

export function decodeRunDetails(value: unknown, endpoint: string, expectedRunId: string): RunDetails {
  const row = object(value, endpoint, "$");
  const id = nonEmptyText(required(row, "id", endpoint, "$"), endpoint, "$.id");
  if (id !== expectedRunId) fail(endpoint, "$.id", "does not match the request");
  return {
    id,
    model: optional(row, "model", (v, p) => nonEmptyText(v, endpoint, p), endpoint, "$"),
    substrate: optional(row, "substrate", (v, p) => nonEmptyText(v, endpoint, p), endpoint, "$"),
    parentRunId: optional(row, "parent_run_id", (v, p) => nullableText(v, endpoint, p), endpoint, "$"),
    identity: "identity" in row ? object(row.identity, endpoint, "$.identity") : {},
  };
}

export interface CreatedRun {
  runId: string;
  answer: string;
}

export function decodeCreatedRun(value: unknown, endpoint = "/v1/chat/completions"): CreatedRun {
  const row = object(value, endpoint, "$");
  const choices = array(required(row, "choices", endpoint, "$"), endpoint, "$.choices");
  if (choices.length !== 1) fail(endpoint, "$.choices", "must contain exactly one completion");
  const choice = object(choices[0], endpoint, "$.choices[0]");
  const message = object(required(choice, "message", endpoint, "$.choices[0]"), endpoint, "$.choices[0].message");
  return {
    runId: nonEmptyText(required(row, "clozn_run_id", endpoint, "$"), endpoint, "$.clozn_run_id"),
    answer: text(required(message, "content", endpoint, "$.choices[0].message"), endpoint, "$.choices[0].message.content"),
  };
}
