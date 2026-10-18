import { ContractError } from "./contracts";

type JsonRecord = Record<string, unknown>;

export interface ReductionComparison {
  schemaVersion: "clozn.reduction-comparison.v1";
  parentRunId: string;
  childRunId: string;
  context: {
    parentSourceIds: readonly string[];
    retainedSourceIds: readonly string[];
    removedSourceIds: readonly string[];
  };
  renderedPromptTokens: {
    parent: number | null;
    child: number | null;
    saved: number | null;
    improved: boolean;
  };
  preservation: {
    answerExactMatch: boolean;
    finishReasonMatch: boolean;
    exact: boolean;
  };
  generationIdentity: {
    parent: JsonRecord;
    child: JsonRecord;
    match: boolean;
  };
}

function fail(endpoint: string, path: string, message: string): never {
  throw new ContractError(endpoint, `${path} ${message}`);
}

function object(value: unknown, endpoint: string, path: string): JsonRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail(endpoint, path, "must be an object");
  return value as JsonRecord;
}

function required(row: JsonRecord, key: string, endpoint: string, path: string): unknown {
  if (!(key in row)) fail(endpoint, path, `is missing required field ${key}`);
  return row[key];
}

function text(value: unknown, endpoint: string, path: string): string {
  if (typeof value !== "string" || !value.trim()) fail(endpoint, path, "must be a non-empty string");
  return value;
}

function booleanValue(value: unknown, endpoint: string, path: string): boolean {
  if (typeof value !== "boolean") fail(endpoint, path, "must be a boolean");
  return value;
}

function nullableInteger(value: unknown, endpoint: string, path: string): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isSafeInteger(value)) fail(endpoint, path, "must be an integer or null");
  return value;
}

function stringArray(value: unknown, endpoint: string, path: string): string[] {
  if (!Array.isArray(value)) fail(endpoint, path, "must be an array");
  return value.map((item, index) => text(item, endpoint, `${path}[${index}]`));
}

export function decodeRunComparison(
  value: unknown,
  endpoint = "/runs/compare",
  expectedParent?: string,
  expectedChild?: string,
): ReductionComparison {
  const row = object(value, endpoint, "$");
  if (required(row, "schema_version", endpoint, "$") !== "clozn.reduction-comparison.v1") {
    fail(endpoint, "$.schema_version", "must be clozn.reduction-comparison.v1");
  }
  if (required(row, "ok", endpoint, "$") !== true) fail(endpoint, "$.ok", "must be true");
  const parentRunId = text(required(row, "parent_run_id", endpoint, "$"), endpoint, "$.parent_run_id");
  const childRunId = text(required(row, "child_run_id", endpoint, "$"), endpoint, "$.child_run_id");
  if (expectedParent !== undefined && parentRunId !== expectedParent) fail(endpoint, "$.parent_run_id", "does not match the request");
  if (expectedChild !== undefined && childRunId !== expectedChild) fail(endpoint, "$.child_run_id", "does not match the request");

  const context = object(required(row, "context", endpoint, "$"), endpoint, "$.context");
  const tokens = object(required(row, "rendered_prompt_tokens", endpoint, "$"), endpoint, "$.rendered_prompt_tokens");
  const preservation = object(required(row, "preservation", endpoint, "$"), endpoint, "$.preservation");
  const identity = object(required(row, "generation_identity", endpoint, "$"), endpoint, "$.generation_identity");
  return {
    schemaVersion: "clozn.reduction-comparison.v1",
    parentRunId,
    childRunId,
    context: {
      parentSourceIds: stringArray(required(context, "parent_source_ids", endpoint, "$.context"), endpoint, "$.context.parent_source_ids"),
      retainedSourceIds: stringArray(required(context, "retained_source_ids", endpoint, "$.context"), endpoint, "$.context.retained_source_ids"),
      removedSourceIds: stringArray(required(context, "removed_source_ids", endpoint, "$.context"), endpoint, "$.context.removed_source_ids"),
    },
    renderedPromptTokens: {
      parent: nullableInteger(required(tokens, "parent", endpoint, "$.rendered_prompt_tokens"), endpoint, "$.rendered_prompt_tokens.parent"),
      child: nullableInteger(required(tokens, "child", endpoint, "$.rendered_prompt_tokens"), endpoint, "$.rendered_prompt_tokens.child"),
      saved: nullableInteger(required(tokens, "saved", endpoint, "$.rendered_prompt_tokens"), endpoint, "$.rendered_prompt_tokens.saved"),
      improved: booleanValue(required(tokens, "improved", endpoint, "$.rendered_prompt_tokens"), endpoint, "$.rendered_prompt_tokens.improved"),
    },
    preservation: {
      answerExactMatch: booleanValue(required(preservation, "answer_exact_match", endpoint, "$.preservation"), endpoint, "$.preservation.answer_exact_match"),
      finishReasonMatch: booleanValue(required(preservation, "finish_reason_match", endpoint, "$.preservation"), endpoint, "$.preservation.finish_reason_match"),
      exact: booleanValue(required(preservation, "exact", endpoint, "$.preservation"), endpoint, "$.preservation.exact"),
    },
    generationIdentity: {
      parent: object(required(identity, "parent", endpoint, "$.generation_identity"), endpoint, "$.generation_identity.parent"),
      child: object(required(identity, "child", endpoint, "$.generation_identity"), endpoint, "$.generation_identity.child"),
      match: booleanValue(required(identity, "match", endpoint, "$.generation_identity"), endpoint, "$.generation_identity.match"),
    },
  };
}
