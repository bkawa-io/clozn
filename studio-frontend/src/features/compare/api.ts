import type {
  PairedDeltaFinding,
  PairedDeltaFindingStatus,
  PairedDeltaRow,
  PairedDeltaSummaryAxis,
  PairedDeltaValue,
} from "../../components/PairedDelta";

/**
 * The Compare screen owns this narrow transport boundary because these documents are useful only at
 * the point where a person chooses a recorded pair. Keeping it here prevents the generic runtime
 * model from acquiring a second, partially-parsed representation of a run.
 */

type JsonRecord = Record<string, unknown>;

const AXES = [
  ["model", "Model"],
  ["adapter", "Adapter"],
  ["template", "Template"],
  ["context", "Context"],
  ["sampling", "Sampling"],
  ["engine", "Engine"],
  ["tool_parse", "Tool parse"],
  ["output", "Output"],
] as const;

const FINDING_STATUSES = [
  "observed",
  "eliminated",
  "reproduced",
  "correlated",
  "causally_supported",
] as const satisfies readonly PairedDeltaFindingStatus[];

export interface RunComparisonDocument {
  runA: string;
  runB: string;
  rows: PairedDeltaRow[];
  summaryAxes: PairedDeltaSummaryAxis[];
  findings: PairedDeltaFinding[];
  privacyLimited: boolean;
  generatedAt?: string;
}

export interface RunChangeTestBudget {
  maxRuns?: number;
  maxSeconds?: number;
  runsUsed?: number;
  remainingRuns?: number;
  remainingSeconds?: number;
}

export interface RunChangeTestEntry {
  kind: string;
  status: string;
  ran: boolean;
  runsUsed?: number;
  reason: string;
  stopReason?: string;
  evidenceRunIds: string[];
}

export interface RunChangeTestDocument {
  runA: string;
  runB: string;
  status: string;
  dryRun: boolean;
  budget: RunChangeTestBudget;
  tests: RunChangeTestEntry[];
  summary: {
    classification?: string;
    causallySupported: string[];
    entangled?: boolean;
  };
}

function record(value: unknown): JsonRecord {
  return value != null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord
    : {};
}

function records(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(record) : [];
}

function string(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function errorMessage(value: JsonRecord, status: number): string {
  return string(value.error) ?? `Comparison request failed (${status}).`;
}

async function jsonRequest(url: string, init?: RequestInit): Promise<JsonRecord> {
  const response = await fetch(url, init);
  let body: JsonRecord = {};
  try {
    body = record(await response.json());
  } catch {
    // A status code is still an honest reason when an intermediary removes a JSON error body.
  }
  if (!response.ok) throw new Error(errorMessage(body, response.status));
  return body;
}

function presentValue(value: unknown, wasRecorded: boolean): PairedDeltaValue {
  if (!wasRecorded) return "Not recorded";
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return "Recorded value could not be displayed";
  }
}

function absenceReason(item: JsonRecord, dimension: string, kind: "unavailable" | "diff_failed"): string {
  return string(item.note)
    ?? `The comparison artifact marked ${dimension} as ${kind.replaceAll("_", " ")} without a more specific reason.`;
}

function differenceRows(value: unknown): PairedDeltaRow[] {
  return records(value).map((item, index) => {
    const dimension = string(item.dimension) ?? `Unlabelled dimension ${index + 1}`;
    const rawKind = string(item.kind);
    const rawRank = number(item.rank);
    const rank = rawRank ?? index;
    const note = string(item.note);
    const hasA = Object.prototype.hasOwnProperty.call(item, "value_a");
    const hasB = Object.prototype.hasOwnProperty.call(item, "value_b");

    switch (rawKind) {
      case "changed":
        return {
          id: `${dimension}-${index}`,
          dimension,
          kind: "changed",
          rank,
          valueA: presentValue(item.value_a, hasA),
          valueB: presentValue(item.value_b, hasB),
          note,
        };
      case "added":
        return {
          id: `${dimension}-${index}`,
          dimension,
          kind: "added",
          rank,
          valueB: presentValue(item.value_b, hasB),
          note,
        };
      case "removed":
        return {
          id: `${dimension}-${index}`,
          dimension,
          kind: "removed",
          rank,
          valueA: presentValue(item.value_a, hasA),
          note,
        };
      case "unavailable":
      case "diff_failed":
        return {
          id: `${dimension}-${index}`,
          dimension,
          kind: rawKind,
          rank,
          reason: absenceReason(item, dimension, rawKind),
        };
      default:
        // An unknown wire kind cannot be made into a visual delta. Preserve its position, but expose the
        // parser boundary as a failed comparison instead of implying that the dimension matched.
        return {
          id: `${dimension}-${index}`,
          dimension,
          kind: "diff_failed",
          rank,
          reason: `The comparison artifact reported an unsupported change state${rawKind ? ` (${rawKind})` : ""}.`,
          note,
        };
    }
  });
}

function summaryAxes(value: unknown): PairedDeltaSummaryAxis[] {
  const axes = record(value);
  return AXES.map(([id, label]) => {
    const axis = record(axes[id]);
    const status = string(axis.status);
    if (status === "changed" || status === "unchanged" || status === "unavailable") {
      return { id, label, status, note: string(axis.note) };
    }
    return {
      id,
      label,
      status: "unavailable",
      note: status
        ? `The comparison artifact reported an unsupported state (${status}) for this axis.`
        : "The comparison artifact did not report this axis.",
    };
  });
}

function findings(value: unknown): PairedDeltaFinding[] {
  return records(value).flatMap((item, index) => {
    const status = string(item.status);
    const label = string(item.classification);
    const summary = string(item.summary);
    if (!status || !label || !summary || !FINDING_STATUSES.includes(status as PairedDeltaFindingStatus)) {
      // A malformed finding has no safely-renderable ordinal position. It is deliberately omitted rather
      // than weakened into "observed" or upgraded into a causal claim.
      return [];
    }
    return [{ id: `${label}-${index}`, label, summary, status: status as PairedDeltaFindingStatus }];
  });
}

/** Fetches only the recorded structural comparison. Planning and execution stay behind explicit C8 clicks. */
export async function loadRunComparison(
  runA: string,
  runB: string,
  signal?: AbortSignal,
): Promise<RunComparisonDocument> {
  const query = new URLSearchParams({ a: runA, b: runB });
  const body = await jsonRequest(`/runs/compare?${query.toString()}`, { signal });
  const schemaVersion = string(body.schema_version);
  if (schemaVersion !== "clozn.run-diff.v1") {
    throw new Error("Comparison response did not contain a clozn.run-diff.v1 document.");
  }

  return {
    runA: string(body.run_a) ?? runA,
    runB: string(body.run_b) ?? runB,
    rows: differenceRows(body.differences),
    summaryAxes: summaryAxes(body.summary_axes),
    findings: findings(body.findings),
    privacyLimited: body.privacy_limited === true,
    generatedAt: string(body.generated_at),
  };
}

function budget(value: unknown): RunChangeTestBudget {
  const item = record(value);
  return {
    maxRuns: number(item.max_runs),
    maxSeconds: number(item.max_seconds),
    runsUsed: number(item.runs_used),
    remainingRuns: number(item.remaining_runs),
    remainingSeconds: number(item.remaining_seconds),
  };
}

function changeTests(value: unknown): RunChangeTestEntry[] {
  return records(value).map((item, index) => ({
    kind: string(item.kind) ?? `unknown-${index + 1}`,
    status: string(item.status) ?? "unavailable",
    ran: item.ran === true,
    runsUsed: number(item.runs_used),
    reason: string(item.reason) ?? "The change-test artifact did not record a reason.",
    stopReason: string(item.stop_reason),
    evidenceRunIds: records(item.evidence)
      .map((evidence) => string(evidence.run_id))
      .filter((id): id is string => id != null),
  }));
}

function changeTestDocument(value: JsonRecord): RunChangeTestDocument {
  const schemaVersion = string(value.schema_version);
  if (schemaVersion !== "clozn.run-change-test.v1") {
    throw new Error("Change-test response did not contain a clozn.run-change-test.v1 document.");
  }
  const summary = record(value.summary);
  return {
    runA: string(value.run_a) ?? "",
    runB: string(value.run_b) ?? "",
    status: string(value.status) ?? "inconclusive",
    dryRun: value.dry_run === true,
    budget: budget(value.budget),
    tests: changeTests(value.tests),
    summary: {
      classification: string(summary.classification),
      causallySupported: Array.isArray(summary.causally_supported)
        ? summary.causally_supported.filter((item): item is string => typeof item === "string")
        : [],
      entangled: summary.entangled === true ? true : undefined,
    },
  };
}

/** A dry-run is still initiated only by the user: it creates a zero-run bounded plan, never a child run. */
export async function previewRunChangeTest(
  runA: string,
  runB: string,
  signal?: AbortSignal,
): Promise<RunChangeTestDocument> {
  const body = await jsonRequest("/runs/compare/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ a: runA, b: runB, plan: true, max_runs: 4, max_seconds: 120 }),
    signal,
  });
  return changeTestDocument(body);
}

/** Executes only the swaps a preview identified as available, under that preview's recorded cap. */
export async function runChangeTest(
  runA: string,
  runB: string,
  tests: readonly string[],
  budget: RunChangeTestBudget,
  signal?: AbortSignal,
): Promise<RunChangeTestDocument> {
  const body = await jsonRequest("/runs/compare/test", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      a: runA,
      b: runB,
      tests,
      ...(budget.maxRuns != null ? { max_runs: budget.maxRuns } : {}),
      ...(budget.maxSeconds != null ? { max_seconds: budget.maxSeconds } : {}),
    }),
    signal,
  });
  return changeTestDocument(body);
}

export function plannedChangeTests(document: RunChangeTestDocument): RunChangeTestEntry[] {
  return document.tests.filter((test) => test.stopReason === "planned" && test.status === "not_run");
}
