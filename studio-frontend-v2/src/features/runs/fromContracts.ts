import type { RunRecord } from "../../data/contracts";
import type { JournalRun } from "./presenters";

function recordedDate(run: RunRecord): string | undefined {
  if (run.createdAt) return run.createdAt;
  if (run.createdTs != null) return new Date(run.createdTs * 1_000).toISOString();
  return undefined;
}

export function toJournalRun(run: RunRecord): JournalRun {
  return {
    id: run.id,
    prompt: run.promptSummary ?? undefined,
    response: run.responseSummary ?? undefined,
    model: run.model ?? undefined,
    source: run.source ?? run.client ?? undefined,
    createdAt: recordedDate(run),
    finishReason: run.finishReason ?? undefined,
    parentRunId: run.parentRunId ?? undefined,
    sessionKey: run.sessionKey ?? undefined,
    flags: [...(run.flags ?? [])],
    warningCount: run.warningCount ?? 0,
  };
}

export function toJournalRuns(runs: readonly RunRecord[]): JournalRun[] {
  return runs.map(toJournalRun);
}
