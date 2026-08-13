export interface JournalRun {
  id: string;
  prompt?: string;
  response?: string;
  model?: string;
  source?: string;
  createdAt?: string;
  finishReason?: string;
  parentRunId?: string;
  sessionKey?: string;
  flags: string[];
  warningCount: number;
}

export type RunState = "complete" | "truncated" | "failed" | "recorded";

export function runTitle(run: JournalRun): string {
  return run.prompt?.trim() || `Run ${shortId(run.id)}`;
}

export function shortId(id: string): string {
  return id.length > 9 ? id.slice(-8) : id;
}

export function runState(run: JournalRun): RunState {
  if (run.flags.includes("error") || run.finishReason === "error" || run.finishReason === "failed") return "failed";
  if (run.flags.includes("truncated") || run.finishReason === "length") return "truncated";
  if (run.finishReason) return "complete";
  return "recorded";
}

export function journalDay(value?: string): string {
  if (!value) return "TIME NOT RECORDED";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  const today = new Date();
  const localDate = date.toDateString();
  if (localDate === today.toDateString()) return "TODAY";
  const yesterday = new Date(today);
  yesterday.setDate(today.getDate() - 1);
  if (localDate === yesterday.toDateString()) return "YESTERDAY";
  return new Intl.DateTimeFormat(undefined, { month: "long", day: "numeric", year: date.getFullYear() === today.getFullYear() ? undefined : "numeric" })
    .format(date)
    .toUpperCase();
}

export function journalTime(value?: string): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
}

export function groupRunsByDay(runs: readonly JournalRun[]): Array<{ day: string; runs: JournalRun[] }> {
  const groups: Array<{ day: string; runs: JournalRun[] }> = [];
  for (const run of runs) {
    const day = journalDay(run.createdAt);
    const existing = groups.at(-1);
    if (existing?.day === day) existing.runs.push(run);
    else groups.push({ day, runs: [run] });
  }
  return groups;
}

export function filterJournal(runs: readonly JournalRun[], query: string): JournalRun[] {
  const needle = query.trim().toLocaleLowerCase();
  if (!needle) return [...runs];
  return runs.filter((run) => [run.id, run.prompt, run.response, run.model, run.source]
    .some((value) => value?.toLocaleLowerCase().includes(needle)));
}
