/**
 * Client-side store for C4's investigation history -- one entry per "ask another question" chip click
 * or resolved free-text route, kept as its own vocabulary, separate from anything resembling a chat/
 * topic conversation transcript (notes/EPIC_ROADMAP_A-F_Q_M.md's own C4 acceptance criteria: "topic
 * follow-up and investigation visually distinct" and "investigations revisitable from the parent run").
 *
 * WHERE THIS LIVES, AND WHY
 * --------------------------
 * There is no backend investigation-history endpoint -- a FINDING from this feature's implementation,
 * not a gap this file papers over. Adding one would be a server change, outside this feature's
 * ownership (`studio-frontend/src/**` only). Within that boundary the only real options were plain
 * React state, which is lost the moment Lens remounts on a run change (`key={params.runId ?? "latest"}`
 * in `src/panels/lens.tsx`) -- so a user who opens "Why?", reads the finding, and comes back would find
 * their own investigation gone -- or a client-side store that survives exactly that round trip.
 * `window.localStorage` is the one already available without inventing a second persistence mechanism
 * this codebase doesn't otherwise have. The key is shaped like this codebase's own `clozn.<name>.v1`
 * server artifact names for the same reason a filename matches a schema version elsewhere in this repo
 * (recognizable, greppable) -- but nothing here is server-persisted, versioned by a schema registry, or
 * shared across a browser; this file is the one and only authority on this shape.
 *
 * Every entry records ONLY what is needed to describe and re-open an investigation: which run, which
 * question, when, and -- for a free-text route -- the literal text the user typed. It never stores the
 * run's own prompt, response, or evidence content. Entries are capped at MAX_ENTRIES, oldest dropped
 * first, so a long-lived browser profile never accumulates an unbounded log.
 */

import type { InvestigationQuestionId } from "./askAnotherQuestion";

const STORAGE_KEY = "clozn.investigation-history.v1";
const MAX_ENTRIES = 200;

export interface InvestigationHistoryEntry {
  entryId: string;
  runId: string;
  questionId: InvestigationQuestionId;
  questionLabel: string;
  targetDescription: string;
  origin: "chip" | "free_text";
  /** Present only when `origin === "free_text"` -- the literal text the user typed, echoed back for
   * their own history, never re-interpreted. */
  queryText?: string;
  ts: number;
}

function hasStorage(): boolean {
  try {
    return typeof window !== "undefined" && Boolean(window.localStorage);
  } catch {
    // A browser can throw reading `localStorage` itself under some privacy settings -- treat that
    // exactly like "no store available", never let it crash the panel.
    return false;
  }
}

function isEntry(value: unknown): value is InvestigationHistoryEntry {
  if (!value || typeof value !== "object") return false;
  const item = value as Record<string, unknown>;
  return typeof item.entryId === "string"
    && typeof item.runId === "string"
    && typeof item.questionId === "string"
    && typeof item.questionLabel === "string"
    && typeof item.targetDescription === "string"
    && (item.origin === "chip" || item.origin === "free_text")
    && typeof item.ts === "number"
    && (item.queryText === undefined || typeof item.queryText === "string");
}

function readAll(): InvestigationHistoryEntry[] {
  if (!hasStorage()) return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isEntry) : [];
  } catch {
    // Malformed JSON (a hand-edited store, a future format this build predates) degrades to an empty
    // history rather than throwing -- the panel must still render.
    return [];
  }
}

function writeAll(entries: InvestigationHistoryEntry[]): void {
  if (!hasStorage()) return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
  } catch {
    // Best effort -- a full or disabled store just means this session's investigations do not persist;
    // the panel keeps working from whatever is already in React state.
  }
}

/** Every recorded investigation across every run, newest first, optionally filtered to one run -- used
 * both by the "past investigations for this run" list (`runId` supplied) and by anything that later
 * wants the full cross-run log. */
export function loadInvestigationHistory(runId?: string): InvestigationHistoryEntry[] {
  const all = readAll().sort((a, b) => b.ts - a.ts);
  return runId ? all.filter((entry) => entry.runId === runId) : all;
}

export function recordInvestigation(
  input: Omit<InvestigationHistoryEntry, "entryId" | "ts">,
): InvestigationHistoryEntry {
  const entry: InvestigationHistoryEntry = {
    ...input,
    entryId: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
    ts: Date.now(),
  };
  writeAll([entry, ...readAll()]);
  return entry;
}
