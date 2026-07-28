import type { CellStatus } from "./types";

/** Presentation-only formatting. None of this re-derives a fact -- it only reformats values the server
 * already computed and verified (see api.ts's header comment on why the summary is trusted, not
 * recomputed). */

export function formatPassRate(rate: number | null): string {
  return rate == null ? "UNSCORED" : `${(rate * 100).toFixed(1)}%`;
}

export function formatTimestamp(value: string | null): string {
  if (!value) return "—";
  const match = value.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/);
  return match ? `${match[1]}-${match[2]}-${match[3]} ${match[4]}:${match[5]}:${match[6]}` : value;
}

export function shortId(id: string, length = 10): string {
  return id.length <= length ? id : `…${id.slice(-length)}`;
}

const STATUS_LABELS: Record<string, string> = {
  pass: "PASS",
  fail: "FAIL",
  error: "ERROR",
  unscored: "UNSCORED",
};

export function statusLabel(status: string): string {
  return STATUS_LABELS[status] ?? status.toUpperCase();
}

/** Priority order when collapsing several replicate-seed cells at one (suite, case, variant) coordinate
 * into a single matrix badge: an error or a failed assertion is more informative than a pass, so it
 * wins the badge even when most seeds passed -- the replicate count alongside it shows the split. */
const STATUS_PRIORITY: CellStatus[] = ["error", "fail", "pass", "unscored"];

export function dominantStatus(counts: Record<string, number>): CellStatus {
  for (const status of STATUS_PRIORITY) {
    if ((counts[status] ?? 0) > 0) return status;
  }
  return "unscored";
}
