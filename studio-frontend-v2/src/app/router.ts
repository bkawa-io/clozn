/** Display-independent coordinates retained while an investigation crosses Studio surfaces. */
export interface RouteTextRange {
  start: number;
  end: number;
}

export interface RouteLocus extends RouteTextRange {
  id: string;
}

/** A selected Compare region remains identifiable even when its prose range is absent on one side. */
export interface ComparisonRouteSelection {
  runAId: string;
  runBId: string;
  differenceId: string;
  a?: RouteTextRange;
  b?: RouteTextRange;
}

export type StudioRoute =
  | { surface: "runs" }
  | { surface: "inspect"; runId: string; comparison?: ComparisonRouteSelection }
  | { surface: "time-travel"; runId?: string; mode?: "turn" | "token"; tokenPosition?: number; breakpointId?: string; rivalTokenId?: number; answerLocus?: RouteLocus; sourceLocus?: RouteLocus; comparison?: ComparisonRouteSelection }
  | { surface: "compare"; runA?: string; runB?: string; selectedDifference?: ComparisonRouteSelection }
  | { surface: "mri"; runId?: string }
  | { surface: "runtime" };

function decode(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    return decodeURIComponent(value);
  } catch {
    return undefined;
  }
}

function queryFromHash(hash: string): URLSearchParams {
  const question = hash.indexOf("?");
  return new URLSearchParams(question < 0 ? "" : hash.slice(question + 1));
}

function integer(value: string | null): number | undefined {
  if (value === null || !/^(?:0|[1-9]\d*)$/.test(value)) return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : undefined;
}

function range(query: URLSearchParams, prefix: "a" | "b" | "answer" | "source"): RouteTextRange | undefined {
  const start = integer(query.get(`${prefix}Start`));
  const end = integer(query.get(`${prefix}End`));
  return start !== undefined && end !== undefined && end > start ? { start, end } : undefined;
}

function locus(query: URLSearchParams, prefix: "answer" | "source"): RouteLocus | undefined {
  const id = query.get(prefix);
  const coordinates = range(query, prefix);
  return id && coordinates ? { id, ...coordinates } : undefined;
}

function comparisonSelection(query: URLSearchParams, expected?: { runAId?: string; runBId?: string }): ComparisonRouteSelection | undefined {
  const runAId = query.get("compareA");
  const runBId = query.get("compareB");
  const differenceId = query.get("difference");
  if (!runAId || !runBId || !differenceId) return undefined;
  if ((expected?.runAId && expected.runAId !== runAId) || (expected?.runBId && expected.runBId !== runBId)) return undefined;
  return { runAId, runBId, differenceId, a: range(query, "a"), b: range(query, "b") };
}

function appendRange(query: URLSearchParams, prefix: "a" | "b" | "answer" | "source", value: RouteTextRange | undefined): void {
  if (!value) return;
  query.set(`${prefix}Start`, String(value.start));
  query.set(`${prefix}End`, String(value.end));
}

function appendComparison(query: URLSearchParams, selection: ComparisonRouteSelection | undefined): void {
  if (!selection) return;
  query.set("compareA", selection.runAId);
  query.set("compareB", selection.runBId);
  query.set("difference", selection.differenceId);
  appendRange(query, "a", selection.a);
  appendRange(query, "b", selection.b);
}

function withQuery(path: string, query: URLSearchParams): string {
  const value = query.toString();
  return value ? `${path}?${value}` : path;
}

export function readRoute(hash = window.location.hash): StudioRoute {
  const path = hash.replace(/^#\/?/, "").split("?")[0];
  const parts = path.split("/").filter(Boolean);
  const query = queryFromHash(hash);
  if (parts[0] === "runs" && parts[1]) {
    const runId = decode(parts[1]) ?? parts[1];
    const comparison = comparisonSelection(query);
    return comparison ? { surface: "inspect", runId, comparison } : { surface: "inspect", runId };
  }
  if (parts[0] === "inspect" && parts[1]) {
    const runId = decode(parts[1]) ?? parts[1];
    const comparison = comparisonSelection(query);
    return comparison ? { surface: "inspect", runId, comparison } : { surface: "inspect", runId };
  }
  if (parts[0] === "time-travel") {
    const runId = decode(parts[1]);
    const answerLocus = locus(query, "answer");
    const sourceLocus = locus(query, "source");
    const tokenPosition = integer(query.get("position"));
    const breakpointId = query.get("breakpoint") || undefined;
    const rivalTokenId = integer(query.get("rival"));
    const mode = query.get("mode") === "token" ? "token" as const : undefined;
    const comparison = comparisonSelection(query);
    return { surface: "time-travel", ...(mode ? { mode } : {}), ...(runId ? { runId } : {}), ...(tokenPosition !== undefined ? { tokenPosition } : {}), ...(breakpointId ? { breakpointId } : {}), ...(rivalTokenId !== undefined ? { rivalTokenId } : {}), ...(answerLocus ? { answerLocus } : {}), ...(sourceLocus ? { sourceLocus } : {}), ...(comparison ? { comparison } : {}) };
  }
  if (parts[0] === "compare") {
    const runA = decode(parts[1]);
    const runB = decode(parts[2]);
    const selectedDifference = comparisonSelection(query, { runAId: runA, runBId: runB });
    return { surface: "compare", ...(runA ? { runA } : {}), ...(runB ? { runB } : {}), ...(selectedDifference ? { selectedDifference } : {}) };
  }
  if (parts[0] === "mri") return { surface: "mri", runId: decode(parts[1]) };
  if (parts[0] === "runtime") return { surface: "runtime" };
  return { surface: "runs" };
}

export function routeHref(route: StudioRoute): string {
  switch (route.surface) {
    case "runs": return "#/runs";
    case "inspect": {
      const query = new URLSearchParams();
      appendComparison(query, route.comparison);
      return `#/runs/${encodeURIComponent(route.runId)}${withQuery("", query)}`;
    }
    case "time-travel": {
      const query = new URLSearchParams();
      if (route.mode === "token") query.set("mode", "token");
      if (route.tokenPosition !== undefined) query.set("position", String(route.tokenPosition));
      if (route.breakpointId) query.set("breakpoint", route.breakpointId);
      if (route.rivalTokenId !== undefined) query.set("rival", String(route.rivalTokenId));
      if (route.answerLocus) { query.set("answer", route.answerLocus.id); appendRange(query, "answer", route.answerLocus); }
      if (route.sourceLocus) { query.set("source", route.sourceLocus.id); appendRange(query, "source", route.sourceLocus); }
      appendComparison(query, route.comparison);
      return `#/time-travel${route.runId ? `/${encodeURIComponent(route.runId)}` : ""}${withQuery("", query)}`;
    }
    case "compare": {
      const query = new URLSearchParams();
      const selection = route.selectedDifference;
      // Avoid emitting a self-contradictory route when a caller mixes a selection from another pair.
      if (selection && selection.runAId === route.runA && selection.runBId === route.runB) appendComparison(query, selection);
      return route.runA && route.runB
        ? `#/compare/${encodeURIComponent(route.runA)}/${encodeURIComponent(route.runB)}${withQuery("", query)}`
        : "#/compare";
    }
    case "mri": return route.runId ? `#/mri/${encodeURIComponent(route.runId)}` : "#/mri";
    case "runtime": return "#/runtime";
  }
}
