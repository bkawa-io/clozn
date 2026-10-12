export const INSPECT_VIEWS = [
  "read",
  "context-effects",
  "context",
  "receipt",
  "minimal-context",
  "actions",
  "claims",
] as const;

export type InspectView = (typeof INSPECT_VIEWS)[number];

type InspectRouteBase = {
  surface: "inspect";
  runId: string;
  sessionId?: string;
};

type InspectReaderRoute = InspectRouteBase & (
  | {
      view?: "read" | "context-effects";
      answerStart: number;
      answerEnd: number;
      sourceId?: string;
    }
  | {
      view?: "read" | "context-effects";
      answerStart?: never;
      answerEnd?: never;
      sourceId?: never;
    }
);

type InspectSimpleRoute = InspectRouteBase & {
  view: Exclude<InspectView, "read" | "context-effects">;
  answerStart?: never;
  answerEnd?: never;
  sourceId?: never;
};

export type InspectRoute = InspectReaderRoute | InspectSimpleRoute;

export type CompareRoute =
  | { surface: "compare"; runA?: never; runB?: never }
  | { surface: "compare"; runA: string; runB: string }
  | { surface: "compare"; mode: "matrix"; experimentId?: string; runA?: never; runB?: never };

export type StudioRoute =
  | { surface: "sessions" }
  | { surface: "session"; sessionId: string; runId?: string }
  | InspectRoute
  | CompareRoute
  | { surface: "runtime" };

function decodePathPart(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const decoded = decodeURIComponent(value);
    return decoded.trim() ? decoded : undefined;
  } catch {
    return undefined;
  }
}

function nonEmpty(value: string | null): string | undefined {
  return value?.trim() ? value : undefined;
}

function coordinate(value: string | null): number | undefined {
  if (value === null || !/^(0|[1-9]\d*)$/.test(value)) return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : undefined;
}

function answerRange(
  startValue: string | null,
  endValue: string | null,
): { answerStart: number; answerEnd: number } | undefined {
  const answerStart = coordinate(startValue);
  const answerEnd = coordinate(endValue);
  return answerStart !== undefined && answerEnd !== undefined && answerEnd > answerStart
    ? { answerStart, answerEnd }
    : undefined;
}

function inspectView(value: string | null): InspectView | undefined {
  return INSPECT_VIEWS.find((candidate) => candidate === value);
}

function hasOnlyQueryKeys(query: URLSearchParams, allowed: readonly string[]): boolean {
  const allowedKeys = new Set(allowed);
  const keys = [...query.keys()];
  return new Set(keys).size === keys.length && keys.every((key) => allowedKeys.has(key));
}

export function readRoute(hash = window.location.hash): StudioRoute {
  const routeText = hash.replace(/^#\/?/, "");
  const queryIndex = routeText.indexOf("?");
  const path = queryIndex === -1 ? routeText : routeText.slice(0, queryIndex);
  const queryString = queryIndex === -1 ? "" : routeText.slice(queryIndex + 1);
  const parts = path.split("/").filter(Boolean);
  const query = new URLSearchParams(queryString);

  if (parts[0] === "inspect" && parts.length === 2) {
    const runId = decodePathPart(parts[1]);
    if (!runId) return { surface: "sessions" };
    const sessionId = nonEmpty(query.get("sessionId"));
    if (query.has("sessionId") && !sessionId) return { surface: "sessions" };
    const viewValue = query.get("view");
    const view = viewValue === null ? undefined : inspectView(viewValue);
    if (viewValue !== null && !view) return { surface: "sessions" };

    if (view && view !== "read" && view !== "context-effects") {
      if (!hasOnlyQueryKeys(query, ["view", "sessionId"])) {
        return { surface: "sessions" };
      }
      return { surface: "inspect", runId, view, ...(sessionId ? { sessionId } : {}) };
    }

    if (!hasOnlyQueryKeys(query, ["view", "sessionId", "answerStart", "answerEnd", "sourceId"])) {
      return { surface: "sessions" };
    }

    const startValue = query.get("answerStart");
    const endValue = query.get("answerEnd");
    const sourceValue = query.get("sourceId");
    const range = answerRange(startValue, endValue);
    if (!range && (startValue !== null || endValue !== null || sourceValue !== null)) {
      return { surface: "sessions" };
    }
    if (!range) {
      return {
        surface: "inspect",
        runId,
        ...(view ? { view } : {}),
        ...(sessionId ? { sessionId } : {}),
      };
    }
    const sourceId = nonEmpty(sourceValue);
    if (sourceValue !== null && !sourceId) return { surface: "sessions" };
    return {
      surface: "inspect",
      runId,
      ...(view ? { view } : {}),
      ...(sessionId ? { sessionId } : {}),
      ...range,
      ...(sourceId ? { sourceId } : {}),
    };
  }


  if (!queryString && parts.length === 1 && parts[0] === "compare") return { surface: "compare" };
  if (!queryString && parts[0] === "compare" && parts[1] === "matrix" && parts.length <= 3) {
    if (parts.length === 2) return { surface: "compare", mode: "matrix" };
    const experimentId = decodePathPart(parts[2]);
    return experimentId
      ? { surface: "compare", mode: "matrix", experimentId }
      : { surface: "sessions" };
  }
  if (!queryString && parts[0] === "compare" && parts.length === 3) {
    const runA = decodePathPart(parts[1]);
    const runB = decodePathPart(parts[2]);
    return runA && runB && runA !== runB
      ? { surface: "compare", runA, runB }
      : { surface: "sessions" };
  }

  if (parts.length === 1 && parts[0] === "runtime" && !queryString) {
    return { surface: "runtime" };
  }
  if (parts[0] === "sessions" && parts.length === 2) {
    const sessionId = decodePathPart(parts[1]);
    if (!sessionId) return { surface: "sessions" };
    if (!hasOnlyQueryKeys(query, ["run"])) return { surface: "sessions" };
    const runId = nonEmpty(query.get("run"));
    if (query.has("run") && !runId) return { surface: "sessions" };
    return { surface: "session", sessionId, ...(runId ? { runId } : {}) };
  }
  if (parts.length === 0 || (parts.length === 1 && parts[0] === "sessions" && !queryString)) {
    return { surface: "sessions" };
  }
  return { surface: "sessions" };
}

function queryEntry(name: string, value: string | number): string {
  return `${name}=${encodeURIComponent(String(value))}`;
}

export function routeHref(route: StudioRoute): string {
  if (route.surface === "sessions") return "#/sessions";
  if (route.surface === "runtime") return "#/runtime";
  if (route.surface === "session") {
    const query = route.runId ? `?${queryEntry("run", route.runId)}` : "";
    return `#/sessions/${encodeURIComponent(route.sessionId)}${query}`;
  }

  if (route.surface === "compare") {
    if ("mode" in route && route.mode === "matrix") {
      return route.experimentId?.trim()
        ? `#/compare/matrix/${encodeURIComponent(route.experimentId)}`
        : "#/compare/matrix";
    }
    return route.runA?.trim() && route.runB?.trim() && route.runA !== route.runB
      ? `#/compare/${encodeURIComponent(route.runA)}/${encodeURIComponent(route.runB)}`
      : "#/compare";
  }

  if (!route.runId.trim()) return "#/sessions";
  const query: string[] = [];
  if (route.view) query.push(queryEntry("view", route.view));
  if (route.sessionId?.trim()) query.push(queryEntry("sessionId", route.sessionId));
  if (
    (!route.view || route.view === "read" || route.view === "context-effects")
    &&
    Number.isSafeInteger(route.answerStart)
    && Number.isSafeInteger(route.answerEnd)
    && route.answerStart !== undefined
    && route.answerEnd !== undefined
    && route.answerStart >= 0
    && route.answerEnd > route.answerStart
  ) {
    query.push(queryEntry("answerStart", route.answerStart));
    query.push(queryEntry("answerEnd", route.answerEnd));
    if (route.sourceId?.trim()) query.push(queryEntry("sourceId", route.sourceId));
  }
  return `#/inspect/${encodeURIComponent(route.runId)}${query.length ? `?${query.join("&")}` : ""}`;
}
