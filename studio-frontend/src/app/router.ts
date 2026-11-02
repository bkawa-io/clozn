export const INSPECT_VIEWS = ["context-effects", "lean-context"] as const;

export type InspectView = (typeof INSPECT_VIEWS)[number];
export type EffectsGranularity = "message" | "adaptive";
export type ScoringMode = "fast" | "exact";

export type InspectRoute = {
  surface: "inspect";
  runId: string;
  conversationId?: string;
  view?: InspectView;
  answerStart?: number;
  answerEnd?: number;
  sourceId?: string;
  granularity?: EffectsGranularity;
  armBudget?: number;
  scoringMode?: ScoringMode;
};

export type RunsRoute = { surface: "runs"; conversationId?: string };
export type StudioRoute = RunsRoute | InspectRoute;

function decodePathPart(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    const decoded = decodeURIComponent(value);
    return decoded.trim() ? decoded : undefined;
  } catch {
    return undefined;
  }
}

function coordinate(value: string | null): number | undefined {
  if (value === null || !/^(0|[1-9]\d*)$/.test(value)) return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) ? parsed : undefined;
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
    if (!runId || !hasOnlyQueryKeys(query, [
      "view", "conversationId", "answerStart", "answerEnd", "sourceId", "granularity", "armBudget", "scoringMode",
    ])) {
      return { surface: "runs" };
    }
    const viewValue = query.get("view");
    const view = viewValue === null
      ? undefined
      : INSPECT_VIEWS.find((candidate) => candidate === viewValue);
    if (viewValue !== null && !view) return { surface: "runs" };
    const conversationValue = query.get("conversationId");
    const conversationId = conversationValue?.trim() || undefined;
    if (conversationValue !== null && !conversationId) return { surface: "runs" };

    const startValue = query.get("answerStart");
    const endValue = query.get("answerEnd");
    const sourceValue = query.get("sourceId");
    const granularityValue = query.get("granularity");
    const budgetValue = query.get("armBudget");
    const scoringModeValue = query.get("scoringMode");
    if (view === "lean-context" && (
      startValue !== null || endValue !== null || sourceValue !== null
      || granularityValue !== null || budgetValue !== null || scoringModeValue !== null
    )) {
      return { surface: "runs" };
    }
    if ((granularityValue === null) !== (budgetValue === null)) return { surface: "runs" };
    const granularity = granularityValue === "message" || granularityValue === "adaptive"
      ? granularityValue
      : undefined;
    const armBudget = coordinate(budgetValue);
    if (granularityValue !== null && (!granularity || armBudget === undefined || armBudget < 1)) {
      return { surface: "runs" };
    }
    const scoringMode = scoringModeValue === "fast" || scoringModeValue === "exact"
      ? scoringModeValue
      : undefined;
    if (scoringModeValue !== null && !scoringMode) return { surface: "runs" };
    const configuration: Pick<InspectRoute, "granularity" | "armBudget" | "scoringMode"> = {
      ...(granularity && armBudget !== undefined ? { granularity, armBudget } : {}),
      ...(scoringMode ? { scoringMode } : {}),
    };
    if (startValue === null && endValue === null && sourceValue === null) {
      return {
        surface: "inspect",
        runId,
        ...(conversationId ? { conversationId } : {}),
        ...(view ? { view } : {}),
        ...configuration,
      };
    }
    const answerStart = coordinate(startValue);
    const answerEnd = coordinate(endValue);
    if (answerStart === undefined || answerEnd === undefined || answerEnd <= answerStart) {
      return { surface: "runs" };
    }
    let sourceId: string | undefined;
    if (sourceValue !== null) {
      sourceId = sourceValue.trim() ? sourceValue : undefined;
      if (!sourceId) return { surface: "runs" };
    }
    return {
      surface: "inspect",
      runId,
      ...(conversationId ? { conversationId } : {}),
      ...(view ? { view } : {}),
      ...configuration,
      answerStart,
      answerEnd,
      ...(sourceId ? { sourceId } : {}),
    };
  }

  if (
    parts.length === 0
    || (parts.length === 1 && (parts[0] === "runs" || parts[0] === "conversations") && !queryString)
  ) {
    return { surface: "runs" };
  }
  if (parts.length === 2 && parts[0] === "conversations" && !queryString) {
    const conversationId = decodePathPart(parts[1]);
    if (conversationId) return { surface: "runs", conversationId };
  }
  return { surface: "runs" };
}

function queryEntry(name: string, value: string | number): string {
  return `${name}=${encodeURIComponent(String(value))}`;
}

export function routeHref(route: StudioRoute): string {
  if (route.surface === "runs") {
    return route.conversationId?.trim()
      ? `#/conversations/${encodeURIComponent(route.conversationId)}`
      : "#/runs";
  }
  if (!route.runId.trim()) return "#/runs";
  const query: string[] = [];
  if (route.conversationId?.trim()) query.push(queryEntry("conversationId", route.conversationId));
  if (route.view) query.push(queryEntry("view", route.view));
  if (route.view !== "lean-context" && (route.scoringMode === "fast" || route.scoringMode === "exact")) {
    query.push(queryEntry("scoringMode", route.scoringMode));
  }
  if (
    route.view !== "lean-context"
    && (route.granularity === "message" || route.granularity === "adaptive")
    && Number.isSafeInteger(route.armBudget)
    && route.armBudget !== undefined
    && route.armBudget > 0
  ) {
    query.push(queryEntry("granularity", route.granularity));
    query.push(queryEntry("armBudget", route.armBudget));
  }
  if (
    route.view !== "lean-context"
    && Number.isSafeInteger(route.answerStart)
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
