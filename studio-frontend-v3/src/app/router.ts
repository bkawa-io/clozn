export type StudioRoute =
  | { surface: "sessions" }
  | { surface: "session"; sessionId: string; runId?: string };

function decode(value: string | undefined): string | undefined {
  if (!value) return undefined;
  try {
    return decodeURIComponent(value);
  } catch {
    return undefined;
  }
}

export function readRoute(hash = window.location.hash): StudioRoute {
  const [path, queryString] = hash.replace(/^#\/?/, "").split("?");
  const parts = path.split("/").filter(Boolean);
  const query = new URLSearchParams(queryString ?? "");
  if (parts[0] === "sessions" && parts[1]) {
    const sessionId = decode(parts[1]) ?? parts[1];
    const runId = query.get("run");
    return { surface: "session", sessionId, ...(runId ? { runId } : {}) };
  }
  return { surface: "sessions" };
}

export function routeHref(route: StudioRoute): string {
  if (route.surface === "sessions") return "#/sessions";
  const query = route.runId ? `?run=${encodeURIComponent(route.runId)}` : "";
  return `#/sessions/${encodeURIComponent(route.sessionId)}${query}`;
}
