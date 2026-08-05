import { Icon } from "../components/Icon";
import { Compare } from "../features/compare/Compare";
import type { PanelContext, StudioPanel } from "./types";

function matrixParams(hash: string, prefix: "compare" | "experiments"): Record<string, string> | null {
  const pattern = prefix === "compare"
    ? /^#\/compare\/(?:matrix|experiments)(?:\/([^/?]+))?\/?(?:\?(.*))?$/
    : /^#\/experiments(?:\/([^/?]+))?\/?(?:\?(.*))?$/;
  const match = hash.match(pattern);
  if (!match) return null;
  const params: Record<string, string> = {
    mode: "matrix",
    matrixBase: prefix === "experiments" ? "legacy" : "canonical",
  };
  if (match[1]) params.id = decodeURIComponent(match[1]);
  if (match[2] != null) params.q = match[2];
  return params;
}

const panel: StudioPanel = {
  id: "compare",
  navLabel: "Compare",
  order: 40,
  icon: () => <Icon name="compare" />,
  match: (hash) => {
    // Compare claims old experiment URLs before the hidden compatibility panel can. That keeps Compare
    // highlighted in the rail while preserving shareable legacy hashes and their query-string filters.
    const legacyMatrix = matrixParams(hash, "experiments");
    if (legacyMatrix) return legacyMatrix;
    const canonicalMatrix = matrixParams(hash, "compare");
    if (canonicalMatrix) return canonicalMatrix;

    const pair = hash.match(/^#\/compare\/([^/?]+)\/([^/?]+)\/?(?:\?.*)?$/);
    if (pair) return { runA: decodeURIComponent(pair[1]), runB: decodeURIComponent(pair[2]) };

    const bare = hash.match(/^#\/compare\/?(?:\?(.*))?$/);
    if (!bare) return null;
    if (bare[1] && new URLSearchParams(bare[1]).get("mode") === "matrix") {
      return { mode: "matrix", matrixBase: "canonical", q: bare[1] };
    }
    return {};
  },
  routeName: (params) => params.mode === "matrix" ? "COMPARE MATRIX" : "COMPARE",
  modeChip: ({ params }) => params.mode === "matrix" ? "MATRIX" : "A / B",
  Component: ({ runtime, inspectorOpen, params }: PanelContext) => (
    <Compare
      key={params.mode === "matrix"
        ? `matrix:${params.matrixBase ?? "canonical"}:${params.id ?? ""}:${params.q ?? ""}`
        : `runs:${params.runA ?? ""}:${params.runB ?? ""}`}
      runtime={runtime}
      initialA={params.runA}
      initialB={params.runB}
      inspectorOpen={inspectorOpen}
      mode={params.mode === "matrix" ? "matrix" : "runs"}
      initialExperimentId={params.id}
      rawExperimentQuery={params.q}
      experimentRouteBase={params.matrixBase === "legacy" ? "#/experiments" : "#/compare/matrix"}
    />
  ),
};

export default panel;
