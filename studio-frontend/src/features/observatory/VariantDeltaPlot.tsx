import type { ObservatoryData } from "../../data/types";
import type { TokenAlignment } from "../compare/alignment";

interface VariantDeltaPlotProps {
  current: ObservatoryData;
  reference: ObservatoryData | null;
  alignment: TokenAlignment;
  selectedToken: number;
}

export function VariantDeltaPlot({
  current,
  reference,
  alignment,
  selectedToken,
}: VariantDeltaPlotProps) {
  const count = Math.max(2, current.tokens.length);
  const points = current.tokens.flatMap((token, index) => {
    const columnIndex = alignment.columnByB.get(index);
    const referenceIndex = columnIndex == null ? undefined : alignment.columns[columnIndex]?.aIndex;
    const referenceToken = referenceIndex == null ? undefined : reference?.tokens[referenceIndex];
    if (!referenceToken) return [];
    const delta = (token.confidence ?? 0) - (referenceToken.confidence ?? 0);
    return [{
      index,
      x: 4 + index * (92 / Math.max(1, count - 1)),
      y: 50 - Math.max(-1, Math.min(1, delta)) * 38,
      delta,
    }];
  });
  const path = points.map((point, index) => `${index ? "L" : "M"} ${point.x} ${point.y}`).join(" ");
  const selectedX = 4 + selectedToken * (92 / Math.max(1, count - 1));

  return (
    <svg className="variant-delta-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Aligned confidence difference">
      <path className="variant-delta-grid" d="M0 12H100M0 50H100M0 88H100M20 0V100M40 0V100M60 0V100M80 0V100" />
      <path className="variant-delta-area" d={path ? `${path} L ${points.at(-1)?.x ?? 4} 50 L ${points[0]?.x ?? 4} 50 Z` : ""} />
      <path className="variant-delta-line" d={path} />
      {points.map((point) => <circle cx={point.x} cy={point.y} r="1.1" key={point.index} />)}
      <line className="variant-delta-cursor" x1={selectedX} y1="4" x2={selectedX} y2="96" />
    </svg>
  );
}
