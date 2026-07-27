import type { TokenReading } from "../../data/types";

export function ConfidencePlot({ tokens, selectedToken }: { tokens: TokenReading[]; selectedToken: number }) {
  const maxEntropy = tokens.reduce((maximum, token) => Math.max(maximum, token.entropy), 0.001);
  const sampleLimit = 512;
  const step = Math.max(1, Math.ceil(tokens.length / sampleLimit));
  const sampleIndexes = new Set<number>([0, Math.max(0, tokens.length - 1), selectedToken]);
  for (let index = 0; index < tokens.length; index += step) sampleIndexes.add(index);
  const sampled = [...sampleIndexes].sort((a, b) => a - b);
  const point = (index: number, value: number) => {
    const x = 4 + index * (92 / Math.max(1, tokens.length - 1));
    const y = 92 - value * 78;
    return `${x},${y}`;
  };
  const confidencePoints = sampled
    .map((index) => point(index, tokens[index]?.confidence ?? 0))
    .join(" ");
  const entropyPoints = sampled
    .map((index) => point(index, (tokens[index]?.entropy ?? 0) / maxEntropy))
    .join(" ");
  const selectedX = 4 + selectedToken * (92 / Math.max(1, tokens.length - 1));
  const selectedY = 92 - (tokens[selectedToken]?.confidence ?? 0) * 78;

  return (
    <svg className="signal-plot" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Token confidence plot">
      <defs>
        <linearGradient id="confidence-fill" x1="0" y1="0" x2="0" y2="1">
          <stop stopColor="var(--signal-mint)" stopOpacity=".34" />
          <stop offset="1" stopColor="var(--signal-cyan)" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path className="plot-grid" d="M0 25H100M0 50H100M0 75H100M20 0V100M40 0V100M60 0V100M80 0V100" />
      <polygon className="plot-area confidence-area" points={`4,96 ${confidencePoints} 96,96`} />
      <polyline className="plot-line secondary" points={entropyPoints} />
      <polyline className="plot-line confidence-line" points={confidencePoints} />
      <line className="plot-cursor" x1={selectedX} y1="5" x2={selectedX} y2="95" />
      <circle className="plot-point" cx={selectedX} cy={selectedY} r="1.7" />
    </svg>
  );
}
