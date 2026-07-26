import type { LayerReading } from "../../data/types";

export function SignalPlot({ layers, selectedLayer }: { layers: LayerReading[]; selectedLayer: number }) {
  const points = layers.map((layer, index) => {
    const x = 6 + index * (88 / Math.max(1, layers.length - 1));
    const y = 88 - layer.energy * 12;
    return `${x},${y}`;
  }).join(" ");
  const selected = Math.max(0, layers.findIndex((layer) => layer.layer === selectedLayer));
  const selectedX = 6 + selected * (88 / Math.max(1, layers.length - 1));

  return (
    <svg className="signal-plot" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Layer energy plot">
      <defs>
        <linearGradient id="signal-fill" x1="0" y1="0" x2="0" y2="1">
          <stop stopColor="var(--signal-violet)" stopOpacity=".42" />
          <stop offset="1" stopColor="var(--signal-cyan)" stopOpacity="0" />
        </linearGradient>
        <filter id="signal-glow">
          <feGaussianBlur stdDeviation="1.2" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      <path className="plot-grid" d="M0 25H100M0 50H100M0 75H100M20 0V100M40 0V100M60 0V100M80 0V100" />
      <polygon className="plot-area" points={`6,96 ${points} 94,96`} />
      <polyline className="plot-line secondary" points={layers.map((layer, index) => {
        const x = 6 + index * (88 / Math.max(1, layers.length - 1));
        const y = 80 - layer.activation * 48 + Math.sin(index * 2) * 7;
        return `${x},${y}`;
      }).join(" ")} />
      <polyline className="plot-line" points={points} />
      <line className="plot-cursor" x1={selectedX} y1="5" x2={selectedX} y2="95" />
      <circle className="plot-point" cx={selectedX} cy={88 - layers[selected].energy * 12} r="1.7" />
    </svg>
  );
}
