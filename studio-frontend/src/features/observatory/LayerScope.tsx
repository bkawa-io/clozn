import { useMemo, type CSSProperties } from "react";
import type { LayerReading } from "../../data/types";

interface LayerScopeProps {
  layers: LayerReading[];
  selectedLayer: number;
  rotation: number;
  zoom: number;
  onSelectLayer: (layer: number) => void;
}

const panelX = [8, 24.5, 41, 57.5, 74, 90.5];
const panelY = [49, 45, 51, 45, 49, 46];

function signalPath(index: number, from: number, to: number): string {
  const x1 = panelX[from] * 10;
  const x2 = panelX[to] * 10;
  const band = (index - 4.5) * 9;
  const y1 = panelY[from] * 4.2 + band;
  const y2 = panelY[to] * 4.2 + Math.sin(index * 1.7 + to) * 32 + band * 0.3;
  const bend = (x2 - x1) * 0.46;
  return `M ${x1} ${y1} C ${x1 + bend} ${y1 - 38}, ${x2 - bend} ${y2 + 38}, ${x2} ${y2}`;
}

function layerDots(layer: LayerReading) {
  return Array.from({ length: 88 }, (_, index) => {
    const x = index % 8;
    const y = Math.floor(index / 8);
    const wave = Math.sin(index * 1.73 + layer.layer * 0.41) * 0.5 + 0.5;
    const active = wave < layer.activation;
    return (
      <i
        className={active ? "is-lit" : ""}
        key={index}
        style={{ "--dot-delay": `${(x * 17 + y * 11) % 90}ms` } as CSSProperties}
      />
    );
  });
}

export function LayerScope({ layers, selectedLayer, rotation, zoom, onSelectLayer }: LayerScopeProps) {
  const paths = useMemo(() => (
    layers.slice(0, -1).flatMap((_, layerIndex) =>
      Array.from({ length: 10 }, (_, pathIndex) => ({
        key: `${layerIndex}-${pathIndex}`,
        d: signalPath(pathIndex, layerIndex, layerIndex + 1),
        tone: (pathIndex + layerIndex) % 3,
      })),
    )
  ), [layers]);

  return (
    <div
      className="layer-scope"
      aria-label="Layer activation scope"
      style={{
        "--scope-scale": 0.9 + ((zoom - 30) / 70) * 0.12,
        "--scope-turn": `${(rotation - 40) * 0.16}deg`,
      } as CSSProperties}
    >
      <div className="scope-haze" aria-hidden="true" />
      <svg className="signal-field" viewBox="0 0 1000 420" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id="flow-cyan" x1="0" x2="1">
            <stop stopColor="#80e6e6" stopOpacity=".05" />
            <stop offset=".45" stopColor="#c3fcff" stopOpacity=".9" />
            <stop offset="1" stopColor="#8acbff" stopOpacity=".12" />
          </linearGradient>
          <linearGradient id="flow-violet" x1="0" x2="1">
            <stop stopColor="#9beaf0" stopOpacity=".08" />
            <stop offset=".5" stopColor="#dbb9ff" stopOpacity=".95" />
            <stop offset="1" stopColor="#a9a8ff" stopOpacity=".12" />
          </linearGradient>
          <linearGradient id="flow-pink" x1="0" x2="1">
            <stop stopColor="#9bdfee" stopOpacity=".08" />
            <stop offset=".5" stopColor="#ffb6e9" stopOpacity=".94" />
            <stop offset="1" stopColor="#c1b6ff" stopOpacity=".1" />
          </linearGradient>
          <filter id="flow-glow" x="-30%" y="-60%" width="160%" height="220%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>
        {paths.map((path) => (
          <g key={path.key}>
            <path className={`flow-line-base tone-${path.tone}`} d={path.d} pathLength="1" />
            <path className={`flow-line tone-${path.tone}`} d={path.d} pathLength="1" />
          </g>
        ))}
        {layers.map((layer, index) => (
          <g className="flow-node" key={layer.layer}>
            <circle cx={panelX[index] * 10} cy={panelY[index] * 4.2} r="21" />
            <circle cx={panelX[index] * 10} cy={panelY[index] * 4.2} r="4.5" />
          </g>
        ))}
      </svg>

      <div className="layer-stage">
        {layers.map((layer, index) => (
          <button
            className={`layer-plane hue-${layer.hue} ${selectedLayer === layer.layer ? "is-selected" : ""}`}
            key={layer.layer}
            type="button"
            onClick={() => onSelectLayer(layer.layer)}
            style={{
              "--layer-x": `${panelX[index]}%`,
              "--layer-y": `${panelY[index]}%`,
              "--layer-z": `${index % 2 ? -3 : 3}deg`,
            } as CSSProperties}
            aria-label={`Layer ${layer.layer}, ${layer.stage}`}
            aria-pressed={selectedLayer === layer.layer}
          >
            <span className="layer-label"><b>L{layer.layer}</b>{layer.stage}</span>
            <span className="plane-glass">
              <span className="dot-field">{layerDots(layer)}</span>
              <span className="plane-scan" />
            </span>
          </button>
        ))}
      </div>

      <div className="scope-scale" aria-label="Activation scale">
        <span>MAX</span>
        <i><b style={{ height: "72%" }} /></i>
        <span>0</span>
      </div>
    </div>
  );
}
