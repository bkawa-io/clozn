import type { ObservatoryData } from "./types";

export const DEMO_OBSERVATORY: ObservatoryData = {
  id: "demo_0142",
  label: "Layer scope",
  model: "Qwen2.5-7B-Instruct",
  quant: "Q5_K_M",
  createdAt: "14:32:18",
  duration: "2.8 s",
  mode: "demo",
  layerEvidence: "demo",
  layers: [
    { layer: 0, stage: "INPUT", activation: 0.24, energy: 1.82, stability: 0.91, features: 18, hue: "cyan" },
    { layer: 6, stage: "EARLY", activation: 0.41, energy: 2.14, stability: 0.82, features: 31, hue: "mint" },
    { layer: 12, stage: "MIDDLE", activation: 0.68, energy: 3.72, stability: 0.64, features: 44, hue: "violet" },
    { layer: 18, stage: "COMMIT", activation: 0.92, energy: 5.21, stability: 0.78, features: 27, hue: "pink" },
    { layer: 24, stage: "LATE", activation: 0.57, energy: 3.08, stability: 0.88, features: 22, hue: "magenta" },
    { layer: 30, stage: "OUTPUT", activation: 0.36, energy: 1.94, stability: 0.96, features: 12, hue: "peach" },
  ],
  tokens: [
    { text: "The", entropy: 0.22, source: "system" },
    { text: "transparent", entropy: 0.64, source: "context 2" },
    { text: "layer", entropy: 0.38, source: "context 2" },
    { text: "reveals", entropy: 0.81, source: "context 4" },
    { text: "where", entropy: 1.12, source: "context 4" },
    { text: "the", entropy: 0.28, source: "system" },
    { text: "response", entropy: 0.73, source: "context 1" },
    { text: "commits", entropy: 1.44, source: "context 3" },
    { text: ".", entropy: 0.13, source: "weights" },
  ],
  candidates: [
    { token: "commits", score: 0.72, delta: 0.18 },
    { token: "settles", score: 0.19, delta: -0.08 },
    { token: "forms", score: 0.09, delta: -0.04 },
  ],
  sources: [],
  configuration: {
    activeDials: {},
    memoryCards: [],
    adapters: [],
    changes: [],
  },
};
