import type { ObservatoryData } from "./types";

export const DEMO_OBSERVATORY: ObservatoryData = {
  id: "demo_0142",
  label: "Layer scope",
  model: "Qwen2.5-7B-Instruct",
  quant: "Q5_K_M",
  createdAt: "14:32:18",
  duration: "2.8 s",
  mode: "demo",
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
