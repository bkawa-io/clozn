import { useEffect, useState } from "react";
import {
  loadModelWorkspace,
  type EngineModel,
  type ModelWorkspaceData,
} from "./api";

interface ModelProps {
  inspectorOpen: boolean;
}

type ModelView = "capabilities" | "stack" | "inventory";

const capabilityLabels: Record<string, { label: string; object: string }> = {
  streaming: { label: "Token streaming", object: "Incremental generation output" },
  sampling: { label: "Sampling", object: "Temperature, top-p, top-k, repeat penalty" },
  steering: { label: "Activation steering", object: "Tone and concept intervention routes" },
  state_stream: { label: "State stream", object: "Runtime state telemetry" },
  jlens: { label: "J-lens", object: "Layer concept readout" },
  sae: { label: "Sparse features", object: "SAE feature readout" },
  infill: { label: "Infill", object: "Masked-span generation" },
  revise: { label: "Revision", object: "Revision pass" },
  score_arms: { label: "Arm scoring", object: "Intervention arm score readout" },
};

const views: Array<{ id: ModelView; label: string }> = [
  { id: "stack", label: "CONFIGURATION STACK" },
  { id: "capabilities", label: "CAPABILITIES" },
  { id: "inventory", label: "MODEL INVENTORY" },
];

function shortHash(value?: string) {
  return value ? `${value.slice(0, 12)}…` : "—";
}

function sizeText(value?: number) {
  if (value == null) return "—";
  return `${(value / 1e9).toFixed(2)} GB`;
}

function modelValue(engine: EngineModel | undefined, key: keyof EngineModel) {
  const value = engine?.[key];
  return value == null || value === "" ? "—" : String(value);
}

export function Model({ inspectorOpen }: ModelProps) {
  const [view, setView] = useState<ModelView>("stack");
  const [data, setData] = useState<ModelWorkspaceData>({
    axes: [],
    errors: {},
  });
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [selectedCapability, setSelectedCapability] = useState("");
  const [selectedModelPath, setSelectedModelPath] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    void loadModelWorkspace(controller.signal).then((next) => {
      if (controller.signal.aborted) return;
      setData(next);
      setStatus(next.engine ? "ready" : "error");
      setSelectedCapability(Object.keys(next.engine?.capabilities ?? {})[0] ?? "");
      setSelectedModelPath(next.engine?.model ?? next.localModels?.[0]?.path ?? "");
    }).catch(() => {
      if (!controller.signal.aborted) setStatus("error");
    });
    return () => controller.abort();
  }, []);

  const engine = data.engine;
  const layers = Array.from({ length: engine?.layers ?? 0 }, (_, layer) => layer);
  const capabilityRows = Object.entries(engine?.capabilities ?? {}).sort(([a], [b]) => a.localeCompare(b));
  const activeAxes = data.axes.filter((axis) => Math.abs(axis.value) > .0001);
  const calibratedAxes = data.axes.filter((axis) => axis.calibrated);
  const selectedInventory = data.localModels?.find((model) => model.path === selectedModelPath);
  const selectedCapabilityState = engine?.capabilities[selectedCapability];
  const selectedCapabilityInfo = capabilityLabels[selectedCapability] ?? {
    label: selectedCapability || "—",
    object: selectedCapability ? "Engine capability flag" : "—",
  };
  const navigationCounts: Record<ModelView, string | number> = {
    capabilities: capabilityRows.filter(([, available]) => available).length,
    stack: 1 + activeAxes.length,
    inventory: data.localModels?.length ?? "—",
  };

  return (
    <>
      <aside className="instrument model-map" aria-labelledby="model-map-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">MODEL SCOPE</span>
            <h2 id="model-map-title">Model</h2>
          </div>
          <strong>{status.toUpperCase()}</strong>
        </header>
        <nav className="model-sections" aria-label="Model sections">
          {views.map((item) => (
            <button
              type="button"
              className={view === item.id ? "is-active" : ""}
              aria-pressed={view === item.id}
              onClick={() => setView(item.id)}
              key={item.id}
            >
              <span>{item.label}</span>
              <b>{navigationCounts[item.id]}</b>
            </button>
          ))}
        </nav>
        <section className="model-serving-summary">
          <header><span>SERVING</span><b>{engine ? "UP" : "—"}</b></header>
          <strong>{engine?.modelName ?? "ENGINE UNAVAILABLE"}</strong>
          <dl>
            <div><dt>Architecture</dt><dd>{modelValue(engine, "architecture")}</dd></div>
            <div><dt>Quant</dt><dd>{modelValue(engine, "quant")}</dd></div>
            <div><dt>Device</dt><dd>{modelValue(engine, "device")}</dd></div>
            <div><dt>Layers</dt><dd>{modelValue(engine, "layers")}</dd></div>
            <div><dt>Context</dt><dd>{modelValue(engine, "context")}</dd></div>
            <div><dt>Protocol</dt><dd>{modelValue(engine, "protocolVersion")}</dd></div>
          </dl>
        </section>
      </aside>

      <section className="instrument model-console" aria-labelledby="model-console-title">
        {view === "capabilities" ? (
          <>
            <header className="instrument-head model-console-head">
              <div>
                <span className="eyebrow">ENGINE CONTRACT</span>
                <h1 id="model-console-title">Capabilities</h1>
              </div>
              <div className="model-head-stats">
                <span><b>AVAILABLE</b>{capabilityRows.filter(([, available]) => available).length}</span>
                <span><b>UNAVAILABLE</b>{capabilityRows.filter(([, available]) => !available).length}</span>
              </div>
            </header>
            <div className="model-capability-grid">
              {capabilityRows.map(([name, available]) => {
                const info = capabilityLabels[name] ?? { label: name, object: "Engine capability flag" };
                return (
                  <button
                    type="button"
                    className={`${selectedCapability === name ? "is-selected" : ""} ${available ? "is-available" : "is-unavailable"}`}
                    aria-pressed={selectedCapability === name}
                    onClick={() => setSelectedCapability(name)}
                    key={name}
                  >
                    <i />
                    <span>{info.label}</span>
                    <strong>{available ? "AVAILABLE" : "UNAVAILABLE"}</strong>
                    <small>{info.object}</small>
                  </button>
                );
              })}
              {!capabilityRows.length && <div className="model-unavailable">0 CAPABILITY FLAGS</div>}
            </div>
          </>
        ) : view === "stack" ? (
          <>
            <header className="instrument-head model-console-head">
              <div>
                <span className="eyebrow">ACTIVE CONFIGURATION</span>
                <h1 id="model-console-title">Configuration stack</h1>
              </div>
              <a className="model-behavior-link" href="#/behavior">OPEN BEHAVIOR</a>
            </header>
            <div className="model-stack-stage">
              <section className="model-architecture-compact" aria-label="Model architecture">
                <header>
                  <span>ARCHITECTURE</span>
                  <strong>{engine?.architecture?.toUpperCase() ?? "—"}</strong>
                  <dl>
                    <div><dt>LAYERS</dt><dd>{engine?.layers ?? "—"}</dd></div>
                    <div><dt>EMBEDDING</dt><dd>{engine?.embedding ?? "—"}</dd></div>
                    <div><dt>CONTEXT</dt><dd>{engine?.context?.toLocaleString() ?? "—"}</dd></div>
                    <div><dt>VOCABULARY</dt><dd>{engine?.vocabulary?.toLocaleString() ?? "—"}</dd></div>
                  </dl>
                </header>
                <div className="model-architecture-layers" aria-label={`${layers.length} transformer layers`}>
                  <span>INPUT</span>
                  <div style={{ gridTemplateColumns: `repeat(${Math.max(layers.length, 1)}, minmax(3px, 1fr))` }}>
                    {layers.map((layer) => <i title={`L${layer}`} key={layer} />)}
                  </div>
                  <span>OUTPUT</span>
                </div>
              </section>
              <article className="model-stack-row is-base">
                <b>01</b>
                <div><span>BASE MODEL</span><strong>{engine?.modelName ?? "—"}</strong></div>
                <output>{engine?.quant ?? "—"}</output>
              </article>
              <article className="model-stack-row">
                <b>02</b>
                <div><span>LORA / ADAPTER METADATA</span><strong>UNREPORTED</strong></div>
                <output>—</output>
              </article>
              <article className="model-stack-row is-active">
                <b>03</b>
                <div><span>TONE STEERING</span><strong>{activeAxes.length} ACTIVE AXES</strong></div>
                <output>{calibratedAxes.length}/{data.axes.length} CAL</output>
              </article>
              <div className="model-axis-strip">
                {activeAxes.map((axis) => (
                  <span key={axis.name}><b>{axis.name}</b>{axis.value >= 0 ? "+" : ""}{axis.value.toFixed(2)}</span>
                ))}
              </div>
              <article className={`model-stack-row ${engine?.capabilities.jlens ? "is-active" : ""}`}>
                <b>04</b>
                <div><span>CONCEPT READOUT</span><strong>{engine?.capabilities.jlens ? "J-LENS AVAILABLE" : "J-LENS UNAVAILABLE"}</strong></div>
                <output>{engine?.capabilities.jlens ? "ON" : "OFF"}</output>
              </article>
            </div>
          </>
        ) : (
          <>
            <header className="instrument-head model-console-head">
              <div>
                <span className="eyebrow">READ-ONLY INVENTORY</span>
                <h1 id="model-console-title">Model inventory</h1>
              </div>
              <div className="model-head-stats">
                <span><b>REPORTED</b>{data.localModels?.length ?? "—"}</span>
                <span><b>SERVING</b>{engine ? 1 : 0}</span>
              </div>
            </header>
            <div className="model-inventory-stage">
              <button
                type="button"
                className={`model-inventory-row is-serving ${selectedModelPath === engine?.model ? "is-selected" : ""}`}
                aria-pressed={selectedModelPath === engine?.model}
                onClick={() => setSelectedModelPath(engine?.model ?? "")}
              >
                <span><b>SERVING</b><strong>{engine?.modelName ?? "—"}</strong></span>
                <span>{engine?.quant ?? "—"}</span>
                <span>PATH</span>
                <span>{sizeText(data.localModels?.find((model) => model.path === engine?.model)?.sizeBytes)}</span>
              </button>
              {(data.localModels ?? []).filter((model) => model.path !== engine?.model).map((model) => (
                <button
                  type="button"
                  className={selectedModelPath === model.path ? "model-inventory-row is-selected" : "model-inventory-row"}
                  aria-pressed={selectedModelPath === model.path}
                  onClick={() => setSelectedModelPath(model.path)}
                  key={model.path}
                >
                  <span><b>INSTALLED</b><strong>{model.filename}</strong></span>
                  <span>{model.quant ?? "—"}</span>
                  <span>FILE</span>
                  <span>{sizeText(model.sizeBytes)}</span>
                </button>
              ))}
              {data.errors.inventory && (
                <div className="model-inventory-error">
                  <strong>INVENTORY ROUTE UNAVAILABLE</strong>
                  <span>{data.errors.inventory}</span>
                </div>
              )}
            </div>
          </>
        )}
      </section>

      {inspectorOpen && (
        <aside className="instrument model-inspector" aria-labelledby="model-inspector-title">
          <header className="instrument-head compact">
            <div>
              <span className="eyebrow">SELECTION</span>
              <h2 id="model-inspector-title">
                {view === "capabilities"
                  ? "Capability"
                  : view === "stack"
                    ? "Configuration"
                    : "Model file"}
              </h2>
            </div>
            <strong>{view.toUpperCase()}</strong>
          </header>
          {view === "capabilities" ? (
            <>
              <section className={`model-capability-inspector ${selectedCapabilityState ? "is-available" : ""}`}>
                <span>{selectedCapability}</span>
                <strong>{selectedCapabilityInfo.label}</strong>
                <b>{selectedCapabilityState ? "AVAILABLE" : "UNAVAILABLE"}</b>
              </section>
              <section className="model-inspector-facts">
                <dl>
                  <div><dt>Contract key</dt><dd>{selectedCapability || "—"}</dd></div>
                  <div><dt>Object</dt><dd>{selectedCapabilityInfo.object}</dd></div>
                  <div><dt>Source</dt><dd>/engine/health</dd></div>
                </dl>
              </section>
            </>
          ) : view === "stack" ? (
            <section className="model-inspector-facts">
              <dl>
                <div><dt>Base model</dt><dd>{engine?.modelName ?? "—"}</dd></div>
                <div><dt>Adapter identity</dt><dd>UNREPORTED</dd></div>
                <div><dt>Active tone axes</dt><dd>{activeAxes.length}</dd></div>
                <div><dt>Calibrated axes</dt><dd>{calibratedAxes.length} / {data.axes.length}</dd></div>
                <div><dt>Profile</dt><dd>{data.activeProfile || "—"}</dd></div>
              </dl>
              <a className="model-inspector-action" href="#/behavior">EDIT INTERVENTIONS</a>
            </section>
          ) : (
            <>
              <section className="model-file-selection">
                <span>{selectedModelPath === engine?.model ? "SERVING" : "INSTALLED"}</span>
                <strong>{selectedInventory?.filename ?? engine?.modelName ?? "—"}</strong>
              </section>
              <section className="model-inspector-facts">
                <dl>
                  <div><dt>Quant</dt><dd>{selectedInventory?.quant ?? engine?.quant ?? "—"}</dd></div>
                  <div><dt>Size</dt><dd>{sizeText(selectedInventory?.sizeBytes)}</dd></div>
                  <div><dt>SHA256</dt><dd>{shortHash(selectedInventory?.sha256 ?? engine?.sha256)}</dd></div>
                  <div><dt>Switch route</dt><dd>UNAVAILABLE</dd></div>
                </dl>
                <code>clozn serve &lt;model&gt;</code>
              </section>
            </>
          )}
        </aside>
      )}

      <section className="instrument model-readout" aria-labelledby="model-readout-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">IDENTITY</span>
            <h2 id="model-readout-title">Serving model</h2>
          </div>
          <strong>{engine?.architecture?.toUpperCase() ?? "—"}</strong>
        </header>
        <div className="model-identity-row">
          <span><b>MODEL</b>{engine?.modelName ?? "—"}</span>
          <span><b>SHA256</b>{shortHash(engine?.sha256)}</span>
          <span><b>MODE</b>{engine?.mode ?? "—"}</span>
          <span><b>DEVICE</b>{engine?.device ?? "—"}{engine?.gpuLayers == null ? "" : ` · ${engine.gpuLayers} OFFLOAD`}</span>
          <span><b>PROTOCOL</b>{engine?.protocolVersion ?? "—"}</span>
        </div>
      </section>
    </>
  );
}
