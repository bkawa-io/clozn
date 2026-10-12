import { useEffect, useMemo, useState } from "react";
import { ContractError } from "../../data/contracts";
import { studioApi } from "../../data/client";
import type { LocalModelRecord, RuntimeHealth } from "../../data/runtime";
import "./runtime.css";

type LoadValue<T> =
  | { state: "loading" }
  | { state: "ready"; value: T }
  | { state: "unavailable"; message: string }
  | { state: "malformed"; message: string };

interface RuntimeState {
  health: LoadValue<RuntimeHealth>;
  inventory: LoadValue<readonly LocalModelRecord[]>;
}

const initialState: RuntimeState = {
  health: { state: "loading" },
  inventory: { state: "loading" },
};

function failure(error: unknown): LoadValue<never> {
  if (error instanceof ContractError) return { state: "malformed", message: error.message };
  return { state: "unavailable", message: error instanceof Error ? error.message : "Request failed." };
}

function basename(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).at(-1) ?? path;
}

function shortHash(value: string | null | undefined): string {
  return value ? `${value.slice(0, 12)}…` : "Not reported";
}

function formatBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) {
    value /= 1000;
    unit += 1;
  }
  return `${unit === 0 || value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

function stateLabel(state: RuntimeState): string {
  if (state.health.state === "loading" || state.inventory.state === "loading") return "Reading";
  if (state.health.state === "ready" && state.inventory.state === "ready") return "Reported";
  if (state.health.state === "ready" || state.inventory.state === "ready") return "Partial";
  return "Unavailable";
}

function SourceState({ value, label }: { value: LoadValue<unknown>; label: string }) {
  if (value.state === "loading") return <p className="runtime-state">Reading {label}…</p>;
  if (value.state === "malformed") {
    return <p className="runtime-state is-error" role="alert">Malformed {label} contract.<span>{value.message}</span></p>;
  }
  if (value.state === "unavailable") {
    return <p className="runtime-state is-error" role="alert">{label} unavailable.<span>{value.message}</span></p>;
  }
  return null;
}

function WorkerHealth({ health }: { health: Extract<RuntimeHealth, { kind: "worker" }> }) {
  const flags = Object.entries(health.capabilities).sort(([left], [right]) => left.localeCompare(right));
  return (
    <>
      <dl className="runtime-facts">
        <div><dt>Model</dt><dd>{basename(health.model)}</dd></div>
        <div><dt>Architecture</dt><dd>{health.architecture}</dd></div>
        <div><dt>Mode</dt><dd>{health.mode}</dd></div>
        <div><dt>Device</dt><dd>{health.device ?? health.backend ?? "Not reported"}</dd></div>
        <div><dt>Context</dt><dd>{health.contextTokens?.toLocaleString() ?? "Not reported"}</dd></div>
        <div><dt>GPU layers</dt><dd>{health.gpuLayers?.toLocaleString() ?? "Not reported"}</dd></div>
        <div><dt>Protocol</dt><dd>{health.protocolVersion ?? "Not reported"}</dd></div>
        <div><dt>Model SHA256</dt><dd>{shortHash(health.modelSha256)}</dd></div>
      </dl>
      <section className="runtime-subsection" aria-labelledby="runtime-capabilities-title">
        <div className="runtime-subsection-heading">
          <h3 id="runtime-capabilities-title">Capability flags</h3>
          <span>{flags.length} reported</span>
        </div>
        {flags.length ? (
          <ul className="runtime-capabilities">
            {flags.map(([name, available]) => (
              <li key={name} data-available={available ? "true" : "false"}>
                <i aria-hidden="true" />
                <code>{name}</code>
                <span>{available ? "Available" : "Unavailable"}</span>
              </li>
            ))}
          </ul>
        ) : <p className="runtime-empty">No boolean capability flags were reported.</p>}
      </section>
    </>
  );
}

function ManagedHealth({ health }: { health: Extract<RuntimeHealth, { kind: "managed" }> }) {
  return (
    <>
      <dl className="runtime-facts">
        <div><dt>Default model</dt><dd>{health.defaultModelId}</dd></div>
        <div><dt>Resident</dt><dd>{health.residentCount} of {health.configuredCount}</dd></div>
        <div><dt>Resident limit</dt><dd>{health.maxLoadedModels}</dd></div>
        <div><dt>Preload set</dt><dd>{health.preloadModelIds.length || "None"}</dd></div>
      </dl>
      <section className="runtime-subsection" aria-labelledby="runtime-workers-title">
        <div className="runtime-subsection-heading">
          <h3 id="runtime-workers-title">Configured models</h3>
          <span>{health.models.length} reported</span>
        </div>
        <ul className="runtime-worker-list">
          {health.models.map((model) => (
            <li key={model.modelId}>
              <div><strong>{model.modelId}</strong>{model.default && <span>Default</span>}</div>
              <code>{model.runtimeKeySha256.slice(0, 12)}…</code>
              <span className={`runtime-worker-state state-${model.state}`}>{model.state}</span>
              {model.failureCode && <small>{model.failureCode}</small>}
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}

function Inventory({ models, activePath }: { models: readonly LocalModelRecord[]; activePath?: string }) {
  if (!models.length) return <p className="runtime-empty">No local GGUF files were reported.</p>;
  return (
    <ul className="runtime-inventory-list">
      {models.map((model) => (
        <li key={model.path}>
          <div>
            <strong>{model.filename}</strong>
            {activePath === model.path && <span>Serving</span>}
          </div>
          <dl>
            <div><dt>Quant</dt><dd>{model.quant ?? "Not reported"}</dd></div>
            <div><dt>Size</dt><dd>{formatBytes(model.sizeBytes)}</dd></div>
            <div><dt>SHA256</dt><dd>{shortHash(model.sha256)}</dd></div>
          </dl>
          <code>{model.path}</code>
        </li>
      ))}
    </ul>
  );
}

export function RuntimeSurface() {
  const [runtime, setRuntime] = useState<RuntimeState>(initialState);
  useEffect(() => {
    const controller = new AbortController();
    void Promise.allSettled([
      studioApi.runtimeHealth(controller.signal),
      studioApi.localModels(controller.signal),
    ]).then(([health, inventory]) => {
      if (controller.signal.aborted) return;
      setRuntime({
        health: health.status === "fulfilled" ? { state: "ready", value: health.value } : failure(health.reason),
        inventory: inventory.status === "fulfilled" ? { state: "ready", value: inventory.value } : failure(inventory.reason),
      });
    });
    return () => controller.abort();
  }, []);

  const label = stateLabel(runtime);
  const activePath = useMemo(() => (
    runtime.health.state === "ready" && runtime.health.value.kind === "worker"
      ? runtime.health.value.model
      : undefined
  ), [runtime.health]);

  return (
    <div className="runtime-surface">
      <header className="runtime-heading">
        <div>
          <span className="eyebrow">INSTALLATION</span>
          <h1>Runtime</h1>
          <p>Read-only engine state and local model inventory.</p>
        </div>
        <span className={`runtime-load-state state-${label.toLowerCase()}`}>{label}</span>
      </header>

      <div className="runtime-grid">
        <section className="runtime-panel" aria-labelledby="runtime-engine-title">
          <header>
            <div><span>ENGINE</span><h2 id="runtime-engine-title">Serving runtime</h2></div>
            <code>/engine/health</code>
          </header>
          <SourceState value={runtime.health} label="engine health" />
          {runtime.health.state === "ready" && (
            runtime.health.value.kind === "worker"
              ? <WorkerHealth health={runtime.health.value} />
              : <ManagedHealth health={runtime.health.value} />
          )}
        </section>

        <section className="runtime-panel" aria-labelledby="runtime-inventory-title">
          <header>
            <div><span>FILES</span><h2 id="runtime-inventory-title">Local GGUF inventory</h2></div>
            <code>/models/local</code>
          </header>
          <SourceState value={runtime.inventory} label="model inventory" />
          {runtime.inventory.state === "ready" && <Inventory models={runtime.inventory.value} activePath={activePath} />}
        </section>
      </div>

      <aside className="runtime-boundary">
        <strong>Read-only boundary</strong>
        <p>This surface reports current server facts. It does not load, switch, download, or remove models.</p>
      </aside>
    </div>
  );
}
