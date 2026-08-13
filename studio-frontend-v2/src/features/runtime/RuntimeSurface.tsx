import { useEffect, useState } from "react";
import {
  capabilitiesFor,
  composeRuntimeState,
  isOccupyingSlot,
  lifecycleLabel,
  lifecycleTone,
  type RuntimeCapability,
  type RuntimeModelRecord,
  type RuntimeQueueSnapshot,
  type RuntimeSnapshot,
  type RuntimeSurfacePhase,
} from "./model";
import "./runtime.css";

export type {
  CapabilityState,
  RuntimeCapability,
  RuntimeIdentity,
  RuntimeLifecycle,
  RuntimeModelRecord,
  RuntimeSnapshot,
  RuntimeSurfacePhase,
  RuntimeTelemetry,
  RuntimeTelemetryMetric,
} from "./model";

export interface RuntimeSurfaceProps {
  snapshot?: RuntimeSnapshot;
  phase?: RuntimeSurfacePhase;
  error?: string;
  initialSelectedModelId?: string;
  onRefresh?: () => void;
}

function StatusChip({ label, tone = "neutral", detail }: { label: string; tone?: "ready" | "transition" | "failed" | "neutral" | "selected"; detail?: string }) {
  return <span className={`runtime-status is-${tone}`} aria-label={detail ? `${label}: ${detail}` : label}>{label}</span>;
}

function Detail({ label, children }: { label: string; children: React.ReactNode }) {
  return <div className="runtime-detail"><dt>{label}</dt><dd>{children}</dd></div>;
}

function PresentationState({ phase, error }: { phase: RuntimeSurfacePhase; error?: string }) {
  if (phase === "loading") return <p className="runtime-presentation-state is-loading" role="status">Reading current runtime state…</p>;
  if (phase === "error") return <p className="runtime-presentation-state is-error" role="alert">Runtime data unavailable · {error ?? "The runtime did not return a usable response."}</p>;
  if (phase === "unavailable") return <p className="runtime-presentation-state">Runtime inventory is not available from this service.</p>;
  if (phase === "stale") return <p className="runtime-presentation-state" role="status">Showing the most recent runtime snapshot; refresh to confirm current state.</p>;
  return null;
}

function ControlStrip({ snapshot }: { snapshot?: RuntimeSnapshot }) {
  const state = composeRuntimeState(snapshot);
  const models = snapshot?.models;
  const defaultFailure = models?.find((model) => model.isDefault && model.state === "failed");
  const controlTone = state === "READY" ? "ready" : state === "DEGRADED" ? "transition" : state === "UNREACHABLE" ? "failed" : "neutral";
  const readiness = snapshot?.readiness === "ready" ? "READY" : snapshot?.readiness === "not_ready" ? "NOT READY" : "NOT REPORTED";
  const protocol = snapshot?.protocol === "compatible" ? "COMPATIBLE" : snapshot?.protocol === "incompatible" ? "INCOMPATIBLE" : snapshot?.protocolVersion ? `REPORTED ${snapshot.protocolVersion}` : "NOT REPORTED";
  const resident = snapshot?.residentCount;
  const cap = snapshot?.maxLoadedModels;

  return (
    <section className="runtime-control instrument" aria-labelledby="runtime-control-title">
      <header className="runtime-heading">
        <div><span className="eyebrow">CURRENT OPERATIONAL STATE</span><h1 id="runtime-control-title">Models / Runtime</h1></div>
        <StatusChip label={state} tone={controlTone} detail="Composed Studio runtime state" />
      </header>
      <dl className="runtime-control-grid">
        <Detail label="Service"><StatusChip label={snapshot?.service === "live" ? "LIVE" : snapshot?.service === "unreachable" ? "UNREACHABLE" : "NOT REPORTED"} tone={snapshot?.service === "live" ? "ready" : snapshot?.service === "unreachable" ? "failed" : "neutral"} detail="Gateway service liveness" /></Detail>
        <Detail label="Inference"><StatusChip label={readiness} tone={snapshot?.readiness === "ready" ? "ready" : "neutral"} detail={snapshot?.readinessDetail ?? "Inference readiness"} />{snapshot?.readinessDetail && <small>{snapshot.readinessDetail}</small>}</Detail>
        <Detail label="Residency">{resident !== undefined && cap !== undefined ? <><strong>{resident} / {cap} slots</strong><small>{snapshot?.configuredCount ?? "Not reported"} configured</small></> : <strong>Not reported</strong>}</Detail>
        <Detail label="Queue"><QueueFact queue={snapshot?.queue} legacyWaiting={snapshot?.queueCount} /></Detail>
        <Detail label="Protocol"><StatusChip label={protocol} tone={snapshot?.protocol === "incompatible" ? "failed" : "neutral"} detail={snapshot?.protocol ? "Runtime protocol compatibility" : snapshot?.protocolVersion ? "Reported gateway-to-worker wire version; compatibility not assessed" : "Runtime protocol not reported"} /></Detail>
      </dl>
      {defaultFailure && <p className="runtime-composed-note"><strong>Default model failed.</strong> A secondary ready worker may still serve requests; this is not a claim that all requests are healthy.</p>}
    </section>
  );
}

function QueueFact({ queue, legacyWaiting }: { queue?: RuntimeQueueSnapshot; legacyWaiting?: number | null }) {
  if (queue) {
    const hasValue = queue.active !== undefined || queue.waiting !== undefined || queue.capacity !== undefined;
    if (!hasValue) return <strong>Not reported</strong>;
    return <>
      <strong>{queue.waiting === undefined ? "Waiting not reported" : `${queue.waiting} waiting`}</strong>
      <small>{queue.active === undefined ? "Active not reported" : `${queue.active} active`}{queue.capacity === undefined ? "" : ` · ${queue.capacity} capacity`}</small>
    </>;
  }
  return legacyWaiting !== undefined && legacyWaiting !== null ? <><strong>{legacyWaiting} waiting</strong><small>Current snapshot</small></> : <strong>Not reported</strong>;
}

function ResidencyRack({ snapshot }: { snapshot?: RuntimeSnapshot }) {
  const capacity = snapshot?.maxLoadedModels;
  const occupants = snapshot?.models?.filter((model) => isOccupyingSlot(model.state)) ?? [];
  if (capacity === undefined) return <section className="runtime-rack instrument"><header><span className="eyebrow">BOUNDED WORKERS</span><h2>Residency Rack</h2></header><p className="runtime-absence">Residency inventory not reported. CLOZN is not guessing worker capacity.</p></section>;
  const slots = Array.from({ length: capacity }, (_, index) => occupants[index]);
  return (
    <section className="runtime-rack instrument" aria-labelledby="residency-rack-title">
      <header><div><span className="eyebrow">BOUNDED WORKERS</span><h2 id="residency-rack-title">Residency Rack</h2></div><span className="mono">{occupants.length} occupied / {capacity} slots</span></header>
      <ol className="residency-slots">
        {slots.map((model, index) => <li key={model?.modelId ?? `empty-${index}`} className={`residency-slot${model ? ` is-${lifecycleTone(model.state)}` : " is-empty"}`}>
          <span className="slot-ordinal">WORKER {index + 1}</span>
          {model ? <><strong>{model.modelId}</strong><StatusChip label={lifecycleLabel(model.state)} tone={lifecycleTone(model.state)} detail={`Worker ${index + 1} lifecycle`} />{model.isDefault && <small>Default model</small>}{model.workerGeneration != null && <small>Generation {model.workerGeneration}</small>}{model.workerGenerationId && <small title={model.workerGenerationId}>Process generation reported</small>}</> : <><strong>Empty</strong><span className="runtime-muted">No resident worker</span></>}
        </li>)}
      </ol>
    </section>
  );
}

function ConfiguredModels({ models, selectedId, onSelect }: { models?: readonly RuntimeModelRecord[]; selectedId?: string; onSelect: (modelId: string) => void }) {
  return (
    <section className="configured-models instrument" aria-labelledby="configured-models-title">
      <header><div><span className="eyebrow">CONFIGURATION</span><h2 id="configured-models-title">Configured Models</h2></div><span>{models?.length ?? "No"} reported</span></header>
      {!models ? <p className="runtime-absence">Model configuration unavailable. Service liveness does not establish residency.</p> : models.length === 0 ? <p className="runtime-absence">No models are configured.</p> : (
        <div className="configured-model-table" role="listbox" aria-label="Configured models">
          <div className="configured-model-head" aria-hidden="true"><span>Model</span><span>Lifecycle</span><span>Markers</span><span>Worker</span></div>
          {models.map((model) => <button type="button" role="option" key={model.modelId} aria-selected={selectedId === model.modelId} className={selectedId === model.modelId ? "is-selected" : undefined} onClick={() => onSelect(model.modelId)}>
            <strong title={model.modelId}>{model.modelId}</strong><StatusChip label={lifecycleLabel(model.state)} tone={lifecycleTone(model.state)} detail={`Lifecycle: ${lifecycleLabel(model.state)}`} /><span className="model-markers">{model.isDefault && <em>DEFAULT</em>}{model.preloaded && <em>PRELOAD</em>}{!model.isDefault && !model.preloaded && <span>—</span>}</span><span className="mono">{model.workerGeneration == null ? "—" : `GEN ${model.workerGeneration}`}</span>
          </button>)}
        </div>
      )}
    </section>
  );
}

function CapabilityBay({ capabilities }: { capabilities: readonly RuntimeCapability[] }) {
  return <section className="instrument-bay-section" aria-labelledby="capability-bay-title"><header><span className="eyebrow">LIVE CAPABILITY</span><h3 id="capability-bay-title">Capability Bay</h3></header><p className="runtime-caption">Runtime availability is not exact-model qualification.</p><dl className="capability-list">{capabilities.map((capability) => <div key={capability.label}><dt>{capability.label}</dt><dd><StatusChip label={capability.state.toUpperCase()} tone={capability.state === "available now" ? "ready" : "neutral"} detail={`${capability.label}: ${capability.state}`} />{capability.detail && <small>{capability.detail}</small>}</dd></div>)}</dl></section>;
}

function TelemetryBay({ snapshot }: { snapshot?: RuntimeSnapshot }) {
  const telemetry = snapshot?.telemetry;
  if (!telemetry) return <section className="instrument-bay-section runtime-telemetry" aria-labelledby="telemetry-title"><header><span className="eyebrow">OPTIONAL RESOURCE INSTRUMENTS</span><h3 id="telemetry-title">Resource Telemetry</h3></header><p className="runtime-absence">Resource telemetry not reported by this runtime.</p><p className="runtime-caption">CLOZN is not synthesizing RAM, VRAM, or utilization from model names or platform assumptions.</p></section>;
  if (telemetry.availability !== "available") return <section className="instrument-bay-section runtime-telemetry" aria-labelledby="telemetry-title"><header><span className="eyebrow">OPTIONAL RESOURCE INSTRUMENTS</span><h3 id="telemetry-title">Resource Telemetry</h3></header><StatusChip label={telemetry.availability.toUpperCase()} tone={telemetry.availability === "error" ? "failed" : "neutral"} detail={`Resource telemetry ${telemetry.availability}`} /><p className="runtime-absence">{telemetry.detail ?? "Resource telemetry is not available from this runtime."}</p></section>;
  return <section className="instrument-bay-section runtime-telemetry" aria-labelledby="telemetry-title"><header><span className="eyebrow">OPTIONAL RESOURCE INSTRUMENTS</span><h3 id="telemetry-title">Resource Telemetry</h3></header><dl className="telemetry-instruments">{telemetry.metrics?.map((metric) => <div key={metric.label}><dt>{metric.label}</dt><dd>{metric.value}</dd></div>) ?? <div><dt>Measurements</dt><dd>Available; no metrics reported</dd></div>}</dl><p className="runtime-caption">Provider: {telemetry.provider ?? "Not reported"} · Device: {telemetry.device ?? "Not reported"} · Observed: {telemetry.observedAt ?? "Not reported"}</p></section>;
}

function InstrumentBay({ model, snapshot, removedSelection }: { model?: RuntimeModelRecord; snapshot?: RuntimeSnapshot; removedSelection: boolean }) {
  if (!model) return <aside className="selected-instrument-bay instrument" aria-labelledby="instrument-bay-title"><header><span className="eyebrow">SELECTED MODEL</span><h2 id="instrument-bay-title">Instrument Bay</h2></header><p className="runtime-absence">{removedSelection ? "Selected model is no longer configured." : "Select a configured model to inspect its live runtime facts."}</p><TelemetryBay snapshot={snapshot} /></aside>;
  const identity = snapshot?.identityByModelId?.[model.modelId];
  const capabilities = capabilitiesFor(model, snapshot?.capabilitiesByModelId?.[model.modelId]);
  const identityRows: [string, string | number | null | undefined][] = [["Runtime key fingerprint", model.runtimeKeyFingerprint], ["Worker identity", model.workerIdentity], ["Artifact format", identity?.artifactFormat], ["Artifact SHA", identity?.artifactSha], ["Quantization", identity?.quantization], ["Backend / device", identity?.backendDevice], ["Context size", identity?.contextSize], ["Engine build", identity?.engineBuild], ["Template fingerprint", identity?.templateFingerprint], ["Adapter identity", identity?.adapterIdentity]];
  return <aside className="selected-instrument-bay instrument" aria-labelledby="instrument-bay-title"><header><div><span className="eyebrow">SELECTED MODEL</span><h2 id="instrument-bay-title">Instrument Bay</h2><p>{model.modelId}</p></div><StatusChip label={lifecycleLabel(model.state)} tone={lifecycleTone(model.state)} detail="Selected model lifecycle" /></header><section className="instrument-bay-section"><h3>Current lifecycle</h3><dl className="instrument-facts"><Detail label="State"><StatusChip label={lifecycleLabel(model.state)} tone={lifecycleTone(model.state)} /></Detail><Detail label="Default">{model.isDefault ? "Yes" : "No"}</Detail><Detail label="Preloaded">{model.preloaded === undefined ? "Not reported" : model.preloaded ? "Yes" : "No"}</Detail><Detail label="Worker generation">{model.workerGeneration ?? "Not reported"}</Detail>{model.workerGenerationId && <Detail label="Process generation"><code>{model.workerGenerationId}</code></Detail>}{model.failureCode && <Detail label="Failure code"><code>{model.failureCode}</code></Detail>}</dl></section><section className="instrument-bay-section"><header><span className="eyebrow">PROVENANCE FACTS</span><h3>Runtime identity</h3></header><dl className="instrument-facts">{identityRows.map(([label, value]) => <Detail key={label} label={label}>{value ?? "Not reported by current runtime contract"}</Detail>)}</dl></section><CapabilityBay capabilities={capabilities} /><TelemetryBay snapshot={snapshot} /></aside>;
}

export function RuntimeSurface({ snapshot, phase = "ready", error, initialSelectedModelId, onRefresh }: RuntimeSurfaceProps) {
  const [selectedId, setSelectedId] = useState<string | undefined>(() => initialSelectedModelId ?? snapshot?.models?.[0]?.modelId);
  const [removedSelection, setRemovedSelection] = useState(false);
  const models = snapshot?.models;
  const selected = models?.find((model) => model.modelId === selectedId);
  useEffect(() => {
    if (!models?.length) return;
    if (!selectedId && !removedSelection) { setSelectedId(initialSelectedModelId ?? models[0].modelId); return; }
    if (!selectedId) return;
    if (!models.some((model) => model.modelId === selectedId)) { setSelectedId(undefined); setRemovedSelection(true); }
  }, [initialSelectedModelId, models, removedSelection, selectedId]);
  const selectModel = (modelId: string) => { setRemovedSelection(false); setSelectedId(modelId); };

  return <main className="runtime-surface"><div className="runtime-surface-topline"><span className="eyebrow">GREENFIELD STUDIO</span>{onRefresh && <button type="button" className="runtime-refresh" onClick={onRefresh}>Refresh runtime</button>}</div><PresentationState phase={phase} error={error} /><ControlStrip snapshot={snapshot} /><div className="runtime-workbench"><div className="runtime-primary-stack"><ResidencyRack snapshot={snapshot} /><ConfiguredModels models={models} selectedId={selectedId} onSelect={selectModel} /></div><InstrumentBay model={selected} snapshot={snapshot} removedSelection={removedSelection} /></div></main>;
}
