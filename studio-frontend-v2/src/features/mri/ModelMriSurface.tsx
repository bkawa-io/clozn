import { Fragment, useEffect, useMemo, useState } from "react";
import {
  capabilityLabel,
  channelGuide,
  defaultMriLocus,
  evidenceLabel,
  evidenceTone,
  locusKey,
  observationAt,
  observationsFor,
  sameLocus,
  type MriChannel,
  type MriLocus,
  type MriObservation,
  type MriSpecimen,
  type MriSurfacePhase,
} from "./model";
import "./model-mri.css";

export type {
  MriArtifactMode, MriCapabilityState, MriChannel, MriChannelFamily, MriChannelKind,
  MriEvidence, MriLayer, MriLocus, MriObservation, MriSourceReference, MriSpecimen, MriSurfacePhase, MriToken,
} from "./model";

export interface ModelMriSurfaceProps {
  specimen?: MriSpecimen;
  phase?: MriSurfacePhase;
  error?: string;
  initialChannelId?: string;
  /** A controlled cross-surface selection; use the same run/sequence/token/layer locus in routing state. */
  selection?: MriLocus;
  onSelectionChange?: (locus: MriLocus, observation?: MriObservation) => void;
  onRefresh?: () => void;
}

function Chip({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "measured" | "unsupported" | "neutral" | "failed" | "selected" }) {
  return <span className={`mri-chip is-${tone}`}>{children}</span>;
}

function stateMessage(phase: MriSurfacePhase, error?: string) {
  if (phase === "loading") return <p className="mri-state" role="status">Reading recorded internal evidence…</p>;
  if (phase === "unavailable") return <p className="mri-state">Internal evidence is unavailable for this run.</p>;
  if (phase === "error") return <p className="mri-state is-error" role="alert">Model MRI unavailable · {error ?? "The read endpoint did not return a usable specimen."}</p>;
  if (phase === "stale") return <p className="mri-state" role="status">Showing the most recent specimen; refresh to confirm its availability.</p>;
  return null;
}

function channelTone(channel: MriChannel): "measured" | "neutral" {
  return channel.capability === "available" ? "measured" : "neutral";
}

function tokenName(text: string): string {
  return text === "" ? "∅" : text.replaceAll("\n", "↵").replaceAll(" ", "·");
}

function Cell({ locus, observation, selected, onSelect }: { locus: MriLocus; observation?: MriObservation; selected: boolean; onSelect: (locus: MriLocus, observation?: MriObservation) => void }) {
  const label = `Token ${locus.tokenIndex}, layer ${locus.layerIndex}: ${evidenceLabel(observation?.evidence)}`;
  const symbol = !observation ? "·" : observation.evidence.kind === "measured" ? observation.evidence.finding === "supported" ? "●" : "○" : observation.evidence.kind === "failed" ? "×" : "—";
  return <button type="button" className={`mri-atlas-cell is-${evidenceTone(observation?.evidence)}${selected ? " is-selected" : ""}`} onClick={() => onSelect(locus, observation)} aria-label={label} aria-pressed={selected} title={label}>{symbol}</button>;
}

function Atlas({ specimen, observations, selection, onSelect }: { specimen: MriSpecimen; observations: readonly MriObservation[]; selection?: MriLocus; onSelect: (locus: MriLocus, observation?: MriObservation) => void }) {
  return <section className="mri-atlas instrument" aria-labelledby="mri-atlas-title">
    <header><div><span className="eyebrow">ATLAS · TOKEN × LAYER</span><h2 id="mri-atlas-title">Recorded slice coverage</h2></div><span className="mri-caption">Cells are evidence states, never an activity scale.</span></header>
    <div className="mri-atlas-scroll">
      <div className="mri-atlas-grid" style={{ gridTemplateColumns: `minmax(6rem, auto) repeat(${specimen.tokens.length}, minmax(2.8rem, 1fr))` }}>
        <span className="mri-corner">LAYER \ TOKEN</span>
        {specimen.tokens.map((token) => <button key={token.index} type="button" className={`mri-token-head${selection?.tokenIndex === token.index ? " is-selected" : ""}`} onClick={() => { const locus = { runId: specimen.runId, sequenceId: specimen.sequenceId, tokenIndex: token.index, layerIndex: selection?.layerIndex ?? specimen.layers[0]?.index ?? 0 }; onSelect(locus, observationAt(observations, locus)); }} title={token.label ?? token.text}><span>{tokenName(token.text)}</span><small>T{token.index}</small></button>)}
        {specimen.layers.map((layer) => <Fragment key={`layer-row-${layer.index}`}>
          <span className={`mri-layer-label${selection?.layerIndex === layer.index ? " is-selected" : ""}`}>L{layer.index}{layer.label && <small>{layer.label}</small>}</span>
          {specimen.tokens.map((token) => {
            const locus = { runId: specimen.runId, sequenceId: specimen.sequenceId, tokenIndex: token.index, layerIndex: layer.index };
            return <Cell key={locusKey(locus)} locus={locus} observation={observationAt(observations, locus)} selected={sameLocus(locus, selection)} onSelect={onSelect} />;
          })}
        </Fragment>)}
      </div>
    </div>
    <footer className="mri-legend"><span><i className="is-measured">●</i> Measured</span><span><i className="is-unsupported">○</i> Measured, unsupported</span><span><i className="is-neutral">—</i> Explicit absence</span><span><i className="is-neutral">·</i> Not captured</span></footer>
  </section>;
}

function DepthSlice({ specimen, observations, selection, onSelect }: { specimen: MriSpecimen; observations: readonly MriObservation[]; selection?: MriLocus; onSelect: (locus: MriLocus, observation?: MriObservation) => void }) {
  const token = specimen.tokens.find((item) => item.index === selection?.tokenIndex);
  if (!selection || !token) return null;
  return <section className="mri-depth instrument" aria-labelledby="mri-depth-title"><header><div><span className="eyebrow">DEPTH SLICE · ONE TOKEN</span><h2 id="mri-depth-title">{token.label ?? tokenName(token.text)}</h2></div><span className="mono">TOKEN {token.index}</span></header><ol>{specimen.layers.map((layer) => { const locus = { ...selection, layerIndex: layer.index }; const observation = observationAt(observations, locus); return <li key={layer.index}><button type="button" className={sameLocus(locus, selection) ? "is-selected" : undefined} onClick={() => onSelect(locus, observation)}><span>L{layer.index}{layer.label ? ` · ${layer.label}` : ""}</span><Chip tone={sameLocus(locus, selection) ? "selected" : evidenceTone(observation?.evidence)}>{evidenceLabel(observation?.evidence)}</Chip></button></li>; })}</ol></section>;
}

function LocusInspector({ specimen, channel, observation, selection }: { specimen: MriSpecimen; channel?: MriChannel; observation?: MriObservation; selection?: MriLocus }) {
  if (!selection) return <aside className="mri-inspector instrument"><span className="eyebrow">LOCUS INSPECTOR</span><h2>Select a coordinate</h2><p>No stable token × layer coordinate is available from this specimen.</p></aside>;
  const token = specimen.tokens.find((item) => item.index === selection.tokenIndex);
  const layer = specimen.layers.find((item) => item.index === selection.layerIndex);
  const absence = !observation ? "This coordinate was not captured for the selected channel. CLOZN does not infer an empty or zero reading." : undefined;
  const evidenceDetail = observation?.evidence.kind === "measured"
    ? observation.evidence.detail
    : observation?.evidence?.reason;
  return <aside className="mri-inspector instrument" aria-labelledby="mri-inspector-title">
    <header><div><span className="eyebrow">LOCUS INSPECTOR</span><h2 id="mri-inspector-title">Selected coordinate</h2></div>{observation && <Chip tone={evidenceTone(observation.evidence)}>{evidenceLabel(observation.evidence)}</Chip>}</header>
    <dl className="mri-facts"><div><dt>Run</dt><dd className="mono">{selection.runId}</dd></div><div><dt>Sequence</dt><dd className="mono">{selection.sequenceId}</dd></div><div><dt>Token</dt><dd>{token?.label ?? tokenName(token?.text ?? "")}&nbsp; <span className="mono">T{selection.tokenIndex}</span></dd></div><div><dt>Layer</dt><dd>L{selection.layerIndex}{layer?.label ? ` · ${layer.label}` : ""}</dd></div></dl>
    <section className="mri-inspector-section"><span className="eyebrow">EVIDENCE</span><h3>{observation ? evidenceLabel(observation.evidence) : "Not captured"}</h3><p>{absence ?? evidenceDetail ?? "The endpoint did not retain a human-readable annotation for this measurement."}</p>{observation?.findings && observation.findings.length > 0 && <ul className="mri-findings">{observation.findings.map((finding, index) => <li key={`${finding}-${index}`}>{finding}</li>)}</ul>}{observation?.sourceTokens && observation.sourceTokens.length > 0 && <div className="mri-source-tokens"><span className="eyebrow">RECORDED SOURCE POSITIONS</span>{observation.sourceTokens.map((source) => <span key={source.tokenIndex}>T{source.tokenIndex}{source.label ? ` · ${source.label}` : ""}</span>)}</div>}</section>
    <section className="mri-inspector-section"><span className="eyebrow">INSTRUMENT QUALIFICATION</span><h3>{channel?.label ?? "No channel selected"}</h3>{channel ? <><Chip tone={channelTone(channel)}>{capabilityLabel(channel.capability)}</Chip><p>{channel.reason ?? channelGuide(channel.kind)}</p><dl className="mri-channel-facts"><div><dt>Family</dt><dd>{channel.family}</dd></div><div><dt>Provenance</dt><dd>{channel.artifactMode ?? "Not reported"}</dd></div><div><dt>Method</dt><dd>{channel.method ?? "Not reported"}</dd></div>{channel.artifactIdentity && <div><dt>Artifact</dt><dd className="mono">{channel.artifactIdentity}</dd></div>}</dl></> : <p>Choose an instrument to inspect its qualification.</p>}</section>
  </aside>;
}

export function ModelMriSurface({ specimen, phase = "ready", error, initialChannelId, selection: controlledSelection, onSelectionChange, onRefresh }: ModelMriSurfaceProps) {
  const [channelId, setChannelId] = useState<string | undefined>(() => initialChannelId ?? specimen?.channels[0]?.id);
  const [localSelection, setLocalSelection] = useState<MriLocus | undefined>(() => controlledSelection ?? defaultMriLocus(specimen));
  const selection = controlledSelection ?? localSelection;
  useEffect(() => { if (!specimen?.channels.some((channel) => channel.id === channelId)) setChannelId(specimen?.channels[0]?.id); }, [channelId, specimen]);
  useEffect(() => { if (!controlledSelection && specimen && (!selection || selection.runId !== specimen.runId || selection.sequenceId !== specimen.sequenceId)) setLocalSelection(defaultMriLocus(specimen)); }, [controlledSelection, selection, specimen]);
  const channel = specimen?.channels.find((item) => item.id === channelId);
  const observations = useMemo(() => observationsFor(channelId, specimen), [channelId, specimen]);
  const selectedObservation = selection ? observationAt(observations, selection) : undefined;
  const choose = (locus: MriLocus, observation?: MriObservation) => { if (!controlledSelection) setLocalSelection(locus); onSelectionChange?.(locus, observation); };

  if (!specimen) return <main className="model-mri-surface"><div className="mri-topline"><span className="eyebrow">MODEL MRI</span>{onRefresh && <button type="button" onClick={onRefresh}>Refresh evidence</button>}</div>{stateMessage(phase, error) ?? <p className="mri-state">No internal specimen has been supplied for this run.</p>}</main>;
  return <main className="model-mri-surface">
    <div className="mri-topline"><span className="eyebrow">MODEL MRI · RECORDED INTERNAL EVIDENCE</span>{onRefresh && <button type="button" onClick={onRefresh}>Refresh evidence</button>}</div>
    {stateMessage(phase, error)}
    <section className="mri-control instrument" aria-labelledby="mri-title"><header><div><span className="eyebrow">SYNCHRONIZED SLICE VIEWER</span><h1 id="mri-title">What was visible at this point?</h1><p>Move the recorded response beneath one qualified instrument at a time.</p></div><span className="mono">RUN {specimen.runId}</span></header><div className="mri-channel-picker"><label htmlFor="mri-channel">Instrument</label><select id="mri-channel" value={channelId ?? ""} onChange={(event) => setChannelId(event.target.value)}>{specimen.channels.map((item) => <option key={item.id} value={item.id}>{item.label} · {capabilityLabel(item.capability)}</option>)}</select>{channel && <><Chip tone={channelTone(channel)}>{capabilityLabel(channel.capability)}</Chip><span className="mri-caption">{channelGuide(channel.kind)}</span></>}</div></section>
    <div className="mri-workbench"><div className="mri-primary"><Atlas specimen={specimen} observations={observations} selection={selection} onSelect={choose} /><DepthSlice specimen={specimen} observations={observations} selection={selection} onSelect={choose} /></div><LocusInspector specimen={specimen} channel={channel} observation={selectedObservation} selection={selection} /></div>
  </main>;
}
