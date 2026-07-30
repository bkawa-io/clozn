import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import { loadRunInspection } from "../../data/api";
import type { ForkState, ObservatoryData, RuntimeState } from "../../data/types";
import { alignTokens } from "../compare/alignment";
import { ConfidencePlot } from "./ConfidencePlot";
import { LayerScope } from "./LayerScope";
import { TraceScope } from "./TraceScope";
import { VariantDeltaPlot } from "./VariantDeltaPlot";
import { VariantScope } from "./VariantScope";
import type { ScopeSelectionState, ScopeUrlState, ScopeView } from "./urlState";
import { describeVariant, dialDifferences } from "./variant";

export interface ObservatoryProps {
  data: ObservatoryData;
  runtime: RuntimeState;
  inspectorOpen: boolean;
  runStatus: "idle" | "loading" | "error";
  forkState: ForkState;
  onSelectRun: (runId: string) => void;
  onFork: (position: number, token: string) => void;
  initialState?: ScopeUrlState;
  onStateChange?: (state: ScopeSelectionState) => void;
}

function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`;
}

function initialToken(data: ObservatoryData) {
  if (!data.tokens.length) return 0;
  let weakest = 0;
  for (let index = 1; index < data.tokens.length; index += 1) {
    if ((data.tokens[index].confidence ?? 1) < (data.tokens[weakest].confidence ?? 1)) weakest = index;
  }
  return data.mode === "run" ? weakest : Math.min(7, data.tokens.length - 1);
}

function defaultReferenceId(data: ObservatoryData, runtime: RuntimeState) {
  if (data.parentRunId && data.parentRunId !== data.id) return data.parentRunId;
  const samePrompt = runtime.runs.find((run) =>
    run.id !== data.id
    && Boolean(data.prompt)
    && run.prompt.trim() === data.prompt?.trim());
  return samePrompt?.id ?? "";
}

function clampToken(data: ObservatoryData, requested?: number) {
  if (!data.tokens.length) return 0;
  return Math.max(0, Math.min(data.tokens.length - 1, requested ?? initialToken(data)));
}

function clampLayer(runtime: RuntimeState, requested?: number) {
  const value = Math.max(0, requested ?? 0);
  const count = runtime.engine?.layerCount;
  return count == null || count <= 0 ? value : Math.min(count - 1, value);
}

function initialView(data: ObservatoryData, requested?: ScopeView): ScopeView {
  if (requested === "layers" && (data.mode !== "run" || !data.response?.trim())) return "trace";
  if (requested === "variants" && data.mode !== "run") return "trace";
  return requested ?? "trace";
}

function ObservatoryWorkspace({
  data,
  runtime,
  inspectorOpen,
  runStatus,
  forkState,
  onSelectRun,
  onFork,
  initialState,
  onStateChange,
}: ObservatoryProps) {
  const [selectedLayer, setSelectedLayer] = useState(() => clampLayer(runtime, initialState?.layer));
  const [selectedToken, setSelectedToken] = useState(() => clampToken(data, initialState?.token));
  const [view, setView] = useState<ScopeView>(() => initialView(data, initialState?.view));
  const [forkToken, setForkToken] = useState("");
  const [variantReferenceId, setVariantReferenceId] = useState(() => {
    const requested = initialState?.reference;
    if (requested === data.id) return "";
    return requested ?? defaultReferenceId(data, runtime);
  });
  const [variantReference, setVariantReference] = useState<ObservatoryData | null>(null);
  const [variantStatus, setVariantStatus] = useState<"idle" | "loading" | "error">("idle");

  const layersAvailable = data.mode === "run" && Boolean(data.response?.trim());
  const activeView: ScopeView = view === "layers" && !layersAvailable
    ? "trace"
    : view === "variants" && data.mode !== "run"
      ? "trace"
      : view;
  const token = data.tokens[selectedToken] ?? data.tokens[0];
  const candidates = token?.alternatives?.length ? token.alternatives : data.candidates;
  const variantOptions = runtime.runs.filter((run) => run.id !== data.id);
  const variantAlignment = useMemo(
    () => alignTokens(variantReference?.tokens ?? [], data.tokens),
    [variantReference, data.tokens],
  );
  const variantRelation = variantReference ? describeVariant(variantReference, data) : undefined;
  const variantDials = variantReference ? dialDifferences(variantReference, data) : [];
  const variantColumnIndex = variantAlignment.columnByB.get(selectedToken);
  const variantColumn = variantColumnIndex == null ? undefined : variantAlignment.columns[variantColumnIndex];
  const variantReferenceToken = variantColumn?.aIndex == null
    ? undefined
    : variantReference?.tokens[variantColumn.aIndex];
  const variantIdentity = variantColumn?.kind === "same"
    ? "MATCHED REFERENCE"
    : variantColumn?.kind === "b-only"
      ? "INSERTED"
      : variantColumn?.kind === "changed"
        ? "CHANGED"
        : "UNALIGNED";
  const variantConfidenceDelta = token && variantReferenceToken
    ? (token.confidence ?? 0) - (variantReferenceToken.confidence ?? 0)
    : undefined;
  const tapeLimit = 120;
  const tapeStart = data.tokens.length > tapeLimit
    ? Math.max(0, Math.min(data.tokens.length - tapeLimit, selectedToken - Math.floor(tapeLimit / 2)))
    : 0;
  const tapeEnd = Math.min(data.tokens.length, tapeStart + tapeLimit);
  const tapeTokens = data.tokens.slice(tapeStart, tapeEnd);

  useEffect(() => {
    onStateChange?.({
      view: activeView,
      token: selectedToken,
      reference: variantReferenceId || undefined,
      layer: selectedLayer,
    });
  }, [activeView, onStateChange, selectedLayer, selectedToken, variantReferenceId]);

  useEffect(() => {
    if (!variantReferenceId || variantReferenceId === data.id) {
      setVariantReference(null);
      setVariantStatus("idle");
      return;
    }
    const controller = new AbortController();
    setVariantStatus("loading");
    void loadRunInspection(variantReferenceId, controller.signal).then((reference) => {
      if (controller.signal.aborted) return;
      setVariantReference(reference);
      setVariantStatus("idle");
    }).catch(() => {
      if (controller.signal.aborted) return;
      setVariantReference(null);
      setVariantStatus("error");
    });
    return () => controller.abort();
  }, [data.id, variantReferenceId]);

  useEffect(() => {
    setForkToken(candidates.find((candidate) => candidate.token !== token?.text)?.token ?? "");
  }, [selectedToken, data.id]);

  function handleTokenKeys(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next = index;
    if (event.key === "ArrowRight") next = Math.min(data.tokens.length - 1, index + 1);
    else if (event.key === "ArrowLeft") next = Math.max(0, index - 1);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = data.tokens.length - 1;
    else return;
    event.preventDefault();
    setSelectedToken(next);
    requestAnimationFrame(() => {
      document.querySelector<HTMLButtonElement>(`[data-token-index="${next}"]`)?.focus();
    });
  }

  return (
    <>
      <section className="instrument scope-instrument" aria-labelledby="scope-title">
        <header className="instrument-head">
          <div>
            <span className="eyebrow">{data.mode === "run" ? "COMPLETED RUN" : "MODEL INTERIOR"}</span>
            <h1 id="scope-title">
              {activeView === "trace"
                ? "Token sources"
                : activeView === "variants"
                  ? "Variant provenance"
                  : "Layer evidence"}
            </h1>
          </div>
          <div className="head-metrics">
            <span><b>LAYERS</b>{runtime.engine?.layerCount ?? "—"}</span>
            <span><b>TOKENS</b>{data.tokens.length}</span>
            <span><b>DURATION</b>{data.duration}</span>
          </div>
          <label className="run-picker">
            <span>RUN</span>
            <select
              value={data.mode === "run" ? data.id : ""}
              disabled={runStatus === "loading"}
              onChange={(event) => onSelectRun(event.target.value)}
            >
              <option value="">Demo</option>
              {runtime.runs.map((run) => <option key={run.id} value={run.id}>{run.label}</option>)}
            </select>
          </label>
        </header>

        <div className="scope-body">
          <nav className="scope-tabs" aria-label="Scope view">
            <button
              className={activeView === "trace" ? "is-active" : ""}
              type="button"
              disabled={!data.tokens.length}
              aria-pressed={activeView === "trace"}
              onClick={() => setView("trace")}
            >
              SOURCES
            </button>
            <button
              className={activeView === "variants" ? "is-active" : ""}
              type="button"
              disabled={data.mode !== "run" || !data.tokens.length || !variantOptions.length}
              aria-pressed={activeView === "variants"}
              onClick={() => setView("variants")}
            >
              VARIANTS
            </button>
            <button
              className={activeView === "layers" ? "is-active" : ""}
              type="button"
              disabled={!layersAvailable}
              aria-pressed={activeView === "layers"}
              title={!layersAvailable ? "A completed run response is required" : undefined}
              onClick={() => setView("layers")}
            >
              LAYERS
            </button>
          </nav>

          {activeView === "layers" ? (
            <LayerScope
              runId={data.id}
              text={data.response ?? ""}
              engine={runtime.engine}
              workspaceReadouts={data.workspaceReadouts ?? []}
              selectedToken={selectedToken}
              selectedLayer={selectedLayer}
              onSelectLayer={setSelectedLayer}
            />
          ) : activeView === "variants" ? (
            <VariantScope
              current={data}
              reference={variantReference}
              referenceId={variantReferenceId}
              referenceStatus={variantStatus}
              referenceOptions={variantOptions}
              alignment={variantAlignment}
              relation={variantRelation}
              selectedToken={selectedToken}
              onSelectToken={setSelectedToken}
              onSelectReference={setVariantReferenceId}
            />
          ) : (
            <TraceScope
              sources={data.contextSources ?? data.sources}
              coverage={data.contextCoverage}
              tokens={data.tokens}
              selectedToken={selectedToken}
              onSelectToken={setSelectedToken}
            />
          )}
        </div>

        {activeView === "layers" ? (
          <footer className="scope-controls trace-controls layer-controls">
            <span><b>RECORDED POSITION</b>{selectedToken + 1} / {data.tokens.length}</span>
            <span><b>SELECTED LAYER</b>L{selectedLayer}</span>
            <span><b>WORKER</b>{runtime.engine?.model ?? "UNAVAILABLE"}</span>
            <span className="measurement-chip">POST-HOC</span>
          </footer>
        ) : activeView === "variants" ? (
          <footer className="scope-controls trace-controls variant-controls">
            <span><b>POSITION</b>{selectedToken + 1} / {data.tokens.length}</span>
            <span><b>IDENTITY</b>{variantIdentity}</span>
            <span>
              <b>CONFIDENCE Δ</b>
              {variantConfidenceDelta == null
                ? "—"
                : `${variantConfidenceDelta >= 0 ? "+" : ""}${variantConfidenceDelta.toFixed(4)}`}
            </span>
            <span className="measurement-chip">STRUCTURAL</span>
          </footer>
        ) : (
          <footer className="scope-controls trace-controls">
            <span><b>POSITION</b>{selectedToken + 1} / {data.tokens.length}</span>
            <span><b>CONFIDENCE</b>{token?.confidence == null ? "—" : formatPercent(token.confidence)}</span>
            <span><b>TOP-K ENTROPY</b>{token ? `${token.entropy.toFixed(3)} bits` : "—"}</span>
            <span className="measurement-chip">{data.mode === "run" ? "RECORDED" : "DEMO DATA"}</span>
          </footer>
        )}
      </section>

      {inspectorOpen && (
        <aside className="instrument inspector" aria-labelledby="inspector-title">
          <header className="instrument-head compact">
            <div>
              <span className="eyebrow">SELECTION</span>
              <h2 id="inspector-title">
                {activeView === "layers"
                  ? "Layer inspector"
                  : activeView === "variants"
                    ? "Variant inspector"
                    : "Token inspector"}
              </h2>
            </div>
            <strong className="layer-number">{activeView === "layers" ? `L${selectedLayer}` : `#${selectedToken + 1}`}</strong>
          </header>

          {activeView === "layers" ? (
            <section className="inspector-section layer-summary">
              <dl className="metric-list">
                <div><dt>Recorded token</dt><dd>{token?.text || "∅"}</dd></div>
                <div><dt>Recorded position</dt><dd>{selectedToken + 1} / {data.tokens.length}</dd></div>
                <div><dt>Selected layer</dt><dd>L{selectedLayer}</dd></div>
                <div><dt>Analysis</dt><dd>POST-HOC · CURRENT WORKER</dd></div>
                <div><dt>Worker model</dt><dd>{runtime.engine?.model ?? "UNAVAILABLE"}</dd></div>
                <div><dt>J-lens</dt><dd>{runtime.engine?.jlens ? "LOADED" : "NOT LOADED"}</dd></div>
                <div><dt>SAE</dt><dd>{runtime.engine?.sae ? "LOADED" : "NOT LOADED"}</dd></div>
              </dl>
            </section>
          ) : activeView === "variants" && token ? (
            <section className="inspector-section token-summary variant-token-summary">
              <div className="variant-token-pair">
                <span><b>REFERENCE</b>{variantReferenceToken?.text || "∅"}</span>
                <i>→</i>
                <span><b>CURRENT</b>{token.text || "∅"}</span>
              </div>
              <strong>{variantIdentity}</strong>
              <dl className="metric-list">
                <div><dt>Current position</dt><dd>{selectedToken + 1} / {data.tokens.length}</dd></div>
                <div><dt>Reference position</dt><dd>{variantColumn?.aIndex == null ? "—" : variantColumn.aIndex + 1}</dd></div>
                <div><dt>Current confidence</dt><dd>{token.confidence?.toFixed(4) ?? "—"}</dd></div>
                <div><dt>Reference confidence</dt><dd>{variantReferenceToken?.confidence?.toFixed(4) ?? "—"}</dd></div>
                <div>
                  <dt>Confidence Δ</dt>
                  <dd>
                    {variantConfidenceDelta == null
                      ? "—"
                      : `${variantConfidenceDelta >= 0 ? "+" : ""}${variantConfidenceDelta.toFixed(4)}`}
                  </dd>
                </div>
              </dl>
            </section>
          ) : token ? (
            <section className="inspector-section token-summary">
              <strong>{token.text || "∅"}</strong>
              <span className={`band-chip band-${token.band ?? "none"}`}>{token.band?.toUpperCase() ?? "UNBANDED"}</span>
              <dl className="metric-list">
                <div><dt>Position</dt><dd>{selectedToken + 1} / {data.tokens.length}</dd></div>
                <div><dt>Confidence</dt><dd>{token.confidence == null ? "—" : token.confidence.toFixed(4)}</dd></div>
                <div><dt>Top-k entropy</dt><dd>{token.entropy.toFixed(4)} bits</dd></div>
              </dl>
            </section>
          ) : null}

          <section className="inspector-section">
            <div className="section-title">
              <h3>{activeView === "layers" ? "Recorded token distribution" : "Token distribution"}</h3>
              <span>TOP-K</span>
            </div>
            <div className="candidate-list">
              {candidates.map((candidate, index) => (
                <button
                  type="button"
                  className={`${index === 0 ? "candidate is-leading" : "candidate"} ${forkToken === candidate.token ? "is-fork-choice" : ""}`}
                  disabled={index === 0 || data.mode !== "run"}
                  aria-pressed={index === 0 ? undefined : forkToken === candidate.token}
                  onClick={() => setForkToken(candidate.token)}
                  key={`${candidate.token}-${index}`}
                >
                  <span>{candidate.token || "∅"}</span>
                  <i><b style={{ width: formatPercent(Math.max(0, candidate.score)) }} /></i>
                  <output>{candidate.score.toFixed(4)}</output>
                </button>
              ))}
            </div>
            {activeView === "trace" && data.mode === "run" && (
              <div className="fork-control">
                <div>
                  <span>FORK TOKEN {selectedToken + 1}</span>
                  <strong>{token?.text || "∅"} <i>→</i> {forkToken || "—"}</strong>
                </div>
                <button
                  type="button"
                  disabled={!forkToken || forkState.status === "loading"}
                  onClick={() => onFork(selectedToken, forkToken)}
                >
                  {forkState.status === "loading" ? "FORKING" : "FORK RUN"}
                </button>
              </div>
            )}
            {forkState.status === "error" && (
              <p className="fork-result is-error" role="status">{forkState.message}</p>
            )}
            {forkState.status === "success" && (
              <div className="fork-result" role="status">
                <span>CHILD {forkState.childId}</span>
                <a href={`#/compare/${encodeURIComponent(forkState.parentId)}/${encodeURIComponent(forkState.childId)}`}>
                  COMPARE PARENT / CHILD
                </a>
              </div>
            )}
          </section>

          {activeView === "trace" && (
            <section className="inspector-section">
              <div className="section-title"><h3>Sources</h3><span>{token?.sources?.length ?? 0}</span></div>
              <div className="token-sources">
                {(token?.sources ?? []).map((source) => (
                  <div key={source.sourceId}>
                    <strong>{source.label}</strong>
                    <span className={`effect-${source.effect}`}>{source.effect}</span>
                    <output>{source.deltaNats >= 0 ? "+" : ""}{source.deltaNats.toFixed(4)} nats</output>
                  </div>
                ))}
                {!token?.sources?.length && <span className="unavailable">UNRESOLVED</span>}
              </div>
            </section>
          )}
          {activeView === "variants" && (
            <section className="inspector-section variant-evidence">
              <div className="section-title"><h3>Variant evidence</h3><span>{variantRelation?.kind.toUpperCase() ?? "—"}</span></div>
              <dl className="metric-list">
                <div><dt>Reference</dt><dd>{variantReference?.id.slice(-6) ?? "—"}</dd></div>
                <div><dt>Current</dt><dd>{data.id.slice(-6)}</dd></div>
                <div><dt>Basis</dt><dd>{variantRelation?.evidence ?? "—"}</dd></div>
                <div><dt>Adapter reference</dt><dd>{variantReference?.configuration.adapters.join(", ") || "UNREPORTED"}</dd></div>
                <div><dt>Adapter current</dt><dd>{data.configuration.adapters.join(", ") || "UNREPORTED"}</dd></div>
              </dl>
              {variantDials.length > 0 && (
                <div className="variant-dial-deltas">
                  <header><span>DIAL DIFFERENCES</span><b>{variantDials.length}</b></header>
                  {variantDials.slice(0, 6).map((dial) => (
                    <div key={dial.name}>
                      <span>{dial.name}</span>
                      <output>{dial.reference >= 0 ? "+" : ""}{dial.reference.toFixed(2)} → {dial.current >= 0 ? "+" : ""}{dial.current.toFixed(2)}</output>
                    </div>
                  ))}
                </div>
              )}
              {variantReference && (
                <a href={`#/compare/${encodeURIComponent(variantReference.id)}/${encodeURIComponent(data.id)}`}>
                  OPEN FULL COMPARE
                </a>
              )}
            </section>
          )}
        </aside>
      )}

      <section className="instrument residual-panel" aria-labelledby="residual-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">{activeView === "variants" ? "ALIGNED TOKENS" : "TOKENS"}</span>
            <h2 id="residual-title">
              {activeView === "layers"
                ? "Recorded confidence trace"
                : activeView === "variants"
                  ? "Confidence difference"
                  : "Confidence trace"}
            </h2>
          </div>
          <div className="legend">
            <span className="violet">{activeView === "variants" ? "CURRENT − REFERENCE" : "CONFIDENCE"}</span>
            <span className="cyan">{activeView === "variants" ? "ZERO" : "TOP-K ENTROPY"}</span>
          </div>
        </header>
        <div className="plot-wrap">
          {activeView === "variants"
              ? (
                  <VariantDeltaPlot
                    current={data}
                    reference={variantReference}
                    alignment={variantAlignment}
                    selectedToken={selectedToken}
                  />
                )
              : <ConfidencePlot tokens={data.tokens} selectedToken={selectedToken} />}
          <div className="plot-axis">
            <><span>1</span><span>{Math.max(1, Math.ceil(data.tokens.length / 2))}</span><span>{data.tokens.length}</span></>
          </div>
        </div>
      </section>

      <section className="instrument token-panel" aria-labelledby="token-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">OUTPUT</span>
            <h2 id="token-title">Token tape</h2>
          </div>
          <span className="token-count">{selectedToken + 1} / {data.tokens.length}</span>
        </header>
        <div
          className={`token-tape ${data.tokens.length > 16 ? "is-dense" : ""} ${data.tokens.length > tapeLimit ? "is-windowed" : ""}`}
          role="listbox"
          aria-label="Output tokens"
        >
          {tapeStart > 0 && <span className="tape-gap" aria-hidden="true">1–{tapeStart}</span>}
          {tapeTokens.map((item, offset) => {
            const index = tapeStart + offset;
            return (
            <button
              type="button"
              role="option"
              aria-selected={selectedToken === index}
              tabIndex={selectedToken === index ? 0 : -1}
              data-token-index={index}
              className={`${selectedToken === index ? "is-selected" : ""} band-${item.band ?? "none"}`}
              onClick={() => setSelectedToken(index)}
              onKeyDown={(event) => handleTokenKeys(event, index)}
              key={`${item.text}-${index}`}
              title={item.text}
            >
              <span>{item.text || "∅"}</span>
              <i style={{ height: `${Math.max(8, Math.min(100, item.entropy / 1.6 * 100))}%` }} />
            </button>
            );
          })}
          {tapeEnd < data.tokens.length && (
            <span className="tape-gap" aria-hidden="true">{tapeEnd + 1}–{data.tokens.length}</span>
          )}
        </div>
        <div className="tape-scrubber">
          <input
            aria-label="Selected token"
            type="range"
            min="0"
            max={Math.max(0, data.tokens.length - 1)}
            value={selectedToken}
            disabled={!data.tokens.length}
            onChange={(event) => setSelectedToken(Number(event.target.value))}
          />
        </div>
      </section>
    </>
  );
}

export function Observatory(props: ObservatoryProps) {
  const { data, initialState, runtime } = props;
  const resetKey = [
    data.id,
    initialState?.view ?? "",
    initialState?.token ?? "",
    initialState?.reference ?? "",
    initialState?.layer ?? "",
    runtime.engine?.layerCount ?? "",
  ].join("\u0000");
  return <ObservatoryWorkspace key={resetKey} {...props} />;
}
