import { useEffect, useMemo, useState, type CSSProperties } from "react";
import type { RuntimeState, WorkspaceReadout } from "../../data/types";
import type { WorkbenchReadoutMeasurements } from "../../data/tokenWorkbench";
import { loadLayerEvidence, type CausalTraceEvidence, type LayerEvidence } from "./layerApi";
import type { ActionState } from "./useTokenWorkbench";

interface LayerScopeProps {
  runId: string;
  text: string;
  engine?: RuntimeState["engine"];
  workspaceReadouts: WorkspaceReadout[];
  /** The workbench `readouts` section's recorded per-token measurements, when evidence has loaded --
   * genuinely new facts (logprob) this Studio did not previously surface anywhere. */
  measurements?: WorkbenchReadoutMeasurements;
  selectedToken: number;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
  /** Causal trace is now one of the token workbench's four generic actions (Milestone F) -- this
   * component only RENDERS its result; `useTokenWorkbench`/ActionTray own running, polling, and
   * cancelling it, so a token selected here and one selected in the action tray can never disagree
   * about whether a trace is in flight. */
  causalAction: ActionState;
  onRunCausalTrace: () => void;
  onCancelCausalTrace: () => void;
}

type EvidenceState =
  | { status: "loading" }
  | { status: "done"; evidence: LayerEvidence }
  | { status: "error"; message: string };

function percentile(values: number[], fraction: number): number {
  if (!values.length) return 1;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))] || 1;
}

function formatScore(value: number): string {
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(3);
}

function isStoredFeature(readout: WorkspaceReadout): boolean {
  const provider = readout.provider.toLowerCase();
  const providerType = readout.providerType?.toLowerCase() ?? "";
  return provider.includes("sae")
    || providerType.includes("sae")
    || providerType === "engine_concepts"
    || readout.readoutKind === "concept";
}

export function LayerScope({
  runId,
  text,
  engine,
  workspaceReadouts,
  measurements,
  selectedToken,
  selectedLayer,
  onSelectLayer,
  causalAction,
  onRunCausalTrace,
  onCancelCausalTrace,
}: LayerScopeProps) {
  const [evidenceState, setEvidenceState] = useState<EvidenceState>({ status: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setEvidenceState({ status: "loading" });
    void loadLayerEvidence(text, controller.signal)
      .then((evidence) => {
        if (!controller.signal.aborted) setEvidenceState({ status: "done", evidence });
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setEvidenceState({
          status: "error",
          message: error instanceof Error ? error.message : "Layer evidence unavailable",
        });
      });
    return () => controller.abort();
  }, [runId, text]);

  const evidence = evidenceState.status === "done" ? evidenceState.evidence : undefined;
  const residual = evidence?.residual;
  const lens = evidence?.jlens;
  const featureReadouts = useMemo(
    () => workspaceReadouts.filter(isStoredFeature),
    [workspaceReadouts],
  );
  const selectedFeatures = featureReadouts.filter((readout) =>
    readout.tokenIndex === selectedToken || readout.position === selectedToken);
  const workerTokenCount = residual?.nTokens || lens?.layers[0]?.tokens.length || 0;
  const workerPosition = Math.max(0, Math.min(selectedToken, Math.max(0, workerTokenCount - 1)));
  const residualScale = useMemo(() => {
    if (!residual?.available) return 1;
    return percentile(residual.norms.flat().map((value) => Math.log1p(Math.max(0, value))), 0.95);
  }, [residual]);
  const lensAtPosition = useMemo(() => {
    if (!lens?.available) return [];
    return lens.layers.map((layer) => ({
      layer: layer.layer,
      token: layer.tokens[Math.min(selectedToken, Math.max(0, layer.tokens.length - 1))],
      candidates: layer.readouts[Math.min(selectedToken, Math.max(0, layer.readouts.length - 1))] ?? [],
    }));
  }, [lens, selectedToken]);
  const lensLeadChanges = lensAtPosition.reduce((count, layer, index) => (
    index > 0 && layer.candidates[0]?.piece !== lensAtPosition[index - 1].candidates[0]?.piece
      ? count + 1
      : count
  ), 0);
  const lastLeadChange = [...lensAtPosition].reverse().find((layer, reverseIndex) => {
    const index = lensAtPosition.length - 1 - reverseIndex;
    return index > 0 && layer.candidates[0]?.piece !== lensAtPosition[index - 1].candidates[0]?.piece;
  })?.layer;

  const causalBusy = causalAction.phase === "running" || causalAction.phase === "cancelling";
  // useTokenWorkbench's `ActionState.artifact` is a plain union across all four actions (see that
  // module's own doc comment on why) -- this is the one place that knows THIS prop only ever carries a
  // causal-trace artifact, so it casts once here rather than at every read site below.
  const causalArtifact = causalAction.artifact as CausalTraceEvidence | undefined;

  if (evidenceState.status === "loading") {
    return (
      <div className="layer-analysis is-loading" aria-live="polite">
        <span>READING CURRENT WORKER</span>
      </div>
    );
  }

  if (evidenceState.status === "error") {
    return (
      <div className="layer-analysis is-loading" role="status">
        <span>{evidenceState.message}</span>
      </div>
    );
  }

  return (
    <div className="layer-analysis">
      <div className="layer-evidence-strip" aria-label="Layer evidence availability">
        <span className={residual?.available ? "is-available" : "is-unavailable"}>
          <b>RESIDUAL</b>
          {residual?.available ? `${residual.nLayer} × ${residual.nTokens}` : "UNAVAILABLE"}
        </span>
        <span className={lens?.available ? "is-available" : "is-unavailable"}>
          <b>J-LENS</b>
          {lens?.available ? `${lens.layers.length} FITTED LAYERS` : "NOT LOADED"}
        </span>
        <span className={featureReadouts.length ? "is-available" : "is-unavailable"}>
          <b>SAE / CONCEPTS</b>
          {featureReadouts.length
            ? `${featureReadouts.length} STORED`
            : engine?.sae
              ? "LOADED · NOT RECORDED"
              : "NOT LOADED"}
        </span>
        <span><b>CAUSAL</b>ON DEMAND</span>
        <span className={measurements?.logprob != null ? "is-available" : "is-unavailable"}>
          <b>RECORDED LOGPROB</b>
          {measurements?.logprob != null ? measurements.logprob.toFixed(4) : "UNRECORDED"}
        </span>
      </div>

      <div className="layer-analysis-grid">
        <section className="layer-readout residual-readout" aria-labelledby="residual-map-title">
          <header>
            <div>
              <span className="eyebrow">RESIDUAL MAGNITUDE</span>
              <h3 id="residual-map-title">Layer × token map</h3>
            </div>
            <div className="readout-tags">
              <span>POST-HOC · CURRENT WORKER</span>
              <span>LOG1P DISPLAY</span>
            </div>
          </header>
          {residual?.available ? (
            <>
              <div
                className="residual-matrix"
                style={{ "--matrix-columns": residual.nTokens } as CSSProperties}
              >
                {residual.norms.map((row, layerIndex) => (
                  <button
                    type="button"
                    className={selectedLayer === layerIndex ? "matrix-row is-selected" : "matrix-row"}
                    key={layerIndex}
                    aria-pressed={selectedLayer === layerIndex}
                    aria-label={`Layer ${layerIndex}, mean residual norm ${formatScore(residual.layerMean[layerIndex] ?? 0)}`}
                    onClick={() => onSelectLayer(layerIndex)}
                  >
                    <b>L{layerIndex}</b>
                    <span className="matrix-cells">
                      {row.map((value, position) => {
                        const level = Math.min(
                          1,
                          Math.log1p(Math.max(0, value)) / residualScale,
                        );
                        return (
                          <i
                            className={position === workerPosition ? "is-position" : ""}
                            key={position}
                            title={`L${layerIndex} · worker position ${position + 1} · norm ${formatScore(value)}`}
                            style={{ "--heat": level } as CSSProperties}
                          />
                        );
                      })}
                    </span>
                    <output>{formatScore(residual.layerMean[layerIndex] ?? 0)}</output>
                  </button>
                ))}
              </div>
              <div className="matrix-token-axis" aria-hidden="true">
                <span>1</span>
                <span>WORKER TOKENS</span>
                <span>{residual.nTokens}</span>
              </div>
              <p className="readout-note">
                {residual.truncated ? `FIRST ${residual.textChars} RESPONSE CHARACTERS · ` : ""}
                RE-TOKENIZED BY {engine?.model ?? "CURRENT WORKER"}
              </p>
            </>
          ) : (
            <p className="readout-empty">{residual?.reason ?? "Residual layer summary unavailable"}</p>
          )}
        </section>

        <section className="layer-readout lens-readout" aria-labelledby="jlens-title">
          <header>
            <div>
              <span className="eyebrow">J-LENS TRAJECTORY</span>
              <h3 id="jlens-title">Candidate by fitted layer</h3>
            </div>
            {lens?.available && (
              <div className="readout-tags">
                <span>{lensLeadChanges} LEAD CHANGES</span>
                <span>{lastLeadChange == null ? "NO LEAD CHANGE" : `LAST AT L${lastLeadChange}`}</span>
              </div>
            )}
          </header>
          {lens?.available ? (
            <>
              <div className="lens-layers">
                {lensAtPosition.map((layer, index) => {
                  const changed = index > 0
                    && layer.candidates[0]?.piece !== lensAtPosition[index - 1].candidates[0]?.piece;
                  return (
                    <button
                      type="button"
                      className={`${selectedLayer === layer.layer ? "is-selected" : ""} ${changed ? "is-change" : ""}`}
                      onClick={() => onSelectLayer(layer.layer)}
                      aria-pressed={selectedLayer === layer.layer}
                      key={layer.layer}
                    >
                      <span><b>L{layer.layer}</b>{changed ? "LEAD CHANGE" : ""}</span>
                      <strong>{layer.candidates[0]?.piece || "∅"}</strong>
                      {layer.candidates.slice(0, 3).map((candidate) => (
                        <i key={`${candidate.piece}-${candidate.score}`}>
                          <span>{candidate.piece || "∅"}</span>
                          <output>{formatScore(candidate.score)}</output>
                        </i>
                      ))}
                    </button>
                  );
                })}
              </div>
              <p className="readout-note">
                WORKER POSITION {Math.min(selectedToken + 1, lens.layers[0]?.tokens.length || 1)}
                {lens.truncated ? ` · FIRST ${lens.textChars} RESPONSE CHARACTERS` : ""}
              </p>
            </>
          ) : (
            <p className="readout-empty">{lens?.reason ?? "J-lens unavailable"}</p>
          )}
        </section>

        <section className="layer-readout feature-readout" aria-labelledby="sae-title">
          <header>
            <div>
              <span className="eyebrow">STORED FEATURE READOUTS</span>
              <h3 id="sae-title">SAE / concepts</h3>
            </div>
            <div className="readout-tags">
              <span>RUN TRACE</span>
              <span>POSITION {selectedToken + 1}</span>
            </div>
          </header>
          {selectedFeatures.length ? (
            <div className="feature-provider-list">
              {selectedFeatures.map((readout, readoutIndex) => (
                <div key={`${readout.provider}-${readout.layer}-${readoutIndex}`}>
                  <header>
                    <strong>{readout.provider}</strong>
                    <span>{readout.providerType ?? "UNCLASSIFIED"} · {readout.layer == null ? "L—" : `L${readout.layer}`}</span>
                  </header>
                  {readout.topReadouts.slice(0, 8).map((feature) => (
                    <p key={`${feature.label}-${feature.score}`}>
                      <span>{feature.label}</span>
                      <output>{formatScore(feature.score)}</output>
                    </p>
                  ))}
                </div>
              ))}
            </div>
          ) : (
            <p className="readout-empty">
              {featureReadouts.length
                ? "No stored feature readout at this token position"
                : engine?.sae
                  ? "SAE is loaded; this run did not record feature readouts"
                  : "No SAE or concept readout recorded"}
            </p>
          )}
        </section>

        <section className="layer-readout causal-readout" aria-labelledby="causal-title">
          <header>
            <div>
              <span className="eyebrow">CONTROLLED INTERVENTION</span>
              <h3 id="causal-title">Causal sites</h3>
            </div>
            <div className="causal-readout-buttons">
              {causalBusy && causalAction.job?.cancellable && (
                <button type="button" onClick={onCancelCausalTrace}>CANCEL</button>
              )}
              <button
                type="button"
                disabled={causalAction.phase === "unavailable" || causalBusy}
                onClick={onRunCausalTrace}
              >
                {causalBusy ? "TRACING" : `TRACE TOKEN ${selectedToken + 1}`}
              </button>
            </div>
          </header>
          {/* Causal trace is a shared action-tray action now (Milestone F) -- this section only renders
              whatever `causalAction` (owned by useTokenWorkbench) currently reports; the button above is
              a convenience shortcut into the SAME `runAction("causal_trace")` call the action tray uses,
              never a second orchestration path. */}
          {causalAction.phase === "unavailable" && (
            <p className="readout-empty">{causalAction.reason ?? "Causal trace unavailable"}</p>
          )}
          {causalAction.phase === "idle" && (
            <p className="readout-empty">Not run for this token</p>
          )}
          {(causalAction.phase === "running" || causalAction.phase === "cancelling") && (
            <p className="readout-empty">
              Running matched-random controls
              {causalAction.job ? ` · ${causalAction.job.progress.phase} ${causalAction.job.progress.percent}%` : ""}
            </p>
          )}
          {(causalAction.phase === "cancelled" || causalAction.phase === "error") && (
            <p className="readout-empty is-error">{causalAction.reason ?? "Causal trace did not complete"}</p>
          )}
          {(causalAction.phase === "cached" || causalAction.phase === "completed") && causalArtifact && !causalArtifact.ok && (
            <p className="readout-empty is-error">
              {causalArtifact.blocked || causalArtifact.error || "Causal trace unavailable"}
            </p>
          )}
          {(causalAction.phase === "cached" || causalAction.phase === "completed") && causalArtifact?.ok && (
            <div className="causal-results">
              <dl>
                <div><dt>Target</dt><dd>{causalArtifact.target?.piece || "∅"} · {causalArtifact.target?.pos ?? selectedToken}</dd></div>
                <div><dt>Survivors</dt><dd>{causalArtifact.survivorCount ?? causalArtifact.nodes.length} / {causalArtifact.candidateCount}</dd></div>
                <div><dt>Control</dt><dd>{causalArtifact.verdict ?? "—"}</dd></div>
                <div><dt>Noise floor</dt><dd>{causalArtifact.noiseFloor == null ? "—" : formatScore(causalArtifact.noiseFloor)}</dd></div>
              </dl>
              <div className="causal-node-list">
                {causalArtifact.nodes.slice(0, 8).map((node) => (
                  <p key={`${node.layer}-${node.pos}`}>
                    <span>L{node.layer} · P{node.pos + 1}</span>
                    <output>{node.deltaFull >= 0 ? "+" : ""}{formatScore(node.deltaFull)} Δlogp</output>
                  </p>
                ))}
                {!causalArtifact.nodes.length && <span>NO SITES ABOVE CONTROL</span>}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
