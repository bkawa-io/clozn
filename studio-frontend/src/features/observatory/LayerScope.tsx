import { useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import type { RuntimeState, WorkspaceReadout } from "../../data/types";
import {
  loadCausalTrace,
  loadLayerEvidence,
  type CausalTraceEvidence,
  type LayerEvidence,
} from "./layerApi";

interface LayerScopeProps {
  runId: string;
  text: string;
  engine?: RuntimeState["engine"];
  workspaceReadouts: WorkspaceReadout[];
  selectedToken: number;
  selectedLayer: number;
  onSelectLayer: (layer: number) => void;
}

type EvidenceState =
  | { status: "loading" }
  | { status: "done"; evidence: LayerEvidence }
  | { status: "error"; message: string };

type CausalState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "done"; evidence: CausalTraceEvidence }
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
  selectedToken,
  selectedLayer,
  onSelectLayer,
}: LayerScopeProps) {
  const [evidenceState, setEvidenceState] = useState<EvidenceState>({ status: "loading" });
  const [causalState, setCausalState] = useState<CausalState>({ status: "idle" });
  const causalController = useRef<AbortController | null>(null);

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

  useEffect(() => {
    causalController.current?.abort();
    setCausalState({ status: "idle" });
    return () => causalController.current?.abort();
  }, [runId, selectedToken]);

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

  async function runCausalTrace() {
    causalController.current?.abort();
    const controller = new AbortController();
    causalController.current = controller;
    setCausalState({ status: "loading" });
    try {
      const result = await loadCausalTrace(runId, selectedToken, controller.signal);
      if (!controller.signal.aborted) setCausalState({ status: "done", evidence: result });
    } catch (error) {
      if (controller.signal.aborted) return;
      setCausalState({
        status: "error",
        message: error instanceof Error ? error.message : "Causal trace unavailable",
      });
    }
  }

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
            <button
              type="button"
              disabled={causalState.status === "loading"}
              onClick={runCausalTrace}
            >
              {causalState.status === "loading" ? "TRACING" : `TRACE TOKEN ${selectedToken + 1}`}
            </button>
          </header>
          {causalState.status === "idle" && (
            <p className="readout-empty">Not run for this token</p>
          )}
          {causalState.status === "loading" && (
            <p className="readout-empty">Running matched-random controls</p>
          )}
          {causalState.status === "error" && (
            <p className="readout-empty is-error">{causalState.message}</p>
          )}
          {causalState.status === "done" && !causalState.evidence.ok && (
            <p className="readout-empty is-error">
              {causalState.evidence.blocked || causalState.evidence.error || "Causal trace unavailable"}
            </p>
          )}
          {causalState.status === "done" && causalState.evidence.ok && (
            <div className="causal-results">
              <dl>
                <div><dt>Target</dt><dd>{causalState.evidence.target?.piece || "∅"} · {causalState.evidence.target?.pos ?? selectedToken}</dd></div>
                <div><dt>Survivors</dt><dd>{causalState.evidence.survivorCount ?? causalState.evidence.nodes.length} / {causalState.evidence.candidateCount}</dd></div>
                <div><dt>Control</dt><dd>{causalState.evidence.verdict ?? "—"}</dd></div>
                <div><dt>Noise floor</dt><dd>{causalState.evidence.noiseFloor == null ? "—" : formatScore(causalState.evidence.noiseFloor)}</dd></div>
              </dl>
              <div className="causal-node-list">
                {causalState.evidence.nodes.slice(0, 8).map((node) => (
                  <p key={`${node.layer}-${node.pos}`}>
                    <span>L{node.layer} · P{node.pos + 1}</span>
                    <output>{node.deltaFull >= 0 ? "+" : ""}{formatScore(node.deltaFull)} Δlogp</output>
                  </p>
                ))}
                {!causalState.evidence.nodes.length && <span>NO SITES ABOVE CONTROL</span>}
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
