import { Fragment, useEffect, useMemo, useState } from "react";
import { EvidenceMark } from "../../components/EvidenceMark";
import { PairedDelta } from "../../components/PairedDelta";
import { TypedActionOffer } from "../../components/TypedActionOffer";
import { loadRunInspection } from "../../data/api";
import type { ObservatoryData, RuntimeState, TokenReading } from "../../data/types";
import { Experiments } from "../experiments/Experiments";
import { alignTokens, type TokenAlignment } from "./alignment";
import {
  loadRunComparison,
  plannedChangeTests,
  previewRunChangeTest,
  runChangeTest,
  type RunChangeTestDocument,
  type RunChangeTestEntry,
  type RunComparisonDocument,
} from "./api";

export type CompareMode = "runs" | "matrix";

export interface CompareProps {
  runtime: RuntimeState;
  initialA?: string;
  initialB?: string;
  inspectorOpen: boolean;
  /** Compare currently has two backed object types: recorded run pairs and experiment result matrices. */
  mode?: CompareMode;
  initialExperimentId?: string;
  rawExperimentQuery?: string;
  /** Legacy experiment hashes retain their own base when filters/cell selection are rewritten. */
  experimentRouteBase?: string;
}

function shortId(id: string) {
  return id.slice(-6);
}

function visibleToken(text: string) {
  if (!text) return "∅";
  if (!text.trim()) return text.includes("\n") ? "↵" : "\u00a0";
  return text
    .replace(/\r\n|\r|\n/g, "")
    .replace(/\t/g, "⇥")
    .replaceAll(" ", "\u00a0");
}

function signed(value: number, digits = 4) {
  if (!Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function tokenIdentity(a?: TokenReading, b?: TokenReading) {
  if (!a || !b) return "UNALIGNED";
  return a.text === b.text ? "SAME" : "CHANGED";
}

function errorText(error: unknown): string {
  return error instanceof Error && error.message ? error.message : "The request failed without a recorded reason.";
}

function CompareModePicker({ mode }: { mode: CompareMode }) {
  return (
    <label className="compare-object-picker">
      <span>OBJECT TYPE</span>
      <select
        aria-label="Comparison object type"
        value={mode}
        onChange={(event) => {
          // These are the only two comparison objects this frontend can load today. Do not advertise a
          // model/template comparison merely because its labels would fit in this picker.
          window.location.hash = event.target.value === "matrix" ? "#/compare/matrix" : "#/compare";
        }}
      >
        <option value="runs">Recorded runs</option>
        <option value="matrix">Experiment result matrix</option>
      </select>
    </label>
  );
}

function ConfidenceComparison({
  a,
  b,
  selectedA,
  selectedB,
}: {
  a: TokenReading[];
  b: TokenReading[];
  selectedA?: number;
  selectedB?: number;
}) {
  const count = Math.max(a.length, b.length, 2);
  const points = (tokens: TokenReading[]) => tokens.map((token, index) => {
    const x = 4 + index * (92 / Math.max(1, count - 1));
    const y = 92 - (token.confidence ?? 0) * 78;
    return `${x},${y}`;
  }).join(" ");
  const selectedX = (index: number) => 4 + index * (92 / Math.max(1, count - 1));

  return (
    <svg className="compare-plot-svg" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="A and B confidence plot">
      <path className="compare-plot-grid" d="M0 25H100M0 50H100M0 75H100M20 0V100M40 0V100M60 0V100M80 0V100" />
      <polyline className="compare-line is-a" points={points(a)} />
      <polyline className="compare-line is-b" points={points(b)} />
      {selectedA != null && (
        <line className="compare-cursor is-a" x1={selectedX(selectedA)} y1="4" x2={selectedX(selectedA)} y2="96" />
      )}
      {selectedB != null && (
        <line className="compare-cursor is-b" x1={selectedX(selectedB)} y1="4" x2={selectedX(selectedB)} y2="96" />
      )}
    </svg>
  );
}

function TokenSequence({
  label,
  tone,
  tokens,
  alignment,
  selectedColumn,
  onSelect,
}: {
  label: string;
  tone: "a" | "b";
  tokens: TokenReading[];
  alignment: TokenAlignment;
  selectedColumn: number;
  onSelect: (column: number) => void;
}) {
  const columnByToken = tone === "a" ? alignment.columnByA : alignment.columnByB;
  return (
    <div className={`compare-token-row is-${tone}`}>
      <strong>{label}</strong>
      <div className="compare-sequence" role="listbox" aria-label={`Run ${label} output tokens`}>
        {tokens.map((token, index) => {
          const hasBreak = token.text.includes("\n");
          const columnIndex = columnByToken.get(index);
          const kind = columnIndex == null ? `${tone}-only` : alignment.columns[columnIndex].kind;
          const changed = kind !== "same";
          return (
            <Fragment key={`${index}-${token.text}`}>
              <button
                type="button"
                role="option"
                aria-selected={selectedColumn === columnIndex}
                aria-label={`${label} token ${index + 1}: ${token.text || "blank"}`}
                className={`${selectedColumn === columnIndex ? "is-selected" : ""} ${changed ? `is-changed is-${kind}` : "is-matched"}`}
                onClick={() => {
                  if (columnIndex != null) onSelect(columnIndex);
                }}
              >
                {visibleToken(token.text)}
              </button>
              {hasBreak && <span className="compare-break" aria-hidden="true" />}
            </Fragment>
          );
        })}
      </div>
    </div>
  );
}

function changeCost(document: RunChangeTestDocument): string {
  const parts: string[] = [];
  if (document.budget.maxRuns != null) parts.push(`up to ${document.budget.maxRuns} child run${document.budget.maxRuns === 1 ? "" : "s"}`);
  if (document.budget.maxSeconds != null) parts.push(`up to ${document.budget.maxSeconds} seconds`);
  return parts.length
    ? `This bounded execution may consume ${parts.join(" and ")}.`
    : "The preview omitted a numeric budget; the server still enforces its own execution limits.";
}

function blockerReason(tests: readonly RunChangeTestEntry[]): string {
  const reasons = [...new Set(tests.map((test) => test.reason).filter(Boolean))];
  return reasons.length
    ? reasons.join(" ")
    : "No replayable context, template, or sampling change was available for this recorded pair.";
}

function ChangeTestReceipt({ document }: { document: RunChangeTestDocument }) {
  return (
    <section className="compare-change-receipt" aria-labelledby="compare-change-receipt-title">
      <header>
        <span className="eyebrow">CONTROLLED CHANGE TEST</span>
        <h3 id="compare-change-receipt-title">{document.dryRun ? "Plan only" : document.status.replaceAll("_", " ")}</h3>
      </header>
      <p>
        {document.dryRun
          ? "This document is a zero-run preview; it did not execute a model or establish causal support."
          : "Each completed arm names a persisted child run below; no conclusion is stronger than the recorded test status."}
      </p>
      <ul aria-label="Controlled change-test results">
        {document.tests.map((test) => (
          <li key={`${test.kind}-${test.status}`} data-change-test-status={test.status}>
            <strong>{test.kind}</strong>
            <span>{test.status.replaceAll("_", " ")}</span>
            <p>{test.reason}</p>
            {test.evidenceRunIds.length > 0 && <small>Evidence runs: {test.evidenceRunIds.join(", ")}</small>}
          </li>
        ))}
      </ul>
      {document.summary.causallySupported.length > 0 && (
        <p className="compare-change-summary">
          Causally supported: {document.summary.causallySupported.join(", ")}.
        </p>
      )}
    </section>
  );
}

function ControlledTestOffer({
  plan,
  result,
  phase,
  error,
  onPreview,
  onRun,
}: {
  plan: RunChangeTestDocument | null;
  result: RunChangeTestDocument | null;
  phase: "idle" | "planning" | "previewed" | "executing" | "executed";
  error: string | null;
  onPreview: () => void;
  onRun: () => void;
}) {
  if (result) return <ChangeTestReceipt document={result} />;

  if (!plan) {
    const isPlanning = phase === "planning";
    return (
      <div className="compare-action-offer">
        <TypedActionOffer
          title="Preview a bounded change test"
          absence={{
            state: "not_measured",
            reason: "No controlled change test has been planned or run for this pair.",
          }}
          cost="Preview is model-free and starts no child runs; it only reports which recorded swaps could be tested."
          preconditions={[
            "A and B must remain two distinct recorded runs.",
            "The server must be able to read the pair's recorded inputs and identities.",
            "Only context, template, and sampling swaps are supported by this instrument.",
          ]}
          action={isPlanning
            ? {
              availability: "blocked",
              label: "Preparing change-test preview",
              blockerReason: "The explicit preview request is still in progress.",
            }
            : { availability: "available", label: "Preview change test", onAction: onPreview }}
        />
        {error && <p className="compare-action-error" role="alert">Preview unavailable: {error}</p>}
      </div>
    );
  }

  const available = plannedChangeTests(plan);
  const unavailable = plan.tests.filter((test) => !available.includes(test));
  const hasRecordedBudget = plan.budget.maxRuns != null && plan.budget.maxSeconds != null;
  const canRun = available.length > 0 && hasRecordedBudget;
  const isExecuting = phase === "executing";
  const unavailableReason = hasRecordedBudget
    ? blockerReason(unavailable.length ? unavailable : plan.tests)
    : "The change-test preview omitted its run or time cap, so Studio cannot offer an execution with an honest cost.";

  return (
    <div className="compare-action-offer">
      <TypedActionOffer
        title="Run the previewed controlled test"
        absence={canRun
          ? {
            state: "not_measured",
            reason: "A bounded plan exists, but no live control or treatment arm has run for this pair.",
          }
          : {
            state: "unavailable",
            reason: unavailableReason,
          }}
        cost={changeCost(plan)}
        preconditions={[
          `Preview selected ${available.length} replayable swap${available.length === 1 ? "" : "s"}: ${available.map((test) => test.kind).join(", ") || "none"}.`,
          "The candidate model worker must be ready when execution begins.",
          "The server rechecks model identity and replay qualification before any child arm starts.",
        ]}
        action={!canRun || isExecuting
          ? {
            availability: "blocked",
            label: isExecuting ? "Running controlled test" : "Run controlled test",
            blockerReason: isExecuting
              ? "The explicit execution request is still in progress."
              : unavailableReason,
          }
          : { availability: "available", label: "Run controlled test", onAction: onRun }}
      />
      {error && <p className="compare-action-error" role="alert">Controlled test unavailable: {error}</p>}
    </div>
  );
}

function RunComparison({ runtime, initialA, initialB, inspectorOpen }: Omit<CompareProps, "mode" | "initialExperimentId" | "rawExperimentQuery" | "experimentRouteBase">) {
  const [idA, setIdA] = useState(initialA ?? "");
  const [idB, setIdB] = useState(initialB ?? "");
  const [runA, setRunA] = useState<ObservatoryData | null>(null);
  const [runB, setRunB] = useState<ObservatoryData | null>(null);
  const [inspectionStatus, setInspectionStatus] = useState<"idle" | "loading" | "error">("idle");
  const [comparison, setComparison] = useState<RunComparisonDocument | null>(null);
  const [comparisonStatus, setComparisonStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState(0);
  const [tokenDrilldownOpen, setTokenDrilldownOpen] = useState(false);
  const [changePlan, setChangePlan] = useState<RunChangeTestDocument | null>(null);
  const [changeResult, setChangeResult] = useState<RunChangeTestDocument | null>(null);
  const [changePhase, setChangePhase] = useState<"idle" | "planning" | "previewed" | "executing" | "executed">("idle");
  const [changeError, setChangeError] = useState<string | null>(null);
  const canCompare = Boolean(idA && idB && idA !== idB);

  useEffect(() => {
    if (!runtime.runs.length) return;
    setIdA((current) => current || initialA || runtime.runs[1]?.id || runtime.runs[0].id);
    setIdB((current) => current || initialB || runtime.runs[0].id);
  }, [initialA, initialB, runtime.runs]);

  useEffect(() => {
    // A plan names one exact pair. It must not survive a selector change and be mistaken for a new pair's
    // evidence, even when the two pairs happen to share a model or a parent run.
    setChangePlan(null);
    setChangeResult(null);
    setChangePhase("idle");
    setChangeError(null);
    setTokenDrilldownOpen(false);
  }, [idA, idB]);

  useEffect(() => {
    if (!canCompare) {
      setRunA(null);
      setRunB(null);
      setInspectionStatus("idle");
      setComparison(null);
      setComparisonStatus("idle");
      setComparisonError(null);
      return;
    }

    const controller = new AbortController();
    setRunA(null);
    setRunB(null);
    setInspectionStatus("loading");
    setComparison(null);
    setComparisonStatus("loading");
    setComparisonError(null);

    void Promise.all([
      loadRunInspection(idA, controller.signal),
      loadRunInspection(idB, controller.signal),
    ]).then(([nextA, nextB]) => {
      if (controller.signal.aborted) return;
      const nextAlignment = alignTokens(nextA.tokens, nextB.tokens);
      setRunA(nextA);
      setRunB(nextB);
      setSelectedColumn(nextAlignment.firstChangedColumn < 0 ? 0 : nextAlignment.firstChangedColumn);
      setInspectionStatus("idle");
    }).catch((error) => {
      if (!controller.signal.aborted) {
        setInspectionStatus("error");
        setComparisonError((current) => current ?? errorText(error));
      }
    });

    void loadRunComparison(idA, idB, controller.signal).then((next) => {
      if (controller.signal.aborted) return;
      setComparison(next);
      setComparisonStatus("ready");
      history.replaceState(null, "", `#/compare/${encodeURIComponent(idA)}/${encodeURIComponent(idB)}`);
    }).catch((error) => {
      if (!controller.signal.aborted) {
        setComparisonStatus("error");
        setComparisonError(errorText(error));
      }
    });

    return () => controller.abort();
  }, [canCompare, idA, idB]);

  const alignment = useMemo(
    () => alignTokens(runA?.tokens ?? [], runB?.tokens ?? []),
    [runA, runB],
  );
  const selected = alignment.columns[selectedColumn];
  const selectedAIndex = selected?.aIndex;
  const selectedBIndex = selected?.bIndex;
  const tokenA = selectedAIndex == null ? undefined : runA?.tokens[selectedAIndex];
  const tokenB = selectedBIndex == null ? undefined : runB?.tokens[selectedBIndex];
  const firstChanged = alignment.firstChangedColumn < 0
    ? undefined
    : alignment.columns[alignment.firstChangedColumn];
  const firstHunk = firstChanged
    ? `A${firstChanged.aIndex == null ? "—" : `#${firstChanged.aIndex + 1}`} · B${firstChanged.bIndex == null ? "—" : `#${firstChanged.bIndex + 1}`}`
    : "—";
  const selectedPosition = `A${selectedAIndex == null ? "—" : `#${selectedAIndex + 1}`} · B${selectedBIndex == null ? "—" : `#${selectedBIndex + 1}`}`;
  const relation = runA && runB
    ? runB.parentRunId === runA.id
      ? "A → B"
      : runA.parentRunId === runB.id
        ? "B → A"
        : "PEERS"
    : "—";
  const maxTokens = Math.max(runA?.tokens.length ?? 0, runB?.tokens.length ?? 0);

  const previewControlledTest = async () => {
    setChangePhase("planning");
    setChangeError(null);
    try {
      const next = await previewRunChangeTest(idA, idB);
      setChangePlan(next);
      setChangePhase("previewed");
    } catch (error) {
      setChangePhase("idle");
      setChangeError(errorText(error));
    }
  };

  const executeControlledTest = async () => {
    if (!changePlan) return;
    const selectedTests = plannedChangeTests(changePlan).map((test) => test.kind);
    if (!selectedTests.length) return;
    setChangePhase("executing");
    setChangeError(null);
    try {
      const next = await runChangeTest(idA, idB, selectedTests, changePlan.budget);
      setChangeResult(next);
      setChangePhase("executed");
    } catch (error) {
      setChangePhase("previewed");
      setChangeError(errorText(error));
    }
  };

  return (
    <>
      <section className="instrument compare-hero" aria-labelledby="compare-title">
        <header className="instrument-head compare-head">
          <div>
            <span className="eyebrow">A / B STRUCTURAL COMPARISON</span>
            <h1 id="compare-title">What moved</h1>
          </div>
          <div className="compare-pickers">
            <CompareModePicker mode="runs" />
            <label>
              <span>A RUN</span>
              <select value={idA} onChange={(event) => setIdA(event.target.value)} disabled={inspectionStatus === "loading"}>
                <option value="">Select run A</option>
                {runtime.runs.map((run) => <option key={`a-${run.id}`} value={run.id}>{run.label}</option>)}
              </select>
            </label>
            <label>
              <span>B RUN</span>
              <select value={idB} onChange={(event) => setIdB(event.target.value)} disabled={inspectionStatus === "loading"}>
                <option value="">Select run B</option>
                {runtime.runs.map((run) => <option key={`b-${run.id}`} value={run.id}>{run.label}</option>)}
              </select>
            </label>
          </div>
        </header>
        <div className="compare-token-disclosure">
          <div>
            <strong>Token alignment</strong>
            <span>Optional drill-down; token sequence is not the primary statement of what changed.</span>
          </div>
          <button
            type="button"
            aria-expanded={tokenDrilldownOpen}
            onClick={() => setTokenDrilldownOpen((open) => !open)}
          >
            {tokenDrilldownOpen ? "Hide token alignment" : "Show token alignment"}
          </button>
        </div>

        {tokenDrilldownOpen && (
          <div className="compare-stage">
            {inspectionStatus === "error" && <div className="compare-state is-error">TOKEN ALIGNMENT UNAVAILABLE</div>}
            {!runA || !runB ? (
              <div className="compare-state">{inspectionStatus === "loading" ? "LOADING RUNS" : "SELECT TWO DISTINCT RUNS"}</div>
            ) : (
              <>
                <div className="compare-branch-field" aria-hidden="true"><i /><i /><b /></div>
                <TokenSequence
                  label="A"
                  tone="a"
                  tokens={runA.tokens}
                  alignment={alignment}
                  selectedColumn={selectedColumn}
                  onSelect={setSelectedColumn}
                />
                <div className="compare-seam">
                  <i />
                  <span>
                    {alignment.firstChangedColumn < 0
                      ? "NO IDENTITY CHANGE"
                      : `FIRST HUNK · ${firstHunk} · ${alignment.hunks} ${alignment.hunks === 1 ? "HUNK" : "HUNKS"}`}
                  </span>
                  <i />
                </div>
                <TokenSequence
                  label="B"
                  tone="b"
                  tokens={runB.tokens}
                  alignment={alignment}
                  selectedColumn={selectedColumn}
                  onSelect={setSelectedColumn}
                />
              </>
            )}
          </div>
        )}
      </section>

      <section className="instrument compare-delta-panel" aria-labelledby="compare-delta-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">RECORDED STRUCTURAL DIFFERENCE</span>
            <h2 id="compare-delta-title">Paired delta</h2>
          </div>
          {comparison?.generatedAt && <time className="compare-generated-at">Computed {comparison.generatedAt}</time>}
        </header>
        {!canCompare && (
          <p className="compare-primary-state">Select two distinct recorded runs to establish a comparison.</p>
        )}
        {canCompare && comparisonStatus === "loading" && <p className="compare-primary-state">LOADING STRUCTURAL COMPARISON</p>}
        {canCompare && comparisonStatus === "error" && (
          <div className="compare-primary-absence">
            <EvidenceMark
              variant="chip"
              state="unavailable"
              label="Structural comparison unavailable"
              reason={comparisonError ?? "The structural comparison could not be loaded."}
            />
          </div>
        )}
        {comparisonStatus === "ready" && comparison && (
          <>
            <PairedDelta
              title="What moved between A and B"
              rows={comparison.rows}
              summaryAxes={comparison.summaryAxes}
              findings={comparison.findings}
              aLabel={runA?.label ?? shortId(idA)}
              bLabel={runB?.label ?? shortId(idB)}
            />
            {comparison.rows.length === 0 && (
              <p className="compare-primary-state">
                No changed dimensions were reported. The eight axes above state what the comparison examined.
              </p>
            )}
            {comparison.privacyLimited && (
              <p className="compare-privacy-note">
                Privacy-limited comparison: one or more dimensions were compared through retained hashes or counts rather than full content.
              </p>
            )}
            <ControlledTestOffer
              plan={changePlan}
              result={changeResult}
              phase={changePhase}
              error={changeError}
              onPreview={() => void previewControlledTest()}
              onRun={() => void executeControlledTest()}
            />
          </>
        )}
      </section>

      {tokenDrilldownOpen && inspectorOpen && (
        <aside className="instrument compare-inspector" aria-labelledby="compare-inspector-title">
          <header className="instrument-head compact">
            <div>
              <span className="eyebrow">ALIGNED POSITION</span>
              <h2 id="compare-inspector-title">A/B inspector</h2>
            </div>
            <strong className="compare-position">{selectedPosition}</strong>
          </header>

          <section className="compare-readout is-a">
            <header><span>A</span><a href={runA ? `#/runs/${encodeURIComponent(runA.id)}/diagnostics/generation` : "#/diagnostics"}>{runA ? shortId(runA.id) : "—"}</a></header>
            <strong>{tokenA?.text || "∅"}</strong>
            <dl>
              <div><dt>Confidence</dt><dd>{tokenA?.confidence?.toFixed(4) ?? "—"}</dd></div>
              <div><dt>Top-k entropy</dt><dd>{tokenA ? `${tokenA.entropy.toFixed(4)} bits` : "—"}</dd></div>
              <div><dt>Band</dt><dd>{tokenA?.band?.toUpperCase() ?? "—"}</dd></div>
            </dl>
          </section>

          <section className="compare-readout is-b">
            <header><span>B</span><a href={runB ? `#/runs/${encodeURIComponent(runB.id)}/diagnostics/generation` : "#/diagnostics"}>{runB ? shortId(runB.id) : "—"}</a></header>
            <strong>{tokenB?.text || "∅"}</strong>
            <dl>
              <div><dt>Confidence</dt><dd>{tokenB?.confidence?.toFixed(4) ?? "—"}</dd></div>
              <div><dt>Top-k entropy</dt><dd>{tokenB ? `${tokenB.entropy.toFixed(4)} bits` : "—"}</dd></div>
              <div><dt>Band</dt><dd>{tokenB?.band?.toUpperCase() ?? "—"}</dd></div>
            </dl>
          </section>

          <section className="compare-deltas">
            <div><span>IDENTITY</span><strong>{tokenIdentity(tokenA, tokenB)}</strong></div>
            <div><span>CONFIDENCE B − A</span><strong>{tokenA && tokenB ? signed((tokenB.confidence ?? 0) - (tokenA.confidence ?? 0)) : "—"}</strong></div>
            <div><span>ENTROPY B − A</span><strong>{tokenA && tokenB ? `${signed(tokenB.entropy - tokenA.entropy)} bits` : "—"}</strong></div>
          </section>
        </aside>
      )}

      {tokenDrilldownOpen && (
        <>
          <section className="instrument compare-plot-panel" aria-labelledby="compare-plot-title">
            <header className="instrument-head compact">
              <div>
                <span className="eyebrow">TOKENS</span>
                <h2 id="compare-plot-title">Confidence comparison</h2>
              </div>
              <div className="compare-legend"><span className="is-a">A</span><span className="is-b">B</span></div>
            </header>
            <div className="compare-plot-wrap">
              <ConfidenceComparison
                a={runA?.tokens ?? []}
                b={runB?.tokens ?? []}
                selectedA={selectedAIndex}
                selectedB={selectedBIndex}
              />
              <div className="compare-axis"><span>1</span><span>{Math.max(1, Math.ceil(maxTokens / 2))}</span><span>{maxTokens || "—"}</span></div>
            </div>
          </section>

          <section className="instrument compare-relation" aria-labelledby="compare-relation-title">
            <header className="instrument-head compact">
              <div>
                <span className="eyebrow">RUNS</span>
                <h2 id="compare-relation-title">Pair details</h2>
              </div>
              <div className="compare-metrics">
                <span><b>MATCHED TOKENS</b>{alignment.matched} / {maxTokens}</span>
                <span><b>FIRST HUNK</b>{firstHunk}</span>
                <span><b>RELATION</b>{relation}</span>
              </div>
            </header>
            <div className="compare-pair">
              <div><b>A</b><span>{runA ? shortId(runA.id) : "—"}</span><small>{runA?.duration ?? "—"}</small></div>
              <i>{relation}</i>
              <div><b>B</b><span>{runB ? shortId(runB.id) : "—"}</span><small>{runB?.duration ?? "—"}</small></div>
            </div>
            <dl className="compare-pair-meta">
              <div><dt>A model</dt><dd>{runA?.model ?? "—"}</dd></div>
              <div><dt>B model</dt><dd>{runB?.model ?? "—"}</dd></div>
              <div><dt>A flags</dt><dd>{runA?.flags?.join(", ") || "—"}</dd></div>
              <div><dt>B flags</dt><dd>{runB?.flags?.join(", ") || "—"}</dd></div>
            </dl>
          </section>
        </>
      )}
    </>
  );
}

function CompareMatrix({
  initialExperimentId,
  rawExperimentQuery,
  experimentRouteBase = "#/compare/matrix",
}: Pick<CompareProps, "initialExperimentId" | "rawExperimentQuery" | "experimentRouteBase">) {
  return (
    <div className="compare-matrix-mode">
      <section className="instrument compare-matrix-intro" aria-labelledby="compare-matrix-title">
        <header className="instrument-head compare-head">
          <div>
            <span className="eyebrow">A / B OVER A RECORDED SUITE</span>
            <h1 id="compare-matrix-title">Experiment matrix</h1>
          </div>
          <div className="compare-pickers"><CompareModePicker mode="matrix" /></div>
        </header>
        <p>
          Matrix results compare recorded baseline and variant outcomes. An unavailable coordinate is not a failed assertion.
        </p>
      </section>
      <Experiments
        id={initialExperimentId}
        rawQuery={rawExperimentQuery}
        routeBase={experimentRouteBase}
        presentation="compare"
      />
    </div>
  );
}

export function Compare({
  mode = "runs",
  runtime,
  initialA,
  initialB,
  inspectorOpen,
  initialExperimentId,
  rawExperimentQuery,
  experimentRouteBase,
}: CompareProps) {
  return mode === "matrix"
    ? <CompareMatrix
        initialExperimentId={initialExperimentId}
        rawExperimentQuery={rawExperimentQuery}
        experimentRouteBase={experimentRouteBase}
      />
    : <RunComparison runtime={runtime} initialA={initialA} initialB={initialB} inspectorOpen={inspectorOpen} />;
}
