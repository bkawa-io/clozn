import { Fragment, useEffect, useMemo, useState } from "react";
import { loadRunInspection } from "../../data/api";
import type { ObservatoryData, RuntimeState, TokenReading } from "../../data/types";
import { alignTokens, type TokenAlignment } from "./alignment";

interface CompareProps {
  runtime: RuntimeState;
  initialA?: string;
  initialB?: string;
  inspectorOpen: boolean;
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

export function Compare({ runtime, initialA, initialB, inspectorOpen }: CompareProps) {
  const [idA, setIdA] = useState(initialA ?? "");
  const [idB, setIdB] = useState(initialB ?? "");
  const [runA, setRunA] = useState<ObservatoryData | null>(null);
  const [runB, setRunB] = useState<ObservatoryData | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [selectedColumn, setSelectedColumn] = useState(0);

  useEffect(() => {
    if (!runtime.runs.length) return;
    setIdA((current) => current || initialA || runtime.runs[1]?.id || runtime.runs[0].id);
    setIdB((current) => current || initialB || runtime.runs[0].id);
  }, [initialA, initialB, runtime.runs]);

  useEffect(() => {
    if (!idA || !idB) return;
    const controller = new AbortController();
    setStatus("loading");
    void Promise.all([
      loadRunInspection(idA, controller.signal),
      loadRunInspection(idB, controller.signal),
    ]).then(([nextA, nextB]) => {
      const nextAlignment = alignTokens(nextA.tokens, nextB.tokens);
      setRunA(nextA);
      setRunB(nextB);
      setSelectedColumn(nextAlignment.firstChangedColumn < 0 ? 0 : nextAlignment.firstChangedColumn);
      setStatus("idle");
      history.replaceState(null, "", `#/compare/${encodeURIComponent(idA)}/${encodeURIComponent(idB)}`);
    }).catch(() => {
      if (!controller.signal.aborted) setStatus("error");
    });
    return () => controller.abort();
  }, [idA, idB]);

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

  return (
    <>
      <section className="instrument compare-hero" aria-labelledby="compare-title">
        <header className="instrument-head compare-head">
          <div>
            <span className="eyebrow">RUN COMPARISON</span>
            <h1 id="compare-title">Token divergence</h1>
          </div>
          <div className="compare-metrics">
            <span><b>MATCHED TOKENS</b>{alignment.matched} / {maxTokens}</span>
            <span><b>FIRST HUNK</b>{firstHunk}</span>
            <span><b>RELATION</b>{relation}</span>
          </div>
          <div className="compare-pickers">
            <label>
              <span>A RUN</span>
              <select value={idA} onChange={(event) => setIdA(event.target.value)} disabled={status === "loading"}>
                <option value="">Select run A</option>
                {runtime.runs.map((run) => <option key={`a-${run.id}`} value={run.id}>{run.label}</option>)}
              </select>
            </label>
            <label>
              <span>B RUN</span>
              <select value={idB} onChange={(event) => setIdB(event.target.value)} disabled={status === "loading"}>
                <option value="">Select run B</option>
                {runtime.runs.map((run) => <option key={`b-${run.id}`} value={run.id}>{run.label}</option>)}
              </select>
            </label>
          </div>
        </header>

        <div className="compare-stage">
          {status === "error" && <div className="compare-state is-error">RUN LOAD FAILED</div>}
          {!runA || !runB ? (
            <div className="compare-state">{status === "loading" ? "LOADING RUNS" : "SELECT TWO RUNS"}</div>
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
      </section>

      {inspectorOpen && (
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
  );
}
