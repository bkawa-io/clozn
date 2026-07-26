import {
  Fragment,
  useEffect,
  useMemo,
  useState,
  type KeyboardEvent,
  type MouseEvent,
} from "react";
import { loadRunConcepts, loadRunInspection, measureRunInfluenceMap } from "../../data/api";
import type {
  ObservatoryData,
  RunConcepts,
  RuntimeState,
  SourceReading,
  TokenReading,
} from "../../data/types";
import { ConfidencePlot } from "../observatory/ConfidencePlot";
import {
  aggregateSources,
  buildResponseClaims,
  influenceSplit,
  summarizeRange,
  weakestTokenInRange,
} from "./analysis";

interface LensProps {
  runtime: RuntimeState;
  initialRunId?: string;
  inspectorOpen: boolean;
}

type LensMode = "sources" | "shakiness" | "influences" | "concepts";
type ConceptState =
  | { status: "idle" | "loading" }
  | { status: "done"; data: RunConcepts };
type TokenRange = { start: number; end: number };

const modes: Array<{ id: LensMode; label: string }> = [
  { id: "sources", label: "SOURCES" },
  { id: "shakiness", label: "SHAKINESS" },
  { id: "influences", label: "INFLUENCES" },
  { id: "concepts", label: "CONCEPTS" },
];

function shortId(id: string) {
  return id.slice(-6);
}

function initialToken(data: ObservatoryData) {
  if (!data.tokens.length) return 0;
  let weakest = 0;
  for (let index = 1; index < data.tokens.length; index += 1) {
    if ((data.tokens[index].confidence ?? 1) < (data.tokens[weakest].confidence ?? 1)) weakest = index;
  }
  return weakest;
}

function dominantInfluence(token?: TokenReading) {
  return [...(token?.sources ?? [])].sort(
    (a, b) => Math.abs(b.deltaNats) - Math.abs(a.deltaNats),
  )[0];
}

function readableToken(text: string) {
  return text
    .replace(/\r\n|\r|\n/g, "")
    .replaceAll(" ", "\u00a0")
    .replaceAll("\t", "\u00a0\u00a0\u00a0\u00a0");
}

function tokenBreaks(text: string) {
  return text.match(/\r\n|\r|\n/g)?.length ?? 0;
}

function nextIndex(indexes: number[], current: number, direction: 1 | -1) {
  if (!indexes.length) return current;
  if (direction === 1) return indexes.find((index) => index > current) ?? indexes[0];
  return [...indexes].reverse().find((index) => index < current) ?? indexes[indexes.length - 1];
}

function bandCounts(tokens: TokenReading[]) {
  return tokens.reduce(
    (counts, token) => {
      if (token.band) counts[token.band] += 1;
      return counts;
    },
    { strong: 0, okay: 0, shaky: 0 },
  );
}

function SourceCard({
  source,
  selected,
  linkedTokens,
  onSelect,
}: {
  source: SourceReading;
  selected: boolean;
  linkedTokens: number;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      className={`lens-source ${selected ? "is-selected" : ""}`}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <span>{source.role || "CONTEXT"}</span>
      <strong>{source.text}</strong>
      <small>{linkedTokens} {linkedTokens === 1 ? "TOKEN" : "TOKENS"}</small>
    </button>
  );
}

export function Lens({ runtime, initialRunId, inspectorOpen }: LensProps) {
  const [runId, setRunId] = useState(initialRunId ?? "");
  const [data, setData] = useState<ObservatoryData | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [mode, setMode] = useState<LensMode>("sources");
  const [selectedToken, setSelectedToken] = useState(0);
  const [selectedRange, setSelectedRange] = useState<TokenRange | null>(null);
  const [rangeAnchor, setRangeAnchor] = useState(0);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [concepts, setConcepts] = useState<ConceptState>({ status: "idle" });
  const [sourceStatus, setSourceStatus] = useState<"idle" | "measuring" | "error">("idle");

  useEffect(() => {
    if (!runtime.runs.length) return;
    setRunId((current) => current || initialRunId || runtime.runs[0].id);
  }, [initialRunId, runtime.runs]);

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    setStatus("loading");
    setSelectedSourceId(null);
    setSelectedRange(null);
    setConcepts({ status: "idle" });
    setSourceStatus("idle");
    void loadRunInspection(runId, controller.signal).then((inspection) => {
      if (controller.signal.aborted) return;
      const nextToken = initialToken(inspection);
      setData(inspection);
      setSelectedToken(nextToken);
      setRangeAnchor(nextToken);
      setStatus("idle");
      history.replaceState(null, "", `#/runs/${encodeURIComponent(runId)}`);
    }).catch(() => {
      if (!controller.signal.aborted) setStatus("error");
    });
    return () => controller.abort();
  }, [runId]);

  useEffect(() => {
    if (mode !== "concepts" || !data || concepts.status !== "idle") return;
    const controller = new AbortController();
    setConcepts({ status: "loading" });
    void loadRunConcepts(data.id, undefined, controller.signal).then((result) => {
      if (!controller.signal.aborted) setConcepts({ status: "done", data: result });
    }).catch(() => {
      if (!controller.signal.aborted) {
        setConcepts({
          status: "done",
          data: {
            available: false,
            reason: "J-lens unavailable",
            availableLayers: [],
            tokens: [],
            readouts: [],
          },
        });
      }
    });
    return () => controller.abort();
  }, [data, mode]);

  const tokens = data?.tokens ?? [];
  const selected = tokens[selectedToken];
  const counts = bandCounts(tokens);
  const claims = useMemo(() => buildResponseClaims(tokens), [tokens]);
  const rangeSummary = selectedRange
    ? summarizeRange(tokens, selectedRange.start, selectedRange.end)
    : null;
  const selectedClaimIndex = selectedRange
    ? claims.findIndex(
      (claim) => claim.start === selectedRange.start && claim.end === selectedRange.end,
    )
    : -1;
  const activeClaimIndex = claims.findIndex(
    (claim) => selectedToken >= claim.start && selectedToken <= claim.end,
  );
  useEffect(() => {
    if (activeClaimIndex < 0) return;
    const frame = requestAnimationFrame(() => {
      document.querySelector<HTMLElement>(`[data-lens-claim="${activeClaimIndex}"]`)
        ?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
    return () => cancelAnimationFrame(frame);
  }, [activeClaimIndex]);
  const rangeSources = useMemo(
    () => selectedRange
      ? aggregateSources(tokens, selectedRange.start, selectedRange.end)
      : [],
    [selectedRange, tokens],
  );
  const answerSources = useMemo(
    () => aggregateSources(tokens, 0, Math.max(0, tokens.length - 1)),
    [tokens],
  );
  const mostLinkedSource = [...answerSources].sort(
    (a, b) => b.tokenCount - a.tokenCount || Math.abs(b.deltaNats) - Math.abs(a.deltaNats),
  )[0];
  const directionCounts = useMemo(() => influenceSplit(tokens), [tokens]);
  const recordedTokenCount = tokens.filter((token) => token.text).length;
  const linkedTokenCount = tokens.filter((token) => token.text && token.sources?.length).length;
  const lowestClaimIndex = claims.reduce((lowest, claim, index) => {
    if (claim.meanConfidence == null) return lowest;
    if (lowest < 0 || (claims[lowest].meanConfidence ?? Infinity) > claim.meanConfidence) return index;
    return lowest;
  }, -1);
  const linkedIndexes = useMemo(
    () => selectedSourceId
      ? tokens.flatMap((token, index) =>
        token.text && token.sources?.some((source) => source.sourceId === selectedSourceId) ? [index] : [])
      : [],
    [selectedSourceId, tokens],
  );
  const linkedSet = new Set(linkedIndexes);
  const selectedSource = data?.sources.find((source) => source.id === selectedSourceId);
  const sourceCounts = useMemo(() => {
    const output = new Map<string, number>();
    for (const token of tokens) {
      if (!token.text) continue;
      for (const source of token.sources ?? []) {
        output.set(source.sourceId, (output.get(source.sourceId) ?? 0) + 1);
      }
    }
    return output;
  }, [tokens]);
  const markedIndexes = tokens.flatMap((token, index) => {
    if (!token.text) return [];
    if (mode === "shakiness") return token.band === "okay" || token.band === "shaky" ? [index] : [];
    if (mode === "concepts") {
      const conceptData = concepts.status === "done" ? concepts.data : null;
      return conceptData?.available && conceptData.readouts[index]?.length ? [index] : [];
    }
    return token.sources?.length ? [index] : [];
  });
  const strongestSourceIndex = tokens.reduce((best, token, index) => {
    const value = token.text ? Math.abs(dominantInfluence(token)?.deltaNats ?? 0) : 0;
    const bestValue = Math.abs(dominantInfluence(tokens[best])?.deltaNats ?? 0);
    return value > bestValue ? index : best;
  }, 0);
  const conceptData = concepts.status === "done" ? concepts.data : null;
  const conceptsAligned = Boolean(
    data?.response
    && conceptData?.available
    && conceptData.tokens.length === tokens.length
    && conceptData.tokens.join("") === data.response,
  );
  const selectedConcepts = conceptsAligned ? conceptData?.readouts[selectedToken] ?? [] : [];

  function selectToken(index: number, extend = false) {
    setSelectedSourceId(null);
    if (extend) {
      setSelectedRange({
        start: Math.min(rangeAnchor, index),
        end: Math.max(rangeAnchor, index),
      });
    } else {
      setSelectedRange(null);
      setRangeAnchor(index);
    }
    setSelectedToken(index);
  }

  function selectClaim(index: number) {
    const claim = claims[index];
    if (!claim) return;
    setSelectedSourceId(null);
    setSelectedRange({ start: claim.start, end: claim.end });
    setRangeAnchor(claim.start);
    setSelectedToken(weakestTokenInRange(tokens, claim.start, claim.end));
  }

  function selectAdjacentClaim(direction: 1 | -1) {
    if (!claims.length) return;
    const current = activeClaimIndex >= 0 ? activeClaimIndex : 0;
    selectClaim((current + direction + claims.length) % claims.length);
  }

  function handleTokenKeys(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next = index;
    if (event.key === "ArrowRight") next = Math.min(tokens.length - 1, index + 1);
    else if (event.key === "ArrowLeft") next = Math.max(0, index - 1);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tokens.length - 1;
    else return;
    event.preventDefault();
    selectToken(next, event.shiftKey);
    requestAnimationFrame(() => {
      document.querySelector<HTMLButtonElement>(`[data-lens-token="${next}"]`)?.focus();
    });
  }

  async function measureSources() {
    if (!data || sourceStatus === "measuring") return;
    setSourceStatus("measuring");
    try {
      await measureRunInfluenceMap(data.id);
      const inspection = await loadRunInspection(data.id);
      const nextToken = Math.min(selectedToken, Math.max(0, inspection.tokens.length - 1));
      setData(inspection);
      setSelectedToken(nextToken);
      setRangeAnchor(nextToken);
      setSelectedRange(null);
      setSourceStatus("idle");
    } catch {
      setSourceStatus("error");
    }
  }

  return (
    <>
      <section className="instrument lens-context" aria-labelledby="lens-context-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">INPUT</span>
            <h2 id="lens-context-title">Context</h2>
          </div>
          <strong>{data?.sources.length ?? 0} SPANS</strong>
        </header>
        <div className="lens-context-body">
          <section className="lens-prompt">
            <span>PROMPT</span>
            <p>{data?.prompt || "—"}</p>
          </section>
          {data?.sources.length ? (
            <div className="lens-source-list">
              {data.sources.map((source) => (
                <SourceCard
                  key={source.id}
                  source={source}
                  selected={source.id === selectedSourceId}
                  linkedTokens={sourceCounts.get(source.id) ?? 0}
                  onSelect={() => {
                    setMode("sources");
                    setSelectedRange(null);
                    setSelectedSourceId((current) => current === source.id ? null : source.id);
                  }}
                />
              ))}
            </div>
          ) : (
            <div className="lens-source-empty">
              <span>{sourceStatus === "error" ? "SOURCE MEASUREMENT FAILED" : "SOURCE MAP UNAVAILABLE"}</span>
              <button
                type="button"
                disabled={!data || sourceStatus === "measuring"}
                onClick={() => void measureSources()}
              >{sourceStatus === "measuring" ? "MEASURING SOURCES" : "MEASURE SOURCES"}</button>
            </div>
          )}
        </div>
      </section>

      <section className={`instrument lens-reader mode-${mode}`} aria-labelledby="lens-title">
        <header className="instrument-head lens-head">
          <div>
            <span className="eyebrow">RECORDED RUN</span>
            <h1 id="lens-title">Response</h1>
          </div>
          <div className="lens-metrics">
            <span><b>TOKENS</b>{tokens.length}</span>
            <span><b>CLAIMS</b>{claims.length}</span>
            <span><b>LINKED</b>{linkedTokenCount}</span>
            <span><b>SHAKY</b>{counts.shaky}</span>
          </div>
          <label className="lens-run-picker">
            <span>RUN</span>
            <select
              value={runId}
              disabled={status === "loading"}
              onChange={(event) => setRunId(event.target.value)}
            >
              {runtime.runs.map((run) => <option key={run.id} value={run.id}>{run.label}</option>)}
            </select>
          </label>
        </header>

        <nav className="lens-modes" aria-label="Response overlay">
          {modes.map((item) => (
            <button
              type="button"
              className={mode === item.id ? "is-active" : ""}
              aria-pressed={mode === item.id}
              onClick={() => {
                setMode(item.id);
                if (item.id !== "sources") setSelectedSourceId(null);
              }}
              key={item.id}
            >{item.label}</button>
          ))}
        </nav>

        <div className="lens-response-stage">
          {status === "error" && <div className="lens-state is-error">RUN LOAD FAILED</div>}
          {!data ? (
            <div className="lens-state">{status === "loading" ? "LOADING RUN" : "SELECT A RUN"}</div>
          ) : tokens.length ? (
            <>
              {claims.length > 0 && (
                <section className="lens-claims" aria-labelledby="lens-claims-title">
                  <header>
                    <div>
                      <strong id="lens-claims-title">CLAIM GROUPS</strong>
                      <span>BOUNDARY DERIVED</span>
                    </div>
                    <div className="lens-claim-nav">
                      <button type="button" onClick={() => selectAdjacentClaim(-1)}>PREV</button>
                      <button type="button" onClick={() => selectAdjacentClaim(1)}>NEXT</button>
                    </div>
                  </header>
                  <div className="lens-claim-list">
                    {claims.map((claim, index) => (
                      <button
                        type="button"
                        className={[
                          "lens-claim",
                          activeClaimIndex === index ? "is-current" : "",
                          selectedClaimIndex === index ? "is-selected" : "",
                        ].join(" ")}
                        aria-pressed={selectedClaimIndex === index}
                        data-lens-claim={index}
                        onClick={() => selectClaim(index)}
                        key={`${claim.start}-${claim.end}`}
                      >
                        <span>C{index + 1}</span>
                        <strong>{claim.text.replace(/\s+/g, " ")}</strong>
                        <small>
                          {claim.tokenCount} TOK · {claim.shakyCount} SHAKY · {claim.linkedCount} LINKED
                        </small>
                      </button>
                    ))}
                  </div>
                </section>
              )}
              <div className="lens-response-text" role="listbox" aria-label="Response tokens">
                {tokens.map((token, index) => {
                  const influence = dominantInfluence(token);
                  const isSourceMatch = selectedSourceId ? linkedSet.has(index) : false;
                  const isSourceMuted = selectedSourceId ? !isSourceMatch : false;
                  const isSpanSelected = selectedRange
                    ? index >= selectedRange.start && index <= selectedRange.end
                    : false;
                  const hasConcept = conceptsAligned && Boolean(conceptData?.readouts[index]?.length);
                  return (
                    <Fragment key={`${index}-${token.text}`}>
                      <button
                        type="button"
                        role="option"
                        aria-selected={index === selectedToken}
                        tabIndex={index === selectedToken ? 0 : -1}
                        data-lens-token={index}
                        className={[
                          "lens-token",
                          index === selectedToken ? "is-selected" : "",
                          isSpanSelected ? "is-span-selected" : "",
                          token.sources?.length ? "has-source" : "",
                          isSourceMatch ? "is-source-match" : "",
                          isSourceMuted ? "is-source-muted" : "",
                          influence ? `effect-${influence.effect}` : "",
                          `band-${token.band ?? "none"}`,
                          hasConcept ? "has-concept" : "",
                          !token.text && index === tokens.length - 1 ? "is-terminal" : "",
                          !readableToken(token.text) && tokenBreaks(token.text) ? "is-break-only" : "",
                        ].join(" ")}
                        onClick={(event: MouseEvent<HTMLButtonElement>) => selectToken(index, event.shiftKey)}
                        onKeyDown={(event) => handleTokenKeys(event, index)}
                        aria-label={`Token ${index + 1}: ${token.text || "blank"}`}
                      >{readableToken(token.text)}</button>
                      {Array.from({ length: tokenBreaks(token.text) }, (_, breakIndex) => (
                        <br key={`break-${index}-${breakIndex}`} />
                      ))}
                    </Fragment>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="lens-response-plain">{data.response || "NO RECORDED RESPONSE"}</div>
          )}
          {mode === "concepts" && conceptData && !conceptData.available && (
            <div className="lens-concept-state">
              <strong>CONCEPT READOUT UNAVAILABLE</strong>
              <span>{conceptData.reason}</span>
            </div>
          )}
        </div>
      </section>

      {inspectorOpen && (
        <aside className="instrument lens-inspector" aria-labelledby="lens-inspector-title">
          <header className="instrument-head compact">
            <div>
              <span className="eyebrow">
                {selectedSource
                  ? "CONTEXT SPAN"
                  : rangeSummary
                    ? selectedClaimIndex >= 0 ? `CLAIM ${selectedClaimIndex + 1}` : "TOKEN SPAN"
                    : "SELECTION"}
              </span>
              <h2 id="lens-inspector-title">
                {selectedSource ? "Source inspector" : rangeSummary ? "Span inspector" : "Token inspector"}
              </h2>
            </div>
            <strong>
              {selectedSource
                ? `${linkedIndexes.length} TOKENS`
                : rangeSummary ? `${rangeSummary.tokenCount} TOKENS` : `#${selectedToken + 1}`}
            </strong>
          </header>

          {selectedSource ? (
            <div className="lens-inspector-body">
              <section className="lens-selected-source">
                <span>{selectedSource.role || "CONTEXT"}</span>
                <p>{selectedSource.text}</p>
              </section>
              <div className="lens-linked-output">
                <span>LINKED OUTPUT</span>
                <p>{linkedIndexes.map((index) => tokens[index]?.text).join("") || "—"}</p>
              </div>
              <button
                type="button"
                className="lens-inspector-action"
                disabled={!linkedIndexes.length}
                onClick={() => selectToken(linkedIndexes[0])}
              >SELECT FIRST TOKEN</button>
            </div>
          ) : rangeSummary ? (
            <div className="lens-inspector-body">
              <section className="lens-span-readout">
                <span>{selectedClaimIndex >= 0 ? `CLAIM ${selectedClaimIndex + 1}` : "SELECTED SPAN"}</span>
                <p>{rangeSummary.text}</p>
              </section>
              <dl className="lens-token-facts">
                <div><dt>Position</dt><dd>{rangeSummary.start + 1}–{rangeSummary.end + 1}</dd></div>
                <div><dt>Mean confidence</dt><dd>{rangeSummary.meanConfidence?.toFixed(4) ?? "—"}</dd></div>
                <div><dt>Shaky tokens</dt><dd>{rangeSummary.shakyCount}</dd></div>
                <div><dt>Source-linked</dt><dd>{rangeSummary.linkedCount} / {rangeSummary.tokenCount}</dd></div>
              </dl>
              <section className="lens-evidence">
                <header><span>SPAN SOURCES · Σ TOKEN Δ</span><b>{rangeSources.length}</b></header>
                {rangeSources.map((source) => (
                  <button
                    type="button"
                    className={`lens-evidence-row effect-${source.effect}`}
                    onClick={() => {
                      setMode("sources");
                      setSelectedRange(null);
                      setSelectedSourceId(source.sourceId);
                    }}
                    key={source.sourceId}
                  >
                    <strong>{source.label}</strong>
                    <span>{source.tokenCount} TOKENS</span>
                    <output>{source.deltaNats >= 0 ? "+" : ""}{source.deltaNats.toFixed(4)} Σ nats</output>
                  </button>
                ))}
                {!rangeSources.length && <div className="lens-unavailable">UNRESOLVED</div>}
              </section>
              <button
                type="button"
                className="lens-inspector-action"
                onClick={() => selectToken(
                  weakestTokenInRange(tokens, rangeSummary.start, rangeSummary.end),
                )}
              >SELECT LOWEST-CONFIDENCE TOKEN</button>
              {data && (
                <a
                  className="lens-inspector-action"
                  href={`#/runs/${encodeURIComponent(data.id)}/scope?token=${selectedToken}`}
                >OPEN SELECTED TOKEN IN SCOPE</a>
              )}
            </div>
          ) : selected ? (
            <div className="lens-inspector-body">
              <section className="lens-token-readout">
                <strong>{selected.text || "∅"}</strong>
                <span className={`band-chip band-${selected.band ?? "none"}`}>
                  {selected.band?.toUpperCase() ?? "UNBANDED"}
                </span>
              </section>
              <dl className="lens-token-facts">
                <div><dt>Position</dt><dd>{selectedToken + 1} / {tokens.length}</dd></div>
                <div><dt>Confidence</dt><dd>{selected.confidence?.toFixed(4) ?? "—"}</dd></div>
                <div><dt>Top-k entropy</dt><dd>{selected.entropy.toFixed(4)} bits</dd></div>
                <div><dt>Sources</dt><dd>{selected.sources?.length ?? 0}</dd></div>
              </dl>

              {mode === "concepts" ? (
                <section className="lens-evidence">
                  <header><span>J-LENS · RAW LOGIT</span><b>{conceptData?.layer == null ? "—" : `L${conceptData.layer}`}</b></header>
                  {concepts.status === "loading" && <div className="lens-unavailable">READING CONCEPTS</div>}
                  {conceptData?.available && !conceptsAligned && (
                    <div className="lens-unavailable">TOKEN ALIGNMENT UNAVAILABLE</div>
                  )}
                  {selectedConcepts.map((concept) => (
                    <div className="lens-concept-row" key={concept.piece}>
                      <strong>{concept.piece}</strong>
                      <output>{concept.score.toFixed(3)}</output>
                    </div>
                  ))}
                  {conceptData && !conceptData.available && (
                    <div className="lens-unavailable">{conceptData.reason}</div>
                  )}
                </section>
              ) : (
                <section className="lens-evidence">
                  <header><span>{mode === "influences" ? "SIGNED INFLUENCE" : "SOURCES"}</span><b>{selected.sources?.length ?? 0}</b></header>
                  {(selected.sources ?? []).map((source) => (
                    <button
                      type="button"
                      className={`lens-evidence-row effect-${source.effect}`}
                      onClick={() => {
                        setMode("sources");
                        setSelectedSourceId(source.sourceId);
                      }}
                      key={source.sourceId}
                    >
                      <strong>{source.label}</strong>
                      <span>{source.effect.toUpperCase()}</span>
                      <output>{source.deltaNats >= 0 ? "+" : ""}{source.deltaNats.toFixed(4)} nats</output>
                    </button>
                  ))}
                  {!selected.sources?.length && <div className="lens-unavailable">UNRESOLVED</div>}
                </section>
              )}

              {data && (
                <a
                  className="lens-inspector-action"
                  href={`#/runs/${encodeURIComponent(data.id)}/scope?token=${selectedToken}`}
                >OPEN IN SCOPE</a>
              )}
            </div>
          ) : <div className="lens-unavailable">NO TOKEN TRACE</div>}
        </aside>
      )}

      <section className="instrument lens-map" aria-labelledby="lens-map-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">RESPONSE POSITION</span>
            <h2 id="lens-map-title">Confidence map</h2>
          </div>
          <div className="lens-map-legend" aria-label="Plot legend">
            <span className="is-confidence">CONFIDENCE</span>
            <span className="is-entropy">TOP-K ENTROPY</span>
          </div>
          <div className="lens-map-actions">
            <button type="button" disabled={!markedIndexes.length} onClick={() => selectToken(nextIndex(markedIndexes, selectedToken, -1))}>PREV MARK</button>
            <button type="button" disabled={!markedIndexes.length} onClick={() => selectToken(nextIndex(markedIndexes, selectedToken, 1))}>NEXT MARK</button>
            <button
              type="button"
              disabled={!counts.shaky}
              onClick={() => selectToken(nextIndex(
                tokens.flatMap((token, index) => token.band === "shaky" ? [index] : []),
                selectedToken,
                1,
              ))}
            >NEXT SHAKY</button>
            <button type="button" disabled={!tokens.some((token) => token.text && token.sources?.length)} onClick={() => selectToken(strongestSourceIndex)}>MAX LINK</button>
          </div>
        </header>
        <div className="lens-map-plot">
          <div className="lens-map-graph">
            <ConfidencePlot tokens={tokens} selectedToken={selectedToken} />
            <input
              type="range"
              aria-label="Selected response token"
              min="0"
              max={Math.max(0, tokens.length - 1)}
              value={selectedToken}
              disabled={!tokens.length}
              onChange={(event) => selectToken(Number(event.target.value))}
            />
            <div className="lens-map-axis">
              <span>1</span>
              <span>{tokens.length ? Math.ceil(tokens.length / 2) : "—"}</span>
              <span>{tokens.length || "—"}</span>
            </div>
          </div>
          <aside className="lens-patterns" aria-labelledby="lens-patterns-title">
            <header>
              <strong id="lens-patterns-title">ANSWER PATTERNS</strong>
              <span>TRACE DERIVED</span>
            </header>
            <button
              type="button"
              disabled={lowestClaimIndex < 0}
              onClick={() => selectClaim(lowestClaimIndex)}
            >
              <span>LOWEST CLAIM MEAN</span>
              <strong>
                {lowestClaimIndex >= 0
                  ? `C${lowestClaimIndex + 1} · ${claims[lowestClaimIndex].meanConfidence?.toFixed(4)}`
                  : "—"}
              </strong>
            </button>
            <div>
              <span>SOURCE COVERAGE</span>
              <strong>
                {recordedTokenCount
                  ? `${linkedTokenCount} / ${recordedTokenCount} · ${Math.round(linkedTokenCount / recordedTokenCount * 100)}%`
                  : "—"}
              </strong>
            </div>
            <button
              type="button"
              disabled={!mostLinkedSource}
              onClick={() => {
                if (!mostLinkedSource) return;
                setMode("sources");
                setSelectedRange(null);
                setSelectedSourceId(mostLinkedSource.sourceId);
              }}
            >
              <span>MOST LINKED CONTEXT</span>
              <strong>{mostLinkedSource ? `${mostLinkedSource.tokenCount} TOK · ${mostLinkedSource.label}` : "—"}</strong>
            </button>
            <div>
              <span>DOMINANT EFFECTS</span>
              <strong>{directionCounts.supports} SUPPORT · {directionCounts.suppresses} SUPPRESS</strong>
            </div>
          </aside>
        </div>
      </section>
    </>
  );
}
