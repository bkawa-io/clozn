import {
  Fragment,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
} from "react";
import {
  cancelRunInfluenceMapJob,
  loadRunInfluenceMapJob,
  loadRunConcepts,
  loadRunInspection,
  loadRunPerformance,
  startRunInfluenceMapJob,
} from "../../data/api";
import type {
  InfluenceAbsence,
  InfluenceMapJob,
  ObservatoryData,
  RunConcepts,
  RunPerformance as RunPerformanceData,
  RuntimeState,
  TokenReading,
} from "../../data/types";
import { SlotHost } from "../../components/SlotHost";
import { buildResponseClaims, weakestTokenInRange } from "./analysis";
import { EvidenceDeck } from "./EvidenceDeck";
import { LensContextCanvas } from "./LensContextCanvas";
import { LensSelectionInspector } from "./LensSelectionInspector";
import { ContextReceipt } from "./ContextReceipt";
import { ReceivedContext } from "./ReceivedContext";
import { RunPerformance } from "./RunPerformance";
import { RunEventRail, type RunEventRailEvent } from "./RunEventRail";
import { RunFrame } from "./RunFrame";
import { LensRunNavigator } from "./LensRunNavigator";
import { TimeMachine } from "./TimeMachine";

interface LensProps {
  runtime: RuntimeState;
  initialRunId?: string;
  inspectorOpen: boolean;
}

type LensMode = "sources" | "shakiness" | "influences" | "concepts";
type ConceptState =
  | { status: "idle" | "loading" }
  | { status: "done"; data: RunConcepts };
type PerformanceState =
  | { status: "idle" | "loading" | "error" }
  | { status: "ready"; data: RunPerformanceData };
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

function bandCounts(tokens: TokenReading[]) {
  return tokens.reduce(
    (counts, token) => {
      if (token.band) counts[token.band] += 1;
      return counts;
    },
    { strong: 0, okay: 0, shaky: 0 },
  );
}

function waitForSourcePoll(signal: AbortSignal, delayMs = 250) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const timeout = window.setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, delayMs);
    const abort = () => {
      window.clearTimeout(timeout);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", abort, { once: true });
  });
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
  const [sourceAbsence, setSourceAbsence] = useState<InfluenceAbsence | null>(null);
  const [sourceCache, setSourceCache] = useState<"hit" | "miss" | "unknown" | null>(null);
  const [sourceJob, setSourceJob] = useState<InfluenceMapJob | null>(null);
  const sourceRequest = useRef<AbortController | null>(null);
  const sourceJobId = useRef<string | null>(null);
  const [performance, setPerformance] = useState<PerformanceState>({ status: "idle" });
  const [selectedPerformanceFinding, setSelectedPerformanceFinding] = useState("generation");

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
    setSourceAbsence(null);
    setSourceCache(null);
    setSourceJob(null);
    sourceRequest.current?.abort();
    sourceRequest.current = null;
    sourceJobId.current = null;
    setPerformance({ status: "loading" });
    setSelectedPerformanceFinding("generation");
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
    void loadRunPerformance(runId, controller.signal).then((nextPerformance) => {
      if (!controller.signal.aborted) setPerformance({ status: "ready", data: nextPerformance });
    }).catch(() => {
      if (!controller.signal.aborted) setPerformance({ status: "error" });
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
  const contextSources = data?.contextSources ?? data?.sources ?? [];
  const claims = useMemo(() => buildResponseClaims(tokens), [tokens]);
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
  const linkedTokenCount = tokens.filter((token) => token.text && token.sources?.length).length;
  const linkedIndexes = useMemo(
    () => selectedSourceId
      ? tokens.flatMap((token, index) =>
        token.text && (
          token.sources?.some((source) => source.sourceId === selectedSourceId)
          || token.observedSources?.some((source) => source.sourceId === selectedSourceId)
        ) ? [index] : [])
      : [],
    [selectedSourceId, tokens],
  );
  const linkedSet = new Set(linkedIndexes);
  const selectedSource = contextSources.find((source) => source.id === selectedSourceId);
  const conceptData = concepts.status === "done" ? concepts.data : null;
  const conceptsAligned = Boolean(
    data?.response
    && conceptData?.available
    && conceptData.tokens.length === tokens.length
    && conceptData.tokens.join("") === data.response,
  );

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
    const controller = new AbortController();
    sourceRequest.current = controller;
    sourceJobId.current = null;
    setSourceStatus("measuring");
    setSourceAbsence(null);
    setSourceJob(null);
    let job: InfluenceMapJob | null = null;
    try {
      job = await startRunInfluenceMapJob(data.id, controller.signal);
      sourceJobId.current = job.jobId;
      setSourceJob(job);
      while (
        job.state !== "completed"
        && job.state !== "failed"
        && job.state !== "cancelled"
      ) {
        await waitForSourcePoll(controller.signal);
        job = await loadRunInfluenceMapJob(data.id, job.jobId, controller.signal);
        setSourceJob(job);
      }
    } catch (error) {
      if (controller.signal.aborted) {
        setSourceStatus("idle");
        return;
      }
      setSourceAbsence({
        kind: "network_error",
        message: error instanceof Error ? error.message : "source measurement request failed",
      });
      setSourceStatus("error");
      return;
    } finally {
      if (sourceRequest.current === controller) sourceRequest.current = null;
      if (job && sourceJobId.current === job.jobId) sourceJobId.current = null;
    }

    if (job.state === "cancelled") {
      setSourceStatus("idle");
      return;
    }
    if (job.state === "failed") {
      setSourceAbsence({
        kind: "server_error",
        message: job.error?.message ?? "source measurement did not produce available evidence",
      });
      setSourceStatus("error");
      return;
    }
    setSourceCache(job.cached ? "hit" : "miss");
    try {
      const inspection = await loadRunInspection(data.id);
      const nextToken = Math.min(selectedToken, Math.max(0, inspection.tokens.length - 1));
      setData(inspection);
      setSelectedToken(nextToken);
      setRangeAnchor(nextToken);
      setSelectedRange(null);
      setSourceStatus("idle");
    } catch {
      setSourceAbsence({ kind: "network_error", message: "the measurement completed but the run could not be reloaded" });
      setSourceStatus("error");
    }
  }

  async function stopWaitingForSources() {
    if (sourceStatus !== "measuring") return;
    const controller = sourceRequest.current;
    const jobId = sourceJobId.current;
    if (data && jobId) {
      try {
        const job = await cancelRunInfluenceMapJob(data.id, jobId);
        setSourceJob(job);
        if (job.state === "completed") {
          setSourceAbsence({
            kind: "server_error",
            message: "the measurement completed before cancellation was accepted; stopped waiting locally",
          });
        }
      } catch (error) {
        setSourceAbsence({
          kind: "network_error",
          message: `cancellation could not be confirmed; stopped waiting locally and the server job may still finish${
            error instanceof Error ? `: ${error.message}` : ""
          }`,
        });
      }
    } else {
      setSourceAbsence({
        kind: "network_error",
        message: "stopped waiting before a server job ID was available; server cancellation could not be requested",
      });
    }
    controller?.abort();
    if (sourceRequest.current === controller) sourceRequest.current = null;
    if (sourceJobId.current === jobId) sourceJobId.current = null;
    setSourceStatus("idle");
  }

  const selectedRun = runtime.runs.find((run) => run.id === runId) ?? null;
  const finishReason = selectedRun?.finishReason
    ?? (performance.status === "ready" ? performance.data.finishReason : undefined);
  const finishMarker = selectedRun
    ? {
      reason: finishReason,
      truncated: selectedRun.finishReason === "length" || selectedRun.flags.includes("truncated"),
      tokenIndex: tokens.length ? tokens.length - 1 : undefined,
    }
    : null;
  const runEvents: RunEventRailEvent[] = selectedRun
    ? [
      {
        id: "run-recorded",
        label: "Run recorded",
        kind: "run-start",
        timestamp: selectedRun.createdAt,
        status: "complete",
      },
      {
        id: "response-recorded",
        label: "Response recorded",
        kind: "generation",
        detail: `${tokens.filter((token) => token.text).length} output tokens`,
        status: "complete",
      },
      ...(selectedRun.flags.length
        ? [{
          id: "run-warning",
          label: "Run warnings",
          kind: "warning" as const,
          detail: selectedRun.flags.join(" · "),
          status: "warning" as const,
        }]
        : []),
      {
        id: "run-finish",
        label: finishReason ? "Generation finished" : "Run captured",
        kind: "run-finish",
        detail: finishReason ?? "Finish reason unavailable",
        status: finishMarker?.truncated ? "warning" : "complete",
      },
    ]
    : [];

  return (
    <>
      <RunFrame
        run={selectedRun}
        runtime={runtime}
        performance={performance.status === "ready" ? performance.data : null}
        lineage={{
          parent: selectedRun?.parentRunId
            ? {
              id: selectedRun.parentRunId,
              label: runtime.runs.find((run) => run.id === selectedRun.parentRunId)?.label,
              href: `#/runs/${encodeURIComponent(selectedRun.parentRunId)}`,
            }
            : null,
          children: runtime.runs
            .filter((run) => run.parentRunId === runId)
            .map((run) => ({
              id: run.id,
              label: run.label,
              href: `#/runs/${encodeURIComponent(run.id)}`,
            })),
        }}
        title="Selected run"
      />
      <LensRunNavigator
        runs={runtime.runs}
        selectedRunId={runId}
        onSelectRun={(nextRunId) => setRunId(nextRunId)}
      />
      <LensContextCanvas
        data={data}
        selectedSourceId={selectedSourceId}
        onSelectedSourceChange={(sourceId) => {
          setMode("sources");
          setSelectedRange(null);
          setSelectedSourceId(sourceId);
        }}
        deliveryContent={runId ? <ReceivedContext runId={runId} /> : null}
        renderedContent={runId ? (
          <ContextReceipt
            runId={runId}
            defaultDetailedOpen
            defaultAdvancedOpen
          />
        ) : null}
        supplementaryContent={runId ? (
          <details className="lens-context-tools">
            <summary>Investigation tools</summary>
            <SlotHost slot="lens.evidence" data={{ runId }} exclude={["context-receipt"]} />
          </details>
        ) : null}
        sourceMeasurementStatus={sourceStatus}
        sourceMeasurementJob={sourceJob}
        sourceMeasurementCache={sourceCache}
        sourceAbsence={sourceAbsence ?? data?.influenceAbsence}
        onMeasureSources={() => void measureSources()}
        onStopWaitingForSources={() => void stopWaitingForSources()}
      />

      {runId && <TimeMachine runId={runId} />}

      <section className={`instrument lens-reader mode-${mode}`} aria-labelledby="lens-title">
        <header className="lens-output-head">
          <div>
            <span className="eyebrow">GENERATED RESPONSE</span>
            <h2 id="lens-title">Output</h2>
          </div>
          <div className="lens-output-summary" aria-label="Output summary">
            <span>{tokens.length} tokens</span>
            <span>{claims.length} claims</span>
            <span>{linkedTokenCount} linked</span>
            <span>{counts.shaky} shaky</span>
          </div>
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
        <LensSelectionInspector
          selection={selectedSource
            ? { kind: "source", sourceId: selectedSource.id, source: selectedSource }
            : selectedRange
              ? {
                kind: "span",
                start: selectedRange.start,
                end: selectedRange.end,
                claimIndex: selectedClaimIndex >= 0 ? selectedClaimIndex : undefined,
              }
              : selected
                ? { kind: "token", index: selectedToken }
                : null}
          tokens={tokens}
          sources={contextSources}
          influenceAbsence={sourceAbsence ?? data?.influenceAbsence}
          influenceMethod={data?.influenceMethod}
          influenceThresholds={data?.influenceThresholds}
          contextCoverage={data?.contextCoverage}
          tokenTrace={tokens.length
            ? { state: "available", provenance: "recorded" }
            : { state: "not_captured", detail: "This run did not retain a token trace." }}
          sourceEvidence={data?.influenceMethod
            ? { state: "available", provenance: "measured" }
            : sourceStatus === "measuring"
              ? { state: "not_measured", detail: "Source influence measurement is in progress." }
              : { state: "not_measured", detail: "Source influence has not been measured for this run." }}
          onSelectToken={(index) => selectToken(index)}
          onSelectSource={(sourceId) => {
            setMode("sources");
            setSelectedRange(null);
            setSelectedSourceId(sourceId);
          }}
          scopeHref={data ? `#/runs/${encodeURIComponent(data.id)}/scope?token=${selectedToken}` : undefined}
          actions={data && !data.influenceMethod ? (
            <button
              type="button"
              disabled={sourceStatus === "measuring"}
              onClick={() => void measureSources()}
            >{sourceStatus === "measuring" ? "Measuring sources" : "Measure sources"}</button>
          ) : undefined}
          events={<RunEventRail events={runEvents} ariaLabel="Recorded events related to this run" />}
        />
      )}


      <EvidenceDeck
        title="Run evidence"
        defaultHeight={390}
        evidenceLanes={{
          tokens,
          selectedToken,
          selectedRange,
          rangeAnchor,
          onSelectionChange: ({ tokenIndex, range }) => {
            setSelectedSourceId(null);
            setSelectedToken(tokenIndex);
            setSelectedRange(range);
            setRangeAnchor(range?.start ?? tokenIndex);
          },
          sourceAvailability: data?.influenceMethod
            ? { available: true }
            : { available: false, reason: "Source influence was not measured for this response." },
          semanticEventsAvailability: {
            available: false,
            reason: "No semantic event trace is recorded in this run payload.",
          },
          finish: finishMarker,
        }}
        sections={{
          events: {
            content: runEvents.length
              ? <div className="lens-deck-events"><RunEventRail events={runEvents} ariaLabel="Recorded run events" /></div>
              : undefined,
            availability: runEvents.length
              ? { state: "available" }
              : { state: "not_captured", detail: "No recorded run events are available." },
          },
          performance: {
            content: (
              <RunPerformance
                data={performance.status === "ready" ? performance.data : undefined}
                status={performance.status}
                selectedFindingId={selectedPerformanceFinding}
                onSelectFinding={setSelectedPerformanceFinding}
              />
            ),
          },
          lineage: selectedRun?.parentRunId || runtime.runs.some((run) => run.parentRunId === runId)
            ? {
              content: (
                <section className="lens-deck-lineage" aria-label="Run lineage">
                  {selectedRun?.parentRunId && (
                    <a href={`#/runs/${encodeURIComponent(selectedRun.parentRunId)}`}>
                      <span>PARENT</span>
                      <strong>{runtime.runs.find((run) => run.id === selectedRun.parentRunId)?.label ?? selectedRun.parentRunId}</strong>
                    </a>
                  )}
                  {runtime.runs.filter((run) => run.parentRunId === runId).map((run) => (
                    <a href={`#/runs/${encodeURIComponent(run.id)}`} key={run.id}>
                      <span>CHILD</span>
                      <strong>{run.label}</strong>
                    </a>
                  ))}
                </section>
              ),
            }
            : {
              availability: {
                state: "available",
                detail: "This run has no recorded parent, child, retry, or branch relationship.",
              },
            },
        }}
      />

    </>
  );
}
