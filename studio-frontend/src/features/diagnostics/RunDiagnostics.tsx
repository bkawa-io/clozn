import { useEffect, useMemo, useState, type ReactNode } from "react";
import { EvidenceMark } from "../../components/EvidenceMark";
import { SpanWaterfall } from "../../components/SpanWaterfall";
import { loadRunInspection, loadRunPerformance } from "../../data/api";
import type {
  ObservatoryData,
  RunPerformance as RunPerformanceData,
  RuntimeState,
  TokenReading,
} from "../../data/types";
import { SlotHost } from "../../components/SlotHost";
import { ContextReceipt } from "../lens/ContextReceipt";
import { EvidenceLanes } from "../lens/EvidenceLanes";
import { ReceivedContext } from "../lens/ReceivedContext";
import { RunEventRail, type RunEventRailEvent } from "../lens/RunEventRail";
import { RunWorkspaceHeader } from "../lens/RunWorkspaceHeader";
import { TimeMachine } from "../lens/TimeMachine";
import { aggregateSources, buildResponseClaims } from "../lens/analysis";
import "./RunDiagnostics.css";

export const DIAGNOSTIC_SECTIONS = [
  { id: "overview", label: "Overview" },
  { id: "investigate", label: "Investigate" },
  { id: "delivery", label: "Prompt delivery" },
  { id: "rendered", label: "Rendered prompt" },
  { id: "context", label: "Context & sources" },
  { id: "influence", label: "Source influence" },
  { id: "generation", label: "Generation" },
  { id: "claims", label: "Claims & evaluations" },
  { id: "runtime", label: "Runtime & performance" },
  { id: "events", label: "Events" },
  { id: "lineage", label: "Lineage" },
  { id: "timemachine", label: "Time machine" },
  { id: "raw", label: "Raw artifacts" },
] as const;

export type DiagnosticSectionId = (typeof DIAGNOSTIC_SECTIONS)[number]["id"];

interface RunDiagnosticsProps {
  runtime: RuntimeState;
  initialRunId?: string;
  initialView?: string;
  sessionId?: string;
}

export type PerformanceState =
  | { status: "idle" | "loading" | "error" }
  | { status: "ready"; data: RunPerformanceData };

function diagnosticSection(value?: string): DiagnosticSectionId {
  return DIAGNOSTIC_SECTIONS.some((section) => section.id === value)
    ? value as DiagnosticSectionId
    : "overview";
}

function sessionRun(runtime: RuntimeState, sessionId?: string) {
  return sessionId ? runtime.runs.find((run) => run.sessionKey === sessionId)?.id : undefined;
}

function meaningfulTokens(tokens: readonly TokenReading[]) {
  return tokens.filter((token) => token.text).length;
}

function readableValue(value: unknown) {
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3);
  if (typeof value === "string" || typeof value === "boolean") return String(value);
  if (value == null) return "Not recorded";
  try { return JSON.stringify(value); } catch { return "Unavailable"; }
}

function milliseconds(value?: number) {
  if (value == null) return "Not recorded";
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

function RuntimeMetric({ label, value, source }: { label: string; value: string; source?: string }) {
  return (
    <article>
      <span>{label}</span>
      <strong>{value}</strong>
      {source && <small>{source}</small>}
    </article>
  );
}

/** One performance request belongs to the timing instrument, not to the whole run reader. Both the
 * retained Diagnostics compatibility surface and S2's Timing section call this hook, so their request
 * handling cannot drift into two subtly different "no trace" stories. */
export function useRunPerformance(runId: string): PerformanceState {
  const [performance, setPerformance] = useState<PerformanceState>({ status: "idle" });

  useEffect(() => {
    if (!runId) {
      setPerformance({ status: "idle" });
      return;
    }
    const controller = new AbortController();
    setPerformance({ status: "loading" });
    void loadRunPerformance(runId, controller.signal).then((result) => {
      if (!controller.signal.aborted) setPerformance({ status: "ready", data: result });
    }).catch(() => {
      if (!controller.signal.aborted) setPerformance({ status: "error" });
    });
    return () => controller.abort();
  }, [runId]);

  return performance;
}

export function RuntimeReport({ performance }: { performance: PerformanceState }) {
  if (performance.status !== "ready") {
    return performance.status === "error"
      ? (
        <EvidenceMark
          variant="chip"
          state="unavailable"
          label="Performance artifact unavailable"
          reason="The recorded performance trace request failed for this run."
        />
      )
      : <p className="diagnostics-empty" role="status">Loading recorded performance…</p>;
  }

  const data = performance.data;
  const findings = [
    ...(data.diagnosis?.findings ?? []),
    ...(data.diagnosis?.cutoff ? [data.diagnosis.cutoff] : []),
    ...(data.diagnosis?.auxiliary ? [data.diagnosis.auxiliary] : []),
  ].filter((finding, index, all) => all.findIndex((candidate) => candidate.id === finding.id) === index);
  const rules = data.rules;

  return (
    <div className="diagnostics-runtime-report">
      <section className="diagnostics-runtime-metrics" aria-label="Runtime summary">
        <RuntimeMetric label="Total duration" value={milliseconds(data.totalDuration?.value)} source={data.totalDuration?.source} />
        <RuntimeMetric label="Generation" value={milliseconds(data.generationDuration?.value)} source={data.generationDuration?.source} />
        <RuntimeMetric label="Throughput" value={data.throughput ? `${data.throughput.value.toFixed(1)} tok/s` : "Not recorded"} source={data.throughput?.kind.replaceAll("_", " ")} />
        <RuntimeMetric label="Prompt tokens" value={readableValue(data.promptTokens?.value)} source={data.promptTokens?.source} />
        <RuntimeMetric label="Output tokens" value={readableValue(data.generatedTokens?.value)} source={data.generatedTokens?.source} />
        <RuntimeMetric label="Context window" value={readableValue(data.contextWindowTokens?.value)} source={data.contextWindowTokens?.source} />
        <RuntimeMetric label="Device" value={data.device ?? "Not recorded"} source={data.gpuLayers == null ? undefined : `${data.gpuLayers} GPU layers`} />
        <RuntimeMetric label="Finish" value={data.finishReason?.replaceAll("_", " ") ?? "Not recorded"} source={data.samplerMode} />
      </section>

      <section className="diagnostics-runtime-block diagnostics-runtime-waterfall">
        {rules ? (
          <SpanWaterfall
            title="Timing breakdown"
            phases={rules.phases}
            aggregation={rules.aggregation}
          />
        ) : (
          <EvidenceMark
            variant="chip"
            state="not_measured"
            label="Timing phases not recorded"
            reason="This performance artifact has no clock-domain timing trace."
          />
        )}
      </section>

      <section className="diagnostics-runtime-block">
        <header><div><span className="eyebrow">OBSERVED EVIDENCE</span><h3>Run findings</h3></div><strong>{findings.length} findings</strong></header>
        {data.diagnosis?.summary && <p className="diagnostics-runtime-summary">{data.diagnosis.summary}</p>}
        <div className="diagnostics-runtime-findings">
          {findings.map((finding) => (
            <article className={`status-${finding.status.replaceAll("_", "-")}`} key={finding.id}>
              <header><strong>{finding.id.replaceAll("_", " ")}</strong><span>{finding.status.replaceAll("_", " ")}</span></header>
              <p>{finding.text}</p>
              {finding.evidence.length > 0 && <dl>{finding.evidence.map((item) => <div key={item.path}><dt>{item.path}</dt><dd>{readableValue(item.value)}</dd></div>)}</dl>}
            </article>
          ))}
          {!findings.length && <p className="diagnostics-empty">No diagnosis findings were retained.</p>}
        </div>
      </section>

      <section className="diagnostics-runtime-block">
        <header><div><span className="eyebrow">VERSIONED RULE ENGINE</span><h3>Likely-cause checks</h3></div><strong>{rules?.diagnoses.length ?? 0} rules</strong></header>
        <p className="diagnostics-runtime-boundary">Observed and correlated rules are diagnostic evidence, not proof of causation.</p>
        <div className="diagnostics-runtime-rules">
          {rules?.diagnoses.map((rule) => (
            <article className={`status-${rule.status.replaceAll("_", "-")}`} key={rule.rule}>
              <header>
                <div><strong>{rule.rule.replaceAll("_", " ")}</strong><small>{rule.ruleVersion}</small></div>
                <span>{rule.status.replaceAll("_", " ")}{rule.evidenceState ? ` · ${rule.evidenceState.replaceAll("_", " ")}` : ""}</span>
              </header>
              <p>{rule.likelyCause ?? rule.reason ?? "No further detail was recorded for this rule."}</p>
              {rule.possibleFix && <p className="diagnostics-runtime-fix"><b>Possible fix</b>{rule.possibleFix}</p>}
              {rule.evidence.length > 0 && <dl>{rule.evidence.map((item, index) => <div key={`${item.path}-${index}`}><dt>{item.path}</dt><dd>{readableValue(item.value)}</dd></div>)}</dl>}
            </article>
          ))}
          {!rules && <p className="diagnostics-empty">No performance rule report is available for this run.</p>}
        </div>
      </section>
    </div>
  );
}

/** S2's Timing section deliberately owns only the performance request. A failed trace cannot turn the
 * reader, context receipt, or time machine into a page-wide error because none of them consume it. */
export function RunTimingInstrument({ runId }: { runId: string }) {
  const performance = useRunPerformance(runId);
  return (
    <section className="run-timing-instrument" aria-labelledby="run-timing-title">
      <header className="run-timing-instrument-head">
        <div>
          <span className="eyebrow">TIMING</span>
          <h2 id="run-timing-title">Recorded performance</h2>
        </div>
        <span>READ ONLY</span>
      </header>
      <RuntimeReport performance={performance} />
    </section>
  );
}

function SectionFrame({ title, children }: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="diagnostics-section-frame" aria-label={title}>
      <div className="diagnostics-section-body">{children}</div>
    </section>
  );
}

function AvailabilityCard({ label, state, detail }: { label: string; state: string; detail: string }) {
  const className = state === "available" ? "is-available" : state === "warning" ? "is-warning" : "is-unavailable";
  return (
    <article className={`diagnostics-availability-card ${className}`}>
      <span>{label}</span>
      <strong>{state.replaceAll("_", " ")}</strong>
      <p>{detail}</p>
    </article>
  );
}

function runEvents(run: RuntimeState["runs"][number] | null, data: ObservatoryData | null): RunEventRailEvent[] {
  if (!run) return [];
  return [
    { id: "recorded", label: "Run recorded", kind: "run-start", timestamp: run.createdAt, status: "complete" },
    {
      id: "input",
      label: "Input retained",
      kind: "prompt",
      detail: `${(data?.contextSources ?? data?.sources ?? []).length} recorded spans`,
      status: data ? "complete" : "unavailable",
    },
    {
      id: "generation",
      label: "Generation recorded",
      kind: "generation",
      detail: `${meaningfulTokens(data?.tokens ?? [])} output tokens`,
      status: data ? "complete" : "unavailable",
    },
    ...run.flags.map((flag, index): RunEventRailEvent => ({
      id: `flag-${index}`,
      label: flag.replaceAll("_", " "),
      kind: "warning",
      status: "warning",
    })),
    {
      id: "finish",
      label: "Generation finished",
      kind: "run-finish",
      detail: run.finishReason ?? "Finish reason not recorded",
      status: run.finishReason === "length" ? "warning" : "complete",
    },
  ];
}

export function RunDiagnostics({ runtime, initialRunId, initialView, sessionId }: RunDiagnosticsProps) {
  const requestedRunId = initialRunId || sessionRun(runtime, sessionId) || runtime.runs[0]?.id || "";
  const [runId, setRunId] = useState(requestedRunId);
  const [data, setData] = useState<ObservatoryData | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const view = diagnosticSection(initialView);
  const performance = useRunPerformance(runId);

  useEffect(() => {
    if (requestedRunId && requestedRunId !== runId) setRunId(requestedRunId);
  }, [requestedRunId, runId]);

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    setStatus("loading");
    void loadRunInspection(runId, controller.signal).then((inspection) => {
      if (controller.signal.aborted) return;
      setData(inspection);
      setStatus("idle");
      history.replaceState(null, "", `#/runs/${encodeURIComponent(runId)}/diagnostics/${view}`);
    }).catch(() => {
      if (!controller.signal.aborted) {
        setData(null);
        setStatus("error");
      }
    });
    return () => controller.abort();
  }, [runId]);

  useEffect(() => {
    if (!runId || status !== "idle") return;
    history.replaceState(null, "", `#/runs/${encodeURIComponent(runId)}/diagnostics/${view}`);
  }, [runId, status, view]);

  const selectedRun = runtime.runs.find((run) => run.id === runId) ?? null;
  const contextSources = data?.contextSources ?? data?.sources ?? [];
  const claims = useMemo(() => buildResponseClaims(data?.tokens ?? []), [data]);
  const influence = useMemo(
    () => data?.tokens.length ? aggregateSources(data.tokens, 0, data.tokens.length - 1) : [],
    [data],
  );
  const events = useMemo(() => runEvents(selectedRun, data), [data, selectedRun]);
  const children = runtime.runs.filter((run) => run.parentRunId === runId);
  const parent = selectedRun?.parentRunId
    ? runtime.runs.find((run) => run.id === selectedRun.parentRunId)
    : undefined;

  function content() {
    if (!runId) {
      return <div className="diagnostics-loading">Loading recorded diagnostics…</div>;
    }

    // These two sections own their own fetches, keyed only by run id, and each renders its own typed
    // loading / not-measured / unavailable state. Gating them behind THIS page's run-inspection fetch
    // would replace those honest states with one generic "could not be loaded" that is not even true
    // of them -- so they resolve before the gate below.
    if (view === "investigate") {
      // The whole `lens.evidence` slot as SIBLINGS on purpose: Ask Another Question routes by scrolling
      // to an in-page anchor owned by one of the panels beneath it, so splitting them across sections
      // would break every one of its targets (see data/askAnotherQuestion.ts). Panel `order` puts AAQ first.
      return (
        <SectionFrame title="Investigate">
          <SlotHost slot="lens.evidence" data={{ runId }} />
        </SectionFrame>
      );
    }
    if (view === "timemachine") {
      return (
        <SectionFrame title="Time machine">
          <TimeMachine runId={runId} />
        </SectionFrame>
      );
    }

    if (status === "loading") {
      return <div className="diagnostics-loading">Loading recorded diagnostics…</div>;
    }
    if (status === "error" || !data) {
      return <div className="diagnostics-loading is-error">The selected run diagnostics could not be loaded.</div>;
    }

    switch (view) {
      case "overview":
        return (
          <SectionFrame title="Run overview">
            <div className="diagnostics-overview-grid">
              <AvailabilityCard label="Input" state={data.prompt || contextSources.length ? "available" : "not captured"} detail={`${contextSources.length} recorded input spans`} />
              <AvailabilityCard label="Token trace" state={data.tokens.length ? "available" : "not captured"} detail={`${meaningfulTokens(data.tokens)} meaningful tokens`} />
              <AvailabilityCard label="Source influence" state={data.influenceMethod ? "available" : "not measured"} detail={data.influenceMethod?.caveat ?? "No source intervention artifact is attached to this run."} />
              <AvailabilityCard label="Performance" state={performance.status === "ready" ? "available" : "unavailable"} detail={performance.status === "ready" ? "Recorded runtime and diagnosis facts are available." : "No performance artifact could be loaded."} />
              <AvailabilityCard label="Warnings" state={selectedRun?.flags.length ? "warning" : "available"} detail={selectedRun?.flags.join(" · ") || "No warning flags are attached to this run."} />
              <AvailabilityCard label="Lineage" state={parent || children.length ? "available" : "not recorded"} detail={`${parent ? 1 : 0} parent · ${children.length} children`} />
            </div>
            <div className="diagnostics-overview-copy">
              <section><span>INPUT</span><p>{data.prompt || "No readable prompt was retained."}</p></section>
              <section><span>OUTPUT</span><p>{data.response || "No readable response was retained."}</p></section>
            </div>
          </SectionFrame>
        );
      case "delivery":
        return (
          <SectionFrame title="Prompt delivery">
            <ReceivedContext runId={runId} />
          </SectionFrame>
        );
      case "rendered":
        return (
          <SectionFrame title="Rendered prompt">
            <ContextReceipt runId={runId} defaultDetailedOpen defaultAdvancedOpen />
          </SectionFrame>
        );
      case "context":
        return (
          <SectionFrame title="Context and sources">
            <div className="diagnostics-context-list">
              {contextSources.map((source, index) => (
                <article key={source.id}>
                  <header><b>{index + 1}</b><strong>{source.label ?? source.role ?? "Input span"}</strong><span>{source.kind ?? source.role}</span></header>
                  <p>{source.text}</p>
                  <dl>
                    <div><dt>Source ID</dt><dd>{source.id}</dd></div>
                    <div><dt>Segment</dt><dd>{source.segmentId ?? "Not recorded"}</dd></div>
                    <div><dt>Measurement</dt><dd>{source.measured === true ? source.clearEffect === false ? "Measured below floor" : "Measured" : "Not measured"}</dd></div>
                  </dl>
                </article>
              ))}
              {!contextSources.length && <p className="diagnostics-empty">No readable context sources were recorded.</p>}
            </div>
          </SectionFrame>
        );
      case "influence":
        return (
          <SectionFrame title="Source influence">
            {data.influenceMethod ? (
              <>
                <div className="diagnostics-method">
                  <span>{data.influenceMethod.mode.replaceAll("_", " ")}</span>
                  <p>{data.influenceMethod.caveat}</p>
                  <p><b>Claim boundary:</b> {data.influenceMethod.claimLimit}</p>
                </div>
                <div className="diagnostics-influence-list">
                  {influence.map((source) => (
                    <article className={`effect-${source.effect}`} key={source.sourceId}>
                      <div><strong>{source.label}</strong><small>{source.sourceId}</small></div>
                      <dl>
                        <div><dt>Σ token effect</dt><dd>{source.deltaNats >= 0 ? "+" : ""}{source.deltaNats.toFixed(4)} nats</dd></div>
                        <div><dt>Cleared</dt><dd>{source.clearTokenCount} tokens</dd></div>
                        <div><dt>Below floor</dt><dd>{source.observedTokenCount} tokens</dd></div>
                      </dl>
                    </article>
                  ))}
                  {!influence.length && <p className="diagnostics-empty">The measurement completed without any recorded source links.</p>}
                </div>
              </>
            ) : <p className="diagnostics-empty">Source influence was not measured for this run. Diagnostics does not start measurements.</p>}
          </SectionFrame>
        );
      case "generation":
        return (
          <SectionFrame title="Generation">
            <p className="diagnostics-output-text">{data.response || "No readable response was retained."}</p>
            <EvidenceLanes
              tokens={data.tokens}
              selectedToken={0}
              sourceAvailability={data.influenceMethod
                ? { available: true }
                : { available: false, reason: "Source influence was not measured for this run." }}
              semanticEventsAvailability={{ available: false, reason: "No semantic token events were captured." }}
              finish={{
                reason: selectedRun?.finishReason,
                truncated: selectedRun?.finishReason === "length" || selectedRun?.flags.includes("truncated"),
              }}
            />
          </SectionFrame>
        );
      case "claims":
        return (
          <SectionFrame title="Claims and evaluations">
            <p className="diagnostics-boundary-note">Claim boundaries are derived for navigation. No external evaluator result is implied.</p>
            <div className="diagnostics-claims-list">
              {claims.map((claim, index) => (
                <article key={`${claim.start}-${claim.end}`}>
                  <header><b>C{index + 1}</b><span>{claim.start + 1}–{claim.end + 1}</span></header>
                  <p>{claim.text}</p>
                  <dl>
                    <div><dt>Mean confidence</dt><dd>{claim.meanConfidence == null ? "Not recorded" : `${Math.round(claim.meanConfidence * 100)}%`}</dd></div>
                    <div><dt>Shaky tokens</dt><dd>{claim.shakyCount}</dd></div>
                    <div><dt>Source-linked</dt><dd>{claim.linkedCount} / {claim.tokenCount}</dd></div>
                  </dl>
                </article>
              ))}
              {!claims.length && <p className="diagnostics-empty">No claim boundaries could be derived from the retained output.</p>}
            </div>
          </SectionFrame>
        );
      case "runtime":
        return (
          <SectionFrame title="Runtime and performance">
            <RuntimeReport performance={performance} />
          </SectionFrame>
        );
      case "events":
        return (
          <SectionFrame title="Run events">
            <RunEventRail events={events} ariaLabel="Recorded run events" />
          </SectionFrame>
        );
      case "lineage":
        return (
          <SectionFrame title="Run lineage">
            <div className="diagnostics-lineage">
              {parent && <a href={`#/runs/${encodeURIComponent(parent.id)}/diagnostics/overview`}><span>Parent</span><strong>{parent.label}</strong><small>{parent.id}</small></a>}
              <article className="is-current"><span>Current</span><strong>{selectedRun?.label ?? data.label}</strong><small>{runId}</small></article>
              {children.map((child) => <a href={`#/runs/${encodeURIComponent(child.id)}/diagnostics/overview`} key={child.id}><span>Child</span><strong>{child.label}</strong><small>{child.id}</small></a>)}
            </div>
          </SectionFrame>
        );
      case "raw":
        return (
          <SectionFrame title="Raw artifacts">
            <pre className="diagnostics-raw">{JSON.stringify({ inspection: data, performance: performance.status === "ready" ? performance.data : null }, null, 2)}</pre>
          </SectionFrame>
        );
    }
  }

  return (
    <section className="run-diagnostics-page" aria-label="Run Diagnostics">
      <div className="run-diagnostics-layout">
        <nav className="run-diagnostics-nav" aria-label="Diagnostic sections">
          {DIAGNOSTIC_SECTIONS.map((section) => (
            <a
              className={section.id === view ? "is-active" : ""}
              aria-current={section.id === view ? "page" : undefined}
              href={runId ? `#/runs/${encodeURIComponent(runId)}/diagnostics/${section.id}` : "#/diagnostics"}
              key={section.id}
            >
              <strong>{section.label}</strong>
            </a>
          ))}
        </nav>
        <main className="run-diagnostics-main">{content()}</main>
      </div>
      <RunWorkspaceHeader
        run={selectedRun}
        performance={performance.status === "ready" ? performance.data : null}
        active="diagnostics"
      />
    </section>
  );
}
