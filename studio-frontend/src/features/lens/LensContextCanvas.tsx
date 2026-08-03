import { useState, type ReactNode } from "react";
import type {
  LegacyLimits,
  ReceiptRendered,
  ReceiptSegment,
  ReceiptTransformation,
} from "../../data/context-receipt";
import type {
  InfluenceAbsence,
  InfluenceMapJob,
  ObservatoryData,
  SourceReading,
} from "../../data/types";
import { describeAbsence } from "./EvidenceCaveat";
import "./LensContextCanvas.css";

export type LensContextRepresentation = "conversation" | "delivery" | "rendered";
export type LensContextDeliveryStatus = "delivered" | "partial" | "truncated" | "unavailable";

/** Exact rendered text is deliberately opt-in; Lens's ordinary `prompt` must never be presented as it. */
export interface LensContextRenderedPrompt extends ReceiptRendered {
  text?: string;
}

/**
 * Receipt-backed facts for the Delivery and Rendered representations. Every field is optional because
 * the feature must distinguish a missing receipt fact from an empty, explicitly recorded collection.
 */
export interface LensContextDelivery {
  status: LensContextDeliveryStatus;
  detail?: string;
  requested?: readonly ReceiptSegment[];
  delivered?: readonly ReceiptSegment[];
  assembled?: readonly ReceiptSegment[];
  omitted?: readonly ReceiptSegment[];
  transformations?: readonly ReceiptTransformation[];
  rendered?: LensContextRenderedPrompt;
  limits?: LegacyLimits;
}

export type LensSourceMeasurementStatus = "idle" | "measuring" | "error";

export interface LensContextCanvasProps {
  data?: ObservatoryData | null;
  selectedSourceId?: string | null;
  onSelectedSourceChange?: (sourceId: string | null) => void;
  /** Controlled when supplied; otherwise the component starts in Conversation and owns tab state. */
  representation?: LensContextRepresentation;
  onRepresentationChange?: (representation: LensContextRepresentation) => void;
  /** Receipt-backed delivery state. It is never inferred from source maps or coverage. */
  delivery?: LensContextDelivery;
  /** Dedicated receipt UI. When present, it owns the Delivery tab instead of the fallback facts below. */
  deliveryContent?: ReactNode;
  /** Explicit-disclosure content for the Rendered tab; mounted only while that tab is active. */
  renderedContent?: ReactNode;
  /** Non-receipt investigation UI shown outside the three primary representations. */
  supplementaryContent?: ReactNode;
  sourceMeasurementStatus?: LensSourceMeasurementStatus;
  sourceMeasurementJob?: InfluenceMapJob | null;
  sourceMeasurementCache?: "hit" | "miss" | "unknown" | null;
  sourceAbsence?: InfluenceAbsence | null;
  onMeasureSources?: () => void;
  onStopWaitingForSources?: () => void;
  /** @deprecated Use `supplementaryContent`; children are never treated as delivery evidence. */
  children?: ReactNode;
}

const FALLBACK_DELIVERY: LensContextDelivery = {
  status: "unavailable",
  detail: "Lens inspection does not record a delivery receipt for this run.",
};

const REPRESENTATIONS: Array<{ id: LensContextRepresentation; label: string }> = [
  { id: "conversation", label: "Conversation" },
  { id: "delivery", label: "Delivery" },
  { id: "rendered", label: "Rendered" },
];

function sourceLabel(source: SourceReading) {
  return source.label ?? source.role ?? "CONTEXT";
}

function sourceRole(source: SourceReading) {
  return source.role ? source.role.toUpperCase() : "UNKNOWN ROLE";
}

function isRetrievedContext(source: SourceReading) {
  const kind = source.kind?.toLowerCase() ?? "";
  return ["retrieval", "document", "attachment", "file", "memory", "reference", "search_result"]
    .some((candidate) => kind === candidate || kind.startsWith(`${candidate}_`));
}

function contextSources(data?: ObservatoryData | null) {
  return data?.contextSources ?? data?.sources ?? [];
}

function clearedTokenCounts(data?: ObservatoryData | null) {
  const counts = new Map<string, number>();
  for (const token of data?.tokens ?? []) {
    if (!token.text) continue;
    // `sources` contains only cleared-floor links. Below-floor observations are intentionally excluded.
    for (const source of token.sources ?? []) {
      counts.set(source.sourceId, (counts.get(source.sourceId) ?? 0) + 1);
    }
  }
  return counts;
}

function deliveryCopy(delivery: LensContextDelivery) {
  switch (delivery.status) {
    case "delivered":
      return { title: "DELIVERED", detail: delivery.detail ?? "A receipt confirms input reached the model." };
    case "partial":
      return { title: "PARTIALLY DELIVERED", detail: delivery.detail ?? "A receipt shows only part of the input reached the model." };
    case "truncated":
      return { title: "TRUNCATED", detail: delivery.detail ?? "A receipt reports input truncation before generation." };
    case "unavailable":
      return { title: "DELIVERY UNAVAILABLE", detail: delivery.detail ?? "No delivery receipt is available for this run." };
    default: {
      const exhaustive: never = delivery.status;
      return exhaustive;
    }
  }
}

function measurementCopy({
  data,
  status,
  job,
  cache,
  absence,
}: {
  data?: ObservatoryData | null;
  status: LensSourceMeasurementStatus;
  job?: InfluenceMapJob | null;
  cache?: "hit" | "miss" | "unknown" | null;
  absence?: InfluenceAbsence | null;
}) {
  if (status === "measuring") {
    return {
      title: job
        ? `MEASURING CONTEXT SUPPORT · ${job.progress.phase.toUpperCase()} · ${job.progress.completedUnits}/${job.progress.totalUnits}`
        : "STARTING CONTEXT SUPPORT MEASUREMENT",
      detail: "No support claim is available until the measurement completes and the run is reloaded.",
    };
  }
  if (status === "error") {
    const described = describeAbsence(absence ?? {
      kind: "server_error",
      message: "The source measurement did not produce available evidence.",
    });
    return { title: described.title, detail: described.detail };
  }
  if (data?.influenceMethod) {
    return {
      title: cache === "hit"
        ? "MEASURED CONTEXT SUPPORT · CACHED"
        : cache === "miss"
          ? "MEASURED CONTEXT SUPPORT · NEWLY MEASURED"
          : "MEASURED CONTEXT SUPPORT",
    };
  }
  const described = describeAbsence(absence ?? data?.influenceAbsence ?? { kind: "not_measured" });
  return { title: described.title, detail: described.detail };
}

function sourceEvidenceCopy(source: SourceReading, clearedTokens: number) {
  // Do not promote a stale token link unless this source is explicitly known to have been measured.
  if (source.measured === true && clearedTokens > 0) {
    return `${clearedTokens} OUTPUT ${clearedTokens === 1 ? "TOKEN" : "TOKENS"} CLEARED`;
  }
  if (source.measured === true && source.clearEffect === false) return "MEASURED · NO EFFECT CLEARED";
  if (source.measured === true && source.clearEffect === true) return "MEASURED · EFFECT SUMMARY RECORDED";
  if (source.measured === true) return "MEASURED · RESULT NOT RECORDED";
  return "NOT MEASURED";
}

function SourceCard({
  source,
  selected,
  clearedTokens,
  onSelectedSourceChange,
}: {
  source: SourceReading;
  selected: boolean;
  clearedTokens: number;
  onSelectedSourceChange?: (sourceId: string | null) => void;
}) {
  const contents = (
    <>
      <span className="lens-context-canvas-card-label">{sourceLabel(source)}</span>
      <span className="lens-context-canvas-card-role">{sourceRole(source)}</span>
      <strong>{source.text}</strong>
      <small className={source.measured === true && source.clearEffect === false ? "is-below-floor" : undefined}>
        {sourceEvidenceCopy(source, clearedTokens)}
      </small>
    </>
  );
  if (!onSelectedSourceChange) return <article className="lens-context-canvas-card">{contents}</article>;
  return (
    <button
      type="button"
      className={`lens-context-canvas-card ${selected ? "is-selected" : ""}`}
      aria-pressed={selected}
      onClick={() => onSelectedSourceChange(selected ? null : source.id)}
    >
      {contents}
    </button>
  );
}

function SourceGroup({
  title,
  sources,
  selectedSourceId,
  counts,
  onSelectedSourceChange,
  empty,
}: {
  title: string;
  sources: SourceReading[];
  selectedSourceId?: string | null;
  counts: Map<string, number>;
  onSelectedSourceChange?: (sourceId: string | null) => void;
  empty: string;
}) {
  return (
    <section className="lens-context-canvas-source-group">
      <header><h3>{title}</h3><span>{sources.length}</span></header>
      {sources.length ? (
        <div className="lens-context-canvas-card-list">
          {sources.map((source) => (
            <SourceCard
              key={source.id}
              source={source}
              selected={source.id === selectedSourceId}
              clearedTokens={counts.get(source.id) ?? 0}
              onSelectedSourceChange={onSelectedSourceChange}
            />
          ))}
        </div>
      ) : <p className="lens-context-canvas-empty">{empty}</p>}
    </section>
  );
}

function segmentLabel(segment: ReceiptSegment, index: number) {
  return segment.sourceLabel ?? segment.sourceType ?? segment.clientSourceId ?? segment.segmentId ?? `SEGMENT ${index + 1}`;
}

function segmentDetail(segment: ReceiptSegment) {
  const facts = [
    segment.deliveredTokens == null ? undefined : `${segment.deliveredTokens.toLocaleString()} TOK`,
    segment.deliveredBytes == null ? undefined : `${segment.deliveredBytes.toLocaleString()} B`,
    segment.included === true ? "INCLUDED" : segment.included === false ? "OMITTED" : undefined,
  ].filter((value): value is string => Boolean(value));
  return facts.length ? facts.join(" · ") : "METADATA NOT RECORDED";
}

function DeliveryStage({
  title,
  segments,
  empty,
}: {
  title: string;
  segments?: readonly ReceiptSegment[];
  empty: string;
}) {
  return (
    <section className="lens-context-canvas-delivery-stage">
      <header><h3>{title}</h3><span>{segments === undefined ? "UNAVAILABLE" : `${segments.length} RECORDED`}</span></header>
      {segments === undefined ? (
        <p>{empty}</p>
      ) : segments.length === 0 ? (
        <p>NO SEGMENTS RECORDED</p>
      ) : (
        <ol>
          {segments.map((segment, index) => (
            <li key={`${segment.segmentId ?? segment.clientSourceId ?? title}-${index}`}>
              <strong>{segmentLabel(segment, index)}</strong>
              <span>{segmentDetail(segment)}</span>
              {(segment.reason || segment.redactionState) && (
                <small>{segment.reason ?? segment.redactionState}</small>
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function DeliveryFacts({ delivery }: { delivery: LensContextDelivery }) {
  const state = deliveryCopy(delivery);
  const limits = delivery.limits;
  const rendered = delivery.rendered;
  const limitFacts = [
    ["PROMPT TOKENS", limits?.promptTokens],
    ["CONTEXT WINDOW", limits?.contextWindowTokens],
    ["REQUESTED OUTPUT", limits?.requestedMaxTokens],
    ["GENERATED TOKENS", limits?.generatedTokens],
  ].filter(([, value]) => value != null) as Array<[string, number]>;
  const renderedFacts = [
    ["RENDERED TOKENS", rendered?.tokens ?? rendered?.tokenCount],
    ["RENDERED BYTES", rendered?.bytes],
    ["TEMPLATE", rendered?.templateFingerprint],
    ["CONTENT", rendered?.contentAvailable === undefined ? undefined : rendered.contentAvailable ? "AVAILABLE" : "WITHHELD"],
  ].filter(([, value]) => value != null) as Array<[string, string | number]>;

  return (
    <div className="lens-context-canvas-delivery-view">
      <section className={`lens-context-canvas-delivery-summary is-${delivery.status}`} aria-label="Delivery receipt summary">
        <span>RECEIPT STATUS</span>
        <strong>{state.title}</strong>
        <p>{state.detail}</p>
      </section>
      <div className="lens-context-canvas-delivery-stages">
        <DeliveryStage title="Requested" segments={delivery.requested} empty="Requested input was not recorded in this receipt." />
        <DeliveryStage title="Delivered" segments={delivery.delivered} empty="Delivered-segment facts were not recorded in this receipt." />
        <DeliveryStage title="Assembled" segments={delivery.assembled} empty="Assembly facts were not recorded in this receipt." />
      </div>
      <section className="lens-context-canvas-delivery-facts" aria-label="Rendered prompt and limits facts">
        <header><h3>Rendered &amp; limits</h3><span>RECORDED FACTS ONLY</span></header>
        {renderedFacts.length || limitFacts.length ? (
          <dl>
            {renderedFacts.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{typeof value === "number" ? value.toLocaleString() : value}</dd></div>)}
            {limitFacts.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value.toLocaleString()}</dd></div>)}
          </dl>
        ) : <p>No rendered-prompt or limit facts were recorded in this receipt.</p>}
      </section>
      {(delivery.omitted || delivery.transformations) && (
        <section className="lens-context-canvas-delivery-notes" aria-label="Delivery omissions and transformations">
          {delivery.omitted && <DeliveryStage title="Omitted" segments={delivery.omitted} empty="NO OMITTED SEGMENTS RECORDED" />}
          {delivery.transformations && (
            <div>
              <h3>Transformations</h3>
              {delivery.transformations.length ? (
                <ul>{delivery.transformations.map((item, index) => <li key={`${item.reason}-${index}`}>{item.detail ?? item.reason}</li>)}</ul>
              ) : <p>NO TRANSFORMATIONS RECORDED</p>}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function RenderedPrompt({ delivery }: { delivery: LensContextDelivery }) {
  const rendered = delivery.rendered;
  const hasText = typeof rendered?.text === "string";
  const metadata = [
    ["SHA-256", rendered?.sha256],
    ["BYTES", rendered?.bytes],
    ["TOKENS", rendered?.tokens ?? rendered?.tokenCount],
    ["TEMPLATE", rendered?.templateFingerprint],
    ["ESTIMATED", rendered?.estimated === undefined ? undefined : rendered.estimated ? "YES" : "NO"],
  ].filter(([, value]) => value != null) as Array<[string, string | number]>;
  return (
    <div className="lens-context-canvas-rendered-view">
      <section className="lens-context-canvas-rendered-prompt">
        <header><h3>Exact model-facing prompt</h3><span>{hasText ? "RECORDED" : "UNAVAILABLE"}</span></header>
        {hasText ? (
          <pre>{rendered.text}</pre>
        ) : (
          <p>{rendered?.contentAvailable === false
            ? "Exact rendered text was withheld by the receipt's privacy or retention policy."
            : "Exact rendered prompt text was not recorded for this run."}</p>
        )}
      </section>
      <section className="lens-context-canvas-rendered-metadata" aria-label="Rendered prompt metadata">
        <header><h3>Rendered metadata</h3><span>RECORDED FACTS ONLY</span></header>
        {metadata.length ? (
          <dl>{metadata.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{typeof value === "number" ? value.toLocaleString() : value}</dd></div>)}</dl>
        ) : <p>No rendered-prompt metadata was recorded in this receipt.</p>}
      </section>
    </div>
  );
}

/** Presentational input stage. It owns only an optional representation tab; all run/source/job state stays with the parent. */
export function LensContextCanvas({
  data,
  selectedSourceId = null,
  onSelectedSourceChange,
  representation,
  onRepresentationChange,
  delivery = FALLBACK_DELIVERY,
  deliveryContent,
  renderedContent,
  supplementaryContent,
  sourceMeasurementStatus = "idle",
  sourceMeasurementJob,
  sourceMeasurementCache,
  sourceAbsence,
  onMeasureSources,
  onStopWaitingForSources,
  children,
}: LensContextCanvasProps) {
  const [internalRepresentation, setInternalRepresentation] = useState<LensContextRepresentation>("conversation");
  const activeRepresentation = representation ?? internalRepresentation;
  const sources = contextSources(data);
  const messages = sources.filter((source) => !isRetrievedContext(source));
  const retrieved = sources.filter(isRetrievedContext);
  const counts = clearedTokenCounts(data);
  const measurementState = measurementCopy({ data, status: sourceMeasurementStatus, job: sourceMeasurementJob, cache: sourceMeasurementCache, absence: sourceAbsence });
  const supplementary = supplementaryContent ?? children;

  function selectRepresentation(next: LensContextRepresentation) {
    if (representation === undefined) setInternalRepresentation(next);
    onRepresentationChange?.(next);
  }

  return (
    <section className="lens-context-canvas" aria-labelledby="lens-context-canvas-title">
      <header className="lens-context-canvas-head">
        <div><span className="eyebrow">INPUT / RETRIEVAL</span><h2 id="lens-context-canvas-title">Prompt &amp; context</h2></div>
        <p>{sources.length} {sources.length === 1 ? "SOURCE" : "SOURCES"}</p>
      </header>
      <nav className="lens-context-canvas-tabs" role="tablist" aria-label="Prompt and context representation">
        {REPRESENTATIONS.map((item) => (
          <button
            key={item.id}
            id={`lens-context-tab-${item.id}`}
            type="button"
            role="tab"
            aria-selected={activeRepresentation === item.id}
            aria-controls={`lens-context-panel-${item.id}`}
            className={activeRepresentation === item.id ? "is-selected" : ""}
            onClick={() => selectRepresentation(item.id)}
          >{item.label}</button>
        ))}
      </nav>
      <div className="lens-context-canvas-layout">
        {activeRepresentation === "conversation" && (
          <section id="lens-context-panel-conversation" role="tabpanel" aria-labelledby="lens-context-tab-conversation" className="lens-context-canvas-conversation-view">
            <section className="lens-context-canvas-prompt" aria-labelledby="lens-context-canvas-prompt-title">
              <header><span>RECORDED USER PROMPT</span><span>{data?.prompt ? "AVAILABLE" : "UNAVAILABLE"}</span></header>
              <h3 id="lens-context-canvas-prompt-title">User prompt</h3>
              <p>{data?.prompt || "Prompt text was not recorded for this run."}</p>
            </section>
            <section className="lens-context-canvas-measurement" aria-label="Source measurement">
              <div><span>SOURCE MEASUREMENT</span><strong>{measurementState.title}</strong>{measurementState.detail && <p>{measurementState.detail}</p>}</div>
              {sourceMeasurementStatus === "measuring" ? (
                onStopWaitingForSources && <button type="button" onClick={onStopWaitingForSources}>STOP WAITING</button>
              ) : onMeasureSources ? <button type="button" disabled={!data} onClick={onMeasureSources}>MEASURE SOURCES</button> : null}
            </section>
            {data?.influenceMethod && (
              <section className="lens-context-canvas-boundary" aria-label="Measurement boundary">
                <header><span>{data.influenceMethod.mode.replaceAll("_", " ").toUpperCase()}</span>{data.influenceThresholds?.cellAbsDeltaNats != null && <b>FLOOR {data.influenceThresholds.cellAbsDeltaNats.toFixed(4)} NATS/TOKEN</b>}</header>
                {data.contextCoverage && <p>{data.contextCoverage.measuredSources} / {data.contextCoverage.totalSources} SOURCES MEASURED{!data.contextCoverage.complete && data.contextCoverage.omittedSources > 0 ? ` · ${data.contextCoverage.omittedSources} OMITTED FROM MEASUREMENT` : ""}</p>}
                <p>{data.influenceMethod.caveat}</p><p><b>DOES NOT LICENSE </b>{data.influenceMethod.claimLimit}</p>
              </section>
            )}
            <SourceGroup title="Messages and instructions" sources={messages} selectedSourceId={selectedSourceId} counts={counts} onSelectedSourceChange={onSelectedSourceChange} empty="No message context was recorded in the Lens inspection." />
            <SourceGroup title="Retrieved context" sources={retrieved} selectedSourceId={selectedSourceId} counts={counts} onSelectedSourceChange={onSelectedSourceChange} empty="No retrieved context was recorded in the Lens inspection." />
          </section>
        )}
        {activeRepresentation === "delivery" && (
          <section id="lens-context-panel-delivery" role="tabpanel" aria-labelledby="lens-context-tab-delivery" className="lens-context-canvas-primary-panel">
            {deliveryContent ?? <DeliveryFacts delivery={delivery} />}
          </section>
        )}
        {activeRepresentation === "rendered" && (
          <section id="lens-context-panel-rendered" role="tabpanel" aria-labelledby="lens-context-tab-rendered" className="lens-context-canvas-primary-panel">
            {renderedContent ?? <RenderedPrompt delivery={delivery} />}
          </section>
        )}
        {supplementary && <aside className="lens-context-canvas-supplementary" aria-label="Supplementary investigation evidence">{supplementary}</aside>}
      </div>
    </section>
  );
}
