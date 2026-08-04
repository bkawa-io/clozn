import { useEffect, useMemo, useState } from "react";
import {
  loadRunInvestigation,
  loadSpanAddresses,
  type InvestigationState,
  type ReceivedContextSection,
  type ReceivedSegment,
  type RunInvestigationReceipt,
  type SpanAddress,
  type SpanAddressDocument,
} from "../../data/received-context";
import { ContextReceipt } from "./ContextReceipt";

interface ReceivedContextProps {
  runId: string;
}

type Resource<T> =
  | { status: "idle" | "loading" }
  | { status: "failed" }
  | { status: "ready"; value: T };

type ReceiptGroup = "delivered" | "assembled" | "omitted";

function shortId(value: string | undefined) {
  if (!value) return "UNAVAILABLE";
  return value.length > 22 ? `${value.slice(0, 12)}...${value.slice(-6)}` : value;
}

function count(value: number | undefined, suffix = "") {
  return value == null ? "UNAVAILABLE" : `${value.toLocaleString()}${suffix}`;
}

function statusLabel(state: InvestigationState) {
  return state.replaceAll("_", " ").toUpperCase();
}

function addressKeyCandidates(segment: ReceivedSegment): string[] {
  return [
    segment.segmentId,
    segment.clientSourceId,
  ].filter((value): value is string => Boolean(value));
}

function addressIndex(addresses: SpanAddress[]): Map<string, SpanAddress> {
  const index = new Map<string, SpanAddress>();
  const ordered = [...addresses].sort((a, b) => {
    const aContext = a.nativeRef.collection === "context_receipt.delivered"
      || a.nativeRef.collection === "run.messages";
    const bContext = b.nativeRef.collection === "context_receipt.delivered"
      || b.nativeRef.collection === "run.messages";
    return Number(bContext) - Number(aContext);
  });
  for (const address of ordered) {
    for (const key of [
      address.nativeRef.segmentId,
      address.nativeRef.clientSourceId,
      address.nativeRef.id,
    ]) {
      if (key && !index.has(key)) index.set(key, address);
    }
  }
  return index;
}

function addressFor(segment: ReceivedSegment, index: Map<string, SpanAddress>) {
  for (const key of addressKeyCandidates(segment)) {
    const address = index.get(key);
    if (address) return address;
  }
  return undefined;
}

function groupLabel(group: ReceiptGroup) {
  if (group === "delivered") return "DELIVERED";
  if (group === "assembled") return "ASSEMBLED";
  return "OMITTED";
}

function privacyState(segment: ReceivedSegment, address: SpanAddress | undefined) {
  if (
    segment.redactionState === "redacted"
    || segment.redactionState === "hash_only"
    || address?.resolution.state === "redacted"
  ) return "redacted";
  return undefined;
}

function syntheticLegacySegments(
  received: ReceivedContextSection,
  addresses: SpanAddress[],
): ReceivedSegment[] {
  if (received.delivered.length) return received.delivered;
  return addresses
    .filter((address) =>
      address.nativeRef.collection === "run.messages"
      || address.nativeRef.collection === "context_receipt.delivered")
    .map((address, index) => ({
      segmentId: address.nativeRef.segmentId ?? address.nativeRef.id,
      clientSourceId: address.nativeRef.clientSourceId,
      sourceLabel: address.nativeRef.sourceLabel ?? "legacy message",
      sourceType: "message",
      originalOrder: index,
      redactionState: address.resolution.state === "redacted" ? "redacted" : undefined,
    }));
}

function SpanLink({ runId, address }: { runId: string; address: SpanAddress }) {
  return (
    <a
      className="received-context-span-link"
      href={`/runs/${encodeURIComponent(runId)}/span-addresses#${address.addressId}`}
      aria-label={`Stable span ${address.addressId}`}
      title={address.addressId}
    >
      {shortId(address.addressId)}
    </a>
  );
}

function ReceiptRow({
  runId,
  group,
  segment,
  index,
  spanIndex,
  spansStatus,
}: {
  runId: string;
  group: ReceiptGroup;
  segment: ReceivedSegment;
  index: number;
  spanIndex: Map<string, SpanAddress>;
  spansStatus: Resource<SpanAddressDocument>["status"];
}) {
  const address = addressFor(segment, spanIndex);
  const redaction = privacyState(segment, address);
  const omitted = group === "omitted" || segment.included === false;
  const groupState = omitted ? "omitted" : group;
  const resolution = address?.resolution.state;
  return (
    <li
      className={`received-context-row is-${groupState} ${redaction ? "is-redacted" : ""}`}
      data-receipt-state={groupState}
    >
      <span className="received-context-order">
        {segment.originalOrder ?? index}
      </span>
      <div className="received-context-row-main">
        <strong>{segment.sourceLabel ?? segment.sourceType ?? "UNLABELED INPUT"}</strong>
        <small title={segment.clientSourceId ?? segment.segmentId}>
          {shortId(segment.clientSourceId ?? segment.segmentId)}
        </small>
      </div>
      <div className="received-context-cost">
        {segment.deliveredTokens != null && <span>{count(segment.deliveredTokens, " TOK")}</span>}
        <span>{count(segment.deliveredBytes, " B")}</span>
      </div>
      <div className="received-context-statuses">
        <span className={`received-context-state is-${groupState}`}>{groupLabel(groupState)}</span>
        {redaction && <span className="received-context-state is-redacted">REDACTED</span>}
        {resolution === "unavailable" && (
          <span className="received-context-state is-unavailable">UNAVAILABLE</span>
        )}
        {resolution === "drifted" && (
          <span className="received-context-state is-drifted">HASH DRIFT</span>
        )}
      </div>
      <div className="received-context-address">
        {address ? (
          <SpanLink runId={runId} address={address} />
        ) : (
          <span className="received-context-no-span">
            {spansStatus === "loading" || spansStatus === "idle"
              ? "RESOLVING ID"
              : "ID UNAVAILABLE"}
          </span>
        )}
      </div>
      {(segment.reason || segment.detail || address?.resolution.reason) && (
        <p>
          {segment.detail ?? segment.reason ?? address?.resolution.reason}
        </p>
      )}
    </li>
  );
}

function ReceiptGroupView({
  runId,
  title,
  group,
  segments,
  spanIndex,
  spansStatus,
}: {
  runId: string;
  title: string;
  group: ReceiptGroup;
  segments: ReceivedSegment[];
  spanIndex: Map<string, SpanAddress>;
  spansStatus: Resource<SpanAddressDocument>["status"];
}) {
  const ordered = [...segments].sort(
    (a, b) => (a.originalOrder ?? 0) - (b.originalOrder ?? 0),
  );
  return (
    <section className={`received-context-group is-${group}`} aria-labelledby={`received-${group}`}>
      <header>
        <h4 id={`received-${group}`}>{title}</h4>
        <b>{ordered.length}</b>
      </header>
      {ordered.length ? (
        <ol className="received-context-list">
          {ordered.map((segment, index) => (
            <ReceiptRow
              runId={runId}
              group={group}
              segment={segment}
              index={index}
              spanIndex={spanIndex}
              spansStatus={spansStatus}
              key={`${segment.segmentId ?? segment.clientSourceId ?? index}-${index}`}
            />
          ))}
        </ol>
      ) : (
        <div className="received-context-empty">NONE RECORDED</div>
      )}
    </section>
  );
}

function EvidenceNotice({
  state,
  reason,
}: {
  state: InvestigationState;
  reason?: string;
}) {
  if (!["unavailable", "failed", "inconclusive"].includes(state)) return null;
  const failed = state === "failed";
  return (
    <div
      className={`received-context-notice ${failed ? "is-failed" : "is-unavailable"}`}
      role={failed ? "alert" : "note"}
    >
      <strong>{failed ? "EVIDENCE FAILED" : `${statusLabel(state)}`}</strong>
      <span>{reason ?? "No delivery evidence is available for this run."}</span>
    </div>
  );
}

function CostStrip({
  received,
  renderedAddress,
  runId,
}: {
  received: ReceivedContextSection;
  renderedAddress?: SpanAddress;
  runId: string;
}) {
  const deliveredBytes = received.delivered.reduce(
    (total, segment) => total + (segment.deliveredBytes ?? 0),
    0,
  );
  const tokenCost = received.rendered?.tokens
    ?? received.rendered?.tokenCount
    ?? received.limits.promptTokens;
  const byteCost = received.rendered?.bytes ?? (deliveredBytes || undefined);
  return (
    <dl className="received-context-cost-strip">
      <div>
        <dt>PROMPT TOKENS</dt>
        <dd>{count(tokenCost)}</dd>
      </div>
      <div>
        <dt>RENDERED BYTES</dt>
        <dd>{count(byteCost, " B")}</dd>
      </div>
      <div>
        <dt>CONTEXT WINDOW</dt>
        <dd>{count(received.limits.contextWindowTokens)}</dd>
      </div>
      <div>
        <dt>RENDERED SPAN</dt>
        <dd>
          {renderedAddress
            ? <SpanLink runId={runId} address={renderedAddress} />
            : "UNAVAILABLE"}
        </dd>
      </div>
    </dl>
  );
}

export function ReceivedContext({ runId }: ReceivedContextProps) {
  const [investigation, setInvestigation] = useState<Resource<RunInvestigationReceipt>>({
    status: "idle",
  });
  const [spans, setSpans] = useState<Resource<SpanAddressDocument>>({ status: "idle" });
  const [authorizedOpen, setAuthorizedOpen] = useState(false);

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    setInvestigation({ status: "loading" });
    setSpans({ status: "loading" });
    setAuthorizedOpen(false);

    void loadRunInvestigation(runId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setInvestigation({ status: "ready", value });
    }).catch(() => {
      if (!controller.signal.aborted) setInvestigation({ status: "failed" });
    });
    void loadSpanAddresses(runId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setSpans({ status: "ready", value });
    }).catch(() => {
      if (!controller.signal.aborted) setSpans({ status: "failed" });
    });
    return () => controller.abort();
  }, [runId]);

  const spanAddresses = spans.status === "ready" ? spans.value.addresses : [];
  const spanIndex = useMemo(() => addressIndex(spanAddresses), [spanAddresses]);
  const received = investigation.status === "ready"
    ? investigation.value.receivedContext
    : undefined;
  const delivered = received
    ? syntheticLegacySegments(received, spanAddresses)
    : [];
  const renderedAddress = spanAddresses.find(
    (address) => address.kind === "rendered_prompt_segment",
  );
  const contextSource = spans.status === "ready"
    ? spans.value.sourceArtifacts.find((artifact) =>
        artifact.schema.includes("context-receipt")
        || artifact.schema.includes("context_receipt")
        || artifact.schema.includes("run-record"))
    : undefined;
  const sourceRedacted = contextSource?.nativeStatus === "redacted";

  return (
    <section className="received-context" id="received-context-title" aria-label="What did the model receive?">
      {received && (
        <header className="received-context-head">
          <span className={`received-context-overall is-${received.state}`}>
            {statusLabel(received.state)}
          </span>
        </header>
      )}

      <p className="received-context-boundary">
        Delivery shows what reached prompt assembly. It does not prove that a passage changed the answer.
      </p>

      {investigation.status === "idle" || investigation.status === "loading" ? (
        <div className="received-context-empty">LOADING DELIVERY EVIDENCE</div>
      ) : investigation.status === "failed" || !received ? (
        <div className="received-context-notice is-failed" role="alert">
          <strong>INVESTIGATION REQUEST FAILED</strong>
          <span>The recorded delivery view could not be loaded.</span>
        </div>
      ) : (
        <>
          <EvidenceNotice state={received.state} reason={received.reason} />
          {(sourceRedacted || received.privacy === "off") && (
            <div className="received-context-notice is-redacted" role="note">
              <strong>REDACTED</strong>
              <span>
                {contextSource?.reason
                  ?? "The run did not retain readable context text under its privacy settings."}
              </span>
            </div>
          )}
          <div className="received-context-privacy">
            <span>VIEW PRIVACY</span>
            <strong>METADATA ONLY</strong>
            <small>
              Stored receipt: {(received.privacy ?? "unknown").replaceAll("_", " ").toUpperCase()}
            </small>
          </div>
          <CostStrip received={received} renderedAddress={renderedAddress} runId={runId} />

          {spans.status === "failed" && (
            <div className="received-context-notice is-failed" role="alert">
              <strong>STABLE SPAN REQUEST FAILED</strong>
              <span>Delivery evidence remains visible, but stable span links are unavailable.</span>
            </div>
          )}

          <ReceiptGroupView
            runId={runId}
            title="Request as delivered"
            group="delivered"
            segments={delivered}
            spanIndex={spanIndex}
            spansStatus={spans.status}
          />
          <ReceiptGroupView
            runId={runId}
            title="Context as assembled"
            group="assembled"
            segments={received.assembled}
            spanIndex={spanIndex}
            spansStatus={spans.status}
          />
          <ReceiptGroupView
            runId={runId}
            title="Omitted before generation"
            group="omitted"
            segments={received.omitted}
            spanIndex={spanIndex}
            spansStatus={spans.status}
          />

          <section className="received-context-authorized">
            <p>
              Exact messages and the rendered prompt are not copied into these metadata APIs.
            </p>
            <button
              type="button"
              aria-expanded={authorizedOpen}
              onClick={() => setAuthorizedOpen((open) => !open)}
            >
              {authorizedOpen ? "CLOSE AUTHORIZED RECEIPT" : "OPEN AUTHORIZED CONTEXT RECEIPT"}
            </button>
            {authorizedOpen && (
              <div className="received-context-authorized-body">
                <ContextReceipt runId={runId} />
              </div>
            )}
          </section>
        </>
      )}
    </section>
  );
}
