import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { routeHref, type InspectRoute } from "../../app/router";
import { describeError, isContractError, studioApi } from "../../data/client";
import type {
  InvestigationState,
  ReceivedContextSection,
  ReceivedSegment,
  RunInvestigation,
  SpanAddress,
  SpanAddressDocument,
} from "../../data/receivedContext";
import "./receivedContext.css";

type Resource<T> =
  | { phase: "loading" }
  | { phase: "ready"; value: T }
  | { phase: "unavailable"; message: string }
  | { phase: "malformed"; message: string };

type ReceiptGroup = "delivered" | "assembled" | "omitted";

interface CompositionEntry {
  id: string;
  label: string;
  value: number;
  omitted: boolean;
  reduced: boolean;
  reason?: string;
}

type Composition =
  | { state: "ready"; entries: readonly CompositionEntry[]; unit: "tokens" | "bytes"; total?: number }
  | { state: "not_measured"; reason: string };

function shortId(value: string | undefined): string {
  if (!value) return "Unavailable";
  return value.length > 24 ? `${value.slice(0, 12)}…${value.slice(-7)}` : value;
}

function sentence(value: string): string {
  const spaced = value.replaceAll("_", " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function stateLabel(state: InvestigationState): string {
  return sentence(state);
}

function count(value: number | null | undefined, suffix = ""): string {
  return value === undefined || value === null ? "Not recorded" : `${value.toLocaleString()}${suffix}`;
}

function resourceFailure<T>(error: unknown): Resource<T> {
  return {
    phase: isContractError(error) ? "malformed" : "unavailable",
    message: describeError(error),
  };
}

function addressIndex(addresses: readonly SpanAddress[]): Map<string, SpanAddress> {
  const index = new Map<string, SpanAddress>();
  const ordered = [...addresses].sort((left, right) => {
    const preferred = (address: SpanAddress) => Number(
      address.nativeRef.collection === "context_receipt.delivered"
      || address.nativeRef.collection === "run.messages",
    );
    return preferred(right) - preferred(left);
  });
  for (const address of ordered) {
    for (const key of [address.nativeRef.segmentId, address.nativeRef.clientSourceId, address.nativeRef.id]) {
      if (key && !index.has(key)) index.set(key, address);
    }
  }
  return index;
}

function addressFor(segment: ReceivedSegment, index: ReadonlyMap<string, SpanAddress>): SpanAddress | undefined {
  for (const key of [segment.segmentId, segment.clientSourceId]) {
    if (!key) continue;
    const address = index.get(key);
    if (address) return address;
  }
  return undefined;
}

function segmentKey(segment: ReceivedSegment, index: number): string {
  return segment.segmentId ?? segment.clientSourceId ?? `receipt-row-${index}`;
}

function contextComposition(
  received: ReceivedContextSection,
  delivered: readonly ReceivedSegment[],
): Composition {
  const rows = new Map<string, ReceivedSegment>();
  const merge = (segment: ReceivedSegment, index: number) => {
    const key = segmentKey(segment, index);
    const previous = rows.get(key);
    rows.set(key, {
      ...previous,
      ...segment,
      deliveredTokens: segment.deliveredTokens ?? previous?.deliveredTokens,
      deliveredBytes: segment.deliveredBytes ?? previous?.deliveredBytes,
      included: segment.included ?? previous?.included,
      reason: segment.reason ?? previous?.reason,
      detail: segment.detail ?? previous?.detail,
    });
  };
  delivered.forEach(merge);
  (received.omitted ?? []).forEach(merge);
  const merged = [...rows.entries()];
  if (!merged.length) {
    return { state: "not_measured", reason: "No delivered or omitted segment counts were recorded." };
  }
  const tokensComplete = merged.every(([, segment]) => segment.deliveredTokens !== undefined);
  const bytesComplete = merged.every(([, segment]) => segment.deliveredBytes !== undefined);
  if (!tokensComplete && !bytesComplete) {
    return {
      state: "not_measured",
      reason: "At least one segment lacks an exact token or byte count, so a composition would be misleading.",
    };
  }
  const unit = tokensComplete ? "tokens" : "bytes";
  const omittedIds = new Set((received.omitted ?? []).map(segmentKey));
  const entries = merged.map(([id, segment]) => ({
    id,
    label: segment.sourceLabel ?? segment.sourceType ?? "Unlabelled input",
    value: unit === "tokens" ? segment.deliveredTokens! : segment.deliveredBytes!,
    omitted: segment.included === false || omittedIds.has(id),
    reduced: Boolean(segment.redactionState && segment.redactionState !== "full"),
    reason: segment.reason ?? segment.detail,
  }));
  const total = unit === "tokens"
    ? received.rendered?.tokens
    : received.rendered?.bytes;
  return { state: "ready", entries, unit, ...(total === undefined ? {} : { total }) };
}

function CompositionView({ composition }: { composition: Composition }) {
  if (composition.state === "not_measured") {
    return (
      <section className="received-composition is-unavailable" aria-labelledby="received-composition-title">
        <div><span>Composition</span><h3 id="received-composition-title">Prompt composition not measured</h3></div>
        <p>{composition.reason}</p>
      </section>
    );
  }
  const unitLabel = composition.unit === "tokens" ? "tokens" : "bytes";
  const accessible = composition.entries
    .map((entry) => `${entry.label} ${entry.value.toLocaleString()} ${unitLabel}${entry.omitted ? " omitted" : ""}`)
    .join(", ");
  return (
    <section className="received-composition" aria-labelledby="received-composition-title">
      <div>
        <span>Exact {unitLabel}</span>
        <h3 id="received-composition-title">Prompt composition</h3>
      </div>
      <div className="received-composition-bar" role="img" aria-label={`Prompt composition: ${accessible}`}>
        {composition.entries.map((entry) => (
          <span
            className={`${entry.omitted ? "is-omitted" : entry.reduced ? "is-reduced" : "is-present"}`}
            key={entry.id}
            style={{ "--segment-weight": Math.max(entry.value, 0.01) } as CSSProperties}
            title={`${entry.label}: ${entry.value.toLocaleString()} ${unitLabel}${entry.reason ? ` · ${entry.reason}` : ""}`}
          />
        ))}
      </div>
      <ul>
        {composition.entries.map((entry) => (
          <li key={entry.id} className={entry.omitted ? "is-omitted" : ""}>
            <i aria-hidden="true" />
            <span>{entry.label}</span>
            <b>{entry.value.toLocaleString()} {unitLabel}</b>
            {entry.omitted && <em>Omitted</em>}
            {entry.reason && <small>{sentence(entry.reason)}</small>}
          </li>
        ))}
      </ul>
      {composition.total !== undefined && <p>Total rendered: {composition.total.toLocaleString()} {unitLabel}</p>}
    </section>
  );
}

function SpanLink({ runId, address }: { runId: string; address: SpanAddress }) {
  return (
    <a
      className="received-span-link"
      href={`/runs/${encodeURIComponent(runId)}/span-addresses#${address.addressId}`}
      aria-label={`Stable span ${address.addressId}`}
      title={address.addressId}
    >
      {shortId(address.addressId)}
    </a>
  );
}

function RowState({ group, segment, address }: {
  group: ReceiptGroup;
  segment: ReceivedSegment;
  address?: SpanAddress;
}) {
  const omitted = group === "omitted" || segment.included === false;
  const redacted = segment.redactionState === "redacted"
    || segment.redactionState === "hash_only"
    || address?.resolution.state === "redacted";
  return (
    <div className="received-row-states">
      <span className={`state-${omitted ? "omitted" : group}`}>{omitted ? "Omitted" : sentence(group)}</span>
      {redacted && <span className="state-redacted">Redacted</span>}
      {address?.resolution.state === "unavailable" && <span className="state-unavailable">Unavailable</span>}
      {address?.resolution.state === "drifted" && <span className="state-drifted">Hash drift</span>}
    </div>
  );
}

function SegmentRow({
  runId,
  group,
  segment,
  ordinal,
  spans,
  spanIndex,
}: {
  runId: string;
  group: ReceiptGroup;
  segment: ReceivedSegment;
  ordinal: number;
  spans: Resource<SpanAddressDocument>;
  spanIndex: ReadonlyMap<string, SpanAddress>;
}) {
  const address = addressFor(segment, spanIndex);
  const reason = segment.detail ?? segment.reason ?? address?.resolution.reason;
  return (
    <li className="received-segment-row">
      <span className="received-segment-order">{segment.originalOrder ?? ordinal}</span>
      <div className="received-segment-identity">
        <strong>{segment.sourceLabel ?? segment.sourceType ?? "Unlabelled input"}</strong>
        <code title={segment.clientSourceId ?? segment.segmentId}>{shortId(segment.clientSourceId ?? segment.segmentId)}</code>
      </div>
      <dl>
        <div><dt>Tokens</dt><dd>{count(segment.deliveredTokens)}</dd></div>
        <div><dt>Bytes</dt><dd>{count(segment.deliveredBytes, " B")}</dd></div>
      </dl>
      <RowState group={group} segment={segment} address={address} />
      <div className="received-segment-address">
        <span>Stable span</span>
        {address
          ? <SpanLink runId={runId} address={address} />
          : <small>{spans.phase === "loading" ? "Resolving ID" : "ID unavailable"}</small>}
      </div>
      {reason && <p>{sentence(reason)}</p>}
    </li>
  );
}

function SegmentGroup({
  runId,
  title,
  group,
  segments,
  spans,
  spanIndex,
}: {
  runId: string;
  title: string;
  group: ReceiptGroup;
  segments: readonly ReceivedSegment[];
  spans: Resource<SpanAddressDocument>;
  spanIndex: ReadonlyMap<string, SpanAddress>;
}) {
  const ordered = [...segments].sort((left, right) => (left.originalOrder ?? 0) - (right.originalOrder ?? 0));
  return (
    <section className="received-segment-group" aria-labelledby={`received-${group}-title`}>
      <header>
        <h3 id={`received-${group}-title`}>{title}</h3>
        <span>{ordered.length.toLocaleString()}</span>
      </header>
      {ordered.length ? (
        <ol>
          {ordered.map((segment, index) => (
            <SegmentRow
              key={`${segment.segmentId ?? segment.clientSourceId ?? index}:${index}`}
              runId={runId}
              group={group}
              segment={segment}
              ordinal={index}
              spans={spans}
              spanIndex={spanIndex}
            />
          ))}
        </ol>
      ) : <p className="received-empty-group">None recorded</p>}
    </section>
  );
}

function EvidenceState({ received }: { received: ReceivedContextSection }) {
  if (!["unavailable", "failed", "inconclusive"].includes(received.state)) return null;
  return (
    <p className={`received-notice state-${received.state}`} role={received.state === "failed" ? "alert" : "note"}>
      <strong>{stateLabel(received.state)}</strong>
      <span>{received.reason ?? "No delivery evidence is available for this Run."}</span>
    </p>
  );
}

function InvestigationFailure({ resource }: { resource: Resource<RunInvestigation> }) {
  if (resource.phase === "loading" || resource.phase === "ready") return null;
  return (
    <p className="received-primary-state is-error" role="alert">
      {resource.phase === "malformed" ? "Malformed investigation contract." : "Investigation request unavailable."}
      <span>{resource.message}</span>
    </p>
  );
}

export function ReceivedContextSurface({ route }: { route: InspectRoute }) {
  const [investigation, setInvestigation] = useState<Resource<RunInvestigation>>({ phase: "loading" });
  const [spans, setSpans] = useState<Resource<SpanAddressDocument>>({ phase: "loading" });

  useEffect(() => {
    const controller = new AbortController();
    setInvestigation({ phase: "loading" });
    setSpans({ phase: "loading" });
    void studioApi.runInvestigation(route.runId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setInvestigation({ phase: "ready", value });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setInvestigation(resourceFailure(error));
    });
    void studioApi.spanAddresses(route.runId, controller.signal).then((value) => {
      if (!controller.signal.aborted) setSpans({ phase: "ready", value });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) setSpans(resourceFailure(error));
    });
    return () => controller.abort();
  }, [route.runId]);

  const received = investigation.phase === "ready" ? investigation.value.receivedContext : undefined;
  const addresses = spans.phase === "ready" ? spans.value.addresses : [];
  const indexedAddresses = useMemo(() => addressIndex(addresses), [addresses]);
  const delivered = received?.delivered ?? [];
  const composition = useMemo(
    () => received ? contextComposition(received, delivered) : undefined,
    [delivered, received],
  );
  const renderedAddress = addresses.find((address) => address.kind === "rendered_prompt_segment");
  const contextArtifact = spans.phase === "ready"
    ? spans.value.sourceArtifacts.find((artifact) => (
        artifact.schema.includes("context-receipt")
        || artifact.schema.includes("context_receipt")
        || artifact.schema.includes("run-record")
      ))
    : undefined;
  const sourceRedacted = contextArtifact?.nativeStatus === "redacted" || received?.privacy === "off";
  const deliveredBytes = delivered.length > 0 && delivered.every((segment) => segment.deliveredBytes !== undefined)
    ? delivered.reduce((sum, segment) => sum + segment.deliveredBytes!, 0)
    : undefined;
  const promptTokens = received?.rendered?.tokens;
  const renderedBytes = received?.rendered?.bytes ?? deliveredBytes;
  const sessionHref = route.sessionId
    ? routeHref({ surface: "session", sessionId: route.sessionId, runId: route.runId })
    : routeHref({ surface: "sessions" });
  const effectsHref = routeHref({
    surface: "inspect",
    runId: route.runId,
    view: "context-effects",
    ...(route.sessionId ? { sessionId: route.sessionId } : {}),
  });
  const receiptHref = routeHref({
    surface: "inspect",
    runId: route.runId,
    view: "receipt",
    ...(route.sessionId ? { sessionId: route.sessionId } : {}),
  });

  return (
    <div className="inspect-surface received-context-surface">
      <header className="inspect-heading">
        <div>
          <a className="back-link" href={sessionHref}>← {route.sessionId ? "Session" : "Sessions"}</a>
          <h1>Inspect</h1>
          <code>{shortId(route.runId)}</code>
        </div>
        {received && <span className={`received-overall state-${received.state}`}>{stateLabel(received.state)}</span>}
      </header>

      <section className="inspect-instrument received-context-instrument" aria-labelledby="received-context-title">
        <header>
          <div><span>RECORDED DELIVERY</span><h2 id="received-context-title">Received context</h2></div>
          <code>GET only · metadata</code>
        </header>
        <div className="received-boundary">
          <p>Delivery shows what reached prompt assembly. It does not prove that a passage changed the answer.</p>
          <a href={effectsHref}>Inspect measured context effects</a>
        </div>

        {investigation.phase === "loading" && (
          <p className="received-primary-state" role="status">Reading delivery evidence…</p>
        )}
        <InvestigationFailure resource={investigation} />

        {received && (
          <>
            <EvidenceState received={received} />
            {sourceRedacted && (
              <p className="received-notice state-redacted" role="note">
                <strong>Redacted</strong>
                <span>{contextArtifact?.reason ?? "Readable context text was not retained under this Run’s privacy settings."}</span>
              </p>
            )}
            <div className="received-privacy">
              <div><span>View privacy</span><strong>Metadata only</strong></div>
              <small>Stored receipt: {sentence(received.privacy ?? "unknown")}</small>
            </div>
            <dl className="received-costs">
              <div><dt>Prompt tokens</dt><dd>{count(promptTokens)}</dd></div>
              <div><dt>Rendered bytes</dt><dd>{count(renderedBytes, " B")}</dd></div>
              <div><dt>Context window</dt><dd>{count(received.contextWindowTokens)}</dd></div>
              <div><dt>Rendered span</dt><dd>{renderedAddress ? <SpanLink runId={route.runId} address={renderedAddress} /> : "Unavailable"}</dd></div>
            </dl>

            {composition && <CompositionView composition={composition} />}

            {spans.phase === "malformed" && (
              <p className="received-notice state-failed" role="alert">
                <strong>Malformed stable span contract</strong>
                <span>Delivery evidence remains visible, but stable span IDs are unavailable. {spans.message}</span>
              </p>
            )}
            {spans.phase === "unavailable" && (
              <p className="received-notice state-failed" role="alert">
                <strong>Stable span request unavailable</strong>
                <span>Delivery evidence remains visible, but stable span IDs are unavailable. {spans.message}</span>
              </p>
            )}

            <div className="received-groups">
              <SegmentGroup
                runId={route.runId}
                title="Request as delivered"
                group="delivered"
                segments={delivered}
                spans={spans}
                spanIndex={indexedAddresses}
              />
              <SegmentGroup
                runId={route.runId}
                title="Context as assembled"
                group="assembled"
                segments={received.assembled ?? []}
                spans={spans}
                spanIndex={indexedAddresses}
              />
              <SegmentGroup
                runId={route.runId}
                title="Omitted before generation"
                group="omitted"
                segments={received.omitted ?? []}
                spans={spans}
                spanIndex={indexedAddresses}
              />
            </div>

            <section className="received-disclosure" aria-labelledby="received-disclosure-title">
              <div>
                <span>Literal prompt text</span>
                <h3 id="received-disclosure-title">Open the existing Context Receipt</h3>
                <p>These metadata reads do not copy messages or the rendered prompt. The Context Receipt owns explicit retained-text disclosure.</p>
              </div>
              <a href={receiptHref}>Open Context Receipt</a>
            </section>
          </>
        )}
      </section>

      <aside className="received-read-boundary">
        <strong>Read-only boundary</strong>
        <p>This view reads stored investigation and span metadata only. It does not score, reconstruct, or modify the Run.</p>
      </aside>
    </div>
  );
}
