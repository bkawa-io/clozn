import type { ReactNode } from "react";
import type {
  ContextCoverage,
  InfluenceAbsence,
  InfluenceMethod,
  InfluenceThresholds,
  SourceReading,
  TokenReading,
  TokenSourceReading,
} from "../../data/types";
import { aggregateSources, summarizeRange, weakestTokenInRange } from "./analysis";
import { describeAbsence, evidenceStateBadge } from "./EvidenceCaveat";
import "./lens-selection-inspector.css";

export type LensInspectorSelection =
  | { kind: "source"; sourceId: string; source?: SourceReading }
  | { kind: "span"; start: number; end: number; claimIndex?: number }
  | { kind: "token"; index: number };

/**
 * Availability is intentionally separate from the link evidence state. For example, a recorded
 * source can be available while its influence is not measured; neither fact permits a causal claim.
 */
export type LensInspectorAvailability =
  | { state: "available"; provenance?: "recorded" | "derived" | "measured"; detail?: string }
  | { state: "not_captured"; detail?: string }
  | { state: "not_measured"; detail?: string }
  | { state: "unavailable"; detail?: string }
  | { state: "privacy_limited"; detail?: string };

export interface LensSelectionInspectorProps {
  selection: LensInspectorSelection | null;
  /** The recorded token trace; no fetch or state ownership belongs to this component. */
  tokens?: readonly TokenReading[];
  /** Context/source captures used to resolve a source selection and describe delivery provenance. */
  sources?: readonly SourceReading[];
  /** Typed source-map absence from the existing influence API. */
  influenceAbsence?: InfluenceAbsence | null;
  influenceMethod?: InfluenceMethod;
  influenceThresholds?: InfluenceThresholds;
  contextCoverage?: ContextCoverage;
  /** Explicitly distinguishes a trace that was not captured from a currently empty trace. */
  tokenTrace?: LensInspectorAvailability;
  /** Optional source-artifact state when it is known independently of influenceAbsence. */
  sourceEvidence?: LensInspectorAvailability;
  onSelectToken?: (index: number) => void;
  onSelectSource?: (sourceId: string) => void;
  /** Caller-owned actions that apply to the current selection. */
  actions?: ReactNode;
  /** A semantic events rail, timing phase list, or other compact evidence-deck content. */
  events?: ReactNode;
  /** Additional feature-owned evidence shown after the standard inspector information. */
  children?: ReactNode;
  scopeHref?: string;
  className?: string;
}

type AvailabilityView = {
  label: string;
  detail?: string;
  className: string;
  provenance?: string;
};

function humanize(value: string) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function signedNats(value: number) {
  return `${value >= 0 ? "+" : ""}${value.toFixed(4)} nats`;
}

function sourceLabel(source: SourceReading) {
  return source.label ?? source.role ?? source.kind ?? "Context source";
}

function availabilityView(value?: LensInspectorAvailability): AvailabilityView | null {
  if (!value || value.state === "available") {
    return value?.provenance
      ? { label: "Available", provenance: humanize(value.provenance), detail: value.detail, className: "is-available" }
      : null;
  }
  switch (value.state) {
    case "not_captured":
      return { label: "Not captured", detail: value.detail, className: "is-not-captured" };
    case "not_measured":
      return { label: "Not measured", detail: value.detail, className: "is-not-measured" };
    case "unavailable":
      return { label: "Unavailable", detail: value.detail, className: "is-unavailable" };
    case "privacy_limited":
      return { label: "Privacy-limited", detail: value.detail, className: "is-privacy-limited" };
    default: {
      const exhaustive: never = value;
      return exhaustive;
    }
  }
}

function absenceView(absence?: InfluenceAbsence | null): AvailabilityView | null {
  if (!absence) return null;
  const described = describeAbsence(absence);
  switch (absence.kind) {
    case "not_measured":
      return { label: "Not measured", detail: described.detail, className: "is-not-measured" };
    case "no_worker":
      return { label: "Unavailable", detail: described.detail, className: "is-unavailable" };
    case "typed":
    case "invalid_request":
    case "server_error":
    case "network_error":
      return { label: "Unavailable", detail: described.detail, className: "is-unavailable" };
    default: {
      const exhaustive: never = absence;
      return exhaustive;
    }
  }
}

function AvailabilityNotice({ value }: { value: AvailabilityView | null }) {
  if (!value) return null;
  return (
    <section className={`lens-selection-availability ${value.className}`} aria-label="Evidence availability">
      <strong>{value.label}</strong>
      {value.provenance && <span>{value.provenance}</span>}
      {value.detail && <p>{value.detail}</p>}
    </section>
  );
}

function MeasurementProvenance({
  method,
  thresholds,
  coverage,
}: {
  method?: InfluenceMethod;
  thresholds?: InfluenceThresholds;
  coverage?: ContextCoverage;
}) {
  if (!method) return null;
  return (
    <section className="lens-selection-provenance" aria-label="Influence measurement provenance">
      <header>
        <span>Measured evidence</span>
        <b>{humanize(method.mode)}</b>
      </header>
      <dl>
        {thresholds?.cellAbsDeltaNats != null && (
          <div><dt>Effect floor</dt><dd>{thresholds.cellAbsDeltaNats.toFixed(4)} nats/token</dd></div>
        )}
        {coverage && (
          <div><dt>Coverage</dt><dd>{coverage.measuredSources} / {coverage.totalSources} sources measured</dd></div>
        )}
      </dl>
      <p>{method.caveat}</p>
      <p><b>Does not license:</b> {method.claimLimit}</p>
    </section>
  );
}

function EvidenceRow({
  source,
  onSelectSource,
}: {
  source: TokenSourceReading;
  onSelectSource?: (sourceId: string) => void;
}) {
  const badge = evidenceStateBadge(source.evidenceState);
  const content = (
    <>
      <span className="lens-selection-evidence-dot" aria-hidden="true" />
      <span className="lens-selection-evidence-copy">
        <strong>{source.label}</strong>
        <small>{source.effect} · {badge.label}</small>
      </span>
      <output>{signedNats(source.deltaNats)}</output>
    </>
  );
  return onSelectSource ? (
    <button
      type="button"
      className={`lens-selection-evidence-row ${badge.className} effect-${source.effect}`}
      onClick={() => onSelectSource(source.sourceId)}
    >{content}</button>
  ) : <div className={`lens-selection-evidence-row ${badge.className} effect-${source.effect}`}>{content}</div>;
}

function EvidenceGroup({
  title,
  sources,
  onSelectSource,
}: {
  title: string;
  sources: readonly TokenSourceReading[];
  onSelectSource?: (sourceId: string) => void;
}) {
  if (!sources.length) return null;
  return (
    <section className="lens-selection-evidence-group" aria-label={title}>
      <header><span>{title}</span><b>{sources.length}</b></header>
      <div>{sources.map((source, index) => <EvidenceRow source={source} onSelectSource={onSelectSource} key={`${source.sourceId}-${index}`} />)}</div>
    </section>
  );
}

function InspectorActions({
  children,
  scopeHref,
}: {
  children?: ReactNode;
  scopeHref?: string;
}) {
  if (!children && !scopeHref) return null;
  return (
    <footer className="lens-selection-actions" aria-label="Selection actions">
      {children}
      {scopeHref && <a href={scopeHref}>Open in scope</a>}
    </footer>
  );
}

function SourceSelection({
  source,
  tokens,
  sourceEvidence,
  influenceAbsence,
  onSelectToken,
  onSelectSource,
  actions,
  scopeHref,
}: {
  source: SourceReading;
  tokens: TokenReading[];
  sourceEvidence?: LensInspectorAvailability;
  influenceAbsence?: InfluenceAbsence | null;
  onSelectToken?: (index: number) => void;
  onSelectSource?: (sourceId: string) => void;
  actions?: ReactNode;
  scopeHref?: string;
}) {
  const linked = tokens.flatMap((token, index) => (
    token.text && (token.sources?.some((link) => link.sourceId === source.id)
      || token.observedSources?.some((link) => link.sourceId === source.id)) ? [index] : []
  ));
  const cleared = tokens.flatMap((token) => (token.sources ?? []).filter(
    (link) => link.sourceId === source.id && link.evidenceState === "causally_supported",
  ));
  const observed = tokens.flatMap((token) => [
    ...(token.sources ?? []).filter((link) => link.sourceId === source.id && link.evidenceState === "observed"),
    ...(token.observedSources ?? []).filter((link) => link.sourceId === source.id),
  ]);
  const derivedAvailability = source.measured === false
    ? { label: "Not measured", detail: "This source was recorded but has not been included in source measurement.", className: "is-not-measured" }
    : source.clearEffect === false && !cleared.length
      ? { label: "Measured · no cleared link", detail: "Measurement ran; no link from this source cleared the effect floor.", className: "is-measured-no-clear" }
      : availabilityView(sourceEvidence) ?? absenceView(influenceAbsence);

  return (
    <>
      <section className="lens-selection-readout">
        <span>Recorded context source</span>
        <h3>{sourceLabel(source)}</h3>
        <p>{source.text || "Content was not captured for this source."}</p>
      </section>

      <dl className="lens-selection-facts">
        <div><dt>Role</dt><dd>{humanize(source.role || "context")}</dd></div>
        <div><dt>Kind</dt><dd>{humanize(source.kind ?? "recorded source")}</dd></div>
        {source.segmentId && <div><dt>Segment</dt><dd><code>{source.segmentId}</code></dd></div>}
        {source.messageIndex != null && <div><dt>Message</dt><dd>{source.messageIndex + 1}</dd></div>}
        <div><dt>Linked output</dt><dd>{linked.length} token{linked.length === 1 ? "" : "s"}</dd></div>
      </dl>

      <AvailabilityNotice value={derivedAvailability} />
      <EvidenceGroup title="Cleared links" sources={cleared} onSelectSource={onSelectSource} />
      <EvidenceGroup title="Observed below floor" sources={observed} onSelectSource={onSelectSource} />

      {linked.length > 0 && (
        <section className="lens-selection-linked-output" aria-label="Linked output">
          <span>Linked output</span>
          <p>{linked.map((index) => tokens[index]?.text).join("")}</p>
        </section>
      )}

      <InspectorActions scopeHref={scopeHref}>
        {onSelectToken && linked.length > 0 && (
          <button type="button" onClick={() => onSelectToken(linked[0])}>Select first linked token</button>
        )}
        {actions}
      </InspectorActions>
    </>
  );
}

function SpanSelection({
  start,
  end,
  claimIndex,
  tokens,
  sourceEvidence,
  influenceAbsence,
  onSelectToken,
  onSelectSource,
  actions,
  scopeHref,
}: {
  start: number;
  end: number;
  claimIndex?: number;
  tokens: TokenReading[];
  sourceEvidence?: LensInspectorAvailability;
  influenceAbsence?: InfluenceAbsence | null;
  onSelectToken?: (index: number) => void;
  onSelectSource?: (sourceId: string) => void;
  actions?: ReactNode;
  scopeHref?: string;
}) {
  const summary = summarizeRange(tokens, start, end);
  const aggregates = aggregateSources(tokens, start, end);
  const unavailable = aggregates.length ? null : availabilityView(sourceEvidence) ?? absenceView(influenceAbsence);

  return (
    <>
      <section className="lens-selection-readout">
        <span>{claimIndex == null ? "Selected token span" : `Claim ${claimIndex + 1}`}</span>
        <h3>{summary.text || "Empty recorded span"}</h3>
      </section>
      <dl className="lens-selection-facts">
        <div><dt>Position</dt><dd>{summary.start + 1}–{summary.end + 1}</dd></div>
        <div><dt>Tokens</dt><dd>{summary.tokenCount}</dd></div>
        <div><dt>Mean confidence</dt><dd>{summary.meanConfidence?.toFixed(4) ?? "Not captured"}</dd></div>
        <div><dt>Shaky tokens</dt><dd>{summary.shakyCount}</dd></div>
        <div><dt>Source-linked</dt><dd>{summary.linkedCount} / {summary.tokenCount}</dd></div>
      </dl>

      {aggregates.length > 0 ? (
        <section className="lens-selection-aggregate-evidence" aria-label="Span source evidence">
          <header><span>Source evidence across span</span><b>{aggregates.length}</b></header>
          <div>
            {aggregates.map((source) => {
              const state = source.clearTokenCount && source.observedTokenCount
                ? "mixed"
                : source.clearTokenCount ? "cleared" : "observed";
              const content = (
                <>
                  <span className="lens-selection-evidence-dot" aria-hidden="true" />
                  <span className="lens-selection-evidence-copy">
                    <strong>{source.label}</strong>
                    <small>
                      {source.clearTokenCount ? `${source.clearTokenCount} cleared` : ""}
                      {source.clearTokenCount && source.observedTokenCount ? " · " : ""}
                      {source.observedTokenCount ? `${source.observedTokenCount} below floor` : ""}
                    </small>
                  </span>
                  <output>{signedNats(source.deltaNats)}</output>
                </>
              );
              return onSelectSource ? (
                <button type="button" className={`lens-selection-evidence-row is-${state}`} onClick={() => onSelectSource(source.sourceId)} key={source.sourceId}>{content}</button>
              ) : <div className={`lens-selection-evidence-row is-${state}`} key={source.sourceId}>{content}</div>;
            })}
          </div>
        </section>
      ) : <AvailabilityNotice value={unavailable} />}

      <InspectorActions scopeHref={scopeHref}>
        {onSelectToken && summary.tokenCount > 0 && (
          <button type="button" onClick={() => onSelectToken(weakestTokenInRange(tokens, summary.start, summary.end))}>
            Select lowest-confidence token
          </button>
        )}
        {actions}
      </InspectorActions>
    </>
  );
}

function TokenSelection({
  index,
  tokens,
  tokenTrace,
  sourceEvidence,
  influenceAbsence,
  onSelectSource,
  actions,
  scopeHref,
}: {
  index: number;
  tokens: TokenReading[];
  tokenTrace?: LensInspectorAvailability;
  sourceEvidence?: LensInspectorAvailability;
  influenceAbsence?: InfluenceAbsence | null;
  onSelectSource?: (sourceId: string) => void;
  actions?: ReactNode;
  scopeHref?: string;
}) {
  const token = tokens[index];
  if (!token) {
    return (
      <>
        <section className="lens-selection-readout">
          <span>Token selection</span>
          <h3>Token trace unavailable</h3>
        </section>
        <AvailabilityNotice value={availabilityView(tokenTrace ?? {
          state: "not_captured",
          detail: "This run did not retain the token trace required for token inspection.",
        })} />
        <InspectorActions scopeHref={scopeHref}>{actions}</InspectorActions>
      </>
    );
  }

  const cleared = (token.sources ?? []).filter((source) => source.evidenceState === "causally_supported");
  const observed = [
    ...(token.sources ?? []).filter((source) => source.evidenceState === "observed"),
    ...(token.observedSources ?? []),
  ];
  const noLinkAvailability = cleared.length || observed.length
    ? null
    : availabilityView(sourceEvidence) ?? absenceView(influenceAbsence) ?? {
      label: "Not measured",
      detail: "No source measurement is recorded for this token.",
      className: "is-not-measured",
    };

  return (
    <>
      <section className="lens-selection-token-readout">
        <span>Recorded output token</span>
        <h3>{token.text || "∅"}</h3>
        <b className={`band-${token.band ?? "none"}`}>{token.band ? humanize(token.band) : "Unbanded"}</b>
      </section>
      <dl className="lens-selection-facts">
        <div><dt>Position</dt><dd>{index + 1} / {tokens.length}</dd></div>
        <div><dt>Confidence</dt><dd>{token.confidence?.toFixed(4) ?? "Not captured"}</dd></div>
        <div><dt>Top-k entropy</dt><dd>{Number.isFinite(token.entropy) ? `${token.entropy.toFixed(4)} bits` : "Not captured"}</dd></div>
        <div><dt>Alternatives</dt><dd>{token.alternatives?.length ?? 0}</dd></div>
      </dl>

      <EvidenceGroup title="Cleared links" sources={cleared} onSelectSource={onSelectSource} />
      <EvidenceGroup title="Observed below floor" sources={observed} onSelectSource={onSelectSource} />
      <AvailabilityNotice value={noLinkAvailability} />

      {token.alternatives?.length ? (
        <section className="lens-selection-alternatives" aria-label="Recorded token alternatives">
          <header><span>Recorded alternatives</span><b>{token.alternatives.length}</b></header>
          <div>{token.alternatives.slice(0, 5).map((candidate, candidateIndex) => (
            <p key={`${candidate.token}-${candidateIndex}`}><code>{candidate.token || "∅"}</code><span>{candidate.score.toFixed(3)}</span></p>
          ))}</div>
        </section>
      ) : null}
      <InspectorActions scopeHref={scopeHref}>{actions}</InspectorActions>
    </>
  );
}

/**
 * Selection-driven Debug inspector. It presents only the selected source, span/claim, or token plus
 * evidence directly related to it; data loading and URL selection are deliberately owned by the host.
 */
export function LensSelectionInspector({
  selection,
  tokens = [],
  sources = [],
  influenceAbsence,
  influenceMethod,
  influenceThresholds,
  contextCoverage,
  tokenTrace,
  sourceEvidence,
  onSelectToken,
  onSelectSource,
  actions,
  events,
  children,
  scopeHref,
  className,
}: LensSelectionInspectorProps) {
  const recordedTokens = [...tokens];
  const selectedSource = selection?.kind === "source"
    ? selection.source ?? sources.find((source) => source.id === selection.sourceId)
    : undefined;
  const selectionTitle = selection?.kind === "source"
    ? "Source"
    : selection?.kind === "span"
      ? selection.claimIndex == null ? "Token span" : "Claim"
      : selection?.kind === "token"
        ? "Token"
        : "Selection";

  return (
    <aside className={["lens-selection-inspector", className].filter(Boolean).join(" ")} aria-labelledby="lens-selection-inspector-title">
      <header className="lens-selection-inspector-header">
        <span>Selection inspector</span>
        <h2 id="lens-selection-inspector-title">{selectionTitle}</h2>
      </header>

      <div className="lens-selection-inspector-body">
        {!selection ? (
          <section className="lens-selection-empty">
            <h3>Select an object</h3>
            <p>Select a source, claim/span, or response token to inspect its recorded and measured evidence.</p>
          </section>
        ) : selection.kind === "source" && selectedSource ? (
          <SourceSelection
            source={selectedSource}
            tokens={recordedTokens}
            sourceEvidence={sourceEvidence}
            influenceAbsence={influenceAbsence}
            onSelectToken={onSelectToken}
            onSelectSource={onSelectSource}
            actions={actions}
            scopeHref={scopeHref}
          />
        ) : selection.kind === "source" ? (
          <>
            <section className="lens-selection-readout"><span>Context source</span><h3>Source capture unavailable</h3></section>
            <AvailabilityNotice value={availabilityView(sourceEvidence ?? {
              state: "not_captured",
              detail: "The selected source is not present in this run's captured context.",
            })} />
            <InspectorActions scopeHref={scopeHref}>{actions}</InspectorActions>
          </>
        ) : selection.kind === "span" ? (
          <SpanSelection
            {...selection}
            tokens={recordedTokens}
            sourceEvidence={sourceEvidence}
            influenceAbsence={influenceAbsence}
            onSelectToken={onSelectToken}
            onSelectSource={onSelectSource}
            actions={actions}
            scopeHref={scopeHref}
          />
        ) : (
          <TokenSelection
            index={selection.index}
            tokens={recordedTokens}
            tokenTrace={tokenTrace}
            sourceEvidence={sourceEvidence}
            influenceAbsence={influenceAbsence}
            onSelectSource={onSelectSource}
            actions={actions}
            scopeHref={scopeHref}
          />
        )}

        {selection && <MeasurementProvenance method={influenceMethod} thresholds={influenceThresholds} coverage={contextCoverage} />}
        {events && <section className="lens-selection-inspector-events" aria-label="Related recorded events">{events}</section>}
        {children && <section className="lens-selection-inspector-extra" aria-label="Additional selection evidence">{children}</section>}
      </div>
    </aside>
  );
}
