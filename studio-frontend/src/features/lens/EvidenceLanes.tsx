import type { CSSProperties, MouseEvent } from "react";
import type { TokenReading } from "../../data/types";
import "./EvidenceLanes.css";

/** A closed, inclusive response-token interval. */
export interface EvidenceTokenRange {
  start: number;
  end: number;
}

export interface EvidenceLaneSelection {
  tokenIndex: number;
  range: EvidenceTokenRange | null;
}

/**
 * Pass this when the caller knows that a lane was not produced. In particular, an unavailable
 * source map must not be represented by an empty source array: the latter means the map ran and
 * found no links for a token.
 */
export interface EvidenceAvailability {
  available: boolean;
  reason?: string;
}

export type SemanticEvidenceEventKind =
  | "boundary"
  | "claim"
  | "citation"
  | "tool"
  | "warning"
  | "custom";

/** A recorded event whose token interval can be selected from the lane. */
export interface SemanticEvidenceEvent {
  id: string;
  label: string;
  startToken: number;
  endToken?: number;
  detail?: string;
  kind?: SemanticEvidenceEventKind;
}

/** The terminal state of a recorded response. `tokenIndex` defaults to the last response token. */
export interface EvidenceFinishMarker {
  reason?: string | null;
  truncated?: boolean;
  tokenIndex?: number;
  detail?: string;
}

export interface EvidenceLanesProps {
  tokens: readonly TokenReading[];
  selectedToken: number;
  selectedRange?: EvidenceTokenRange | null;
  /** Optional anchor used when a lane is shift-clicked to extend an existing selection. */
  rangeAnchor?: number;
  /** Primary controlled-selection callback. It fires for token, range, event, and finish-marker clicks. */
  onSelectionChange?: (selection: EvidenceLaneSelection) => void;
  /** Optional convenience callbacks for hosts that keep token and range state separately. */
  onSelectToken?: (tokenIndex: number) => void;
  onSelectRange?: (range: EvidenceTokenRange | null) => void;
  confidenceAvailability?: EvidenceAvailability;
  entropyAvailability?: EvidenceAvailability;
  sourceAvailability?: EvidenceAvailability;
  semanticEventsAvailability?: EvidenceAvailability;
  semanticEvents?: readonly SemanticEvidenceEvent[];
  finish?: EvidenceFinishMarker | null;
  onSelectSemanticEvent?: (event: SemanticEvidenceEvent) => void;
  className?: string;
}

type NormalizedEvent = SemanticEvidenceEvent & EvidenceTokenRange;

const EMPTY_RANGE: EvidenceTokenRange | null = null;

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function clampIndex(index: number, tokenCount: number) {
  return Math.max(0, Math.min(tokenCount - 1, Math.round(index)));
}

function normalizeRange(range: EvidenceTokenRange | null | undefined, tokenCount: number) {
  if (!range || tokenCount === 0 || !finite(range.start) || !finite(range.end)) return EMPTY_RANGE;
  return {
    start: clampIndex(Math.min(range.start, range.end), tokenCount),
    end: clampIndex(Math.max(range.start, range.end), tokenCount),
  };
}

function resolvesAvailability(
  supplied: EvidenceAvailability | undefined,
  fallbackAvailable: boolean,
  fallbackReason: string,
): EvidenceAvailability {
  if (supplied) return supplied;
  return fallbackAvailable ? { available: true } : { available: false, reason: fallbackReason };
}

function sourceLinks(token: TokenReading) {
  return token.sources ?? [];
}

function observedSourceLinks(token: TokenReading) {
  return token.observedSources ?? [];
}

function dominantSource(token: TokenReading) {
  return [...sourceLinks(token)].sort((left, right) => Math.abs(right.deltaNats) - Math.abs(left.deltaNats))[0];
}

function sourceCellLabel(token: TokenReading, index: number) {
  const links = sourceLinks(token);
  const observed = observedSourceLinks(token);
  if (links.length) {
    const dominant = dominantSource(token);
    const total = links.reduce((sum, link) => sum + link.deltaNats, 0);
    return `Source influence token ${index + 1}, ${links.length} cleared ${links.length === 1 ? "link" : "links"}, ${
      dominant?.effect ?? "neutral"
    }, ${total >= 0 ? "+" : ""}${total.toFixed(3)} nats`;
  }
  if (observed.length) {
    return `Source influence token ${index + 1}, measured below the evidence floor, ${observed.length} ${
      observed.length === 1 ? "link" : "links"
    }`;
  }
  return `Source influence token ${index + 1}, no cleared source link`;
}

function confidenceLabel(token: TokenReading | undefined, index: number) {
  if (!finite(token?.confidence)) return `Confidence token ${index + 1}, unavailable`;
  return `Confidence token ${index + 1}, ${Math.round(token.confidence * 100)} percent`;
}

function entropyLabel(token: TokenReading | undefined, index: number) {
  if (!finite(token?.entropy)) return `Entropy and shakiness token ${index + 1}, unavailable`;
  return `Entropy and shakiness token ${index + 1}, ${token.entropy.toFixed(3)} bits, ${
    token.band ?? "unbanded"
  }`;
}

function selectionClass(index: number, selectedToken: number, selectedRange: EvidenceTokenRange | null) {
  return [
    "evidence-lanes-cell",
    index === selectedToken ? "is-selected" : "",
    selectedRange && index >= selectedRange.start && index <= selectedRange.end ? "is-in-range" : "",
  ].filter(Boolean).join(" ");
}

function readableReason(reason: string | null | undefined) {
  return reason ? reason.replaceAll("_", " ").toUpperCase() : "REASON NOT RECORDED";
}

function laneStyle(tokenCount: number): CSSProperties {
  return { "--evidence-token-count": tokenCount } as CSSProperties;
}

function intersects(event: EvidenceTokenRange, selection: EvidenceTokenRange) {
  return event.start <= selection.end && event.end >= selection.start;
}

function availabilityText(availability: EvidenceAvailability, defaultText: string) {
  return availability.reason || defaultText;
}

/**
 * A compact, controlled companion to a response reader. Every lane has exactly the same token grid,
 * so a token or interval selected in one is visible in every other lane.
 */
export function EvidenceLanes({
  tokens,
  selectedToken,
  selectedRange,
  rangeAnchor,
  onSelectionChange,
  onSelectToken,
  onSelectRange,
  confidenceAvailability,
  entropyAvailability,
  sourceAvailability,
  semanticEventsAvailability,
  semanticEvents,
  finish,
  onSelectSemanticEvent,
  className,
}: EvidenceLanesProps) {
  const tokenCount = tokens.length;
  const activeToken = tokenCount ? clampIndex(selectedToken, tokenCount) : 0;
  const activeRange = normalizeRange(selectedRange, tokenCount);
  const activeSelection: EvidenceTokenRange = activeRange ?? { start: activeToken, end: activeToken };
  const activeRangeAnchor = tokenCount
    ? clampIndex(rangeAnchor ?? activeToken, tokenCount)
    : 0;
  const canSelect = Boolean(onSelectionChange || onSelectToken || onSelectRange);
  const confidence = resolvesAvailability(
    confidenceAvailability,
    tokens.some((token) => finite(token.confidence)),
    "Confidence was not recorded for this response.",
  );
  const entropy = resolvesAvailability(
    entropyAvailability,
    tokens.some((token) => finite(token.entropy)),
    "Entropy was not recorded for this response.",
  );
  const sources = resolvesAvailability(
    sourceAvailability,
    tokens.some((token) => sourceLinks(token).length > 0 || observedSourceLinks(token).length > 0),
    "Source influence was not measured for this response.",
  );
  const events = resolvesAvailability(
    semanticEventsAvailability,
    semanticEvents !== undefined,
    "No semantic event trace was recorded for this response.",
  );
  const finishAvailability = resolvesAvailability(
    undefined,
    finish !== undefined && finish !== null,
    "The response finish marker was not recorded.",
  );
  const maxEntropy = Math.max(0.001, ...tokens.flatMap((token) => finite(token.entropy) ? [token.entropy] : []));
  const normalizedEvents: NormalizedEvent[] = (semanticEvents ?? []).flatMap((event) => {
    if (tokenCount === 0 || !finite(event.startToken)) return [];
    const range = normalizeRange({ start: event.startToken, end: event.endToken ?? event.startToken }, tokenCount);
    return range ? [{ ...event, ...range }] : [];
  });
  const eventsAtToken = (index: number) => normalizedEvents.filter((event) => index >= event.start && index <= event.end);
  const eventsInSelection = normalizedEvents.filter((event) => intersects(event, activeSelection));
  const finishIndex = finish && tokenCount ? clampIndex(finish.tokenIndex ?? tokenCount - 1, tokenCount) : null;

  function changeSelection(tokenIndex: number, range: EvidenceTokenRange | null) {
    onSelectionChange?.({ tokenIndex, range });
    onSelectToken?.(tokenIndex);
    onSelectRange?.(range);
  }

  function selectToken(index: number, event?: MouseEvent<HTMLButtonElement>) {
    const extend = Boolean(event?.shiftKey);
    const range = extend
      ? {
        start: Math.min(activeRangeAnchor, index),
        end: Math.max(activeRangeAnchor, index),
      }
      : null;
    changeSelection(index, range);
  }

  function selectEvent(event: NormalizedEvent) {
    const range = event.start === event.end ? null : { start: event.start, end: event.end };
    changeSelection(event.start, range);
    onSelectSemanticEvent?.(event);
  }

  const selectedTokenData = tokens[activeToken];
  const selectedSourceLinks = selectedTokenData ? sourceLinks(selectedTokenData) : [];
  const selectedObservedLinks = selectedTokenData ? observedSourceLinks(selectedTokenData) : [];
  const selectionLabel = activeRange
    ? `Tokens ${activeRange.start + 1}–${activeRange.end + 1}`
    : `Token ${activeToken + 1}`;

  return (
    <section className={["evidence-lanes", className].filter(Boolean).join(" ")} aria-label="Evidence lanes">
      <header className="evidence-lanes-head">
        <output aria-live="polite">{tokenCount ? selectionLabel.toUpperCase() : "NO TOKEN TRACE"}</output>
      </header>

      {!tokenCount ? (
        <p className="evidence-lanes-empty">No token-level evidence was recorded for this response.</p>
      ) : (
        <div className="evidence-lanes-scroll">
          <div className="evidence-lanes-grid" style={laneStyle(tokenCount)}>
            <section className="evidence-lanes-lane" aria-labelledby="evidence-confidence-title">
              <header>
                <div>
                  <strong id="evidence-confidence-title">CONFIDENCE</strong>
                  <span>RECORDED TOKEN PROBABILITY</span>
                </div>
                <output>
                  {confidence.available
                    ? finite(selectedTokenData?.confidence)
                      ? `${Math.round(selectedTokenData.confidence * 100)}% · ${selectedTokenData.confidence.toFixed(3)}`
                      : "UNAVAILABLE AT SELECTION"
                    : "UNAVAILABLE"}
                </output>
              </header>
              {confidence.available ? (
                <div className="evidence-lanes-track" role="group" aria-label="Confidence by response token">
                  {tokens.map((token, index) => {
                    const value = finite(token.confidence) ? Math.max(0, Math.min(1, token.confidence)) : 0;
                    return (
                      <button
                        type="button"
                        className={`${selectionClass(index, activeToken, activeRange)} evidence-lanes-confidence-cell ${
                          finite(token.confidence) ? "" : "is-missing"
                        }`}
                        style={{ "--evidence-value": value } as CSSProperties}
                        aria-label={confidenceLabel(token, index)}
                        aria-pressed={index === activeToken}
                        disabled={!canSelect}
                        onClick={(event) => selectToken(index, event)}
                        key={`confidence-${index}`}
                      />
                    );
                  })}
                </div>
              ) : <p className="evidence-lanes-unavailable">CONFIDENCE UNAVAILABLE · {availabilityText(confidence, "Not recorded")}</p>}
            </section>

            <section className="evidence-lanes-lane" aria-labelledby="evidence-entropy-title">
              <header>
                <div>
                  <strong id="evidence-entropy-title">ENTROPY / SHAKINESS</strong>
                  <span>TOP-K ENTROPY · RECORDED BAND</span>
                </div>
                <output>
                  {entropy.available && finite(selectedTokenData?.entropy)
                    ? `${selectedTokenData.entropy.toFixed(3)} BITS · ${(selectedTokenData.band ?? "UNBANDED").toUpperCase()}`
                    : "UNAVAILABLE"}
                </output>
              </header>
              {entropy.available ? (
                <div className="evidence-lanes-track" role="group" aria-label="Entropy and shakiness by response token">
                  {tokens.map((token, index) => {
                    const value = finite(token.entropy) ? Math.max(0, Math.min(1, token.entropy / maxEntropy)) : 0;
                    return (
                      <button
                        type="button"
                        className={`${selectionClass(index, activeToken, activeRange)} evidence-lanes-entropy-cell band-${token.band ?? "none"} ${
                          finite(token.entropy) ? "" : "is-missing"
                        }`}
                        style={{ "--evidence-value": value } as CSSProperties}
                        aria-label={entropyLabel(token, index)}
                        aria-pressed={index === activeToken}
                        disabled={!canSelect}
                        onClick={(event) => selectToken(index, event)}
                        key={`entropy-${index}`}
                      />
                    );
                  })}
                </div>
              ) : <p className="evidence-lanes-unavailable">ENTROPY UNAVAILABLE · {availabilityText(entropy, "Not recorded")}</p>}
            </section>

            <section className="evidence-lanes-lane" aria-labelledby="evidence-source-title">
              <header>
                <div>
                  <strong id="evidence-source-title">SOURCE SUPPORT / INFLUENCE</strong>
                  <span>CLEARED LINKS · BELOW-FLOOR READINGS KEPT SEPARATE</span>
                </div>
                <output>
                  {sources.available
                    ? selectedSourceLinks.length
                      ? `${selectedSourceLinks.length} CLEARED · ${selectedSourceLinks.reduce((sum, link) => sum + link.deltaNats, 0).toFixed(3)} NATS`
                      : selectedObservedLinks.length
                        ? `${selectedObservedLinks.length} BELOW FLOOR`
                        : "NO CLEARED LINK"
                    : "UNAVAILABLE"}
                </output>
              </header>
              {sources.available ? (
                <div className="evidence-lanes-track" role="group" aria-label="Source influence by response token">
                  {tokens.map((token, index) => {
                    const links = sourceLinks(token);
                    const observed = observedSourceLinks(token);
                    const dominant = dominantSource(token);
                    return (
                      <button
                        type="button"
                        className={`${selectionClass(index, activeToken, activeRange)} evidence-lanes-source-cell ${
                          links.length ? `is-cleared effect-${dominant?.effect ?? "neutral"}` : observed.length ? "is-observed" : "is-empty"
                        }`}
                        aria-label={sourceCellLabel(token, index)}
                        aria-pressed={index === activeToken}
                        disabled={!canSelect}
                        onClick={(event) => selectToken(index, event)}
                        key={`source-${index}`}
                      >
                        {links.length ? <span>{links.length}</span> : observed.length ? <span>·</span> : null}
                      </button>
                    );
                  })}
                </div>
              ) : <p className="evidence-lanes-unavailable">SOURCE EVIDENCE UNAVAILABLE · {availabilityText(sources, "Not measured")}</p>}
            </section>

            <section className="evidence-lanes-lane" aria-labelledby="evidence-events-title">
              <header>
                <div>
                  <strong id="evidence-events-title">SEMANTIC EVENTS</strong>
                  <span>RECORDED BOUNDARIES, CLAIMS, TOOLS, OR WARNINGS</span>
                </div>
                <output>
                  {events.available
                    ? eventsInSelection.length
                      ? `${eventsInSelection.length} IN ${selectionLabel.toUpperCase()}`
                      : "NONE AT SELECTION"
                    : "UNAVAILABLE"}
                </output>
              </header>
              {events.available ? (
                <>
                  <div className="evidence-lanes-track" role="group" aria-label="Semantic events by response token">
                    {tokens.map((_, index) => {
                      const tokenEvents = eventsAtToken(index);
                      return (
                        <button
                          type="button"
                          className={`${selectionClass(index, activeToken, activeRange)} evidence-lanes-event-cell ${
                            tokenEvents.length ? `has-event event-${tokenEvents[0].kind ?? "custom"}` : ""
                          }`}
                          aria-label={tokenEvents.length
                            ? `Semantic event token ${index + 1}, ${tokenEvents.map((event) => event.label).join(", ")}`
                            : `Semantic event token ${index + 1}, none recorded`}
                          aria-pressed={index === activeToken}
                          disabled={!canSelect}
                          onClick={(event) => selectToken(index, event)}
                          key={`event-${index}`}
                        >{tokenEvents.length ? <span>{tokenEvents.length}</span> : null}</button>
                      );
                    })}
                  </div>
                  {normalizedEvents.length ? (
                    <div className="evidence-lanes-event-list" aria-label="Recorded semantic events">
                      {normalizedEvents.map((event) => (
                        <button
                          type="button"
                          className={`event-${event.kind ?? "custom"}`}
                          aria-pressed={intersects(event, activeSelection)}
                          disabled={!canSelect}
                          onClick={() => selectEvent(event)}
                          key={event.id}
                        >
                          <span>{event.kind ?? "custom"}</span>
                          <strong>{event.label}</strong>
                          <small>#{event.start + 1}{event.end === event.start ? "" : `–${event.end + 1}`}</small>
                        </button>
                      ))}
                    </div>
                  ) : <p className="evidence-lanes-recorded-empty">NO SEMANTIC EVENTS RECORDED</p>}
                </>
              ) : <p className="evidence-lanes-unavailable">SEMANTIC EVENTS UNAVAILABLE · {availabilityText(events, "Not recorded")}</p>}
            </section>

            <section className="evidence-lanes-lane evidence-lanes-finish" aria-labelledby="evidence-finish-title">
              <header>
                <div>
                  <strong id="evidence-finish-title">FINISH / TRUNCATION</strong>
                  <span>TERMINAL RESPONSE MARKER</span>
                </div>
                <output>
                  {finishAvailability.available
                    ? `${finish?.truncated ? "TRUNCATED · " : ""}${readableReason(finish?.reason)}`
                    : "UNAVAILABLE"}
                </output>
              </header>
              {finishAvailability.available && finishIndex !== null ? (
                <div className="evidence-lanes-track" role="group" aria-label="Response finish marker">
                  {tokens.map((_, index) => index === finishIndex ? (
                    <button
                      type="button"
                      className={`${selectionClass(index, activeToken, activeRange)} evidence-lanes-finish-cell ${
                        finish?.truncated ? "is-truncated" : ""
                      }`}
                      aria-label={`${finish?.truncated ? "Truncated" : "Finish"} marker at token ${index + 1}, ${
                        readableReason(finish?.reason).toLowerCase()
                      }`}
                      aria-pressed={index === activeToken}
                      disabled={!canSelect}
                      onClick={(event) => selectToken(index, event)}
                      key="finish-marker"
                    >
                      <span>{finish?.truncated ? "TRUNCATED" : "FINISH"}</span>
                    </button>
                  ) : <span className="evidence-lanes-finish-gap" aria-hidden="true" key={`finish-${index}`} />)}
                </div>
              ) : <p className="evidence-lanes-unavailable">FINISH MARKER UNAVAILABLE · {availabilityText(finishAvailability, "Not recorded")}</p>}
              {finish?.detail && <p className="evidence-lanes-finish-detail">{finish.detail}</p>}
            </section>
          </div>
        </div>
      )}
    </section>
  );
}
