import { EvidenceMark } from "./EvidenceMark";
import "./SpanWaterfall.css";

/**
 * C3 -- a clock-domain-aware timing readout. This deliberately accepts a small, independent view of
 * phases rather than `RunPerformance` / `PerformanceRuleReport`: a timing visualization must not become
 * coupled to the route response that happens to use it first. The aggregation is equally independent
 * because it is duration arithmetic, not another phase on any process's clock.
 *
 * The important boundary is spatial as well as type-level. Gateway and worker monotonic clocks do not
 * share an origin, so this component groups by `clockOwner` and only lays out `startNs` beside spans in
 * that exact local `clockDomain`. Similar-looking left edges in different lanes are never an alignment
 * claim; the visible note below is part of the component rather than caller-supplied copy so it cannot
 * be accidentally dropped at a future call site.
 */

export type SpanWaterfallMeasurement = "measured" | "estimated";

export type SpanWaterfallPhaseAggregation = "exclusive" | "overlapping" | "context_only";

/** One recorded timing span. `durationNs` is optional on purpose: absence is evidence about capture,
 * not an implicit zero duration, and renders as an EvidenceMark instead of a collapsed bar. */
export interface SpanWaterfallPhase {
  id?: string;
  name: string;
  durationNs?: number;
  startNs?: number;
  owner?: string;
  clockOwner?: string;
  clockDomain?: string;
  measurement?: SpanWaterfallMeasurement;
  aggregation?: SpanWaterfallPhaseAggregation;
  scope?: string;
  includes?: readonly string[];
}

/** Whole-request accounting. These values are intentionally not nested in `SpanWaterfallPhase`: adding
 * them to a clock lane would imply a shared start position that the trace schema expressly does not have. */
export interface SpanWaterfallAggregation {
  knownDurationNs?: number;
  unaccountedDurationNs?: number;
  wallClockTotalNs?: number;
  measurementCoverage?: number;
  consistency?: "consistent" | "known_exceeds_wall";
}

export interface SpanWaterfallProps {
  phases: readonly SpanWaterfallPhase[];
  aggregation?: SpanWaterfallAggregation;
  title?: string;
  className?: string;
}

interface IndexedPhase {
  phase: SpanWaterfallPhase;
  index: number;
}

interface ClockDomainGroup {
  domain: string;
  phases: IndexedPhase[];
}

interface ClockOwnerLane {
  owner: string;
  domains: ClockDomainGroup[];
}

interface PhasePresentation {
  measurement: SpanWaterfallMeasurement;
  aggregation: SpanWaterfallPhaseAggregation;
  processStartup: boolean;
  excludedFromKnown: boolean;
  exclusionLabels: string[];
}

const UNKNOWN_CLOCK_OWNER = "Clock owner not recorded";
const UNKNOWN_CLOCK_DOMAIN = "Clock domain not recorded";
const DURATION_ABSENCE_REASON = "This span has no recorded duration, so it cannot be drawn as a zero-width bar.";

function recordedNonNegative(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function recordedLabel(value: string | undefined, fallback: string): string {
  const trimmed = value?.trim();
  return trimmed || fallback;
}

/** The conversion happens exactly once, before either the bar label or the accounting key is rendered.
 * Keeping the threshold in nanoseconds avoids the common bug of treating raw ns as if they were ms. */
export function formatDurationNs(durationNs: number | undefined): string {
  if (!recordedNonNegative(durationNs)) return "Not recorded";
  if (durationNs < 1_000_000_000) {
    const milliseconds = durationNs / 1_000_000;
    return `${Number.isInteger(milliseconds) ? milliseconds : milliseconds.toFixed(2).replace(/0+$/, "").replace(/\.$/, "")} ms`;
  }
  return `${(durationNs / 1_000_000_000).toFixed(2)} s`;
}

// Kept as a named alias for callers that use the trace schema's `duration_ns` vocabulary directly.
export const formatNanoseconds = formatDurationNs;

function formatCoverage(coverage: number): string {
  const percentage = coverage * 100;
  const display = Number.isInteger(percentage)
    ? percentage
    : percentage.toFixed(1).replace(/0+$/, "").replace(/\.$/, "");
  return `${display}%`;
}

function phasePresentation(phase: SpanWaterfallPhase): PhasePresentation {
  // The trace producer defaults omitted fields to measured/exclusive for backwards-compatible worker
  // frames. Preserve that contract here, while treating process startup as outside the in-request total
  // even if a malformed future caller mistakenly labels it exclusive.
  const measurement = phase.measurement ?? "measured";
  const aggregation = phase.aggregation ?? "exclusive";
  const scope = `${phase.scope ?? ""} ${phase.name}`.toLowerCase().replaceAll(/[\s-]+/g, "_");
  const processStartup = /(^|_)process_?startup($|_)/.test(scope) || /(^|_)startup($|_)/.test(scope);
  const excludedFromKnown = measurement !== "measured"
    || aggregation !== "exclusive"
    || processStartup;
  const exclusionLabels = [
    aggregation === "overlapping" ? "Overlapping" : undefined,
    aggregation === "context_only" ? "Context only" : undefined,
    processStartup ? "Process startup" : undefined,
    measurement === "estimated" ? "Estimated" : undefined,
  ].filter((label): label is string => Boolean(label));

  return { measurement, aggregation, processStartup, excludedFromKnown, exclusionLabels };
}

function groupClockOwners(phases: readonly SpanWaterfallPhase[]): ClockOwnerLane[] {
  const lanes: ClockOwnerLane[] = [];

  phases.forEach((phase, index) => {
    const owner = recordedLabel(phase.clockOwner, UNKNOWN_CLOCK_OWNER);
    const domain = recordedLabel(phase.clockDomain, UNKNOWN_CLOCK_DOMAIN);
    let lane = lanes.find((candidate) => candidate.owner === owner);
    if (!lane) {
      lane = { owner, domains: [] };
      lanes.push(lane);
    }
    let domainGroup = lane.domains.find((candidate) => candidate.domain === domain);
    if (!domainGroup) {
      domainGroup = { domain, phases: [] };
      lane.domains.push(domainGroup);
    }
    domainGroup.phases.push({ phase, index });
  });

  return lanes;
}

function domainExtent(phases: readonly IndexedPhase[]): number {
  // A domain's scale is local. We intentionally do not take a max across owner lanes: doing so would
  // turn unrelated monotonic origins into a visual synchronization claim.
  const extent = phases.reduce((largest, { phase }) => {
    if (!recordedNonNegative(phase.durationNs)) return largest;
    const end = recordedNonNegative(phase.startNs) ? phase.startNs + phase.durationNs : phase.durationNs;
    return Math.max(largest, end);
  }, 0);
  return Math.max(extent, 1);
}

function barStyle(phase: SpanWaterfallPhase, extentNs: number) {
  const durationNs = phase.durationNs;
  if (!recordedNonNegative(durationNs)) return undefined;
  const width = Math.min(100, (durationNs / extentNs) * 100);
  const startNs = phase.startNs;
  return recordedNonNegative(startNs)
    ? { marginInlineStart: `${Math.min(100, (startNs / extentNs) * 100)}%`, width: `${width}%` }
    : { width: `${width}%` };
}

function phaseClassName(presentation: PhasePresentation): string {
  return [
    "span-waterfall-bar",
    presentation.aggregation === "overlapping" && "is-overlapping",
    presentation.aggregation === "context_only" && "is-context-only",
    presentation.processStartup && "is-process-startup",
    presentation.measurement === "estimated" && "is-estimated",
  ].filter(Boolean).join(" ");
}

function phaseTooltip(phase: SpanWaterfallPhase, presentation: PhasePresentation): string {
  return [
    phase.name.replaceAll("_", " "),
    formatDurationNs(phase.durationNs),
    `measurement: ${presentation.measurement}`,
    `clockDomain: ${recordedLabel(phase.clockDomain, UNKNOWN_CLOCK_DOMAIN)}`,
    `aggregation: ${presentation.aggregation}`,
    presentation.excludedFromKnown ? "excluded from known in-request arithmetic" : "included in known in-request arithmetic",
  ].join(" · ");
}

function PhaseRow({ item, extentNs }: { item: IndexedPhase; extentNs: number }) {
  const { phase, index } = item;
  const presentation = phasePresentation(phase);
  const hasDuration = recordedNonNegative(phase.durationNs);
  const phaseKey = phase.id ?? `${phase.name}-${index}`;
  const hasStartOffset = recordedNonNegative(phase.startNs);
  // Older traces legitimately omit `includes`; treating that as an empty list preserves their meaning
  // without manufacturing a second absence treatment for a field that is merely supplementary detail.
  const includes = phase.includes ?? [];

  return (
    <article className="span-waterfall-phase" data-phase-id={phaseKey}>
      <div className="span-waterfall-phase-heading">
        <strong>{phase.name.replaceAll("_", " ")}</strong>
        {phase.scope && <span>{phase.scope.replaceAll("_", " ")}</span>}
      </div>
      <div className="span-waterfall-phase-timeline">
        <div className="span-waterfall-phase-track" aria-label={`${phase.name} local clock placement`}>
          {hasDuration ? (
            <span
              className={phaseClassName(presentation)}
              data-start-offset-ns={hasStartOffset ? phase.startNs : undefined}
              style={barStyle(phase, extentNs)}
              title={phaseTooltip(phase, presentation)}
            />
          ) : (
            <EvidenceMark
              variant="chip"
              state="not_measured"
              label="Duration not recorded"
              reason={DURATION_ABSENCE_REASON}
            />
          )}
        </div>
        <b className="span-waterfall-phase-duration">{formatDurationNs(phase.durationNs)}</b>
      </div>
      <div className="span-waterfall-phase-facts">
        {phase.owner && <span>owner: {phase.owner}</span>}
        <span>measurement: {presentation.measurement}</span>
        <span>aggregation: {presentation.aggregation.replaceAll("_", " ")}</span>
        {includes.length > 0 && <span>includes: {includes.join(", ")}</span>}
        {!hasStartOffset && hasDuration && <span>Start offset not recorded</span>}
        {presentation.exclusionLabels.map((label) => <span key={label}>{label}</span>)}
        {presentation.excludedFromKnown && <strong>Excluded from known in-request arithmetic</strong>}
      </div>
    </article>
  );
}

function ClockDomain({ group }: { group: ClockDomainGroup }) {
  const extentNs = domainExtent(group.phases);
  const hasRecordedOffset = group.phases.some(({ phase }) => recordedNonNegative(phase.startNs));

  return (
    <div className="span-waterfall-domain" role="group" aria-label={`Clock domain: ${group.domain}`}>
      <header className="span-waterfall-domain-header">
        <span>Clock domain</span>
        <strong>{group.domain}</strong>
      </header>
      <p className="span-waterfall-domain-note">
        {hasRecordedOffset
          ? "Start offsets are positioned only within this local clock domain."
          : "No start offsets were recorded; these bars show duration only, not a shared sequence."}
      </p>
      <div className="span-waterfall-domain-track">
        {group.phases.map((item) => <PhaseRow item={item} extentNs={extentNs} key={item.phase.id ?? `${item.phase.name}-${item.index}`} />)}
      </div>
    </div>
  );
}

function Accounting({ aggregation }: { aggregation: SpanWaterfallAggregation }) {
  const knownDurationNs = recordedNonNegative(aggregation.knownDurationNs) ? aggregation.knownDurationNs : undefined;
  const unaccountedDurationNs = recordedNonNegative(aggregation.unaccountedDurationNs)
    ? aggregation.unaccountedDurationNs
    : undefined;
  const wallClockTotalNs = recordedNonNegative(aggregation.wallClockTotalNs) ? aggregation.wallClockTotalNs : undefined;
  // The accounting strip represents the additive duration ledger, not the largest individual entry.
  // A missing (or smaller) wall total must not stretch known time to 100% and hide the unaccounted gap.
  const additiveDurationNs = (knownDurationNs ?? 0) + (unaccountedDurationNs ?? 0);
  const total = Math.max(additiveDurationNs, wallClockTotalNs ?? 0, 1);
  const knownWidth = knownDurationNs == null ? 0 : (knownDurationNs / total) * 100;
  const unaccountedWidth = unaccountedDurationNs == null ? 0 : (unaccountedDurationNs / total) * 100;
  const measurementCoverage = recordedNonNegative(aggregation.measurementCoverage)
    ? aggregation.measurementCoverage
    : undefined;

  return (
    <section className="span-waterfall-accounting" aria-label="Request accounting">
      <header>
        <div>
          <span>Request accounting</span>
          <strong>Duration totals only</strong>
        </div>
        {wallClockTotalNs != null && <b>{formatDurationNs(wallClockTotalNs)} wall total</b>}
      </header>
      <p>
        Known and unaccounted durations are additive accounting, not a shared cross-process timeline.
      </p>
      <div className="span-waterfall-accounting-track" role="img" aria-label="Known and unaccounted request duration">
        {knownDurationNs != null && (
          <span
            className="span-waterfall-accounting-known"
            style={{ width: `${knownWidth}%` }}
            title={`Known measured duration: ${formatDurationNs(knownDurationNs)}`}
          />
        )}
        {unaccountedDurationNs != null && unaccountedDurationNs > 0 && (
          <span
            className="span-waterfall-accounting-gap"
            style={{ width: `${unaccountedWidth}%` }}
            title={`Unaccounted gap: ${formatDurationNs(unaccountedDurationNs)}`}
          />
        )}
      </div>
      <dl className="span-waterfall-accounting-key">
        <div>
          <dt>Known measured duration</dt>
          <dd>{formatDurationNs(knownDurationNs)}</dd>
        </div>
        {unaccountedDurationNs != null ? (
          <div className="is-unaccounted">
            <dt>Unaccounted gap</dt>
            <dd>{formatDurationNs(unaccountedDurationNs)}</dd>
          </div>
        ) : (
          <div className="is-unaccounted">
            <dt>Unaccounted gap</dt>
            <dd>
              <EvidenceMark
                state="not_measured"
                label="Not recorded"
                reason="The trace did not record an unaccounted duration."
              />
            </dd>
          </div>
        )}
        {measurementCoverage != null && (
          <div>
            <dt>Measurement coverage</dt>
            <dd>{formatCoverage(measurementCoverage)}</dd>
          </div>
        )}
        {aggregation.consistency && (
          <div>
            <dt>Consistency</dt>
            <dd>{aggregation.consistency.replaceAll("_", " ")}</dd>
          </div>
        )}
      </dl>
    </section>
  );
}

export function SpanWaterfall({ phases, aggregation, title = "Timing waterfall", className }: SpanWaterfallProps) {
  const lanes = groupClockOwners(phases);

  return (
    <section className={["span-waterfall", className].filter(Boolean).join(" ")} aria-label={title}>
      <header className="span-waterfall-header">
        <div>
          <span>Recorded timing spans</span>
          <h3>{title}</h3>
        </div>
      </header>
      <p className="span-waterfall-boundary">
        Separate clock-owner lanes are not mutually aligned. Start offsets are preserved only within one local clock domain.
      </p>
      {aggregation && <Accounting aggregation={aggregation} />}
      <div className="span-waterfall-lanes">
        {lanes.map((lane) => (
          <section className="span-waterfall-lane" aria-label={`Clock owner: ${lane.owner}`} key={lane.owner}>
            <header className="span-waterfall-lane-header">
              <span>Clock owner</span>
              <strong>{lane.owner}</strong>
            </header>
            <div className="span-waterfall-lane-domains">
              {lane.domains.map((group) => <ClockDomain group={group} key={group.domain} />)}
            </div>
          </section>
        ))}
        {!lanes.length && <p className="span-waterfall-empty">No timing spans were recorded.</p>}
      </div>
    </section>
  );
}
