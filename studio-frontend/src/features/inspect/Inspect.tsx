import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type RefObject,
} from "react";
import { routeHref, type InspectRoute } from "../../app/router";
import { describeError, isContractError } from "../../data/client";
import {
  type ContextInvestigationEffect,
  type ContextInvestigationLocus,
  type ContextInvestigationQuery,
  type ContextInvestigationReader,
  type ContextInvestigationSource,
  type ReaderStatus,
  type SourceMeasurementStatus,
  type UnicodeRange,
} from "../../data/contextInvestigation";
import { contextInvestigationApi } from "../../data/contextInvestigationClient";
import {
  isTerminalContextEffectsJob,
  type ContextEffectsJob,
} from "../../data/contextInvestigationJobs";
import {
  isTerminalLeanContextJob,
  type LeanContextJob,
} from "../../data/leanContext";
import {
  codePointLength,
  isValidCodePointRange,
  segmentAnnotatedText,
  sliceCodePoints,
  type AnnotatedTextSegment,
  type CodePointRange,
} from "./unicodeRanges";
import { MinimalContextSurface } from "./MinimalContext";
import { RunDetailsDisclosure } from "./ContextReceipt";
import "./inspect.css";

type Resource<T> =
  | { phase: "loading" }
  | { phase: "ready"; value: T }
  | { phase: "unavailable"; error: string }
  | { phase: "malformed"; error: string };

type OptionalResource<T> = { phase: "idle" } | Resource<T>;
type EvidenceDirection = "support" | "suppression" | "mixed";
type MobilePane = "context" | "answer";
type InvestigationView = "links" | "efficiency" | "lean";

interface ActiveRange extends CodePointRange {
  key: string;
}

interface RelationshipGeometry {
  sourceId: string;
  direction: EvidenceDirection;
  mapY: number;
  localY?: number;
  offscreen?: "above" | "below";
  registeredOnly: boolean;
}

interface CoverageGeometry {
  sourceId: string;
  status: SourceMeasurementStatus;
  mapY: number;
}

interface EvidenceGeometry {
  width: number;
  height: number;
  answerMapY: number;
  answerLocalY?: number;
  answerOffscreen?: "above" | "below";
  relationships: RelationshipGeometry[];
  coverage: CoverageGeometry[];
}

const EMPTY_GEOMETRY: EvidenceGeometry = {
  width: 112,
  height: 480,
  answerMapY: 240,
  relationships: [],
  coverage: [],
};

const MAX_TRANSIENT_PATHS = 6;

function useResource<T>(requestKey: string, loader: (signal: AbortSignal) => Promise<T>): Resource<T> {
  const [entry, setEntry] = useState<{ requestKey: string; resource: Resource<T> }>({
    requestKey,
    resource: { phase: "loading" },
  });
  useEffect(() => {
    const controller = new AbortController();
    setEntry({ requestKey, resource: { phase: "loading" } });
    void loader(controller.signal).then((value) => {
      if (!controller.signal.aborted) setEntry({ requestKey, resource: { phase: "ready", value } });
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setEntry({
        requestKey,
        resource: {
          phase: isContractError(error) ? "malformed" : "unavailable",
          error: describeError(error),
        },
      });
    });
    return () => controller.abort();
  }, [loader, requestKey]);
  return entry.requestKey === requestKey ? entry.resource : { phase: "loading" };
}

function useOptionalResource<T>(requestKey: string | undefined, loader?: (signal: AbortSignal) => Promise<T>): OptionalResource<T> {
  const [entry, setEntry] = useState<{ requestKey?: string; resource: OptionalResource<T> }>({
    requestKey,
    resource: { phase: loader ? "loading" : "idle" },
  });
  useEffect(() => {
    if (!loader || !requestKey) {
      setEntry({ requestKey: undefined, resource: { phase: "idle" } });
      return undefined;
    }
    const controller = new AbortController();
    setEntry({ requestKey, resource: { phase: "loading" } });
    void loader(controller.signal).then((value) => {
      if (!controller.signal.aborted) setEntry({ requestKey, resource: { phase: "ready", value } });
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setEntry({
        requestKey,
        resource: {
          phase: isContractError(error) ? "malformed" : "unavailable",
          error: describeError(error),
        },
      });
    });
    return () => controller.abort();
  }, [loader, requestKey]);
  const expectedKey = loader ? requestKey : undefined;
  return entry.requestKey === expectedKey ? entry.resource : loader ? { phase: "loading" } : { phase: "idle" };
}

function rangeKey(range: CodePointRange): string {
  return `${range.start}:${range.end}`;
}

function tupleRange(range: UnicodeRange): CodePointRange {
  return { start: range[0], end: range[1] };
}

function rangesEqual(left: CodePointRange | undefined, right: CodePointRange | undefined): boolean {
  return Boolean(left && right && left.start === right.start && left.end === right.end);
}

function rangesIntersect(left: CodePointRange, right: CodePointRange): boolean {
  return left.start < right.end && right.start < left.end;
}

function statusLabel(status: ReaderStatus): string {
  switch (status) {
    case "completed": return "Measured";
    case "partial": return "Partial";
    case "not_measured": return "Not measured";
    case "stale": return "Stale";
    case "unavailable": return "Unavailable";
  }
}

function sourceStatusLabel(status: SourceMeasurementStatus): string {
  switch (status) {
    case "measured": return "Measured";
    case "not_measured": return "Not measured";
    case "unavailable": return "Unavailable";
    case "failed": return "Failed";
  }
}

function directionLabel(direction: ContextInvestigationEffect["direction"]): string {
  switch (direction) {
    case "support": return "Support";
    case "suppression": return "Suppression";
    case "below_floor": return "Below floor";
    case "not_measured": return "Not measured";
    case "unavailable": return "Unavailable";
    case "failed": return "Failed";
  }
}

function isAboveFloorEffect(
  effect: ContextInvestigationEffect,
): effect is ContextInvestigationEffect & { direction: "support" | "suppression" } {
  return effect.direction === "support" || effect.direction === "suppression";
}

function shortId(value: string | null | undefined): string {
  if (!value) return "—";
  return value.length > 19 ? `${value.slice(0, 8)}…${value.slice(-7)}` : value;
}

function plural(count: number, singular: string, pluralForm = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : pluralForm}`;
}

function excerpt(value: string, limit = 64): string {
  const compact = value.replace(/\s+/g, " ").trim();
  return compact.length > limit ? `${compact.slice(0, limit - 1).trimEnd()}…` : compact;
}

function sourceInfluence(source: ContextInvestigationSource | undefined): number {
  if (!source || source.summary.totalContinuationDeltaNats === null) return 0;
  return Math.abs(source.summary.totalContinuationDeltaNats);
}

function heatDirection(value: number): EvidenceDirection {
  if (value > 0) return "support";
  if (value < 0) return "suppression";
  return "mixed";
}

function heatStyle(intensity: number): CSSProperties {
  const normalized = Math.max(0, Math.min(1, intensity));
  return { "--heat-alpha": `${Math.round(5 + normalized * 25)}%` } as CSSProperties;
}

function sourceRecordedText(
  reader: ContextInvestigationReader,
  source: ContextInvestigationSource,
): string {
  const block = reader.context.blocks.find((item) => item.messageIndex === source.messageIndex);
  return block
    ? sliceCodePoints(block.text, source.unicodeRange[0], source.unicodeRange[1])
    : source.label ?? source.sourceId;
}

function strongestSource(
  sourceIds: readonly string[],
  sourceById: ReadonlyMap<string, ContextInvestigationSource>,
): ContextInvestigationSource | undefined {
  return sourceIds
    .flatMap((sourceId) => {
      const source = sourceById.get(sourceId);
      return source ? [source] : [];
    })
    .sort((left, right) => sourceInfluence(right) - sourceInfluence(left))[0];
}

function strongestLocusForSource(
  reader: ContextInvestigationReader,
  sourceId: string,
): ContextInvestigationLocus | undefined {
  return reader.loci
    .filter((locus) => locus.sourceId === sourceId)
    .sort((left, right) => right.absoluteDeltaNats - left.absoluteDeltaNats)[0];
}

function directionForLoci(loci: readonly ContextInvestigationLocus[]): EvidenceDirection {
  const directions = new Set(loci.map((locus) => locus.direction));
  return directions.size === 1 ? [...directions][0] : "mixed";
}

function elementSourceIds(element: HTMLElement): string[] {
  try {
    const value: unknown = JSON.parse(element.dataset.sourceIds ?? "[]");
    return Array.isArray(value) && value.every((item) => typeof item === "string") ? value : [];
  } catch {
    return [];
  }
}

function sourceElements(container: HTMLElement, sourceId: string): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>("[data-source-ids]"))
    .filter((element) => elementSourceIds(element).includes(sourceId));
}

function indexedSourceFragments(container: HTMLElement): Map<string, DOMRect[]> {
  const result = new Map<string, DOMRect[]>();
  for (const element of container.querySelectorAll<HTMLElement>("[data-source-ids]")) {
    const fragments = Array.from(element.getClientRects());
    for (const sourceId of elementSourceIds(element)) {
      const current = result.get(sourceId) ?? [];
      current.push(...fragments);
      result.set(sourceId, current);
    }
  }
  return result;
}

function answerElements(container: HTMLElement, range: CodePointRange): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>("[data-answer-start][data-answer-end]"))
    .filter((element) => {
      const start = Number(element.dataset.answerStart);
      const end = Number(element.dataset.answerEnd);
      return Number.isSafeInteger(start) && Number.isSafeInteger(end)
        && rangesIntersect(range, { start, end });
    });
}

function elementFragments(elements: readonly HTMLElement[]): DOMRect[] {
  return elements.flatMap((element) => Array.from(element.getClientRects()));
}

function nearestVisibleFragment(
  fragments: readonly DOMRect[],
  viewport: Pick<DOMRect, "top" | "bottom" | "height">,
): DOMRect | undefined {
  const center = viewport.top + viewport.height / 2;
  return fragments
    .filter((rect) => rect.bottom > viewport.top && rect.top < viewport.bottom && rect.width > 0 && rect.height > 0)
    .sort((left, right) =>
      Math.abs(left.top + left.height / 2 - center) - Math.abs(right.top + right.height / 2 - center),
    )[0];
}

function normalizedY(range: CodePointRange, length: number, height: number): number {
  if (length <= 0 || height <= 0) return Math.max(24, height / 2);
  const ratio = ((range.start + range.end) / 2) / length;
  return Math.max(24, Math.min(height - 24, ratio * height));
}

function clampY(value: number, height: number): number {
  return Math.max(24, Math.min(height - 24, value));
}

function useEvidenceGeometry(
  contextRef: RefObject<HTMLElement | null>,
  answerRef: RefObject<HTMLElement | null>,
  mapRef: RefObject<HTMLDivElement | null>,
  reader: ContextInvestigationReader,
  activeRange: ActiveRange | undefined,
  activeSourceIds: readonly string[],
  directionBySource: ReadonlyMap<string, EvidenceDirection>,
): EvidenceGeometry {
  const [geometry, setGeometry] = useState<EvidenceGeometry>(EMPTY_GEOMETRY);
  useLayoutEffect(() => {
    const context = contextRef.current;
    const answer = answerRef.current;
    const map = mapRef.current;
    if (!context || !answer || !map) return undefined;
    let frame: number | undefined;
    let cancelled = false;

    const contextEnd = reader.context.blocks.length
      ? reader.context.blocks[reader.context.blocks.length - 1].displayUnicodeRange[1]
      : 0;
    const answerLength = reader.answer.status === "available" ? codePointLength(reader.answer.text) : 0;

    const update = () => {
      frame = undefined;
      if (cancelled) return;
      const mapRect = map.getBoundingClientRect();
      const contextRect = context.getBoundingClientRect();
      const answerRect = answer.getBoundingClientRect();
      const contextHeadingBottom = context.querySelector<HTMLElement>(".inspect-pane-heading")?.getBoundingClientRect().bottom
        ?? contextRect.top;
      const answerHeadingBottom = answer.querySelector<HTMLElement>(".inspect-pane-heading")?.getBoundingClientRect().bottom
        ?? answerRect.top;
      const contextViewport = {
        top: Math.max(contextRect.top, contextHeadingBottom),
        bottom: contextRect.bottom,
        height: Math.max(0, contextRect.bottom - Math.max(contextRect.top, contextHeadingBottom)),
      };
      const answerViewport = {
        top: Math.max(answerRect.top, answerHeadingBottom),
        bottom: answerRect.bottom,
        height: Math.max(0, answerRect.bottom - Math.max(answerRect.top, answerHeadingBottom)),
      };
      const width = Math.max(mapRect.width, 72);
      const height = Math.max(mapRect.height, 160);
      const contextHidden = contextRect.width < 1 || contextRect.height < 1;
      const answerHidden = answerRect.width < 1 || answerRect.height < 1;
      const fragmentsBySource = contextHidden ? new Map<string, DOMRect[]>() : indexedSourceFragments(context);

      const sourcePoint = (source: ContextInvestigationSource) => {
        const fallback = normalizedY(tupleRange(source.displayUnicodeRange), contextEnd, height);
        if (contextHidden) return { mapY: fallback, registeredOnly: true as const };
        const fragments = fragmentsBySource.get(source.sourceId) ?? [];
        const visible = nearestVisibleFragment(fragments, contextViewport);
        if (visible) {
          const center = visible.top + visible.height / 2;
          return {
            mapY: clampY(center - mapRect.top, height),
            localY: clampY(center - contextRect.top, contextRect.height),
            registeredOnly: false as const,
          };
        }
        if (fragments.length) {
          const above = fragments.every((rect) => rect.bottom <= contextViewport.top);
          const below = fragments.every((rect) => rect.top >= contextViewport.bottom);
          if (above || below) {
            return {
              mapY: above ? 24 : height - 24,
              offscreen: above ? "above" as const : "below" as const,
              registeredOnly: false as const,
            };
          }
        }
        return { mapY: fallback, registeredOnly: true as const };
      };

      let answerMapY = height / 2;
      let answerLocalY: number | undefined;
      let answerOffscreen: "above" | "below" | undefined;
      if (activeRange) {
        const fragments = elementFragments(answerElements(answer, activeRange));
        const visible = answerHidden ? undefined : nearestVisibleFragment(fragments, answerViewport);
        if (visible) {
          const center = visible.top + visible.height / 2;
          answerMapY = clampY(center - mapRect.top, height);
          answerLocalY = clampY(center - answerRect.top, answerRect.height);
        } else if (!answerHidden && fragments.length) {
          const above = fragments.every((rect) => rect.bottom <= answerViewport.top);
          const below = fragments.every((rect) => rect.top >= answerViewport.bottom);
          if (above || below) {
            answerOffscreen = above ? "above" : "below";
            answerMapY = above ? 24 : height - 24;
          } else {
            answerMapY = normalizedY(activeRange, answerLength, height);
          }
        } else {
          answerMapY = normalizedY(activeRange, answerLength, height);
        }
      }

      const sourceById = new Map(reader.sources.map((source) => [source.sourceId, source]));
      const relationships = activeSourceIds.flatMap((sourceId) => {
        const source = sourceById.get(sourceId);
        if (!source) return [];
        return [{
          sourceId,
          direction: directionBySource.get(sourceId) ?? "mixed",
          ...sourcePoint(source),
        }];
      });
      const coverage = reader.sources.map((source) => ({
        sourceId: source.sourceId,
        status: source.measurement.status,
        mapY: normalizedY(tupleRange(source.displayUnicodeRange), contextEnd, height),
      }));
      setGeometry({ width, height, answerMapY, answerLocalY, answerOffscreen, relationships, coverage });
    };

    const schedule = () => {
      if (frame !== undefined) return;
      frame = window.requestAnimationFrame(update);
    };
    const observer = typeof ResizeObserver === "undefined" ? undefined : new ResizeObserver(schedule);
    observer?.observe(context);
    observer?.observe(answer);
    observer?.observe(map);
    context.addEventListener("scroll", schedule, { passive: true });
    answer.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    void document.fonts?.ready.then(schedule);
    update();
    return () => {
      cancelled = true;
      if (frame !== undefined) window.cancelAnimationFrame(frame);
      observer?.disconnect();
      context.removeEventListener("scroll", schedule);
      answer.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [activeRange, activeSourceIds, answerRef, contextRef, directionBySource, mapRef, reader]);
  return geometry;
}

function ReaderState({ resource }: { resource: Resource<ContextInvestigationReader> }) {
  if (resource.phase === "loading") return <p className="inspect-state" role="status">Loading Inspect.</p>;
  if (resource.phase === "malformed") {
    return <p className="inspect-state is-failed" role="alert">Malformed evidence contract. <span>{resource.error}</span></p>;
  }
  if (resource.phase === "unavailable") {
    return <p className="inspect-state is-unavailable" role="alert">Inspect unavailable. <span>{resource.error}</span></p>;
  }
  return null;
}

function SequenceOverview({
  reader,
  activeRange,
  directions,
}: {
  reader: ContextInvestigationReader;
  activeRange?: ActiveRange;
  directions: ReadonlySet<EvidenceDirection>;
}) {
  const contextLength = reader.context.blocks.length
    ? reader.context.blocks[reader.context.blocks.length - 1].displayUnicodeRange[1]
    : 0;
  const answerLength = reader.answer.status === "available" ? codePointLength(reader.answer.text) : 0;
  const uniqueAnswerRanges = new Map<string, CodePointRange>();
  for (const locus of reader.loci) {
    const range = tupleRange(locus.answerUnicodeRange);
    uniqueAnswerRanges.set(rangeKey(range), range);
  }
  const styleFor = (range: CodePointRange, length: number) => ({
    "--range-start": `${length ? (range.start / length) * 100 : 0}%`,
    "--range-size": `${length ? ((range.end - range.start) / length) * 100 : 0}%`,
  }) as CSSProperties;
  const counts = reader.sources.reduce<Record<SourceMeasurementStatus, number>>((result, source) => {
    result[source.measurement.status] += 1;
    return result;
  }, { measured: 0, not_measured: 0, unavailable: 0, failed: 0 });
  const directionClasses = [
    directions.has("support") ? "has-support" : "",
    directions.has("suppression") ? "has-suppression" : "",
    directions.has("mixed") ? "has-mixed" : "",
  ].filter(Boolean).join(" ");
  return (
    <div
      className={`inspect-overview${activeRange ? " is-active" : ""}${directionClasses ? ` ${directionClasses}` : ""}`}
      role="img"
      aria-label={`Context coverage: ${plural(counts.measured, "measured source")}, ${plural(counts.not_measured, "not measured source")}, ${plural(counts.unavailable, "unavailable source")}, ${plural(counts.failed, "failed source")}. ${plural(uniqueAnswerRanges.size, "measured answer range")}.`}
    >
      <div className="overview-track overview-context">
        <span className="overview-label">Context</span>
        <div className="overview-rail" aria-hidden="true">
          {reader.sources.map((source) => (
            <i
              key={source.sourceId}
              className={`overview-range state-${source.measurement.status.replace("_", "-")}`}
              style={styleFor(tupleRange(source.displayUnicodeRange), contextLength)}
            />
          ))}
        </div>
      </div>
      <div className="overview-registration" aria-hidden="true"><i /><i /><i /></div>
      <div className="overview-track overview-answer">
        <span className="overview-label">Answer</span>
        <div className="overview-rail" aria-hidden="true">
          {[...uniqueAnswerRanges.entries()].map(([key, range]) => (
            <i
              key={key}
              className={`overview-range state-measured${activeRange && rangesIntersect(activeRange, range) ? " is-selected" : ""}`}
              style={styleFor(range, answerLength)}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function contextSegmentClass(
  segment: AnnotatedTextSegment,
  sourceById: ReadonlyMap<string, ContextInvestigationSource>,
  activeSourceIds: ReadonlySet<string>,
  focusedSourceId?: string,
): string {
  const sources = segment.annotationIds.flatMap((id) => {
    const source = sourceById.get(id);
    return source ? [source] : [];
  });
  const classes = ["context-segment"];
  if (sources.some((source) => source.measurement.status === "measured")) classes.push("has-measurement");
  if (sources.some((source) => source.measurement.status === "not_measured")) classes.push("has-not-measured");
  if (sources.some((source) => source.measurement.status === "unavailable")) classes.push("has-unavailable");
  if (sources.some((source) => source.measurement.status === "failed")) classes.push("has-failed");
  if (segment.annotationIds.some((id) => activeSourceIds.has(id))) classes.push("is-related");
  if (focusedSourceId && segment.annotationIds.some((id) => activeSourceIds.has(id)) && !segment.annotationIds.includes(focusedSourceId)) {
    classes.push("is-quieted");
  }
  if (focusedSourceId && segment.annotationIds.includes(focusedSourceId)) classes.push("is-focused");
  return classes.join(" ");
}

function ContextPane({
  reader,
  paneRef,
  activeSourceIds,
  focusedSourceId,
  directionBySource,
  canFocus,
  expanded,
  maxInfluence,
  onFocusSource,
  onPreviewSource,
  onToggleExpanded,
}: {
  reader: ContextInvestigationReader;
  paneRef: RefObject<HTMLElement | null>;
  activeSourceIds: ReadonlySet<string>;
  focusedSourceId?: string;
  directionBySource: ReadonlyMap<string, EvidenceDirection>;
  canFocus: boolean;
  expanded: boolean;
  maxInfluence: number;
  onFocusSource: (sourceId: string) => void;
  onPreviewSource: (sourceId?: string) => void;
  onToggleExpanded: () => void;
}) {
  const sourceById = useMemo(() => new Map(reader.sources.map((source) => [source.sourceId, source])), [reader.sources]);
  const sourcesByMessage = useMemo(() => {
    const result = new Map<number, ContextInvestigationSource[]>();
    for (const source of reader.sources) {
      const current = result.get(source.messageIndex) ?? [];
      current.push(source);
      result.set(source.messageIndex, current);
    }
    return result;
  }, [reader.sources]);
  return (
    <section className="inspect-pane context-pane" ref={paneRef} aria-labelledby="inspect-context-heading">
      <header className="inspect-pane-heading">
        <h2 id="inspect-context-heading">Context</h2>
        <button
          className="pane-expand"
          type="button"
          aria-expanded={expanded}
          onClick={onToggleExpanded}
        >{expanded ? "Collapse" : "Expand heatmap"}</button>
      </header>
      <div className="inspect-prose context-prose">
        {reader.context.blocks.map((block) => {
          const blockStart = block.displayUnicodeRange[0];
          const blockSources = sourcesByMessage.get(block.messageIndex) ?? [];
          const wholeSources = blockSources.filter((source) => source.granularity === "whole_segment");
          const wholeIds = wholeSources.map((source) => source.sourceId);
          const relatedWhole = wholeIds.filter((id) => activeSourceIds.has(id));
          const strongestWhole = strongestSource(wholeIds, sourceById);
          const focusableWhole = expanded
            ? strongestWhole?.sourceId
            : canFocus && relatedWhole.length === 1 ? relatedWhole[0] : undefined;
          const wholeDirections = new Set(relatedWhole.map((id) => directionBySource.get(id)).filter(Boolean));
          const wholeDirection = wholeDirections.size === 1
            ? [...wholeDirections][0]
            : wholeDirections.size > 1 ? "mixed" : undefined;
          const blockClasses = ["context-block"];
          if (wholeSources.some((source) => source.measurement.status === "measured")) blockClasses.push("has-whole-measurement");
          if (wholeSources.some((source) => source.measurement.status === "not_measured")) blockClasses.push("has-whole-not-measured");
          if (wholeSources.some((source) => source.measurement.status === "unavailable")) blockClasses.push("has-whole-unavailable");
          if (wholeSources.some((source) => source.measurement.status === "failed")) blockClasses.push("has-whole-failed");
          if (relatedWhole.length) blockClasses.push("is-related");
          if (focusedSourceId && relatedWhole.length && !relatedWhole.includes(focusedSourceId)) blockClasses.push("is-quieted");
          if (focusedSourceId && wholeIds.includes(focusedSourceId)) blockClasses.push("is-focused");
          const annotations = blockSources
            .filter((source) => source.granularity === "exact_span")
            .map((source) => ({
              id: source.sourceId,
              start: source.displayUnicodeRange[0] - blockStart,
              end: source.displayUnicodeRange[1] - blockStart,
            }));
          const segments = segmentAnnotatedText(block.text, annotations);
          return (
            <p
              key={`${block.messageIndex}:${block.segmentId}`}
              className={blockClasses.join(" ")}
              data-message-index={block.messageIndex}
              data-source-ids={wholeIds.length ? JSON.stringify(wholeIds) : undefined}
              data-direction={expanded && strongestWhole && strongestWhole.summary.totalContinuationDeltaNats !== null
                ? heatDirection(strongestWhole.summary.totalContinuationDeltaNats)
                : wholeDirection}
              data-heat={expanded && strongestWhole ? "true" : undefined}
              style={expanded && strongestWhole
                ? heatStyle(sourceInfluence(strongestWhole) / Math.max(maxInfluence, 1e-9))
                : undefined}
              onMouseEnter={expanded && focusableWhole ? () => onPreviewSource(focusableWhole) : undefined}
              onMouseLeave={expanded && focusableWhole ? () => onPreviewSource(undefined) : undefined}
            >
              {focusableWhole && (
                <button
                  className="context-block-focus"
                  type="button"
                  aria-label={`Focus ${wholeDirection === "mixed" ? "mixed measured" : wholeDirection ?? "measured"} relationship from “${excerpt(block.text)}”`}
                  onClick={() => onFocusSource(focusableWhole)}
                ><span aria-hidden="true" /></button>
              )}
              {segments.map((segment) => {
                const related = segment.annotationIds.filter((id) => activeSourceIds.has(id));
                const strongest = strongestSource(segment.annotationIds, sourceById);
                const focusableSource = expanded
                  ? strongest?.sourceId
                  : canFocus && related.length === 1 ? related[0] : undefined;
                const directions = new Set(related.map((id) => directionBySource.get(id)).filter(Boolean));
                const direction = directions.size === 1 ? [...directions][0] : directions.size > 1 ? "mixed" : undefined;
                const heatValue = strongest?.summary.totalContinuationDeltaNats;
                return (
                  <span
                    key={`${block.messageIndex}:${segment.start}:${segment.end}`}
                    className={contextSegmentClass(segment, sourceById, activeSourceIds, focusedSourceId)}
                    data-source-ids={segment.annotationIds.length ? JSON.stringify(segment.annotationIds) : undefined}
                    data-direction={expanded && heatValue !== null && heatValue !== undefined
                      ? heatDirection(heatValue)
                      : direction}
                    data-heat={expanded && strongest ? "true" : undefined}
                    style={expanded && strongest
                      ? heatStyle(sourceInfluence(strongest) / Math.max(maxInfluence, 1e-9))
                      : undefined}
                    role={focusableSource ? "button" : undefined}
                    tabIndex={focusableSource ? 0 : undefined}
                    aria-label={focusableSource ? `Focus “${excerpt(segment.text)}”. ${sourceStatusLabel(sourceById.get(focusableSource)?.measurement.status ?? "not_measured")}.` : undefined}
                    onMouseEnter={expanded && focusableSource ? () => onPreviewSource(focusableSource) : undefined}
                    onMouseLeave={expanded && focusableSource ? () => onPreviewSource(undefined) : undefined}
                    onClick={focusableSource ? () => onFocusSource(focusableSource) : undefined}
                    onKeyDown={focusableSource ? (event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onFocusSource(focusableSource);
                      }
                    } : undefined}
                  >{segment.text}</span>
                );
              })}
            </p>
          );
        })}
      </div>
    </section>
  );
}

function AnswerPane({
  reader,
  paneRef,
  segments,
  activeRange,
  lockedRange,
  lockedRelationshipCount,
  expanded,
  maxInfluence,
  onPreview,
  onLock,
  onClear,
  onToggleExpanded,
}: {
  reader: ContextInvestigationReader;
  paneRef: RefObject<HTMLElement | null>;
  segments: readonly AnnotatedTextSegment[];
  activeRange?: ActiveRange;
  lockedRange?: ActiveRange;
  lockedRelationshipCount?: number;
  expanded: boolean;
  maxInfluence: number;
  onPreview: (range?: ActiveRange) => void;
  onLock: (range: ActiveRange) => void;
  onClear: () => void;
  onToggleExpanded: () => void;
}) {
  const interactive = useMemo(() => segments.filter((segment) => segment.annotationIds.length > 0), [segments]);
  const interactiveIndex = useMemo(
    () => new Map(interactive.map((segment, index) => [rangeKey(segment), index])),
    [interactive],
  );
  const nodes = useRef(new Map<string, HTMLElement>());
  const [rovingKey, setRovingKey] = useState<string>();
  const lockedRovingKey = lockedRange
    ? interactive.find((segment) => rangesIntersect(lockedRange, segment))
    : undefined;
  const effectiveRovingKey = lockedRovingKey
    ? rangeKey(lockedRovingKey)
    : interactive.some((segment) => rangeKey(segment) === rovingKey)
      ? rovingKey
      : interactive[0] ? rangeKey(interactive[0]) : undefined;
  const locusSourceById = useMemo(
    () => new Map(reader.loci.map((locus) => [locus.locusId, locus.sourceId])),
    [reader.loci],
  );
  const locusById = useMemo(
    () => new Map(reader.loci.map((locus) => [locus.locusId, locus])),
    [reader.loci],
  );
  const focusAt = (index: number) => {
    const next = interactive[Math.max(0, Math.min(interactive.length - 1, index))];
    if (next) {
      const key = rangeKey(next);
      setRovingKey(key);
      nodes.current.get(key)?.focus();
    }
  };
  const handleKey = (event: ReactKeyboardEvent<HTMLElement>, segment: AnnotatedTextSegment, index: number) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onLock({ start: segment.start, end: segment.end, key: rangeKey(segment) });
    } else if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      event.preventDefault();
      focusAt(index + 1);
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      event.preventDefault();
      focusAt(index - 1);
    } else if (event.key === "Home") {
      event.preventDefault();
      focusAt(0);
    } else if (event.key === "End") {
      event.preventDefault();
      focusAt(interactive.length - 1);
    } else if (event.key === "Escape") {
      event.preventDefault();
      onClear();
    }
  };
  return (
    <section className="inspect-pane answer-pane" ref={paneRef} aria-labelledby="inspect-answer-heading">
      <header className="inspect-pane-heading">
        <h2 id="inspect-answer-heading">Answer</h2>
        <div className="pane-heading-actions">
          {reader.answer.status === "available" && <span>{plural(reader.answer.tokens.length, "token")}</span>}
          <button
            className="pane-expand"
            type="button"
            aria-expanded={expanded}
            onClick={onToggleExpanded}
          >{expanded ? "Collapse" : "Expand heatmap"}</button>
        </div>
      </header>
      {reader.answer.status === "available" ? (
        <div className="inspect-prose answer-prose">
          <p>
            {segments.map((segment) => {
              const measured = segment.annotationIds.length > 0;
              const range = { start: segment.start, end: segment.end, key: rangeKey(segment) };
              const active = activeRange && rangesIntersect(activeRange, range);
              const locked = lockedRange && rangesIntersect(lockedRange, range);
              const index = measured ? interactiveIndex.get(range.key) ?? -1 : -1;
              const relationshipCount = new Set(segment.annotationIds.flatMap((id) => {
                const sourceId = locusSourceById.get(id);
                return sourceId ? [sourceId] : [];
              })).size;
              const announcedCount = locked && lockedRelationshipCount !== undefined
                ? lockedRelationshipCount
                : relationshipCount;
              const relationshipNoun = locked && lockedRelationshipCount !== undefined
                ? "above-floor relationship"
                : "measured relationship";
              const segmentLoci = segment.annotationIds.flatMap((id) => {
                const locus = locusById.get(id);
                return locus ? [locus] : [];
              });
              const segmentInfluence = segmentLoci.reduce((sum, locus) => sum + locus.absoluteDeltaNats, 0);
              return (
                <span
                  key={range.key}
                  ref={measured ? (node) => {
                    if (node) nodes.current.set(range.key, node);
                    else nodes.current.delete(range.key);
                  } : undefined}
                  className={`answer-segment${measured ? " has-measurement" : ""}${active ? " is-active" : ""}${locked ? " is-locked" : ""}`}
                  data-answer-start={segment.start}
                  data-answer-end={segment.end}
                  data-direction={expanded && segmentLoci.length ? directionForLoci(segmentLoci) : undefined}
                  data-heat={expanded && segmentLoci.length ? "true" : undefined}
                  style={expanded && segmentLoci.length
                    ? heatStyle(segmentInfluence / Math.max(maxInfluence, 1e-9))
                    : undefined}
                  role={measured ? "button" : undefined}
                  tabIndex={measured ? (range.key === effectiveRovingKey ? 0 : -1) : undefined}
                  aria-pressed={measured ? Boolean(locked) : undefined}
                  aria-label={measured ? `${segment.text}; ${plural(announcedCount, relationshipNoun)}` : undefined}
                  onMouseEnter={measured && !lockedRange ? () => onPreview(range) : undefined}
                  onMouseLeave={measured && !lockedRange ? () => onPreview(undefined) : undefined}
                  onFocus={measured ? () => {
                    setRovingKey(range.key);
                    if (!lockedRange) onPreview(range);
                  } : undefined}
                  onBlur={measured && !lockedRange ? () => onPreview(undefined) : undefined}
                  onClick={measured ? () => onLock(range) : undefined}
                  onKeyDown={measured ? (event) => handleKey(event, segment, index) : undefined}
                >{segment.text}</span>
              );
            })}
          </p>
        </div>
      ) : <p className="pane-unavailable">Answer unavailable.</p>}
    </section>
  );
}

function relationshipPath(sourceX: number, sourceY: number, answerX: number, answerY: number): string {
  const distance = Math.abs(answerX - sourceX);
  return `M ${sourceX} ${sourceY} C ${sourceX + distance * .44} ${sourceY}, ${answerX - distance * .44} ${answerY}, ${answerX} ${answerY}`;
}

function EvidenceMap({
  mapRef,
  geometry,
  activeRange,
  locked,
  focusedSourceId,
  visibleSourceIds,
  sourceLabels,
  onFocusSource,
  onOffscreen,
  onAnswerOffscreen,
}: {
  mapRef: RefObject<HTMLDivElement | null>;
  geometry: EvidenceGeometry;
  activeRange?: ActiveRange;
  locked: boolean;
  focusedSourceId?: string;
  visibleSourceIds: ReadonlySet<string>;
  sourceLabels: ReadonlyMap<string, string>;
  onFocusSource: (sourceId: string) => void;
  onOffscreen: (direction: "above" | "below") => void;
  onAnswerOffscreen: () => void;
}) {
  const supportGradient = `support-${useId().replaceAll(":", "")}`;
  const suppressionGradient = `suppression-${useId().replaceAll(":", "")}`;
  const mixedGradient = `mixed-${useId().replaceAll(":", "")}`;
  const axisX = geometry.width * .34;
  const gateX = geometry.width - 14;
  const registeredRelationships = geometry.relationships.filter((relationship) => relationship.offscreen === undefined);
  const mappedRelationships = geometry.relationships.filter((relationship) => visibleSourceIds.has(relationship.sourceId));
  const relationshipIndex = new Map(mappedRelationships.map((relationship, index) => [relationship.sourceId, index]));
  const visibleThreads = mappedRelationships.filter((relationship) =>
    relationship.offscreen === undefined
    && relationship.localY !== undefined
    && geometry.answerLocalY !== undefined
    && geometry.answerOffscreen === undefined,
  );
  const aboveCount = geometry.relationships.filter((relationship) => relationship.offscreen === "above").length;
  const belowCount = geometry.relationships.filter((relationship) => relationship.offscreen === "below").length;
  const hiddenCount = focusedSourceId ? 0 : Math.max(0, registeredRelationships.length - mappedRelationships.length);
  const directionClasses = [
    geometry.relationships.some((relationship) => relationship.direction === "support") ? "has-support" : "",
    geometry.relationships.some((relationship) => relationship.direction === "suppression") ? "has-suppression" : "",
    geometry.relationships.some((relationship) => relationship.direction === "mixed") ? "has-mixed" : "",
  ].filter(Boolean).join(" ");
  return (
    <div className={`evidence-map${activeRange ? " is-active" : ""}${directionClasses ? ` ${directionClasses}` : ""}`} ref={mapRef} role="group" aria-label="Influence relationships">
      <svg className="evidence-map-svg" viewBox={`0 0 ${geometry.width} ${geometry.height}`} preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <linearGradient id={supportGradient} x1="0" x2="1">
            <stop offset="0" stopColor="var(--evidence-support)" />
            <stop offset=".58" stopColor="var(--status-success)" />
            <stop offset="1" stopColor="var(--action-bright)" />
          </linearGradient>
          <linearGradient id={suppressionGradient} x1="0" x2="1">
            <stop offset="0" stopColor="var(--evidence-suppression)" />
            <stop offset=".58" stopColor="var(--evidence-spectral-pink)" />
            <stop offset="1" stopColor="var(--action-bright)" />
          </linearGradient>
          <linearGradient id={mixedGradient} x1="0" x2="1">
            <stop offset="0" stopColor="var(--evidence-measured)" />
            <stop offset=".58" stopColor="var(--evidence-spectral-pink)" />
            <stop offset="1" stopColor="var(--action-bright)" />
          </linearGradient>
        </defs>
        <line className="map-axis" x1={axisX} x2={axisX} y1="18" y2={geometry.height - 18} />
        <line className="map-gate-rail" x1={gateX} x2={gateX} y1="18" y2={geometry.height - 18} />
        {geometry.coverage.map((mark) => (
          <line
            key={mark.sourceId}
            className={`coverage-mark state-${mark.status.replace("_", "-")}`}
            x1={axisX - 5}
            x2={axisX + 5}
            y1={mark.mapY}
            y2={mark.mapY}
          />
        ))}
        {visibleThreads.map((relationship) => {
          const sourceX = axisX + ((relationshipIndex.get(relationship.sourceId) ?? 0) % 3) * 4;
          const path = relationshipPath(sourceX, relationship.mapY, gateX, geometry.answerMapY);
          const gradient = relationship.direction === "suppression"
            ? suppressionGradient
            : relationship.direction === "mixed" ? mixedGradient : supportGradient;
          return (
            <g key={relationship.sourceId} className={`relationship-path direction-${relationship.direction}`}>
              <path className="relationship-glow" d={path} stroke={`url(#${gradient})`} />
              <path className="relationship-signal" d={path} stroke={`url(#${gradient})`} />
            </g>
          );
        })}
        {activeRange && <line className="answer-gate" x1={gateX - 5} x2={gateX + 1} y1={geometry.answerMapY} y2={geometry.answerMapY} />}
      </svg>
      {mappedRelationships.filter((relationship) => relationship.offscreen === undefined).map((relationship) => {
        const left = axisX + ((relationshipIndex.get(relationship.sourceId) ?? 0) % 3) * 4;
        const sourceLabel = sourceLabels.get(relationship.sourceId) ?? shortId(relationship.sourceId);
        const className = `map-node direction-${relationship.direction}${focusedSourceId === relationship.sourceId ? " is-focused" : ""}`;
        const style = { "--map-node-x": `${left}px`, "--map-node-y": `${relationship.mapY}px` } as CSSProperties;
        return locked ? (
          <button
            key={relationship.sourceId}
            className={className}
            style={style}
            type="button"
            aria-label={`Focus ${relationship.direction === "mixed" ? "mixed measured" : relationship.direction} relationship from “${sourceLabel}”`}
            aria-pressed={focusedSourceId === relationship.sourceId}
            onClick={() => onFocusSource(relationship.sourceId)}
          ><span aria-hidden="true" /></button>
        ) : (
          <span key={relationship.sourceId} className={className} style={style} aria-hidden="true"><span /></span>
        );
      })}
      {aboveCount > 0 && (
        locked ? (
          <button className="offscreen-mark is-above" type="button" onClick={() => onOffscreen("above")} aria-label={`${plural(aboveCount, "related context locus", "related context loci")} above`}>
            <span aria-hidden="true">↑</span>{aboveCount}
          </button>
        ) : <span className="offscreen-mark is-above" aria-hidden="true"><span>↑</span>{aboveCount}</span>
      )}
      {belowCount > 0 && (
        locked ? (
          <button className="offscreen-mark is-below" type="button" onClick={() => onOffscreen("below")} aria-label={`${plural(belowCount, "related context locus", "related context loci")} below`}>
            <span aria-hidden="true">↓</span>{belowCount}
          </button>
        ) : <span className="offscreen-mark is-below" aria-hidden="true"><span>↓</span>{belowCount}</span>
      )}
      {geometry.answerOffscreen && locked && (
        <button
          className={`answer-offscreen-mark is-${geometry.answerOffscreen}`}
          type="button"
          onClick={onAnswerOffscreen}
          aria-label={`Selected answer ${geometry.answerOffscreen}`}
        >
          <span aria-hidden="true">{geometry.answerOffscreen === "above" ? "↑" : "↓"}</span>
        </button>
      )}
      {hiddenCount > 0 && (
        <span className="relationship-overflow" aria-label={`${plural(hiddenCount, "additional relationship")} not drawn`}>+{hiddenCount}</span>
      )}
    </div>
  );
}

function QueryStateLine({ resource }: { resource: OptionalResource<ContextInvestigationQuery> }) {
  if (resource.phase === "loading") return <span role="status">Loading evidence.</span>;
  if (resource.phase === "malformed") return <span className="is-failed" role="alert">Malformed evidence.</span>;
  if (resource.phase === "unavailable") return <span className="is-unavailable" role="alert">Evidence unavailable.</span>;
  if (resource.phase === "ready" && resource.value.status !== "completed") {
    return (
      <span className={`query-status status-${resource.value.status.replace("_", "-")}`} role={resource.value.status === "stale" || resource.value.status === "unavailable" ? "alert" : "status"}>
        {statusLabel(resource.value.status)}{resource.value.reason ? ` · ${resource.value.reason}` : ""}
      </span>
    );
  }
  return null;
}

function EvidenceInspector({
  reader,
  range,
  query,
  sourceIds,
  focusedSourceId,
  onPrevious,
  onNext,
  onClear,
}: {
  reader: ContextInvestigationReader;
  range: ActiveRange;
  query: OptionalResource<ContextInvestigationQuery>;
  sourceIds: readonly string[];
  focusedSourceId?: string;
  onPrevious: () => void;
  onNext: () => void;
  onClear: () => void;
}) {
  const answerText = reader.answer.status === "available"
    ? sliceCodePoints(reader.answer.text, range.start, range.end)
    : "";
  const ready = query.phase === "ready" ? query.value : undefined;
  const focusedEffect = ready?.effects.find((effect) => effect.sourceId === focusedSourceId);
  const focusedIndex = focusedSourceId ? sourceIds.indexOf(focusedSourceId) : -1;
  const counts = ready?.effects.reduce((result, effect) => {
    result[effect.direction] = (result[effect.direction] ?? 0) + 1;
    return result;
  }, {} as Record<string, number>);
  const countSummary = counts ? [
    `${counts.support ?? 0} support`,
    `${counts.suppression ?? 0} suppression`,
    `${counts.below_floor ?? 0} below floor`,
    ...(counts.not_measured ? [`${counts.not_measured} not measured`] : []),
    ...(counts.unavailable ? [`${counts.unavailable} unavailable`] : []),
    ...(counts.failed ? [`${counts.failed} failed`] : []),
  ].join(" · ") : undefined;
  return (
    <footer className="evidence-inspector" aria-live="polite">
      <div className="inspector-selection">
        <span className="eyebrow">Answer</span>
        <p>{answerText}</p>
      </div>
      <div className="inspector-evidence">
        {focusedEffect ? (
          <>
            <span className="eyebrow">Relationship</span>
            <div className="effect-reading">
              {focusedEffect.deltaNats !== null && (
                <strong>{focusedEffect.deltaNats > 0 ? "+" : ""}{focusedEffect.deltaNats.toFixed(2)} <small>Δ nats</small></strong>
              )}
              <span className={`evidence-state direction-${focusedEffect.direction}`}>{directionLabel(focusedEffect.direction)}</span>
            </div>
            <div className="effect-basis">Delete source · recorded continuation · floor {ready?.measurementFloorNats.toFixed(2)}</div>
            <QueryStateLine resource={query} />
          </>
        ) : (
          <>
            <span className="eyebrow">Influence</span>
            <div className="effect-counts">
              <strong>{plural(sourceIds.length, "above-floor relationship")}</strong>
              {countSummary && <span>{countSummary}</span>}
            </div>
            <QueryStateLine resource={query} />
          </>
        )}
      </div>
      <div className="inspector-navigation">
        <button type="button" onClick={onPrevious} aria-label="Previous related context">←</button>
        <span>{focusedIndex >= 0 ? `${focusedIndex + 1} / ${sourceIds.length}` : `${sourceIds.length}`}</span>
        <button type="button" onClick={onNext} aria-label="Next related context">→</button>
        <button className="inspector-clear" type="button" onClick={onClear}>Clear</button>
      </div>
      {focusedEffect && (
        <details className="evidence-provenance">
          <summary>Evidence</summary>
          <dl>
            <div><dt>Source</dt><dd><code>{shortId(focusedEffect.sourceId)}</code></dd></div>
            <div><dt>Experiment</dt><dd><code>{shortId(focusedEffect.experimentId)}</code></dd></div>
            <div><dt>Observation</dt><dd><code>{shortId(focusedEffect.interventionObservationId)}</code></dd></div>
          </dl>
        </details>
      )}
    </footer>
  );
}

function HeatmapPeek({
  reader,
  pane,
  sourceId,
  range,
}: {
  reader: ContextInvestigationReader;
  pane: MobilePane;
  sourceId?: string;
  range?: ActiveRange;
}) {
  const sourceById = useMemo(
    () => new Map(reader.sources.map((source) => [source.sourceId, source])),
    [reader.sources],
  );
  const loci = useMemo(() => {
    if (pane === "context" && sourceId) {
      return reader.loci
        .filter((locus) => locus.sourceId === sourceId)
        .sort((left, right) => right.absoluteDeltaNats - left.absoluteDeltaNats);
    }
    if (pane === "answer" && range) {
      return reader.loci
        .filter((locus) => rangesIntersect(range, tupleRange(locus.answerUnicodeRange)))
        .sort((left, right) => right.absoluteDeltaNats - left.absoluteDeltaNats);
    }
    return [];
  }, [pane, range, reader.loci, sourceId]);
  const source = sourceId ? sourceById.get(sourceId) : undefined;
  if (!source && !range) {
    return (
      <aside className="heatmap-peek" aria-live="polite">
        <span className="eyebrow">Heatmap</span>
        <p>Hover highlighted text to preview its strongest links. Click to pin the evidence.</p>
      </aside>
    );
  }
  return (
    <aside className="heatmap-peek" aria-live="polite">
      <div>
        <span className="eyebrow">{pane === "context" ? "Context span" : "Answer span"}</span>
        <p>{source
          ? excerpt(source.label?.trim() || sourceRecordedText(reader, source), 180)
          : reader.answer.status === "available" && range
            ? excerpt(sliceCodePoints(reader.answer.text, range.start, range.end), 180)
            : "Selection unavailable"}</p>
      </div>
      <div>
        <span className="eyebrow">Linked text</span>
        {loci.length ? (
          <ul>
            {loci.slice(0, 3).map((locus) => {
              const linkedSource = sourceById.get(locus.sourceId);
              return (
                <li key={locus.locusId}>
                  <span className={`evidence-state direction-${locus.direction}`}>{directionLabel(locus.direction)}</span>
                  <strong>{locus.deltaNats > 0 ? "+" : ""}{locus.deltaNats.toFixed(2)} Δ</strong>
                  <span>{pane === "context"
                    ? excerpt(locus.answerText, 110)
                    : excerpt(linkedSource ? sourceRecordedText(reader, linkedSource) : locus.sourceId, 110)}</span>
                </li>
              );
            })}
          </ul>
        ) : <p>No above-floor link was recorded for this text.</p>}
      </div>
    </aside>
  );
}

function InspectReader({
  reader,
  route,
}: {
  reader: ContextInvestigationReader;
  route: InspectRoute;
}) {
  const [hoveredRange, setHoveredRange] = useState<ActiveRange | undefined>();
  const [hoveredSourceId, setHoveredSourceId] = useState<string | undefined>();
  const [expandedPane, setExpandedPane] = useState<MobilePane | undefined>();
  const [mobilePane, setMobilePane] = useState<MobilePane>("answer");
  const contextRef = useRef<HTMLElement>(null);
  const answerRef = useRef<HTMLElement>(null);
  const mapRef = useRef<HTMLDivElement>(null);
  const answer = reader.answer.status === "available" ? reader.answer : undefined;
  const lockedRange = useMemo<ActiveRange | undefined>(() => {
    if (!answer || route.answerStart === undefined || route.answerEnd === undefined) return undefined;
    const range = { start: route.answerStart, end: route.answerEnd };
    return isValidCodePointRange(answer.text, range, { allowEmpty: false })
      ? { ...range, key: rangeKey(range) }
      : undefined;
  }, [answer, route.answerEnd, route.answerStart]);
  const activeRange = lockedRange ?? hoveredRange;
  const maxSourceInfluence = useMemo(
    () => Math.max(0, ...reader.sources.map(sourceInfluence)),
    [reader.sources],
  );
  const maxAnswerInfluence = useMemo(() => {
    const totals = new Map<string, number>();
    for (const locus of reader.loci) {
      const key = `${locus.answerUnicodeRange[0]}:${locus.answerUnicodeRange[1]}`;
      totals.set(key, (totals.get(key) ?? 0) + locus.absoluteDeltaNats);
    }
    return Math.max(0, ...totals.values());
  }, [reader.loci]);

  const answerSegments = useMemo(() => {
    if (!answer) return [];
    const annotations = reader.loci.map((locus) => ({
      id: locus.locusId,
      start: locus.answerUnicodeRange[0],
      end: locus.answerUnicodeRange[1],
    }));
    if (lockedRange && !annotations.some((item) => item.start === lockedRange.start && item.end === lockedRange.end)) {
      annotations.push({ id: "locked-selection", start: lockedRange.start, end: lockedRange.end });
    }
    return segmentAnnotatedText(answer.text, annotations).map((segment) => ({
      ...segment,
      annotationIds: segment.annotationIds.filter((id) => id !== "locked-selection"),
    }));
  }, [answer, lockedRange, reader.loci]);

  const queryLoader = useMemo(() => lockedRange && reader.capabilities.contextEffects === "measurable"
    ? (signal: AbortSignal) => contextInvestigationApi.query(route.runId, {
      answerStart: lockedRange.start,
      answerEnd: lockedRange.end,
      floorNats: reader.presentation.measurementFloorNats,
    }, signal)
    : undefined, [lockedRange, reader.capabilities.contextEffects, reader.presentation.measurementFloorNats, route.runId]);
  const queryKey = lockedRange && reader.capabilities.contextEffects === "measurable"
    ? `${route.runId}:${lockedRange.start}:${lockedRange.end}:${reader.presentation.measurementFloorNats}`
    : undefined;
  const rawQuery = useOptionalResource(queryKey, queryLoader);
  const query = useMemo<OptionalResource<ContextInvestigationQuery>>(() => {
    if (lockedRange && reader.capabilities.contextEffects === "unavailable") {
      return { phase: "unavailable", error: reader.reason ?? "Context effects are unavailable." };
    }
    if (rawQuery.phase !== "ready") return rawQuery;
    const sourceIds = new Set(reader.sources.map((source) => source.sourceId));
    if (rawQuery.value.effects.some((effect) => !sourceIds.has(effect.sourceId))) {
      return { phase: "malformed", error: "Query references a source outside the reader projection." };
    }
    const readerExperimentId = reader.measurement.experimentId;
    if (readerExperimentId && rawQuery.value.effects.some((effect) => effect.experimentId !== readerExperimentId)) {
      return { phase: "malformed", error: "Query and reader experiment identities do not match." };
    }
    return rawQuery;
  }, [lockedRange, rawQuery, reader.capabilities.contextEffects, reader.measurement.experimentId, reader.reason, reader.sources]);
  const activeLoci = useMemo(() => activeRange
    ? reader.loci.filter((locus) => rangesIntersect(activeRange, tupleRange(locus.answerUnicodeRange)))
    : [], [activeRange, reader.loci]);
  const sourceById = useMemo(() => new Map(reader.sources.map((source) => [source.sourceId, source])), [reader.sources]);
  const sourceLabels = useMemo(() => {
    const blockByMessage = new Map(reader.context.blocks.map((block) => [block.messageIndex, block]));
    return new Map(reader.sources.map((source) => {
      const block = blockByMessage.get(source.messageIndex);
      const recordedText = block
        ? sliceCodePoints(block.text, source.unicodeRange[0], source.unicodeRange[1])
        : source.sourceId;
      return [source.sourceId, excerpt(source.label?.trim() || recordedText || source.sourceId)];
    }));
  }, [reader.context.blocks, reader.sources]);
  const queryAboveFloorEffects = useMemo(() => lockedRange && query.phase === "ready"
    ? query.value.effects.filter(isAboveFloorEffect)
    : undefined, [lockedRange, query]);
  const activeSourceIds = useMemo(() => [...new Set(
    (queryAboveFloorEffects ?? activeLoci).map((item) => item.sourceId),
  )].filter((sourceId) => sourceById.has(sourceId)).sort((left, right) => {
    const leftRange = sourceById.get(left)?.displayUnicodeRange ?? [0, 0];
    const rightRange = sourceById.get(right)?.displayUnicodeRange ?? [0, 0];
    return leftRange[0] - rightRange[0] || left.localeCompare(right);
  }), [activeLoci, queryAboveFloorEffects, sourceById]);
  const focusedSourceId = lockedRange && route.sourceId && activeSourceIds.includes(route.sourceId)
    ? route.sourceId
    : undefined;
  const displayedFocusedSourceId = hoveredSourceId ?? focusedSourceId;
  const directionBySource = useMemo(() => {
    if (queryAboveFloorEffects) {
      return new Map(queryAboveFloorEffects.map((effect) => [effect.sourceId, effect.direction]));
    }
    const directions = new Map<string, Set<ContextInvestigationLocus["direction"]>>();
    for (const locus of activeLoci) {
      const current = directions.get(locus.sourceId) ?? new Set();
      current.add(locus.direction);
      directions.set(locus.sourceId, current);
    }
    return new Map([...directions].map(([sourceId, values]) => [
      sourceId,
      values.size === 1 ? [...values][0] : "mixed" as const,
    ]));
  }, [activeLoci, queryAboveFloorEffects]);
  const geometry = useEvidenceGeometry(
    contextRef,
    answerRef,
    mapRef,
    reader,
    activeRange,
    activeSourceIds,
    directionBySource,
  );
  const visibleSourceIds = useMemo(() => {
    if (displayedFocusedSourceId) return new Set([displayedFocusedSourceId]);
    const visible = geometry.relationships
      .filter((relationship) => relationship.offscreen === undefined && relationship.localY !== undefined)
      .map((relationship) => relationship.sourceId);
    const candidates = visible.length ? visible : activeSourceIds;
    return new Set(candidates.slice(0, MAX_TRANSIENT_PATHS));
  }, [activeSourceIds, displayedFocusedSourceId, geometry.relationships]);

  const setCoordinate = useCallback((range?: ActiveRange, sourceId?: string) => {
    window.location.hash = routeHref(range ? {
      surface: "inspect",
      runId: route.runId,
      answerStart: range.start,
      answerEnd: range.end,
      ...(sourceId ? { sourceId } : {}),
    } : {
      surface: "inspect",
      runId: route.runId,
    });
  }, [route.runId]);

  useEffect(() => {
    if (!lockedRange) return undefined;
    const clearOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setCoordinate();
    };
    window.addEventListener("keydown", clearOnEscape);
    return () => window.removeEventListener("keydown", clearOnEscape);
  }, [lockedRange, setCoordinate]);

  const focusSource = useCallback((sourceId: string) => {
    if (!lockedRange) return;
    const next = focusedSourceId === sourceId ? undefined : sourceId;
    setCoordinate(lockedRange, next);
    if (next) setMobilePane("context");
  }, [focusedSourceId, lockedRange, setCoordinate]);

  const previewContextSource = useCallback((sourceId?: string) => {
    setHoveredSourceId(sourceId);
    if (lockedRange) return;
    const locus = sourceId ? strongestLocusForSource(reader, sourceId) : undefined;
    setHoveredRange(locus ? {
      start: locus.answerUnicodeRange[0],
      end: locus.answerUnicodeRange[1],
      key: `${locus.answerUnicodeRange[0]}:${locus.answerUnicodeRange[1]}`,
    } : undefined);
  }, [lockedRange, reader]);

  const lockContextSource = useCallback((sourceId: string) => {
    const locus = strongestLocusForSource(reader, sourceId);
    if (!locus) return;
    setHoveredSourceId(undefined);
    setHoveredRange(undefined);
    setCoordinate({
      start: locus.answerUnicodeRange[0],
      end: locus.answerUnicodeRange[1],
      key: `${locus.answerUnicodeRange[0]}:${locus.answerUnicodeRange[1]}`,
    }, sourceId);
  }, [reader, setCoordinate]);

  useEffect(() => {
    if (!focusedSourceId || !contextRef.current) return;
    setMobilePane("context");
    const viewport = contextRef.current;
    const target = sourceElements(viewport, focusedSourceId)[0];
    if (!target) return;
    const targetRect = target.getBoundingClientRect();
    const viewportRect = viewport.getBoundingClientRect();
    const top = viewport.scrollTop + targetRect.top - viewportRect.top - viewport.clientHeight / 2 + targetRect.height / 2;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    viewport.scrollTo?.({ top: Math.max(0, top), behavior: reduced ? "auto" : "smooth" });
  }, [focusedSourceId]);

  const navigateSource = useCallback((step: -1 | 1) => {
    if (!lockedRange || !activeSourceIds.length) return;
    const current = focusedSourceId ? activeSourceIds.indexOf(focusedSourceId) : (step > 0 ? -1 : 0);
    const nextIndex = (current + step + activeSourceIds.length) % activeSourceIds.length;
    setCoordinate(lockedRange, activeSourceIds[nextIndex]);
    setMobilePane("context");
  }, [activeSourceIds, focusedSourceId, lockedRange, setCoordinate]);

  const navigateOffscreen = useCallback((direction: "above" | "below") => {
    const candidates = geometry.relationships.filter((relationship) => relationship.offscreen === direction);
    if (!candidates.length || !activeRange) return;
    const ordered = candidates.sort((left, right) => {
      const leftStart = sourceById.get(left.sourceId)?.displayUnicodeRange[0] ?? 0;
      const rightStart = sourceById.get(right.sourceId)?.displayUnicodeRange[0] ?? 0;
      return direction === "above" ? rightStart - leftStart : leftStart - rightStart;
    });
    setCoordinate(activeRange, ordered[0].sourceId);
    setMobilePane("context");
  }, [activeRange, geometry.relationships, setCoordinate, sourceById]);

  const navigateAnswer = useCallback(() => {
    if (!activeRange) return;
    setMobilePane("answer");
    window.requestAnimationFrame(() => {
      const viewport = answerRef.current;
      const target = viewport ? answerElements(viewport, activeRange)[0] : undefined;
      if (!viewport || !target) return;
      const targetRect = target.getBoundingClientRect();
      const viewportRect = viewport.getBoundingClientRect();
      const top = viewport.scrollTop + targetRect.top - viewportRect.top - viewport.clientHeight / 2 + targetRect.height / 2;
      const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      viewport.scrollTo?.({ top: Math.max(0, top), behavior: reduced ? "auto" : "smooth" });
    });
  }, [activeRange]);

  const relatedSet = useMemo(() => new Set(activeSourceIds), [activeSourceIds]);
  const activeDirections = useMemo(() => new Set(directionBySource.values()), [directionBySource]);
  const displayedRelationships = geometry.relationships.filter((relationship) => visibleSourceIds.has(relationship.sourceId));
  return (
    <>
      <div className="inspect-pane-switch" role="group" aria-label="Readable pane">
        <button type="button" aria-pressed={mobilePane === "context"} onClick={() => setMobilePane("context")}>Context</button>
        <button type="button" aria-pressed={mobilePane === "answer"} onClick={() => setMobilePane("answer")}>Answer</button>
      </div>
      <SequenceOverview reader={reader} activeRange={activeRange} directions={activeDirections} />
      <div className={`inspect-reader mobile-${mobilePane}${lockedRange ? " has-lock" : ""}${expandedPane ? ` is-expanded is-expanded-${expandedPane}` : ""}`}>
        <div className="inspect-pane-slot context-slot">
          <ContextPane
            reader={reader}
            paneRef={contextRef}
            activeSourceIds={relatedSet}
            focusedSourceId={displayedFocusedSourceId}
            directionBySource={directionBySource}
            canFocus={Boolean(lockedRange)}
            expanded={expandedPane === "context"}
            maxInfluence={maxSourceInfluence}
            onFocusSource={expandedPane === "context" ? lockContextSource : focusSource}
            onPreviewSource={previewContextSource}
            onToggleExpanded={() => {
              setHoveredSourceId(undefined);
              setHoveredRange(undefined);
              setExpandedPane((current) => current === "context" ? undefined : "context");
            }}
          />
          <div className="local-registration-layer context-registration" aria-hidden="true">
            {displayedRelationships.filter((relationship) => relationship.localY !== undefined).map((relationship) => (
              <i
                key={relationship.sourceId}
                className={`direction-${relationship.direction}`}
                style={{ "--registration-y": `${relationship.localY}px` } as CSSProperties}
              />
            ))}
          </div>
        </div>
        <EvidenceMap
          mapRef={mapRef}
          geometry={geometry}
          activeRange={activeRange}
          locked={Boolean(lockedRange)}
          focusedSourceId={displayedFocusedSourceId}
          visibleSourceIds={visibleSourceIds}
          sourceLabels={sourceLabels}
          onFocusSource={focusSource}
          onOffscreen={navigateOffscreen}
          onAnswerOffscreen={navigateAnswer}
        />
        <div className="inspect-pane-slot answer-slot">
          <AnswerPane
            reader={reader}
            paneRef={answerRef}
            segments={answerSegments}
            activeRange={activeRange}
            lockedRange={lockedRange}
            lockedRelationshipCount={lockedRange && query.phase === "ready" ? activeSourceIds.length : undefined}
            expanded={expandedPane === "answer"}
            maxInfluence={maxAnswerInfluence}
            onPreview={(range) => {
              setHoveredSourceId(undefined);
              setHoveredRange(range);
            }}
            onLock={(range) => {
              if (rangesEqual(lockedRange, range)) setCoordinate();
              else {
                setCoordinate(range);
                setMobilePane("answer");
              }
            }}
            onClear={() => setCoordinate()}
            onToggleExpanded={() => {
              setHoveredSourceId(undefined);
              setHoveredRange(undefined);
              setExpandedPane((current) => current === "answer" ? undefined : "answer");
            }}
          />
          <div className="local-registration-layer answer-registration" aria-hidden="true">
            {activeRange && geometry.answerLocalY !== undefined && (
              <i style={{ "--registration-y": `${geometry.answerLocalY}px` } as CSSProperties} />
            )}
          </div>
        </div>
      </div>
      {expandedPane && (
        <HeatmapPeek
          reader={reader}
          pane={expandedPane}
          sourceId={expandedPane === "context" ? hoveredSourceId ?? focusedSourceId : undefined}
          range={expandedPane === "answer" ? activeRange : undefined}
        />
      )}
      {route.answerStart !== undefined && !lockedRange && (
        <p className="selection-unavailable" role="status">Selection unavailable.</p>
      )}
      {lockedRange && (
        <>
          <EvidenceInspector
            reader={reader}
            range={lockedRange}
            query={query}
            sourceIds={activeSourceIds}
            focusedSourceId={focusedSourceId}
            onPrevious={() => navigateSource(-1)}
            onNext={() => navigateSource(1)}
            onClear={() => setCoordinate()}
          />
        </>
      )}
    </>
  );
}

function phaseLabel(value: string): string {
  return value.replaceAll("_", " ");
}

function EffectsPanel({
  reader,
  route,
  onCompleted,
}: {
  reader: ContextInvestigationReader;
  route: InspectRoute;
  onCompleted: () => void;
}) {
  const [job, setJob] = useState<ContextEffectsJob | undefined>();
  const [action, setAction] = useState<"idle" | "starting" | "cancelling">("idle");
  const [error, setError] = useState<string | undefined>();
  const [pollRevision, setPollRevision] = useState(0);
  const actionController = useRef<AbortController | undefined>(undefined);
  const completedJobId = useRef<string | undefined>(undefined);
  const sourceIds = reader.measurement.sourceUniverse;

  useEffect(() => {
    actionController.current?.abort();
    actionController.current = undefined;
    completedJobId.current = undefined;
    setJob(undefined);
    setAction("idle");
    setError(undefined);
    setPollRevision(0);
  }, [route.runId]);

  useEffect(() => {
    if (!job || isTerminalContextEffectsJob(job)) return undefined;
    const controller = new AbortController();
    let timer: number | undefined;
    const poll = async () => {
      try {
        const next = await contextInvestigationApi.effectsJob(
          route.runId, job.jobId, controller.signal,
        );
        if (controller.signal.aborted) return;
        setJob(next);
        if (next.state === "completed" && completedJobId.current !== next.jobId) {
          completedJobId.current = next.jobId;
          onCompleted();
        }
        if (!isTerminalContextEffectsJob(next)) {
          timer = window.setTimeout(() => { void poll(); }, 750);
        }
      } catch (pollError: unknown) {
        if (!controller.signal.aborted) setError(describeError(pollError));
      }
    };
    void poll();
    return () => {
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [job?.jobId, job?.state, onCompleted, pollRevision, route.runId]);

  const start = async () => {
    actionController.current?.abort();
    const controller = new AbortController();
    actionController.current = controller;
    setAction("starting");
    setError(undefined);
    try {
      const next = await contextInvestigationApi.startEffects(
        route.runId, { sourceIds }, controller.signal,
      );
      if (!controller.signal.aborted) {
        setJob(next);
        if (next.state === "completed" && completedJobId.current !== next.jobId) {
          completedJobId.current = next.jobId;
          onCompleted();
        }
      }
    } catch (startError: unknown) {
      if (!controller.signal.aborted) setError(describeError(startError));
    } finally {
      if (!controller.signal.aborted) setAction("idle");
    }
  };

  const cancel = async () => {
    if (!job?.cancellable) return;
    actionController.current?.abort();
    const controller = new AbortController();
    actionController.current = controller;
    setAction("cancelling");
    setError(undefined);
    try {
      const next = await contextInvestigationApi.cancelEffects(
        route.runId, job.jobId, controller.signal,
      );
      if (!controller.signal.aborted) setJob(next);
    } catch (cancelError: unknown) {
      if (!controller.signal.aborted) setError(describeError(cancelError));
    } finally {
      if (!controller.signal.aborted) setAction("idle");
    }
  };

  const active = Boolean(job && !isTerminalContextEffectsJob(job));
  return (
    <section className="effects-search-panel" aria-label="Context measurement">
      <div className="effects-search-copy">
        <span className="eyebrow">Measurement</span>
        <strong>{sourceIds.length ? plural(sourceIds.length, "source") : "No measurable sources"}</strong>
        <span>Each source is removed once and scored against the recorded answer.</span>
      </div>
      <div className="effects-search-action">
        {active ? (
          <button type="button" disabled={!job?.cancellable || action !== "idle"} onClick={() => { void cancel(); }}>
            {action === "cancelling" ? "Cancelling…" : "Cancel"}
          </button>
        ) : (
          <button type="button" disabled={!sourceIds.length || action !== "idle"} onClick={() => { void start(); }}>
            {action === "starting" ? "Starting…" : job ? "Run again" : "Measure context"}
          </button>
        )}
      </div>
      {job && (
        <div className={`effects-search-progress state-${job.state}`} role="status" aria-live="polite">
          <span>{job.state === "completed" ? "Completed" : job.state === "failed" ? "Failed" : job.state === "cancelled" ? "Cancelled" : phaseLabel(job.progress.phase)}</span>
          <strong>{job.progress.percent.toFixed(0)}%</strong>
          {!isTerminalContextEffectsJob(job) && <progress max="100" value={job.progress.percent} />}
          {job.result && (
            <small>
              {job.result.completedSourceCount} measured · {job.result.unavailableSourceCount} unavailable · {job.result.failedSourceCount} failed · {job.result.notMeasuredSourceCount} not measured
            </small>
          )}
          {job.error && <small className="is-failed">{job.error.message}</small>}
        </div>
      )}
      {error && (
        <p className="effects-search-error" role="alert">
          <span>{error}</span>
          {active && (
            <button type="button" onClick={() => {
              setError(undefined);
              setPollRevision((value) => value + 1);
            }}>Retry status</button>
          )}
        </p>
      )}
    </section>
  );
}

interface AnswerInfluenceRow {
  range: ActiveRange;
  text: string;
  influence: number;
  direction: EvidenceDirection;
  sourceCount: number;
}

function EfficiencyOverview({
  reader,
  onOpenSource,
  onOpenAnswer,
}: {
  reader: ContextInvestigationReader;
  onOpenSource: (sourceId: string, range: ActiveRange) => void;
  onOpenAnswer: (range: ActiveRange) => void;
}) {
  const contextRows = useMemo(() => reader.sources
    .filter((source) => source.measurement.status === "measured")
    .map((source) => ({
      source,
      text: source.label?.trim() || sourceRecordedText(reader, source),
      influence: sourceInfluence(source),
      locus: strongestLocusForSource(reader, source.sourceId),
    }))
    .sort((left, right) => right.influence - left.influence), [reader]);
  const answerRows = useMemo(() => {
    const grouped = new Map<string, ContextInvestigationLocus[]>();
    for (const locus of reader.loci) {
      const key = `${locus.answerUnicodeRange[0]}:${locus.answerUnicodeRange[1]}`;
      const current = grouped.get(key) ?? [];
      current.push(locus);
      grouped.set(key, current);
    }
    return [...grouped.values()].map<AnswerInfluenceRow>((loci) => {
      const [first] = loci;
      const range = {
        start: first.answerUnicodeRange[0],
        end: first.answerUnicodeRange[1],
        key: `${first.answerUnicodeRange[0]}:${first.answerUnicodeRange[1]}`,
      };
      return {
        range,
        text: first.answerText,
        influence: loci.reduce((sum, locus) => sum + locus.absoluteDeltaNats, 0),
        direction: directionForLoci(loci),
        sourceCount: new Set(loci.map((locus) => locus.sourceId)).size,
      };
    }).sort((left, right) => right.influence - left.influence);
  }, [reader.loci]);
  const maxContext = contextRows[0]?.influence ?? 0;
  const maxAnswer = answerRows[0]?.influence ?? 0;
  const contextHighlights = contextRows.length > 1
    ? [{ row: contextRows[0], label: "Highest" }, { row: contextRows[contextRows.length - 1], label: "Lowest" }]
    : contextRows.map((row) => ({ row, label: "Highest" }));
  const answerHighlights = answerRows.length > 1
    ? [{ row: answerRows[0], label: "Highest" }, { row: answerRows[answerRows.length - 1], label: "Lowest" }]
    : answerRows.map((row) => ({ row, label: "Highest" }));
  return (
    <section className="efficiency-overview" aria-labelledby="efficiency-heading">
      <header className="overview-heading">
        <div>
          <span className="eyebrow">Summary</span>
          <h2 id="efficiency-heading">Influence overview</h2>
        </div>
        <p>Ranks measured text by its recorded-answer effect. This measures influence, not correctness.</p>
      </header>
      <div className="efficiency-grid">
        <section aria-labelledby="context-ranking-heading">
          <header>
            <h3 id="context-ranking-heading">Context</h3>
            <span>{plural(contextRows.length, "measured span")}</span>
          </header>
          {contextRows.length ? (
            <ol className="influence-ranking">
              {contextHighlights.map(({ row, label }) => (
                <li key={row.source.sourceId}>
                  <button
                    type="button"
                    disabled={!row.locus}
                    onClick={() => {
                      if (!row.locus) return;
                      onOpenSource(row.source.sourceId, {
                        start: row.locus.answerUnicodeRange[0],
                        end: row.locus.answerUnicodeRange[1],
                        key: `${row.locus.answerUnicodeRange[0]}:${row.locus.answerUnicodeRange[1]}`,
                      });
                    }}
                  >
                    <span className="ranking-copy">
                      <span>{label}</span>
                      <strong>{excerpt(row.text, 116)}</strong>
                      <small>{row.source.summary.totalContinuationDeltaNats !== null
                        ? `${row.source.summary.totalContinuationDeltaNats > 0 ? "+" : ""}${row.source.summary.totalContinuationDeltaNats.toFixed(2)} total Δ nats`
                        : "No score"}</small>
                    </span>
                    <span className="ranking-meter" aria-hidden="true"><i style={{ "--meter-size": `${maxContext ? row.influence / maxContext * 100 : 0}%` } as CSSProperties} /></span>
                  </button>
                </li>
              ))}
            </ol>
          ) : <p className="overview-empty">Measure Context Effects to rank context.</p>}
        </section>
        <section aria-labelledby="answer-ranking-heading">
          <header>
            <h3 id="answer-ranking-heading">Answer</h3>
            <span>{plural(answerRows.length, "affected span")}</span>
          </header>
          {answerRows.length ? (
            <ol className="influence-ranking">
              {answerHighlights.map(({ row, label }) => (
                <li key={row.range.key}>
                  <button type="button" onClick={() => onOpenAnswer(row.range)}>
                    <span className="ranking-copy">
                      <span>{label}</span>
                      <strong>{excerpt(row.text, 116)}</strong>
                      <small>{row.influence.toFixed(2)} absolute Δ nats · {plural(row.sourceCount, "source")}</small>
                    </span>
                    <span className={`ranking-meter direction-${row.direction}`} aria-hidden="true"><i style={{ "--meter-size": `${maxAnswer ? row.influence / maxAnswer * 100 : 0}%` } as CSSProperties} /></span>
                  </button>
                </li>
              ))}
            </ol>
          ) : <p className="overview-empty">No above-floor answer effects were recorded.</p>}
        </section>
      </div>
    </section>
  );
}

function LeanContextPanel({
  reader,
  route,
}: {
  reader: ContextInvestigationReader;
  route: InspectRoute;
}) {
  const [job, setJob] = useState<LeanContextJob | undefined>();
  const [action, setAction] = useState<"idle" | "starting" | "cancelling">("idle");
  const [error, setError] = useState<string | undefined>();
  const [pollRevision, setPollRevision] = useState(0);
  const actionController = useRef<AbortController | undefined>(undefined);

  useEffect(() => {
    actionController.current?.abort();
    actionController.current = undefined;
    setJob(undefined);
    setAction("idle");
    setError(undefined);
    setPollRevision(0);
  }, [route.runId]);

  useEffect(() => {
    if (!job || isTerminalLeanContextJob(job)) return undefined;
    const controller = new AbortController();
    let timer: number | undefined;
    const poll = async () => {
      try {
        const next = await contextInvestigationApi.leanContextJob(
          route.runId, job.jobId, controller.signal,
        );
        if (controller.signal.aborted) return;
        setJob(next);
        if (!isTerminalLeanContextJob(next)) {
          timer = window.setTimeout(() => { void poll(); }, 750);
        }
      } catch (pollError: unknown) {
        if (!controller.signal.aborted) setError(describeError(pollError));
      }
    };
    void poll();
    return () => {
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [job?.jobId, job?.state, pollRevision, route.runId]);

  const start = async () => {
    actionController.current?.abort();
    const controller = new AbortController();
    actionController.current = controller;
    setAction("starting");
    setError(undefined);
    try {
      const next = await contextInvestigationApi.startLeanContext(route.runId, controller.signal);
      if (!controller.signal.aborted) setJob(next);
    } catch (startError: unknown) {
      if (!controller.signal.aborted) setError(describeError(startError));
    } finally {
      if (!controller.signal.aborted) setAction("idle");
    }
  };

  const cancel = async () => {
    if (!job?.cancellable) return;
    actionController.current?.abort();
    const controller = new AbortController();
    actionController.current = controller;
    setAction("cancelling");
    setError(undefined);
    try {
      const next = await contextInvestigationApi.cancelLeanContext(
        route.runId, job.jobId, controller.signal,
      );
      if (!controller.signal.aborted) setJob(next);
    } catch (cancelError: unknown) {
      if (!controller.signal.aborted) setError(describeError(cancelError));
    } finally {
      if (!controller.signal.aborted) setAction("idle");
    }
  };

  const result = job?.result;
  const removedSources = result?.best?.removedSourceIds.flatMap((sourceId) => {
    const source = reader.sources.find((item) => item.sourceId === sourceId);
    return source ? [source] : [];
  }) ?? [];
  const active = Boolean(job && !isTerminalLeanContextJob(job));
  return (
    <section className="lean-context-panel" aria-labelledby="lean-context-heading">
      <header className="overview-heading">
        <div>
          <span className="eyebrow">Bounded search</span>
          <h2 id="lean-context-heading">Lean Context</h2>
        </div>
        <p>Searches for a leaner context within a 16-execution budget while the recorded answer stays inside the balanced loss tolerance.</p>
      </header>
      <div className="lean-controls">
        <div className="lean-tolerance">
          <span className="eyebrow">Balanced tolerance</span>
          <strong>≤ 0.05 mean loss · ≤ 0.70 max token loss</strong>
          <small>Only probability lost from the recorded answer counts; gains do not offset losses.</small>
        </div>
        {active ? (
          <button type="button" disabled={!job?.cancellable || action !== "idle"} onClick={() => { void cancel(); }}>
            {action === "cancelling" ? "Cancelling…" : "Cancel search"}
          </button>
        ) : (
          <button type="button" disabled={!reader.measurement.sourceUniverse.length || action !== "idle"} onClick={() => { void start(); }}>
            {action === "starting" ? "Starting…" : job ? "Run again" : "Find lean context"}
          </button>
        )}
      </div>
      {job && (
        <div className={`lean-progress state-${job.state}`} role="status" aria-live="polite">
          <span>{job.state === "completed" ? "Search complete" : job.state === "failed" ? "Search failed" : job.state === "cancelled" ? "Search cancelled" : phaseLabel(job.progress.phase)}</span>
          <strong>{job.progress.percent.toFixed(0)}%</strong>
          {!isTerminalLeanContextJob(job) && <progress max="100" value={job.progress.percent} />}
          {job.error && <small className="is-failed">{job.error.message}</small>}
        </div>
      )}
      {result?.status === "completed" && result.original && result.best && result.reduction && (
        <div className="lean-result">
          <div className="lean-result-summary">
            <article>
              <span>Prompt tokens</span>
              <strong>{result.original.renderedPromptTokenCost} <i>→</i> {result.best.renderedPromptTokenCost}</strong>
              <small>{result.reduction.removedPromptTokens} removed</small>
            </article>
            <article>
              <span>Context retained</span>
              <strong>{(result.reduction.retainedPromptTokenFraction * 100).toFixed(0)}%</strong>
              <small>{plural(result.reduction.retainedSourceCount, "span")} retained</small>
            </article>
            <article>
              <span>Search work</span>
              <strong>{result.budget.usedNewExecutions} / {result.budget.maxNewExecutions}</strong>
              <small>{plural(result.budget.reusedObservationCount, "measurement")} reused</small>
            </article>
          </div>
          <p className="lean-claim">Best found within this search budget. It is not a proof of the mathematically smallest context.</p>
          <section className="lean-removed" aria-labelledby="lean-removed-heading">
            <header>
              <h3 id="lean-removed-heading">Removed context</h3>
              <span>{plural(removedSources.length, "span")}</span>
            </header>
            {removedSources.length ? (
              <ol>
                {removedSources.map((source) => (
                  <li key={source.sourceId}>{excerpt(source.label?.trim() || sourceRecordedText(reader, source), 180)}</li>
                ))}
              </ol>
            ) : <p>No tested reduction stayed inside the balanced tolerance.</p>}
          </section>
          {result.best.loss && (
            <div className="lean-loss">
              <span>Best candidate loss</span>
              <strong>{result.best.loss.meanPositiveSupportLossNatsPerToken.toFixed(3)} mean · {result.best.loss.maxTokenSupportLossNats.toFixed(3)} max nats</strong>
              <small>{plural(result.best.loss.positiveLossTokenCount, "answer token")} lost support</small>
            </div>
          )}
        </div>
      )}
      {result && result.status !== "completed" && (
        <p className="lean-unavailable">{result.reason ?? "Lean Context is unavailable for this run."}</p>
      )}
      {!job && <p className="lean-empty">Run this after Context Effects to reuse its single-source measurements and spend the remaining budget on combined removals.</p>}
      {error && (
        <p className="effects-search-error" role="alert">
          <span>{error}</span>
          {active && <button type="button" onClick={() => {
            setError(undefined);
            setPollRevision((value) => value + 1);
          }}>Retry status</button>}
        </p>
      )}
    </section>
  );
}

function ContextInspectSurface({ route }: { route: InspectRoute }) {
  const [readerRevision, setReaderRevision] = useState(0);
  const [view, setView] = useState<InvestigationView>("links");
  const readerLoader = useCallback(
    (signal: AbortSignal) => contextInvestigationApi.reader(route.runId, signal),
    [route.runId],
  );
  const readerResource = useResource(`${route.runId}:${readerRevision}`, readerLoader);
  const refreshReader = useCallback(() => setReaderRevision((value) => value + 1), []);
  useEffect(() => setView("links"), [route.runId]);
  const openLinks = useCallback((range: ActiveRange, sourceId?: string) => {
    setView("links");
    window.location.hash = routeHref({
      surface: "inspect",
      runId: route.runId,
      answerStart: range.start,
      answerEnd: range.end,
      ...(sourceId ? { sourceId } : {}),
    });
  }, [route.runId]);
  return (
    <div className="inspect-surface">
      <header className="inspect-heading">
        <div>
          <a className="back-link" href={routeHref({ surface: "runs" })}>← Runs</a>
          <h1>Context Effects</h1>
          <code>{shortId(route.runId)}</code>
        </div>
        {readerResource.phase === "ready" && (
          <span className={`reader-status status-${readerResource.value.status.replace("_", "-")}`}>
            <i aria-hidden="true" />{statusLabel(readerResource.value.status)}
          </span>
        )}
      </header>
      <div className="inspect-layout">
        <section className="inspect-instrument" aria-label="Influence">
          <ReaderState resource={readerResource} />
          {readerResource.phase === "ready" && (
            <>
              {readerResource.value.capabilities.contextEffects === "measurable" && (
                <EffectsPanel
                  reader={readerResource.value}
                  route={route}
                  onCompleted={refreshReader}
                />
              )}
              {readerResource.value.reason && readerResource.value.status !== "completed" && (
                <p className={`reader-reason status-${readerResource.value.status.replace("_", "-")}`} role={readerResource.value.status === "stale" || readerResource.value.status === "unavailable" ? "alert" : "status"}>
                  {readerResource.value.reason}
                </p>
              )}
              <nav className="investigation-views" aria-label="Investigation view">
                <button type="button" aria-current={view === "links" ? "page" : undefined} onClick={() => setView("links")}>Linked view</button>
                <button type="button" aria-current={view === "efficiency" ? "page" : undefined} onClick={() => setView("efficiency")}>Efficiency</button>
                <button type="button" aria-current={view === "lean" ? "page" : undefined} onClick={() => setView("lean")}>Lean Context</button>
              </nav>
              {readerResource.value.answer.status === "available" || readerResource.value.context.blocks.length > 0 ? (
                <>
                  <div hidden={view !== "links"}>
                    <InspectReader reader={readerResource.value} route={route} />
                  </div>
                  <div hidden={view !== "efficiency"}>
                    <EfficiencyOverview
                      reader={readerResource.value}
                      onOpenSource={(sourceId, range) => openLinks(range, sourceId)}
                      onOpenAnswer={(range) => openLinks(range)}
                    />
                  </div>
                  <div hidden={view !== "lean"}>
                    <LeanContextPanel reader={readerResource.value} route={route} />
                  </div>
                </>
              ) : <p className="inspect-state is-unavailable">Recorded prose unavailable.</p>}
            </>
          )}
        </section>
      </div>
    </div>
  );
}

export function InspectSurface({ route }: { route: InspectRoute }) {
  return (
    <>
      {route.view === "minimal-context"
        ? <MinimalContextSurface route={route} />
        : <ContextInspectSurface route={route} />}
      <RunDetailsDisclosure runId={route.runId} />
    </>
  );
}
