import {
  useCallback,
  useEffect,
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
import {
  contextInvestigationApi,
  type ContextEffectsRequest,
  type ScoringMode,
} from "../../data/contextInvestigationClient";
import {
  isTerminalContextEffectsJob,
  type ContextEffectsJob,
} from "../../data/contextInvestigationJobs";
import {
  isTerminalContextInteractionsJob,
  type ContextInteractionEstimate,
  type ContextInteractionsJob,
} from "../../data/contextInteractions";
import {
  isTerminalLeanContextJob,
  type LeanContextJob,
} from "../../data/leanContext";
import {
  codePointLength,
  isValidCodePointRange,
  semanticTextRanges,
  segmentAnnotatedText,
  sliceCodePoints,
  type AnnotatedTextSegment,
  type CodePointRange,
} from "./unicodeRanges";
import { RunDetailsDisclosure } from "./ContextReceipt";
import { answerDisplayScale, answerHeatForRange, answerHeatForLens, signedContributions, type AnswerHeat, type AnswerLens } from "./answerHeat";
import "./inspect.css";

type Resource<T> =
  | { phase: "loading" }
  | { phase: "ready"; value: T }
  | { phase: "unavailable"; error: string }
  | { phase: "malformed"; error: string };

type OptionalResource<T> = { phase: "idle" } | Resource<T>;
type EvidenceDirection = "support" | "suppression" | "mixed";
type MobilePane = "context" | "answer";
type InvestigationView = "links" | "overview";

interface ActiveRange extends CodePointRange {
  key: string;
}

interface LinkedRangeContour {
  range: ActiveRange;
  direction: EvidenceDirection;
}

type AnswerHeatScale = ReturnType<typeof answerDisplayScale>;
interface AnswerPassage {
  range: AnnotatedTextSegment;
  key: string;
  heat: AnswerHeat;
}

interface InfluenceFocusItem {
  key: string;
  direction: "support" | "suppression";
  delta: number;
  text: string;
  onSelect: () => void;
}

interface InfluenceFocusModel {
  key: string;
  title: string;
  items: readonly InfluenceFocusItem[];
  primaryDirection?: "support" | "suppression";
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

interface AnswerAnchorGeometry {
  key: string;
  mapY: number;
  localY?: number;
  offscreen?: "above" | "below";
  direction?: EvidenceDirection;
}

interface EvidenceGeometry {
  width: number;
  height: number;
  answerAnchors: AnswerAnchorGeometry[];
  relationships: RelationshipGeometry[];
  coverage: CoverageGeometry[];
}

const EMPTY_GEOMETRY: EvidenceGeometry = {
  width: 112,
  height: 480,
  answerAnchors: [],
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

function mergeLinkedRangeContours(
  ranges: readonly ActiveRange[],
  directions: ReadonlyMap<string, EvidenceDirection>,
): LinkedRangeContour[] {
  const sorted = [...ranges].sort((left, right) => left.start - right.start || left.end - right.end);
  const merged: LinkedRangeContour[] = [];
  for (const range of sorted) {
    const direction = directions.get(range.key) ?? "mixed";
    const previous = merged.at(-1);
    if (previous && previous.direction === direction && range.start <= previous.range.end) {
      const combined = { start: previous.range.start, end: Math.max(previous.range.end, range.end) };
      merged[merged.length - 1] = { direction, range: { ...combined, key: rangeKey(combined) } };
    } else {
      merged.push({ range: { ...range }, direction });
    }
  }
  return merged;
}

function semanticRangeFor(
  ranges: readonly CodePointRange[],
  target: CodePointRange,
): ActiveRange {
  const selected = ranges
    .filter((range) => rangesIntersect(range, target))
    .sort((left, right) => {
      const leftOverlap = Math.min(left.end, target.end) - Math.max(left.start, target.start);
      const rightOverlap = Math.min(right.end, target.end) - Math.max(right.start, target.start);
      return rightOverlap - leftOverlap || left.start - right.start;
    })[0] ?? target;
  return { ...selected, key: rangeKey(selected) };
}

function robustCeiling(values: readonly number[]): number {
  const positive = values.filter((value) => value > 0).sort((left, right) => left - right);
  if (!positive.length) return 0;
  return positive[Math.max(0, Math.ceil(positive.length * .9) - 1)];
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

type VisualRelationship = ContextInvestigationLocus
  | (ContextInvestigationEffect & { direction: "support" | "suppression" });

function relationshipMagnitude(relationship: VisualRelationship): number {
  return "absoluteDeltaNats" in relationship
    ? relationship.absoluteDeltaNats
    : Math.abs(relationship.deltaNats ?? 0);
}

const MAX_PINNED_THREADS_PER_DIRECTION = 3;

function strongestRelationshipsPerDirection<T extends VisualRelationship>(
  relationships: readonly T[],
  limit = MAX_PINNED_THREADS_PER_DIRECTION,
): T[] {
  return (["support", "suppression"] as const).flatMap((direction) => {
    return relationships
      .filter((relationship) => relationship.direction === direction)
      .sort((left, right) => relationshipMagnitude(right) - relationshipMagnitude(left))
      .slice(0, limit);
  });
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

function sourceHeat(source: ContextInvestigationSource | undefined): number {
  // Context is a source-side view: encode the strongest answer-token effect
  // instead of summing across the whole continuation, which flattens long answers.
  return source?.summary.maxAbsTokenDeltaNats ?? 0;
}

function heatDirection(value: number): EvidenceDirection {
  if (value > 0) return "support";
  if (value < 0) return "suppression";
  return "mixed";
}

function heatBand(normalized: number): number {
  if (normalized <= 0) return -1;
  if (normalized <= .18) return 0;
  if (normalized <= .34) return 1;
  if (normalized <= .52) return 2;
  if (normalized <= .70) return 3;
  if (normalized <= .88) return 4;
  return 5;
}

function heatBandStyle(band: number): CSSProperties {
  const visible = band >= 0;
  const index = Math.max(0, Math.min(5, band));
  const support = [
    "var(--support-ramp-1)",
    "var(--support-ramp-2)",
    "var(--support-ramp-3)",
    "var(--support-ramp-4)",
    "var(--support-ramp-5)",
    "var(--support-ramp-6)",
  ] as const;
  const suppression = [
    "var(--suppression-ramp-0)",
    "var(--suppression-ramp-1)",
    "var(--suppression-ramp-2)",
    "var(--suppression-ramp-3)",
    "var(--suppression-ramp-4)",
    "var(--suppression-ramp-5)",
  ] as const;
  const supportFill = [
    "var(--heatmap-support-fill-1)",
    "var(--heatmap-support-fill-2)",
    "var(--heatmap-support-fill-3)",
    "var(--heatmap-support-fill-4)",
    "var(--heatmap-support-fill-5)",
    "var(--heatmap-support-fill-5)",
  ] as const;
  const suppressionFill = [
    "var(--heatmap-suppression-fill-1)",
    "var(--heatmap-suppression-fill-2)",
    "var(--heatmap-suppression-fill-3)",
    "var(--heatmap-suppression-fill-4)",
    "var(--heatmap-suppression-fill-5)",
    "var(--heatmap-suppression-fill-5)",
  ] as const;
  return {
    "--heat-band": band,
    "--heat-edge-width": band >= 4 ? "2px" : "0px",
    "--heat-support-color": support[index],
    "--heat-suppression-color": suppression[index],
    "--heat-support-fill": visible ? supportFill[index] : "var(--heatmap-support-fill-0)",
    "--heat-suppression-fill": visible ? suppressionFill[index] : "var(--heatmap-suppression-fill-0)",
  } as CSSProperties;
}

function heatStyle(intensity: number): CSSProperties {
  return heatBandStyle(heatBand(Math.max(0, Math.min(1, intensity))));
}

function answerHeatStyle(heat: AnswerHeat, scale: AnswerHeatScale): CSSProperties {
  const relative = Math.min(1, heat.intensity / scale.ceiling);
  const band = heatBand(relative);
  const signedTotal = heat.support + heat.suppression;
  const supportShare = signedTotal ? heat.support / signedTotal * 100 : 50;
  return {
    ...heatBandStyle(band),
    "--answer-bar-height": `${Math.max(8, relative * 100)}%`,
    "--answer-fill-strength": `${[28, 40, 52, 70, 84, 100][Math.max(0, band)]}%`,
    "--heat-support-share": `${supportShare}%`,
    "--heat-suppression-share": `${100 - supportShare}%`,
  } as CSSProperties;
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

function LeanAnswerDiffText({
  text,
  other,
  kind,
}: {
  text: string;
  other: string;
  kind: "before" | "after";
}) {
  if (text === other) return <>{text}</>;
  const points = Array.from(text);
  const otherPoints = Array.from(other);
  let prefixLength = 0;
  while (
    prefixLength < points.length
    && prefixLength < otherPoints.length
    && points[prefixLength] === otherPoints[prefixLength]
  ) prefixLength += 1;
  let suffixLength = 0;
  while (
    suffixLength < points.length - prefixLength
    && suffixLength < otherPoints.length - prefixLength
    && points[points.length - suffixLength - 1] === otherPoints[otherPoints.length - suffixLength - 1]
  ) suffixLength += 1;
  const changedEnd = points.length - suffixLength;
  const changed = points.slice(prefixLength, changedEnd).join("");
  return (
    <>
      {points.slice(0, prefixLength).join("")}
      <span className={`lean-answer-change is-${kind}`}>
        {changed || <span aria-label="No text">∅</span>}
      </span>
      {points.slice(changedEnd).join("")}
    </>
  );
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
    .sort((left, right) => sourceHeat(right) - sourceHeat(left))[0];
}

function strongestLocusForSource(
  reader: ContextInvestigationReader,
  sourceId: string,
): ContextInvestigationLocus | undefined {
  return reader.loci
    .filter((locus) => locus.sourceId === sourceId)
    .sort((left, right) => right.absoluteDeltaNats - left.absoluteDeltaNats)[0];
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

interface ContinuousOutlineGeometry {
  width: number;
  height: number;
  path: string;
}

function continuousOutlinePath(
  pane: HTMLElement,
  elements: readonly HTMLElement[],
): ContinuousOutlineGeometry | undefined {
  const paneRect = pane.getBoundingClientRect();
  const fragments = elements.flatMap((element) => Array.from(element.getClientRects()))
    .filter((rect) => rect.width > 0 && rect.height > 0)
    .map((rect) => ({
      top: rect.top - paneRect.top + pane.scrollTop - 2,
      right: rect.right - paneRect.left + pane.scrollLeft + 3,
      bottom: rect.bottom - paneRect.top + pane.scrollTop + 2,
      left: rect.left - paneRect.left + pane.scrollLeft - 3,
    }))
    .sort((left, right) => left.top - right.top || left.left - right.left);
  if (!fragments.length) return undefined;
  const rows: typeof fragments = [];
  for (const fragment of fragments) {
    const previous = rows.at(-1);
    if (previous && Math.abs(previous.top - fragment.top) < 2) {
      previous.left = Math.min(previous.left, fragment.left);
      previous.right = Math.max(previous.right, fragment.right);
      previous.bottom = Math.max(previous.bottom, fragment.bottom);
    } else rows.push({ ...fragment });
  }
  const connected = rows.map((row, index) => ({
    ...row,
    top: index === 0 ? row.top : (rows[index - 1].bottom + row.top) / 2,
    bottom: index === rows.length - 1 ? row.bottom : (row.bottom + rows[index + 1].top) / 2,
  }));
  const first = connected[0];
  const last = connected[connected.length - 1];
  const corner = Math.min(6, (first.right - first.left) / 4, (first.bottom - first.top) / 4);
  const downwardStep = (currentX: number, nextX: number, y: number) => {
    const direction = Math.sign(nextX - currentX);
    if (!direction) return ` V ${y}`;
    const radius = Math.min(6, Math.abs(nextX - currentX) / 2);
    return ` V ${y - radius} Q ${currentX} ${y} ${currentX + direction * radius} ${y}`
      + ` H ${nextX - direction * radius} Q ${nextX} ${y} ${nextX} ${y + radius}`;
  };
  const upwardStep = (currentX: number, nextX: number, y: number) => {
    const direction = Math.sign(nextX - currentX);
    if (!direction) return ` V ${y}`;
    const radius = Math.min(6, Math.abs(nextX - currentX) / 2);
    return ` V ${y + radius} Q ${currentX} ${y} ${currentX + direction * radius} ${y}`
      + ` H ${nextX - direction * radius} Q ${nextX} ${y} ${nextX} ${y - radius}`;
  };
  let path = `M ${first.left + corner} ${first.top} H ${first.right - corner}`
    + ` Q ${first.right} ${first.top} ${first.right} ${first.top + corner}`;
  for (let index = 0; index < connected.length - 1; index += 1) {
    path += downwardStep(connected[index].right, connected[index + 1].right, connected[index].bottom);
  }
  const bottomCorner = Math.min(6, (last.right - last.left) / 4, (last.bottom - last.top) / 4);
  path += ` V ${last.bottom - bottomCorner} Q ${last.right} ${last.bottom} ${last.right - bottomCorner} ${last.bottom}`
    + ` H ${last.left + bottomCorner} Q ${last.left} ${last.bottom} ${last.left} ${last.bottom - bottomCorner}`;
  for (let index = connected.length - 1; index > 0; index -= 1) {
    path += upwardStep(connected[index].left, connected[index - 1].left, connected[index].top);
  }
  path += ` V ${first.top + corner} Q ${first.left} ${first.top} ${first.left + corner} ${first.top} Z`;
  return {
    width: Math.max(pane.scrollWidth, pane.clientWidth),
    height: Math.max(pane.scrollHeight, pane.clientHeight),
    path,
  };
}

function ContinuousSpanOutline({
  paneRef,
  sourceId,
  answerRange,
  pinned = false,
  linkedDirection,
}: {
  paneRef: RefObject<HTMLElement | null>;
  sourceId?: string;
  answerRange?: ActiveRange;
  pinned?: boolean;
  linkedDirection?: EvidenceDirection;
}) {
  const [geometry, setGeometry] = useState<ContinuousOutlineGeometry>();
  const targetKey = sourceId ?? answerRange?.key;
  useEffect(() => {
    const pane = paneRef.current;
    if (!pane || !targetKey) {
      setGeometry(undefined);
      return undefined;
    }
    const elements = sourceId
      ? sourceElements(pane, sourceId).filter((element) => (
        element.classList.contains("context-segment") || element.classList.contains("context-block")
      ))
      : answerRange
        ? answerElements(pane, answerRange).filter((element) => element.classList.contains("answer-segment"))
        : [];
    if (!elements.length) {
      setGeometry(undefined);
      return undefined;
    }
    let frame: number | undefined;
    const update = () => {
      frame = undefined;
      setGeometry(continuousOutlinePath(pane, elements));
    };
    const schedule = () => {
      if (frame === undefined) frame = window.requestAnimationFrame(update);
    };
    update();
    const observer = typeof ResizeObserver === "undefined" ? undefined : new ResizeObserver(schedule);
    observer?.observe(pane);
    elements.forEach((element) => observer?.observe(element));
    window.addEventListener("resize", schedule);
    return () => {
      if (frame !== undefined) window.cancelAnimationFrame(frame);
      observer?.disconnect();
      window.removeEventListener("resize", schedule);
    };
  }, [answerRange, paneRef, sourceId, targetKey]);
  if (!geometry) return null;
  return (
    <svg
      className={`continuous-span-outline ${linkedDirection ? `is-linked direction-${linkedDirection}` : pinned ? "is-pinned" : "is-preview"}`}
      width={geometry.width}
      height={geometry.height}
      viewBox={`0 0 ${geometry.width} ${geometry.height}`}
      aria-hidden="true"
    >
      <path className="continuous-span-outline-halo" d={geometry.path} />
      <path className="continuous-span-outline-edge" d={geometry.path} />
    </svg>
  );
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
  activeRanges: readonly ActiveRange[],
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

      const answerAnchors = activeRanges.map((activeRange): AnswerAnchorGeometry => {
        let mapY = height / 2;
        let localY: number | undefined;
        let offscreen: "above" | "below" | undefined;
        const fragments = elementFragments(answerElements(answer, activeRange));
        const visible = answerHidden ? undefined : nearestVisibleFragment(fragments, answerViewport);
        if (visible) {
          const center = visible.top + visible.height / 2;
          mapY = clampY(center - mapRect.top, height);
          localY = clampY(center - answerRect.top, answerRect.height);
        } else if (!answerHidden && fragments.length) {
          const above = fragments.every((rect) => rect.bottom <= answerViewport.top);
          const below = fragments.every((rect) => rect.top >= answerViewport.bottom);
          if (above || below) {
            offscreen = above ? "above" : "below";
            mapY = above ? 24 : height - 24;
          } else {
            mapY = normalizedY(activeRange, answerLength, height);
          }
        } else {
          mapY = normalizedY(activeRange, answerLength, height);
        }
        const directions = new Set(reader.loci
          .filter((locus) => activeSourceIds.includes(locus.sourceId)
            && rangesIntersect(activeRange, tupleRange(locus.answerUnicodeRange)))
          .map((locus) => locus.direction));
        const direction = directions.size === 1
          ? [...directions][0]
          : directions.size > 1 ? "mixed" as const : undefined;
        return {
          key: activeRange.key,
          mapY,
          ...(localY === undefined ? {} : { localY }),
          ...(offscreen ? { offscreen } : {}),
          ...(direction ? { direction } : {}),
        };
      });

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
      setGeometry({ width, height, answerAnchors, relationships, coverage });
    };

    const schedule = () => {
      if (frame !== undefined) return;
      frame = window.requestAnimationFrame(update);
    };
    const observer = typeof ResizeObserver === "undefined" ? undefined : new ResizeObserver(schedule);
    observer?.observe(context);
    observer?.observe(answer);
    observer?.observe(map);
    // Bubbled passages can expand/reorder without changing the pane's size.
    const contentObserver = typeof MutationObserver === "undefined" ? undefined : new MutationObserver(schedule);
    contentObserver?.observe(context, { childList: true, subtree: true });
    contentObserver?.observe(answer, { childList: true, subtree: true });
    context.addEventListener("scroll", schedule, { passive: true });
    answer.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    void document.fonts?.ready.then(schedule);
    update();
    return () => {
      cancelled = true;
      if (frame !== undefined) window.cancelAnimationFrame(frame);
      observer?.disconnect();
      contentObserver?.disconnect();
      context.removeEventListener("scroll", schedule);
      answer.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [activeRanges, activeSourceIds, answerRef, contextRef, directionBySource, mapRef, reader]);
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

function AnswerLensToolbar({
  lens,
  onChange,
  onExplore,
  exploring,
}: {
  lens: AnswerLens;
  onChange: (lens: AnswerLens) => void;
  onExplore?: () => void;
  exploring: boolean;
}) {
  const descriptions: Record<AnswerLens, string> = {
    influence: "Support and suppression together per token",
    tension: "Support and suppression affecting the same answer tokens",
    suppression: "Negative context effect per token",
  };
  const options: readonly { lens: AnswerLens; label: string }[] = [
    { lens: "influence", label: "Overall" },
    { lens: "tension", label: "Tension" },
    { lens: "suppression", label: "Suppression" },
  ];
  return (
    <div className="answer-lens-toolbar">
      <span>Answer lens</span>
      <div role="group" aria-label="Answer heatmap lens">
        {options.map((option) => (
          <button
            key={option.lens}
            type="button"
            aria-pressed={lens === option.lens}
            onClick={() => onChange(option.lens)}
          >{option.label}</button>
        ))}
      </div>
      {onExplore && <button type="button" className="explore-answer" onClick={onExplore}>{exploring ? "Next passage →" : "Explore strongest →"}</button>}
      <p aria-live="polite">{descriptions[lens]}</p>
    </div>
  );
}

function SequenceOverview({
  reader,
  activeRanges,
  directions,
  answerLens,
  answerEntries,
  answerScale,
  onSelectContext,
  onSelectAnswer,
}: {
  reader: ContextInvestigationReader;
  activeRanges: readonly ActiveRange[];
  directions: ReadonlySet<EvidenceDirection>;
  answerLens: AnswerLens;
  answerEntries: readonly AnswerPassage[];
  answerScale: AnswerHeatScale;
  onSelectContext: (sourceId: string) => void;
  onSelectAnswer: (range: ActiveRange) => void;
}) {
  const answerLensLabel = answerLens === "influence" ? "overall" : answerLens;
  const contextLength = reader.context.blocks.length
    ? reader.context.blocks[reader.context.blocks.length - 1].displayUnicodeRange[1]
    : 0;
  const answerLength = reader.answer.status === "available" ? codePointLength(reader.answer.text) : 0;
  const sourceCeiling = robustCeiling(reader.sources.map(sourceHeat));
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
      className={`inspect-overview${activeRanges.length ? " is-active" : ""}${directionClasses ? ` ${directionClasses}` : ""}`}
      role="group"
      aria-label={`Context coverage: ${plural(counts.measured, "measured source")}, ${plural(counts.not_measured, "not measured source")}, ${plural(counts.unavailable, "unavailable source")}, ${plural(counts.failed, "failed source")}. ${plural(answerEntries.length, "answer phrase")}. Answer lens: ${answerLensLabel}.`}
    >
      <div className="overview-track overview-context">
        <span className="overview-label">Context · click to explore</span>
        <div className="overview-rail">
          {reader.sources.map((source) => (
            <button
              key={source.sourceId}
              type="button"
              aria-label={`Explore context: ${excerpt(sourceRecordedText(reader, source), 100)}`}
              onClick={() => onSelectContext(source.sourceId)}
              disabled={!strongestLocusForSource(reader, source.sourceId)}
              className={`overview-range state-${source.measurement.status.replace("_", "-")}`}
              data-direction={source.summary.totalContinuationDeltaNats === null
                ? undefined
                : heatDirection(source.summary.totalContinuationDeltaNats)}
              data-heat={source.measurement.status === "measured" ? "true" : undefined}
              style={{
                ...styleFor(tupleRange(source.displayUnicodeRange), contextLength),
                ...(source.measurement.status === "measured"
                  ? heatStyle(sourceHeat(source) / Math.max(sourceCeiling, 1e-9))
                  : {}),
              }}
            />
          ))}
        </div>
      </div>
      <div className="overview-registration" aria-hidden="true"><i /><i /><i /></div>
      <div className="overview-track overview-answer" data-lens={answerLens}>
        <span className="overview-label">
          Answer · click to explore
          <span className="overview-scale" title="Height and brightness show effect per token within this answer and lens. The scale runs from zero to the highest passage score, with a minimum upper endpoint of 0.10.">
            0–{answerScale.ceiling.toFixed(3)} nats/token
          </span>
        </span>
        <div className="overview-rail">
          {answerEntries.map(({ key, range, heat }) => (
            <button
              key={key}
              type="button"
              aria-label={`Explore ${answerLensLabel}: ${excerpt(range.text, 100)}`}
              title={range.annotationIds.length ? `${heat.intensity.toFixed(3)} nats/token · ${range.text.trim()}` : undefined}
              onClick={() => onSelectAnswer({ ...range, key })}
              disabled={!range.annotationIds.length}
              className={`overview-range state-measured${activeRanges.some((activeRange) => rangesIntersect(activeRange, range)) ? " is-selected" : ""}`}
              data-direction={heat.total > 0 ? heat.direction : undefined}
              data-heat={heat.total > 0 ? "true" : undefined}
              data-lens={answerLens}
              style={{
                ...styleFor(range, answerLength),
                ...(heat.total > 0 ? answerHeatStyle(heat, answerScale) : {}),
              }}
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

function InfluenceFocusPanel({ model }: { model: InfluenceFocusModel }) {
  const support = model.items
    .filter((item) => item.direction === "support")
    .sort((left, right) => right.delta - left.delta);
  const suppression = model.items
    .filter((item) => item.direction === "suppression")
    .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta));
  const groups = [
    { direction: "support" as const, title: "Top positive influence", items: support },
    { direction: "suppression" as const, title: "Top negative influence", items: suppression },
  ].sort((left, right) => {
    if (!model.primaryDirection) return 0;
    if (left.direction === model.primaryDirection) return -1;
    if (right.direction === model.primaryDirection) return 1;
    return 0;
  });
  return (
    <aside className="influence-focus-panel" aria-label={model.title}>
      <div className="influence-focus-groups">
        {groups.map((group) => (
            <section className={`influence-focus-group direction-${group.direction}`} key={group.direction}>
            <header>
              <h4>{group.title}</h4>
            </header>
            {group.items.length ? (
              <ol>
                {group.items.slice(0, 1).map((item) => (
                  <li key={item.key}>
                    <button type="button" onClick={item.onSelect} aria-label={`Focus ${group.direction} span: ${excerpt(item.text, 90)}`}
                    >
                      <span className="influence-focus-copy">{excerpt(item.text, 320)}</span>
                      <b>{item.delta > 0 ? "+" : "−"}{Math.abs(item.delta).toFixed(2)} Δ</b>
                    </button>
                  </li>
                ))}
              </ol>
            ) : <p>No above-floor {group.direction} was recorded.</p>}
          </section>
        ))}
      </div>
    </aside>
  );
}

function ContextPane({
  reader,
  paneRef,
  activeSourceIds,
  focusedSourceId,
  directionBySource,
  focusModel,
  pinned,
  selectionActive,
  linkedSelection,
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
  focusModel?: InfluenceFocusModel;
  pinned: boolean;
  selectionActive: boolean;
  linkedSelection: boolean;
  expanded: boolean;
  maxInfluence: number;
  onFocusSource: (sourceId: string) => void;
  onPreviewSource: (sourceId?: string) => void;
  onToggleExpanded: () => void;
}) {
  const sourceById = useMemo(() => new Map(reader.sources.map((source) => [source.sourceId, source])), [reader.sources]);
  const linkedSourceIds = useMemo(
    () => new Set(reader.loci.map((locus) => locus.sourceId)),
    [reader.loci],
  );
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
    <section className={`inspect-pane context-pane${selectionActive ? " has-linked-selection" : ""}`} ref={paneRef} aria-labelledby="inspect-context-heading">
      <header className="inspect-pane-heading">
        <h2 id="inspect-context-heading">Context</h2>
        <div className="pane-heading-actions">
          <button
            className="pane-expand"
            type="button"
            aria-expanded={expanded}
            onClick={onToggleExpanded}
          >{expanded ? "Collapse" : "Expand heatmap"}</button>
        </div>
      </header>
      {focusModel && <InfluenceFocusPanel model={focusModel} />}
      <div className="inspect-prose context-prose">
        {reader.context.blocks.map((block) => {
          const blockStart = block.displayUnicodeRange[0];
          const blockSources = sourcesByMessage.get(block.messageIndex) ?? [];
          const wholeSources = blockSources.filter((source) => source.granularity === "whole_segment");
          const wholeIds = wholeSources.map((source) => source.sourceId);
          const relatedWhole = wholeIds.filter((id) => activeSourceIds.has(id));
          const strongestWhole = strongestSource(
            wholeIds.filter((sourceId) => linkedSourceIds.has(sourceId)),
            sourceById,
          );
          const focusableWhole = strongestWhole?.sourceId;
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
          const focusWholeOnSurface = Boolean(focusableWhole) && !segments.some((segment) => (
            segment.annotationIds.some((sourceId) => linkedSourceIds.has(sourceId))
          ));
          return (
            <p
              key={`${block.messageIndex}:${block.segmentId}`}
              className={blockClasses.join(" ")}
              data-message-index={block.messageIndex}
              data-source-ids={wholeIds.length ? JSON.stringify(wholeIds) : undefined}
              data-direction={strongestWhole && strongestWhole.summary.totalContinuationDeltaNats !== null
                ? heatDirection(strongestWhole.summary.totalContinuationDeltaNats)
                : wholeDirection}
              data-link-direction={relatedWhole.length ? wholeDirection : undefined}
              data-heat={strongestWhole ? "true" : undefined}
              style={strongestWhole
                ? heatStyle(sourceHeat(strongestWhole) / Math.max(maxInfluence, 1e-9))
                : undefined}
              role={focusWholeOnSurface ? "button" : undefined}
              tabIndex={focusWholeOnSurface ? 0 : undefined}
              aria-pressed={focusWholeOnSurface ? Boolean(pinned && focusedSourceId === focusableWhole) : undefined}
              aria-label={focusWholeOnSurface ? `Pin context passage: “${excerpt(block.text)}”` : undefined}
              onMouseEnter={focusableWhole ? () => onPreviewSource(focusableWhole) : undefined}
              onMouseLeave={focusableWhole ? () => onPreviewSource(undefined) : undefined}
              onFocus={focusWholeOnSurface ? () => onPreviewSource(focusableWhole) : undefined}
              onBlur={focusWholeOnSurface ? () => onPreviewSource(undefined) : undefined}
              onClick={focusWholeOnSurface ? () => onFocusSource(focusableWhole!) : undefined}
              onKeyDown={focusWholeOnSurface ? (event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onFocusSource(focusableWhole!);
                }
              } : undefined}
            >
              {focusableWhole && (
                focusWholeOnSurface
                  ? <span className="context-block-focus" aria-hidden="true"><span /></span>
                  : <button
                    className="context-block-focus"
                    type="button"
                    aria-label={`Focus ${wholeDirection === "mixed" ? "mixed measured" : wholeDirection ?? "measured"} relationship from “${excerpt(block.text)}”`}
                    onClick={() => onFocusSource(focusableWhole)}
                  ><span aria-hidden="true" /></button>
              )}
              {segments.map((segment) => {
                const related = segment.annotationIds.filter((id) => activeSourceIds.has(id));
                const strongest = strongestSource(
                  segment.annotationIds.filter((sourceId) => linkedSourceIds.has(sourceId)),
                  sourceById,
                );
                const focusableSource = strongest?.sourceId;
                const directions = new Set(related.map((id) => directionBySource.get(id)).filter(Boolean));
                const direction = directions.size === 1 ? [...directions][0] : directions.size > 1 ? "mixed" : undefined;
                const heatValue = strongest?.summary.totalContinuationDeltaNats;
                return (
                  <span
                    key={`${block.messageIndex}:${segment.start}:${segment.end}`}
                    className={contextSegmentClass(segment, sourceById, activeSourceIds, focusedSourceId)}
                    data-source-ids={segment.annotationIds.length ? JSON.stringify(segment.annotationIds) : undefined}
                    data-direction={heatValue !== null && heatValue !== undefined
                      ? heatDirection(heatValue)
                      : direction}
                    data-link-direction={related.length ? direction : undefined}
                    data-heat={strongest ? "true" : undefined}
                    style={strongest
                      ? heatStyle(sourceHeat(strongest) / Math.max(maxInfluence, 1e-9))
                      : undefined}
                    role={focusableSource ? "button" : undefined}
                    tabIndex={focusableSource ? 0 : undefined}
                    aria-label={focusableSource ? `Focus “${excerpt(segment.text)}”. ${sourceStatusLabel(sourceById.get(focusableSource)?.measurement.status ?? "not_measured")}.` : undefined}
                    onMouseEnter={focusableSource ? () => onPreviewSource(focusableSource) : undefined}
                    onMouseLeave={focusableSource ? () => onPreviewSource(undefined) : undefined}
                    onFocus={focusableSource ? () => onPreviewSource(focusableSource) : undefined}
                    onBlur={focusableSource ? () => onPreviewSource(undefined) : undefined}
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
      {linkedSelection && [...activeSourceIds].map((sourceId) => (
        <ContinuousSpanOutline
          key={`linked:${sourceId}`}
          paneRef={paneRef}
          sourceId={sourceId}
          linkedDirection={directionBySource.get(sourceId) ?? "mixed"}
        />
      ))}
      <ContinuousSpanOutline paneRef={paneRef} sourceId={focusedSourceId} pinned={pinned} />
    </section>
  );
}

function AnswerPane({
  reader,
  paneRef,
  passages,
  onExplore,
  activeRange,
  lockedRange,
  linkedRanges,
  linkedDirectionsByRange,
  lockedRelationshipCount,
  focusModel,
  pinned,
  expanded,
  lens,
  heatCeilings,
  onPreview,
  onLock,
  onClear,
  onToggleExpanded,
}: {
  reader: ContextInvestigationReader;
  paneRef: RefObject<HTMLElement | null>;
  passages: readonly AnswerPassage[];
  onExplore: (range: ActiveRange) => void;
  activeRange?: ActiveRange;
  lockedRange?: ActiveRange;
  linkedRanges?: readonly ActiveRange[];
  linkedDirectionsByRange?: ReadonlyMap<string, EvidenceDirection>;
  lockedRelationshipCount?: number;
  focusModel?: InfluenceFocusModel;
  pinned: boolean;
  expanded: boolean;
  lens: AnswerLens;
  heatCeilings: AnswerHeatScale;
  onPreview: (range?: ActiveRange) => void;
  onLock: (range: ActiveRange) => void;
  onClear: () => void;
  onToggleExpanded: () => void;
}) {
  const lensLabel = lens === "influence" ? "overall effect" : lens;
  const linkedContours = useMemo(
    () => mergeLinkedRangeContours(linkedRanges ?? [], linkedDirectionsByRange ?? new Map()),
    [linkedDirectionsByRange, linkedRanges],
  );
  const interactive = useMemo(() => passages.map(item => item.range).filter(range => range.annotationIds.length > 0), [passages]);
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
  const peakPassage = heatCeilings.standout ? passages.find(item => item.heat.intensity === heatCeilings.peak) : undefined;
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
    <section className={`inspect-pane answer-pane${pinned || linkedRanges?.length ? " has-linked-selection" : ""}`} ref={paneRef} aria-labelledby="inspect-answer-heading">
      <header className="inspect-pane-heading">
        <h2 id="inspect-answer-heading">Answer</h2>
        <div className="pane-heading-actions">
          {reader.answer.status === "available" && <span>{plural(reader.answer.tokens.length, "token")}</span>}
          {reader.answer.status === "text_only" && <span>Recorded response</span>}
          {reader.answer.status === "available" && (
            <button
              className="pane-expand"
              type="button"
              aria-expanded={expanded}
              onClick={onToggleExpanded}
            >{expanded ? "Collapse" : "Expand heatmap"}</button>
          )}
        </div>
      </header>
      {peakPassage && !focusModel && (
        <button className="passage-spotlight" type="button"
          onClick={() => onExplore({ ...peakPassage.range, key: peakPassage.key })}
          title="This peak is at least 1.5 times the median measured passage and reaches 0.10 nats/token. Click to inspect the context.">
          <span><b>Highest {lensLabel}</b><span>{heatCeilings.peak.toFixed(3)} nats/token
            {heatCeilings.median > 0 && ` · ${(heatCeilings.peak / heatCeilings.median).toFixed(1)}× median`}
            {heatCeilings.peakCount > 1 && ` · ${heatCeilings.peakCount} tied`}
          </span></span>
          <strong>{peakPassage.range.text.trim()} <i aria-hidden="true">↗</i></strong>
        </button>
      )}
      {!peakPassage && !focusModel && reader.measurement.completedSourceCount > 0 && passages.length > 1 && (
        <p className="passage-scale-note">{heatCeilings.peak === 0
          ? `No measured ${lensLabel} in this answer.`
          : heatCeilings.peak < .1
            ? `Peak ${lensLabel} is only ${heatCeilings.peak.toPrecision(2)} nats/token.`
            : `No distinct ${lensLabel} peak. The highest passage is less than 1.5× the median.`}</p>
      )}
      {focusModel && <InfluenceFocusPanel model={focusModel} />}
      {reader.answer.status === "available" ? (
        <div className="inspect-prose answer-prose">
          <p>
            {passages.map(({ range: segment, heat }) => {
              const measured = segment.annotationIds.length > 0;
              const range = { start: segment.start, end: segment.end, key: rangeKey(segment) };
              const sourceLinked = Boolean(linkedRanges?.some((linkedRange) => rangesIntersect(linkedRange, range)));
              const linkDirection = sourceLinked ? linkedDirectionsByRange?.get(range.key) : undefined;
              const active = sourceLinked || Boolean(activeRange && rangesIntersect(activeRange, range));
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
              return (
                <span
                  key={range.key}
                  ref={measured ? (node) => {
                    if (node) nodes.current.set(range.key, node);
                    else nodes.current.delete(range.key);
                  } : undefined}
                  className={`answer-segment${measured ? " has-measurement" : ""}${active ? " is-active" : ""}${sourceLinked ? " is-source-linked" : ""}${locked ? " is-locked" : ""}`}
                  data-answer-start={segment.start}
                  data-answer-end={segment.end}
                  data-direction={heat.total ? heat.direction : undefined}
                  data-link-direction={linkDirection}
                  data-heat={heat.total ? "true" : undefined}
                  data-lens={lens}
                  data-peak={heatCeilings.standout && heat.intensity === heatCeilings.peak ? "true" : undefined}
                  style={heat.total
                    ? answerHeatStyle(heat, heatCeilings)
                    : undefined}
                  role={measured ? "button" : undefined}
                  tabIndex={measured ? (range.key === effectiveRovingKey ? 0 : -1) : undefined}
                  aria-pressed={measured ? Boolean(locked) : undefined}
                  aria-label={measured ? `${segment.text}; ${plural(announcedCount, relationshipNoun)}` : undefined}
                  onMouseEnter={measured ? () => onPreview(range) : undefined}
                  onMouseLeave={measured ? () => onPreview(undefined) : undefined}
                  onFocus={measured ? () => {
                    setRovingKey(range.key);
                    onPreview(range);
                  } : undefined}
                  onBlur={measured ? () => onPreview(undefined) : undefined}
                  onClick={measured ? () => onLock(range) : undefined}
                  onKeyDown={measured ? (event) => handleKey(event, segment, index) : undefined}
                >{segment.text}</span>
              );
            })}
          </p>
        </div>
      ) : reader.answer.status === "text_only" ? (
        <div className="inspect-prose answer-prose answer-text-only">
          <p>{reader.answer.text}</p>
          <aside>
            <strong>Response recorded; heatmap not ready yet.</strong>
            <span>This older turn has no saved token trace. Measure Context Effects to reconstruct token-aligned evidence.</span>
          </aside>
        </div>
      ) : <p className="pane-unavailable">No recorded response is available for this turn.</p>}
      {linkedContours.map(({ range, direction }) => (
        <ContinuousSpanOutline
          key={`linked:${range.key}`}
          paneRef={paneRef}
          answerRange={range}
          linkedDirection={direction}
        />
      ))}
      <ContinuousSpanOutline paneRef={paneRef} answerRange={activeRange} pinned={pinned} />
    </section>
  );
}

function relationshipPath(
  sourceX: number,
  sourceY: number,
  answerX: number,
  answerY: number,
  sourceOffscreen?: "above" | "below",
  answerOffscreen?: "above" | "below",
): string {
  const distance = Math.abs(answerX - sourceX);
  const sourceControl = sourceOffscreen
    ? [sourceX, sourceY + (sourceOffscreen === "above" ? 32 : -32)]
    : [sourceX + distance * .44, sourceY];
  const answerControl = answerOffscreen
    ? [answerX, answerY + (answerOffscreen === "above" ? 32 : -32)]
    : [answerX - distance * .44, answerY];
  return `M ${sourceX} ${sourceY} C ${sourceControl[0]} ${sourceControl[1]}, ${answerControl[0]} ${answerControl[1]}, ${answerX} ${answerY}`;
}

function offscreenTipPath(x: number, y: number, direction: "above" | "below"): string {
  const offset = direction === "above" ? 5 : -5;
  return `M ${x - 4} ${y + offset} L ${x} ${y} L ${x + 4} ${y + offset}`;
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
  onClear,
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
  onAnswerOffscreen: (direction: "above" | "below") => void;
  onClear: () => void;
}) {
  const axisX = geometry.width / 2;
  const gateX = geometry.width - 14;
  const registeredRelationships = geometry.relationships.filter((relationship) => relationship.offscreen === undefined);
  const mappedRelationships = geometry.relationships.filter((relationship) => visibleSourceIds.has(relationship.sourceId));
  const relationshipIndex = new Map(mappedRelationships.map((relationship, index) => [relationship.sourceId, index]));
  const visibleThreads = mappedRelationships.flatMap((relationship) => (
    (relationship.localY !== undefined || relationship.offscreen !== undefined)
      ? geometry.answerAnchors
        .filter((anchor) => anchor.localY !== undefined || anchor.offscreen !== undefined)
        .map((anchor) => ({
          relationship,
          anchor,
          direction: geometry.relationships.length === 1 && geometry.answerAnchors.length > 1
            ? anchor.direction ?? relationship.direction
            : relationship.direction,
        }))
      : []
  ));
  const aboveCount = geometry.relationships.filter((relationship) => relationship.offscreen === "above").length;
  const belowCount = geometry.relationships.filter((relationship) => relationship.offscreen === "below").length;
  const answerAboveCount = geometry.answerAnchors.filter((anchor) => anchor.offscreen === "above").length;
  const answerBelowCount = geometry.answerAnchors.filter((anchor) => anchor.offscreen === "below").length;
  const hiddenCount = focusedSourceId ? 0 : Math.max(0, registeredRelationships.length - mappedRelationships.length);
  const directionClasses = [
    geometry.relationships.some((relationship) => relationship.direction === "support") ? "has-support" : "",
    geometry.relationships.some((relationship) => relationship.direction === "suppression") ? "has-suppression" : "",
    geometry.relationships.some((relationship) => relationship.direction === "mixed") ? "has-mixed" : "",
  ].filter(Boolean).join(" ");
  return (
    <div
      className={`evidence-map${geometry.answerAnchors.length ? " is-active" : ""}${locked ? " is-locked" : geometry.answerAnchors.length ? " is-preview" : ""}${directionClasses ? ` ${directionClasses}` : ""}`}
      ref={mapRef}
      role="group"
      aria-label={`Influence relationships${locked ? ", pinned" : activeRange ? ", preview" : ""}`}
    >
      <svg className="evidence-map-svg" viewBox={`0 0 ${geometry.width} ${geometry.height}`} preserveAspectRatio="none" aria-hidden="true">
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
        {visibleThreads.map(({ relationship, anchor, direction }) => {
          const sourceX = axisX + ((relationshipIndex.get(relationship.sourceId) ?? 0) % 3) * 4;
          const path = relationshipPath(
            sourceX,
            relationship.mapY,
            gateX,
            anchor.mapY,
            relationship.offscreen,
            anchor.offscreen,
          );
          const offscreenClasses = [
            relationship.offscreen ? `is-source-${relationship.offscreen}` : "",
            anchor.offscreen ? `is-answer-${anchor.offscreen}` : "",
          ].filter(Boolean).join(" ");
          return (
            <g key={`${relationship.sourceId}:${anchor.key}`} className={`relationship-path direction-${direction}${offscreenClasses ? ` ${offscreenClasses}` : ""}`}>
              <path className="relationship-glow" d={path} />
              <path className="relationship-signal" d={path} />
              {relationship.offscreen && (
                <path className="relationship-scroll-tip" d={offscreenTipPath(sourceX, relationship.mapY, relationship.offscreen)} />
              )}
              {anchor.offscreen && (
                <path className="relationship-scroll-tip" d={offscreenTipPath(gateX, anchor.mapY, anchor.offscreen)} />
              )}
            </g>
          );
        })}
        {geometry.answerAnchors.map((anchor) => (
          <line key={anchor.key} className="answer-gate" x1={gateX - 5} x2={gateX + 1} y1={anchor.mapY} y2={anchor.mapY} />
        ))}
      </svg>
      {locked && (
        <button className="map-pin-state" type="button" onClick={onClear} aria-label="Clear pinned selection">
          <span>Pinned</span><b aria-hidden="true">×</b>
        </button>
      )}
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
      {answerAboveCount > 0 && locked && (
        <button
          className="answer-offscreen-mark is-above"
          type="button"
          onClick={() => onAnswerOffscreen("above")}
          aria-label={`${plural(answerAboveCount, "linked answer passage")} above`}
        >
          <span aria-hidden="true">↑</span>{answerAboveCount > 1 ? answerAboveCount : null}
        </button>
      )}
      {answerBelowCount > 0 && locked && (
        <button
          className="answer-offscreen-mark is-below"
          type="button"
          onClick={() => onAnswerOffscreen("below")}
          aria-label={`${plural(answerBelowCount, "linked answer passage")} below`}
        >
          <span aria-hidden="true">↓</span>{answerBelowCount > 1 ? answerBelowCount : null}
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
  relationshipCount,
  focusedSourceId,
  onPrevious,
  onNext,
  onClear,
}: {
  reader: ContextInvestigationReader;
  range: ActiveRange;
  query: OptionalResource<ContextInvestigationQuery>;
  sourceIds: readonly string[];
  relationshipCount?: number;
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
            <span className="eyebrow">Net effect over selection</span>
            <div className="effect-reading">
              {focusedEffect.deltaNats !== null && (
                <strong>{focusedEffect.deltaNats > 0 ? "+" : ""}{focusedEffect.deltaNats.toFixed(2)} <small>Δ nats</small></strong>
              )}
              <span className={`evidence-state direction-${focusedEffect.direction}`}>{directionLabel(focusedEffect.direction)}</span>
            </div>
            <div className="effect-basis">Delete source · recorded continuation · floor {ready?.measurementFloorNats.toFixed(2)}</div>
            <div className="effect-basis">Positive and negative token effects can cancel in this total.</div>
            <QueryStateLine resource={query} />
          </>
        ) : (
          <>
            <span className="eyebrow">Net selection relationships</span>
            <div className="effect-counts">
              <strong>{plural(relationshipCount ?? sourceIds.length, "above-floor relationship")}</strong>
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
  answerLens,
  expanded = false,
}: {
  reader: ContextInvestigationReader;
  pane: MobilePane;
  sourceId?: string;
  range?: ActiveRange;
  answerLens: AnswerLens;
  expanded?: boolean;
}) {
  const sourceById = useMemo(
    () => new Map(reader.sources.map((source) => [source.sourceId, source])),
    [reader.sources],
  );
  const links = useMemo(() => {
    let loci: ContextInvestigationLocus[] = [];
    if (pane === "context" && sourceId) {
      loci = reader.loci.filter((locus) => locus.sourceId === sourceId);
    } else if (pane === "answer" && range) {
      loci = reader.loci.filter((locus) => rangesIntersect(range, tupleRange(locus.answerUnicodeRange)));
    }
    if (reader.answer.status !== "available") return [];
    const answer = reader.answer;
    const ranges = pane === "answer" && range ? [range] : semanticTextRanges(answer.text);
    return ranges.flatMap((textRange) => [...signedContributions(loci, answer.tokens, textRange)].flatMap(([id, values]) => {
      const positive = [...values.values()].reduce((sum, value) => sum + Math.max(0, value), 0);
      const negative = [...values.values()].reduce((sum, value) => sum + Math.min(0, value), 0);
      return [positive, negative].filter((delta) => delta !== 0).map((delta) => ({
        key: `${id}:${rangeKey(textRange)}:${Math.sign(delta)}`,
        sourceId: id,
        text: pane === "context" ? sliceCodePoints(answer.text, textRange.start, textRange.end)
          : sourceById.has(id) ? sourceRecordedText(reader, sourceById.get(id)!) : id,
        delta,
      }));
    })).sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta));
  }, [pane, range, reader, sourceById, sourceId]);
  const source = sourceId ? sourceById.get(sourceId) : undefined;
  const support = links.reduce((total, item) => item.delta > 0 ? total + item.delta : total, 0);
  const suppression = links.reduce((total, item) => item.delta < 0 ? total + Math.abs(item.delta) : total, 0);
  const linkedSourceCount = new Set(links.map((item) => item.sourceId)).size;
  const strongestSupport = links.find((item) => item.delta > 0);
  const strongestSuppression = links.find((item) => item.delta < 0);
  const opposing = [strongestSupport, strongestSuppression].filter((item) => item !== undefined);
  const previewLinks = answerLens === "suppression" ? links.filter((item) => item.delta < 0)
    : answerLens === "tension" ? [...opposing, ...links.filter((item) => !opposing.includes(item))] : links;
  const lensLabel = answerLens === "influence"
    ? "overall effect"
    : answerLens === "tension" ? "context tension" : "suppression";
  const key = (
    <header className="heatmap-key">
      <span className="eyebrow">Heatmap</span>
      {answerLens === "influence" && <span className="heatmap-key-item direction-support"><i />Support</span>}
      {answerLens !== "tension" && <span className="heatmap-key-item direction-suppression"><i />Suppression</span>}
      {answerLens === "tension" && <span className="heatmap-key-item direction-tension"><i />Opposing pull</span>}
      <span className="heatmap-key-intensity"><i />Lower <b /> Higher {lensLabel} / token</span>
    </header>
  );
  if (!source && !range) {
    const explanation = answerLens === "influence"
      ? "Brightness compares effect per token within this answer. Hover to preview; click to bring the related passages into view."
      : answerLens === "tension"
        ? "Brighter phrases have stronger, balanced opposing effects on the same tokens. This is context tension, not a confidence score."
        : "Brighter phrases were pushed against more strongly by parts of the context. This shows negative pull, not model uncertainty.";
    return (
      <aside className="heatmap-peek is-empty" aria-live="polite">
        {key}
        <p>{explanation}</p>
      </aside>
    );
  }
  return (
    <aside className="heatmap-peek" aria-live="polite">
      {key}
      <div>
        <span className="eyebrow">{pane === "context" ? "Context span" : "Answer span"}</span>
        <p>{source
          ? excerpt(source.label?.trim() || sourceRecordedText(reader, source), 180)
          : reader.answer.status === "available" && range
            ? excerpt(sliceCodePoints(reader.answer.text, range.start, range.end), 180)
            : "Selection unavailable"}</p>
        <div className="heatmap-totals" aria-label={`${support.toFixed(2)} support delta nats, ${suppression.toFixed(2)} suppression delta nats, ${linkedSourceCount} linked sources`}>
          <span className="direction-support">+{support.toFixed(2)} support</span>
          <span className="direction-suppression">−{suppression.toFixed(2)} suppression</span>
          <span>{plural(linkedSourceCount, "linked source")}</span>
        </div>
      </div>
      <div>
        <span className="eyebrow">Corresponding text</span>
        {previewLinks.length ? (
          <ul>
            {previewLinks.slice(0, 3).map(({ key, text, delta }) => {
              const direction = delta > 0 ? "support" : "suppression";
              return (
                <li key={key}>
                  <span className={`evidence-state direction-${direction}`}>{directionLabel(direction)}</span>
                  <strong>{delta > 0 ? "+" : ""}{delta.toFixed(2)} Δ</strong>
                  <span>{excerpt(text, expanded ? 240 : 110)}</span>
                </li>
              );
            })}
          </ul>
        ) : <p>{answerLens === "suppression" ? "No above-floor suppression was recorded for this text." : "No above-floor link was recorded for this text."}</p>}
      </div>
    </aside>
  );
}

export function InspectReader({
  reader,
  route,
  querySelection = contextInvestigationApi.query,
  initialAnswerLens = "tension",
}: {
  reader: ContextInvestigationReader;
  route: InspectRoute;
  querySelection?: typeof contextInvestigationApi.query;
  initialAnswerLens?: AnswerLens;
}) {
  const [hoveredRange, setHoveredRange] = useState<ActiveRange | undefined>();
  const [hoveredSourceId, setHoveredSourceId] = useState<string | undefined>();
  const [expandedPane, setExpandedPane] = useState<MobilePane | undefined>();
  const [mobilePane, setMobilePane] = useState<MobilePane>("answer");
  const [answerLens, setAnswerLens] = useState<AnswerLens>(initialAnswerLens);
  const [selectionOrigin, setSelectionOrigin] = useState<MobilePane>(() => route.sourceId ? "context" : "answer");
  const contextRef = useRef<HTMLElement>(null);
  const answerRef = useRef<HTMLElement>(null);
  const mapRef = useRef<HTMLDivElement>(null);
  const readingWorkspaceRef = useRef<HTMLDivElement>(null);
  const hoverClearTimer = useRef<number | undefined>(undefined);
  const cancelHoverClear = useCallback(() => {
    if (hoverClearTimer.current !== undefined) {
      window.clearTimeout(hoverClearTimer.current);
      hoverClearTimer.current = undefined;
    }
  }, []);
  useEffect(() => () => cancelHoverClear(), [cancelHoverClear]);
  useLayoutEffect(() => {
    if (!expandedPane) return;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    readingWorkspaceRef.current?.querySelector<HTMLElement>('.pane-expand[aria-expanded="true"]')?.focus({ preventScroll: true });
    return () => {
      document.body.style.overflow = previousOverflow;
      if (previousFocus?.isConnected) previousFocus.focus({ preventScroll: true });
    };
  }, [expandedPane]);
  useEffect(() => setSelectionOrigin(route.sourceId ? "context" : "answer"), [route.runId]);
  const answer = reader.answer.status === "available" ? reader.answer : undefined;
  const answerPhraseRanges = useMemo(
    () => answer ? semanticTextRanges(answer.text) : [],
    [answer],
  );
  const lockedRange = useMemo<ActiveRange | undefined>(() => {
    if (!answer || route.answerStart === undefined || route.answerEnd === undefined) return undefined;
    const range = { start: route.answerStart, end: route.answerEnd };
    return isValidCodePointRange(answer.text, range, { allowEmpty: false })
      ? { ...range, key: rangeKey(range) }
      : undefined;
  }, [answer, route.answerEnd, route.answerStart]);
  const contextPinnedSourceId = selectionOrigin === "context"
    && route.sourceId
    && reader.sources.some((source) => source.sourceId === route.sourceId)
    ? route.sourceId
    : undefined;
  const previewing = !lockedRange && Boolean(hoveredRange || hoveredSourceId);
  const activeRange = lockedRange ?? hoveredRange;
  const maxSourceInfluence = useMemo(
    () => robustCeiling(reader.sources.map(sourceHeat)),
    [reader.sources],
  );

  const passageEvidence = useMemo<AnswerPassage[]>(() => {
    if (!answer) return [];
    return answerPhraseRanges.map(range => {
      const loci = reader.loci.filter(locus => rangesIntersect(range, tupleRange(locus.answerUnicodeRange)));
      return {
        key: rangeKey(range),
        range: { ...range, text: sliceCodePoints(answer.text, range.start, range.end), annotationIds: loci.map(locus => locus.locusId) },
        heat: answerHeatForRange(loci, answer.tokens, range),
      };
    });
  }, [answer, answerPhraseRanges, reader.loci]);
  const answerPassages = useMemo(() => passageEvidence.map(item => ({
    ...item, heat: answerHeatForLens(item.heat, answerLens),
  })), [passageEvidence, answerLens]);
  const activeAnswerRanges = useMemo<readonly ActiveRange[]>(() => {
    if (!contextPinnedSourceId) return activeRange ? [activeRange] : [];
    const sourceLoci = strongestRelationshipsPerDirection(
      reader.loci.filter((locus) => locus.sourceId === contextPinnedSourceId),
    );
    const linked = answerPassages
      .filter(({ range }) => sourceLoci.some((locus) => rangesIntersect(range, tupleRange(locus.answerUnicodeRange))))
      .map(({ range, key }) => ({ ...range, key }));
    return linked;
  }, [activeRange, answerPassages, contextPinnedSourceId, reader.loci]);
  const answerHeatCeilings = useMemo(() => answerDisplayScale(
    // Each scored arm covers the continuation. Include quiet passages in the
    // median even when none of their token deltas cleared the reader's floor.
    reader.measurement.completedSourceCount ? answerPassages.filter(item => item.heat.tokenCount > 0).map(item => item.heat.intensity) : [],
  ), [answerPassages, reader.measurement.completedSourceCount]);

  const queryLoader = useMemo(() => lockedRange && reader.capabilities.contextEffects === "measurable"
    ? (signal: AbortSignal) => querySelection(route.runId, {
      answerStart: lockedRange.start,
      answerEnd: lockedRange.end,
      floorNats: reader.presentation.measurementFloorNats,
      sourceIds: reader.measurement.sourceUniverse,
      scoringMode: route.scoringMode ?? "fast",
    }, signal)
    : undefined, [lockedRange, querySelection, reader.capabilities.contextEffects, reader.measurement.sourceUniverse, reader.presentation.measurementFloorNats, route.runId, route.scoringMode]);
  const queryKey = lockedRange && reader.capabilities.contextEffects === "measurable"
    ? `${route.runId}:${route.scoringMode ?? "fast"}:${lockedRange.start}:${lockedRange.end}:${reader.presentation.measurementFloorNats}:${reader.measurement.sourceUniverse.join(",")}`
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
  const activeLoci = useMemo(() => contextPinnedSourceId
    ? reader.loci.filter((locus) => locus.sourceId === contextPinnedSourceId)
    : activeRange
      ? reader.loci.filter((locus) => rangesIntersect(activeRange, tupleRange(locus.answerUnicodeRange)))
      : [], [activeRange, contextPinnedSourceId, reader.loci]);
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
  const lockedAboveFloorEffects = useMemo(() => lockedRange && query.phase === "ready"
    ? query.value.effects.filter(isAboveFloorEffect)
    : undefined, [lockedRange, query]);
  const queryAboveFloorEffects = previewing || contextPinnedSourceId ? undefined : lockedAboveFloorEffects;
  const visualRelationships = useMemo<readonly VisualRelationship[]>(() => {
    const relationships: readonly VisualRelationship[] = queryAboveFloorEffects ?? activeLoci;
    return lockedRange ? strongestRelationshipsPerDirection(relationships) : relationships;
  }, [activeLoci, lockedRange, queryAboveFloorEffects]);
  const activeSourceIds = useMemo(() => [...new Set(
    visualRelationships.map((item) => item.sourceId),
  )].filter((sourceId) => sourceById.has(sourceId)).sort((left, right) => {
    const leftRange = sourceById.get(left)?.displayUnicodeRange ?? [0, 0];
    const rightRange = sourceById.get(right)?.displayUnicodeRange ?? [0, 0];
    return leftRange[0] - rightRange[0] || left.localeCompare(right);
  }), [sourceById, visualRelationships]);
  const focusedSourceId = lockedRange && route.sourceId && (
    activeSourceIds.includes(route.sourceId) || activeLoci.some((locus) => locus.sourceId === route.sourceId)
  )
    ? route.sourceId
    : undefined;
  const displayedFocusedSourceId = focusedSourceId ?? (lockedRange ? undefined : hoveredSourceId);
  const directionBySource = useMemo(() => {
    const directions = new Map<string, Set<"support" | "suppression">>();
    for (const relationship of visualRelationships) {
      const current = directions.get(relationship.sourceId) ?? new Set();
      current.add(relationship.direction);
      directions.set(relationship.sourceId, current);
    }
    return new Map([...directions].map(([sourceId, values]) => [
      sourceId,
      values.size === 1 ? [...values][0] : "mixed" as const,
    ]));
  }, [visualRelationships]);
  const linkedDirectionsByRange = useMemo(() => {
    const result = new Map<string, EvidenceDirection>();
    if (!contextPinnedSourceId) return result;
    for (const range of activeAnswerRanges) {
      const directions = new Set(visualRelationships
        .filter((relationship): relationship is ContextInvestigationLocus => (
          "answerUnicodeRange" in relationship
          && rangesIntersect(range, tupleRange(relationship.answerUnicodeRange))
        ))
        .map((relationship) => relationship.direction));
      if (directions.size === 1) result.set(range.key, [...directions][0]);
      else if (directions.size > 1) result.set(range.key, "mixed");
    }
    return result;
  }, [activeAnswerRanges, contextPinnedSourceId, visualRelationships]);
  const geometry = useEvidenceGeometry(
    contextRef,
    answerRef,
    mapRef,
    reader,
    activeAnswerRanges,
    activeSourceIds,
    directionBySource,
  );
  const visibleSourceIds = useMemo(() => {
    if (displayedFocusedSourceId) return new Set([displayedFocusedSourceId]);
    if (lockedRange) return new Set(activeSourceIds.slice(0, MAX_TRANSIENT_PATHS));
    const visible = geometry.relationships
      .filter((relationship) => relationship.offscreen === undefined && relationship.localY !== undefined)
      .map((relationship) => relationship.sourceId);
    const candidates = visible.length ? visible : activeSourceIds;
    return new Set(candidates.slice(0, MAX_TRANSIENT_PATHS));
  }, [activeSourceIds, displayedFocusedSourceId, geometry.relationships, lockedRange]);

  const setCoordinate = useCallback((range?: ActiveRange, sourceId?: string) => {
    window.location.hash = routeHref(range ? {
      surface: "inspect",
      runId: route.runId,
      ...(route.conversationId ? { conversationId: route.conversationId } : {}),
      ...(route.scoringMode ? { scoringMode: route.scoringMode } : {}),
      ...(route.granularity && route.armBudget !== undefined ? {
        granularity: route.granularity,
        armBudget: route.armBudget,
      } : {}),
      answerStart: range.start,
      answerEnd: range.end,
      ...(sourceId ? { sourceId } : {}),
    } : {
      surface: "inspect",
      runId: route.runId,
      ...(route.conversationId ? { conversationId: route.conversationId } : {}),
      ...(route.scoringMode ? { scoringMode: route.scoringMode } : {}),
      ...(route.granularity && route.armBudget !== undefined ? {
        granularity: route.granularity,
        armBudget: route.armBudget,
      } : {}),
    });
  }, [route.armBudget, route.conversationId, route.granularity, route.runId, route.scoringMode]);

  const clearSelection = useCallback(() => {
    cancelHoverClear();
    setHoveredSourceId(undefined);
    setHoveredRange(undefined);
    setCoordinate();
  }, [cancelHoverClear, setCoordinate]);

  useEffect(() => {
    if (!lockedRange) return undefined;
    const clearOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      clearSelection();
    };
    window.addEventListener("keydown", clearOnEscape);
    return () => window.removeEventListener("keydown", clearOnEscape);
  }, [clearSelection, lockedRange]);

  const focusSource = useCallback((sourceId: string) => {
    if (!lockedRange) return;
    const next = focusedSourceId === sourceId ? undefined : sourceId;
    setCoordinate(lockedRange, next);
    if (next) setMobilePane("context");
  }, [focusedSourceId, lockedRange, setCoordinate]);

  const previewContextSource = useCallback((sourceId?: string) => {
    cancelHoverClear();
    if (lockedRange) return;
    if (!sourceId) {
      hoverClearTimer.current = window.setTimeout(() => {
        hoverClearTimer.current = undefined;
        setHoveredSourceId(undefined);
        setHoveredRange(undefined);
      }, 90);
      return;
    }
    setHoveredSourceId(sourceId);
    const locus = strongestLocusForSource(reader, sourceId);
    setHoveredRange(locus
      ? semanticRangeFor(answerPhraseRanges, tupleRange(locus.answerUnicodeRange))
      : undefined);
  }, [answerPhraseRanges, cancelHoverClear, lockedRange, reader]);

  const previewAnswerRange = useCallback((range?: ActiveRange) => {
    cancelHoverClear();
    if (lockedRange) return;
    if (!range) {
      hoverClearTimer.current = window.setTimeout(() => {
        hoverClearTimer.current = undefined;
        setHoveredRange(undefined);
      }, 90);
      return;
    }
    setHoveredSourceId(undefined);
    setHoveredRange(range);
  }, [cancelHoverClear, lockedRange]);

  const lockContextSource = useCallback((sourceId: string) => {
    if (selectionOrigin === "context" && focusedSourceId === sourceId) {
      clearSelection();
      return;
    }
    const locus = strongestLocusForSource(reader, sourceId);
    if (!locus) return;
    cancelHoverClear();
    setSelectionOrigin("context");
    setHoveredSourceId(undefined);
    setHoveredRange(undefined);
    setCoordinate(
      semanticRangeFor(answerPhraseRanges, tupleRange(locus.answerUnicodeRange)),
      sourceId,
    );
  }, [answerPhraseRanges, cancelHoverClear, clearSelection, focusedSourceId, reader, selectionOrigin, setCoordinate]);

  useEffect(() => {
    if (!focusedSourceId || !contextRef.current || selectionOrigin === "answer") return;
    setMobilePane("context");
    const viewport = contextRef.current;
    const target = sourceElements(viewport, focusedSourceId)[0];
    if (!target) return;
    const targetRect = target.getBoundingClientRect();
    const viewportRect = viewport.getBoundingClientRect();
    const top = viewport.scrollTop + targetRect.top - viewportRect.top - viewport.clientHeight / 2 + targetRect.height / 2;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    viewport.scrollTo?.({ top: Math.max(0, top), behavior: reduced ? "auto" : "smooth" });
  }, [focusedSourceId, selectionOrigin]);

  useEffect(() => {
    if (!lockedRange) return;
    const viewport = selectionOrigin === "answer" ? contextRef.current : answerRef.current;
    if (!viewport) return;
    window.requestAnimationFrame(() => {
      const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      viewport.scrollTo?.({ top: 0, behavior: reduced ? "auto" : "smooth" });
    });
  }, [lockedRange?.key, selectionOrigin]);

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

  const scrollToAnswer = useCallback((range: ActiveRange) => {
    setMobilePane("answer");
    window.requestAnimationFrame(() => {
      const viewport = answerRef.current;
      const target = viewport ? answerElements(viewport, range)[0] : undefined;
      if (!viewport || !target) return;
      const targetRect = target.getBoundingClientRect();
      const viewportRect = viewport.getBoundingClientRect();
      const top = viewport.scrollTop + targetRect.top - viewportRect.top - viewport.clientHeight / 2 + targetRect.height / 2;
      const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
      viewport.scrollTo?.({ top: Math.max(0, top), behavior: reduced ? "auto" : "smooth" });
    });
  }, []);

  const navigateAnswer = useCallback((direction: "above" | "below") => {
    const keys = new Set(geometry.answerAnchors
      .filter((anchor) => anchor.offscreen === direction)
      .map((anchor) => anchor.key));
    const candidates = activeAnswerRanges.filter((range) => keys.has(range.key));
    const next = direction === "above" ? candidates.at(-1) : candidates[0];
    if (next) scrollToAnswer(next);
  }, [activeAnswerRanges, geometry.answerAnchors, scrollToAnswer]);

  const exploreAnswer = useCallback((range: ActiveRange) => {
    setSelectionOrigin("answer");
    setHoveredRange(undefined);
    setHoveredSourceId(undefined);
    setCoordinate(range);
    scrollToAnswer(range);
  }, [setCoordinate, scrollToAnswer]);

  const rankedPassages = useMemo(() => answerPassages.filter(item => item.heat.intensity > 0)
    .sort((a, b) => b.heat.intensity - a.heat.intensity).map(item => ({ ...item.range, key: item.key })), [answerPassages]);

  const focusAnswerRange = useCallback((range: ActiveRange) => {
    if (!focusedSourceId) return;
    setHoveredSourceId(undefined);
    setHoveredRange(undefined);
    setCoordinate(range, focusedSourceId);
    setMobilePane("answer");
  }, [focusedSourceId, setCoordinate]);

  const contextFocusModel = useMemo<InfluenceFocusModel | undefined>(() => {
    if (selectionOrigin !== "answer" || !lockedRange || !answer) return undefined;
    const items = [...signedContributions(reader.loci, answer.tokens, lockedRange)].flatMap(([sourceId, values]) => {
      const source = sourceById.get(sourceId);
      if (!source) return [];
      const positive = [...values.values()].reduce((sum, value) => sum + Math.max(0, value), 0);
      const negative = [...values.values()].reduce((sum, value) => sum + Math.min(0, value), 0);
      return [positive, negative].filter((delta) => delta !== 0).map((delta) => ({
        key: `${sourceId}:${delta > 0 ? "support" : "suppression"}`,
        direction: delta > 0 ? "support" as const : "suppression" as const,
        delta,
        text: sourceRecordedText(reader, source),
        onSelect: () => focusSource(sourceId),
      }));
    });
    return {
      key: `answer:${lockedRange.key}:${reader.measurement.experimentId ?? "unmeasured"}`,
      title: answerLens === "tension"
        ? "Context competing around this answer span"
        : answerLens === "suppression" ? "Context suppressing this answer span" : "Most influential context",
      items,
      primaryDirection: answerLens === "suppression" ? "suppression" : undefined,
    };
  }, [answer, answerLens, focusSource, lockedRange, reader, selectionOrigin, sourceById]);

  const answerFocusModel = useMemo<InfluenceFocusModel | undefined>(() => {
    if (selectionOrigin !== "context" || !focusedSourceId || !answer) return undefined;
    const source = sourceById.get(focusedSourceId);
    if (!source) return undefined;
    const sourceLoci = reader.loci.filter((locus) => locus.sourceId === focusedSourceId);
    const items = answerPhraseRanges.flatMap((range) => {
      const heat = answerHeatForRange(sourceLoci, answer.tokens, range);
      const active = { ...range, key: rangeKey(range) };
      return [heat.support, -heat.suppression].filter((delta) => delta !== 0).map((delta) => ({
        key: `${active.key}:${Math.sign(delta)}`,
        direction: delta > 0 ? "support" as const : "suppression" as const,
        delta,
        text: sliceCodePoints(answer.text, range.start, range.end),
        onSelect: () => focusAnswerRange(active),
      }));
    });
    return {
      key: `context:${focusedSourceId}:${reader.measurement.experimentId ?? "unmeasured"}`,
      title: "Most influenced answer spans",
      items,
      primaryDirection: answerLens === "suppression" ? "suppression" : undefined,
    };
  }, [answer, answerLens, answerPhraseRanges, focusAnswerRange, focusedSourceId, reader, selectionOrigin, sourceById]);

  const relatedSet = useMemo(() => new Set(activeSourceIds), [activeSourceIds]);
  const activeDirections = useMemo(() => new Set(directionBySource.values()), [directionBySource]);
  const displayedRelationships = geometry.relationships.filter((relationship) => visibleSourceIds.has(relationship.sourceId));
  const peekSourceId = hoveredSourceId ?? (hoveredRange ? undefined : focusedSourceId);
  const peekRange = peekSourceId ? undefined : activeRange;
  return (
    <>
      <div
        ref={readingWorkspaceRef}
        className={`inspect-reading-workspace${expandedPane ? " is-expanded" : ""}`}
        role={expandedPane ? "dialog" : undefined}
        aria-modal={expandedPane ? true : undefined}
        aria-label={expandedPane ? `Expanded ${expandedPane} heatmap` : undefined}
        onKeyDownCapture={(event) => {
          if (!expandedPane) return;
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            setExpandedPane(undefined);
          } else if (event.key === "Tab") {
            const targets = [...event.currentTarget.querySelectorAll<HTMLElement>('button:not(:disabled), a[href], [tabindex="0"]')]
              .filter((element) => element.getClientRects().length > 0);
            const first = targets[0];
            const last = targets[targets.length - 1];
            if (event.shiftKey && document.activeElement === first) {
              event.preventDefault();
              last?.focus({ preventScroll: true });
            } else if (!event.shiftKey && document.activeElement === last) {
              event.preventDefault();
              first?.focus({ preventScroll: true });
            }
          }
        }}
      >
      <div className="inspect-pane-switch" role="group" aria-label="Readable pane">
        <button type="button" aria-pressed={mobilePane === "answer"} onClick={() => {
          setMobilePane("answer");
          if (expandedPane) setExpandedPane("answer");
        }}>Answer</button>
        <button type="button" aria-pressed={mobilePane === "context"} onClick={() => {
          setMobilePane("context");
          if (expandedPane) setExpandedPane("context");
        }}>Context</button>
      </div>
      <SequenceOverview
        reader={reader}
        activeRanges={activeAnswerRanges}
        directions={activeDirections}
        answerLens={answerLens}
        answerEntries={answerPassages}
        answerScale={answerHeatCeilings}
        onSelectContext={lockContextSource}
        onSelectAnswer={exploreAnswer}
      />
      <AnswerLensToolbar lens={answerLens} onChange={setAnswerLens} exploring={Boolean(lockedRange)}
        onExplore={rankedPassages.length ? () => {
          const current = rankedPassages.findIndex((range) => rangesEqual(range, lockedRange));
          exploreAnswer(rankedPassages[(current + 1) % rankedPassages.length]);
        } : undefined}
      />
      <div className={`inspect-reader mobile-${mobilePane}${lockedRange ? " has-lock" : ""}${expandedPane ? ` is-expanded is-expanded-${expandedPane}` : ""}`}>
        <div className="inspect-pane-slot context-slot">
          <ContextPane
            reader={reader}
            paneRef={contextRef}
            activeSourceIds={relatedSet}
            focusedSourceId={displayedFocusedSourceId}
            directionBySource={directionBySource}
            focusModel={contextFocusModel}
            pinned={Boolean(lockedRange && focusedSourceId && selectionOrigin === "context")}
            selectionActive={Boolean(lockedRange)}
            linkedSelection={Boolean(lockedRange && selectionOrigin === "answer")}
            expanded={expandedPane === "context"}
            maxInfluence={maxSourceInfluence}
            onFocusSource={lockContextSource}
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
          onClear={clearSelection}
        />
        <div className="inspect-pane-slot answer-slot">
          <AnswerPane
            reader={reader}
            paneRef={answerRef}
            passages={answerPassages}
            onExplore={exploreAnswer}
            activeRange={selectionOrigin === "answer" ? activeRange : undefined}
            lockedRange={selectionOrigin === "answer" ? lockedRange : undefined}
            linkedRanges={contextPinnedSourceId ? activeAnswerRanges : undefined}
            linkedDirectionsByRange={linkedDirectionsByRange}
            lockedRelationshipCount={selectionOrigin === "answer" && lockedRange && query.phase === "ready"
              ? query.value.effects.filter(isAboveFloorEffect).length
              : undefined}
            focusModel={answerFocusModel}
            pinned={Boolean(lockedRange && selectionOrigin === "answer")}
            expanded={expandedPane === "answer"}
            lens={answerLens}
            heatCeilings={answerHeatCeilings}
            onPreview={previewAnswerRange}
            onLock={(range) => {
              setSelectionOrigin("answer");
              if (lockedRange && rangesIntersect(lockedRange, range)) clearSelection();
              else {
                setCoordinate(range);
                setMobilePane("answer");
              }
            }}
            onClear={clearSelection}
            onToggleExpanded={() => {
              setHoveredSourceId(undefined);
              setHoveredRange(undefined);
              setExpandedPane((current) => current === "answer" ? undefined : "answer");
            }}
          />
          <div className="local-registration-layer answer-registration" aria-hidden="true">
            {geometry.answerAnchors.filter((anchor) => anchor.localY !== undefined).map((anchor) => (
              <i key={anchor.key} style={{ "--registration-y": `${anchor.localY}px` } as CSSProperties} />
            ))}
          </div>
        </div>
      </div>
      <HeatmapPeek
        reader={reader}
        pane={peekSourceId ? "context" : "answer"}
        sourceId={peekSourceId}
        range={peekRange}
        answerLens={answerLens}
        expanded={Boolean(expandedPane)}
      />
      </div>
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
            relationshipCount={lockedAboveFloorEffects?.length}
            focusedSourceId={focusedSourceId}
            onPrevious={() => navigateSource(-1)}
            onNext={() => navigateSource(1)}
            onClear={clearSelection}
          />
        </>
      )}
    </>
  );
}

function phaseLabel(value: string): string {
  return value.replaceAll("_", " ");
}

function contextPartitionSources(
  reader: ContextInvestigationReader,
): ContextInvestigationSource[] {
  const selected: ContextInvestigationSource[] = [];
  for (const block of reader.context.blocks) {
    const sources = reader.sources.filter((source) => source.messageIndex === block.messageIndex);
    const spans = sources
      .filter((source) => source.granularity === "exact_span")
      .sort((left, right) => left.unicodeRange[0] - right.unicodeRange[0]);
    if (spans.length > 1) {
      selected.push(...spans);
      continue;
    }
    const root = sources.find((source) => source.granularity === "whole_segment");
    if (root) selected.push(root);
    else if (spans[0]) selected.push(spans[0]);
  }
  return selected;
}

function contextPartitionBounds(reader: ContextInvestigationReader): {
  minimum: number;
  maximum: number;
} {
  const partition = contextPartitionSources(reader);
  const minimum = new Set(partition.map((source) => source.messageIndex)).size;
  const maximum = partition.length;
  return { minimum, maximum: Math.max(minimum, maximum) };
}

function EffectsPanel({
  reader,
  route,
  onReaderUpdate,
}: {
  reader: ContextInvestigationReader;
  route: InspectRoute;
  onReaderUpdate: (reader: ContextInvestigationReader) => void;
}) {
  const [job, setJob] = useState<ContextEffectsJob | undefined>();
  const [action, setAction] = useState<"idle" | "starting" | "cancelling">("idle");
  const [error, setError] = useState<string | undefined>();
  const [pollRevision, setPollRevision] = useState(0);
  const actionController = useRef<AbortController | undefined>(undefined);
  const pollController = useRef<AbortController | undefined>(undefined);
  const evidenceController = useRef<AbortController | undefined>(undefined);
  const publishedEvidenceKey = useRef<string | undefined>(undefined);
  const sourceIds = reader.measurement.sourceUniverse;
  const customCandidates = useMemo(() => contextPartitionSources(reader), [reader]);
  const customCandidateKey = customCandidates.map((source) => source.sourceId).join(",");
  const [customMode, setCustomMode] = useState(false);
  const [customSourceIds, setCustomSourceIds] = useState<string[]>([]);
  const [settingsOpen, setSettingsOpen] = useState(
    reader.status !== "completed" && reader.status !== "partial",
  );
  const selectedCustomSourceIds = useMemo(() => {
    const candidates = new Set(customCandidates.map((source) => source.sourceId));
    return customSourceIds.filter((sourceId) => candidates.has(sourceId));
  // The candidate IDs, rather than changing reader evidence, define legality.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [customCandidateKey, customSourceIds]);
  const bounds = useMemo(() => contextPartitionBounds(reader), [reader]);
  const [defaultArmBudget] = useState(() => Math.max(bounds.minimum, sourceIds.length));
  const granularity = route.granularity ?? "adaptive";
  const armBudget = route.armBudget ?? defaultArmBudget;
  const scoringMode: ScoringMode = route.scoringMode ?? "fast";
  const configuredRequest = useMemo(() => ({
    granularity,
    maxArms: armBudget,
    scoringMode,
  }), [armBudget, granularity, scoringMode]);
  const experimentRequest = useMemo<ContextEffectsRequest>(() => customMode
    ? { sourceIds: selectedCustomSourceIds, scoringMode }
    : configuredRequest, [configuredRequest, customMode, scoringMode, selectedCustomSourceIds]);
  const experimentRequestKey = customMode
    ? `${scoringMode}:custom:${selectedCustomSourceIds.join(",")}`
    : `${scoringMode}:${granularity}:${armBudget}`;
  const plannedSourceIds = customMode ? selectedCustomSourceIds : sourceIds;

  const configure = (nextGranularity: "message" | "adaptive", nextBudget: number) => {
    window.location.hash = routeHref({
      ...route,
      granularity: nextGranularity,
      armBudget: nextBudget,
      ...(route.scoringMode ? { scoringMode: route.scoringMode } : {}),
    });
  };

  const configureScoring = (nextScoringMode: ScoringMode) => {
    const nextRoute = { ...route };
    if (nextScoringMode === "fast") delete nextRoute.scoringMode;
    else nextRoute.scoringMode = nextScoringMode;
    window.location.hash = routeHref(nextRoute);
  };

  const publishAvailableEvidence = useCallback(async (
    next: ContextEffectsJob,
  ) => {
    const terminal = isTerminalContextEffectsJob(next);
    if (next.progress.completedUnits === 0 && !terminal) return;
    const key = `${next.jobId}:${next.progress.completedUnits}:${terminal ? next.state : "running"}`;
    if (publishedEvidenceKey.current === key) return;
    publishedEvidenceKey.current = key;
    evidenceController.current?.abort();
    const controller = new AbortController();
    evidenceController.current = controller;
    try {
      const updated = await contextInvestigationApi.reader(
        route.runId, controller.signal, experimentRequest,
      );
      if (!controller.signal.aborted) onReaderUpdate(updated);
    } catch (readerError: unknown) {
      publishedEvidenceKey.current = undefined;
      if (!controller.signal.aborted) setError(describeError(readerError));
    } finally {
      if (evidenceController.current === controller) evidenceController.current = undefined;
    }
  }, [experimentRequest, onReaderUpdate, route.runId]);

  useEffect(() => {
    setCustomMode(false);
    setCustomSourceIds([]);
  }, [route.runId]);

  useEffect(() => {
    if (!customMode || !selectedCustomSourceIds.length) return undefined;
    const controller = new AbortController();
    void contextInvestigationApi.reader(
      route.runId, controller.signal, { sourceIds: selectedCustomSourceIds, scoringMode },
    ).then((updated) => {
      if (!controller.signal.aborted) onReaderUpdate(updated);
    }).catch((readerError: unknown) => {
      if (!controller.signal.aborted) setError(describeError(readerError));
    });
    return () => controller.abort();
  }, [customMode, onReaderUpdate, route.runId, scoringMode, selectedCustomSourceIds]);

  useEffect(() => {
    actionController.current?.abort();
    actionController.current = undefined;
    pollController.current?.abort();
    pollController.current = undefined;
    evidenceController.current?.abort();
    evidenceController.current = undefined;
    publishedEvidenceKey.current = undefined;
    setJob(undefined);
    setAction("idle");
    setError(undefined);
    setPollRevision(0);
    return () => evidenceController.current?.abort();
  }, [experimentRequestKey, route.runId]);

  useEffect(() => {
    if (!job || isTerminalContextEffectsJob(job)) return undefined;
    const controller = new AbortController();
    pollController.current = controller;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const next = await contextInvestigationApi.effectsJob(
          route.runId, job.jobId, controller.signal,
        );
        if (controller.signal.aborted) return;
        setJob(next);
        await publishAvailableEvidence(next);
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
      if (pollController.current === controller) pollController.current = undefined;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [job?.jobId, job?.state, pollRevision, publishAvailableEvidence, route.runId]);

  const start = async () => {
    actionController.current?.abort();
    const controller = new AbortController();
    actionController.current = controller;
    setAction("starting");
    setError(undefined);
    try {
      const next = await contextInvestigationApi.startEffects(
        route.runId, experimentRequest, controller.signal,
      );
      if (!controller.signal.aborted) {
        setJob(next);
        await publishAvailableEvidence(next);
      }
    } catch (startError: unknown) {
      if (!controller.signal.aborted) setError(describeError(startError));
    } finally {
      if (!controller.signal.aborted) setAction("idle");
    }
  };

  const cancel = async () => {
    if (!job?.cancellable) return;
    pollController.current?.abort();
    actionController.current?.abort();
    const controller = new AbortController();
    actionController.current = controller;
    setAction("cancelling");
    setError(undefined);
    try {
      const next = await contextInvestigationApi.cancelEffects(
        route.runId, job.jobId, controller.signal,
      );
      if (!controller.signal.aborted) {
        setJob(next);
        await publishAvailableEvidence(next);
      }
    } catch (cancelError: unknown) {
      if (!controller.signal.aborted) setError(describeError(cancelError));
    } finally {
      if (!controller.signal.aborted) setAction("idle");
    }
  };

  const active = Boolean(job && !isTerminalContextEffectsJob(job));
  const progressPending = Boolean(active && job?.progress.completedUnits === 0);
  return (
    <section className="effects-search-panel" aria-label="Context measurement">
      <details
        className="effects-search-settings"
        open={settingsOpen}
        onToggle={(event) => setSettingsOpen(event.currentTarget.open)}
      >
        <summary>
          <div className="effects-search-copy">
            <span className="eyebrow">Measurement setup</span>
            <strong>{plannedSourceIds.length ? plural(plannedSourceIds.length, "context span") : "No sources selected"}</strong>
            <span>{reader.status === "completed" ? "Measurement available" : `Up to ${plural(plannedSourceIds.length, "source score")}, plus a reusable baseline.`}</span>
          </div>
          <span className="effects-settings-action">Configure</span>
        </summary>
        <div className="effects-search-body">
          <div className="effects-search-config">
            <label>
              Granularity
              <select
                value={customMode ? "custom" : granularity}
                disabled={active}
                onChange={(event) => {
                  if (event.target.value === "custom") {
                    setCustomMode(true);
                    setCustomSourceIds([]);
                    return;
                  }
                  setCustomMode(false);
                  const next = event.target.value === "message" ? "message" : "adaptive";
                  configure(next, next === "message" ? bounds.minimum : bounds.maximum);
                }}
              >
                <option value="adaptive">Markdown sections</option>
                <option value="message">Whole messages</option>
                <option value="custom">Custom spans</option>
              </select>
            </label>
            <label>
              Arm budget · {armBudget}
              <input
                type="range"
                min={bounds.minimum}
                max={bounds.maximum}
                step="1"
                value={Math.min(bounds.maximum, Math.max(bounds.minimum, armBudget))}
                disabled={active || customMode || granularity === "message" || bounds.minimum === bounds.maximum}
                onChange={(event) => configure("adaptive", Number(event.target.value))}
              />
            </label>
            <label>
              Scoring
              <select
                value={scoringMode}
                disabled={active}
                onChange={(event) => configureScoring(event.target.value === "exact" ? "exact" : "fast")}
              >
                <option value="fast">Fast · auto</option>
                <option value="exact">Exact · scalar reference</option>
              </select>
            </label>
            <small>{customMode
              ? "Each selected span becomes one removal arm."
              : scoringMode === "fast"
                ? "More arms preserve smaller Markdown sections. Fast chooses native batching for prompt-heavy work and exact prefix-cached scoring when answer decoding would dominate."
                : "More arms preserve smaller Markdown sections. Exact scores each arm through the scalar reference path."}</small>
          </div>
          <div className="effects-search-action">
            {active ? (
              <button type="button" disabled={!job?.cancellable || action !== "idle"} onClick={() => { void cancel(); }}>
                {action === "cancelling" ? "Cancelling…" : "Cancel"}
              </button>
            ) : (
              <button type="button" disabled={!plannedSourceIds.length || action !== "idle"} onClick={() => { void start(); }}>
                {action === "starting" ? "Starting…" : job ? "Run again" : "Measure context"}
              </button>
            )}
          </div>
          {customMode && (
            <fieldset className="custom-source-picker" disabled={active}>
              <legend>Choose exact recorded context spans · {selectedCustomSourceIds.length} selected</legend>
              <div>
                {customCandidates.map((source, index) => (
                  <label key={source.sourceId}>
                    <input
                      type="checkbox"
                      checked={selectedCustomSourceIds.includes(source.sourceId)}
                      onChange={(event) => setCustomSourceIds((current) => event.target.checked
                        ? [...current, source.sourceId]
                        : current.filter((sourceId) => sourceId !== source.sourceId))}
                    />
                    <span>{excerpt(source.label?.trim() || sourceRecordedText(reader, source), 150)}</span>
                    <small>Span {index + 1}</small>
                  </label>
                ))}
              </div>
            </fieldset>
          )}
        </div>
      </details>
      {job && (
        <div className={`effects-search-progress state-${job.state}`} role="status" aria-live="polite">
          <span>{job.state === "completed" ? "Completed" : job.state === "failed" ? "Failed" : job.state === "cancelled" ? "Cancelled" : phaseLabel(job.progress.phase)}</span>
          <strong>{progressPending ? "Working…" : `${job.progress.percent.toFixed(0)}%`}</strong>
          {!isTerminalContextEffectsJob(job) && (
            progressPending ? <progress max="100" /> : <progress max="100" value={job.progress.percent} />
          )}
          {progressPending && (
            <small>Links and heatmaps update as each scoring batch finishes. Foreground chat takes priority between batches.</small>
          )}
          {job.result && (
            <small>
              {job.result.completedSourceCount} measured · {job.result.unavailableSourceCount} unavailable · {job.result.failedSourceCount} failed · {job.result.notMeasuredSourceCount} not measured
              {` · ${job.result.elapsedSeconds.toFixed(1)}s elapsed · ${job.result.executedScoreCount} new / ${job.result.reusedScoreCount} reused scores`}
              {` · ${job.result.scoringMode === "fast" ? "Fast" : "Exact"} requested · ${job.result.evidenceQuality} evidence · ${job.result.executionStrategy.replaceAll("_", " ")}`}
              {job.result.executedScoreCount > 0 && job.result.elapsedSeconds > 0
                ? ` · ${(job.result.executedScoreCount / job.result.elapsedSeconds).toFixed(2)} new scores/s`
                : ""}
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
  supportShare: number;
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
  const answer = reader.answer.status === "available" ? reader.answer : undefined;
  const answerPhraseRanges = useMemo(
    () => answer ? semanticTextRanges(answer.text) : [],
    [answer],
  );
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
    if (!answer) return [];
    const grouped = new Map<string, { range: CodePointRange; loci: ContextInvestigationLocus[] }>();
    for (const locus of reader.loci) {
      const locusRange = tupleRange(locus.answerUnicodeRange);
      for (const range of answerPhraseRanges.filter((item) => rangesIntersect(item, locusRange))) {
        const key = rangeKey(range);
        const current = grouped.get(key) ?? { range, loci: [] };
        current.loci.push(locus);
        grouped.set(key, current);
      }
    }
    return [...grouped.values()].map<AnswerInfluenceRow>(({ range: groupedRange, loci }) => {
      const range = {
        ...groupedRange,
        key: rangeKey(groupedRange),
      };
      const heat = answerHeatForRange(loci, answer.tokens, range);
      return {
        range,
        text: sliceCodePoints(answer.text, range.start, range.end),
        influence: heat.intensity,
        direction: heat.direction,
        supportShare: heat.total ? heat.support / heat.total : 0,
        sourceCount: new Set(loci.map((locus) => locus.sourceId)).size,
      };
    }).sort((left, right) => right.influence - left.influence);
  }, [answer, answerPhraseRanges, reader.loci]);
  const maxContext = contextRows[0]?.influence ?? 0;
  const maxAnswer = answerRows[0]?.influence ?? 0;
  const contextHighlights = contextRows.length > 1
    ? [
      { row: contextRows[0], label: "Strongest" },
      { row: contextRows[contextRows.length - 1], label: "Weakest" },
    ]
    : contextRows.map((row) => ({ row, label: "Strongest" }));
  const answerHighlights = answerRows.length > 1
    ? [
      { row: answerRows[0], label: "Most affected" },
      { row: answerRows[answerRows.length - 1], label: "Least affected" },
    ]
    : answerRows.map((row) => ({ row, label: "Most affected" }));
  return (
    <section className="efficiency-overview" aria-label="Key effects">
      <div className="efficiency-grid">
        <section aria-labelledby="context-ranking-heading">
          <header>
            <div>
              <h3 id="context-ranking-heading">Context</h3>
              <p>Which passages moved the answer most?</p>
            </div>
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
                      onOpenSource(
                        row.source.sourceId,
                        semanticRangeFor(answerPhraseRanges, tupleRange(row.locus.answerUnicodeRange)),
                      );
                    }}
                  >
                    <span className="ranking-copy">
                      <span>{label}</span>
                      <strong>{excerpt(row.text, 240)}</strong>
                      <small>{row.source.summary.totalContinuationDeltaNats !== null
                        ? `${row.source.summary.totalContinuationDeltaNats > 0 ? "+" : ""}${row.source.summary.totalContinuationDeltaNats.toFixed(2)} total Δ nats`
                        : "No score"}</small>
                    </span>
                    <span
                      className={`ranking-meter direction-${heatDirection(row.source.summary.totalContinuationDeltaNats ?? 0)}`}
                      aria-hidden="true"
                    >
                      <i style={{ "--meter-size": `${maxContext ? row.influence / maxContext * 100 : 0}%` } as CSSProperties} />
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          ) : <p className="overview-empty">Measure Context Effects to rank context.</p>}
        </section>
        <section aria-labelledby="answer-ranking-heading">
          <header>
            <div>
              <h3 id="answer-ranking-heading">Answer</h3>
              <p>Which answer passages depended most on context?</p>
            </div>
            <span>{plural(answerRows.length, "affected span")}</span>
          </header>
          {answerRows.length ? (
            <ol className="influence-ranking">
              {answerHighlights.map(({ row, label }) => (
                <li key={row.range.key}>
                  <button type="button" onClick={() => onOpenAnswer(row.range)}>
                    <span className="ranking-copy">
                      <span>{label}</span>
                      <strong>{excerpt(row.text, 240)}</strong>
                      <small>{row.influence.toFixed(2)} absolute Δ nats / token · {plural(row.sourceCount, "source")}</small>
                    </span>
                    <span className={`ranking-meter direction-${row.direction}`} aria-hidden="true"><i style={{ "--meter-size": `${maxAnswer ? row.influence / maxAnswer * 100 : 0}%`, "--support-share": `${row.supportShare * 100}%` } as CSSProperties} /></span>
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

function interactionEstimateStyle(
  estimate: ContextInteractionEstimate,
  ceiling: number,
): CSSProperties {
  const mean = estimate.meanMarginalNats ?? 0;
  const normalized = ceiling ? Math.min(1, Math.abs(mean) / ceiling) : 0;
  const size = normalized * 50;
  const leaveOneOut = estimate.leaveOneOutNats;
  const marker = leaveOneOut === null || !ceiling
    ? 50
    : Math.max(0, Math.min(100, 50 + leaveOneOut / ceiling * 50));
  return {
    "--interaction-size": `${size}%`,
    "--interaction-marker": `${marker}%`,
    "--interaction-alpha": `${Math.round(7 + Math.sqrt(normalized) * 28)}%`,
  } as CSSProperties;
}

function InteractionsPanel({
  reader,
  route,
}: {
  reader: ContextInvestigationReader;
  route: InspectRoute;
}) {
  const [budget, setBudget] = useState(16);
  const [job, setJob] = useState<ContextInteractionsJob | undefined>();
  const [action, setAction] = useState<"idle" | "starting" | "cancelling">("idle");
  const [error, setError] = useState<string | undefined>();
  const [pollRevision, setPollRevision] = useState(0);
  const actionController = useRef<AbortController | undefined>(undefined);
  const maxPairs = Math.min(256, budget * 2);

  useEffect(() => {
    actionController.current?.abort();
    actionController.current = undefined;
    setJob(undefined);
    setAction("idle");
    setError(undefined);
    setPollRevision(0);
  }, [route.runId]);

  useEffect(() => {
    if (!job || isTerminalContextInteractionsJob(job)) return undefined;
    const controller = new AbortController();
    let timer: number | undefined;
    const poll = async () => {
      try {
        const next = await contextInvestigationApi.interactionsJob(
          route.runId, job.jobId, controller.signal,
        );
        if (controller.signal.aborted) return;
        setJob(next);
        if (!isTerminalContextInteractionsJob(next)) {
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
      const next = await contextInvestigationApi.startInteractions(route.runId, {
        maxNewExecutions: budget,
        maxPairs,
        seed: 0,
      }, controller.signal);
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
      const next = await contextInvestigationApi.cancelInteractions(
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
  const active = Boolean(job && !isTerminalContextInteractionsJob(job));
  const progressPending = Boolean(active && job?.progress.completedUnits === 0);
  const sourceById = useMemo(
    () => new Map(reader.sources.map((source) => [source.sourceId, source])),
    [reader.sources],
  );
  const estimates = result?.sourceEstimates ?? [];
  const ceiling = Math.max(0, ...estimates.flatMap((estimate) => [
    Math.abs(estimate.meanMarginalNats ?? 0),
    Math.abs(estimate.leaveOneOutNats ?? 0),
  ]));
  const sampledCount = estimates.filter((estimate) => estimate.sampleCount > 0).length;
  const terminal = Boolean(job && isTerminalContextInteractionsJob(job));
  const displayedEstimates = terminal
    ? estimates.filter((estimate) => estimate.sampleCount > 0)
    : estimates;
  const unsampledCount = estimates.length - sampledCount;
  const throughput = result?.elapsedSeconds && result.budget.usedNewExecutions
    ? result.budget.usedNewExecutions / result.elapsedSeconds
    : null;
  return (
    <section className="interactions-panel" aria-labelledby="interactions-heading">
      <header className="overview-heading">
        <div>
          <span className="eyebrow">Optional · sampled experiment</span>
          <h2 id="interactions-heading">Does each span stay influential when the surrounding context changes?</h2>
        </div>
        <p>Clozn samples different context combinations and watches whether a span's effect stays stable. A large shift suggests interaction with other context; it does not by itself prove redundancy.</p>
      </header>
      <div className="interaction-controls">
        <label>
          <span>New model-execution budget <strong>{budget}</strong></span>
          <input
            type="range"
            min="4"
            max="64"
            step="4"
            value={budget}
            disabled={active}
            onChange={(event) => setBudget(Number(event.target.value))}
          />
          <small>More work samples more coalitions; compatible prior scores are reused for free.</small>
        </label>
        <div className="interaction-action">
          <span>Up to {maxPairs} paired samples · deterministic seed 0</span>
          {active ? (
            <button type="button" disabled={!job?.cancellable || action !== "idle"} onClick={() => { void cancel(); }}>
              {action === "cancelling" ? "Cancelling…" : "Cancel experiment"}
            </button>
          ) : (
            <button type="button" disabled={!reader.measurement.sourceUniverse.length || action !== "idle"} onClick={() => { void start(); }}>
              {action === "starting" ? "Starting…" : job ? "Run again" : "Run stability test"}
            </button>
          )}
        </div>
      </div>
      {job && (
        <div className={`interaction-progress state-${job.state}`} role="status" aria-live="polite">
          <span>{job.state === "completed" ? "Experiment complete" : job.state === "failed" ? "Experiment failed" : job.state === "cancelled" ? "Experiment cancelled" : phaseLabel(job.progress.phase)}</span>
          <strong>{progressPending ? "Preparing…" : `${job.progress.percent.toFixed(0)}%`}</strong>
          {!isTerminalContextInteractionsJob(job) && (
            progressPending ? <progress max="100" /> : <progress max="100" value={job.progress.percent} />
          )}
          {result && (
            <small>
              {plural(result.budget.completedPairs, "pair")} · {result.budget.usedNewExecutions} / {result.budget.maxNewExecutions} new executions · {plural(result.budget.reusedObservationCount, "score")} reused
            </small>
          )}
          {job.error && <small className="is-failed">{job.error.message}</small>}
        </div>
      )}
      {result && estimates.length > 0 && (
        <div className="interaction-heatmap" aria-live="polite">
          <header>
            <div>
              <h3>Effect across changing context</h3>
              <span>{sampledCount} / {estimates.length} spans sampled</span>
            </div>
            <div className="interaction-legend" aria-label="Chart legend">
              <span className="direction-suppression"><i />Suppresses answer</span>
              <span className="direction-support"><i />Supports answer</span>
              <span className="leave-one-out-key"><i />Leave-one-out</span>
            </div>
          </header>
          <ol>
            {displayedEstimates.map((estimate) => {
              const index = estimates.findIndex((candidate) => candidate.sourceId === estimate.sourceId);
              const source = sourceById.get(estimate.sourceId);
              const mean = estimate.meanMarginalNats;
              const direction = mean === null ? "unsampled" : mean < 0 ? "suppression" : "support";
              const difference = estimate.meanMinusLeaveOneOutNats;
              return (
                <li
                  key={estimate.sourceId}
                  className={`interaction-row direction-${direction}`}
                  style={interactionEstimateStyle(estimate, ceiling)}
                >
                  <div className="interaction-source">
                    <span>Span {index + 1}</span>
                    <strong>{excerpt(source?.label?.trim() || (source ? sourceRecordedText(reader, source) : estimate.sourceId), 132)}</strong>
                    <small>{estimate.sampleCount
                      ? `${plural(estimate.sampleCount, "sample")} · ${plural(estimate.uniquePairCount, "unique coalition")}`
                      : "Waiting for a completed pair"}</small>
                  </div>
                  <div className="interaction-plot" aria-label={mean === null
                    ? "Not sampled"
                    : `${mean.toFixed(2)} mean marginal delta nats${estimate.leaveOneOutNats === null ? "" : `, ${estimate.leaveOneOutNats.toFixed(2)} leave-one-out delta nats`}`}>
                    <i className="interaction-axis" aria-hidden="true" />
                    {mean !== null && <i className="interaction-bar" aria-hidden="true" />}
                    {estimate.leaveOneOutNats !== null && <i className="interaction-marker" aria-hidden="true" />}
                  </div>
                  <div className="interaction-value">
                    <strong>{mean === null ? "—" : `${mean > 0 ? "+" : ""}${mean.toFixed(2)} Δ`}</strong>
                    <small>{estimate.sampleStddevNats === null ? "" : `±${estimate.sampleStddevNats.toFixed(2)} sample SD`}</small>
                    {difference !== null && <span>{difference > 0 ? "+" : ""}{difference.toFixed(2)} vs leave-one-out</span>}
                  </div>
                </li>
              );
            })}
          </ol>
          {terminal && unsampledCount > 0 && (
            <p className="interaction-unsampled">{plural(unsampledCount, "span")} not sampled within this budget.</p>
          )}
          <p className="interaction-claim">A gap from leave-one-out means the effect changed with the surrounding context. It is evidence of context dependence, not by itself proof that two spans are redundant.</p>
        </div>
      )}
      {result?.best && (
        <div className="interaction-best">
          <div>
            <span className="eyebrow">Cheapest tolerant context directly tested</span>
            <strong>{plural(result.best.retainedSourceIds.length, "span")} · {result.best.renderedPromptTokens} prompt tokens</strong>
          </div>
          <small>This is the best observed candidate, not a proof of the minimum.</small>
        </div>
      )}
      {result?.elapsedSeconds !== null && result?.elapsedSeconds !== undefined && (
        <p className="interaction-timing">
          Measured elapsed time {result.elapsedSeconds.toFixed(1)}s
          {throughput !== null ? ` · ${throughput.toFixed(2)} new executions/s` : ""}
        </p>
      )}
      {result?.status === "unavailable" && (
        <p className="lean-unavailable">No trustworthy control score was available within this budget.</p>
      )}
      {!job && <p className="lean-empty">Run an opt-in sample to see whether source effects stay stable—or change—when other context is removed. Rows fill as paired experiments finish.</p>}
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

export function LeanContextPanel({
  reader,
  route,
  initialJob,
  allowSearch = true,
}: {
  reader: ContextInvestigationReader;
  route: InspectRoute;
  initialJob?: LeanContextJob;
  allowSearch?: boolean;
}) {
  const [job, setJob] = useState<LeanContextJob | undefined>(initialJob);
  const [action, setAction] = useState<"idle" | "starting" | "cancelling">("idle");
  const [error, setError] = useState<string | undefined>();
  const [pollRevision, setPollRevision] = useState(0);
  const [searchBudget, setSearchBudget] = useState(16);
  const [contextView, setContextView] = useState<"redline" | "kept">("redline");
  const actionController = useRef<AbortController | undefined>(undefined);

  useEffect(() => {
    actionController.current?.abort();
    actionController.current = undefined;
    setJob(initialJob);
    setAction("idle");
    setError(undefined);
    setPollRevision(0);
    setContextView("redline");
  }, [initialJob, route.runId]);

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
    if (!allowSearch) return;
    actionController.current?.abort();
    const controller = new AbortController();
    actionController.current = controller;
    setAction("starting");
    setError(undefined);
    try {
      const next = await contextInvestigationApi.startLeanContext(
        route.runId,
        { maxNewExecutions: searchBudget },
        controller.signal,
      );
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
  const leanSourceById = new Map(reader.sources.map((source) => [source.sourceId, source]));
  const removedSourceIds = new Set(result?.best?.removedSourceIds ?? []);
  const retainedSourceIds = new Set(result?.best?.retainedSourceIds ?? []);
  const leanSources = (result?.sourceIds ?? []).flatMap((sourceId) => {
    const source = leanSourceById.get(sourceId);
    return source ? [source] : [];
  });
  const leanContextBlocks = result ? reader.context.blocks.map((block) => {
    const annotations = leanSources
      .filter((source) => (
        source.messageIndex === block.messageIndex
        && isValidCodePointRange(block.text, {
          start: source.unicodeRange[0],
          end: source.unicodeRange[1],
        })
      ))
      .map((source) => ({
        id: source.sourceId,
        start: source.unicodeRange[0],
        end: source.unicodeRange[1],
      }));
    return {
      ...block,
      segments: segmentAnnotatedText(block.text, annotations),
    };
  }) : [];
  const displayedLeanContextBlocks = contextView === "redline"
    ? leanContextBlocks
    : leanContextBlocks.filter((block) => block.segments.some((segment) => {
      const removed = segment.annotationIds.some((sourceId) => removedSourceIds.has(sourceId));
      const retained = segment.annotationIds.some((sourceId) => retainedSourceIds.has(sourceId));
      return (!removed || retained) && Boolean(segment.text.trim());
    }));
  const recordedAnswer = reader.answer.status === "unavailable" ? undefined : reader.answer.text;
  const generationComparison = result?.generationComparison;
  const fullGeneratedAnswer = generationComparison?.status === "completed"
    ? generationComparison.fullContext?.text
    : undefined;
  const leanGeneratedAnswer = generationComparison?.status === "completed"
    ? generationComparison.leanContext?.text
    : undefined;
  const hasFreshGenerationComparison = fullGeneratedAnswer !== undefined && leanGeneratedAnswer !== undefined;
  const beforeAnswer = fullGeneratedAnswer ?? recordedAnswer;
  const afterAnswer = leanGeneratedAnswer ?? recordedAnswer;
  const removedPromptTokens = result?.reduction?.removedPromptTokens ?? 0;
  const active = Boolean(job && !isTerminalLeanContextJob(job));
  const progressPending = Boolean(active && job?.progress.completedUnits === 0);
  const answerUnavailable = reader.answer.status === "unavailable";
  const generatedAnswerChanged = Boolean(
    hasFreshGenerationComparison && !generationComparison?.exactMatch,
  );
  const generatedAnswerPreserved = Boolean(
    hasFreshGenerationComparison && generationComparison?.exactMatch,
  );
  const generationCheckLabel = generatedAnswerPreserved
    ? "Answer preserved"
    : generatedAnswerChanged
      ? "Answer changed"
      : "Fresh check unavailable";
  const searchControls = (
    <div className="lean-controls">
      <label>
        <span>Search depth</span>
        <select
          value={searchBudget}
          disabled={active}
          onChange={(event) => setSearchBudget(Number(event.target.value))}
        >
          <option value={8}>Quick · 8 trials</option>
          <option value={16}>Balanced · 16 trials</option>
          <option value={32}>Thorough · 32 trials</option>
          <option value={64}>Deep · 64 trials</option>
        </select>
      </label>
      {active ? (
        <button type="button" disabled={!job?.cancellable || action !== "idle"} onClick={() => { void cancel(); }}>
          {action === "cancelling" ? "Cancelling…" : "Cancel search"}
        </button>
      ) : (
        <button type="button" disabled={answerUnavailable || !reader.measurement.sourceUniverse.length || action !== "idle"} onClick={() => { void start(); }}>
          {action === "starting" ? "Starting…" : "Find lean context"}
        </button>
      )}
    </div>
  );
  return (
    <section className="lean-context-panel" aria-labelledby="lean-context-heading">
      {reader.answer.status === "text_only" && (
        <p className="lean-capability-note">
          <strong>The recorded response is available for scoring.</strong>
          Its token heatmap is not available yet, but Lean Context can still teacher-force and compare that response.
        </p>
      )}
      {answerUnavailable && (
        <p className="lean-capability-note is-unavailable">
          <strong>Lean Context cannot run for this turn.</strong>
          There is no recorded response to teacher-force and compare.
        </p>
      )}
      {reader.measurement.sourceUniverse.length === 1 && (
        <p className="lean-capability-note">
          <strong>This turn has only one removable context span.</strong>
          Search depth will not change the single available removal candidate.
        </p>
      )}
      {job && job.state !== "completed" && (
        <div className={`lean-progress state-${job.state}`} role="status" aria-live="polite">
          <span>{job.state === "failed" ? "Search failed" : job.state === "cancelled" ? "Search cancelled" : phaseLabel(job.progress.phase)}</span>
          <strong>{progressPending ? "Working…" : `${job.progress.percent.toFixed(0)}%`}</strong>
          {!isTerminalLeanContextJob(job) && (
            progressPending ? <progress max="100" /> : <progress max="100" value={job.progress.percent} />
          )}
          {progressPending && (
            <small>Existing measurements are reused first. New model work reports after the current search batch finishes.</small>
          )}
          {job.error && <small className="is-failed">{job.error.message}</small>}
        </div>
      )}
      {result?.status !== "completed" && reader.context.blocks.length > 0 && (
        <section className="lean-context-diff lean-context-preview" aria-labelledby="lean-context-heading">
          <header>
            <div>
              <h2 id="lean-context-heading">Original context</h2>
              <span className="lean-result-outcome">{plural(reader.context.blocks.length, "message")}</span>
            </div>
            {searchControls}
          </header>
          <div className="lean-context-document">
            <div className="inspect-prose context-prose lean-context-prose">
              {reader.context.blocks.map((block) => (
                <p
                  key={`${block.messageIndex}:${block.segmentId}`}
                  className="context-block lean-context-block"
                  data-message-index={block.messageIndex}
                  data-role={block.role}
                >{block.text}</p>
              ))}
            </div>
          </div>
        </section>
      )}
      {result?.status === "completed" && result.original && result.best && result.reduction && (
        <div className="lean-result">
          <section className="lean-context-diff" aria-labelledby="lean-context-heading">
            <header>
              <div className="lean-context-title">
                <div>
                  <h2 id="lean-context-heading">{contextView === "redline" ? "Original context" : "Context to keep"}</h2>
                  <nav className="lean-context-views" aria-label="Lean context view">
                    <button type="button" aria-pressed={contextView === "redline"} onClick={() => setContextView("redline")}>Redline</button>
                    <button type="button" aria-pressed={contextView === "kept"} onClick={() => setContextView("kept")}>Kept only</button>
                  </nav>
                </div>
                <span className="lean-result-outcome">{removedPromptTokens > 0
                  ? generatedAnswerChanged
                    ? `${removedPromptTokens} prompt tokens removed · answer changed`
                    : generatedAnswerPreserved
                      ? `${removedPromptTokens} prompt tokens removed · answer preserved`
                      : `${removedPromptTokens} prompt tokens removed · fresh check unavailable`
                  : "Nothing was removed"}</span>
              </div>
              <div className="lean-result-meta">
                <dl>
                  <div>
                    <dt>Prompt tokens</dt>
                    <dd>{result.original.renderedPromptTokenCost} <i>→</i> {result.best.renderedPromptTokenCost}</dd>
                  </div>
                  <div>
                    <dt>Context removed</dt>
                    <dd>{((1 - result.reduction.retainedPromptTokenFraction) * 100).toFixed(0)}%</dd>
                  </div>
                  <div>
                    <dt>Answer</dt>
                    <dd className={generatedAnswerPreserved ? "is-match" : generatedAnswerChanged ? "is-changed" : ""}>
                      {generationCheckLabel}
                    </dd>
                  </div>
                </dl>
              </div>
            </header>
            <div className="lean-context-strip" aria-label={`${result.reduction.removedSourceCount} context spans removed and ${result.reduction.retainedSourceCount} retained`}>
              {leanSources.filter((source) => contextView === "redline" || retainedSourceIds.has(source.sourceId)).map((source) => (
                <i
                  key={source.sourceId}
                  className={removedSourceIds.has(source.sourceId) ? "is-removed" : "is-retained"}
                  style={{ flexGrow: Math.max(1, codePointLength(sourceRecordedText(reader, source))) }}
                  title={`${removedSourceIds.has(source.sourceId) ? "Removed" : "Kept"}: ${excerpt(sourceRecordedText(reader, source), 80)}`}
                />
              ))}
            </div>
            <div className="lean-context-document">
              <div className="inspect-prose context-prose lean-context-prose">
                {displayedLeanContextBlocks.map((block) => (
                  <p
                    key={`${block.messageIndex}:${block.segmentId}`}
                    className="context-block lean-context-block"
                    data-message-index={block.messageIndex}
                    data-role={block.role}
                  >
                    {block.segments.map((segment) => {
                      const segmentRemoved = segment.annotationIds.some((sourceId) => removedSourceIds.has(sourceId));
                      const segmentRetained = segment.annotationIds.some((sourceId) => retainedSourceIds.has(sourceId));
                      if (contextView === "kept" && segmentRemoved && !segmentRetained) return null;
                      if (segmentRemoved && !segmentRetained) {
                        return <del key={`${segment.start}:${segment.end}`} className="lean-source is-removed">{segment.text}</del>;
                      }
                      if (segmentRetained) {
                        return <mark key={`${segment.start}:${segment.end}`} className="lean-source is-retained">{segment.text}</mark>;
                      }
                      return <span key={`${segment.start}:${segment.end}`}>{segment.text}</span>;
                    })}
                  </p>
                ))}
              </div>
            </div>
          </section>
          {beforeAnswer !== undefined && afterAnswer !== undefined && (
            <section className="lean-answer-comparison" aria-labelledby="lean-answer-comparison-heading">
              <header>
                <div>
                  <h3 id="lean-answer-comparison-heading">Answer comparison</h3>
                  <span className="lean-result-outcome">Full context → lean context</span>
                </div>
                <span className={`lean-answer-method ${hasFreshGenerationComparison
                  ? generationComparison?.exactMatch ? "is-match" : "is-changed"
                  : "is-unavailable"}`}>
                  {hasFreshGenerationComparison
                    ? generationComparison?.exactMatch ? "Exact match" : "Answer changed"
                    : "Fresh generation unavailable"}
                </span>
              </header>
              {generatedAnswerPreserved ? (
                <article className="lean-answer-exact">
                  <div className="lean-answer-equation" aria-label="Before and after answers are an exact match">
                    <span><b>Before</b> Full context · {result.original.renderedPromptTokenCost} prompt tokens</span>
                    <i aria-hidden="true">=</i>
                    <span><b>After</b> Lean context · {result.best.renderedPromptTokenCost} prompt tokens</span>
                  </div>
                  <blockquote>{beforeAnswer}</blockquote>
                </article>
              ) : (
                <>
                  <div className="lean-answer-grid">
                    <article className="lean-answer-card is-before">
                      <header>
                        <span>Before</span>
                        <strong>{hasFreshGenerationComparison ? "Full context · regenerated" : "Full context · recorded"}</strong>
                      </header>
                      <blockquote>
                        {hasFreshGenerationComparison
                          ? <LeanAnswerDiffText text={beforeAnswer} other={afterAnswer} kind="before" />
                          : beforeAnswer}
                      </blockquote>
                      <footer>
                        {result.original.renderedPromptTokenCost} prompt tokens
                        {hasFreshGenerationComparison
                          ? ` · greedy · ${generationComparison?.maxNewTokens} token cap`
                          : " · recorded answer"}
                      </footer>
                    </article>
                    <span className="lean-answer-arrow" aria-hidden="true">→</span>
                    <article className={`lean-answer-card is-after${generatedAnswerChanged ? " is-invalid" : ""}`}>
                      <header>
                        <span>After</span>
                        <strong>{hasFreshGenerationComparison ? "Lean context · regenerated" : "Lean context · rescored"}</strong>
                      </header>
                      <blockquote>
                        {hasFreshGenerationComparison
                          ? <LeanAnswerDiffText text={afterAnswer} other={beforeAnswer} kind="after" />
                          : afterAnswer}
                      </blockquote>
                      <footer>
                        {result.best.renderedPromptTokenCost} prompt tokens
                        {hasFreshGenerationComparison ? " · greedy" : ""}
                        {result.best.loss
                          ? ` · +${result.best.loss.meanPositiveSupportLossNatsPerToken.toFixed(3)} mean loss`
                          : " · within tolerance"}
                      </footer>
                    </article>
                  </div>
                  <p className="lean-answer-note">
                    {hasFreshGenerationComparison
                      ? "This candidate passed the recorded-answer score tolerance, but removing the marked context changed a fresh generation. Treat it as a failed behavioral check, not a safe reduction."
                      : `Lean Context could not run the fresh generation check${generationComparison?.reason ? `: ${generationComparison.reason}` : "."} The cards fall back to the recorded answer and score comparison.`}
                  </p>
                </>
              )}
            </section>
          )}
          <details className="lean-method-details">
            <summary>Method &amp; evidence</summary>
            <dl>
              <div>
                <dt>Answer-change tolerance</dt>
                <dd>≤ 0.05 mean · ≤ 0.70 max token loss</dd>
              </div>
              <div>
                <dt>Search work</dt>
                <dd>{result.budget.usedNewExecutions} / {result.budget.maxNewExecutions} new executions · {plural(result.budget.reusedObservationCount, "measurement")} reused</dd>
              </div>
              <div>
                <dt>Best candidate loss</dt>
                <dd>{result.best.loss
                  ? `${result.best.loss.meanPositiveSupportLossNatsPerToken.toFixed(3)} mean · ${result.best.loss.maxTokenSupportLossNats.toFixed(3)} max nats`
                  : "Unavailable"}</dd>
              </div>
            </dl>
            <footer>
              <p>Best candidate found within this search budget—not proof of the mathematically smallest context. Measured in {result.elapsedSeconds.toFixed(1)}s.</p>
              {allowSearch && (
                <button type="button" disabled={action !== "idle"} onClick={() => { void start(); }}>
                  {action === "starting" ? "Starting…" : "Run search again"}
                </button>
              )}
            </footer>
          </details>
        </div>
      )}
      {result && result.status !== "completed" && (
        <p className="lean-unavailable">{result.reason ?? "Lean Context is unavailable for this run."}</p>
      )}
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

function LeanContextSurface({ route }: { route: InspectRoute }) {
  const readerKey = `lean:${route.runId}`;
  const readerResource = useResource(
    readerKey,
    useCallback(
      (signal: AbortSignal) => contextInvestigationApi.reader(
        route.runId,
        signal,
        { scoringMode: "fast" },
      ),
      [route.runId],
    ),
  );
  const reader = readerResource.phase === "ready" ? readerResource.value : undefined;
  return (
    <div className="inspect-surface lean-context-surface">
      <header className="inspect-heading">
        <div>
          <a
            className="back-link"
            href={routeHref({ surface: "runs", ...(route.conversationId ? { conversationId: route.conversationId } : {}) })}
          >← {route.conversationId ? "Conversation" : "Conversations"}</a>
          <h1>Lean Context</h1>
          <code>{shortId(route.runId)}</code>
        </div>
        {reader && reader.status !== "completed" && (
          <span className={`reader-status status-${reader.status.replace("_", "-")}`}>
            <i aria-hidden="true" />{statusLabel(reader.status)}
          </span>
        )}
      </header>
      <div className="inspect-layout">
        <section className="inspect-instrument lean-context-instrument" aria-label="Lean Context">
          <ReaderState resource={readerResource} />
          {reader && <LeanContextPanel reader={reader} route={route} />}
        </section>
      </div>
    </div>
  );
}

function ContextInspectSurface({ route }: { route: InspectRoute }) {
  const [liveReaderEntry, setLiveReaderEntry] = useState<{
    key: string;
    reader: ContextInvestigationReader;
  }>();
  const [view, setView] = useState<InvestigationView>("links");
  const scoringMode = route.scoringMode ?? "fast";
  const readerKey = `${route.runId}:${scoringMode}:${route.granularity ?? "default"}:${route.armBudget ?? "default"}`;
  const readerLoader = useCallback(
    (signal: AbortSignal) => contextInvestigationApi.reader(
      route.runId,
      signal,
      route.granularity && route.armBudget !== undefined ? {
        granularity: route.granularity,
        maxArms: route.armBudget,
        scoringMode,
      } : { scoringMode },
    ),
    [route.armBudget, route.granularity, route.runId, scoringMode],
  );
  const readerResource = useResource(readerKey, readerLoader);
  const liveReader = liveReaderEntry?.key === readerKey
    ? liveReaderEntry.reader
    : undefined;
  const reader = liveReader ?? (readerResource.phase === "ready" ? readerResource.value : undefined);
  const displayedResource: Resource<ContextInvestigationReader> = reader
    ? { phase: "ready", value: reader }
    : readerResource;
  useEffect(() => {
    setView("links");
  }, [route.runId]);
  const updateLiveReader = useCallback((updated: ContextInvestigationReader) => {
    setLiveReaderEntry({ key: readerKey, reader: updated });
  }, [readerKey]);
  const openLinks = useCallback((range: ActiveRange, sourceId?: string) => {
    setView("links");
    window.location.hash = routeHref({
      surface: "inspect",
      runId: route.runId,
      ...(route.conversationId ? { conversationId: route.conversationId } : {}),
      ...(route.scoringMode ? { scoringMode: route.scoringMode } : {}),
      ...(route.granularity && route.armBudget !== undefined ? {
        granularity: route.granularity,
        armBudget: route.armBudget,
      } : {}),
      answerStart: range.start,
      answerEnd: range.end,
      ...(sourceId ? { sourceId } : {}),
    });
  }, [route.armBudget, route.conversationId, route.granularity, route.runId, scoringMode]);
  return (
    <div className="inspect-surface">
      <header className="inspect-heading">
        <div>
          <a
            className="back-link"
            href={routeHref({ surface: "runs", ...(route.conversationId ? { conversationId: route.conversationId } : {}) })}
          >← {route.conversationId ? "Conversation" : "Conversations"}</a>
          <h1>Context Effects</h1>
          <code>{shortId(route.runId)}</code>
        </div>
        {reader && reader.status !== "completed" && (
          <span className={`reader-status status-${reader.status.replace("_", "-")}`}>
            <i aria-hidden="true" />{statusLabel(reader.status)}
          </span>
        )}
      </header>
      <div className="inspect-layout">
        <section className="inspect-instrument" aria-label="Influence">
          <ReaderState resource={displayedResource} />
          {reader && (
            <>
              {reader.reason && reader.status !== "completed" && (
                <p className={`reader-reason status-${reader.status.replace("_", "-")}`} role={reader.status === "stale" || reader.status === "unavailable" ? "alert" : "status"}>
                  {reader.reason}
                </p>
              )}
              <nav className="investigation-views" aria-label="Investigation view">
                <button type="button" aria-current={view === "links" ? "page" : undefined} onClick={() => setView("links")}>Attribution map</button>
                <button type="button" aria-current={view === "overview" ? "page" : undefined} onClick={() => setView("overview")}>Highlights</button>
              </nav>
              {reader.answer.status === "available" || reader.context.blocks.length > 0 ? (
                <>
                  <div hidden={view !== "links"}>
                    <InspectReader reader={reader} route={route} />
                  </div>
                  <div className="investigation-summary" hidden={view !== "overview"}>
                    <EfficiencyOverview
                      reader={reader}
                      onOpenSource={(sourceId, range) => openLinks(range, sourceId)}
                      onOpenAnswer={(range) => openLinks(range)}
                    />
                    <RunDetailsDisclosure runId={route.runId} />
                    <details className="stability-disclosure">
                      <summary>
                        <span>
                          <strong>Test whether influence changes across context combinations</strong>
                          <small>Optional advanced experiment · uses an execution budget</small>
                        </span>
                        <span>Stability test</span>
                      </summary>
                      <InteractionsPanel reader={reader} route={route} />
                    </details>
                  </div>
                </>
              ) : <p className="inspect-state is-unavailable">Recorded prose unavailable.</p>}
              {reader.capabilities.contextEffects === "measurable" && (
                <EffectsPanel
                  key={readerKey}
                  reader={reader}
                  route={route}
                  onReaderUpdate={updateLiveReader}
                />
              )}
            </>
          )}
        </section>
      </div>
    </div>
  );
}

export function InspectSurface({ route }: { route: InspectRoute }) {
  return route.view === "lean-context"
    ? <LeanContextSurface route={route} />
    : <ContextInspectSurface route={route} />;
}
