import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { EvidenceLanes, type EvidenceLanesProps } from "./EvidenceLanes";
import "./EvidenceDeck.css";

export const EVIDENCE_DECK_SECTIONS = ["evidence", "events", "performance", "lineage"] as const;

export type EvidenceDeckSection = (typeof EVIDENCE_DECK_SECTIONS)[number];

export type EvidenceDeckAvailabilityState =
  | "available"
  | "not_captured"
  | "not_measured"
  | "computing"
  | "failed"
  | "unsupported"
  | "privacy_limited";

/**
 * A section without content must name its actual evidence state. This intentionally keeps a known
 * empty event list (`available`) distinct from an artifact that was never captured (`not_captured`).
 */
export interface EvidenceDeckAvailability {
  state: EvidenceDeckAvailabilityState;
  detail?: string;
}

export interface EvidenceDeckPanel {
  content?: ReactNode;
  availability?: EvidenceDeckAvailability;
  /** Overrides the default, evidence-specific empty state when `availability.state` is available. */
  emptyMessage?: string;
}

export interface EvidenceDeckProps {
  /** Controlled active tab. Omit to let the deck keep its own tab state. */
  selectedSection?: EvidenceDeckSection;
  onSelectedSectionChange?: (section: EvidenceDeckSection) => void;
  /** Controlled collapsed state. Omit to let the deck keep its own collapsed state. */
  collapsed?: boolean;
  defaultCollapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
  /** Controlled deck height in CSS pixels while expanded. */
  height?: number;
  defaultHeight?: number;
  minHeight?: number;
  maxHeight?: number;
  onHeightChange?: (height: number) => void;
  /** A first-class embedding path that preserves EvidenceLanes' controlled token/range callbacks. */
  evidenceLanes?: EvidenceLanesProps;
  /** Optional content or honest availability state for each deck section. */
  sections?: Partial<Record<EvidenceDeckSection, EvidenceDeckPanel>>;
  className?: string;
  title?: string;
}

const DEFAULT_HEIGHT = 360;
const DEFAULT_MIN_HEIGHT = 220;
const DEFAULT_MAX_HEIGHT = 760;
const RESIZE_STEP = 32;

const sectionMeta: Record<EvidenceDeckSection, { label: string; empty: string }> = {
  evidence: {
    label: "Evidence",
    empty: "No additional evidence panel was supplied for this run.",
  },
  events: {
    label: "Events",
    empty: "No semantic run events were recorded.",
  },
  performance: {
    label: "Performance",
    empty: "No recorded performance evidence is available for this run.",
  },
  lineage: {
    label: "Lineage",
    empty: "No parent, child, retry, or branch relationship was recorded for this run.",
  },
};

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, Math.round(value)));
}

function availabilityTitle(state: EvidenceDeckAvailabilityState) {
  return state.replaceAll("_", " ").toUpperCase();
}

function defaultAvailabilityDetail(section: EvidenceDeckSection, state: EvidenceDeckAvailabilityState) {
  const label = sectionMeta[section].label.toLowerCase();
  switch (state) {
    case "available":
      return sectionMeta[section].empty;
    case "not_captured":
      return `${sectionMeta[section].label} evidence was not captured for this run.`;
    case "not_measured":
      return `${sectionMeta[section].label} evidence is eligible but has not been measured.`;
    case "computing":
      return `${sectionMeta[section].label} evidence is currently being computed.`;
    case "failed":
      return `${sectionMeta[section].label} evidence could not be produced; inspect its error receipt.`;
    case "unsupported":
      return `This model or runtime does not support ${label} evidence.`;
    case "privacy_limited":
      return `${sectionMeta[section].label} content is intentionally unavailable under this run's privacy setting.`;
  }
}

function SectionState({ section, panel }: { section: EvidenceDeckSection; panel?: EvidenceDeckPanel }) {
  const availability = panel?.availability ?? { state: "not_captured" as const };
  const detail = availability.detail
    ?? (availability.state === "available" ? panel?.emptyMessage : undefined)
    ?? defaultAvailabilityDetail(section, availability.state);
  return (
    <div className={`evidence-deck-state is-${availability.state}`}>
      <strong>{availabilityTitle(availability.state)}</strong>
      <p>{detail}</p>
    </div>
  );
}

/**
 * Lower Debug deck with independently controlled tab, collapsed, and height state. It owns only the
 * deck chrome: callers retain all run, selection, loading, and measurement state in their feature.
 */
export function EvidenceDeck({
  selectedSection,
  onSelectedSectionChange,
  collapsed,
  defaultCollapsed = false,
  onCollapsedChange,
  height,
  defaultHeight = DEFAULT_HEIGHT,
  minHeight = DEFAULT_MIN_HEIGHT,
  maxHeight = DEFAULT_MAX_HEIGHT,
  onHeightChange,
  evidenceLanes,
  sections = {},
  className,
  title = "Evidence deck",
}: EvidenceDeckProps) {
  const minimumHeight = Math.max(160, minHeight);
  const maximumHeight = Math.max(minimumHeight, maxHeight);
  const [internalSection, setInternalSection] = useState<EvidenceDeckSection>("evidence");
  const [internalCollapsed, setInternalCollapsed] = useState(defaultCollapsed);
  const [internalHeight, setInternalHeight] = useState(() => clamp(defaultHeight, minimumHeight, maximumHeight));
  const cleanupResize = useRef<(() => void) | null>(null);
  const id = useId().replaceAll(":", "");
  const activeSection = selectedSection ?? internalSection;
  const isCollapsed = collapsed ?? internalCollapsed;
  const deckHeight = clamp(height ?? internalHeight, minimumHeight, maximumHeight);
  const activePanel = sections[activeSection];

  useEffect(() => () => cleanupResize.current?.(), []);

  function changeSection(section: EvidenceDeckSection) {
    if (selectedSection === undefined) setInternalSection(section);
    onSelectedSectionChange?.(section);
  }

  function changeCollapsed(next: boolean) {
    if (collapsed === undefined) setInternalCollapsed(next);
    onCollapsedChange?.(next);
  }

  function changeHeight(next: number) {
    const nextHeight = clamp(next, minimumHeight, maximumHeight);
    if (height === undefined) setInternalHeight(nextHeight);
    onHeightChange?.(nextHeight);
  }

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") nextIndex = (index + 1) % EVIDENCE_DECK_SECTIONS.length;
    if (event.key === "ArrowLeft") nextIndex = (index - 1 + EVIDENCE_DECK_SECTIONS.length) % EVIDENCE_DECK_SECTIONS.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = EVIDENCE_DECK_SECTIONS.length - 1;
    if (nextIndex === null) return;
    event.preventDefault();
    const nextSection = EVIDENCE_DECK_SECTIONS[nextIndex];
    changeSection(nextSection);
    document.getElementById(`${id}-${nextSection}-tab`)?.focus();
  }

  function handleResizeKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (isCollapsed) return;
    if (event.key === "ArrowUp") {
      event.preventDefault();
      changeHeight(deckHeight + RESIZE_STEP);
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      changeHeight(deckHeight - RESIZE_STEP);
    } else if (event.key === "Home") {
      event.preventDefault();
      changeHeight(minimumHeight);
    } else if (event.key === "End") {
      event.preventDefault();
      changeHeight(maximumHeight);
    }
  }

  function handleResizePointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (isCollapsed || event.button !== 0) return;
    event.preventDefault();
    cleanupResize.current?.();
    const startY = event.clientY;
    const startHeight = deckHeight;
    const onMove = (moveEvent: PointerEvent) => changeHeight(startHeight + startY - moveEvent.clientY);
    const onEnd = () => cleanup();
    const cleanup = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
      if (cleanupResize.current === cleanup) cleanupResize.current = null;
    };
    cleanupResize.current = cleanup;
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onEnd, { once: true });
    window.addEventListener("pointercancel", onEnd, { once: true });
  }

  const hasPanelContent = activePanel?.content !== undefined && activePanel.content !== null;
  const panelContent = activeSection === "evidence" && !hasPanelContent && evidenceLanes
    ? <EvidenceLanes {...evidenceLanes} />
    : hasPanelContent
      ? activePanel.content
      : <SectionState section={activeSection} panel={activePanel} />;

  return (
    <section
      className={[
        "evidence-deck",
        `is-${activeSection}`,
        isCollapsed ? "is-collapsed" : "",
        className,
      ].filter(Boolean).join(" ")}
      style={isCollapsed ? undefined : { height: `${deckHeight}px` }}
      aria-label={title}
    >
      <div
        className="evidence-deck-resize-handle"
        role="separator"
        aria-label="Resize evidence deck"
        aria-orientation="horizontal"
        aria-valuemin={minimumHeight}
        aria-valuemax={maximumHeight}
        aria-valuenow={deckHeight}
        aria-disabled={isCollapsed || undefined}
        tabIndex={isCollapsed ? -1 : 0}
        onKeyDown={handleResizeKeyDown}
        onPointerDown={handleResizePointerDown}
      >
        <span aria-hidden="true" />
      </div>
      <header className="evidence-deck-header">
        <div className="evidence-deck-title">
          <span>DEBUG / LOWER DETAIL</span>
          <h2>{title}</h2>
        </div>
        <div className="evidence-deck-controls">
          <div className="evidence-deck-tabs" role="tablist" aria-label="Evidence deck sections">
            {EVIDENCE_DECK_SECTIONS.map((section, index) => (
              <button
                type="button"
                role="tab"
                id={`${id}-${section}-tab`}
                aria-controls={`${id}-${section}-panel`}
                aria-selected={activeSection === section}
                tabIndex={activeSection === section ? 0 : -1}
                onClick={() => changeSection(section)}
                onKeyDown={(event) => handleTabKeyDown(event, index)}
                key={section}
              >{sectionMeta[section].label}</button>
            ))}
          </div>
          <button
            type="button"
            className="evidence-deck-collapse"
            aria-expanded={!isCollapsed}
            aria-controls={`${id}-${activeSection}-panel`}
            onClick={() => changeCollapsed(!isCollapsed)}
          >{isCollapsed ? "Expand deck" : "Collapse deck"}</button>
        </div>
      </header>
      {!isCollapsed && (
        <div
          className="evidence-deck-panel"
          id={`${id}-${activeSection}-panel`}
          role="tabpanel"
          aria-labelledby={`${id}-${activeSection}-tab`}
        >
          {panelContent}
        </div>
      )}
    </section>
  );
}
