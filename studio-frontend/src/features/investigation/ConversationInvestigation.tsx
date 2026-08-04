import { useCallback, useEffect, useRef, useState } from "react";
import { findingStatusMeta } from "../lens/DiagnosisRepair";
import {
  describeSessionTraceError,
  loadSessionTracePage,
  SessionTraceNotFoundError,
  type SessionTracePage,
  type TraceBranchChild,
  type TraceCandidate,
  type TraceCandidateKind,
  type TraceDifference,
  type TraceRuleRegistryEntry,
  type TraceSession,
  type TraceTotals,
  type TraceTurn,
  type TraceTurnComparison,
} from "../../data/sessionTrace";
import type { RepairEvidence, RepairFinding, RepairFindingStatus } from "../../data/diagnosisRepair";
import { useTopbar } from "../../panels/topbar";

/**
 * F3 -- the conversation investigation view. Renders `clozn.session-trace.v1` (F2) with D1's five-value
 * evidence vocabulary intact and never collapsed (see `StatusCountsRow` below).
 *
 * WHY THIS CANNOT BE MISTAKEN FOR A CHAT REPLAY
 * ---------------------------------------------------
 * `GET /sessions/<id>/trace` never returns a turn's full message text -- only `prompt_summary`/
 * `response_summary` (≤90 chars, `clozn.runs.summaries._summ`). There is no wire data this view COULD
 * render as a scrollback of full turns even if it wanted to; every turn row is structurally an evidence
 * summary, never a reply. Layout reinforces this on top of that structural fact: dense ledger rows in
 * the same "instrument" language as Runs/Lens, no bubbles, no avatars, an explicit boundary banner. C4's
 * `AskAnotherQuestion.tsx` made the same visual choice for the same reason (see its own doc comment) --
 * this is the first place that promise gets a real chat-shaped neighbor to be un-mistaken for.
 *
 * PAGINATION IS NEVER HIDDEN
 * ------------------------------
 * `turns` accumulates only pages this component explicitly fetched -- one on mount, more only on an
 * explicit "load more" click or an explicit "jump to this candidate" action (see `locate`). Whenever
 * `page.next_cursor` was non-null on the LAST fetched page, the timeline states outright that it is
 * PARTIAL (`investigation-partial-banner`) -- never silently rendered as if it were the whole session.
 *
 * "FIRST SUSPICIOUS TURN" NEVER INVENTED
 * -------------------------------------------
 * `firstWentWrongCandidates` comes straight from the backend's own deterministic, session-wide scan
 * (`clozn.runs.session_trace._first_went_wrong_candidates`) -- it is already complete regardless of how
 * many pages THIS component has fetched (the backend scans the whole session on every call). This
 * component adds no inference on top: a candidate is rendered only when the backend actually returned
 * one for that kind; the absence of a kind renders an explicit "no candidate found" sentence, never a
 * blank space and never a fabricated arrow. See `CandidateCard`.
 */

export interface ConversationInvestigationProps {
  sessionId: string;
}

const TURNS_PER_PAGE = 50;
// 20 additional page fetches at TURNS_PER_PAGE=50 covers store.KEEP (1000 runs store-wide) in the worst
// case -- see clozn/runs/store.py's own retention cap. Bounded and user-triggered only (a candidate
// click, never eager on mount) -- see the module docstring's "PAGINATION IS NEVER HIDDEN" section.
const MAX_LOCATE_PAGES = 20;
const FINDING_STATUS_ORDER: readonly RepairFindingStatus[] =
  ["finding", "not_observed", "unavailable", "pending", "suppressed"];
const CANDIDATE_KIND_ORDER: readonly TraceCandidateKind[] =
  ["first_finding", "first_settings_drift", "first_failed_run"];

function shortId(id: string): string {
  return id.length > 6 ? id.slice(-6) : id;
}

function formatEpochSeconds(ts: number | undefined): string {
  if (!ts) return "—";
  const date = new Date(ts * 1000);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function formatDuration(ms: number | undefined): string {
  if (ms == null) return "—";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`;
}

function evidenceLabel(item: RepairEvidence): string {
  return item.kind === "field" ? `${item.path} = ${JSON.stringify(item.value)}` : `span ${item.addressId.slice(-8)}`;
}

function evidenceHref(runId: string, item: RepairEvidence): string | undefined {
  return item.kind === "text_span"
    ? `#/runs/${encodeURIComponent(runId)}/span-addresses#${item.addressId}`
    : undefined;
}

function differenceLabel(d: TraceDifference): string {
  const hasValues = d.valueA !== undefined || d.valueB !== undefined;
  return hasValues
    ? `${d.dimension} (${d.kind}): ${JSON.stringify(d.valueA)} → ${JSON.stringify(d.valueB)}`
    : `${d.dimension} (${d.kind})`;
}

function candidateKindLabel(kind: TraceCandidateKind): string {
  switch (kind) {
    case "first_finding": return "First diagnostic finding";
    case "first_settings_drift": return "First identity/setting drift";
    case "first_failed_run": return "First failed or cancelled run";
    default: {
      const exhaustive: never = kind;
      return exhaustive;
    }
  }
}

function candidateAbsenceText(kind: TraceCandidateKind): string {
  switch (kind) {
    case "first_finding":
      return "No turn in this session triggered a diagnostic rule finding.";
    case "first_settings_drift":
      return "No identity or generation-setting drift was found between consecutive turns in this session.";
    case "first_failed_run":
      return "No turn in this session recorded an error or a cancelled/failed termination.";
    default: {
      const exhaustive: never = kind;
      return exhaustive;
    }
  }
}

// ------------------------------------------------------------------------------------------- accumulator

interface Accumulated {
  session: TraceSession | null;
  turns: TraceTurn[];
  branchesByParent: Map<string, TraceBranchChild[]>;
  candidates: TraceCandidate[];
  ruleRegistry: TraceRuleRegistryEntry[];
  totals: TraceTotals | null;
  nextCursor: string | null;
  complete: boolean;
  pagesLoaded: number;
}

const EMPTY_ACCUMULATED: Accumulated = {
  session: null, turns: [], branchesByParent: new Map(), candidates: [], ruleRegistry: [], totals: null,
  nextCursor: null, complete: false, pagesLoaded: 0,
};

function applyPage(prev: Accumulated, page: SessionTracePage): Accumulated {
  // `branches` is disjoint per page by construction (each linear turn belongs to exactly one page --
  // see session_trace.py's `_branches_for`), so a plain accumulate-by-parent is correct, never a merge.
  const branchesByParent = new Map(prev.branchesByParent);
  for (const branch of page.branches) branchesByParent.set(branch.parentRunId, branch.children);
  return {
    session: page.session,
    turns: [...prev.turns, ...page.turns],
    branchesByParent,
    // Session-wide and freshly recomputed on every call (see session_trace.py's own docstring) -- the
    // MOST RECENT page's value is always the authoritative one, never merged with an older page's.
    candidates: page.firstWentWrongCandidates,
    ruleRegistry: page.diagnosticRuleRegistry,
    totals: page.totalsThroughThisPage,
    nextCursor: page.page.nextCursor,
    complete: page.page.nextCursor == null,
    pagesLoaded: prev.pagesLoaded + 1,
  };
}

// -------------------------------------------------------------------------------------------- sub-parts

function StatusCountsRow({ counts }: { counts: Record<RepairFindingStatus, number> }) {
  return (
    <ul className="investigation-status-counts" aria-label="Diagnostic rule status breakdown for this turn">
      {FINDING_STATUS_ORDER.map((status) => {
        const meta = findingStatusMeta(status);
        return (
          <li key={status}>
            <span className={`diagnosis-repair-status ${meta.className}`}>
              {counts[status]} {status.replaceAll("_", " ").toUpperCase()}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

function FindingCard({ runId, finding }: { runId: string; finding: Extract<RepairFinding, { status: "finding" }> }) {
  return (
    <article className="investigation-finding">
      <header>
        <b>{finding.ruleId}</b>
        <span>{finding.ruleName.replaceAll("_", " ")}</span>
      </header>
      <p className="investigation-finding-severity">
        {finding.severity.toUpperCase()} SEVERITY · {finding.confidence.replaceAll("_", " ").toUpperCase()} CONFIDENCE
      </p>
      <p>{finding.summary}</p>
      {finding.evidence.length > 0 && (
        <ul className="investigation-finding-evidence" aria-label={`Evidence for ${finding.ruleId}`}>
          {finding.evidence.map((item, index) => {
            const href = evidenceHref(runId, item);
            return <li key={index}>{href ? <a href={href}>{evidenceLabel(item)}</a> : evidenceLabel(item)}</li>;
          })}
        </ul>
      )}
    </article>
  );
}

function TurnComparisonBlock({ comparison }: { comparison?: TraceTurnComparison }) {
  if (!comparison) {
    return (
      <p className="investigation-comparison-note">
        This is the first recorded turn of the session -- there is no earlier turn to compare against.
      </p>
    );
  }
  if (!comparison.available) {
    return (
      <p className="investigation-comparison-note">
        Comparison against the previous turn is unavailable{comparison.reason ? ` -- ${comparison.reason}` : ""}.
      </p>
    );
  }
  const { settingsChanges, contextChanges, classifications } = comparison;
  const nothingChanged = settingsChanges.length === 0 && contextChanges.newSegments.length === 0
    && contextChanges.droppedSegments.length === 0 && contextChanges.otherContextDifferences.length === 0
    && classifications.length === 0;
  if (nothingChanged) {
    return (
      <p className="investigation-comparison-note">
        No identity, setting, or context change was found against the previous turn.
      </p>
    );
  }
  return (
    <div className="investigation-comparison">
      {settingsChanges.length > 0 && (
        <div>
          <b>SETTINGS CHANGED</b>
          <ul>{settingsChanges.map((d, index) => <li key={index}>{differenceLabel(d)}</li>)}</ul>
        </div>
      )}
      {(contextChanges.newSegments.length > 0 || contextChanges.droppedSegments.length > 0
        || contextChanges.carriedForwardSegmentCount > 0) && (
        <div>
          <b>CONTEXT CHANGES</b>
          {contextChanges.newSegments.length > 0 && (
            <p>+{contextChanges.newSegments.length} new segment(s): {contextChanges.newSegments.map((d) => d.dimension).join(", ")}</p>
          )}
          {contextChanges.droppedSegments.length > 0 && (
            <p>−{contextChanges.droppedSegments.length} dropped segment(s): {contextChanges.droppedSegments.map((d) => d.dimension).join(", ")}</p>
          )}
          <p>{contextChanges.carriedForwardSegmentCount} segment(s) carried forward unchanged.</p>
        </div>
      )}
      {contextChanges.otherContextDifferences.length > 0 && (
        <div>
          <b>OTHER CONTEXT DIFFERENCES</b>
          <ul>{contextChanges.otherContextDifferences.map((d, index) => <li key={index}>{differenceLabel(d)}</li>)}</ul>
        </div>
      )}
      {classifications.length > 0 && (
        <div>
          <b>CLASSIFICATIONS</b>
          <ul>
            {classifications.map((c, index) => (
              <li key={index}><b>{c.classification.replaceAll("_", " ")}</b> — {c.summary}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function TurnDetail({
  turn, branchChildren, compareA, compareB, onStage,
}: {
  turn: TraceTurn;
  branchChildren: TraceBranchChild[];
  compareA: string;
  compareB: string;
  onStage: (runId: string, slot: "a" | "b") => void;
}) {
  const highlights = turn.diagnosticHighlights;
  return (
    <div className="investigation-turn-detail">
      <section aria-label={`Diagnostics for turn ${shortId(turn.runId)}`}>
        <b>DIAGNOSTICS</b>
        <StatusCountsRow counts={highlights.statusCounts} />
        {highlights.findings.length > 0
          ? highlights.findings.map((finding) => (
            <FindingCard key={finding.ruleId} runId={turn.runId} finding={finding} />
          ))
          : (
            <p className="investigation-comparison-note">
              No rule in D1&apos;s registry reported a finding for this turn.
            </p>
          )}
        <a href={`#/runs/${encodeURIComponent(turn.runId)}`}>Open this turn in Debug</a>
      </section>

      <section aria-label={`Changes since the previous turn for ${shortId(turn.runId)}`}>
        <b>CHANGES SINCE THE PREVIOUS TURN</b>
        <TurnComparisonBlock comparison={turn.turnComparison} />
      </section>

      {branchChildren.length > 0 && (
        <div className="investigation-branches-inline">
          <b>BRANCHES OFF THIS TURN ({branchChildren.length})</b>
          <ul>
            {branchChildren.map((child) => (
              <li key={child.id}>
                <span>{shortId(child.id)}</span>
                <span>{(child.source ?? "—").toUpperCase()}</span>
                <span>{child.promptSummary || child.responseSummary || "—"}</span>
                <button type="button" onClick={() => onStage(child.id, compareA ? "b" : "a")}>
                  STAGE {compareA ? "B" : "A"}
                </button>
                <a href={`#/runs/${encodeURIComponent(child.id)}`}>OPEN</a>
              </li>
            ))}
          </ul>
        </div>
      )}

      <dl className="investigation-turn-facts">
        <div><dt>Client</dt><dd>{turn.client}</dd></div>
        <div><dt>Finish reason</dt><dd>{turn.finishReason ?? "—"}</dd></div>
        <div><dt>Error</dt><dd>{turn.error ?? "—"}</dd></div>
        <div><dt>Duration</dt><dd>{formatDuration(turn.durationMs)}</dd></div>
        <div><dt>Prompt tokens</dt><dd>{turn.contextUsage?.promptTokens ?? "—"}</dd></div>
        <div><dt>Generated tokens</dt><dd>{turn.contextUsage?.generatedTokens ?? "—"}</dd></div>
        <div><dt>Context window</dt><dd>{turn.contextUsage?.contextWindowTokens ?? "—"}</dd></div>
        <div><dt>Cumulative turns</dt><dd>{turn.cumulative.turnCount}</dd></div>
        <div>
          <dt>Cumulative tokens</dt>
          <dd>{turn.cumulative.promptTokensTotal + turn.cumulative.generatedTokensTotal}</dd>
        </div>
      </dl>

      <div className="investigation-turn-actions">
        <a href={`#/runs/${encodeURIComponent(turn.runId)}/lens`}>OPEN IN LENS</a>
        <a href={`#/runs/${encodeURIComponent(turn.runId)}/diagnostics`}>OPEN DIAGNOSTICS</a>
        <button
          type="button"
          className={compareA === turn.runId ? "is-a is-active" : "is-a"}
          aria-pressed={compareA === turn.runId}
          onClick={() => onStage(turn.runId, "a")}
        >STAGE A</button>
        <button
          type="button"
          className={compareB === turn.runId ? "is-b is-active" : "is-b"}
          aria-pressed={compareB === turn.runId}
          onClick={() => onStage(turn.runId, "b")}
        >STAGE B</button>
      </div>
    </div>
  );
}

function TurnRow({
  turn, expanded, onToggle, branchChildren, compareA, compareB, onStage,
}: {
  turn: TraceTurn;
  expanded: boolean;
  onToggle: () => void;
  branchChildren: TraceBranchChild[];
  compareA: string;
  compareB: string;
  onStage: (runId: string, slot: "a" | "b") => void;
}) {
  const counts = turn.diagnosticHighlights.statusCounts;
  return (
    <li className={`investigation-turn ${turn.redacted ? "is-redacted" : ""}`} id={`investigation-turn-${turn.runId}`}>
      <button type="button" className="investigation-turn-summary" onClick={onToggle} aria-expanded={expanded}>
        <time>{formatEpochSeconds(turn.recordedTs)}</time>
        <span className="investigation-turn-id">{shortId(turn.runId)}</span>
        <span className="investigation-turn-source">{turn.source.toUpperCase()}</span>
        <span className="investigation-turn-copy">
          <strong>{turn.promptSummary || "—"}</strong>
          <small>{turn.responseSummary || "—"}</small>
        </span>
        <span className="investigation-turn-model">{turn.model}</span>
        <span className="investigation-turn-flags">
          {turn.error && <span className="investigation-flag is-error">ERROR</span>}
          {turn.finishReason === "length" && <span className="investigation-flag is-truncated">TRUNCATED</span>}
          {turn.redacted && <span className="investigation-flag is-redacted">REDACTED</span>}
          {counts.finding > 0 && (
            <span className="diagnosis-repair-status is-finding">
              {counts.finding} FINDING{counts.finding === 1 ? "" : "S"}
            </span>
          )}
          {branchChildren.length > 0 && (
            <span className="investigation-flag is-branch">
              {branchChildren.length} BRANCH{branchChildren.length === 1 ? "" : "ES"}
            </span>
          )}
        </span>
        <span className="investigation-turn-caret" aria-hidden="true">{expanded ? "▾" : "▸"}</span>
      </button>
      {expanded && (
        <TurnDetail
          turn={turn}
          branchChildren={branchChildren}
          compareA={compareA}
          compareB={compareB}
          onStage={onStage}
        />
      )}
    </li>
  );
}

function CandidateCard({
  kind, candidate, locating, onLocate,
}: {
  kind: TraceCandidateKind;
  candidate?: TraceCandidate;
  locating: boolean;
  onLocate: (runId: string) => void;
}) {
  return (
    <li className={`investigation-candidate ${candidate ? "has-candidate" : ""}`}>
      <header>
        <span className="investigation-candidate-kind">{candidateKindLabel(kind).toUpperCase()}</span>
        {candidate && <span className="investigation-candidate-when">{formatEpochSeconds(candidate.recordedTs)}</span>}
      </header>
      <p>{candidate ? candidate.summary : candidateAbsenceText(kind)}</p>
      {candidate && (
        <button type="button" disabled={locating} onClick={() => onLocate(candidate.runId)}>
          {locating ? "LOCATING…" : `JUMP TO ${shortId(candidate.runId)}`}
        </button>
      )}
    </li>
  );
}

// -------------------------------------------------------------------------------------------------- root

type Phase = "loading" | "ready" | "error";

export function ConversationInvestigation({ sessionId }: ConversationInvestigationProps) {
  const [phase, setPhase] = useState<Phase>("loading");
  const [errorInfo, setErrorInfo] = useState<{ notFound: boolean; message: string } | null>(null);
  const [data, setData] = useState<Accumulated>(EMPTY_ACCUMULATED);
  const [loadingMore, setLoadingMore] = useState(false);
  const [loadMoreError, setLoadMoreError] = useState<string | null>(null);
  const [locatingRunId, setLocatingRunId] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [compareA, setCompareA] = useState("");
  const [compareB, setCompareB] = useState("");

  // One id per SESSION SELECTION -- mirrors scope.tsx's own selectRun guard: a response for a session
  // this component has since navigated away from can never land on top of the session now showing.
  const requestIdRef = useRef(0);

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    setPhase("loading");
    setErrorInfo(null);
    setData(EMPTY_ACCUMULATED);
    setExpanded(new Set());
    setCompareA("");
    setCompareB("");
    setLoadMoreError(null);
    if (!sessionId) {
      setPhase("error");
      setErrorInfo({ notFound: false, message: "no session id was given" });
      return;
    }
    const controller = new AbortController();
    void loadSessionTracePage(sessionId, { limit: TURNS_PER_PAGE, signal: controller.signal }).then((page) => {
      if (requestIdRef.current !== requestId) return;
      setData((prev) => applyPage(prev, page));
      setPhase("ready");
    }).catch((error: unknown) => {
      if (requestIdRef.current !== requestId) return;
      if (error instanceof DOMException && error.name === "AbortError") return;
      setPhase("error");
      setErrorInfo({ notFound: error instanceof SessionTraceNotFoundError, message: describeSessionTraceError(error) });
    });
    return () => controller.abort();
  }, [sessionId]);

  const loadMore = useCallback(async () => {
    if (loadingMore || data.complete || !data.nextCursor) return;
    const requestId = requestIdRef.current;
    setLoadingMore(true);
    setLoadMoreError(null);
    try {
      const page = await loadSessionTracePage(sessionId, { cursor: data.nextCursor, limit: TURNS_PER_PAGE });
      if (requestIdRef.current !== requestId) return;
      setData((prev) => applyPage(prev, page));
    } catch (error) {
      if (requestIdRef.current !== requestId) return;
      setLoadMoreError(describeSessionTraceError(error));
    } finally {
      if (requestIdRef.current === requestId) setLoadingMore(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- data.nextCursor/data.complete are read fresh
  }, [sessionId, loadingMore, data.complete, data.nextCursor]);

  function scrollToTurn(runId: string) {
    setExpanded((current) => {
      const next = new Set(current);
      next.add(runId);
      return next;
    });
    requestAnimationFrame(() => {
      document.getElementById(`investigation-turn-${runId}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  async function locate(runId: string) {
    if (data.turns.some((turn) => turn.runId === runId)) {
      scrollToTurn(runId);
      return;
    }
    if (data.complete) return; // genuinely not part of this session's trace -- nothing left to load
    const requestId = requestIdRef.current;
    setLocatingRunId(runId);
    setLoadMoreError(null);
    let cursor = data.nextCursor;
    let iterations = 0;
    let found = false;
    try {
      while (cursor && iterations < MAX_LOCATE_PAGES && requestIdRef.current === requestId) {
        const page = await loadSessionTracePage(sessionId, { cursor, limit: TURNS_PER_PAGE });
        if (requestIdRef.current !== requestId) return;
        setData((prev) => applyPage(prev, page));
        iterations += 1;
        cursor = page.page.nextCursor;
        if (page.turns.some((turn) => turn.runId === runId)) {
          found = true;
          break;
        }
      }
    } catch (error) {
      if (requestIdRef.current !== requestId) return;
      setLoadMoreError(describeSessionTraceError(error));
    } finally {
      if (requestIdRef.current === requestId) setLocatingRunId(null);
    }
    if (found && requestIdRef.current === requestId) scrollToTurn(runId);
  }

  function toggleExpanded(runId: string) {
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(runId)) next.delete(runId); else next.add(runId);
      return next;
    });
  }

  function stage(runId: string, slot: "a" | "b") {
    if (slot === "a") {
      setCompareA(runId);
      if (compareB === runId) setCompareB("");
    } else {
      setCompareB(runId);
      if (compareA === runId) setCompareA("");
    }
  }

  const branchCount = [...data.branchesByParent.values()].reduce((n, children) => n + children.length, 0);

  useTopbar(() => ({
    stats: (
      <>
        <span className="top-stat"><b>TURNS LOADED</b>{data.turns.length}</span>
        <span className="top-stat"><b>BRANCHES</b>{branchCount}</span>
      </>
    ),
    modeChip: phase === "loading" ? "LOADING" : phase === "error" ? "ERROR" : data.complete ? "COMPLETE" : "PARTIAL",
  }), [data.turns.length, branchCount, phase, data.complete]);

  if (phase === "loading" && data.turns.length === 0) {
    return (
      <section className="instrument investigation-timeline" aria-labelledby="investigation-title">
        <header className="instrument-head">
          <div><span className="eyebrow">INVESTIGATION</span><h1 id="investigation-title">Loading session…</h1></div>
        </header>
        <p className="investigation-empty">LOADING SESSION TRACE…</p>
      </section>
    );
  }

  if (phase === "error") {
    return (
      <section className="instrument investigation-timeline" aria-labelledby="investigation-title">
        <header className="instrument-head">
          <div><span className="eyebrow">INVESTIGATION</span><h1 id="investigation-title">Session not opened</h1></div>
        </header>
        <p className="investigation-load-error" role="alert">
          {errorInfo?.notFound
            ? `Session "${sessionId}" was not found -- no session record and no run has ever used this id.`
            : (errorInfo?.message ?? "the session trace request failed")}
        </p>
        <p className="investigation-boundary"><a href="#/sessions">Back to sessions</a></p>
      </section>
    );
  }

  const firstFindingCandidate = data.candidates.find((c) => c.kind === "first_finding");

  return (
    <>
      <section className="instrument investigation-timeline" aria-labelledby="investigation-title">
        <header className="instrument-head">
          <div>
            <span className="eyebrow">INVESTIGATION</span>
            <h1 id="investigation-title">{data.session?.title || sessionId}</h1>
          </div>
          <span className={`mode-chip ${data.complete ? "is-complete" : ""}`}>
            {data.complete ? "COMPLETE TRACE" : "PARTIAL TRACE"}
          </span>
        </header>
        <p className="investigation-boundary">
          An evidence timeline over this session&apos;s recorded turns -- prompts and responses show as
          SUMMARIES only, and every diagnostic below is a deterministic rule result computed ahead of
          time, never generated on the fly. This is not a chat replay.
        </p>
        {!data.complete && (
          <p className="investigation-partial-banner" role="status">
            Showing {data.turns.length} loaded turn{data.turns.length === 1 ? "" : "s"} -- PARTIAL TRACE.
            Load more below to see the rest of this session.
          </p>
        )}

        <ol className="investigation-turn-list" aria-label="Session turns, oldest first">
          {data.turns.map((turn) => (
            <TurnRow
              key={turn.runId}
              turn={turn}
              expanded={expanded.has(turn.runId)}
              onToggle={() => toggleExpanded(turn.runId)}
              branchChildren={data.branchesByParent.get(turn.runId) ?? []}
              compareA={compareA}
              compareB={compareB}
              onStage={stage}
            />
          ))}
        </ol>

        <footer className="investigation-timeline-end">
          {data.turns.length === 0 ? (
            <p>No turns recorded for this session yet.</p>
          ) : (
            <>
              <p>
                END OF LOADED TRACE -- {data.complete
                  ? "this is the full session."
                  : `${data.turns.length} turn(s) loaded so far.`}
              </p>
              {firstFindingCandidate && (
                <button type="button" onClick={() => void locate(firstFindingCandidate.runId)}>
                  ↑ JUMP TO FIRST FLAGGED TURN
                </button>
              )}
              {!data.complete && (
                <button type="button" disabled={loadingMore} onClick={() => void loadMore()}>
                  {loadingMore ? "LOADING…" : `LOAD NEXT ${TURNS_PER_PAGE} TURNS`}
                </button>
              )}
            </>
          )}
        </footer>
        {loadMoreError && <p className="investigation-load-error" role="alert">{loadMoreError}</p>}
      </section>

      <aside className="instrument investigation-inspector" aria-label="Investigation summary">
        <header className="instrument-head compact">
          <div><span className="eyebrow">DIAGNOSTICS</span><h2>First suspicious turn</h2></div>
        </header>
        <p className="investigation-candidates-note">
          Deterministic candidates only -- clozn never invents a &quot;first bad turn&quot; when its own
          rules found nothing. Each candidate names a structural fact (a rule fired, a setting changed, a
          run failed), never a claim that it explains the final answer.
        </p>
        <ul className="investigation-candidate-list">
          {CANDIDATE_KIND_ORDER.map((kind) => (
            <CandidateCard
              key={kind}
              kind={kind}
              candidate={data.candidates.find((c) => c.kind === kind)}
              locating={locatingRunId === data.candidates.find((c) => c.kind === kind)?.runId}
              onLocate={(runId) => void locate(runId)}
            />
          ))}
        </ul>

        <header className="instrument-head compact">
          <div><span className="eyebrow">USAGE</span><h2>Cumulative usage</h2></div>
          <span>{data.complete ? "FULL SESSION" : "THROUGH LAST LOADED TURN"}</span>
        </header>
        <dl className="investigation-cumulative-facts">
          <div><dt>Turns</dt><dd>{data.totals?.turnCount ?? 0}</dd></div>
          <div><dt>Duration</dt><dd>{formatDuration(data.totals?.durationMsTotal)}</dd></div>
          <div><dt>Prompt tokens</dt><dd>{data.totals?.promptTokensTotal ?? 0}</dd></div>
          <div><dt>Generated tokens</dt><dd>{data.totals?.generatedTokensTotal ?? 0}</dd></div>
        </dl>
      </aside>

      <section className="instrument investigation-branch-overview" aria-labelledby="investigation-branches-title">
        <header className="instrument-head compact">
          <div><span className="eyebrow">FORKS / RETRIES</span><h2 id="investigation-branches-title">Branches</h2></div>
          <strong>{branchCount} {branchCount === 1 ? "BRANCH" : "BRANCHES"}</strong>
        </header>
        <div className="investigation-branch-tree">
          {data.branchesByParent.size === 0 ? (
            <p className="investigation-empty">No forks or retries on the turns loaded so far.</p>
          ) : (
            [...data.branchesByParent.entries()].map(([parentRunId, children]) => (
              <div className="investigation-branch-group" key={parentRunId}>
                <button type="button" onClick={() => void locate(parentRunId)}>{shortId(parentRunId)}</button>
                <ul>
                  {children.map((child) => (
                    <li key={child.id}>
                      <span>{shortId(child.id)}</span>
                      <span>{(child.source ?? "—").toUpperCase()}</span>
                      <a href={`#/runs/${encodeURIComponent(child.id)}`}>OPEN</a>
                    </li>
                  ))}
                </ul>
              </div>
            ))
          )}
        </div>
      </section>

      <section className="instrument investigation-compare-tray" aria-labelledby="investigation-compare-title">
        <header className="instrument-head compact">
          <div><span className="eyebrow">A / B</span><h2 id="investigation-compare-title">Cross-turn comparison</h2></div>
        </header>
        <p className="investigation-compare-note">
          Stage any two turns (or forked branches) above, then compare their full token-level output --
          this reuses Runs&apos; own A/B comparison surface, never a second implementation.
        </p>
        <div className="runs-slots">
          <button type="button" className="is-a" onClick={() => stage(compareA, "a")}>
            <b>A</b><span>{compareA ? shortId(compareA) : "STAGE A TURN"}</span>
          </button>
          <i>→</i>
          <button type="button" className="is-b" onClick={() => stage(compareB, "b")}>
            <b>B</b><span>{compareB ? shortId(compareB) : "STAGE B TURN"}</span>
          </button>
        </div>
        <div className="runs-tray-actions">
          <button type="button" onClick={() => { setCompareA(""); setCompareB(""); }}>CLEAR</button>
          <a
            className={!compareA || !compareB ? "is-disabled" : ""}
            aria-disabled={!compareA || !compareB}
            href={compareA && compareB
              ? `#/compare/${encodeURIComponent(compareA)}/${encodeURIComponent(compareB)}`
              : undefined}
          >COMPARE</a>
        </div>
      </section>
    </>
  );
}
