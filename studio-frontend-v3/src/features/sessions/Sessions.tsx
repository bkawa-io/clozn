import { useEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import { routeHref } from "../../app/router";
import { describeError, isContractError, studioApi } from "../../data/client";
import type { SessionRecord, SessionTrace, StandaloneRun, TraceBranch, TraceTurn } from "../../data/contracts";
import "./sessions.css";

type LoadState<T> =
  | { phase: "loading" }
  | { phase: "ready"; value: T }
  | { phase: "unavailable"; error: string }
  | { phase: "malformed"; error: string };

function useRead<T>(load: (signal: AbortSignal) => Promise<T>): LoadState<T> {
  const [state, setState] = useState<LoadState<T>>({ phase: "loading" });
  useEffect(() => {
    const controller = new AbortController();
    setState({ phase: "loading" });
    void load(controller.signal).then((value) => {
      if (!controller.signal.aborted) setState({ phase: "ready", value });
    }).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setState({ phase: isContractError(error) ? "malformed" : "unavailable", error: describeError(error) });
    });
    return () => controller.abort();
  }, [load]);
  return state;
}

function formatTimestamp(epochSeconds?: number | null, fallback?: string): string {
  const date = epochSeconds == null ? new Date(fallback ?? "") : new Date(epochSeconds * 1000);
  if (Number.isNaN(date.getTime())) return "Time not recorded";
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric", month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  }).format(date);
}

function shortId(value: string): string {
  return value.length > 18 ? `…${value.slice(-16)}` : value;
}

function plural(count: number, singular: string): string {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}

function StateLine({ phase, error, noun }: { phase: LoadState<unknown>["phase"]; error?: string; noun: string }) {
  if (phase === "loading") return <p className="v3-state" role="status">Loading {noun}.</p>;
  if (phase === "malformed") return <p className="v3-state is-error" role="alert">Malformed {noun} contract. <span>{error}</span></p>;
  if (phase === "unavailable") return <p className="v3-state is-error" role="alert">{noun} unavailable. <span>{error}</span></p>;
  return null;
}

function SessionCard({ session }: { session: SessionRecord }) {
  const activity = session.lastActivityTs ?? session.firstActivityTs;
  return (
    <a className="session-card" href={routeHref({ surface: "session", sessionId: session.id })}>
      <div className="session-card-head">
        <h2>{session.title ?? <code>{session.id}</code>}</h2>
        <span className="session-card-arrow" aria-hidden="true">↗</span>
      </div>
      <div className="session-card-facts">
        <time dateTime={session.createdAt}>{formatTimestamp(activity, session.createdAt)}</time>
        {session.runCount !== undefined && <span>{plural(session.runCount, "turn")}</span>}
        {session.visibility === "hidden" && <span className="status-label">Hidden</span>}
      </div>
      {session.preview && (
        <div className="session-preview" aria-label="Conversation preview">
          {session.preview.promptSummary && <p><b>Prompt</b><span>{session.preview.promptSummary}</span></p>}
          {session.preview.responseSummary && <p><b>Response</b><span>{session.preview.responseSummary}</span></p>}
        </div>
      )}
    </a>
  );
}

function StandaloneRow({ run }: { run: StandaloneRun }) {
  return (
    <li className="standalone-row">
      <div>
        <code>{shortId(run.id)}</code>
        <span>{run.model ?? "Model not recorded"}</span>
      </div>
      <p>{run.promptSummary || run.responseSummary || "—"}</p>
      <time dateTime={run.createdAt ?? undefined}>{formatTimestamp(run.createdTs, run.createdAt ?? undefined)}</time>
    </li>
  );
}

function SessionsIndex() {
  const sessionsLoader = useMemo(() => (signal: AbortSignal) => studioApi.sessions(signal), []);
  const runsLoader = useMemo(() => (signal: AbortSignal) => studioApi.standaloneRuns(signal), []);
  const sessions = useRead(sessionsLoader);
  const runs = useRead(runsLoader);
  const standalone = runs.phase === "ready" ? runs.value.filter((run) => run.sessionKey == null) : [];

  return (
    <div className="sessions-index">
      <header className="page-heading">
        <div>
          <span className="eyebrow">SESSION INDEX</span>
          <h1>Sessions</h1>
        </div>
        {sessions.phase === "ready" && <span className="heading-count">{plural(sessions.value.length, "session")}</span>}
      </header>

      <section className="session-index-section" aria-labelledby="session-list-title">
        <div className="section-heading">
          <h2 id="session-list-title">Conversations</h2>
          {sessions.phase === "ready" && <span>{sessions.value.length}</span>}
        </div>
        <StateLine phase={sessions.phase} error={sessions.phase !== "ready" && sessions.phase !== "loading" ? sessions.error : undefined} noun="sessions" />
        {sessions.phase === "ready" && sessions.value.length === 0 && <p className="empty-state">No sessions recorded.</p>}
        {sessions.phase === "ready" && sessions.value.length > 0 && (
          <div className="session-grid">
            {sessions.value.map((session) => <SessionCard key={session.id} session={session} />)}
          </div>
        )}
      </section>

      <section className="standalone-section" aria-labelledby="standalone-title">
        <div className="section-heading">
          <div>
            <h2 id="standalone-title">Standalone</h2>
            <p>Runs without a session identity.</p>
          </div>
          {runs.phase === "ready" && <span>{standalone.length}</span>}
        </div>
        <StateLine phase={runs.phase} error={runs.phase !== "ready" && runs.phase !== "loading" ? runs.error : undefined} noun="standalone runs" />
        {runs.phase === "ready" && standalone.length === 0 && <p className="empty-state">No standalone runs.</p>}
        {runs.phase === "ready" && standalone.length > 0 && <ol className="standalone-list">{standalone.map((run) => <StandaloneRow key={run.id} run={run} />)}</ol>}
      </section>
    </div>
  );
}

function branchFor(branches: readonly TraceBranch[], runId: string): TraceBranch | undefined {
  return branches.find((branch) => branch.parentRunId === runId);
}

function turnFailure(turn: TraceTurn): boolean {
  return Boolean(turn.error || turn.finishReason === "error" || turn.finishReason === "failed");
}

function turnPosition(index: number, count: number): number {
  return count < 2 ? 50 : (index / (count - 1)) * 100;
}

function selectedRunHref(sessionId: string, runId: string): string {
  return routeHref({ surface: "session", sessionId, runId });
}

function ConversationScrubber({ sessionId, trace, selectedRunId, onSelect }: { sessionId: string; trace: SessionTrace; selectedRunId?: string; onSelect: (runId: string) => void }) {
  const [hoveredRunId, setHoveredRunId] = useState<string | undefined>(selectedRunId ?? trace.turns[0]?.runId);
  const turnNodes = useRef(new Map<string, HTMLElement>());
  const previewTurn = trace.turns.find((turn) => turn.runId === hoveredRunId);

  useEffect(() => {
    setHoveredRunId(selectedRunId ?? trace.turns[0]?.runId);
    if (!selectedRunId) return;
    const node = turnNodes.current.get(selectedRunId);
    if (!node) return;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    node.scrollIntoView?.({ block: "nearest", behavior: reduced ? "auto" : "smooth" });
  }, [selectedRunId, trace.turns]);

  function indexAt(clientY: number, element: HTMLElement): number {
    const rect = element.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (clientY - rect.top) / Math.max(rect.height, 1)));
    return Math.round(ratio * Math.max(trace.turns.length - 1, 0));
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLElement>) {
    const index = indexAt(event.clientY, event.currentTarget);
    setHoveredRunId(trace.turns[index]?.runId);
  }

  function handlePointerDown(event: ReactPointerEvent<HTMLElement>) {
    const target = event.target;
    if (!(target instanceof HTMLElement) || (target !== event.currentTarget && !target.classList.contains("scrubber-track"))) return;
    selectIndex(indexAt(event.clientY, event.currentTarget));
  }

  function selectIndex(index: number) {
    const turn = trace.turns[Math.max(0, Math.min(trace.turns.length - 1, index))];
    if (turn) onSelect(turn.runId);
  }

  if (!trace.turns.length) return null;
  return (
    <aside
      className="conversation-scrubber-column"
      aria-label="Conversation turns"
      onPointerMove={handlePointerMove}
      onPointerDown={handlePointerDown}
      onPointerLeave={() => setHoveredRunId(undefined)}
    >
      <div className="conversation-scrubber">
        <div className="scrubber-track" aria-hidden="true" />
        {trace.turns.map((turn, index) => {
          const branch = branchFor(trace.branches, turn.runId);
          const failure = turnFailure(turn);
          const label = `Turn ${index + 1}${failure ? ", failure recorded" : ""}${branch ? `, ${branch.children.length} branch${branch.children.length === 1 ? "" : "es"} recorded` : ""}`;
          const style = { "--turn-position": `${turnPosition(index, trace.turns.length)}%` } as CSSProperties;
          return (
            <button
              key={turn.runId}
              ref={(node) => { if (node) turnNodes.current.set(turn.runId, node); else turnNodes.current.delete(turn.runId); }}
              className={`scrubber-turn${turn.runId === selectedRunId ? " is-selected" : ""}${failure ? " has-failure" : ""}${branch ? " has-branch" : ""}`}
              style={style}
              type="button"
              aria-label={label}
              aria-current={turn.runId === selectedRunId ? "true" : undefined}
              tabIndex={turn.runId === (selectedRunId ?? trace.turns[0]?.runId) ? 0 : -1}
              onClick={() => selectIndex(index)}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown" || event.key === "ArrowRight") { event.preventDefault(); selectIndex(index + 1); }
                if (event.key === "ArrowUp" || event.key === "ArrowLeft") { event.preventDefault(); selectIndex(index - 1); }
                if (event.key === "Home") { event.preventDefault(); selectIndex(0); }
                if (event.key === "End") { event.preventDefault(); selectIndex(trace.turns.length - 1); }
              }}
            >
              <span aria-hidden="true">{failure ? "×" : branch ? "◇" : "•"}</span>
            </button>
          );
        })}
        {previewTurn && (
          <div className="scrubber-preview" style={{ "--turn-position": `${turnPosition(Math.max(0, trace.turns.indexOf(previewTurn)), trace.turns.length)}%` } as CSSProperties}>
            <b>Turn {trace.turns.indexOf(previewTurn) + 1}</b>
            <span>{previewTurn.promptSummary || previewTurn.responseSummary || "—"}</span>
            <a href={selectedRunHref(sessionId, previewTurn.runId)}>Open run</a>
          </div>
        )}
      </div>
    </aside>
  );
}

function TurnCard({ turn, index, trace, selected }: { turn: TraceTurn; index: number; trace: SessionTrace; selected: boolean }) {
  const branch = branchFor(trace.branches, turn.runId);
  const failure = turnFailure(turn);
  const comparison = turn.turnComparison;
  const comparisonDetail = comparison?.available
    ? comparison.classifications.map((classification) => classification.summary).join(" · ") || [
      ...comparison.settingsChanges.map((change) => change.dimension),
      ...comparison.contextChanges.newSegments.map((change) => change.dimension),
      ...comparison.contextChanges.droppedSegments.map((change) => change.dimension),
    ].join(" · ") || "—"
    : comparison?.reason || "—";
  return (
    <article id={`turn-${turn.runId}`} className={`turn-card${selected ? " is-selected" : ""}`}>
      <header className="turn-card-head">
        <div>
          <span className="turn-number">TURN {index + 1}</span>
          <time dateTime={turn.createdAt}>{formatTimestamp(turn.recordedTs, turn.createdAt)}</time>
        </div>
        <div className="turn-state-labels">
          {failure && <span className="state-failure"><i aria-hidden="true">×</i> Failure recorded</span>}
          {branch && <span className="state-branch"><i aria-hidden="true">◇</i> {plural(branch.children.length, "branch")}</span>}
          <code>{shortId(turn.runId)}</code>
        </div>
      </header>
      <div className="turn-card-meta">
        <span>{turn.model || "Model not recorded"}</span>
        {turn.durationMs !== undefined && <span>{turn.durationMs} ms</span>}
        {turn.finishReason && <span>{turn.finishReason}</span>}
      </div>
      <div className="turn-prose">
        <section><h3>Prompt</h3><p>{turn.redacted ? "Redacted" : turn.promptSummary || "—"}</p></section>
        <section><h3>Response</h3><p>{turn.redacted ? "Redacted" : turn.responseSummary || "—"}</p></section>
      </div>
      <div className="turn-facts">
        <div><dt>Diagnostics</dt><dd>{turn.diagnosticHighlights.statusCounts.finding ? `${turn.diagnosticHighlights.statusCounts.finding} finding${turn.diagnosticHighlights.statusCounts.finding === 1 ? "" : "s"}` : "No findings"}</dd></div>
        <div><dt>Previous turn</dt><dd>{comparisonDetail}</dd></div>
        <div><dt>Cumulative turns</dt><dd>{turn.cumulative.turnCount}</dd></div>
        <div><dt>Cumulative time</dt><dd>{turn.cumulative.durationMsTotal} ms</dd></div>
      </div>
    </article>
  );
}

function Conversation({ sessionId, trace, selectedRunId }: { sessionId: string; trace: SessionTrace; selectedRunId?: string }) {
  function selectRun(runId: string) {
    window.location.hash = selectedRunHref(sessionId, runId);
  }
  return (
    <section className="conversation-surface" aria-labelledby="conversation-title">
      <div className="conversation-heading">
        <div><span className="eyebrow">RECORDED TURNS</span><h2 id="conversation-title">Conversation</h2></div>
        <span>{plural(trace.totalsThroughThisPage.turnCount, "turn")}</span>
      </div>
      {trace.totalsThroughThisPage.turnCount === 0 ? <p className="empty-state">No turns recorded.</p> : (
        <div className="conversation-viewport">
          <div className="conversation-layout">
            <ConversationScrubber sessionId={sessionId} trace={trace} selectedRunId={selectedRunId} onSelect={selectRun} />
            <div className="turn-list">
              {trace.turns.map((turn, index) => <TurnCard key={turn.runId} turn={turn} index={index} trace={trace} selected={turn.runId === selectedRunId} />)}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function SessionDetail({ sessionId, selectedRunId }: { sessionId: string; selectedRunId?: string }) {
  const loader = useMemo(() => (signal: AbortSignal) => Promise.all([studioApi.session(sessionId, signal), studioApi.sessionTrace(sessionId, signal)]), [sessionId]);
  const resource = useRead(loader);
  if (resource.phase !== "ready") return <div className="session-detail"><StateLine phase={resource.phase} error={resource.phase !== "loading" ? resource.error : undefined} noun="session" /></div>;
  const [session, trace] = resource.value;
  const title = session.title ?? trace.session.title;
  return (
    <div className="session-detail">
      <a className="back-link" href={routeHref({ surface: "sessions" })}>← Sessions</a>
      <header className="detail-heading">
        <div>
          <span className="eyebrow">SESSION</span>
          <h1>{title ?? <code>{session.id}</code>}</h1>
          <code className="detail-id">{session.id}</code>
        </div>
        <dl className="detail-facts">
          <div><dt>Created</dt><dd>{formatTimestamp(session.createdTs, session.createdAt)}</dd></div>
          {session.runCount !== undefined && <div><dt>Turns</dt><dd>{session.runCount}</dd></div>}
          {session.visibility === "hidden" && <div><dt>Visibility</dt><dd>Hidden</dd></div>}
        </dl>
      </header>
      <Conversation sessionId={sessionId} trace={trace} selectedRunId={selectedRunId} />
    </div>
  );
}

export function SessionsSurface({ route }: { route: { surface: "sessions" } | { surface: "session"; sessionId: string; runId?: string } }) {
  return route.surface === "sessions" ? <SessionsIndex /> : <SessionDetail sessionId={route.sessionId} selectedRunId={route.runId} />;
}
