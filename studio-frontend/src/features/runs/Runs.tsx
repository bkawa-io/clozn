import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { routeHref, type RunsRoute } from "../../app/router";
import { describeError, isContractError, studioApi } from "../../data/client";
import type { RunDetails, RunSummary } from "../../data/contracts";
import "./runs.css";

type RunsState =
  | { phase: "loading" }
  | { phase: "ready"; runs: readonly RunSummary[] }
  | { phase: "unavailable" | "malformed"; message: string };

type TranscriptState =
  | { key: string; phase: "loading" }
  | {
    key: string;
    phase: "ready";
    details: ReadonlyMap<string, RunDetails>;
    failedRunIds: ReadonlySet<string>;
  };

interface ConversationGroup {
  id: string;
  runs: readonly RunSummary[];
  title: string;
  latest: RunSummary;
}

interface ConversationDay {
  key: string;
  label: string;
  conversations: readonly ConversationGroup[];
}

function runDate(run: RunSummary): Date | undefined {
  const date = run.createdTs == null ? new Date(run.createdAt ?? "") : new Date(run.createdTs * 1000);
  return Number.isNaN(date.getTime()) ? undefined : date;
}

function formatTime(run: RunSummary): string {
  const date = runDate(run);
  if (!date) return "Time not recorded";
  return new Intl.DateTimeFormat(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  }).format(date);
}

function formatClock(run: RunSummary): string {
  const date = runDate(run);
  if (!date) return "—";
  return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
}

function groupConversationDays(conversations: readonly ConversationGroup[]): ConversationDay[] {
  const groups = new Map<string, ConversationDay>();
  for (const conversation of conversations) {
    const date = runDate(conversation.latest);
    const key = date
      ? `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`
      : "undated";
    const current = groups.get(key);
    if (current) {
      groups.set(key, { ...current, conversations: [...current.conversations, conversation] });
      continue;
    }
    groups.set(key, {
      key,
      label: date
        ? new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date)
        : "Undated",
      conversations: [conversation],
    });
  }
  return [...groups.values()];
}

function runTime(run: RunSummary): number {
  if (run.createdTs != null) return run.createdTs * 1000;
  const value = new Date(run.createdAt ?? "").getTime();
  return Number.isNaN(value) ? 0 : value;
}

function matchesSearch(run: RunSummary, query: string): boolean {
  if (!query) return true;
  return [
    run.id,
    run.model,
    run.promptSummary,
    run.responseSummary,
    run.conversationId,
    ...run.flags,
  ].some((value) => String(value ?? "").toLocaleLowerCase().includes(query));
}

function groupConversations(runs: readonly RunSummary[], query = ""): ConversationGroup[] {
  const groups = new Map<string, RunSummary[]>();
  for (const run of runs) {
    const id = run.conversationId || run.id;
    const current = groups.get(id) ?? [];
    current.push(run);
    groups.set(id, current);
  }
  return [...groups].flatMap(([id, groupRuns]) => {
    if (!groupRuns.some((run) => matchesSearch(run, query))) return [];
    const ordered = [...groupRuns].sort((left, right) => {
      const leftTurn = left.conversationTurn;
      const rightTurn = right.conversationTurn;
      if (leftTurn != null && rightTurn != null && leftTurn !== rightTurn) return leftTurn - rightTurn;
      return runTime(left) - runTime(right) || left.id.localeCompare(right.id);
    });
    return [{
      id,
      runs: ordered,
      title: ordered[0].promptSummary || "Untitled conversation",
      latest: ordered[ordered.length - 1],
    }];
  }).sort((left, right) => runTime(right.latest) - runTime(left.latest));
}

function excerpt(value: string | null | undefined, max = 240): string {
  const normalized = (value ?? "").trim().replace(/\s+/gu, " ");
  if (!normalized) return "No recorded text";
  return normalized.length <= max ? normalized : `${normalized.slice(0, max - 1).trimEnd()}…`;
}

function promptText(run: RunSummary, details?: RunDetails): string {
  const messages = details?.messages ?? [];
  const user = [...messages].reverse().find((message) => message.role === "user");
  const lastVisible = [...messages].reverse().find((message) => message.role !== "assistant");
  return user?.content || lastVisible?.content || run.promptSummary || "Prompt not recorded";
}

function responseText(run: RunSummary, details?: RunDetails): string {
  return details?.response ?? run.responseSummary ?? "No recorded answer";
}

function ConversationIndex({ conversations, search, onSearch }: {
  conversations: readonly ConversationGroup[];
  search: string;
  onSearch: (value: string) => void;
}) {
  const days = groupConversationDays(conversations);
  return (
    <>
      <label className="conversation-search">
        <span>Search conversations</span>
        <input
          type="search"
          value={search}
          onChange={(event) => onSearch(event.target.value)}
          placeholder="Opening message, latest reply, model, or ID"
        />
      </label>
      {conversations.length ? (
        <div className="conversation-index-list">
          {days.map((day) => (
            <section className="conversation-day" key={day.key} aria-labelledby={`conversation-day-${day.key}`}>
              <header className="conversation-day-heading">
                <h2 id={`conversation-day-${day.key}`}>{day.label}</h2>
                <span>{day.conversations.length} {day.conversations.length === 1 ? "chat" : "chats"}</span>
              </header>
              <ol>
                {day.conversations.map((conversation) => {
                  const first = conversation.runs[0];
                  const firstExcerpt = excerpt(first.promptSummary, 120);
                  const lastExcerpt = excerpt(conversation.latest.responseSummary, 120);
                  return (
                    <li key={conversation.id}>
                      <a href={routeHref({ surface: "runs", conversationId: conversation.id })}>
                        <header>
                          <time dateTime={conversation.latest.createdAt ?? undefined}>{formatClock(conversation.latest)}</time>
                          <span>{conversation.runs.length} {conversation.runs.length === 1 ? "turn" : "turns"}</span>
                        </header>
                        <div className="conversation-copy">
                          <p className="first-preview" aria-label={`First message: ${firstExcerpt}`}>{firstExcerpt}</p>
                          <p className="last-preview" aria-label={`Latest reply: ${lastExcerpt}`}>{lastExcerpt}</p>
                        </div>
                        <span className="conversation-open" aria-hidden="true">→</span>
                      </a>
                    </li>
                  );
                })}
              </ol>
            </section>
          ))}
        </div>
      ) : <p className="runs-empty">No conversations match “{search.trim()}”.</p>}
    </>
  );
}

function ConversationTranscript({
  conversation,
  transcript,
}: {
  conversation: ConversationGroup;
  transcript?: TranscriptState;
}) {
  const transcriptRef = useRef<HTMLDivElement>(null);
  const turnRefs = useRef(new Map<string, HTMLElement>());
  const [activeRunId, setActiveRunId] = useState(conversation.runs[0]?.id);
  const transcriptReady = transcript?.key === conversation.id && transcript.phase === "ready"
    ? transcript
    : undefined;

  useLayoutEffect(() => {
    const latestRun = conversation.runs[conversation.runs.length - 1];
    const viewport = transcriptRef.current;
    if (!latestRun || !viewport) return;
    setActiveRunId(latestRun.id);
    const settleAtLatestTurn = () => {
      viewport.scrollTop = viewport.scrollHeight;
    };
    settleAtLatestTurn();
    const firstFrame = window.requestAnimationFrame(() => {
      settleAtLatestTurn();
      window.requestAnimationFrame(settleAtLatestTurn);
    });
    return () => window.cancelAnimationFrame(firstFrame);
  }, [conversation.id, conversation.runs, transcriptReady?.details]);

  const updateActiveTurn = useCallback(() => {
    const viewport = transcriptRef.current;
    if (!viewport) return;
    if (viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight <= 2) {
      setActiveRunId(conversation.runs[conversation.runs.length - 1]?.id);
      return;
    }
    const viewportRect = viewport.getBoundingClientRect();
    const targetY = viewportRect.top + Math.min(150, viewportRect.height * .28);
    let nearest: { id: string; distance: number } | undefined;
    for (const run of conversation.runs) {
      const element = turnRefs.current.get(run.id);
      if (!element) continue;
      const rect = element.getBoundingClientRect();
      const distance = rect.bottom < viewportRect.top
        ? viewportRect.top - rect.bottom
        : rect.top > viewportRect.bottom
          ? rect.top - viewportRect.bottom
          : Math.abs(rect.top - targetY);
      if (!nearest || distance < nearest.distance) nearest = { id: run.id, distance };
    }
    if (nearest) setActiveRunId(nearest.id);
  }, [conversation.runs]);

  const jumpTo = (runId: string) => {
    setActiveRunId(runId);
    const viewport = transcriptRef.current;
    const target = turnRefs.current.get(runId);
    if (!viewport || !target) return;
    const top = viewport.scrollTop + target.getBoundingClientRect().top - viewport.getBoundingClientRect().top;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    viewport.scrollTo?.({ top, behavior: reduced ? "auto" : "smooth" });
  };

  return (
    <div className="conversation-chat-shell">
      <div className="conversation-transcript" ref={transcriptRef} onScroll={updateActiveTurn}>
        {conversation.runs.map((run, index) => {
          const details = transcriptReady?.details.get(run.id);
          const prompt = promptText(run, details);
          const answer = responseText(run, details);
          const inspectHref = routeHref({
            surface: "inspect",
            runId: run.id,
            conversationId: conversation.id,
            ...(run.analysisScoringMode ? { scoringMode: run.analysisScoringMode } : {}),
          });
          return (
            <article
              className="chat-exchange"
              id={`turn-${run.id}`}
              key={run.id}
              ref={(element) => {
                if (element) turnRefs.current.set(run.id, element);
                else turnRefs.current.delete(run.id);
              }}
            >
              <section className="chat-message user-message" aria-label={`Turn ${index + 1}, user message`}>
                <a className="chat-message-link" href={inspectHref} aria-label={`Examine turn ${index + 1} from the request`}>
                  <div className="chat-message-body">
                    <header><strong>You</strong><time dateTime={run.createdAt ?? undefined}>{formatTime(run)}</time></header>
                    <p>{prompt}</p>
                  </div>
                </a>
              </section>
              <section className="chat-message assistant-message" aria-label={`Turn ${index + 1}, assistant response`}>
                <a className="chat-message-link" href={inspectHref} aria-label={`Examine turn ${index + 1} from the response`}>
                  <div className="chat-message-body">
                    <header>
                      <strong>Assistant</strong>
                      <span>Turn {run.conversationTurn ?? index + 1}</span>
                    </header>
                    <p>{answer}</p>
                  </div>
                </a>
              </section>
            </article>
          );
        })}
        {transcript?.key === conversation.id && transcript.phase === "loading" && (
          <p className="transcript-loading" role="status">Loading full conversation…</p>
        )}
      </div>
      <nav className="turn-scrubber" aria-label="Conversation turns">
        <span className="turn-scrubber-track" aria-hidden="true" />
        {conversation.runs.map((run, index) => (
          <button
            type="button"
            key={run.id}
            className={run.id === activeRunId ? "is-active" : undefined}
            aria-current={run.id === activeRunId ? "true" : undefined}
            aria-label={`Jump to turn ${index + 1}: ${excerpt(run.promptSummary, 90)}`}
            onClick={() => jumpTo(run.id)}
          >
            <i aria-hidden="true" />
            <span className="turn-scrubber-preview" aria-hidden="true">
              <strong>Turn {run.conversationTurn ?? index + 1}</strong>
              <b>{excerpt(run.promptSummary, 100)}</b>
              <span>{excerpt(run.responseSummary, 130)}</span>
            </span>
          </button>
        ))}
      </nav>
    </div>
  );
}

export function RunsSurface({ route }: { route: RunsRoute }) {
  const [state, setState] = useState<RunsState>({ phase: "loading" });
  const [revision, setRevision] = useState(0);
  const [search, setSearch] = useState("");
  const [transcript, setTranscript] = useState<TranscriptState>();

  useEffect(() => {
    const controller = new AbortController();
    setState({ phase: "loading" });
    void studioApi.runs(controller.signal).then((runs) => {
      if (!controller.signal.aborted) setState({ phase: "ready", runs });
    }).catch((error: unknown) => {
      if (!controller.signal.aborted) {
        setState({ phase: isContractError(error) ? "malformed" : "unavailable", message: describeError(error) });
      }
    });
    return () => controller.abort();
  }, [revision]);

  const allConversations = useMemo(
    () => state.phase === "ready" ? groupConversations(state.runs) : [],
    [state],
  );
  const normalizedSearch = search.trim().toLocaleLowerCase();
  const visibleConversations = useMemo(
    () => state.phase === "ready" ? groupConversations(state.runs, normalizedSearch) : [],
    [normalizedSearch, state],
  );
  const selectedConversation = route.conversationId
    ? allConversations.find((conversation) => conversation.id === route.conversationId)
    : undefined;
  const transcriptKey = selectedConversation?.runs.map((run) => run.id).join("|");

  useEffect(() => {
    if (!selectedConversation || !transcriptKey) {
      setTranscript(undefined);
      return undefined;
    }
    const controller = new AbortController();
    setTranscript({ key: selectedConversation.id, phase: "loading" });
    void Promise.all(selectedConversation.runs.map(async (run) => {
      try {
        return { id: run.id, details: await studioApi.run(run.id, controller.signal) };
      } catch {
        return { id: run.id, details: undefined };
      }
    })).then((rows) => {
      if (controller.signal.aborted) return;
      const details = new Map<string, RunDetails>();
      const failedRunIds = new Set<string>();
      for (const row of rows) {
        if (row.details) details.set(row.id, row.details);
        else failedRunIds.add(row.id);
      }
      setTranscript({ key: selectedConversation.id, phase: "ready", details, failedRunIds });
    });
    return () => controller.abort();
  }, [selectedConversation?.id, transcriptKey]);

  const detailMode = Boolean(route.conversationId);
  const loadingConversation = detailMode && state.phase === "loading";
  return (
    <div className={`runs-surface${detailMode ? " is-conversation" : ""}`}>
      <header className={`runs-heading ${detailMode ? "is-detail" : "is-index"}`}>
        <div>
          {detailMode && <a className="back-link" href={routeHref({ surface: "runs" })}>← All conversations</a>}
          <h1>{selectedConversation
            ? excerpt(selectedConversation.title, 120)
            : loadingConversation ? "Loading conversation…" : detailMode ? "Conversation unavailable" : "Conversations"}</h1>
          {detailMode && selectedConversation && (
            <span className="conversation-turn-count">{selectedConversation.runs.length} {selectedConversation.runs.length === 1 ? "turn" : "turns"}</span>
          )}
        </div>
        {state.phase === "ready" && !detailMode && <span>{visibleConversations.length} conversations</span>}
        {state.phase === "ready" && !detailMode && <button type="button" className="secondary-action" onClick={() => setRevision((value) => value + 1)}>Refresh</button>}
      </header>

      {state.phase === "loading" && <p className="studio-state" role="status">Loading conversations.</p>}
      {(state.phase === "malformed" || state.phase === "unavailable") && (
        <p className="studio-state is-error" role="alert">{state.phase === "malformed" ? "Malformed conversation list." : "Conversation list unavailable."} <span>{state.message}</span></p>
      )}
      {state.phase === "ready" && !state.runs.length && (
        <p className="runs-empty">No conversations recorded yet. Point an OpenAI-compatible chat client at Clozn to begin.</p>
      )}
      {state.phase === "ready" && state.runs.length > 0 && !detailMode && (
        <ConversationIndex conversations={visibleConversations} search={search} onSearch={setSearch} />
      )}
      {state.phase === "ready" && detailMode && selectedConversation && (
        <ConversationTranscript
          conversation={selectedConversation}
          transcript={transcript}
        />
      )}
      {state.phase === "ready" && detailMode && !selectedConversation && (
        <p className="runs-empty">Return to the conversation list and choose an available chat.</p>
      )}
    </div>
  );
}
