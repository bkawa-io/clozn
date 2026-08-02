import { useEffect, useState, type FormEvent } from "react";
import {
  capabilityLabel,
  findInvestigationQuestion,
  INVESTIGATION_QUESTIONS,
  matchInvestigationIntent,
  type InvestigationQuestion,
} from "../../data/askAnotherQuestion";
import {
  loadInvestigationHistory,
  recordInvestigation,
  type InvestigationHistoryEntry,
} from "../../data/investigationHistory";

/**
 * C4 -- "Ask another question" investigation entry point. See `data/askAnotherQuestion.ts`'s own doc
 * comment for the full honesty audit behind each question's capability; this file is presentation and
 * routing only.
 *
 * VISUALLY DISTINCT FROM A CHAT TURN, ON PURPOSE
 * -------------------------------------------------
 * F3 (a conversation/topic investigation view) has not landed yet, so there is no live chat surface in
 * Studio to contrast against today -- but C4's own acceptance criterion ("topic follow-up and
 * investigation visually distinct") is a constraint on THIS panel's shape, not something F3 gets to
 * retrofit later. Nothing here uses a message-bubble/avatar/transcript idiom: the question list is a
 * grid of labelled, stateful buttons (like WhatMattered's legend or DiagnosisRepair's corrective-action
 * list), the free-text field is captioned as a router in its own placeholder, and history renders as a
 * dated, structured log (`<ol>` + `<time>`), not a scrollback of turns. An investigation is a lookup
 * against recorded evidence, never a new conversational reply.
 *
 * ROUTING, NEVER EXPLAINING
 * --------------------------
 * This component fires zero requests. Selecting a question -- by chip or by a free-text match -- only
 * ever does two things: scroll a sibling slot panel into view, and append one entry to investigation
 * history (`data/investigationHistory.ts`). No branch here renders text that was not already a fixed
 * string in `data/askAnotherQuestion.ts`; a free-text query the user typed is only ever echoed back
 * verbatim in quotes, never paraphrased or answered.
 */

export interface AskAnotherQuestionProps {
  runId: string;
}

function scrollToAnchor(anchorId: string) {
  requestAnimationFrame(() => {
    document.getElementById(anchorId)?.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

function formatTimestamp(ts: number): string {
  const date = new Date(ts);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

function QuestionChip({
  question, runId, onSelect,
}: {
  question: InvestigationQuestion;
  runId: string;
  onSelect: () => void;
}) {
  const disabled = question.capability === "unavailable";
  return (
    <li className={`ask-another-chip is-${question.capability}`}>
      <button type="button" disabled={disabled} aria-disabled={disabled} onClick={onSelect}>
        <strong>{question.label}</strong>
        <span className="ask-another-chip-state">{capabilityLabel(question.capability)}</span>
        {/* The claim in data/askAnotherQuestion.ts's own doc comment, made checkable from the running
            UI itself rather than left as a source comment nobody using the app ever sees. */}
        <small className="ask-another-chip-backing">{question.backedBy}</small>
      </button>
      {question.capability !== "available" && <p className="ask-another-chip-reason">{question.reason}</p>}
    </li>
  );
}

export function AskAnotherQuestion({ runId }: AskAnotherQuestionProps) {
  const [query, setQuery] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [history, setHistory] = useState<InvestigationHistoryEntry[]>([]);

  // Investigation history is scoped to THIS run and reloaded whenever the run changes -- the same
  // run-identity guard every other lens.evidence panel applies to its own fetch (WhatMattered.tsx,
  // DiagnosisRepair.tsx, ...), even though this panel reads the browser's own store rather than a GET.
  useEffect(() => {
    setHistory(runId ? loadInvestigationHistory(runId) : []);
    setNotice(null);
    setQuery("");
  }, [runId]);

  function open(question: InvestigationQuestion, origin: "chip" | "free_text", queryText?: string) {
    // Defense in depth, not just a disabled button: this function itself refuses to record or scroll to
    // an "unavailable" question even if called directly, mirroring useTokenWorkbench's own
    // never-trust-just-the-button-state discipline for its action tray.
    if (question.capability === "unavailable") return;
    scrollToAnchor(question.target.anchorId);
    const entry = recordInvestigation({
      runId,
      questionId: question.id,
      questionLabel: question.label,
      targetDescription: question.target.description,
      origin,
      queryText,
    });
    setHistory((current) => [entry, ...current]);
    setNotice(null);
  }

  function submitQuery(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) return;
    const matchedId = matchInvestigationIntent(trimmed);
    const question = matchedId ? findInvestigationQuestion(matchedId) : undefined;
    if (!question || question.capability === "unavailable") {
      // Routing only -- this notice is either the question's own fixed `reason` or a fixed fallback
      // string; never text derived from `trimmed` beyond echoing it back in quotes.
      setNotice(
        question
          ? `"${question.label}" cannot run yet -- ${question.reason}`
          : `"${trimmed}" doesn't match a question clozn can run yet -- pick one below.`,
      );
      return;
    }
    open(question, "free_text", trimmed);
    setQuery("");
  }

  function revisit(entry: InvestigationHistoryEntry) {
    const question = findInvestigationQuestion(entry.questionId);
    if (question.capability === "unavailable") return; // an entry is never recorded for one of these
    scrollToAnchor(question.target.anchorId);
  }

  const runnableCount = INVESTIGATION_QUESTIONS.filter((question) => question.capability !== "unavailable").length;

  return (
    <section className="ask-another" aria-labelledby="ask-another-title">
      <header className="ask-another-head">
        <div>
          <span className="eyebrow">INVESTIGATE</span>
          <h3 id="ask-another-title">Ask another question</h3>
        </div>
        <strong className="ask-another-runnable">
          {runnableCount} / {INVESTIGATION_QUESTIONS.length} RUNNABLE
        </strong>
      </header>

      <p className="ask-another-boundary">
        A directory of the investigations clozn can actually run for this run -- not a chat. Every
        question below either opens real evidence, opens an explicit action, or says plainly why it
        can&apos;t yet; nothing on this panel generates an explanation of its own.
      </p>

      <ul className="ask-another-grid" aria-label="Investigation questions">
        {INVESTIGATION_QUESTIONS.map((question) => (
          <QuestionChip
            key={question.id}
            question={question}
            runId={runId}
            onSelect={() => open(question, "chip")}
          />
        ))}
      </ul>

      <form className="ask-another-freeform" onSubmit={submitQuery}>
        <label htmlFor="ask-another-input">
          <span>ROUTE A QUESTION</span>
          <input
            id="ask-another-input"
            type="text"
            value={query}
            onChange={(event) => { setQuery(event.target.value); setNotice(null); }}
            onKeyDown={(event) => {
              // Keep keyboard submission explicit. Some embedded browser surfaces do not synthesize
              // the implicit submit for a controlled text input inside this labelled form.
              if (event.key === "Enter") {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Routes to one of the questions above -- never answers directly"
          />
        </label>
        <button type="submit" disabled={!query.trim()}>ROUTE</button>
      </form>
      {notice && <p className="ask-another-notice" role="status">{notice}</p>}

      <section className="ask-another-history" aria-label="Past investigations for this run">
        <header className="section-title">
          <h3>Past investigations for this run</h3>
          <span>{history.length}</span>
        </header>
        {history.length === 0 ? (
          <p className="ask-another-history-empty">No investigations recorded yet for this run.</p>
        ) : (
          <ol className="ask-another-history-list">
            {history.map((entry) => (
              <li key={entry.entryId}>
                <button type="button" onClick={() => revisit(entry)}>
                  <time dateTime={new Date(entry.ts).toISOString()}>{formatTimestamp(entry.ts)}</time>
                  <strong>{entry.questionLabel}</strong>
                  {entry.queryText && <span className="ask-another-history-query">&quot;{entry.queryText}&quot;</span>}
                  <small>{entry.targetDescription}</small>
                </button>
              </li>
            ))}
          </ol>
        )}
      </section>
    </section>
  );
}
