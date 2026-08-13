import { useMemo, useState } from "react";
import type { RunMessage, RunRecord } from "../../data/contracts";
import "./turn-time-travel.css";

interface ConversationTurn extends RunMessage {
  id: string;
  kind: "context" | "response";
}

export interface TurnTimeTravelSurfaceProps {
  run: RunRecord;
  family?: readonly RunRecord[];
  linkedSelection?: { answerId?: string; sourceId?: string; differenceId?: string; compareRunId?: string };
  onOpenTokenExecution: (position?: number) => void;
  onOpenCompare?: (parentRunId: string, childRunId: string) => void;
  onInspectRun?: (runId: string) => void;
}

function turnsFor(run: RunRecord): ConversationTurn[] {
  const received = run.assembledMessages ?? run.messages ?? [];
  const turns: ConversationTurn[] = received.map((message, index) => ({ ...message, id: `message-${index}`, kind: "context" }));
  if (run.response) turns.push({ id: "recorded-response", role: "assistant", content: run.response, kind: "response" });
  return turns;
}

function roleLabel(turn: ConversationTurn): string {
  if (turn.sourceLabel?.trim()) return turn.sourceLabel;
  if (turn.role === "assistant") return "Assistant response";
  if (turn.role === "system") return "System instruction";
  if (turn.role === "user") return "User turn";
  return turn.role;
}

export function TurnTimeTravelSurface({ run, family = [], linkedSelection, onOpenTokenExecution, onOpenCompare, onInspectRun }: TurnTimeTravelSurfaceProps) {
  const turns = useMemo(() => turnsFor(run), [run]);
  const [selectedId, setSelectedId] = useState(turns.at(-1)?.id);
  const selected = turns.find((turn) => turn.id === selectedId) ?? turns.at(-1);
  const relatives = family.filter((candidate) => candidate.id !== run.id);
  const children = relatives.filter((candidate) => candidate.parentRunId === run.id);

  return <main className="turn-time-travel"><header className="turn-time-travel__heading"><div><span className="eyebrow">RECORDED EXECUTION / TIME TRAVEL</span><h1>Conversation strand</h1><p>Run <code>{run.id}</code>{run.model && <> · {run.model}</>}</p></div><button type="button" className="primary-action" onClick={() => onOpenTokenExecution()}>Inspect token execution</button></header>
    <div className="turn-time-travel__layout">
      <section className="conversation-strand" aria-labelledby="conversation-strand-title"><header><span className="eyebrow">PRIMARY VIEW</span><h2 id="conversation-strand-title">Recorded turns</h2><p>Conversation order remains readable; forensic token controls stay one level below.</p></header><ol>{turns.map((turn, index) => <li key={turn.id} className={turn.id === selected?.id ? "is-selected" : undefined}><button type="button" aria-pressed={turn.id === selected?.id} onClick={() => setSelectedId(turn.id)}><span>{String(index + 1).padStart(2, "0")} · {roleLabel(turn)}</span><p>{turn.content}</p>{turn.kind === "response" && <em>RECORDED OUTCOME</em>}</button></li>)}</ol></section>
      <aside className="lineage-gutter" aria-labelledby="lineage-gutter-title"><header><span className="eyebrow">LINEAGE GUTTER</span><h2 id="lineage-gutter-title">Branch family</h2></header><div className="lineage-gutter__rail" aria-hidden="true" /><button type="button" className="is-current" onClick={() => onInspectRun?.(run.id)}><span>CURRENT</span><code>{run.id}</code></button>{run.parentRunId && <button type="button" onClick={() => onInspectRun?.(run.parentRunId!)}><span>PARENT</span><code>{run.parentRunId}</code></button>}{children.map((child) => <button type="button" key={child.id} onClick={() => onInspectRun?.(child.id)}><span>CHILD</span><code>{child.id}</code></button>)}{!relatives.length && <p>Immutable original · no recorded children</p>}</aside>
      <aside className="local-fork-focus" aria-labelledby="local-fork-focus-title"><span className="eyebrow">LOCAL FORK FOCUS</span><h2 id="local-fork-focus-title">{selected ? roleLabel(selected) : "No retained turn"}</h2>{selected ? <blockquote>{selected.content}</blockquote> : <p>No readable conversation content was retained.</p>}
        {linkedSelection && <section className="local-fork-focus__link"><strong>Linked investigation locus</strong>{linkedSelection.answerId && <code>answer · {linkedSelection.answerId}</code>}{linkedSelection.sourceId && <code>source · {linkedSelection.sourceId}</code>}{linkedSelection.differenceId && <code>difference · {linkedSelection.differenceId}</code>}{linkedSelection.compareRunId && <code>paired with · {linkedSelection.compareRunId}</code>}<p>The coordinate is retained. Token execution resolves the nearest recorded boundary; it does not pretend a text range is a checkpoint.</p></section>}
        <section className="local-fork-focus__actions"><h3>Choose investigation depth</h3><p>{selected?.kind === "response" ? "Inspect recorded token boundaries, exact-fork eligibility, and reconstructed replay separately." : "This context turn remains part of the immutable parent. Continue to the recorded response before staging a child."}</p><button type="button" className="primary-action" onClick={() => onOpenTokenExecution(selected?.kind === "response" ? 0 : undefined)}>Open token execution</button>{run.parentRunId && onOpenCompare && <button type="button" onClick={() => onOpenCompare(run.parentRunId!, run.id)}>Compare with parent</button>}</section>
      </aside>
    </div>
  </main>;
}
