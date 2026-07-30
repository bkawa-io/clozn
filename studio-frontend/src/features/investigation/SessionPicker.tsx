import { useEffect, useState, type FormEvent } from "react";
import { describeSessionTraceError, loadSessionList, type SessionSummary } from "../../data/sessionTrace";

/**
 * F3's entry point into the conversation investigation view: a directory of recorded sessions
 * (`GET /sessions`, F1) plus a "open by id" router for a session this list does not (yet) show --
 * `list_sessions()` is bounded (`clozn.runs.sessions.list_sessions`'s own docstring: "bounded, not
 * cursor-paginated"), so a session past that bound, or one a caller already knows the raw id/token for,
 * still needs a direct way in. Routing only: this component fires exactly one GET (the list) and never
 * inspects or validates a typed id beyond trimming it -- `clozn.runs.association.session_key()` accepts
 * both an opaque `session_...` id and a raw token to digest, so there is nothing useful to validate
 * client-side that the trace route itself will not already tell the user about honestly.
 */

type ListResource =
  | { status: "loading" }
  | { status: "failed"; message: string }
  | { status: "ready"; sessions: SessionSummary[] };

function formatEpochSeconds(ts: number | undefined): string {
  if (!ts) return "—";
  const date = new Date(ts * 1000);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

export function SessionPicker() {
  const [resource, setResource] = useState<ListResource>({ status: "loading" });
  const [jumpId, setJumpId] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setResource({ status: "loading" });
    void loadSessionList({ limit: 100, signal: controller.signal }).then((sessions) => {
      setResource({ status: "ready", sessions });
    }).catch((error: unknown) => {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setResource({ status: "failed", message: describeSessionTraceError(error) });
    });
    return () => controller.abort();
  }, []);

  function submitJump(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = jumpId.trim();
    if (!trimmed) return;
    location.hash = `#/sessions/${encodeURIComponent(trimmed)}/investigate`;
  }

  return (
    <section className="instrument investigation-picker" aria-labelledby="investigation-picker-title">
      <header className="instrument-head">
        <div>
          <span className="eyebrow">INVESTIGATION</span>
          <h1 id="investigation-picker-title">Conversation sessions</h1>
        </div>
        {resource.status === "ready" && (
          <strong>{resource.sessions.length} SESSION{resource.sessions.length === 1 ? "" : "S"}</strong>
        )}
      </header>

      <p className="investigation-boundary">
        Pick a session to open its investigation view -- a deterministic evidence timeline over its
        recorded turns (F2&apos;s `session-trace` route), never a chat replay. Nothing here re-runs the
        model.
      </p>

      <form className="investigation-jump" onSubmit={submitJump}>
        <label htmlFor="investigation-jump-input">
          <span>OPEN A SESSION BY ID</span>
          <input
            id="investigation-jump-input"
            type="text"
            value={jumpId}
            onChange={(event) => setJumpId(event.target.value)}
            placeholder="session_… or a raw conversation id"
          />
        </label>
        <button type="submit" disabled={!jumpId.trim()}>OPEN</button>
      </form>

      {resource.status === "loading" && <p className="investigation-empty">LOADING SESSIONS…</p>}
      {resource.status === "failed" && <p className="investigation-load-error" role="alert">{resource.message}</p>}
      {resource.status === "ready" && resource.sessions.length === 0 && (
        <p className="investigation-empty">No sessions recorded yet.</p>
      )}
      {resource.status === "ready" && resource.sessions.length > 0 && (
        <ul className="investigation-session-list" role="list" aria-label="Recorded sessions">
          {resource.sessions.map((session) => (
            <li key={session.id}>
              <a href={`#/sessions/${encodeURIComponent(session.id)}/investigate`}>
                <strong>{session.title || session.id}</strong>
                <span>{session.runCount ?? 0} TURN{session.runCount === 1 ? "" : "S"}</span>
                <time>{formatEpochSeconds(session.lastActivityTs ?? session.createdTs)}</time>
                {session.visibility === "hidden" && <span className="investigation-flag">HIDDEN</span>}
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
