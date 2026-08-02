import { useEffect, useState } from "react";
import {
  branchTimeMachine,
  exactBranchTimeMachine,
  loadTimeMachine,
  verifyTimeMachine,
  type TimeMachineDocument,
} from "../../data/timeMachine";
import "../../styles/time-machine.css";

export function TimeMachine({ runId }: { runId: string }) {
  const [document, setDocument] = useState<TimeMachineDocument | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [selectedTurn, setSelectedTurn] = useState(0);
  const [altUser, setAltUser] = useState("");
  const [branching, setBranching] = useState(false);
  const [branchError, setBranchError] = useState<string | null>(null);
  const [childId, setChildId] = useState<string | null>(null);
  const [exactBranching, setExactBranching] = useState(false);
  const [exactBranchError, setExactBranchError] = useState<string | null>(null);
  const [exactChildId, setExactChildId] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [verificationMessage, setVerificationMessage] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    setDocument(null);
    setChildId(null);
    setBranchError(null);
    setExactChildId(null);
    setExactBranchError(null);
    setVerificationMessage(null);
    void loadTimeMachine(runId, controller.signal).then((next) => {
      if (controller.signal.aborted) return;
      setDocument(next);
      setSelectedTurn(next.turns[0]?.turn ?? 0);
      setStatus("ready");
    }).catch(() => {
      if (!controller.signal.aborted) setStatus("error");
    });
    return () => controller.abort();
  }, [runId]);

  async function branch() {
    if (!document || branching) return;
    setBranching(true);
    setBranchError(null);
    setChildId(null);
    try {
      const child = await branchTimeMachine(runId, selectedTurn, altUser);
      const id = typeof child.id === "string" ? child.id : null;
      if (!id) throw new Error("The branch response did not include a child run id.");
      setChildId(id);
    } catch (error) {
      setBranchError(error instanceof Error ? error.message : "The branch could not be created.");
    } finally {
      setBranching(false);
    }
  }

  async function exactBranch() {
    if (!document || exactBranching || altUser.trim()) return;
    setExactBranching(true);
    setExactBranchError(null);
    setExactChildId(null);
    try {
      const result = await exactBranchTimeMachine(runId, selectedTurn);
      const id = typeof result.child_run_id === "string" ? result.child_run_id : null;
      if (!id) throw new Error("The exact child replay did not include a child run id.");
      setExactChildId(id);
    } catch (error) {
      setExactBranchError(error instanceof Error ? error.message : "The exact child replay failed.");
    } finally {
      setExactBranching(false);
    }
  }

  async function verify() {
    if (!document || verifying) return;
    setVerifying(true);
    setVerificationMessage(null);
    try {
      const result = await verifyTimeMachine(runId, selectedTurn);
      setVerificationMessage(result.exactReplay
        ? result.sourceRunId && result.requestedRunId && result.sourceRunId !== result.requestedRunId
          ? "Exact prompt-boundary replay verified from session run " + result.sourceRunId
            + " for requested run " + result.requestedRunId + "."
          : "Exact prompt-boundary replay verified for this run."
        : result.reasons[0]?.message ?? "Exact replay could not be verified for this turn.");
    } catch (error) {
      setVerificationMessage(error instanceof Error ? error.message : "Exact replay verification failed.");
    } finally {
      setVerifying(false);
    }
  }

  return (
    <section className="instrument lens-time-machine" aria-labelledby="lens-time-machine-title">
      <header className="instrument-head compact">
        <div>
          <span className="eyebrow">ANSWER TIME MACHINE</span>
          <h2 id="lens-time-machine-title">Replay and branch</h2>
        </div>
        {document && <strong>{document.state.replaceAll("_", " ").toUpperCase()}</strong>}
      </header>
      {status === "loading" && <p className="lens-time-machine-note" role="status">CHECKING REPLAY ELIGIBILITY…</p>}
      {status === "error" && <p className="lens-time-machine-note is-error" role="alert">TIME MACHINE RECEIPT UNAVAILABLE</p>}
      {status === "ready" && document && (
        <>
          <p className="lens-time-machine-note">
            {document.exactReplay.eligible
              ? "Exact replay is eligible for this run."
              : "Exact replay is not available: current branching regenerates the transcript and does not restore exact KV state."}
          </p>
          <div className="lens-time-machine-controls">
            <label>
              <span>BRANCH FROM TURN</span>
              <select
                aria-label="Branch from turn"
                value={selectedTurn}
                onChange={(event) => setSelectedTurn(Number(event.target.value))}
              >
                {document.turns.map((turn) => (
                  <option key={turn.turn} value={turn.turn} disabled={!turn.branchEligible}>
                    TURN {turn.turn + 1} · {turn.replayFidelity.replaceAll("_", " ")}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>OPTIONAL REPLACEMENT QUESTION</span>
              <input
                aria-label="Optional replacement question"
                value={altUser}
                onChange={(event) => setAltUser(event.target.value)}
                placeholder="Leave blank to re-roll this turn"
              />
            </label>
            <button type="button" onClick={() => void branch()} disabled={branching || !document.eligible}>
              {branching ? "BRANCHING…" : "CREATE CHILD BRANCH"}
            </button>
            <button
              type="button"
              onClick={() => void exactBranch()}
              disabled={exactBranching || !document.eligible || Boolean(altUser.trim())}
              title={altUser.trim()
                ? "Exact child replay cannot change the question; clear the replacement first"
                : "Restore and replay the selected prompt boundary through the exact worker path"}
            >
              {exactBranching ? "REPLAYINGâ€¦" : "CREATE EXACT CHILD REPLAY"}
            </button>
            <button
              type="button"
              onClick={() => void verify()}
              disabled={verifying || selectedTurn !== (document.turns.at(-1)?.turn ?? -1)}
              title={selectedTurn === (document.turns.at(-1)?.turn ?? -1)
                ? "Verify the full run at its prompt boundary"
                : "Earlier turns do not have a persisted exact KV boundary yet"}
            >
              {verifying ? "VERIFYING…" : "VERIFY EXACT PROMPT BOUNDARY"}
            </button>
          </div>
          {childId && <p className="lens-time-machine-success" role="status">Child branch created — <a href={`#/runs/${encodeURIComponent(childId)}`}>OPEN {childId.slice(-8)}</a></p>}
          {document.turns[selectedTurn]?.lastVerification?.sourceRunId &&
            document.turns[selectedTurn].lastVerification?.requestedRunId &&
            document.turns[selectedTurn].lastVerification?.sourceRunId !== document.turns[selectedTurn].lastVerification?.requestedRunId && (
              <p className="lens-time-machine-provenance" role="status">
                LAST EXACT PROOF: TURN {((document.turns[selectedTurn].lastVerification?.sourceTurn ?? selectedTurn) + 1)}
                {" "}from session run{" "}
                <code>{document.turns[selectedTurn].lastVerification?.sourceRunId}</code>
                {" "}for requested run{" "}
                <code>{document.turns[selectedTurn].lastVerification?.requestedRunId}</code>.
              </p>
            )}
          {exactChildId && <p className="lens-time-machine-success" role="status">Exact child replay created — <a href={"#/runs/" + encodeURIComponent(exactChildId)}>OPEN {exactChildId.slice(-8)}</a></p>}
          {exactBranchError && <p className="lens-time-machine-note is-error" role="alert">{exactBranchError}</p>}
          {branchError && <p className="lens-time-machine-note is-error" role="alert">{branchError}</p>}
          {verificationMessage && <p className="lens-time-machine-note" role="status">{verificationMessage}</p>}
        </>
      )}
    </section>
  );
}
