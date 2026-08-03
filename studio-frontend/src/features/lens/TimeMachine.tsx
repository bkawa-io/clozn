import { useEffect, useRef, useState } from "react";
import {
  branchTimeMachine,
  continueTimeMachine,
  exactBranchTimeMachine,
  loadTimeMachine,
  verifyTimeMachine,
  type TimeMachineContinuationReceipt,
  type TimeMachineDocument,
} from "../../data/timeMachine";
import {
  formatSnapshotBytes,
  pinSnapshot,
  previewSnapshot,
  SnapshotRequestError,
  unpinSnapshot,
} from "../../data/snapshots";
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
  const [continuationQuestion, setContinuationQuestion] = useState("");
  const [continuationMaxTokens, setContinuationMaxTokens] = useState("256");
  const [continuing, setContinuing] = useState(false);
  const [continuationReceipt, setContinuationReceipt] = useState<TimeMachineContinuationReceipt | null>(null);
  const [continuationError, setContinuationError] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [verificationMessage, setVerificationMessage] = useState<string | null>(null);
  const [pinBusy, setPinBusy] = useState<"preview" | "pin" | "unpin" | null>(null);
  const [pinPreview, setPinPreview] = useState<{
    sourceRunId: string;
    sizeBytes?: number;
    envelopeBytes?: number;
  } | null>(null);
  const [pinMessage, setPinMessage] = useState<string | null>(null);
  const [cascadeUnpin, setCascadeUnpin] = useState<{ sourceRunId: string; children: string[] } | null>(null);
  const continuationController = useRef<AbortController | null>(null);
  const continuationRequest = useRef(0);

  useEffect(() => {
    const controller = new AbortController();
    continuationController.current?.abort();
    continuationController.current = null;
    continuationRequest.current += 1;
    setStatus("loading");
    setDocument(null);
    setChildId(null);
    setBranchError(null);
    setExactChildId(null);
    setExactBranchError(null);
    setContinuationQuestion("");
    setContinuationMaxTokens("256");
    setContinuing(false);
    setContinuationReceipt(null);
    setContinuationError(null);
    setVerificationMessage(null);
    setPinPreview(null);
    setPinMessage(null);
    setCascadeUnpin(null);
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

  const selected = document?.turns.find((turn) => turn.turn === selectedTurn);
  const selectedSource = selected?.source;
  const sourceRunId = selectedSource?.status === "available" ? selectedSource.runId : undefined;
  const hasStoredPin = selectedSource?.durablePin.status === "stored" && Boolean(selectedSource.durablePin.pin);
  const hasLiveCheckpoint = selected?.snapshot?.hasCache === true
    && selected.snapshot.runId === sourceRunId;
  const continuationMaxTokensValue = Number(continuationMaxTokens);
  const continuationInputValid = Boolean(continuationQuestion.trim())
    && Number.isInteger(continuationMaxTokensValue) && continuationMaxTokensValue > 0;
  const continuationUnavailableReasons = [
    ...(sourceRunId ? [] : [selectedSource?.reasons[0]?.message ?? "No exact source run is available for this turn."]),
    ...(sourceRunId && !hasStoredPin
      ? [
        selectedSource?.durablePin.reason.message
          ?? "No durable restart-safe checkpoint is recorded for this exact source.",
      ]
      : []),
  ];
  const continuationReady = Boolean(document && sourceRunId && hasStoredPin
    && continuationUnavailableReasons.length === 0 && continuationInputValid);

  async function refreshEligibility() {
    const next = await loadTimeMachine(runId);
    setDocument(next);
    setSelectedTurn((current) => next.turns.some((turn) => turn.turn === current)
      ? current
      : (next.turns[0]?.turn ?? 0));
  }

  async function branch() {
    if (!document || branching || continuing) return;
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
    if (!document || exactBranching || continuing || altUser.trim()) return;
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

  async function continueExactHistory() {
    if (!document || continuing || !continuationReady) return;
    continuationController.current?.abort();
    const controller = new AbortController();
    const requestId = continuationRequest.current + 1;
    continuationRequest.current = requestId;
    continuationController.current = controller;
    setContinuing(true);
    setContinuationReceipt(null);
    setContinuationError(null);
    try {
      const result = await continueTimeMachine(runId, {
        turn: selectedTurn,
        user: { content: continuationQuestion },
        maxTokens: continuationMaxTokensValue,
      }, controller.signal);
      if (controller.signal.aborted || continuationRequest.current !== requestId) return;
      setContinuationReceipt(result);
      if (result.status !== "completed") {
        setContinuationError(
          `${result.status.toUpperCase()} AT ${result.failure.stage.replaceAll("_", " ").toUpperCase()} `
          + `(${result.failure.code}): ${result.failure.message}`,
        );
      }
    } catch (error) {
      if (controller.signal.aborted || continuationRequest.current !== requestId) return;
      setContinuationError(
        error instanceof Error ? error.message : "The exact appended-turn continuation failed.",
      );
    } finally {
      if (!controller.signal.aborted && continuationRequest.current === requestId) {
        setContinuing(false);
        continuationController.current = null;
      }
    }
  }

  async function verify() {
    if (!document || verifying || continuing) return;
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

  async function previewPin() {
    if (!sourceRunId || pinBusy || continuing) return;
    setPinBusy("preview");
    setPinPreview(null);
    setPinMessage(null);
    setCascadeUnpin(null);
    try {
      const preview = await previewSnapshot(sourceRunId);
      setPinPreview({ sourceRunId, sizeBytes: preview.sizeBytes, envelopeBytes: preview.envelopeBytes });
    } catch (error) {
      setPinMessage(error instanceof Error ? error.message : "Durable checkpoint preview failed.");
    } finally {
      setPinBusy(null);
    }
  }

  async function pinSource() {
    if (!sourceRunId || pinBusy || continuing || pinPreview?.sourceRunId !== sourceRunId) return;
    setPinBusy("pin");
    setPinMessage(null);
    try {
      await pinSnapshot(sourceRunId);
      setPinPreview(null);
      await refreshEligibility();
      setPinMessage("Durable checkpoint recorded. The source run remains unchanged.");
    } catch (error) {
      setPinMessage(error instanceof Error ? error.message : "Durable checkpoint pin failed.");
    } finally {
      setPinBusy(null);
    }
  }

  async function unpinSource(cascade = false) {
    if (!sourceRunId || pinBusy || continuing) return;
    setPinBusy("unpin");
    setPinMessage(null);
    try {
      await unpinSnapshot(sourceRunId, cascade);
      setPinPreview(null);
      setCascadeUnpin(null);
      await refreshEligibility();
      setPinMessage("Durable checkpoint unpinned. Source and child runs remain immutable.");
    } catch (error) {
      if (
        error instanceof SnapshotRequestError
        && error.code === "snapshot_unpin_has_dependents"
      ) {
        const rawChildren = error.details?.children;
        const children = Array.isArray(rawChildren)
          ? rawChildren.filter((child): child is string => typeof child === "string")
          : [];
        setCascadeUnpin({ sourceRunId, children });
        setPinMessage(
          `This pinned source has ${children.length} direct child run${children.length === 1 ? "" : "s"}; unpin was refused.`,
        );
      } else {
        setPinMessage(error instanceof Error ? error.message : "Durable checkpoint unpin failed.");
      }
    } finally {
      setPinBusy(null);
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
              : "Structural branching regenerates the transcript. Exact same-prompt replay and exact appended-turn continuation are separate, source-bound worker actions."}
          </p>
          <div className="lens-time-machine-controls">
            <label>
              <span>BRANCH FROM TURN</span>
              <select
                aria-label="Branch from turn"
                value={selectedTurn}
                disabled={continuing}
                onChange={(event) => {
                  setSelectedTurn(Number(event.target.value));
                  setPinPreview(null);
                  setPinMessage(null);
                  setCascadeUnpin(null);
                  setContinuationReceipt(null);
                  setContinuationError(null);
                }}
              >
                {document.turns.map((turn) => (
                  <option key={turn.turn} value={turn.turn} disabled={!turn.branchEligible}>
                    TURN {turn.turn + 1} · {turn.replayFidelity.replaceAll("_", " ")}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <section className="lens-time-machine-source" aria-label="Exact source checkpoint">
            <span className="eyebrow">EXACT SOURCE CHECKPOINT</span>
            {sourceRunId ? (
              <>
                <p>
                  {selectedSource?.scope === "session_turn_prompt_boundary"
                    ? "Organic historical source"
                    : "Requested run source"}{" "}
                  <code>{sourceRunId}</code>
                </p>
                <p className={hasStoredPin ? "is-ready" : "is-unavailable"}>
                  {hasStoredPin
                    ? `RESTART-SAFE PIN RECORDED · CHECKED WHEN USED · ${formatSnapshotBytes(selectedSource?.durablePin.pin?.envelopeBytes)}`
                    : "RESTART-SAFE PIN UNAVAILABLE"}
                </p>
                <p className="lens-time-machine-source-detail">{selectedSource?.durablePin.reason.message}</p>
                <p className={hasLiveCheckpoint ? "is-ready" : "is-unavailable"}>
                  {hasLiveCheckpoint
                    ? "LIVE CHECKPOINT CANDIDATE RETAINED · NOT RESTART-SAFE · RECHECKED WHEN USED"
                    : "NO RETAINED LIVE CHECKPOINT FOR THIS EXACT SOURCE"}
                </p>
                {hasStoredPin ? (
                  <div className="lens-time-machine-source-actions">
                    {cascadeUnpin?.sourceRunId === sourceRunId ? (
                      <>
                        <button
                          type="button"
                          className="is-danger"
                          disabled={pinBusy !== null || continuing}
                          onClick={() => void unpinSource(true)}
                        >
                          {pinBusy === "unpin"
                            ? "UNPINNING…"
                            : `CASCADE UNPIN (${cascadeUnpin.children.length} CHILD${cascadeUnpin.children.length === 1 ? "" : "REN"})`}
                        </button>
                        <button type="button" disabled={pinBusy !== null || continuing} onClick={() => setCascadeUnpin(null)}>KEEP PIN</button>
                      </>
                    ) : (
                      <button type="button" disabled={pinBusy !== null || continuing} onClick={() => void unpinSource()}>
                        {pinBusy === "unpin" ? "CHECKING CHILDREN…" : "UNPIN DURABLE CHECKPOINT"}
                      </button>
                    )}
                  </div>
                ) : (
                  <div className="lens-time-machine-source-actions">
                    <button type="button" disabled={pinBusy !== null || continuing} onClick={() => void previewPin()}>
                      {pinBusy === "preview" ? "CHECKING SIZE…" : "PREVIEW DURABLE PIN"}
                    </button>
                    {pinPreview?.sourceRunId === sourceRunId && (
                      <>
                        <button type="button" disabled={pinBusy !== null || continuing} onClick={() => void pinSource()}>
                          {pinBusy === "pin"
                            ? "PINNING…"
                            : `PIN ${formatSnapshotBytes(pinPreview.envelopeBytes)}`}
                        </button>
                        <span className="lens-time-machine-pin-preview" role="status">
                          Writes {formatSnapshotBytes(pinPreview.envelopeBytes)} ({formatSnapshotBytes(pinPreview.sizeBytes)} KV cache).
                        </span>
                      </>
                    )}
                  </div>
                )}
              </>
            ) : (
              <p className="is-unavailable">
                {selectedSource?.reasons[0]?.message ?? "No exact source run is available for this turn."}
              </p>
            )}
          </section>
          <section className="lens-time-machine-operation is-structural" aria-labelledby="structural-branch-title">
            <div>
              <span className="eyebrow">STRUCTURAL ALTERNATE-QUESTION BRANCH</span>
              <p id="structural-branch-title" className="lens-time-machine-note">
                Rebuilds the selected transcript boundary. This is not exact historical-state restoration.
              </p>
            </div>
            <div className="lens-time-machine-controls lens-time-machine-operation-controls">
            <label>
              <span>OPTIONAL REPLACEMENT QUESTION</span>
              <input
                aria-label="Optional replacement question"
                value={altUser}
                onChange={(event) => setAltUser(event.target.value)}
                placeholder="Leave blank to re-roll this turn"
              />
            </label>
            <button type="button" onClick={() => void branch()} disabled={branching || continuing || !document.eligible}>
              {branching ? "BRANCHING…" : "CREATE CHILD BRANCH"}
            </button>
            </div>
          </section>
          <section className="lens-time-machine-operation is-exact-replay" aria-labelledby="exact-replay-title">
            <div>
              <span className="eyebrow">EXACT SAME-PROMPT REPLAY</span>
              <p id="exact-replay-title" className="lens-time-machine-note">
                Replays the same selected prompt boundary. It cannot change the question.
              </p>
            </div>
            <div className="lens-time-machine-controls lens-time-machine-operation-controls">
            <button
              type="button"
              onClick={() => void exactBranch()}
              disabled={exactBranching || continuing || !document.eligible || !sourceRunId || Boolean(altUser.trim())}
              title={altUser.trim()
                ? "Exact child replay cannot change the question; clear the replacement first"
                : sourceRunId
                  ? "Restore and replay the selected prompt boundary through the exact worker path"
                  : selectedSource?.reasons[0]?.message ?? "No exact source run is available for this turn"}
            >
              {exactBranching ? "REPLAYING…" : "CREATE EXACT CHILD REPLAY"}
            </button>
            <button
              type="button"
              onClick={() => void verify()}
              disabled={verifying || continuing || !sourceRunId}
              title={sourceRunId
                ? selectedSource?.scope === "session_turn_prompt_boundary"
                  ? "Verify this historical turn from its exact organic session source"
                  : "Verify the full run at its prompt boundary"
                : selectedSource?.reasons[0]?.message ?? "No exact source run is available for this turn"}
            >
              {verifying ? "VERIFYING…" : "VERIFY EXACT PROMPT BOUNDARY"}
            </button>
            </div>
          </section>
          <section className="lens-time-machine-operation is-exact-continuation" aria-labelledby="exact-continuation-title">
            <div>
              <span className="eyebrow">EXACT APPENDED-TURN CONTINUATION</span>
              <p id="exact-continuation-title" className="lens-time-machine-note">
                Restores the exact source checkpoint, appends only a newly rendered user-turn suffix, then generates an immutable child. It never falls back to structural replay or historical-prefix re-prefill.
              </p>
            </div>
            <div className="lens-time-machine-continuation-preview" aria-live="polite">
              <p>
                REQUESTED PARENT <code>{runId}</code>{" "}→ SOURCE CHECKPOINT{" "}
                {sourceRunId ? <code>{sourceRunId}</code> : "UNRESOLVED"}{" "}→ NEW IMMUTABLE CHILD
              </p>
              {hasStoredPin && <p className="is-ready">DURABLE PIN IS RESTART-SAFE; IMPORT IDENTITY WILL BE RECHECKED.</p>}
              {!hasStoredPin && hasLiveCheckpoint && <p className="is-unavailable">A PROCESS-BOUND CACHE CANDIDATE EXISTS, BUT V1 EXACT CONTINUATION REQUIRES A DURABLE VERIFIED PIN.</p>}
              {continuationUnavailableReasons.length > 0 && (
                <ul className="lens-time-machine-unavailable" role="status">
                  {continuationUnavailableReasons.map((message) => <li key={message}>{message}</li>)}
                </ul>
              )}
            </div>
            <div className="lens-time-machine-controls lens-time-machine-operation-controls">
              <label>
                <span>NEW QUESTION TO APPEND</span>
                <textarea
                  aria-label="New question to append"
                  value={continuationQuestion}
                  disabled={continuing}
                  onChange={(event) => setContinuationQuestion(event.target.value)}
                  placeholder="Required — this becomes one new user turn"
                  required
                />
              </label>
              <label>
                <span>MAX OUTPUT TOKENS</span>
                <input
                  aria-label="Max output tokens"
                  type="number"
                  min="1"
                  step="1"
                  value={continuationMaxTokens}
                  disabled={continuing}
                  onChange={(event) => setContinuationMaxTokens(event.target.value)}
                />
              </label>
              <button
                type="button"
                onClick={() => void continueExactHistory()}
                disabled={continuing || !continuationReady}
                title={continuationUnavailableReasons[0]
                  ?? (!continuationQuestion.trim()
                    ? "Enter the new question that will be appended to the restored history"
                    : !Number.isInteger(continuationMaxTokensValue) || continuationMaxTokensValue < 1
                      ? "Set a positive whole-number output-token limit"
                      : "Restore the exact source checkpoint, append this new user turn, and generate")}
              >
                {continuing ? "CONTINUING…" : "CONTINUE EXACT HISTORY"}
              </button>
            </div>
          </section>
          {childId && <p className="lens-time-machine-success" role="status">Child branch created — <a href={`#/runs/${encodeURIComponent(childId)}`}>OPEN {childId.slice(-8)}</a></p>}
          {selected?.lastVerification?.sourceRunId &&
            selected.lastVerification?.requestedRunId &&
            selected.lastVerification?.sourceRunId !== selected.lastVerification?.requestedRunId && (
              <p className="lens-time-machine-provenance" role="status">
                LAST EXACT PROOF: TURN {((selected.lastVerification?.sourceTurn ?? selectedTurn) + 1)}
                {" "}from session run{" "}
                <code>{selected.lastVerification?.sourceRunId}</code>
                {" "}for requested run{" "}
                <code>{selected.lastVerification?.requestedRunId}</code>.
              </p>
            )}
          {exactChildId && <p className="lens-time-machine-success" role="status">Exact child replay created — <a href={"#/runs/" + encodeURIComponent(exactChildId)}>OPEN {exactChildId.slice(-8)}</a></p>}
          {continuationReceipt?.status === "completed" && (
            <>
              <p className="lens-time-machine-success" role="status">
                Exact appended-turn child created — <a href={"#/runs/" + encodeURIComponent(continuationReceipt.childLineage.childRunId)}>
                  OPEN {continuationReceipt.childLineage.childRunId.slice(-8)}
                </a>
              </p>
              <p className="lens-time-machine-provenance" role="status">
                EXACT CONTINUATION: REQUESTED PARENT <code>{continuationReceipt.childLineage.requestedParentRunId}</code>
                {" "}→ SOURCE CHECKPOINT <code>{continuationReceipt.childLineage.sourceCheckpointRunId}</code>
                {" "}→ CHILD <code>{continuationReceipt.childLineage.childRunId}</code>.
                {" "}{continuationReceipt.sourceCheckpoint.provenance === "durable_pin_import"
                  ? "Durable pin import was restart-safe."
                  : "Live checkpoint restoration was process-bound."}
              </p>
            </>
          )}
          {continuationError && <p className="lens-time-machine-note is-error" role="alert">{continuationError}</p>}
          {exactBranchError && <p className="lens-time-machine-note is-error" role="alert">{exactBranchError}</p>}
          {branchError && <p className="lens-time-machine-note is-error" role="alert">{branchError}</p>}
          {verificationMessage && <p className="lens-time-machine-note" role="status">{verificationMessage}</p>}
          {pinMessage && <p className={cascadeUnpin ? "lens-time-machine-note is-error" : "lens-time-machine-note"} role="status">{pinMessage}</p>}
        </>
      )}
    </section>
  );
}
