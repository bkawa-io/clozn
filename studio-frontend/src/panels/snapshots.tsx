import { useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "../components/Icon";
import {
  formatSnapshotBytes,
  loadSnapshots,
  pinSnapshot,
  previewSnapshot,
  unpinSnapshot,
  type SnapshotManifest,
} from "../data/snapshots";
import type { PanelContext, StudioPanel } from "./types";
import "../styles/snapshots.css";

type LoadState = "idle" | "loading" | "ready" | "failed";

function SnapshotCard({
  snapshot,
  pending,
  onUnpin,
  onConfirm,
}: {
  snapshot: SnapshotManifest;
  pending: boolean;
  onUnpin: () => void;
  onConfirm: (cascade: boolean) => void;
}) {
  return (
    <article className="snapshot-card" aria-label={`Pinned snapshot ${snapshot.runId}`}>
      <header>
        <div>
          <span className="eyebrow">PINNED CHECKPOINT</span>
          <h2>{snapshot.runId}</h2>
        </div>
        <span className="snapshot-pin-id">{snapshot.pinId.slice(-8)}</span>
      </header>
      <dl className="snapshot-facts">
        <div><dt>PINNED</dt><dd>{snapshot.pinnedAt}</dd></div>
        <div><dt>MODEL</dt><dd>{snapshot.identity.architecture ?? "unreported"}</dd></div>
        <div><dt>KV CACHE</dt><dd>{formatSnapshotBytes(snapshot.blob.kvBytes)}</dd></div>
        <div><dt>POSITION</dt><dd>{snapshot.state.nPast ?? "unreported"}</dd></div>
      </dl>
      {snapshot.note && <p className="snapshot-note">{snapshot.note}</p>}
      <div className="snapshot-card-actions">
        {pending ? (
          <>
            <span className="snapshot-confirm-copy">Children may depend on this pin.</span>
            <button type="button" className="is-danger" onClick={() => onConfirm(false)}>CONFIRM UNPIN</button>
            <button type="button" className="is-danger" onClick={() => onConfirm(true)}>CASCADE UNPIN</button>
            <button type="button" onClick={onUnpin}>CANCEL</button>
          </>
        ) : (
          <button type="button" onClick={onUnpin}>UNPIN</button>
        )}
      </div>
    </article>
  );
}

export function SnapshotsPanel({ runtime, params }: PanelContext) {
  const [state, setState] = useState<LoadState>("idle");
  const [snapshots, setSnapshots] = useState<SnapshotManifest[]>([]);
  const [selectedRun, setSelectedRun] = useState(params.runId ?? runtime.runs[0]?.id ?? "");
  const [note, setNote] = useState("");
  const [preview, setPreview] = useState<{ sizeBytes?: number; envelopeBytes?: number } | null>(null);
  const [busy, setBusy] = useState<"preview" | "pin" | "unpin" | null>(null);
  const [pendingUnpin, setPendingUnpin] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const requestRef = useRef(0);

  const availableRuns = useMemo(() => runtime.runs.filter((run) => run.id), [runtime.runs]);

  async function refresh() {
    const requestId = ++requestRef.current;
    setState("loading");
    try {
      const document = await loadSnapshots();
      if (requestRef.current !== requestId) return;
      setSnapshots(document.snapshots);
      setState("ready");
    } catch (error) {
      if (requestRef.current !== requestId) return;
      setState("failed");
      setMessage(error instanceof Error ? error.message : "snapshot list unavailable");
    }
  }

  useEffect(() => { void refresh(); }, []);

  useEffect(() => {
    if (selectedRun && availableRuns.some((run) => run.id === selectedRun)) return;
    setSelectedRun(availableRuns[0]?.id ?? "");
  }, [availableRuns, selectedRun]);

  async function handlePreview() {
    if (!selectedRun || busy) return;
    setBusy("preview");
    setMessage(null);
    try {
      const result = await previewSnapshot(selectedRun, note);
      setPreview(result);
    } catch (error) {
      setPreview(null);
      setMessage(error instanceof Error ? error.message : "snapshot preview failed");
    } finally {
      setBusy(null);
    }
  }

  async function handlePin() {
    if (!selectedRun || busy) return;
    setBusy("pin");
    setMessage(null);
    try {
      await pinSnapshot(selectedRun, note);
      setPreview(null);
      setMessage("Checkpoint pinned. The original run remains unchanged.");
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "snapshot pin failed");
    } finally {
      setBusy(null);
    }
  }

  async function handleUnpin(runId: string, cascade: boolean) {
    setBusy("unpin");
    setMessage(null);
    try {
      await unpinSnapshot(runId, cascade);
      setPendingUnpin(null);
      setMessage(`Unpinned ${runId}.`);
      await refresh();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "snapshot unpin failed");
    } finally {
      setBusy(null);
    }
  }

  return (
    <main className="snapshots-workspace" aria-labelledby="snapshots-title">
      <header className="instrument-head snapshots-head">
        <div>
          <span className="eyebrow">ANSWER TIME MACHINE</span>
          <h1 id="snapshots-title">Durable snapshots</h1>
        </div>
        <span className="mode-chip">{state === "loading" ? "LOADING" : `${snapshots.length} PINNED`}</span>
      </header>
      <p className="snapshots-boundary">
        Pin a run's checkpoint before a risky experiment. Studio shows the storage cost first; pinning
        creates durable evidence and never mutates the source run.
      </p>

      <section className="snapshot-pin-form" aria-labelledby="snapshot-pin-title">
        <header className="section-title"><h2 id="snapshot-pin-title">Pin a run</h2><span>PREVIEW FIRST</span></header>
        <label htmlFor="snapshot-run">RUN</label>
        <select id="snapshot-run" value={selectedRun} onChange={(event) => { setSelectedRun(event.target.value); setPreview(null); }}>
          <option value="">No recorded runs available</option>
          {availableRuns.map((run) => <option value={run.id} key={run.id}>{run.label}</option>)}
        </select>
        <label htmlFor="snapshot-note">NOTE <span>(OPTIONAL)</span></label>
        <input id="snapshot-note" value={note} onChange={(event) => setNote(event.target.value)} placeholder="before the risky rewrite" />
        <div className="snapshot-pin-actions">
          <button type="button" disabled={!selectedRun || busy !== null} onClick={() => void handlePreview()}>
            {busy === "preview" ? "CHECKING SIZE…" : "PREVIEW PIN"}
          </button>
          {preview && (
            <button type="button" className="is-primary" disabled={busy !== null} onClick={() => void handlePin()}>
              {busy === "pin" ? "PINNING…" : `PIN ${formatSnapshotBytes(preview.envelopeBytes)}`}
            </button>
          )}
        </div>
        {preview && <p className="snapshot-preview" role="status">This will write {formatSnapshotBytes(preview.envelopeBytes)} ({formatSnapshotBytes(preview.sizeBytes)} KV cache).</p>}
      </section>

      {message && <p className="snapshot-message" role="status">{message}</p>}
      {state === "failed" && <p className="snapshot-message is-error" role="alert">{message ?? "snapshot list unavailable"}</p>}
      {state === "ready" && snapshots.length === 0 && <p className="snapshot-empty">No durable snapshots are pinned yet.</p>}
      <section className="snapshot-list" aria-label="Pinned snapshots">
        {snapshots.map((snapshot) => (
          <SnapshotCard
            key={snapshot.runId}
            snapshot={snapshot}
            pending={pendingUnpin === snapshot.runId}
            onUnpin={() => { if (busy === null) setPendingUnpin((current) => current === snapshot.runId ? null : snapshot.runId); }}
            onConfirm={(cascade) => void handleUnpin(snapshot.runId, cascade)}
          />
        ))}
      </section>
    </main>
  );
}

const panel: StudioPanel = {
  id: "snapshots",
  navLabel: "Snapshots",
  order: 35,
  icon: () => <Icon name="model" />,
  match: (hash): Record<string, string> | null => {
    const deep = hash.match(/^#\/snapshots\/([^/]+)\/?$/);
    if (deep) return { runId: decodeURIComponent(deep[1]) };
    return /^#\/snapshots\/?$/.test(hash) ? {} : null;
  },
  routeName: () => "SNAPSHOTS",
  Component: SnapshotsPanel,
};

export default panel;
