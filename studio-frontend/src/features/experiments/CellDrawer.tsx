import { useEffect, useState } from "react";
import { loadExperimentCells, reproductionCommand } from "./api";
import { formatTimestamp, statusLabel } from "./format";
import { diffIdentity } from "./identityDiff";
import type { CellSelection } from "./urlState";
import type { CaseDef, ExperimentDetail, FullCell } from "./types";

interface CellDrawerProps {
  experimentId: string;
  detail: ExperimentDetail;
  selection: CellSelection;
  onClose: () => void;
  onSelectSeed: (seed: number) => void;
}

function findCase(detail: ExperimentDetail, suite: string, caseName: string): CaseDef | undefined {
  const suiteDef = detail.manifest.suites[suite as "target" | "guard"];
  return suiteDef?.cases.find((c) => c.name === caseName);
}

function CaseDefinition({ caseDef }: { caseDef: CaseDef | undefined }) {
  if (!caseDef) {
    return <p className="experiments-drawer-unavailable">CASE DEFINITION UNAVAILABLE -- not found in the loaded manifest</p>;
  }
  return (
    <div className="experiments-case-def">
      {caseDef.prompt != null && (
        <div className="experiments-message">
          <span>PROMPT</span>
          <p>{caseDef.prompt}</p>
        </div>
      )}
      {caseDef.messages?.map((message, i) => (
        <div className="experiments-message" key={i}>
          <span>{message.role.toUpperCase()}</span>
          <p>{message.content}</p>
        </div>
      ))}
      {caseDef.expect && (
        <div className="experiments-expect">
          <span>EXPECT</span>
          <pre>{JSON.stringify(caseDef.expect, null, 2)}</pre>
        </div>
      )}
      {!caseDef.expect && caseDef.prove == null && (
        <p className="experiments-drawer-note">No `expect`/`prove` rule -- this case is unscored by design.</p>
      )}
    </div>
  );
}

function OutputPane({ label, cell }: { label: string; cell: FullCell | undefined }) {
  if (!cell) {
    return (
      <div className="experiments-output-pane">
        <header><span>{label}</span><strong>UNAVAILABLE</strong></header>
        <p className="experiments-drawer-unavailable">No cell recorded at this coordinate.</p>
      </div>
    );
  }
  return (
    <div className={`experiments-output-pane is-${cell.status}`}>
      <header>
        <span>{label} · {cell.variant}</span>
        <strong>{statusLabel(cell.status)}</strong>
      </header>
      {cell.response != null ? <p className="experiments-response">{cell.response}</p>
        : <p className="experiments-drawer-unavailable">NO RESPONSE RECORDED</p>}
      {cell.error && <p className="experiments-error-text">ERROR: {cell.error}</p>}
      {cell.assertions.length > 0 && (
        <ul className="experiments-assertions">
          {cell.assertions.map((a, i) => (
            <li key={i} className={`is-${a.status}`}>{a.status.toUpperCase()} · {a.check}</li>
          ))}
        </ul>
      )}
      <div className="experiments-receipt-line">
        {cell.receipts != null
          ? <span className="is-available">RECEIPT AVAILABLE ({String(cell.receipts.mode ?? "recorded")})</span>
          : <span className="is-unavailable">NO RECEIPT RECORDED</span>}
      </div>
    </div>
  );
}

function IdentityDiffTable({ baseline, candidate }: { baseline?: FullCell; candidate?: FullCell }) {
  if (!baseline || !candidate) {
    return <p className="experiments-drawer-unavailable">IDENTITY DIFF UNAVAILABLE -- one side has no recorded run.</p>;
  }
  const fields = diffIdentity(baseline.run?.identity, candidate.run?.identity);
  if (!baseline.run || !candidate.run) {
    return <p className="experiments-drawer-unavailable">IDENTITY UNAVAILABLE -- {!baseline.run ? "baseline" : "candidate"} cell has no run record (generation error).</p>;
  }
  if (fields.length === 0) {
    return <p className="experiments-drawer-note">Neither run carries an identity block.</p>;
  }
  return (
    <table className="experiments-identity-table">
      <thead>
        <tr><th>FIELD</th><th>BASELINE</th><th>CANDIDATE</th></tr>
      </thead>
      <tbody>
        {fields.map((field) => (
          <tr key={field.path} className={`is-${field.status}`}>
            <td>{field.path}</td>
            <td>{field.base ?? "—"}</td>
            <td>{field.candidate ?? "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function copy(text: string) {
  try {
    void navigator.clipboard?.writeText(text);
  } catch {
    // Clipboard access can be denied by permissions policy; the command text is still selectable.
  }
}

function ReproCommand({ label, command }: { label: string; command: string }) {
  return (
    <div className="experiments-repro">
      <span>{label}</span>
      <div className="experiments-repro-line">
        <code>{command}</code>
        <button type="button" onClick={() => copy(command)}>COPY</button>
      </div>
    </div>
  );
}

export function CellDrawer({ experimentId, detail, selection, onClose, onSelectSeed }: CellDrawerProps) {
  const [cells, setCells] = useState<FullCell[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    void loadExperimentCells(
      experimentId,
      { suite: selection.suite, case: selection.case, seed: selection.seed },
      controller.signal,
    ).then((next) => {
      if (controller.signal.aborted) return;
      setCells(next);
      setStatus("ready");
    }).catch(() => {
      if (!controller.signal.aborted) setStatus("error");
    });
    return () => controller.abort();
  }, [experimentId, selection.suite, selection.case, selection.seed]);

  const baselineVariant = detail.summary.baselineVariant ?? detail.manifest.baselineVariant;
  const baselineCell = cells.find((c) => c.variant === baselineVariant);
  const candidateCell = cells.find((c) => c.variant === selection.variant);
  const isBaselineSelected = selection.variant === baselineVariant;
  const caseDef = findCase(detail, selection.suite, selection.case);

  return (
    <aside className="instrument experiments-drawer" aria-labelledby="experiments-drawer-title">
      <header className="instrument-head compact experiments-drawer-head">
        <div>
          <span className="eyebrow">{selection.suite.toUpperCase()} CELL</span>
          <h2 id="experiments-drawer-title">{selection.case}</h2>
        </div>
        <div className="experiments-drawer-actions">
          {detail.seeds.length > 1 && (
            <label className="experiments-seed-picker">
              <span>SEED</span>
              <select value={selection.seed} onChange={(e) => onSelectSeed(Number(e.target.value))}>
                {detail.seeds.map((seed) => <option value={seed} key={seed}>{seed}</option>)}
              </select>
            </label>
          )}
          <button type="button" onClick={onClose} aria-label="Close cell drawer">×</button>
        </div>
      </header>

      {status === "loading" && <div className="experiments-drawer-state">LOADING CELL</div>}
      {status === "error" && <div className="experiments-drawer-state is-error">CELL FETCH FAILED</div>}

      {status === "ready" && (
        <div className="experiments-drawer-body">
          <section>
            <h3>CASE DEFINITION</h3>
            <CaseDefinition caseDef={caseDef} />
          </section>

          <section className="experiments-outputs">
            <h3>OUTPUT</h3>
            {isBaselineSelected ? (
              <OutputPane label="BASELINE" cell={baselineCell} />
            ) : (
              <div className="experiments-outputs-grid">
                <OutputPane label="BASELINE" cell={baselineCell} />
                <OutputPane label="CANDIDATE" cell={candidateCell} />
              </div>
            )}
          </section>

          {!isBaselineSelected && (
            <section>
              <h3>IDENTITY DIFF</h3>
              <p className="experiments-drawer-note">
                Raw field comparison only -- not a causal explanation of any behavior difference.
              </p>
              <IdentityDiffTable baseline={baselineCell} candidate={candidateCell} />
            </section>
          )}

          <section>
            <h3>LOCAL REPRODUCTION</h3>
            {!isBaselineSelected && baselineCell && (
              <ReproCommand
                label="BASELINE"
                command={reproductionCommand(experimentId, {
                  suite: selection.suite, case: selection.case, variant: baselineVariant, seed: selection.seed,
                })}
              />
            )}
            <ReproCommand
              label={isBaselineSelected ? "BASELINE" : "CANDIDATE"}
              command={reproductionCommand(experimentId, {
                suite: selection.suite, case: selection.case, variant: selection.variant, seed: selection.seed,
              })}
            />
            <p className="experiments-drawer-note">
              Assumes the default result path (~/.clozn/experiments/{experimentId}.json); a result saved
              with a custom --out will need its own path substituted.
            </p>
          </section>

          {candidateCell?.run?.identity?.capturedAt && (
            <p className="experiments-drawer-note">Candidate run captured {formatTimestamp(candidateCell.run.identity.capturedAt)}.</p>
          )}
        </div>
      )}
    </aside>
  );
}
