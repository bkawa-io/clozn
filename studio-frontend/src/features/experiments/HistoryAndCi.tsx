import { useEffect, useState } from "react";
import { generateCiPreview, loadExperimentTrends } from "./api";
import { formatPassRate, formatTimestamp, shortId } from "./format";
import type { CompatibleTrends, ExperimentDetail, TrendPoint } from "./types";

function copy(text: string) {
  try {
    void navigator.clipboard?.writeText(text);
  } catch {
    // Text stays selectable if clipboard permissions are denied.
  }
}

function compactIdentity(point: TrendPoint): string[] {
  const labels: string[] = [];
  for (const [field, values] of Object.entries(point.identity)) {
    labels.push(`${field}=${values.map((value) => shortId(value)).join(",")}`);
  }
  if (point.vcs?.commit) labels.push(`commit=${shortId(point.vcs.commit)}`);
  if (point.vcs?.branch) labels.push(`branch=${point.vcs.branch}`);
  return labels;
}

function artifactExpired(point: TrendPoint): boolean {
  const expires = point.artifactProvenance?.expiresAt;
  if (!expires) return false;
  const timestamp = Date.parse(expires);
  return Number.isFinite(timestamp) && timestamp < Date.now();
}

function ArtifactLinks({ point }: { point: TrendPoint }) {
  const provenance = point.artifactProvenance;
  if (!provenance) return <span className="experiments-history-missing">not supplied</span>;
  const expired = artifactExpired(point);
  return (
    <span className="experiments-history-links">
      {expired && <b>EXPIRED</b>}
      {provenance.workflowUrl && (
        <a href={provenance.workflowUrl} target="_blank" rel="noreferrer">workflow</a>
      )}
      {provenance.artifactUrl && (
        <a href={provenance.artifactUrl} target="_blank" rel="noreferrer">artifact</a>
      )}
      {provenance.localOpenCommand && (
        <button type="button" onClick={() => copy(provenance.localOpenCommand!)}>COPY LOCAL OPEN</button>
      )}
    </span>
  );
}

export function HistoryPanel({ experimentId }: { experimentId: string }) {
  const [history, setHistory] = useState<CompatibleTrends | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    void loadExperimentTrends(experimentId, controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return;
        setHistory(value);
        setStatus("ready");
      })
      .catch(() => {
        if (!controller.signal.aborted) setStatus("error");
      });
    return () => controller.abort();
  }, [experimentId]);

  return (
    <section className="instrument experiments-history" aria-labelledby="experiments-history-title">
      <header className="instrument-head compact">
        <div>
          <span className="eyebrow">IDENTITY BEFORE OUTCOMES · COMPATIBLE FINGERPRINTS ONLY</span>
          <h2 id="experiments-history-title">History</h2>
        </div>
        {history && <code>{history.suiteFingerprint.algorithm}:{shortId(history.suiteFingerprint.sha256)}</code>}
      </header>
      {status === "loading" && <p className="experiments-history-state">LOADING HISTORY</p>}
      {status === "error" && <p className="experiments-history-state is-error">HISTORY UNAVAILABLE</p>}
      {status === "ready" && history && (
        <div className="experiments-history-table">
          <div className="experiments-history-head" aria-hidden="true">
            <span>RUN</span><span>IDENTITY</span><span>OUTCOMES</span><span>EVIDENCE</span>
          </div>
          {history.points.map((point) => (
            <div className="experiments-history-row" key={point.experimentId}>
              <span>
                <b>{formatTimestamp(point.createdAt)}</b>
                <small>{shortId(point.experimentId)}</small>
              </span>
              <span>
                {compactIdentity(point).length
                  ? compactIdentity(point).map((label) => <small key={label}>{label}</small>)
                  : <small>identity not supplied</small>}
              </span>
              <span>
                {Object.entries(point.aggregates).map(([variant, aggregate]) => (
                  <small key={variant}>
                    {variant}: T {formatPassRate(aggregate.target?.passRate ?? null)}
                    {" · "}G {formatPassRate(aggregate.guard?.passRate ?? null)}
                  </small>
                ))}
                <small>{point.errorCells} errors · {point.replicateInstability.coordinateCount} unstable</small>
              </span>
              <ArtifactLinks point={point} />
            </div>
          ))}
          {history.points.length === 0 && <p className="experiments-history-state">NO COMPATIBLE HISTORY</p>}
        </div>
      )}
    </section>
  );
}

interface Budgets {
  max_execution_errors: number;
  max_target_regressions: number;
  max_guard_regressions: number;
  min_target_gains: number;
}

export function CiPreviewPanel({ detail }: { detail: ExperimentDetail }) {
  const [resultPath, setResultPath] = useState("");
  const [suitePath, setSuitePath] = useState("");
  const [lockfilePath, setLockfilePath] = useState("");
  const [budgets, setBudgets] = useState<Budgets>({
    max_execution_errors: 0,
    max_target_regressions: 0,
    max_guard_regressions: 0,
    min_target_gains: 0,
  });
  const [yaml, setYaml] = useState("");
  const [cacheKey, setCacheKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [working, setWorking] = useState(false);

  const updateBudget = (name: keyof Budgets, value: string) => {
    setBudgets((current) => ({ ...current, [name]: Math.max(0, Number(value) || 0) }));
  };

  const generate = async () => {
    setWorking(true);
    setError(null);
    try {
      const preview = await generateCiPreview(detail.experimentId, {
        mode: "verify",
        result_path: resultPath,
        ...(suitePath ? { suite_path: suitePath } : {}),
        ...(lockfilePath ? { lockfile_path: lockfilePath } : {}),
        budgets,
      });
      setYaml(preview.workflowYaml);
      setCacheKey(preview.cacheKey);
    } catch (reason) {
      setYaml("");
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setWorking(false);
    }
  };

  return (
    <section className="instrument experiments-ci-preview" aria-labelledby="experiments-ci-title">
      <header className="instrument-head compact">
        <div>
          <span className="eyebrow">VERSIONED LOCAL INPUT CONTRACT · COPY ONLY</span>
          <h2 id="experiments-ci-title">CI configuration preview</h2>
        </div>
      </header>
      <div className="experiments-ci-body">
        <p>
          Repository paths are explicit. Studio generates a preview and never writes workflow files.
          Fingerprint {detail.suiteFingerprint
            ? ` ${detail.suiteFingerprint.algorithm}:${shortId(detail.suiteFingerprint.sha256)}`
            : " unavailable"}.
        </p>
        <div className="experiments-ci-fields">
          <label><span>RESULT PATH</span><input value={resultPath} onChange={(event) => setResultPath(event.target.value)} /></label>
          <label><span>SUITE PATH (OPTIONAL)</span><input value={suitePath} onChange={(event) => setSuitePath(event.target.value)} /></label>
          <label><span>LOCKFILE PATH (OPTIONAL)</span><input value={lockfilePath} onChange={(event) => setLockfilePath(event.target.value)} /></label>
          {(Object.keys(budgets) as (keyof Budgets)[]).map((name) => (
            <label key={name}>
              <span>{name.replaceAll("_", " ").toUpperCase()}</span>
              <input type="number" min={0} value={budgets[name]} onChange={(event) => updateBudget(name, event.target.value)} />
            </label>
          ))}
        </div>
        <button type="button" disabled={!resultPath || working} onClick={() => void generate()}>
          {working ? "GENERATING…" : "GENERATE WORKFLOW PREVIEW"}
        </button>
        {error && <p className="experiments-error-text">{error}</p>}
        {yaml && (
          <div className="experiments-ci-output">
            <span>CACHE KEY · {cacheKey}</span>
            <button type="button" onClick={() => copy(yaml)}>COPY YAML</button>
            <pre>{yaml}</pre>
          </div>
        )}
      </div>
    </section>
  );
}
