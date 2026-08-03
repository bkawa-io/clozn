import type { ReactNode } from "react";
import type { RunPerformance, RunSummary, RuntimeState } from "../../data/types";
import "./run-frame.css";

/** A stable, display-ready description of an artifact associated with a recorded run. */
export interface RunFrameArtifact {
  /** Use the artifact id for a stable list key; it is never displayed as the primary label. */
  id: string;
  label: string;
  kind?: string;
  detail?: string;
  status?: "available" | "pending" | "unavailable" | "error";
  href?: string;
}

export interface RunFrameLineageItem {
  id: string;
  label?: string;
  detail?: string;
  href?: string;
}

/**
 * The frame knows only enough about lineage to describe it. Navigation and loading remain the
 * responsibility of the feature that supplies the items.
 */
export interface RunFrameLineage {
  parent?: RunFrameLineageItem | null;
  children?: readonly RunFrameLineageItem[];
}

export interface RunFrameProps {
  /** A RuntimeState run record, narrowed to the fields the chrome actually presents. */
  run?: Pick<
    RunSummary,
    | "id"
    | "label"
    | "createdAt"
    | "source"
    | "client"
    | "model"
    | "substrate"
    | "finishReason"
    | "parentRunId"
    | "flags"
    | "warningCount"
  > | null;
  /** Overrides the state inferred from the run record (for example, while a live run is active). */
  status?: string;
  /** Overrides the recorded finish reason when the caller has a more current value. */
  finishReason?: string | null;
  /** Runtime snapshot, deliberately narrowed so the frame does not own run selection. */
  runtime?: Pick<RuntimeState, "status" | "engine"> | null;
  /** Optional recorded performance/runtime fields. */
  performance?: Pick<RunPerformance, "device" | "gpuLayers" | "samplerMode"> | null;
  artifacts?: readonly RunFrameArtifact[];
  /** Explicit warning messages. When omitted, the run record's flags are shown instead. */
  warnings?: readonly string[];
  lineage?: RunFrameLineage;
  /** Primary actions, normally links or buttons owned by the calling feature. */
  actions?: ReactNode;
  /** Secondary actions, kept apart from primary run operations in the header. */
  utilityActions?: ReactNode;
  /** The workspace content framed by this run chrome. */
  children?: ReactNode;
  title?: string;
  className?: string;
}

function humanize(value?: string | null) {
  if (!value) return "—";
  return value
    .trim()
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function shortId(id?: string) {
  if (!id) return "—";
  return id.length > 10 ? id.slice(-8) : id;
}

function runStatus(run: RunFrameProps["run"], status?: string) {
  if (status) return status;
  if (!run) return "unselected";
  if (run.flags.includes("error")) return "error";
  if (run.finishReason === "length" || run.flags.includes("truncated")) return "truncated";
  return run.finishReason ? "complete" : "recorded";
}

function statusClass(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "unknown";
}

function LineageLink({ item }: { item: RunFrameLineageItem }) {
  const label = item.label || `Run ${shortId(item.id)}`;
  const contents = <><strong>{label}</strong>{item.detail && <small>{item.detail}</small>}</>;
  return item.href
    ? <a className="run-frame-lineage-item" href={item.href}>{contents}</a>
    : <span className="run-frame-lineage-item">{contents}</span>;
}

function Artifact({ artifact }: { artifact: RunFrameArtifact }) {
  const contents = (
    <>
      <span>{artifact.kind ? `${humanize(artifact.kind)} · ` : ""}{artifact.label}</span>
      <small>{artifact.detail ?? humanize(artifact.status ?? "available")}</small>
    </>
  );
  return artifact.href
    ? <a className={`run-frame-artifact is-${artifact.status ?? "available"}`} href={artifact.href}>{contents}</a>
    : <span className={`run-frame-artifact is-${artifact.status ?? "available"}`}>{contents}</span>;
}

/**
 * Presentational workspace chrome for a single run. It does not fetch, select, or mutate a run;
 * callers supply the record, related summaries, and action content they want to expose.
 */
export function RunFrame({
  run,
  status: statusOverride,
  finishReason: finishReasonOverride,
  runtime,
  performance,
  artifacts = [],
  warnings,
  lineage,
  actions,
  utilityActions,
  children,
  title = "Run workspace",
  className,
}: RunFrameProps) {
  const status = runStatus(run, statusOverride);
  const finishReason = finishReasonOverride === undefined ? run?.finishReason : finishReasonOverride;
  const warningItems = warnings ?? run?.flags ?? [];
  const parent = lineage?.parent ?? (run?.parentRunId
    ? { id: run.parentRunId, label: `Parent ${shortId(run.parentRunId)}` }
    : null);
  const childrenInLineage = lineage?.children ?? [];
  const runtimeModel = runtime?.engine?.model;
  const model = run?.model || runtimeModel || "Unavailable";
  const runtimeState = runtime?.status ? humanize(runtime.status) : "Not connected";

  return (
    <section
      className={["run-frame", `status-${statusClass(status)}`, className].filter(Boolean).join(" ")}
      aria-label={title}
    >
      <header className="run-frame-header">
        <div className="run-frame-identity">
          <span className="eyebrow">RECORDED RUN</span>
          <div>
            <h1>{run?.label || "No run selected"}</h1>
            {run?.id && <code title={run.id}>Run {shortId(run.id)}</code>}
          </div>
          {run?.createdAt && <time dateTime={run.createdAt}>{run.createdAt}</time>}
        </div>

        <div className="run-frame-state" aria-label="Run status">
          <span className={`run-frame-status is-${statusClass(status)}`}>{humanize(status)}</span>
          <span><b>FINISH</b>{humanize(finishReason)}</span>
        </div>

        {(actions || utilityActions) && (
          <div className="run-frame-actions">
            {actions && <div className="run-frame-primary-actions">{actions}</div>}
            {utilityActions && <div className="run-frame-utility-actions">{utilityActions}</div>}
          </div>
        )}
      </header>

      <dl className="run-frame-summary" aria-label="Run environment summary">
        <div><dt>MODEL</dt><dd>{model}</dd></div>
        <div><dt>ARTIFACTS</dt><dd>{artifacts.length ? `${artifacts.length} RECORDED` : "NONE RECORDED"}</dd></div>
        <div><dt>RUNTIME</dt><dd>{runtimeState}</dd></div>
        <div><dt>DEVICE</dt><dd>{performance?.device ?? run?.substrate ?? "—"}</dd></div>
        {performance?.gpuLayers !== undefined && <div><dt>GPU LAYERS</dt><dd>{performance.gpuLayers}</dd></div>}
        {performance?.samplerMode && <div><dt>SAMPLER</dt><dd>{humanize(performance.samplerMode)}</dd></div>}
      </dl>

      {(warningItems.length > 0 || (warnings === undefined && (run?.warningCount ?? 0) > 0)) && (
        <section className="run-frame-warnings" aria-label="Run warnings">
          <span>WARNINGS</span>
          <div>
            {warningItems.map((warning, index) => <p key={`${warning}-${index}`}>{humanize(warning)}</p>)}
            {warnings === undefined && (run?.warningCount ?? 0) > warningItems.length && (
              <p>{run!.warningCount - warningItems.length} additional recorded warning{run!.warningCount - warningItems.length === 1 ? "" : "s"}</p>
            )}
          </div>
        </section>
      )}

      {(artifacts.length > 0 || parent || childrenInLineage.length > 0) && (
        <div className="run-frame-context">
          {artifacts.length > 0 && (
            <section className="run-frame-artifacts" aria-label="Run artifacts">
              <span>ARTIFACTS</span>
              <div>{artifacts.map((artifact) => <Artifact artifact={artifact} key={artifact.id} />)}</div>
            </section>
          )}

          {(parent || childrenInLineage.length > 0) && (
            <section className="run-frame-lineage" aria-label="Run lineage">
              <span>LINEAGE</span>
              <div>
                {parent && <><small>PARENT</small><LineageLink item={parent} /></>}
                {childrenInLineage.length > 0 && (
                  <>
                    <small>CHILD{childrenInLineage.length === 1 ? "" : "REN"}</small>
                    {childrenInLineage.map((child) => <LineageLink item={child} key={child.id} />)}
                  </>
                )}
              </div>
            </section>
          )}
        </div>
      )}

      {children && <div className="run-frame-body">{children}</div>}
    </section>
  );
}
