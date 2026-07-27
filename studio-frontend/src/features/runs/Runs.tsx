import { useEffect, useMemo, useState } from "react";
import { loadRunFacts, loadRunFamily } from "../../data/api";
import type { RunFacts, RunSummary, RuntimeState } from "../../data/types";

interface RunsProps {
  runtime: RuntimeState;
  inspectorOpen: boolean;
}

type StatusFilter = "all" | "complete" | "truncated" | "error" | "flagged";

function shortId(id: string) {
  return id.slice(-6);
}

function timestamp(value: string) {
  if (!value || value === "—") return "—";
  const match = value.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/);
  return match ? `${match[2]}/${match[3]} ${match[4]}:${match[5]}:${match[6]}` : value;
}

function runStatus(run: RunSummary) {
  if (run.flags.includes("error")) return "ERROR";
  if (run.finishReason === "length" || run.flags.includes("truncated")) return "TRUNCATED";
  if (run.finishReason) return "COMPLETE";
  return "RECORDED";
}

function matchesStatus(run: RunSummary, status: StatusFilter) {
  if (status === "all") return true;
  if (status === "flagged") return run.flags.length > 0 || run.warningCount > 0;
  return runStatus(run).toLowerCase() === status;
}

function parentLabel(run: RunSummary, runs: RunSummary[]) {
  if (!run.parentRunId) return "ORIGINAL";
  const parent = runs.find((candidate) => candidate.id === run.parentRunId);
  return parent ? `FROM ${shortId(parent.id)}` : `FROM ${shortId(run.parentRunId)}`;
}

function LineageNode({
  run,
  byParent,
  selectedId,
  onSelect,
}: {
  run: RunSummary;
  byParent: Map<string, RunSummary[]>;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const children = byParent.get(run.id) ?? [];
  return (
    <div className="runs-tree-node">
      <button
        type="button"
        className={run.id === selectedId ? "is-selected" : ""}
        onClick={() => onSelect(run.id)}
        aria-current={run.id === selectedId ? "true" : undefined}
      >
        <i />
        <span>{shortId(run.id)}</span>
        <small>{run.source.toUpperCase()}</small>
      </button>
      {children.length > 0 && (
        <div className="runs-tree-children">
          {children.map((child) => (
            <LineageNode
              key={child.id}
              run={child}
              byParent={byParent}
              selectedId={selectedId}
              onSelect={onSelect}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export function Runs({ runtime, inspectorOpen }: RunsProps) {
  const [selectedId, setSelectedId] = useState("");
  const [family, setFamily] = useState<RunSummary[]>([]);
  const [facts, setFacts] = useState<RunFacts>({ tokenCount: 0, traceAvailable: false });
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("all");
  const [status, setStatus] = useState<StatusFilter>("all");
  const [compareA, setCompareA] = useState("");
  const [compareB, setCompareB] = useState("");

  useEffect(() => {
    if (!runtime.runs.length) return;
    setSelectedId((current) => current || runtime.runs[0].id);
  }, [runtime.runs]);

  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    setFacts({ tokenCount: 0, traceAvailable: false });
    void Promise.all([
      loadRunFamily(selectedId, controller.signal),
      loadRunFacts(selectedId, controller.signal),
    ]).then(([nextFamily, nextFacts]) => {
      if (controller.signal.aborted) return;
      setFamily(nextFamily);
      setFacts(nextFacts);
    }).catch(() => {
      if (!controller.signal.aborted) setFacts({ tokenCount: 0, traceAvailable: false });
    });
    return () => controller.abort();
  }, [selectedId]);

  const sources = useMemo(
    () => [...new Set(runtime.runs.map((run) => run.source))].sort(),
    [runtime.runs],
  );
  const visibleRuns = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return runtime.runs.filter((run) => {
      if (source !== "all" && run.source !== source) return false;
      if (!matchesStatus(run, status)) return false;
      if (!needle) return true;
      return [
        run.id,
        run.prompt,
        run.response,
        run.model,
        run.source,
        ...run.flags,
      ].some((value) => value.toLowerCase().includes(needle));
    });
  }, [query, runtime.runs, source, status]);

  const selected = runtime.runs.find((run) => run.id === selectedId)
    ?? family.find((run) => run.id === selectedId)
    ?? visibleRuns[0];
  const branchCount = runtime.runs.filter((run) => run.parentRunId).length;
  const flaggedCount = runtime.runs.filter((run) => run.flags.length || run.warningCount).length;

  const familyById = new Map(family.map((run) => [run.id, run]));
  const byParent = new Map<string, RunSummary[]>();
  for (const run of family) {
    if (!run.parentRunId || !familyById.has(run.parentRunId)) continue;
    const children = byParent.get(run.parentRunId) ?? [];
    children.push(run);
    children.sort((a, b) => (a.createdTs ?? 0) - (b.createdTs ?? 0));
    byParent.set(run.parentRunId, children);
  }
  const roots = family
    .filter((run) => !run.parentRunId || !familyById.has(run.parentRunId))
    .sort((a, b) => (a.createdTs ?? 0) - (b.createdTs ?? 0));

  function stage(runId: string, slot: "a" | "b") {
    if (slot === "a") {
      setCompareA(runId);
      if (compareB === runId) setCompareB("");
    } else {
      setCompareB(runId);
      if (compareA === runId) setCompareA("");
    }
  }

  return (
    <>
      <section className="instrument runs-ledger" aria-labelledby="runs-title">
        <header className="instrument-head runs-head">
          <div>
            <span className="eyebrow">RUN LOG</span>
            <h1 id="runs-title">Recorded runs</h1>
          </div>
          <div className="runs-metrics">
            <span><b>RECORDED</b>{runtime.runs.length}</span>
            <span><b>VISIBLE</b>{visibleRuns.length}</span>
            <span><b>BRANCHES</b>{branchCount}</span>
            <span><b>FLAGGED</b>{flaggedCount}</span>
          </div>
          <div className="runs-filters">
            <label className="runs-search">
              <span>SEARCH</span>
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Prompt, response, model, ID"
              />
            </label>
            <label>
              <span>SOURCE</span>
              <select value={source} onChange={(event) => setSource(event.target.value)}>
                <option value="all">All sources</option>
                {sources.map((item) => <option key={item} value={item}>{item}</option>)}
              </select>
            </label>
            <label>
              <span>STATUS</span>
              <select value={status} onChange={(event) => setStatus(event.target.value as StatusFilter)}>
                <option value="all">All states</option>
                <option value="complete">Complete</option>
                <option value="truncated">Truncated</option>
                <option value="error">Error</option>
                <option value="flagged">Flagged</option>
              </select>
            </label>
          </div>
        </header>

        <div className="runs-table-wrap">
          <div className="runs-table-head" aria-hidden="true">
            <span>TIME / ID</span>
            <span>PROMPT / RESPONSE</span>
            <span>MODEL</span>
            <span>SOURCE</span>
            <span>DURATION</span>
            <span>STATE</span>
            <span>A / B</span>
          </div>
          <div className="runs-table" role="list" aria-label="Recorded runs">
            {visibleRuns.map((run) => {
              const state = runStatus(run);
              const isSelected = run.id === selected?.id;
              return (
                <div
                  className={`runs-row ${isSelected ? "is-selected" : ""}`}
                  key={run.id}
                  role="listitem"
                >
                  <button
                    className="runs-row-main"
                    type="button"
                    onClick={() => setSelectedId(run.id)}
                    aria-pressed={isSelected}
                  >
                    <span className="runs-identity">
                      <time>{timestamp(run.createdAt)}</time>
                      <small>{shortId(run.id)} · {parentLabel(run, runtime.runs)}</small>
                    </span>
                    <span className="runs-copy">
                      <strong>{run.prompt || "—"}</strong>
                      <small>{run.response || "—"}</small>
                    </span>
                    <span className="runs-model">{run.model}</span>
                    <span className="runs-source">{run.source}</span>
                    <span className="runs-duration">{run.duration}</span>
                    <span className={`runs-status is-${state.toLowerCase()}`}>{state}</span>
                  </button>
                  <span className="runs-stage">
                    <button
                      type="button"
                      className={compareA === run.id ? "is-a is-active" : "is-a"}
                      onClick={() => stage(run.id, "a")}
                      aria-pressed={compareA === run.id}
                      aria-label={`Stage ${shortId(run.id)} as run A`}
                    >A</button>
                    <button
                      type="button"
                      className={compareB === run.id ? "is-b is-active" : "is-b"}
                      onClick={() => stage(run.id, "b")}
                      aria-pressed={compareB === run.id}
                      aria-label={`Stage ${shortId(run.id)} as run B`}
                    >B</button>
                  </span>
                </div>
              );
            })}
            {visibleRuns.length === 0 && <div className="runs-empty">NO MATCHING RUNS</div>}
          </div>
        </div>
      </section>

      {inspectorOpen && (
        <aside className="instrument runs-inspector" aria-labelledby="runs-inspector-title">
          <header className="instrument-head compact">
            <div>
              <span className="eyebrow">SELECTED RUN</span>
              <h2 id="runs-inspector-title">{selected ? shortId(selected.id) : "—"}</h2>
            </div>
            {selected && <span className={`runs-status is-${runStatus(selected).toLowerCase()}`}>{runStatus(selected)}</span>}
          </header>

          {selected ? (
            <div className="runs-inspector-body">
              <div className="runs-actions">
                <a href={`#/runs/${encodeURIComponent(selected.id)}`}>READ</a>
                <a href={`#/runs/${encodeURIComponent(selected.id)}/scope`}>SCOPE</a>
                <button type="button" onClick={() => stage(selected.id, compareA ? "b" : "a")}>
                  {compareA ? "STAGE B" : "STAGE A"}
                </button>
              </div>

              <dl className="runs-facts">
                <div><dt>Created</dt><dd>{timestamp(selected.createdAt)}</dd></div>
                <div><dt>Model</dt><dd>{selected.model}</dd></div>
                <div><dt>Source</dt><dd>{selected.source}</dd></div>
                <div><dt>Client</dt><dd>{selected.client}</dd></div>
                <div><dt>Duration</dt><dd>{selected.duration}</dd></div>
                <div><dt>Finish</dt><dd>{selected.finishReason?.toUpperCase() ?? "—"}</dd></div>
                <div><dt>Trace</dt><dd>{facts.traceAvailable ? `${facts.tokenCount} TOKENS` : "UNAVAILABLE"}</dd></div>
                <div><dt>Parent</dt><dd>{selected.parentRunId ? shortId(selected.parentRunId) : "—"}</dd></div>
              </dl>

              <section className="runs-summary">
                <span>PROMPT</span>
                <p>{selected.prompt || "—"}</p>
              </section>
              <section className="runs-summary">
                <span>RESPONSE</span>
                <p>{selected.response || "—"}</p>
              </section>

              <div className="runs-flags">
                {selected.flags.map((flag) => <span key={flag}>{flag}</span>)}
                {selected.activeDialCount > 0 && <span>{selected.activeDialCount} DIALS</span>}
                {selected.memoryCardCount > 0 && <span>{selected.memoryCardCount} MEMORY</span>}
                {selected.warningCount > 0 && <span>{selected.warningCount} WARNINGS</span>}
                {selected.flags.length === 0
                  && selected.activeDialCount === 0
                  && selected.memoryCardCount === 0
                  && selected.warningCount === 0
                  && <span>NO FLAGS</span>}
              </div>
            </div>
          ) : <div className="runs-empty">SELECT A RUN</div>}
        </aside>
      )}

      <section className="instrument runs-lineage" aria-labelledby="runs-lineage-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">PARENT / CHILD</span>
            <h2 id="runs-lineage-title">Run family</h2>
          </div>
          <strong>{family.length} {family.length === 1 ? "RUN" : "RUNS"}</strong>
        </header>
        <div className="runs-tree">
          {roots.map((root) => (
            <LineageNode
              key={root.id}
              run={root}
              byParent={byParent}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          ))}
          {selected && !family.length && <span className="runs-tree-loading">LOADING FAMILY</span>}
        </div>
      </section>

      <section className="instrument runs-compare-tray" aria-labelledby="runs-compare-title">
        <header className="instrument-head compact">
          <div>
            <span className="eyebrow">A / B</span>
            <h2 id="runs-compare-title">Comparison tray</h2>
          </div>
        </header>
        <div className="runs-slots">
          <button type="button" className="is-a" onClick={() => stage(selected?.id ?? "", "a")}>
            <b>A</b>
            <span>{compareA ? shortId(compareA) : "SELECT RUN"}</span>
          </button>
          <i>→</i>
          <button type="button" className="is-b" onClick={() => stage(selected?.id ?? "", "b")}>
            <b>B</b>
            <span>{compareB ? shortId(compareB) : "SELECT RUN"}</span>
          </button>
        </div>
        <div className="runs-tray-actions">
          <button type="button" onClick={() => { setCompareA(""); setCompareB(""); }}>CLEAR</button>
          <a
            className={!compareA || !compareB ? "is-disabled" : ""}
            aria-disabled={!compareA || !compareB}
            href={compareA && compareB
              ? `#/compare/${encodeURIComponent(compareA)}/${encodeURIComponent(compareB)}`
              : undefined}
          >COMPARE</a>
        </div>
      </section>
    </>
  );
}
