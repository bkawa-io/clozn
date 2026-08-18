import { useEffect, useMemo, useState } from "react";
import { EvidenceMark } from "../../components/EvidenceMark";
import "../../components/EvidenceMark.css";
import { loadRunFacts, loadRunFamily } from "../../data/api";
import type { RunFacts, RunSummary, RuntimeState } from "../../data/types";

export interface RunsProps {
  runtime: RuntimeState;
  inspectorOpen: boolean;
}

export type RunLedgerStatus = "complete" | "truncated" | "error" | "recorded";
export type RunLedgerFlagFilter = "all" | "flagged" | "unflagged";

export interface RunLedgerFilters {
  query: string;
  model: string;
  source: string;
  finishReason: string;
  flagged: RunLedgerFlagFilter;
}

interface RecordedFilterOptions {
  models: string[];
  sources: string[];
  finishReasons: string[];
  hasMissingModels: boolean;
  hasMissingSources: boolean;
  hasMissingFinishReasons: boolean;
}

interface EvidenceIndexItem {
  id: "context" | "influence" | "performance" | "claims";
  abbreviation: string;
  label: string;
  reason: string;
}

const ALL_FILTER = "all";
const NOT_RECORDED_FILTER = "__not_recorded__";
const ADAPTER_FILTER_REASON = "Adapter identity is not included in the loaded run index, so this index cannot filter it.";
const INFLUENCE_FILTER_REASON = "Influence artifact presence is not included in the loaded run index, so this index cannot filter it.";
const CONFIDENCE_ABSENCE_REASON = "Token confidence is not included in the loaded run index; the ledger will not infer it from duration or finish state.";

// The ledger deliberately consumes only RuntimeState.runs. These four artifacts each have their own
// run-scoped instrument, so the index must show an explained absence until that instrument reads it.
const EVIDENCE_INDEX_ITEMS: readonly EvidenceIndexItem[] = [
  {
    id: "context",
    abbreviation: "CTX",
    label: "Context receipt",
    reason: "The loaded run index does not include a context-receipt artifact for this row.",
  },
  {
    id: "influence",
    abbreviation: "INF",
    label: "Influence",
    reason: "The loaded run index does not include a source-influence artifact for this row.",
  },
  {
    id: "performance",
    abbreviation: "PERF",
    label: "Performance",
    reason: "A listed duration is not a loaded performance trace; this index has not read that artifact.",
  },
  {
    id: "claims",
    abbreviation: "CLM",
    label: "Claims",
    reason: "The loaded run index does not include a claim-support artifact for this row.",
  },
];

function recordedText(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed && trimmed !== "—" ? trimmed : undefined;
}

function uniqueRecorded(values: readonly (string | undefined)[]): string[] {
  return [...new Set(values.flatMap((value) => {
    const recorded = recordedText(value);
    return recorded ? [recorded] : [];
  }))].sort((left, right) => left.localeCompare(right));
}

function hasFlag(run: RunSummary) {
  return run.flags.length > 0 || run.warningCount > 0;
}

function shortId(id: string) {
  return id.length > 8 ? id.slice(-8) : id;
}

function timestamp(value: string) {
  if (!value || value === "—") return "TIME NOT RECORDED";
  const match = value.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/);
  return match ? `${match[2]}/${match[3]} ${match[4]}:${match[5]}:${match[6]}` : value;
}

function humanize(value: string | undefined) {
  const recorded = recordedText(value);
  return recorded
    ? recorded.replace(/[_-]+/g, " ").replace(/\s+/g, " ").toUpperCase()
    : "NOT RECORDED";
}

export function runLedgerStatus(run: RunSummary): RunLedgerStatus {
  if (run.flags.includes("error")) return "error";
  if (run.finishReason === "length" || run.flags.includes("truncated")) return "truncated";
  if (recordedText(run.finishReason)) return "complete";
  return "recorded";
}

function statusLabel(status: RunLedgerStatus) {
  switch (status) {
    case "complete": return "COMPLETE";
    case "truncated": return "TRUNCATED";
    case "error": return "ERROR";
    case "recorded": return "RECORDED";
    default: {
      const exhaustive: never = status;
      return exhaustive;
    }
  }
}

function runLabel(run: RunSummary) {
  return recordedText(run.label) || recordedText(run.prompt) || `Run ${shortId(run.id)}`;
}

function parentLabel(run: RunSummary, runs: readonly RunSummary[]) {
  if (!run.parentRunId) return "ORIGINAL";
  const parent = runs.find((candidate) => candidate.id === run.parentRunId);
  return parent ? `FROM ${shortId(parent.id)}` : `FROM ${shortId(run.parentRunId)}`;
}

/** What a derived run says about where it came from.
 *
 * Prefers the parent's own prompt over its id: "from 0_e0116e" asks you to hold an eight-character
 * hash in your head and go find it, while "from A customer was auto-renewed..." tells you which
 * investigation this run belongs to. Falls back to the id when the parent is not in the loaded page
 * -- naming a run we cannot see would be worse than admitting we only have its id.
 */
function derivedFromLabel(run: RunSummary, runs: readonly RunSummary[]) {
  const parent = runs.find((candidate) => candidate.id === run.parentRunId);
  const text = parent && recordedText(parent.prompt);
  if (text) return text.length > 64 ? `${text.slice(0, 64)}…` : text;
  // The parent is not in the loaded page. This used to fall back to shortId(), which slices the last
  // eight characters off a run id and produced things like "from edundant" -- a hash fragment that
  // reads as a corrupted word, not an identifier. Saying we know it came from somewhere, without
  // pretending to name it, is the honest version; the full id is on the element's title.
  return "an earlier run";
}

export function runLedgerFilterOptions(runs: readonly RunSummary[]): RecordedFilterOptions {
  return {
    models: uniqueRecorded(runs.map((run) => run.model)),
    sources: uniqueRecorded(runs.map((run) => run.source)),
    finishReasons: uniqueRecorded(runs.map((run) => run.finishReason)),
    hasMissingModels: runs.some((run) => !recordedText(run.model)),
    hasMissingSources: runs.some((run) => !recordedText(run.source)),
    hasMissingFinishReasons: runs.some((run) => !recordedText(run.finishReason)),
  };
}

function matchesRecordedFilter(value: string | undefined, selected: string) {
  if (selected === ALL_FILTER) return true;
  const recorded = recordedText(value);
  if (selected === NOT_RECORDED_FILTER) return !recorded;
  return recorded === selected;
}

export function filterRunLedger(
  runs: readonly RunSummary[],
  filters: RunLedgerFilters,
): RunSummary[] {
  const needle = filters.query.trim().toLowerCase();
  return runs.filter((run) => {
    if (!matchesRecordedFilter(run.model, filters.model)) return false;
    if (!matchesRecordedFilter(run.source, filters.source)) return false;
    if (!matchesRecordedFilter(run.finishReason, filters.finishReason)) return false;
    if (filters.flagged === "flagged" && !hasFlag(run)) return false;
    if (filters.flagged === "unflagged" && hasFlag(run)) return false;
    if (!needle) return true;
    return [
      run.id,
      run.label,
      run.prompt,
      run.response,
      run.model,
      run.source,
      run.finishReason ?? "",
      ...run.flags,
    ].some((value) => value.toLowerCase().includes(needle));
  });
}

export function runLedgerRoute(runId: string) {
  return `#/runs/${encodeURIComponent(runId)}`;
}

export function runLedgerCompareRoute(runA: string, runB: string): string | undefined {
  // One record cannot be both ends of a comparison. Returning no link is clearer than routing to a
  // self-comparison that visually suggests a real A/B result exists.
  if (!runA || !runB || runA === runB) return undefined;
  return `#/compare/${encodeURIComponent(runA)}/${encodeURIComponent(runB)}`;
}

/**
 * A token-confidence trace is a probability series, not a duration proxy. This presenter deliberately
 * accepts only recorded values: the no-values branch is an EvidenceMark instead of a flat zero line.
 */
export function ConfidenceSparkline({
  values,
  absenceReason = CONFIDENCE_ABSENCE_REASON,
}: {
  values?: readonly number[];
  absenceReason?: string;
}) {
  const recorded = (values ?? []).filter((value): value is number => Number.isFinite(value));
  if (!recorded.length) {
    return (
      <span className="runs-confidence is-absent">
        <EvidenceMark state="not_measured" label="Confidence not recorded" reason={absenceReason} />
        <span>NOT RECORDED</span>
      </span>
    );
  }

  const visible = recorded.length <= 24
    ? recorded
    : Array.from({ length: 24 }, (_, index) => recorded[Math.round((index / 23) * (recorded.length - 1))]);
  const points = visible.map((value, index) => {
    const x = visible.length === 1 ? 50 : (index / (visible.length - 1)) * 100;
    // Confidence is a probability. A malformed out-of-range record is bounded only for geometry; its
    // exact values stay available in the tooltip rather than being silently rewritten as evidence.
    const y = 100 - (Math.max(0, Math.min(1, value)) * 100);
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");

  return (
    <span
      className="runs-confidence is-recorded"
      data-confidence-sparkline="recorded"
      role="img"
      aria-label={`Confidence sparkline from ${recorded.length} recorded token confidences`}
      title={recorded.map((value) => value.toFixed(4)).join(", ")}
    >
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <polyline points={points} />
      </svg>
    </span>
  );
}

function RunIdentityChips({ run }: { run: RunSummary }) {
  const model = recordedText(run.model);
  const source = recordedText(run.source);
  const modelReason = "This run index row did not record a model identity.";
  const sourceReason = "This run index row did not record an entry point.";

  return (
    <span className="runs-identity-chips" role="group" aria-label={`Provenance for ${runLabel(run)}`}>
      <span className={`runs-identity-chip${model ? "" : " is-absent"}`} title={modelReason}>
        <b>MODEL</b>
        <span>{model ?? "NOT RECORDED"}</span>
        {!model && <EvidenceMark state="not_measured" label="Model not recorded" reason={modelReason} />}
      </span>
      <span className="runs-identity-chip is-absent" title={ADAPTER_FILTER_REASON}>
        <b>ADAPTER</b>
        <span>NOT RECORDED</span>
        <EvidenceMark state="not_measured" label="Adapter not recorded" reason={ADAPTER_FILTER_REASON} />
      </span>
      <span className={`runs-identity-chip${source ? "" : " is-absent"}`} title={sourceReason}>
        <b>ENTRY</b>
        <span>{source ?? "NOT RECORDED"}</span>
        {!source && <EvidenceMark state="not_measured" label="Entry point not recorded" reason={sourceReason} />}
      </span>
    </span>
  );
}

/** The card's at-a-glance row.
 *
 * Every signal here is derived from a field `RunSummary` ACTUALLY carries. That constraint is the
 * point. The table this replaced spent roughly half its width on three columns that were structurally
 * incapable of varying: `<ConfidenceSparkline />` was called with no `values` prop so it always
 * rendered NOT RECORDED; `RunEvidenceIndex` hardcoded `state="not_measured"` for all four evidence
 * marks; and the ADAPTER chip was a literal "NOT RECORDED". Four absence marks on every row, in every
 * state, forever -- which trains the eye to ignore exactly the marks that are supposed to mean
 * something.
 *
 * So this renders NOTHING when a run has nothing notable, and the quiet card IS the signal. It does
 * not claim a run is clean -- absence of a warning is not evidence of correctness, and the index
 * simply does not carry per-run evidence availability. That belongs on the run's own page, where it
 * can be fetched and stated honestly, not guessed at from a list payload.
 */
function RunSignals({ run, state }: { run: RunSummary; state: RunLedgerStatus }) {
  const signals: { id: string; tone: string; label: string; title: string }[] = [];

  if (state === "error") {
    signals.push({ id: "error", tone: "error", label: "ERROR", title: "This run recorded an error flag." });
  }
  if (state === "truncated") {
    signals.push({
      id: "truncated",
      tone: "warn",
      label: "CUT OFF",
      title: "Generation stopped at the token limit, so the answer may be unfinished.",
    });
  }
  if (run.lowConfidenceCount && run.tokenCount) {
    // A real measurement, unlike the CONFIDENCE column this replaced -- that one rendered NOT
    // RECORDED on every row because <ConfidenceSparkline /> was mounted with no values prop.
    //
    // A SHARE, not a count. "36 SHAKY" raises the question "36 of what?", and the honest answer
    // (36 of 173 generated tokens) is not something an index should make you do arithmetic on: 36
    // shaky tokens means something completely different in a 40-token answer than in a 400-token
    // one. The exact counts stay in the tooltip for anyone who wants them.
    const share = run.lowConfidenceCount / run.tokenCount;
    signals.push({
      id: "shaky",
      tone: share > 0.25 ? "warn" : "caution",
      label: `${Math.round(share * 100)}% SHAKY`,
      title:
        `${run.lowConfidenceCount} of ${run.tokenCount} generated tokens fell below the confidence `
        + `floor. Mean ${run.confidenceMean}, lowest ${run.confidenceMin}.`,
    });
  }
  if (run.warningCount > 0) {
    signals.push({
      id: "warnings",
      tone: "warn",
      label: run.warningCount === 1 ? "1 WARNING" : `${run.warningCount} WARNINGS`,
      title: "Recorded diagnostic warnings for this run.",
    });
  }
  if (!signals.length) return null;
  return (
    <span className="run-card-signals" aria-label={`Signals for ${runLabel(run)}`}>
      {signals.map((signal) => (
        <span className={`run-signal is-${signal.tone}`} key={signal.id} title={signal.title}>
          <i aria-hidden="true" />{signal.label}
        </span>
      ))}
    </span>
  );
}

function RunEvidenceIndex({ run }: { run: RunSummary }) {
  return (
    <span className="runs-evidence-index" role="group" aria-label={`Evidence index for ${runLabel(run)}`}>
      {EVIDENCE_INDEX_ITEMS.map((item) => (
        <span className="runs-evidence-item" data-evidence={item.id} key={item.id} title={`${item.label}: ${item.reason}`}>
          <small aria-hidden="true">{item.abbreviation}</small>
          <EvidenceMark state="not_measured" label={item.label} reason={item.reason} />
        </span>
      ))}
    </span>
  );
}

function RecordedFilter({
  label,
  value,
  values,
  hasMissing,
  onChange,
}: {
  label: string;
  value: string;
  values: readonly string[];
  hasMissing: boolean;
  onChange: (value: string) => void;
}) {
  const disabled = values.length === 0;
  return (
    <label className="runs-filter">
      <span>{label}</span>
      <select
        aria-label={`${label} filter`}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
      >
        {disabled ? (
          <option value={ALL_FILTER}>{label} not recorded</option>
        ) : (
          <>
            <option value={ALL_FILTER}>All {label.toLowerCase()}</option>
            {hasMissing && <option value={NOT_RECORDED_FILTER}>Not recorded</option>}
            {values.map((item) => <option key={item} value={item}>{item}</option>)}
          </>
        )}
      </select>
    </label>
  );
}

function UnavailableFilter({ label, reason }: { label: string; reason: string }) {
  return (
    <label className="runs-filter is-unavailable">
      <span>{label}</span>
      <select aria-label={`${label} filter`} disabled value={ALL_FILTER} title={reason}>
        <option value={ALL_FILTER}>Not indexed</option>
      </select>
    </label>
  );
}


function SelectionDock({
  selected,
  compareA,
  compareB,
  onStage,
  onClear,
}: {
  selected: RunSummary | undefined;
  compareA: string;
  compareB: string;
  onStage: (runId: string, slot: "a" | "b") => void;
  onClear: () => void;
}) {
  const comparisonRoute = runLedgerCompareRoute(compareA, compareB);
  const selectedLabel = selected ? runLabel(selected) : "Select a recorded run";
  const aIsSelected = Boolean(selected && compareA === selected.id);
  const bIsSelected = Boolean(selected && compareB === selected.id);

  return (
    <aside className="runs-selection-dock" aria-label="Selected run controls">
      <div className="runs-selection-summary">
        <span>SELECTION DOCK</span>
        <strong title={selected?.id}>{selectedLabel}</strong>
        {selected && <small>RUN {shortId(selected.id)}</small>}
      </div>
      <div className="runs-selection-actions">
        {selected ? (
          <a href={runLedgerRoute(selected.id)} aria-label={`Open ${selectedLabel} in Run`}>OPEN RUN</a>
        ) : (
          <span className="is-disabled">OPEN RUN</span>
        )}
        <button
          type="button"
          disabled={!selected}
          aria-pressed={aIsSelected}
          aria-label="Stage selected run as A"
          onClick={() => selected && onStage(selected.id, "a")}
        >A {aIsSelected ? "STAGED" : "STAGE"}</button>
        <button
          type="button"
          disabled={!selected}
          aria-pressed={bIsSelected}
          aria-label="Stage selected run as B"
          onClick={() => selected && onStage(selected.id, "b")}
        >B {bIsSelected ? "STAGED" : "STAGE"}</button>
        {comparisonRoute ? (
          <a href={comparisonRoute} aria-label="Compare staged runs">COMPARE</a>
        ) : (
          <span className="is-disabled" title="Stage two different recorded runs to compare them.">STAGE A + B</span>
        )}
        {(compareA || compareB) && <button type="button" onClick={onClear}>CLEAR</button>}
      </div>
    </aside>
  );
}

export function Runs({ runtime, inspectorOpen }: RunsProps) {
  const [selectedId, setSelectedId] = useState("");
  const [family, setFamily] = useState<RunSummary[]>([]);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [facts, setFacts] = useState<RunFacts>({ tokenCount: 0, traceAvailable: false });
  const [filters, setFilters] = useState<RunLedgerFilters>({
    query: "",
    model: ALL_FILTER,
    source: ALL_FILTER,
    finishReason: ALL_FILTER,
    flagged: "all",
  });

  useEffect(() => {
    if (!runtime.runs.length) {
      setSelectedId("");
      return;
    }
    setSelectedId((current) => runtime.runs.some((run) => run.id === current) ? current : runtime.runs[0].id);
  }, [runtime.runs]);

  useEffect(() => {
    if (!selectedId) return;
    const controller = new AbortController();
    setFamily([]);
    setFacts({ tokenCount: 0, traceAvailable: false });
    void Promise.all([
      loadRunFamily(selectedId, controller.signal),
      loadRunFacts(selectedId, controller.signal),
    ]).then(([nextFamily, nextFacts]) => {
      if (controller.signal.aborted) return;
      setFamily(nextFamily);
      setFacts(nextFacts);
    }).catch(() => {
      if (!controller.signal.aborted) {
        setFamily([]);
        setFacts({ tokenCount: 0, traceAvailable: false });
      }
    });
    return () => controller.abort();
  }, [selectedId]);

  const filterOptions = useMemo(() => runLedgerFilterOptions(runtime.runs), [runtime.runs]);
  const visibleRuns = useMemo(() => filterRunLedger(runtime.runs, filters), [filters, runtime.runs]);
  const selected = runtime.runs.find((run) => run.id === selectedId)
    ?? family.find((run) => run.id === selectedId)
    ?? visibleRuns[0];
  // Shown on the toggle so a collapsed panel can never hide a filter that is silently excluding runs.
  const activeFilterCount = [
    filters.model !== ALL_FILTER,
    filters.source !== ALL_FILTER,
    filters.finishReason !== ALL_FILTER,
    filters.flagged !== "all",
  ].filter(Boolean).length;
  const branchCount = runtime.runs.filter((run) => run.parentRunId).length;
  const flaggedCount = runtime.runs.filter(hasFlag).length;

  function setFilter<Key extends keyof RunLedgerFilters>(key: Key, value: RunLedgerFilters[Key]) {
    setFilters((current) => ({ ...current, [key]: value }));
  }

  return (
    <>
      <section className="instrument runs-ledger" aria-labelledby="runs-title">
        <header className="instrument-head runs-head">
          <div>
            <span className="eyebrow">S1 · RUN INDEX</span>
            <h1 id="runs-title">Recorded runs</h1>
          </div>
          <div className="runs-metrics">
            <span><b>RECORDED</b>{runtime.runs.length}</span>
            <span><b>VISIBLE</b>{visibleRuns.length}</span>
            <span><b>BRANCHES</b>{branchCount}</span>
            <span><b>FLAGGED</b>{flaggedCount}</span>
          </div>
          <label className="runs-search-inline">
            <span className="sr-only">Search runs</span>
            <input
              type="search"
              value={filters.query}
              onChange={(event) => setFilter("query", event.target.value)}
              placeholder="Search prompt, response, model, ID"
            />
          </label>
          <button
            type="button"
            className={`runs-filter-toggle${activeFilterCount ? " is-active" : ""}`}
            aria-expanded={filtersOpen}
            onClick={() => setFiltersOpen((open) => !open)}
          >
            FILTER{activeFilterCount ? ` · ${activeFilterCount}` : ""}
          </button>
          {/* Collapsed by default. Seven controls across the top of the index made the first thing
              you saw a configuration panel rather than your runs -- and six of them are refinements
              you reach for occasionally, while search is the one you use constantly. */}
          {filtersOpen && (
          <div className="runs-filters" aria-label="Run provenance filters">
            <RecordedFilter
              label="MODEL"
              value={filters.model}
              values={filterOptions.models}
              hasMissing={filterOptions.hasMissingModels}
              onChange={(value) => setFilter("model", value)}
            />
            <UnavailableFilter label="ADAPTER" reason={ADAPTER_FILTER_REASON} />
            <RecordedFilter
              label="ENTRY POINT"
              value={filters.source}
              values={filterOptions.sources}
              hasMissing={filterOptions.hasMissingSources}
              onChange={(value) => setFilter("source", value)}
            />
            <RecordedFilter
              label="FINISH REASON"
              value={filters.finishReason}
              values={filterOptions.finishReasons}
              hasMissing={filterOptions.hasMissingFinishReasons}
              onChange={(value) => setFilter("finishReason", value)}
            />
            <UnavailableFilter label="HAS INFLUENCE" reason={INFLUENCE_FILTER_REASON} />
            <label className="runs-filter">
              <span>FLAGGED</span>
              <select
                aria-label="Flagged filter"
                value={filters.flagged}
                onChange={(event) => setFilter("flagged", event.target.value as RunLedgerFlagFilter)}
              >
                <option value="all">All runs</option>
                <option value="flagged">Flagged</option>
                <option value="unflagged">Not flagged</option>
              </select>
            </label>
            <div className="runs-filter-limitations" role="note">
              <span><EvidenceMark state="not_measured" label="Adapter filter unavailable" reason={ADAPTER_FILTER_REASON} /> Adapter identity is not indexed; no run is assumed to be base.</span>
              <span><EvidenceMark state="not_measured" label="Influence filter unavailable" reason={INFLUENCE_FILTER_REASON} /> Influence presence is not indexed; the ledger does not claim a missing map is no influence.</span>
            </div>
          </div>
          )}
        </header>

        <div className="runs-table-wrap">
          <div className="runs-table" role="list" aria-label="Recorded runs">
            {visibleRuns.map((run) => {
              const state = runLedgerStatus(run);
              const isSelected = run.id === selected?.id;
              const duration = recordedText(run.duration);
              return (
                <div
                  className={`runs-row ${isSelected ? "is-selected" : ""}`}
                  data-testid={`run-ledger-row-${run.id}`}
                  key={run.id}
                  role="listitem"
                >
                  <button
                    className="runs-row-main"
                    type="button"
                    onClick={() => setSelectedId(run.id)}
                    aria-pressed={isSelected}
                    aria-label={`Select run ${runLabel(run)}`}
                  >
                    <p className="run-card-prompt">{run.prompt || run.label || shortId(run.id)}</p>
                    {run.response && <p className="run-card-response">{run.response}</p>}
                    {run.parentRunId && (
                      <p className="run-card-derived" title={
                        `Derived from run ${run.parentRunId} -- a replay, fork, or corrective retry `
                        + `rather than a fresh generation.`
                      }>
                        ↳ from {derivedFromLabel(run, runtime.runs)}
                      </p>
                    )}
                    <p className="run-card-meta">
                      <RunSignals run={run} state={state} />
                      <time>{timestamp(run.createdAt)}</time>
                      {duration && <> · {duration}</>}
                      {recordedText(run.model) && <> · {recordedText(run.model)}</>}
                      {recordedText(run.source) && <> · {recordedText(run.source)}</>}
                      {" · "}{shortId(run.id)}
                    </p>
                  </button>
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
            {selected && <span className={`runs-status is-${runLedgerStatus(selected)}`}>{statusLabel(runLedgerStatus(selected))}</span>}
          </header>

          {selected ? (
            <div className="runs-inspector-body">
              <div className="runs-actions">
                <a href={runLedgerRoute(selected.id)}>OPEN RUN</a>
                <a href={`#/runs/${encodeURIComponent(selected.id)}/lens`}>OPEN LENS</a>
                <a href={`#/runs/${encodeURIComponent(selected.id)}/diagnostics`}>DIAGNOSTICS</a>
                {selected.sessionKey && (
                  <a href={`#/sessions/${encodeURIComponent(selected.sessionKey)}/investigate`}>SESSION DIAGNOSTICS</a>
                )}
              </div>

              <dl className="runs-facts">
                <div><dt>Created</dt><dd>{timestamp(selected.createdAt)}</dd></div>
                <div><dt>Model</dt><dd>{recordedText(selected.model) ?? "NOT RECORDED"}</dd></div>
                <div><dt>Adapter</dt><dd>NOT RECORDED IN INDEX</dd></div>
                <div><dt>Entry point</dt><dd>{recordedText(selected.source) ?? "NOT RECORDED"}</dd></div>
                <div><dt>Client</dt><dd>{selected.client}</dd></div>
                <div><dt>Duration</dt><dd>{recordedText(selected.duration) ?? "NOT RECORDED"}</dd></div>
                <div><dt>Finish</dt><dd>{humanize(selected.finishReason)}</dd></div>
                <div><dt>Trace</dt><dd>{facts.traceAvailable ? `${facts.tokenCount} TOKENS` : "NOT RECORDED"}</dd></div>
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
                {selected.warningCount > 0 && <span>{selected.warningCount} WARNINGS</span>}
                {selected.flags.length === 0
                  && selected.warningCount === 0
                  && <span>NO FLAGS</span>}
              </div>
            </div>
          ) : <div className="runs-empty">SELECT A RUN</div>}
        </aside>
      )}

    </>
  );
}
