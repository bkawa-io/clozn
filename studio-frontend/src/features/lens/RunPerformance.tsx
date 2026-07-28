import { useState } from "react";
import type {
  DiagnosisEvidence,
  PerformanceRuleDiagnosis,
  PerformanceRuleEvidence,
  RunDiagnosisFinding,
  RunPerformance as RunPerformanceData,
} from "../../data/types";

interface RunPerformanceProps {
  data?: RunPerformanceData;
  status: "idle" | "loading" | "ready" | "error";
  selectedFindingId: string;
  onSelectFinding: (id: string) => void;
}

const findingLabels: Record<string, string> = {
  total_wall_time: "TOTAL",
  model_load: "LOAD",
  prefill: "PREFILL",
  generation: "DECODE",
  context_pressure: "CONTEXT",
  context_allocation: "KV ALLOC",
  cpu_spill: "CPU SPILL",
};

const ruleLabels: Record<string, string> = {
  large_context: "LARGE CONTEXT",
  slow_decode: "SLOW DECODE",
  cold_model_load: "COLD MODEL LOAD",
  client_backpressure: "CLIENT BACKPRESSURE",
  queue_contention: "QUEUE CONTENTION",
  adapter_reload: "ADAPTER RELOAD",
  memory_pressure: "MEMORY PRESSURE",
};

// "fired"/"not_fired"/"unavailable" are a DIFFERENT vocabulary from the phase findings above --
// see types.ts. The nav label text differs for every status on purpose (not just color), so the
// distinction survives a screenshot, a colorblind viewer, or someone skimming past the dot.
const ruleStatusLabels: Record<PerformanceRuleDiagnosis["status"], string> = {
  fired: "FIRED",
  not_fired: "CHECKED · CLEAN",
  unavailable: "NOT INSTRUMENTED",
};

function duration(value?: number) {
  if (value == null) return "—";
  if (value < 1000) return `${Math.round(value)} ms`;
  return `${(value / 1000).toFixed(value < 10_000 ? 2 : 1)} s`;
}

function integer(value?: number) {
  return value == null ? "—" : Math.round(value).toLocaleString();
}

function rate(value?: number) {
  return value == null ? "—" : `${value.toFixed(value < 10 ? 2 : 1)} tok/s`;
}

function rawValue(value: unknown) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3);
  }
  if (typeof value === "string" || typeof value === "boolean") {
    return String(value);
  }
  if (value == null) return "—";
  try {
    return JSON.stringify(value);
  } catch {
    return "—";
  }
}

function evidenceValue(evidence: DiagnosisEvidence) {
  return rawValue(evidence.value);
}

function ruleEvidenceValue(evidence: PerformanceRuleEvidence) {
  return rawValue(evidence.value);
}

function Metric({
  label,
  value,
  source,
  title,
}: {
  label: string;
  value: string;
  source: string;
  title?: string;
}) {
  return (
    <article className="lens-performance-metric" title={title}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{source}</small>
    </article>
  );
}

function selectedFinding(
  findings: RunDiagnosisFinding[],
  selectedFindingId: string,
) {
  return findings.find((finding) => finding.id === selectedFindingId)
    ?? findings.find((finding) => finding.status === "observed")
    ?? findings[0];
}

function selectedRule(
  diagnoses: PerformanceRuleDiagnosis[],
  selectedRuleId: string | null,
) {
  return diagnoses.find((entry) => entry.rule === selectedRuleId)
    ?? diagnoses.find((entry) => entry.status === "fired")
    ?? diagnoses[0];
}

export function RunPerformance({
  data,
  status,
  selectedFindingId,
  onSelectFinding,
}: RunPerformanceProps) {
  const findings = data?.diagnosis?.findings ?? [];
  const selected = selectedFinding(findings, selectedFindingId);
  const throughputLabel = data?.throughput?.kind === "measured_decode"
    ? "MEASURED DECODE"
    : data?.throughput?.kind === "derived_end_to_end"
      ? "DERIVED END TO END"
      : "UNAVAILABLE";

  // Local, not lifted to Lens.tsx state -- this selection is this component's own concern, and a fresh
  // run resets it naturally on remount via Lens's `key`-less data swap (the rule list itself changes
  // identity every run, so a stale rule id just falls through to the "first fired, else first" default).
  const [selectedRuleId, setSelectedRuleId] = useState<string | null>(null);
  const diagnoses = data?.rules?.diagnoses ?? [];
  const selectedDiagnosis = selectedRule(diagnoses, selectedRuleId);

  return (
    <section className="lens-performance" aria-labelledby="lens-performance-title">
      <header className="lens-performance-head">
        <div>
          <span>RUN DIAGNOSIS</span>
          <strong id="lens-performance-title">Performance</strong>
        </div>
        <div>
          <span>FINISH</span>
          <strong>{data?.finishReason?.toUpperCase() ?? "—"}</strong>
        </div>
      </header>

      <div className="lens-performance-metrics">
        <Metric
          label="TOTAL"
          value={duration(data?.totalDuration?.value)}
          source="MEASURED WALL TIME"
          title={data?.totalDuration?.source}
        />
        <Metric
          label="THROUGHPUT"
          value={rate(data?.throughput?.value)}
          source={throughputLabel}
          title={data?.throughput?.source}
        />
        <Metric
          label="PROMPT"
          value={integer(data?.promptTokens?.value)}
          source="TOKENS"
          title={data?.promptTokens?.source}
        />
        <Metric
          label="OUTPUT"
          value={integer(data?.generatedTokens?.value)}
          source="TOKENS"
          title={data?.generatedTokens?.source}
        />
        <Metric
          label="GENERATION"
          value={duration(data?.generationDuration?.value)}
          source="MEASURED PHASE"
          title={data?.generationDuration?.source}
        />
      </div>

      <div className="lens-performance-diagnosis">
        <nav aria-label="Performance phases">
          {findings.map((finding) => (
            <button
              type="button"
              className={[
                selected?.id === finding.id ? "is-selected" : "",
                `status-${finding.status.replace("_", "-")}`,
              ].join(" ")}
              aria-pressed={selected?.id === finding.id}
              onClick={() => onSelectFinding(finding.id)}
              key={finding.id}
            >
              <i />
              <span>{findingLabels[finding.id] ?? finding.id.replaceAll("_", " ").toUpperCase()}</span>
              <small>{finding.status.replace("_", " ").toUpperCase()}</small>
            </button>
          ))}
        </nav>

        <div className="lens-performance-detail">
          {status === "loading" ? (
            <div className="lens-performance-state">LOADING PERFORMANCE</div>
          ) : status === "error" ? (
            <div className="lens-performance-state is-error">RUN PERFORMANCE UNAVAILABLE</div>
          ) : selected ? (
            <>
              <header>
                <strong>{findingLabels[selected.id] ?? selected.id.replaceAll("_", " ").toUpperCase()}</strong>
                <span className={`status-${selected.status.replace("_", "-")}`}>
                  {selected.status.replace("_", " ").toUpperCase()}
                </span>
              </header>
              <p>{selected.text}</p>
              {selected.evidence.length > 0 && (
                <dl>
                  {selected.evidence.map((evidence) => (
                    <div key={evidence.path}>
                      <dt>{evidence.path}</dt>
                      <dd>{evidenceValue(evidence)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </>
          ) : (
            <div className="lens-performance-state">
              {data ? "PHASE DIAGNOSIS UNAVAILABLE" : "LOADING PERFORMANCE"}
            </div>
          )}
        </div>
      </div>

      <div className="lens-performance-rules-label">
        <span>RULE ENGINE</span>
        <strong>Likely cause</strong>
      </div>

      <div className="lens-performance-rules">
        <nav aria-label="Performance diagnosis rules">
          {diagnoses.map((entry) => (
            <button
              type="button"
              title={`${ruleLabels[entry.rule] ?? entry.rule} — ${ruleStatusLabels[entry.status]}`}
              className={[
                selectedDiagnosis?.rule === entry.rule ? "is-selected" : "",
                `status-${entry.status.replace(/_/g, "-")}`,
              ].join(" ")}
              aria-pressed={selectedDiagnosis?.rule === entry.rule}
              onClick={() => setSelectedRuleId(entry.rule)}
              key={entry.rule}
            >
              <i />
              <span>{ruleLabels[entry.rule] ?? entry.rule.replaceAll("_", " ").toUpperCase()}</span>
              <small>{ruleStatusLabels[entry.status]}</small>
            </button>
          ))}
        </nav>

        <div className="lens-performance-rules-detail">
          {status === "loading" ? (
            <div className="lens-performance-state">LOADING RULE ENGINE</div>
          ) : status === "error" ? (
            <div className="lens-performance-state is-error">RULE ENGINE UNAVAILABLE</div>
          ) : !data?.rules ? (
            // Distinct from "checked every rule, nothing fired": the fetch itself never returned a
            // trace for this run at all, which is a different fact and must read differently -- see
            // the module-level discipline this whole feature exists to enforce.
            <div className="lens-performance-state">
              {data ? "NO PERFORMANCE TRACE FOR THIS RUN" : "LOADING RULE ENGINE"}
            </div>
          ) : selectedDiagnosis ? (
            <>
              <header>
                <strong>
                  {ruleLabels[selectedDiagnosis.rule] ?? selectedDiagnosis.rule.replaceAll("_", " ").toUpperCase()}
                </strong>
                <span className={`status-${selectedDiagnosis.status.replace(/_/g, "-")}`}>
                  {ruleStatusLabels[selectedDiagnosis.status]}
                </span>
              </header>

              {selectedDiagnosis.status === "fired" ? (
                <>
                  <p className="rule-evidence-state">
                    EVIDENCE: {(selectedDiagnosis.evidenceState ?? "observed").toUpperCase()}
                    {selectedDiagnosis.evidenceState === "correlated"
                      && " — correlated with the outcome, not shown to be its cause"}
                  </p>
                  {selectedDiagnosis.likelyCause && <p>{selectedDiagnosis.likelyCause}</p>}
                  {selectedDiagnosis.possibleFix && (
                    <p className="rule-fix"><b>POSSIBLE FIX</b>{selectedDiagnosis.possibleFix}</p>
                  )}
                </>
              ) : (
                // not_fired and unavailable are both rendered from `reason` -- the backend names the
                // exact evidence it checked (not_fired) or the exact evidence it lacks (unavailable).
                // Never a placeholder dash: see clozn/runs/perf_diagnosis.py's module docstring.
                <p>{selectedDiagnosis.reason ?? "No further detail was recorded for this rule."}</p>
              )}

              {selectedDiagnosis.evidence.length > 0 && (
                <dl>
                  {selectedDiagnosis.evidence.map((evidence, index) => (
                    <div key={`${evidence.path}-${index}`}>
                      <dt>{evidence.path}</dt>
                      <dd>{ruleEvidenceValue(evidence)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </>
          ) : (
            <div className="lens-performance-state">RULE ENGINE UNAVAILABLE</div>
          )}
        </div>
      </div>
    </section>
  );
}
