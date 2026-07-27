import type {
  DiagnosisEvidence,
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

function evidenceValue(evidence: DiagnosisEvidence) {
  if (typeof evidence.value === "number") {
    return Number.isInteger(evidence.value)
      ? evidence.value.toLocaleString()
      : evidence.value.toFixed(3);
  }
  if (typeof evidence.value === "string" || typeof evidence.value === "boolean") {
    return String(evidence.value);
  }
  if (evidence.value == null) return "—";
  try {
    return JSON.stringify(evidence.value);
  } catch {
    return "—";
  }
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
    </section>
  );
}
