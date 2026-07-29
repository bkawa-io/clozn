import { useEffect, useMemo, useState } from "react";
import { applyPromotion, previewPromotion } from "./api";
import type { PromotionPreview, PromotionTransaction } from "./types";
import type { CellSelection } from "./urlState";

interface PromotionPanelProps {
  experimentId: string;
  selection: CellSelection;
  canPromote: boolean;
}

function parseReplacements(value: string): Record<string, string> | undefined {
  if (!value.trim()) return undefined;
  const parsed: unknown = JSON.parse(value);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Redactions must be a JSON object mapping exact text to replacement text.");
  }
  for (const [source, replacement] of Object.entries(parsed)) {
    if (!source || typeof replacement !== "string") {
      throw new Error("Every redaction must map a non-empty string to a string.");
    }
  }
  return parsed as Record<string, string>;
}

export function PromotionPanel({ experimentId, selection, canPromote }: PromotionPanelProps) {
  const [destination, setDestination] = useState("");
  const [caseName, setCaseName] = useState(selection.case);
  const [redactions, setRedactions] = useState("");
  const [preview, setPreview] = useState<PromotionPreview | null>(null);
  const [acknowledged, setAcknowledged] = useState<Set<string>>(new Set());
  const [transaction, setTransaction] = useState<PromotionTransaction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [working, setWorking] = useState(false);

  useEffect(() => {
    setCaseName(selection.case);
    setPreview(null);
    setTransaction(null);
    setAcknowledged(new Set());
    setError(null);
  }, [selection.case, selection.seed, selection.suite, selection.variant]);

  const request = () => ({
    suite: selection.suite,
    case: selection.case,
    variant: selection.variant,
    seed: selection.seed,
    destination,
    case_name: caseName,
    replacements: parseReplacements(redactions),
  });

  const allReviewed = useMemo(
    () => preview?.requiredAcknowledgements.every((id) => acknowledged.has(id)) ?? false,
    [acknowledged, preview],
  );

  const runPreview = async () => {
    setWorking(true);
    setError(null);
    setTransaction(null);
    try {
      const next = await previewPromotion(experimentId, request());
      setPreview(next);
      setAcknowledged(new Set());
    } catch (reason) {
      setPreview(null);
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setWorking(false);
    }
  };

  const runApply = async () => {
    if (!preview) return;
    setWorking(true);
    setError(null);
    try {
      const applied = await applyPromotion(experimentId, {
        ...request(),
        expected_destination_hash: preview.expectedDestinationHash,
        acknowledged_findings: [...acknowledged].sort(),
      });
      setTransaction(applied);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setWorking(false);
    }
  };

  if (!canPromote) {
    return (
      <p className="experiments-drawer-unavailable">
        PROMOTION UNAVAILABLE — this cell has no successful recorded run.
      </p>
    );
  }

  return (
    <div className="experiments-promotion">
      <p className="experiments-drawer-note">
        Preview is read-only. Findings are never removed automatically; redact exact text or explicitly
        acknowledge every remaining finding before apply.
      </p>
      <div className="experiments-promotion-fields">
        <label>
          <span>DESTINATION ARTIFACT</span>
          <input
            value={destination}
            onChange={(event) => setDestination(event.target.value)}
            placeholder="my-regression-suite.json"
          />
        </label>
        <label>
          <span>CASE NAME</span>
          <input value={caseName} onChange={(event) => setCaseName(event.target.value)} />
        </label>
        <label className="is-wide">
          <span>EXACT REDACTIONS (OPTIONAL JSON)</span>
          <textarea
            value={redactions}
            onChange={(event) => setRedactions(event.target.value)}
            placeholder={'{"literal secret":"[REDACTED]"}'}
          />
        </label>
      </div>
      <button
        type="button"
        disabled={working || !destination || !caseName}
        onClick={() => void runPreview()}
      >
        {working ? "WORKING…" : "PREVIEW PROMOTION"}
      </button>
      {error && <p className="experiments-error-text">{error}</p>}
      {preview && (
        <div className="experiments-promotion-preview">
          <p>
            {preview.destinationDiff.operation.toUpperCase()} · {preview.destinationDiff.beforeCaseCount}
            {" → "}{preview.destinationDiff.afterCaseCount} cases · source {preview.sourceRunId}
          </p>
          <code>EXPECTED {preview.expectedDestinationHash}</code>
          {preview.redactionFindings.length === 0 ? (
            <p className="is-clean">SCANNER: NO FINDINGS</p>
          ) : (
            <ul>
              {preview.redactionFindings.map((finding) => (
                <li key={finding.id}>
                  <label>
                    <input
                      type="checkbox"
                      checked={acknowledged.has(finding.id)}
                      onChange={(event) => {
                        const next = new Set(acknowledged);
                        if (event.target.checked) next.add(finding.id);
                        else next.delete(finding.id);
                        setAcknowledged(next);
                      }}
                    />
                    <span>{finding.kind} · {finding.path} · {finding.preview}</span>
                  </label>
                </li>
              ))}
            </ul>
          )}
          <button type="button" disabled={working || !allReviewed} onClick={() => void runApply()}>
            APPLY REVIEWED PROMOTION
          </button>
        </div>
      )}
      {transaction && (
        <p className="experiments-promotion-success">
          APPLIED {transaction.transactionId} · transaction {transaction.transactionPath}
          {transaction.backup ? ` · backup ${transaction.backup}` : ""}
        </p>
      )}
    </div>
  );
}
