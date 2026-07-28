import type {
  ContextCoverage,
  InfluenceAbsence,
  InfluenceErrorCode,
  InfluenceEvidenceState,
  InfluenceMethod,
  InfluenceThresholds,
} from "../../data/types";

/**
 * The sentence that licenses (or refuses to license) every delta_nats number the Sources lens shows.
 * The backend already computes `method` (mode, floor, claim_limit, persistent caveat) and a typed
 * `evidence_state`/absence reason per link -- this file is where those get turned into copy instead of
 * staying numbers a reader has to trust blind. See clozn/receipts/context_answer_influence.py and
 * notes/agent_roadmap/07-sources-evidence-lens.md.
 *
 * Every switch below ends in a `never` assignment. That is not decoration: it is what keeps
 * "INCONCLUSIVE" (here: an `observed` link, or a typed absence reason) from ever being rendered as
 * though it were the stronger claim. A value this module hasn't been taught about fails the TypeScript
 * build (`tsc -b` in the verification steps) instead of silently taking whatever label the switch falls
 * through to.
 */

export function evidenceStateBadge(state: InfluenceEvidenceState): { label: string; className: string } {
  switch (state) {
    case "causally_supported":
      return { label: "CLEARED FLOOR", className: "is-evidence-causal" };
    case "observed":
      return { label: "BELOW FLOOR", className: "is-evidence-observed" };
    default: {
      const exhaustive: never = state;
      return exhaustive;
    }
  }
}

function absenceCodeCopy(code: InfluenceErrorCode): { title: string; detail: string } {
  switch (code) {
    case "invalid_run":
      return { title: "RUN NOT MEASURABLE", detail: "This run has no data the measurement can run against." };
    case "no_text_context":
      return { title: "NO CONTEXT TO MEASURE", detail: "The recorded run has no text context spans to measure -- there is nothing to attribute the answer to." };
    case "no_recorded_continuation":
      return { title: "NO RECORDED CONTINUATION", detail: "The run has neither recorded continuation token IDs nor response text to score against." };
    case "scoring_unavailable":
      return { title: "SCORING UNAVAILABLE ON THIS BUILD", detail: "Teacher-forced token scoring is unavailable on the attached worker -- this run cannot be measured until it is." };
    case "invalid_baseline_score":
      return { title: "BASELINE SCORE INVALID", detail: "The baseline scorer returned no finite, aligned token log-probabilities." };
    case "intervention_score_failed":
      return { title: "INTERVENTION SCORING FAILED", detail: "A matched-control measurement did not complete or did not align token-for-token." };
    case "influence_map_error":
      return { title: "SOURCE MEASUREMENT FAILED", detail: "The measurement raised an unexpected error partway through." };
    default: {
      const exhaustive: never = code;
      return exhaustive;
    }
  }
}

/** Turns a typed absence reason into copy, never collapsing distinct reasons into one generic message. */
export function describeAbsence(reason: InfluenceAbsence): { title: string; detail?: string } {
  switch (reason.kind) {
    case "not_measured":
      return {
        title: "SOURCES NOT YET MEASURED",
        detail: "No source map has been computed for this run yet.",
      };
    case "no_worker":
      return {
        title: "SCORING UNAVAILABLE ON THIS BUILD",
        detail: "No worker with token scoring is attached to this server -- a build/deployment limitation, not a property of this run.",
      };
    case "typed":
      return absenceCodeCopy(reason.code);
    case "invalid_request":
      return { title: "MEASUREMENT REQUEST REJECTED", detail: reason.message };
    case "server_error":
      return { title: "SOURCE MEASUREMENT FAILED", detail: reason.message };
    case "network_error":
      return { title: "SOURCE MEASUREMENT FAILED", detail: reason.message };
    default: {
      const exhaustive: never = reason;
      return exhaustive;
    }
  }
}

function formatMode(mode: string): string {
  return mode.replace(/_/g, " ").toUpperCase();
}

export function EvidenceCaveat({ method, thresholds, coverage }: {
  method?: InfluenceMethod;
  thresholds?: InfluenceThresholds;
  coverage?: ContextCoverage;
}) {
  if (!method) return null;
  return (
    <section className="lens-evidence-caveat" aria-label="Measurement method and caveat">
      <header>
        <span>{formatMode(method.mode)}</span>
        {thresholds?.cellAbsDeltaNats != null && (
          <b>FLOOR {thresholds.cellAbsDeltaNats.toFixed(4)} NATS/TOKEN</b>
        )}
      </header>
      {coverage && coverage.totalSources > 0 && (
        <p className="lens-evidence-coverage">
          {coverage.measuredSources} / {coverage.totalSources} SOURCES MEASURED
          {!coverage.complete && coverage.omittedSources > 0
            ? ` · ${coverage.omittedSources} OMITTED (BOUND SELECTION)`
            : ""}
        </p>
      )}
      <p className="lens-evidence-caveat-text">{method.caveat}</p>
      <p className="lens-evidence-claim-limit">
        <b>DOES NOT LICENSE </b>
        {method.claimLimit}
      </p>
    </section>
  );
}
