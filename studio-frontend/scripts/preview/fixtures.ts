/**
 * Fixture bodies for the surface-preview generator (see `capture-surfaces.spec.tsx`).
 *
 * Every function here is copied -- as literally as TypeScript allows -- from the fixture builders
 * already living next to each component in its own `*.test.tsx` file. That is deliberate: the task this
 * generator serves says plainly "do not invent new fixtures where a test fixture exists -- the test data
 * is what the wire format really produces." Copying (rather than importing the `.test.tsx` files
 * directly) avoids re-executing those files' own `describe`/`test` blocks as a side effect of borrowing
 * their fixture literals -- importing a `.test.tsx` module from outside the component test run would
 * register and (depending on the runner) execute its whole suite a second time, which is not what this
 * generator is for.
 *
 * Each builder below names its source file in a comment. If a source fixture changes, this file will
 * drift -- that is an accepted, disclosed cost of not importing directly (see this repo's report on the
 * preview generator for the tradeoff).
 */

// ============================================================================================ DiagnosisRepair
// Source: src/features/lens/DiagnosisRepair.test.tsx

export const FIVE_STATUS_FINDINGS = [
  {
    rule_id: "R01", rule_name: "input_omitted_or_rejected", status: "finding",
    severity: "medium", confidence: "exact",
    summary: "1 input segment was omitted.",
    evidence: [], limitations: [],
    suggested_actions: [{ kind: "resend_context", description: "resend the omitted segment" }],
  },
  {
    rule_id: "R02", rule_name: "context_budget_pressure", status: "not_observed",
    summary: "the prompt used 10% of context tokens.", evidence: [], limitations: [],
  },
  {
    rule_id: "R08", rule_name: "source_below_measurement_floor", status: "unavailable",
    summary: "no readable influence artifact was recorded.", evidence: [], limitations: [],
  },
  {
    rule_id: "R09", rule_name: "source_little_effect", status: "pending",
    summary: "this run never recorded an influence map.", evidence: [], limitations: [],
  },
  {
    rule_id: "R03", rule_name: "conflicting_instructions", status: "suppressed",
    summary: "R03 was suppressed for this evaluation by caller request.", evidence: [], limitations: [],
  },
  {
    rule_id: "R04", rule_name: "duplicate_instructions", status: "finding",
    severity: "low", confidence: "pattern_match",
    summary: "duplicate instructions were found.",
    evidence: [], limitations: [],
    suggested_actions: [],
  },
];

export function diagnosisFindingsBody(runId: string, findings: Record<string, unknown>[] = FIVE_STATUS_FINDINGS) {
  const counts = { finding: 0, not_observed: 0, unavailable: 0, pending: 0, suppressed: 0 };
  for (const finding of findings) counts[finding.status as keyof typeof counts] += 1;
  return {
    findings: {
      schema_version: "clozn.diagnosis-findings.v1",
      generated_at: "2026-01-01T00:00:00Z",
      run_id: runId,
      redacted: false,
      rule_registry: findings.map((f) => ({ rule_id: f.rule_id, rule_name: f.rule_name })),
      suppressed_rule_ids: findings.filter((f) => f.status === "suppressed").map((f) => f.rule_id),
      findings,
      summary: { status_counts: counts },
    },
    narrative: {
      schema_version: "clozn.diagnosis-narrative.v1",
      generated_at: "2026-01-01T00:00:00Z",
      run_id: runId,
      comparison_available: false,
      findings_schema_version: "clozn.diagnosis-findings.v1",
      headline: "no comparison run supplied.",
      registers: { observed_changes: [], measured_effects: [], plausible_but_unproven: [] },
      summary: { counts: { observed_changes: 0, measured_effects: 0, plausible_but_unproven: 0 } },
    },
  };
}

export function correctiveRegistryBody(runId: string) {
  return {
    schema_version: "clozn.action-registry.v1",
    version: "1",
    run_id: runId,
    run_fingerprint: "fp-" + runId,
    actions: [
      {
        id: "less-verbose",
        label: "More concise",
        description: "For this reply, answer concisely.",
        conflicts: [],
        scopes: ["once"],
        eligibility: { eligible: true },
        evaluation_metrics: [],
        backends: [{ type: "prompt_policy", available: true }],
        scope_eligibility: [
          { scope: "once", available: true, prior_hash: "hash-once" },
        ],
      },
      {
        id: "use-context",
        label: "Ground in supplied materials",
        description: "Ask the model to cite the supplied materials and say what is missing.",
        conflicts: [],
        scopes: ["once"],
        eligibility: { eligible: true },
        evaluation_metrics: [],
        backends: [{ type: "prompt_policy", available: true }],
        scope_eligibility: [{ scope: "once", available: true, prior_hash: "hash-once-b" }],
      },
    ],
  };
}

export function previewBody() {
  return {
    preview_id: `fix_preview_${"a".repeat(20)}`,
    status: "ready",
    created_ts: 1700000000,
    expires_ts: 1700003600,
    parent_run_id: "run-demo-diagnosis",
    parent_run_fingerprint: "fp",
    action: { id: "less-verbose", label: "More concise", description: "For this reply, answer concisely." },
    execution: {
      requested_backend: "prompt_policy",
      expected_executed_backend: "prompt_policy",
      expected_fallback: false,
      qualification: "generic",
      qualification_id: "clozn.prompt-policy.generic.v1",
    },
    scope_eligibility: [
      { scope: "once", available: true, prior_hash: "hash-once" },
    ],
    comparison_contract: {
      baseline: "matched greedy replay under the current runtime policy",
      corrected: "matched greedy replay with the bounded action",
      stored_original: "context only; it may have been sampled",
    },
  };
}

// ================================================================================================ WhatMattered
// Source: src/features/lens/WhatMattered.test.tsx

export function investigation(runId: string, overrides: Record<string, unknown> = {}) {
  return {
    schema_version: "clozn.run-investigation.v1",
    run_id: runId,
    sections: {
      received_context: {
        state: "delivered_not_measured",
        privacy: "metadata_only",
        delivered: [],
        assembled: [],
        omitted: [
          {
            segment_id: "seg-tool",
            source_type: "message",
            source_label: "tool",
            original_order: 2,
            reason: "tool_result_pruned",
          },
        ],
        limits: {},
      },
      text_span_addresses: {
        state: "supported",
        privacy: "metadata_only",
        href: `/runs/${runId}/span-addresses`,
        address_count: 4,
      },
      prompt_source_influence: {
        state: "measured_effect",
        privacy: "metadata_only",
        prompt_sources: [
          {
            id: "p.m000",
            segment_id: "seg-system",
            source_label: "system",
            role: "system",
            selected: true,
            start: 0,
            end: 50,
            text_sha256: "a".repeat(64),
            text_bytes: 50,
          },
          {
            id: "p.m001",
            segment_id: "seg-user",
            source_label: "user",
            role: "user",
            selected: false,
            start: 0,
            end: 80,
            text_sha256: "b".repeat(64),
            text_bytes: 80,
          },
        ],
        prompt_spans: [
          {
            id: "p.m000.c000",
            parent_id: "p.m000",
            level: "coarse",
            segment_id: "seg-system",
            source_label: "system",
            role: "system",
            start: 0,
            end: 50,
            text_sha256: "a".repeat(64),
            text_bytes: 50,
          },
        ],
        answer_spans: [
          {
            id: "a.t0000",
            level: "token",
            token_index: 0,
            token_id: 5,
            start: 0,
            end: 3,
            text_sha256: "c".repeat(64),
            text_bytes: 3,
          },
          {
            id: "a.t0001",
            level: "token",
            token_index: 1,
            token_id: 9,
            start: 3,
            end: 6,
            text_sha256: "d".repeat(64),
            text_bytes: 3,
          },
        ],
        links: [
          {
            context_span_id: "p.m000.c000",
            answer_span_id: "a.t0000",
            context_index: 0,
            answer_index: 0,
            delta_nats: 0.9,
            abs_delta_nats: 0.9,
            effect: "supports",
            clears_floor: true,
            evidence_state: "causally_supported",
          },
          {
            context_span_id: "p.m000.c000",
            answer_span_id: "a.t0001",
            context_index: 0,
            answer_index: 1,
            delta_nats: -0.01,
            abs_delta_nats: 0.01,
            effect: "suppresses",
            clears_floor: false,
            evidence_state: "observed",
          },
        ],
        thresholds: {
          cell_abs_delta_nats: 0.05,
          source_clear_rule: "absolute signed cell delta meets or exceeds cell_abs_delta_nats",
          calibration: "fixed_default_not_model_calibrated",
        },
      },
    },
    actions: [
      {
        id: "measure_prompt_source_influence",
        label: "Measure what mattered",
        kind: "measurement",
        method: "POST",
        href: `/runs/${runId}/influence-map/jobs`,
        availability: "ready",
      },
    ],
    unavailable_measurements: [],
    ...overrides,
  };
}

export function unmeasuredInvestigation(runId: string, availability = "ready", reason?: string) {
  const base = investigation(runId);
  return {
    ...base,
    sections: {
      ...base.sections,
      prompt_source_influence: {
        state: "delivered_not_measured",
        privacy: "metadata_only",
        reason: "context delivery is recorded, but no prompt/source influence experiment has run",
        prompt_sources: [],
        prompt_spans: [],
        answer_spans: [],
        links: [],
        thresholds: {},
      },
    },
    actions: [
      {
        id: "measure_prompt_source_influence",
        label: "Measure what mattered",
        kind: "measurement",
        method: "POST",
        href: `/runs/${runId}/influence-map/jobs`,
        availability,
        ...(reason ? { reason } : {}),
      },
    ],
  };
}

export function spanDocument(runId: string) {
  return {
    schema_version: "clozn.text-span-addresses.v1",
    run_id: runId,
    privacy: "metadata_only",
    offset_contract: {
      unit: "unicode_code_points",
      interval: "half_open",
      hash_algorithm: "sha256",
      canonicalization: "exact_string_utf8_v1",
    },
    source_artifacts: [],
    addresses: [],
    lineage: { parent_run_id: null, mappings: [] },
  };
}

// ============================================================================================ ClaimVerification
// Source: src/features/lens/ClaimVerification.test.tsx

interface ClaimSpec {
  text: string;
  category: string;
  status: string;
  method: Record<string, unknown>;
  sourceSpanIds?: string[];
}

export function buildClaimFixture(runId: string, specs: ClaimSpec[]) {
  let cursor = 0;
  const parts: string[] = [];
  const claims: Record<string, unknown>[] = [];
  const results: Record<string, unknown>[] = [];
  specs.forEach((spec, index) => {
    if (index > 0) {
      parts.push(" ");
      cursor += 1;
    }
    const start = cursor;
    parts.push(spec.text);
    cursor += spec.text.length;
    const end = cursor;
    const addressId = `span_claim${String(index).padStart(3, "0")}`;
    claims.push({
      index,
      category: spec.category,
      category_reason: "factual_declarative",
      text_span: {
        address_id: addressId,
        run_id: runId,
        kind: "claim",
        relation_key: `rel_claim${index}`,
        native_ref: { artifact_schema: "clozn.answer-claims.v1", collection: "derived.claims", id: `claim-${index}` },
        resolution: {
          state: "metadata_only",
          canonical: {
            basis: "recorded_answer", unit: "unicode_code_points", interval: "half_open",
            start, end, basis_sha256: "a".repeat(64), span_sha256: "b".repeat(64),
          },
        },
      },
    });
    results.push({
      claim_index: index,
      claim_address_id: addressId,
      status: spec.status,
      method: spec.method,
      ...(spec.sourceSpanIds ? { source_span_ids: spec.sourceSpanIds } : {}),
    });
  });
  return { text: parts.join(""), claims, results };
}

const OFFSET_CONTRACT = {
  unit: "unicode_code_points", interval: "half_open", hash_algorithm: "sha256",
  canonicalization: "exact_string_utf8_v1",
};

export function claimSupportBody(runId: string, fixture: ReturnType<typeof buildClaimFixture>, gate = "ok") {
  return {
    claims: {
      schema_version: "clozn.answer-claims.v1",
      run_id: runId,
      privacy: "metadata_only",
      offset_contract: OFFSET_CONTRACT,
      segmentation: { state: "ok", claim_count: fixture.claims.length },
      answer_source: {
        basis: "recorded_answer", basis_sha256: "c".repeat(64),
        basis_code_points: fixture.text.length, basis_utf8_bytes: fixture.text.length,
      },
      claims: fixture.claims,
    },
    support: {
      schema_version: "clozn.claim-support.v1",
      run_id: runId,
      privacy: "metadata_only",
      offset_contract: OFFSET_CONTRACT,
      source: { claims_schema_version: "clozn.answer-claims.v1", influence_map: { gate } },
      results: fixture.results,
    },
  };
}

export function claimRunBody(runId: string, responseText: string) {
  return { id: runId, response: responseText };
}

export function claimRegistryBody(runId: string, actionIds: string[]) {
  return {
    schema_version: "clozn.action-registry.v1",
    version: "1",
    run_id: runId,
    actions: actionIds.map((id) => ({
      id,
      label: id,
      description: `Corrective description for ${id}.`,
      conflicts: [],
      scopes: ["once"],
      eligibility: { eligible: true },
      evaluation_metrics: [],
      backends: [{ type: "prompt_policy", available: true }],
      scope_eligibility: [{ scope: "once", available: true, prior_hash: "hash-once" }],
    })),
  };
}

export const SHARED_CLAIM_SOURCE_ID = "span_source0000000000001aaaa";
export const OTHER_CLAIM_SOURCE_ID = "span_source0000000000002bbbb";

export const SIX_STATUS_CLAIM_SPECS: ClaimSpec[] = [
  {
    text: "Paris is the capital of France.", category: "factual_claim", status: "supported",
    method: { name: "forced_score_intervention", max_abs_delta_nats: 1.2345 },
    sourceSpanIds: [SHARED_CLAIM_SOURCE_ID],
  },
  {
    text: "The bridge was likely finished around then.", category: "factual_claim", status: "weakly_supported",
    method: { name: "textual_overlap", overlap_fraction: 0.5 }, sourceSpanIds: [OTHER_CLAIM_SOURCE_ID],
  },
  {
    text: "The event happened in 1950.", category: "factual_claim", status: "contradicted",
    method: { name: "numeric_or_date_mismatch" }, sourceSpanIds: [SHARED_CLAIM_SOURCE_ID],
  },
  {
    text: "The moon is made of cheese.", category: "factual_claim", status: "unsupported_by_supplied_materials",
    method: { name: "measured_comparison_no_match" },
  },
  {
    text: "Water boils at exactly 100 degrees.", category: "factual_claim", status: "measurement_unavailable",
    method: { name: "no_influence_map" },
  },
  {
    text: "You should try the new approach.", category: "recommendation", status: "unverifiable_from_available_evidence",
    method: { name: "category_rule" },
  },
];

// =================================================================================== ConversationInvestigation
// Source: src/features/investigation/ConversationInvestigation.test.tsx

export function turnFixture(runId: string, overrides: Record<string, unknown> = {}) {
  return {
    run_id: runId,
    recorded_ts: 1700000000,
    created_at: "2023-11-14T00:00:00",
    source: "chat",
    client: "web",
    model: "test-model",
    prompt_summary: `prompt for ${runId}`,
    response_summary: `response for ${runId}`,
    redacted: false,
    cumulative: { turn_count: 1, duration_ms_total: 100, prompt_tokens_total: 10, generated_tokens_total: 5 },
    diagnostic_highlights: {
      findings: [],
      status_counts: { finding: 0, not_observed: 10, unavailable: 0, pending: 2, suppressed: 0 },
    },
    ...overrides,
  };
}

export function pageFixture(sessionId: string, turns: unknown[], overrides: Record<string, unknown> = {}) {
  return {
    schema_version: "clozn.session-trace.v1",
    generated_at: "2023-11-14T00:00:00Z",
    session_id: sessionId,
    session: { id: sessionId, privacy: {} },
    page: { cursor: null, next_cursor: null, limit: 50, count: turns.length },
    turns,
    branches: [],
    totals_through_this_page: {
      turn_count: turns.length, duration_ms_total: 100, prompt_tokens_total: 10, generated_tokens_total: 5,
    },
    diagnostic_rule_registry: [],
    first_went_wrong_candidates: [],
    ...overrides,
  };
}

// ========================================================================================== SessionPicker
// Source: src/features/investigation/SessionPicker.test.tsx

export function sessionFixture(id: string, overrides: Record<string, unknown> = {}) {
  return {
    id, created_at: "2023-11-14T00:00:00", created_ts: 1700000000,
    privacy: { visibility: "visible" }, materialized_from: "explicit",
    ...overrides,
  };
}

// ========================================================================================= ForkOutcomePanel
// Source: src/features/observatory/ForkOutcomePanel.test.tsx -- these are typed component props, not
// fetch bodies (ForkOutcomePanel takes `outcome`/`note` directly; it never fetches).

export const FORK_OUTCOME_EXACT = {
  note: "exact execution fork: the worker restored its exact recorded KV state and applied the "
    + "forced token there directly on its token id -- no text splice, nothing to retokenize",
  outcome: {
    kind: "exact_execution_fork" as const,
    reasons: [{
      code: "exact_preconditions_met",
      message: "an exact checkpoint was captured and its intervention completed",
    }],
    exactness: {
      regime: "generated_token_live_kv",
      source: "live_kv",
      proofStatus: "confirmed",
      truncateTo: 42,
    },
    unchangedControl: {
      required: true,
      status: "matched",
      result: {
        status: "matched",
        exactMatch: true,
        note: "parent suffix token ids and text matched exactly",
      },
    },
    intervention: {
      type: "force_token",
      tokenId: 4242,
      tokenPiece: "alternate",
      restoreMode: "live_kv_truncated",
    },
    executionId: "fork_exec_abc123",
  },
};

export const FORK_OUTCOME_RECONSTRUCTED = {
  note: "greedy continuation (sample=false): a deterministic what-if",
  outcome: {
    kind: "reconstructed_replay" as const,
    reasons: [{
      code: "checkpoint_not_supplied",
      message: "no exact checkpoint was supplied; the eligible path explicitly reconstructs text",
    }],
    exactness: {
      regime: "reconstructed_text",
      source: "text_retokenization",
      proofStatus: "not_applicable",
    },
    unavoidableDifferences: [
      "kv_state_not_restored",
      "sampler_state_reinitialized",
      "prompt_prefix_retokenized",
      "batch_shape_not_preserved",
    ],
    retokenized: true,
  },
};

export const FORK_OUTCOME_UNAVAILABLE = {
  outcome: {
    kind: "unavailable" as const,
    reasons: [{
      code: "checkpoint_expired",
      message: "the referenced checkpoint has expired or been evicted",
    }],
  },
};

// ==================================================================================================== App shell
// Not copied from a test file -- App.tsx has no .test.tsx of its own (smoke-render.mjs covers its
// routing at the SSR/no-effects level only). This fixture is new, kept intentionally minimal: it exists
// only to get the rail nav + topbar chrome into a "CONNECTED" state, not to exercise any one panel's own
// evidence. See capture-surfaces.spec.tsx's own note on the shell state for what this does and does not
// prove.

export function healthzBody() {
  return { status: "ok" };
}

export function runtimeRunsBody() {
  return {
    runs: [
      {
        id: "run-demo-001", prompt_summary: "Summarize the Q3 incident report",
        response_summary: "The Q3 incident was caused by...", source: "chat", client: "web",
        model: "demo-model-7b", substrate: "gguf", created_at: "2026-07-28T10:00:00Z",
        created_ts: 1753700000, timing: { duration_ms: 1830 },
      },
      {
        id: "run-demo-002", prompt_summary: "Diff the retry against the original",
        response_summary: "The retry is more concise and...", source: "api", client: "sdk",
        model: "demo-model-7b", substrate: "gguf", created_at: "2026-07-28T10:05:00Z",
        created_ts: 1753700300, timing: { duration_ms: 940 },
      },
    ],
  };
}

export function engineHealthBody() {
  return {
    engine: {
      model: "/models/demo-model-7b.gguf",
      n_layer: 32,
      capabilities: { jlens: true, sae: false },
    },
  };
}
