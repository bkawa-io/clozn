import { afterEach, describe, expect, test, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "../../test/render";
import { TimeMachine } from "./TimeMachine";

const receipt = {
  schema_version: "clozn.time-machine-eligibility.v1",
  run_id: "run-1",
  state: "structurally_reproducible",
  eligible: true,
  exact_replay: {
    eligible: false,
    reason: { code: "exact_replay_not_available", message: "restore is pending" },
  },
  reasons: [{ code: "structural_replay_available", message: "messages can be replayed" }],
  turns: [
    {
      turn: 0,
      branch_eligible: true,
      replay_fidelity: "structurally_reproducible",
      exact_replay_eligible: false,
      snapshot: null,
      source: {
        status: "available",
        run_id: "run-turn-0",
        scope: "session_turn_prompt_boundary",
        source_turn: 0,
        durable_pin: {
          status: "unavailable",
          reason: { code: "durable_pin_missing", message: "no restart-safe pin" },
        },
        reasons: [{ code: "organic_session_source_resolved", message: "matched session source" }],
      },
      reasons: [{ code: "structural_replay_only", message: "transcript replay" }],
    },
    {
      turn: 1,
      branch_eligible: true,
      replay_fidelity: "structurally_reproducible",
      exact_replay_eligible: false,
      snapshot: null,
      source: {
        status: "available",
        run_id: "run-1",
        scope: "full_run_prompt_boundary",
        source_turn: 1,
        durable_pin: {
          status: "unavailable",
          reason: { code: "durable_pin_missing", message: "no restart-safe pin" },
        },
        reasons: [{ code: "requested_run_source_resolved", message: "matched requested source" }],
      },
      reasons: [{ code: "structural_replay_only", message: "transcript replay" }],
    },
  ],
};

afterEach(() => vi.restoreAllMocks());

describe("TimeMachine", () => {
  test("loads eligibility and keeps turn selection side-effect free", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(receipt), { status: 200 }),
    );
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByText("STRUCTURALLY REPRODUCIBLE")).toBeInTheDocument());
    fireEvent.change(screen.getByRole("combobox", { name: "Branch from turn" }), { target: { value: "1" } });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "CREATE CHILD BRANCH" })).toBeInTheDocument();
  });

  test("creates a child only after the explicit branch action", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(receipt), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ id: "child-12345678" }), { status: 200 }));
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "CREATE CHILD BRANCH" })).toBeInTheDocument());
    fireEvent.change(screen.getByRole("textbox", { name: "Optional replacement question" }), {
      target: { value: "ask something else" },
    });
    fireEvent.click(screen.getByRole("button", { name: "CREATE CHILD BRANCH" }));
    await waitFor(() => expect(screen.getByText(/Child branch created/)).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const request = fetchMock.mock.calls[1][1] as RequestInit;
    expect(request.method).toBe("POST");
    expect(JSON.parse(String(request.body))).toEqual({ turn: 0, alt_user: "ask something else" });
  });

  test("creates an exact child replay only after its explicit action", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(receipt), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        schema_version: "clozn.time-machine-branch.v1",
        requested_run_id: "run-1",
        source_run_id: "run-1",
        source_turn: 0,
        status: "completed",
        exact_replay: true,
        fidelity: "exact_replay_eligible",
        child_run_id: "run-exact-child",
        execution_fork_execution_id: "fork_exec_1",
        reasons: [{ code: "exact_child_replay_completed", message: "matched" }],
        capture: {},
        execution_fork: {},
      }), { status: 201 }));
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "CREATE EXACT CHILD REPLAY" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "CREATE EXACT CHILD REPLAY" }));
    await waitFor(() => expect(screen.getByText(/Exact child replay created/)).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const request = fetchMock.mock.calls[1][1] as RequestInit;
    expect(request.method).toBe("POST");
    expect(JSON.parse(String(request.body))).toEqual({ turn: 0 });
  });

  test("verifies the selected latest turn only after an explicit action", async () => {
    const verification = {
      schema_version: "clozn.time-machine-verification.v1",
      verification_id: "tmv-1",
      parent_run_id: "run-1",
      turn: 1,
      scope: "full_run_prompt_boundary",
      status: "verified",
      exact_replay: true,
      fidelity: "exact_replay_eligible",
      exactness_regime: "prompt_boundary_reprefill",
      reasons: [{ code: "exact_prompt_boundary_verified", message: "matched" }],
    };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(receipt), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(verification), { status: 201 }));
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "VERIFY EXACT PROMPT BOUNDARY" })).toBeInTheDocument());
    fireEvent.change(screen.getByRole("combobox", { name: "Branch from turn" }), { target: { value: "1" } });
    fireEvent.click(screen.getByRole("button", { name: "VERIFY EXACT PROMPT BOUNDARY" }));
    await waitFor(() => expect(screen.getByText("Exact prompt-boundary replay verified for this run.")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const request = fetchMock.mock.calls[1][1] as RequestInit;
    expect(request.method).toBe("POST");
    expect(JSON.parse(String(request.body))).toEqual({ turn: 1 });
  });

  test("enables an earlier turn's explicit verification when an organic source is resolved", async () => {
    const verification = {
      schema_version: "clozn.time-machine-verification.v1",
      verification_id: "tmv-historical",
      parent_run_id: "run-turn-0",
      requested_run_id: "run-1",
      source_run_id: "run-turn-0",
      source_turn: 0,
      turn: 0,
      scope: "session_turn_prompt_boundary",
      status: "verified",
      exact_replay: true,
      fidelity: "exact_replay_eligible",
      exactness_regime: "prompt_boundary_reprefill",
      reasons: [{ code: "exact_prompt_boundary_verified", message: "matched" }],
    };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(receipt), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(verification), { status: 201 }));
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "VERIFY EXACT PROMPT BOUNDARY" })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: "VERIFY EXACT PROMPT BOUNDARY" }));
    await waitFor(() => expect(screen.getByText(/verified from session run run-turn-0/)).toBeInTheDocument());
    const request = fetchMock.mock.calls[1][1] as RequestInit;
    expect(JSON.parse(String(request.body))).toEqual({ turn: 0 });
  });

  test("previews then pins the selected historical source without mutating its run", async () => {
    const pinnedReceipt = {
      ...receipt,
      turns: receipt.turns.map((turn, index) => index === 0 ? {
        ...turn,
        source: {
          ...turn.source,
          durable_pin: {
            status: "stored",
            reason: { code: "durable_pin_recorded", message: "rechecked when used" },
            pin: { pin_id: "pin-source", pinned_at: "2026-08-02T00:00:00Z", kv_bytes: 2048, envelope_bytes: 3072 },
          },
        },
      } : turn),
    };
    const manifest = {
      schema_version: "clozn.pinned-checkpoint.v1",
      pin_id: "pin_12345678901234567890",
      run_id: "run-turn-0",
      pinned_at: "2026-08-02T00:00:00Z",
      identity: {}, state: {}, blob: {},
    };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(receipt), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        ok: true, preview: true, run_id: "run-turn-0", size_bytes: 2048, envelope_bytes: 3072,
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true, manifest }), { status: 201 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(pinnedReceipt), { status: 200 }));
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "PREVIEW DURABLE PIN" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "PREVIEW DURABLE PIN" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "PIN 3.0 KB" })).toBeInTheDocument());
    let request = fetchMock.mock.calls[1][1] as RequestInit;
    expect(JSON.parse(String(request.body))).toEqual({ preview: true });
    fireEvent.click(screen.getByRole("button", { name: "PIN 3.0 KB" }));
    await waitFor(() => expect(screen.getByText(/Durable checkpoint recorded/)).toBeInTheDocument());
    request = fetchMock.mock.calls[2][1] as RequestInit;
    expect(JSON.parse(String(request.body))).toEqual({ preview: false });
    expect(String(fetchMock.mock.calls[1][0])).toBe("/runs/run-turn-0/snapshot/pin");
    expect(String(fetchMock.mock.calls[2][0])).toBe("/runs/run-turn-0/snapshot/pin");
  });

  test("requires an explicit cascade choice when a source pin has child runs", async () => {
    const pinnedReceipt = {
      ...receipt,
      turns: receipt.turns.map((turn, index) => index === 0 ? {
        ...turn,
        source: {
          ...turn.source,
          durable_pin: {
            status: "stored",
            reason: { code: "durable_pin_recorded", message: "rechecked when used" },
            pin: { pin_id: "pin-source", pinned_at: "2026-08-02T00:00:00Z", kv_bytes: 2048, envelope_bytes: 3072 },
          },
        },
      } : turn),
    };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(pinnedReceipt), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        error: "children depend on this pin", code: "snapshot_unpin_has_dependents", children: ["child-a", "child-b"],
      }), { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ok: true, action: "unpin", run_id: "run-turn-0", cascade: true }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(receipt), { status: 200 }));
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "UNPIN DURABLE CHECKPOINT" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "UNPIN DURABLE CHECKPOINT" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "CASCADE UNPIN (2 CHILDREN)" })).toBeInTheDocument());
    const first = fetchMock.mock.calls[1][1] as RequestInit;
    expect(JSON.parse(String(first.body))).toEqual({ cascade: false });
    fireEvent.click(screen.getByRole("button", { name: "CASCADE UNPIN (2 CHILDREN)" }));
    await waitFor(() => expect(screen.getByText(/Durable checkpoint unpinned/)).toBeInTheDocument());
    const second = fetchMock.mock.calls[2][1] as RequestInit;
    expect(JSON.parse(String(second.body))).toEqual({ cascade: true });
  });

  test("shows historical source-run provenance after an earlier-turn proof", async () => {
    const historicalReceipt = {
      ...receipt,
      turns: receipt.turns.map((turn, index) => index === 0 ? {
        ...turn,
        last_verification: {
          verification_id: "tmv-0",
          status: "verified",
          fidelity: "exact_replay_eligible",
          exact_replay: true,
          message: "matched",
          scope: "session_turn_prompt_boundary",
          requested_run_id: "run-final",
          source_run_id: "run-turn-0",
          source_turn: 0,
        },
      } : turn),
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(historicalReceipt), { status: 200 }),
    );
    render(<TimeMachine runId="run-final" />);
    await waitFor(() => expect(screen.getByText(/LAST EXACT PROOF: TURN 1/)).toBeInTheDocument());
    expect(screen.getAllByText("run-turn-0")).toHaveLength(3);
    expect(screen.getAllByText("run-final")).toHaveLength(2);
  });

  test("previews why exact appended-turn continuation is unavailable without an exact checkpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify(receipt), { status: 200 }),
    );
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "CONTINUE EXACT HISTORY" })).toBeDisabled());
    fireEvent.change(screen.getByRole("textbox", { name: "New question to append" }), {
      target: { value: "What changed?" },
    });
    expect(screen.getAllByText("no restart-safe pin")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "CONTINUE EXACT HISTORY" })).toBeDisabled();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("continues exact history only after an explicit action with the closed v1 body", async () => {
    const continuation = {
      schema_version: "clozn.time-machine-continuation.v1",
      continuation_id: "tmc_1234567890abcdef1234",
      created_ts: 1,
      finished_ts: 2,
      requested_run_id: "run-1",
      source_turn: 0,
      status: "completed",
      request: {
        request_id: "request-1",
        turn: 0,
        append_kind: "new_user_turn",
        user_content_sha256: "a".repeat(64),
        user_content_bytes: 14,
        max_tokens: 37,
        generation_config_sha256: "b".repeat(64),
      },
      source: {
        status: "resolved",
        source_run_id: "run-turn-0",
        source_turn: 0,
        resolution: "exact_organic_session_prefix",
      },
      source_checkpoint: {
        status: "available",
        provenance: "durable_pin_import",
        restart_safe: true,
        source_run_id: "run-turn-0",
        checkpoint_reference_id: "checkpoint-ref-1",
        checkpoint_id: "checkpoint-1",
        pin_id: "pin_12345678901234567890",
      },
      identity: { status: "matched" },
      append: { status: "validated" },
      sampler: { status: "preserved" },
      exactness: {
        status: "confirmed",
        claim: "exact_historical_state_append",
        append_only_execution: true,
        historical_prefix_recomputed: false,
        historical_prefix_retokenized_for_execution: false,
        structural_fallback_used: false,
      },
      worker: { status: "completed" },
      child_lineage: {
        status: "created",
        requested_parent_run_id: "run-1",
        source_checkpoint_run_id: "run-turn-0",
        child_run_id: "run-continuation-child",
        relation: "exact_continuation",
        parent_immutable: true,
        source_immutable: true,
        receipt_persisted: true,
      },
      unavoidable_differences: ["new_append_tokens", "new_generated_suffix"],
      reasons: [{ code: "continuation_completed", message: "persisted" }],
      failure: null,
    };
    const durableReceipt = {
      ...receipt,
      turns: receipt.turns.map((turn, index) => index === 0 ? {
        ...turn,
        source: {
          ...turn.source,
          durable_pin: {
            status: "stored",
            reason: { code: "durable_pin_recorded", message: "restart-safe pin is ready" },
            pin: { pin_id: "pin-source", pinned_at: "2026-08-02T00:00:00Z", kv_bytes: 2048, envelope_bytes: 3072 },
          },
        },
      } : turn),
    };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(durableReceipt), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(continuation), { status: 201 }));
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByText("DURABLE PIN IS RESTART-SAFE; IMPORT IDENTITY WILL BE RECHECKED.")).toBeInTheDocument());
    fireEvent.change(screen.getByRole("textbox", { name: "New question to append" }), {
      target: { value: "What changed?" },
    });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max output tokens" }), {
      target: { value: "37" },
    });
    fireEvent.click(screen.getByRole("button", { name: "CONTINUE EXACT HISTORY" }));
    await waitFor(() => expect(screen.getByText(/Exact appended-turn child created/)).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const request = fetchMock.mock.calls[1][1] as RequestInit;
    expect(request.method).toBe("POST");
    expect(String(fetchMock.mock.calls[1][0])).toBe("/runs/run-1/time-machine/continue");
    expect(JSON.parse(String(request.body))).toEqual({
      turn: 0,
      user: { content: "What changed?" },
      max_tokens: 37,
    });
    expect(screen.getAllByText(/REQUESTED PARENT/).at(-1)).toHaveTextContent("run-1");
    expect(screen.getByText(/EXACT CONTINUATION/)).toHaveTextContent("run-turn-0");
  });

  test("shows typed terminal continuation failures rather than treating them as child creation", async () => {
    const unavailable = {
      schema_version: "clozn.time-machine-continuation.v1",
      continuation_id: "tmc_1234567890abcdef1234",
      created_ts: 1,
      finished_ts: 2,
      requested_run_id: "run-1",
      source_turn: 0,
      status: "unavailable",
      request: {
        request_id: "request-1", turn: 0, append_kind: "new_user_turn",
        user_content_sha256: "a".repeat(64), user_content_bytes: 14, max_tokens: 256,
        generation_config_sha256: "b".repeat(64),
      },
      source: {
        status: "resolved", source_run_id: "run-turn-0", source_turn: 0,
        resolution: "exact_organic_session_prefix",
      },
      source_checkpoint: {
        status: "unavailable",
        reasons: [{ code: "checkpoint_expired", message: "the live checkpoint expired" }],
      },
      identity: { status: "unavailable", reasons: [{ code: "checkpoint_expired", message: "the live checkpoint expired" }] },
      append: { status: "unavailable", reasons: [{ code: "checkpoint_expired", message: "the live checkpoint expired" }] },
      sampler: { status: "unavailable", reasons: [{ code: "checkpoint_expired", message: "the live checkpoint expired" }] },
      exactness: { status: "not_confirmed", reasons: [{ code: "checkpoint_expired", message: "the live checkpoint expired" }] },
      worker: { status: "not_run", reasons: [{ code: "checkpoint_expired", message: "the live checkpoint expired" }] },
      child_lineage: { status: "not_created", reasons: [{ code: "checkpoint_expired", message: "the live checkpoint expired" }] },
      unavoidable_differences: [],
      reasons: [{ code: "checkpoint_expired", message: "the live checkpoint expired" }],
      failure: {
        stage: "checkpoint", code: "checkpoint_expired", message: "the live checkpoint expired", retryable: false,
      },
    };
    const durableReceipt = {
      ...receipt,
      turns: receipt.turns.map((turn, index) => index === 0 ? {
        ...turn,
        snapshot: { run_id: "run-turn-0", turn: 0, has_cache: true },
        source: {
          ...turn.source,
          durable_pin: {
            status: "stored",
            reason: { code: "durable_pin_recorded", message: "restart-safe pin is ready" },
            pin: { pin_id: "pin-source", pinned_at: "2026-08-02T00:00:00Z", kv_bytes: 2048, envelope_bytes: 3072 },
          },
        },
      } : turn),
    };
    const fetchMock = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response(JSON.stringify(durableReceipt), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(unavailable), { status: 409 }));
    render(<TimeMachine runId="run-1" />);
    await waitFor(() => expect(screen.getByText(/DURABLE PIN IS RESTART-SAFE/)).toBeInTheDocument());
    fireEvent.change(screen.getByRole("textbox", { name: "New question to append" }), {
      target: { value: "What changed?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "CONTINUE EXACT HISTORY" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(
      "UNAVAILABLE AT CHECKPOINT (checkpoint_expired): the live checkpoint expired",
    ));
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(screen.queryByText(/Exact appended-turn child created/)).not.toBeInTheDocument();
  });
});
