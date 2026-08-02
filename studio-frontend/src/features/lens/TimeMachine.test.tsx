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
      reasons: [{ code: "structural_replay_only", message: "transcript replay" }],
    },
    {
      turn: 1,
      branch_eligible: true,
      replay_fidelity: "structurally_reproducible",
      exact_replay_eligible: false,
      snapshot: null,
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

  test("verifies only the latest turn after an explicit action", async () => {
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
    expect(screen.getByText("run-turn-0")).toBeInTheDocument();
    expect(screen.getByText("run-final")).toBeInTheDocument();
  });
});
