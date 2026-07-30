import { describe, expect, test } from "vitest";
import { render, screen } from "../../test/render";
import { ForkOutcomePanel } from "./ForkOutcomePanel";

describe("ForkOutcomePanel", () => {
  test("exact_execution_fork reads as the strong result and shows its exactness facts", () => {
    render(
      <ForkOutcomePanel
        note={"exact execution fork: the worker restored its exact recorded KV state and applied the "
          + "forced token there directly on its token id -- no text splice, nothing to retokenize"}
        outcome={{
          kind: "exact_execution_fork",
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
        }}
      />,
    );

    expect(screen.getByText("EXACT EXECUTION FORK")).toBeInTheDocument();
    expect(screen.getByText(/no text splice, nothing to retokenize/i, {
      selector: ".fork-outcome-summary",
    })).toBeInTheDocument();
    expect(screen.getByText("GENERATED TOKEN LIVE KV")).toBeInTheDocument();
    expect(screen.getByText("LIVE KV TRUNCATED")).toBeInTheDocument();
    expect(screen.getByText('FORCE TOKEN → "alternate" (id 4242)')).toBeInTheDocument();
    expect(screen.getByText("MATCHED · EXACT MATCH")).toBeInTheDocument();
    expect(screen.getByText("CONFIRMED")).toBeInTheDocument();
  });

  test("reconstructed_replay reads as visibly weaker and names the retokenization risk", () => {
    render(
      <ForkOutcomePanel
        note="greedy continuation (sample=false): a deterministic what-if"
        outcome={{
          kind: "reconstructed_replay",
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
        }}
      />,
    );

    expect(screen.getByText("RECONSTRUCTED REPLAY")).toBeInTheDocument();
    expect(screen.getByText("RETOKENIZED")).toBeInTheDocument();
    expect(screen.getByText(/BPE token boundaries can shift/i)).toBeInTheDocument();
    expect(screen.getByText(/NOT guaranteed to run on the exact recorded token ids/)).toBeInTheDocument();
    expect(screen.getByText("KV STATE NOT RESTORED")).toBeInTheDocument();
    expect(screen.getByText("SAMPLER STATE REINITIALIZED")).toBeInTheDocument();
    // Never styled as though it were the strong outcome: no exactness metric list, no exact badge text.
    expect(screen.queryByText("EXACT EXECUTION FORK")).not.toBeInTheDocument();
  });

  test("unavailable shows the gateway's typed reason instead of a generic failure", () => {
    render(
      <ForkOutcomePanel
        outcome={{
          kind: "unavailable",
          reasons: [{
            code: "checkpoint_expired",
            message: "the referenced checkpoint has expired or been evicted",
          }],
        }}
      />,
    );

    expect(screen.getByText("FORK UNAVAILABLE")).toBeInTheDocument();
    expect(screen.getByText("CHECKPOINT EXPIRED")).toBeInTheDocument();
    expect(screen.getByText("the referenced checkpoint has expired or been evicted")).toBeInTheDocument();
    expect(screen.queryByText(/fork failed/i)).not.toBeInTheDocument();
  });
});
