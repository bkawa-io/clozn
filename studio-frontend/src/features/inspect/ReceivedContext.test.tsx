import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen, waitFor, within } from "@testing-library/react";
import type { InspectRoute } from "../../app/router";
import { studioApi } from "../../data/client";
import { ContractError } from "../../data/contracts";
import type {
  ReceivedContextSection,
  RunInvestigation,
  SpanAddress,
  SpanAddressDocument,
  SpanResolutionState,
} from "../../data/receivedContext";
import { ReceivedContextSurface } from "./ReceivedContext";

const runId = "run_received";

function route(selectedRunId: string): InspectRoute {
  return { surface: "inspect", runId: selectedRunId, view: "context" };
}

function investigationFixture(
  selectedRunId: string,
  overrides: Partial<ReceivedContextSection> = {},
): RunInvestigation {
  const receivedContext: ReceivedContextSection = {
    state: "delivered_not_measured",
    privacy: "full",
    delivered: [
      {
        segmentId: "seg-system",
        sourceType: "message",
        sourceLabel: "system",
        originalOrder: 0,
        deliveredBytes: 512,
        deliveredTokens: 12,
        included: true,
        redactionState: "full",
      },
      {
        segmentId: "seg-user",
        sourceType: "message",
        sourceLabel: "user",
        originalOrder: 1,
        deliveredBytes: 512,
        included: false,
        reason: "context_budget",
        redactionState: "full",
      },
    ],
    assembled: [
      {
        segmentId: "seg-system",
        sourceType: "message",
        sourceLabel: "system",
        originalOrder: 0,
        included: true,
      },
    ],
    omitted: [
      {
        segmentId: "seg-user",
        sourceType: "message",
        sourceLabel: "user",
        originalOrder: 1,
        reason: "context_budget",
      },
    ],
    rendered: {
      sha256: "a".repeat(64),
      bytes: 1024,
      tokens: 42,
    },
    contextWindowTokens: 8192,
    ...overrides,
  };
  return {
    schemaVersion: "clozn.run-investigation.v1",
    runId: selectedRunId,
    receivedContext,
    spanSection: {
      state: "supported",
      privacy: "metadata_only",
      href: `/runs/${selectedRunId}/span-addresses`,
      addressCount: 3,
    },
  };
}

function spanAddress(
  selectedRunId: string,
  addressId: string,
  segmentId: string,
  options: {
    kind?: SpanAddress["kind"];
    collection?: SpanAddress["nativeRef"]["collection"];
    state?: SpanResolutionState;
    reason?: SpanAddress["resolution"]["reason"];
  } = {},
): SpanAddress {
  const state = options.state ?? "metadata_only";
  return {
    addressId,
    runId: selectedRunId,
    kind: options.kind ?? "delivered_message",
    relationKey: `rel_${addressId.slice(-24)}`,
    nativeRef: {
      artifactSchema: "clozn.context-receipt.v1",
      collection: options.collection ?? "context_receipt.delivered",
      id: segmentId,
      segmentId,
    },
    resolution: {
      state,
      ...(options.reason ? { reason: options.reason } : {}),
      ...(state === "metadata_only" || state === "drifted"
        ? {
            canonical: {
              basis: options.kind === "rendered_prompt_segment"
                ? "rendered_prompt" as const
                : "delivered_message" as const,
              start: 0,
              end: 5,
              basisSha256: "b".repeat(64),
              spanSha256: "c".repeat(64),
            },
          }
        : {}),
    },
  };
}

function spanDocument(
  selectedRunId: string,
  addresses: readonly SpanAddress[] = [
    spanAddress(selectedRunId, "span_111111111111111111111111", "seg-system"),
    spanAddress(selectedRunId, "span_222222222222222222222222", "seg-user"),
    spanAddress(selectedRunId, "span_333333333333333333333333", "rendered-prompt", {
      kind: "rendered_prompt_segment",
      collection: "context_receipt.rendered",
    }),
  ],
  sourceOverrides: Partial<SpanAddressDocument["sourceArtifacts"][number]> = {},
): SpanAddressDocument {
  return {
    schemaVersion: "clozn.text-span-addresses.v1",
    runId: selectedRunId,
    privacy: "metadata_only",
    sourceArtifacts: [{
      schema: "clozn.context-receipt.v1",
      privacy: "full",
      ...sourceOverrides,
    }],
    addresses,
    parentRunId: null,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("Received Context Inspect instrument", () => {
  it("renders the existing metadata-only delivery view without disclosing prompt content", async () => {
    const investigationRead = vi.spyOn(studioApi, "runInvestigation")
      .mockResolvedValue(investigationFixture(runId));
    const spanRead = vi.spyOn(studioApi, "spanAddresses")
      .mockResolvedValue(spanDocument(runId));
    const receiptRead = vi.spyOn(studioApi, "contextReceipt");

    render(<ReceivedContextSurface route={route(runId)} />);

    expect(await screen.findByRole("heading", { name: /received context/i })).toBeInTheDocument();
    expect(screen.getByText(/GET only/i)).toBeInTheDocument();
    expect(screen.getByText(/delivery.*prompt assembly/i)).toBeInTheDocument();
    expect(screen.getByText(/does not prove.*changed the answer/i)).toBeInTheDocument();
    expect(screen.getByText("Metadata only")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByText("1,024 B")).toBeInTheDocument();
    expect(screen.getByText("8,192")).toBeInTheDocument();

    const delivered = screen.getByRole("heading", { name: "Request as delivered" }).closest("section");
    const assembled = screen.getByRole("heading", { name: "Context as assembled" }).closest("section");
    const omitted = screen.getByRole("heading", { name: "Omitted before generation" }).closest("section");
    expect(delivered).not.toBeNull();
    expect(assembled).not.toBeNull();
    expect(omitted).not.toBeNull();
    const deliveredSystemRow = within(delivered!).getByText("system").closest("li");
    expect(deliveredSystemRow).not.toBeNull();
    const tokenLabel = within(deliveredSystemRow!).getByText("Tokens");
    expect(tokenLabel.parentElement).toHaveTextContent("12");
    expect(within(delivered!).getByText("Delivered")).toBeInTheDocument();
    expect(within(delivered!).getByText("Omitted")).toBeInTheDocument();
    expect(within(assembled!).getByText("Assembled")).toBeInTheDocument();
    expect(within(omitted!).getByText(/context budget/i)).toBeInTheDocument();

    const stableSpan = within(delivered!).getByRole("link", {
      name: "Stable span span_111111111111111111111111",
    });
    expect(stableSpan).toHaveAttribute(
      "href",
      "/runs/run_received/span-addresses#span_111111111111111111111111",
    );

    expect(investigationRead).toHaveBeenCalledTimes(1);
    expect(investigationRead.mock.calls[0]?.[0]).toBe(runId);
    expect(investigationRead.mock.calls[0]?.[1]).toBeInstanceOf(AbortSignal);
    expect(spanRead).toHaveBeenCalledTimes(1);
    expect(spanRead.mock.calls[0]?.[0]).toBe(runId);
    expect(spanRead.mock.calls[0]?.[1]).toBeInstanceOf(AbortSignal);
    expect(receiptRead).not.toHaveBeenCalled();
  });

  it("keeps redacted, unavailable, and omitted evidence visibly distinct with reasons", async () => {
    vi.spyOn(studioApi, "runInvestigation").mockResolvedValue(investigationFixture(runId, {
      privacy: "hashes_only",
      delivered: [
        {
          segmentId: "seg-redacted",
          sourceType: "message",
          sourceLabel: "system",
          originalOrder: 0,
          included: true,
          redactionState: "hash_only",
        },
        {
          segmentId: "seg-unavailable",
          sourceType: "message",
          sourceLabel: "user",
          originalOrder: 1,
          included: true,
        },
      ],
      assembled: [],
      omitted: [{
        segmentId: "seg-omitted",
        sourceType: "message",
        sourceLabel: "tool",
        originalOrder: 2,
        reason: "tool_result_pruned",
      }],
      rendered: {},
    }));
    vi.spyOn(studioApi, "spanAddresses").mockResolvedValue(spanDocument(runId, [
      spanAddress(runId, "span_444444444444444444444444", "seg-redacted", {
        state: "redacted",
        reason: "source_text_redacted",
      }),
      spanAddress(runId, "span_555555555555555555555555", "seg-unavailable", {
        state: "unavailable",
        reason: "canonical_basis_unavailable",
      }),
    ], {
      nativeStatus: "redacted",
      privacy: "redacted",
      reason: "run text was removed by the persisted redaction lifecycle",
    }));

    render(<ReceivedContextSurface route={route(runId)} />);

    expect(await screen.findByText(
      "run text was removed by the persisted redaction lifecycle",
    )).toBeInTheDocument();
    expect(screen.getByText(/stored receipt.*hashes only/i)).toBeInTheDocument();

    const redactedRow = screen.getByText("system").closest("li");
    const unavailableRow = screen.getByText("user").closest("li");
    const omittedRow = screen.getByText("tool").closest("li");
    expect(redactedRow).not.toBeNull();
    expect(unavailableRow).not.toBeNull();
    expect(omittedRow).not.toBeNull();
    expect(within(redactedRow!).getByText("Redacted")).toBeInTheDocument();
    expect(within(redactedRow!).getByText(/source text redacted/i)).toBeInTheDocument();
    expect(within(unavailableRow!).getByText("Unavailable")).toBeInTheDocument();
    expect(within(unavailableRow!).getByText(/canonical basis unavailable/i)).toBeInTheDocument();
    expect(within(omittedRow!).getByText("Omitted")).toBeInTheDocument();
    expect(within(omittedRow!).getByText(/tool result pruned/i)).toBeInTheDocument();
  });

  it("preserves delivery evidence when the independent stable-span read fails", async () => {
    vi.spyOn(studioApi, "runInvestigation").mockResolvedValue(investigationFixture(runId));
    vi.spyOn(studioApi, "spanAddresses").mockRejectedValue(new Error("span service offline"));
    const receiptRead = vi.spyOn(studioApi, "contextReceipt");

    render(<ReceivedContextSurface route={route(runId)} />);

    const delivered = (await screen.findByRole("heading", {
      name: "Request as delivered",
    })).closest("section");
    expect(delivered).not.toBeNull();
    expect(within(delivered!).getByText("system")).toBeInTheDocument();
    expect(within(delivered!).getByText("Delivered")).toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(/stable span.*unavailable/i);
    expect(screen.getByRole("alert")).toHaveTextContent("span service offline");
    expect(within(delivered!).getAllByText(/(?:address|id) unavailable/i).length).toBeGreaterThan(0);
    expect(receiptRead).not.toHaveBeenCalled();
  });

  it("distinguishes recorded unavailability, malformed evidence, and transport failure", async () => {
    const investigationRead = vi.spyOn(studioApi, "runInvestigation")
      .mockResolvedValueOnce(investigationFixture("run-unavailable", {
        state: "unavailable",
        reason: "no context receipt was recorded for this run",
        privacy: "off",
        delivered: [],
        assembled: [],
        omitted: [],
        rendered: {},
      }))
      .mockRejectedValueOnce(new ContractError(
        "/runs/run-malformed/investigation",
        "$.sections.received_context.state is unknown",
      ))
      .mockRejectedValueOnce(new Error("gateway offline"));
    vi.spyOn(studioApi, "spanAddresses").mockImplementation((selectedRunId) => (
      Promise.resolve(spanDocument(selectedRunId, []))
    ));

    const view = render(<ReceivedContextSurface route={route("run-unavailable")} />);
    const unavailable = (await screen.findByText("no context receipt was recorded for this run")).closest("[role='note']");
    expect(unavailable).not.toBeNull();
    expect(unavailable).toHaveTextContent("Unavailable");
    expect(unavailable).toHaveTextContent("no context receipt was recorded for this run");

    view.rerender(<ReceivedContextSurface route={route("run-malformed")} />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Malformed investigation contract."));
    expect(screen.getByRole("alert")).toHaveTextContent("state is unknown");

    view.rerender(<ReceivedContextSurface route={route("run-offline")} />);
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("Investigation request unavailable."));
    expect(screen.getByRole("alert")).toHaveTextContent("gateway offline");
    expect(investigationRead).toHaveBeenCalledTimes(3);
  });

  it("does not invent delivery rows or costs from independent span references", async () => {
    vi.spyOn(studioApi, "runInvestigation").mockResolvedValue(investigationFixture(runId, {
      deliveredCount: 1,
      delivered: [],
      assembled: [],
      omitted: [],
      rendered: {},
    }));
    vi.spyOn(studioApi, "spanAddresses").mockResolvedValue(spanDocument(runId, [
      {
        ...spanAddress(runId, "span_777777777777777777777777", "message-0", {
          collection: "run.messages",
        }),
        nativeRef: {
          artifactSchema: "clozn.run.v1",
          collection: "run.messages",
          id: "message-0",
          sourceLabel: "Recorded system message",
        },
      },
    ]));

    render(<ReceivedContextSurface route={route(runId)} />);

    await screen.findByRole("heading", { name: "Request as delivered" });
    expect(screen.queryByText("Recorded system message")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Prompt composition not measured" })).toBeInTheDocument();
    expect(screen.getByText(/No delivered or omitted segment counts were recorded/i)).toBeInTheDocument();
    expect(screen.queryByText("0 B")).not.toBeInTheDocument();
  });

  it("aborts both stale metadata reads and ignores their late results after a Run switch", async () => {
    const oldInvestigation = deferred<RunInvestigation>();
    const oldSpans = deferred<SpanAddressDocument>();
    const newInvestigation = deferred<RunInvestigation>();
    const newSpans = deferred<SpanAddressDocument>();
    const investigationRead = vi.spyOn(studioApi, "runInvestigation")
      .mockImplementation((selectedRunId) => (
        selectedRunId === "run-old" ? oldInvestigation.promise : newInvestigation.promise
      ));
    const spanRead = vi.spyOn(studioApi, "spanAddresses")
      .mockImplementation((selectedRunId) => (
        selectedRunId === "run-old" ? oldSpans.promise : newSpans.promise
      ));

    const view = render(<ReceivedContextSurface route={route("run-old")} />);
    await waitFor(() => expect(investigationRead).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(spanRead).toHaveBeenCalledTimes(1));

    view.rerender(<ReceivedContextSurface route={route("run-new")} />);
    await waitFor(() => expect(investigationRead).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(spanRead).toHaveBeenCalledTimes(2));
    expect(investigationRead.mock.calls[0]?.[1]?.aborted).toBe(true);
    expect(spanRead.mock.calls[0]?.[1]?.aborted).toBe(true);

    await act(async () => {
      newInvestigation.resolve(investigationFixture("run-new", {
        delivered: [{
          segmentId: "seg-new",
          sourceType: "message",
          sourceLabel: "NEW RUN SOURCE",
          originalOrder: 0,
          included: true,
        }],
        assembled: [],
        omitted: [],
      }));
      newSpans.resolve(spanDocument("run-new", [
        spanAddress("run-new", "span_666666666666666666666666", "seg-new"),
      ]));
    });
    expect(await screen.findByText("NEW RUN SOURCE")).toBeInTheDocument();

    await act(async () => {
      oldInvestigation.resolve(investigationFixture("run-old", {
        delivered: [{
          segmentId: "seg-old",
          sourceType: "message",
          sourceLabel: "OLD RUN SOURCE",
          originalOrder: 0,
          included: true,
        }],
      }));
      oldSpans.resolve(spanDocument("run-old"));
    });
    expect(screen.queryByText("OLD RUN SOURCE")).not.toBeInTheDocument();
    expect(screen.getByText("NEW RUN SOURCE")).toBeInTheDocument();
  });
});
