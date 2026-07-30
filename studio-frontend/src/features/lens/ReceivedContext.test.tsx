import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { createFetchController, type PendingFetch } from "../../test/fetch";
import { render, screen, waitFor, within } from "../../test/render";
import { ReceivedContext } from "./ReceivedContext";

function investigation(
  runId: string,
  overrides: Record<string, unknown> = {},
) {
  return {
    schema_version: "clozn.run-investigation.v1",
    run_id: runId,
    sections: {
      received_context: {
        state: "delivered_not_measured",
        privacy: "full",
        delivered: [
          {
            segment_id: "seg-system",
            source_type: "message",
            source_label: "system",
            original_order: 0,
            delivered_bytes: 512,
            delivered_tokens: 12,
            included: true,
            redaction_state: "full",
          },
          {
            segment_id: "seg-user",
            source_type: "message",
            source_label: "user",
            original_order: 1,
            delivered_bytes: 512,
            included: false,
            reason: "context_budget",
            redaction_state: "full",
          },
        ],
        assembled: [
          {
            segment_id: "seg-system",
            source_type: "message",
            source_label: "system",
            original_order: 0,
            included: true,
          },
        ],
        omitted: [
          {
            segment_id: "seg-user",
            source_type: "message",
            source_label: "user",
            original_order: 1,
            reason: "context_budget",
          },
        ],
        rendered: {
          sha256: "a".repeat(64),
          bytes: 1024,
          tokens: 42,
          content_available: true,
        },
        limits: {
          prompt_tokens: 42,
          context_window_tokens: 8192,
        },
        ...overrides,
      },
      text_span_addresses: {
        state: "supported",
        privacy: "metadata_only",
        href: `/runs/${runId}/span-addresses`,
        address_count: 3,
      },
    },
    actions: [],
    unavailable_measurements: [],
  };
}

function address(
  addressId: string,
  id: string,
  options: {
    kind?: string;
    state?: string;
    reason?: string;
    collection?: string;
  } = {},
) {
  return {
    address_id: addressId,
    run_id: "fixture",
    kind: options.kind ?? "delivered_message",
    relation_key: `rel_${addressId.slice(-24)}`,
    native_ref: {
      artifact_schema: "clozn.context-receipt.v1",
      collection: options.collection ?? "context_receipt.delivered",
      id,
      segment_id: id,
    },
    resolution: {
      state: options.state ?? "metadata_only",
      ...(options.reason ? { reason: options.reason } : {}),
      ...(
        options.state === "redacted" || options.state === "unavailable"
          ? {}
          : {
              canonical: {
                basis: options.kind === "rendered_prompt_segment"
                  ? "rendered_prompt"
                  : "delivered_message",
                unit: "unicode_code_points",
                interval: "half_open",
                start: 0,
                end: 5,
                basis_sha256: "b".repeat(64),
                span_sha256: "c".repeat(64),
              },
            }
      ),
    },
  };
}

function spanDocument(
  runId: string,
  addresses = [
    address("span_111111111111111111111111", "seg-system"),
    address("span_222222222222222222222222", "seg-user"),
    address("span_333333333333333333333333", "rendered-prompt", {
      kind: "rendered_prompt_segment",
      collection: "context_receipt.rendered",
    }),
  ],
  sourceOverrides: Record<string, unknown> = {},
) {
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
    source_artifacts: [{
      schema: "clozn.context-receipt.v1",
      privacy: "full",
      ...sourceOverrides,
    }],
    addresses,
    lineage: { parent_run_id: null, mappings: [] },
  };
}

function pathOf(request: PendingFetch) {
  return typeof request.input === "string"
    ? request.input
    : request.input instanceof URL
      ? request.input.pathname
      : request.input.url;
}

function requestFor(requests: PendingFetch[], suffix: string) {
  const request = requests.find((item) => pathOf(item).endsWith(suffix));
  if (!request) throw new Error(`missing request ending ${suffix}`);
  return request;
}

async function initialRequests(controller: ReturnType<typeof createFetchController>) {
  await waitFor(() => expect(controller.requests).toHaveLength(2));
  return {
    investigation: requestFor(controller.requests, "/investigation"),
    spans: requestFor(controller.requests, "/span-addresses"),
  };
}

describe("What did the model receive panel", () => {
  test("renders delivery states, available costs, and stable span links before authorized content", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const user = userEvent.setup();

    render(<ReceivedContext runId="run-one" />);
    const requests = await initialRequests(controller);
    controller.respondJson(requests.investigation, investigation("run-one"));
    controller.respondJson(requests.spans, spanDocument("run-one"));

    expect(await screen.findByRole("heading", {
      name: "What did the model receive?",
    })).toBeInTheDocument();
    expect(await screen.findByText("42")).toBeInTheDocument();
    expect(screen.getByText("1,024 B")).toBeInTheDocument();
    expect(screen.getByText("8,192")).toBeInTheDocument();

    const delivered = screen.getByRole("heading", {
      name: "Request as delivered",
    }).closest("section");
    const assembled = screen.getByRole("heading", {
      name: "Context as assembled",
    }).closest("section");
    const omitted = screen.getByRole("heading", {
      name: "Omitted before generation",
    }).closest("section");
    expect(delivered).not.toBeNull();
    expect(assembled).not.toBeNull();
    expect(omitted).not.toBeNull();
    expect(within(delivered!).getAllByText("DELIVERED")).toHaveLength(1);
    expect(within(delivered!).getAllByText("OMITTED")).toHaveLength(1);
    expect(within(assembled!).getByText("ASSEMBLED")).toBeInTheDocument();
    expect(within(omitted!).getByText("OMITTED")).toBeInTheDocument();
    expect(within(delivered!).getByText("12 TOK")).toBeInTheDocument();

    const systemSpan = screen.getAllByRole("link", {
      name: "Stable span span_111111111111111111111111",
    })[0];
    expect(systemSpan).toHaveAttribute(
      "href",
      "/runs/run-one/span-addresses#span_111111111111111111111111",
    );
    expect(controller.requests.some((request) =>
      pathOf(request).endsWith("/context-receipt"))).toBe(false);

    await user.click(screen.getByRole("button", {
      name: "OPEN AUTHORIZED CONTEXT RECEIPT",
    }));
    await waitFor(() => expect(controller.requests).toHaveLength(3));
    const authorized = requestFor(controller.requests, "/context-receipt");
    expect(pathOf(authorized)).toBe("/runs/run-one/context-receipt");
    controller.respondJson(authorized, {
      run_id: "run-one",
      shape: "absent",
      context_receipt: {},
    });
    expect(await screen.findByText(
      "NO CONTEXT RECEIPT WAS RECORDED FOR THIS RUN",
    )).toBeInTheDocument();
  });

  test("keeps redacted, unavailable, omitted, and failed presentations distinct", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const privateView = render(<ReceivedContext runId="run-private" />);
    const requests = await initialRequests(controller);
    controller.respondJson(requests.investigation, investigation("run-private", {
      privacy: "hashes_only",
      delivered: [
        {
          segment_id: "seg-redacted",
          source_type: "message",
          source_label: "system",
          original_order: 0,
          included: true,
          redaction_state: "hash_only",
        },
        {
          segment_id: "seg-unavailable",
          source_type: "message",
          source_label: "user",
          original_order: 1,
          included: true,
        },
      ],
      assembled: [],
      omitted: [{
        segment_id: "seg-omitted",
        source_type: "message",
        source_label: "tool",
        original_order: 2,
        reason: "tool_result_pruned",
      }],
    }));
    controller.respondJson(requests.spans, spanDocument("run-private", [
      address("span_444444444444444444444444", "seg-redacted", {
        state: "redacted",
        reason: "source_text_redacted",
      }),
      address("span_555555555555555555555555", "seg-unavailable", {
        state: "unavailable",
        reason: "canonical_basis_unavailable",
      }),
    ], {
      native_status: "redacted",
      privacy: "redacted",
      reason: "run text was removed by the persisted redaction lifecycle",
    }));

    expect(await screen.findByText(
      "run text was removed by the persisted redaction lifecycle",
    )).toBeInTheDocument();
    expect(screen.getAllByText("REDACTED").length).toBeGreaterThan(1);
    expect(screen.getAllByText("UNAVAILABLE").length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", {
      name: "Omitted before generation",
    })).toBeInTheDocument();
    expect(document.querySelector(".received-context-notice.is-redacted")).not.toBeNull();
    expect(document.querySelector(".received-context-state.is-unavailable")).not.toBeNull();
    privateView.unmount();

    const failedController = createFetchController();
    vi.stubGlobal("fetch", failedController.fetch);
    const view = render(<ReceivedContext runId="run-failed" />);
    const failedRequests = await initialRequests(failedController);
    failedController.respondJson(
      failedRequests.investigation,
      investigation("run-failed"),
    );
    failedController.respondJson(
      failedRequests.spans,
      { error: "span projection failed" },
      { status: 500 },
    );

    expect(await screen.findByText("STABLE SPAN REQUEST FAILED")).toBeInTheDocument();
    const failedNotice = screen.getByText("STABLE SPAN REQUEST FAILED").closest("[role='alert']");
    expect(failedNotice).toHaveClass("is-failed");
    expect(screen.getAllByText("ID UNAVAILABLE").length).toBeGreaterThan(0);
    view.unmount();
  });

  test("renders an unavailable investigation differently from a request failure", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const unavailableView = render(<ReceivedContext runId="run-unavailable" />);
    const requests = await initialRequests(controller);
    controller.respondJson(requests.investigation, investigation("run-unavailable", {
      state: "unavailable",
      reason: "no context receipt was recorded for this run",
      privacy: "off",
      delivered: [],
      assembled: [],
      omitted: [],
      rendered: {},
      limits: {},
    }));
    controller.respondJson(requests.spans, spanDocument("run-unavailable", []));

    const unavailable = await screen.findByText("no context receipt was recorded for this run");
    expect(unavailable.closest("[role='note']")).toHaveClass("is-unavailable");
    expect(screen.queryByText("INVESTIGATION REQUEST FAILED")).not.toBeInTheDocument();
    unavailableView.unmount();

    const failedController = createFetchController();
    vi.stubGlobal("fetch", failedController.fetch);
    render(<ReceivedContext runId="run-request-failed" />);
    const failedRequests = await initialRequests(failedController);
    failedController.respondJson(
      failedRequests.investigation,
      { error: "failed" },
      { status: 500 },
    );
    failedController.respondJson(failedRequests.spans, spanDocument("run-request-failed", []));
    expect(await screen.findByText("INVESTIGATION REQUEST FAILED")).toBeInTheDocument();
    expect(screen.getByText("INVESTIGATION REQUEST FAILED").closest("[role='alert']"))
      .toHaveClass("is-failed");
  });

  test("aborts stale requests and cannot render an older run after selection changes", async () => {
    const controller = createFetchController();
    vi.stubGlobal("fetch", controller.fetch);
    const view = render(<ReceivedContext runId="run-old" />);
    await waitFor(() => expect(controller.requests).toHaveLength(2));
    const oldInvestigation = requestFor(controller.requests, "/run-old/investigation");
    const oldSpans = requestFor(controller.requests, "/run-old/span-addresses");

    view.rerender(<ReceivedContext runId="run-new" />);
    await waitFor(() => expect(controller.requests).toHaveLength(4));
    expect(oldInvestigation.signal?.aborted).toBe(true);
    expect(oldSpans.signal?.aborted).toBe(true);
    const newInvestigation = requestFor(controller.requests, "/run-new/investigation");
    const newSpans = requestFor(controller.requests, "/run-new/span-addresses");
    controller.respondJson(newInvestigation, investigation("run-new", {
      delivered: [{
        segment_id: "seg-new",
        source_label: "NEW RUN SOURCE",
        source_type: "message",
        original_order: 0,
        included: true,
      }],
      assembled: [],
      omitted: [],
    }));
    controller.respondJson(newSpans, spanDocument("run-new", [
      address("span_666666666666666666666666", "seg-new"),
    ]));

    expect(await screen.findByText("NEW RUN SOURCE")).toBeInTheDocument();
    controller.respondJson(oldInvestigation, investigation("run-old", {
      delivered: [{
        segment_id: "seg-old",
        source_label: "OLD RUN SOURCE",
        source_type: "message",
        original_order: 0,
        included: true,
      }],
    }));
    controller.respondJson(oldSpans, spanDocument("run-old"));
    await waitFor(() => expect(screen.queryByText("OLD RUN SOURCE")).not.toBeInTheDocument());
    expect(screen.getByText("NEW RUN SOURCE")).toBeInTheDocument();
  });
});
