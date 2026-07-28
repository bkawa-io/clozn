import type {
  ObservatoryData,
  SourceReading,
  TokenReading,
  TokenSourceReading,
} from "./types";

export type StressFixtureName = "thread" | "code" | "rag" | "agent";

interface FixtureSpec {
  name: StressFixtureName;
  model: string;
  prompt: string;
  response: string;
  sources: SourceReading[];
  promptTokens: number;
}

function tokenize(text: string): string[] {
  return text.match(/\r\n|\n|\t| +|[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|./gu) ?? [];
}

function measuredIds(sources: SourceReading[]): Set<string> {
  const selected = new Set<string>();
  const policy = sources.find((source) => source.role === "system" || source.role === "developer");
  if (policy) selected.add(policy.id);
  for (const source of [...sources].reverse()) {
    if (selected.size >= 8) break;
    selected.add(source.id);
  }
  return selected;
}

function decorateTokens(text: string, sources: SourceReading[]): TokenReading[] {
  const measured = sources.filter((source) => source.measured);
  return tokenize(text).map((piece, index) => {
    const confidence = Math.max(.18, Math.min(.98, .78 + Math.sin(index * .37) * .17));
    const entropy = Math.max(.05, 1.6 - confidence + Math.abs(Math.cos(index * .19)) * .44);
    const linked: TokenSourceReading[] = [];
    if (piece.trim() && measured.length && index % 3 === 0) {
      const primary = measured[(Math.floor(index / 3) + index) % measured.length];
      linked.push({
        sourceId: primary.id,
        label: primary.label || `${primary.role} context`,
        effect: index % 29 === 0 ? "suppresses" : "supports",
        deltaNats: index % 29 === 0 ? -.08 - index % 7 / 100 : .09 + index % 11 / 100,
        evidenceState: "causally_supported",
      });
      if (index % 13 === 0) {
        const secondary = measured[(measured.indexOf(primary) + 3) % measured.length];
        if (secondary.id !== primary.id) {
          linked.push({
            sourceId: secondary.id,
            label: secondary.label || `${secondary.role} context`,
            effect: "supports",
            deltaNats: .06 + index % 5 / 100,
            evidenceState: "causally_supported",
          });
        }
      }
    }
    return {
      text: piece,
      confidence,
      entropy,
      band: confidence < .54 ? "shaky" : confidence < .76 ? "okay" : "strong",
      source: linked[0]?.label,
      sources: linked,
      alternatives: [
        { token: piece || "∅", score: confidence, delta: 0 },
        { token: piece.trim() ? `${piece.trim()}_alt` : " ", score: Math.max(.01, 1 - confidence), delta: .1 - confidence },
      ],
    };
  });
}

function threadSpec(): FixtureSpec {
  const sources: SourceReading[] = [
    {
      id: "stress.thread.system",
      text: "Answer the arithmetic question directly, then check the computed result against the stated percentage.",
      role: "system",
      kind: "policy",
      label: "answer policy",
      groupId: "system",
      messageIndex: 0,
      measured: true,
    },
    {
      id: "stress.thread.question",
      text: "What is ten percent of 768?",
      role: "user",
      kind: "message",
      label: "user question",
      groupId: "question",
      messageIndex: 1,
      measured: true,
    },
    {
      id: "stress.thread.rule",
      text: "Ten percent is represented by the decimal multiplier 0.1.",
      role: "user",
      kind: "retrieval",
      label: "percentage rule",
      groupId: "reference",
      messageIndex: 2,
      measured: true,
    },
    {
      id: "stress.thread.example",
      text: "Multiplying a quantity by 0.1 moves the decimal point one place to the left.",
      role: "user",
      kind: "retrieval",
      label: "worked example",
      groupId: "reference",
      messageIndex: 3,
      measured: true,
    },
    {
      id: "stress.thread.check",
      text: "The result should be smaller than 768 and greater than zero.",
      role: "user",
      kind: "retrieval",
      label: "range check",
      groupId: "reference",
      messageIndex: 4,
      measured: true,
    },
  ];
  return {
    name: "thread",
    model: "Qwen2.5-3B-Instruct",
    prompt: sources[1].text,
    response: "To find 10% of 768, multiply 768 by 0.1. The result is 76.8, which is smaller than 768 and matches the percentage rule.",
    sources,
    promptTokens: 96,
  };
}

function codeSource(index: number): SourceReading {
  const filename = `src/services/checkout/handler_${String(index).padStart(2, "0")}.ts`;
  const body = Array.from({ length: 16 }, (_, line) =>
    `export async function handler${index}_${line}(request: Request, deps: Dependencies) {\n`
    + `  const record = await deps.store.read(request.params.id);\n`
    + `  return record ? serialize(record, { includeAudit: ${line % 2 === 0} }) : null;\n`
    + `}\n`,
  ).join("\n");
  return {
    id: `stress.code.file.${String(index).padStart(2, "0")}`,
    text: body,
    role: "user",
    kind: "file",
    label: `${filename}:1-${body.split("\n").length}`,
    groupId: filename,
    messageIndex: index + 4,
    start: 0,
    end: body.length,
  };
}

function codeSpec(): FixtureSpec {
  const fixed: SourceReading[] = [
    {
      id: "stress.code.system",
      text: "Follow the repository TypeScript conventions. Preserve public API behavior and return typed errors.",
      role: "system",
      kind: "policy",
      label: "system policy",
      groupId: "system",
      messageIndex: 0,
    },
    {
      id: "stress.code.developer",
      text: "Use AbortSignal for cancellation. Do not introduce dependencies. Update tests with each behavior change.",
      role: "developer",
      kind: "message",
      label: "project instructions",
      groupId: "developer",
      messageIndex: 1,
    },
    {
      id: "stress.code.repo",
      text: Array.from({ length: 80 }, (_, index) =>
        `src/services/checkout/module_${index}.ts: createHandler${index}, validateRequest${index}`,
      ).join("\n"),
      role: "user",
      kind: "repository_map",
      label: "repository map",
      groupId: "repository map",
      messageIndex: 2,
    },
    {
      id: "stress.code.issue",
      text: "Refactor the checkout handlers to share cancellation and error normalization. Return a patch and tests.",
      role: "user",
      kind: "issue",
      label: "issue #1842",
      groupId: "issue",
      messageIndex: 3,
    },
  ];
  const files = Array.from({ length: 22 }, (_, index) => codeSource(index));
  const sources = [...fixed, ...files];
  const selected = measuredIds(sources);
  sources.forEach((source) => { source.measured = selected.has(source.id); });
  const response = [
    "```ts\n",
    "type HandlerResult<T> = { ok: true; value: T } | { ok: false; error: HandlerError };\n\n",
    ...Array.from({ length: 72 }, (_, index) =>
      `export async function checkoutHandler${index}(\n`
      + `  request: CheckoutRequest,\n`
      + `  dependencies: CheckoutDependencies,\n`
      + `  signal: AbortSignal,\n`
      + `): Promise<HandlerResult<CheckoutRecord>> {\n`
      + `  signal.throwIfAborted();\n`
      + `  try {\n`
      + `    const record = await dependencies.store.read(request.id, { signal });\n`
      + `    return record\n`
      + `      ? { ok: true, value: normalizeCheckout(record, ${index}) }\n`
      + `      : { ok: false, error: { code: \"NOT_FOUND\", id: request.id } };\n`
      + `  } catch (error) {\n`
      + `    return { ok: false, error: normalizeHandlerError(error) };\n`
      + `  }\n`
      + `}\n\n`,
    ),
    "```\n",
  ].join("");
  return {
    name: "code",
    model: "Qwen2.5-Coder-3B",
    prompt: fixed[3].text,
    response,
    sources,
    promptTokens: 16_384,
  };
}

function ragSpec(): FixtureSpec {
  const sources: SourceReading[] = [
    {
      id: "stress.rag.system",
      text: "Answer from the supplied documents and preserve conflicting evidence.",
      role: "system",
      kind: "policy",
      label: "answer policy",
      groupId: "system",
      messageIndex: 0,
    },
    {
      id: "stress.rag.question",
      text: "Compare the reported causes of the production outage and identify unresolved disagreements.",
      role: "user",
      kind: "message",
      label: "user question",
      groupId: "question",
      messageIndex: 1,
    },
    ...Array.from({ length: 24 }, (_, index): SourceReading => {
      const text = Array.from({ length: 14 }, (_, paragraph) =>
        `Incident ${index + 1}, section ${paragraph + 1}. Service latency changed after deployment `
        + `${1200 + index}. The report attributes ${paragraph % 2 ? "cache pressure" : "connection churn"} `
        + `to region ${String.fromCharCode(65 + index % 5)}, with confidence ${62 + (index + paragraph) % 31}%.`,
      ).join("\n\n");
      return {
        id: `stress.rag.chunk.${String(index).padStart(2, "0")}`,
        text,
        role: "user",
        kind: "retrieval",
        label: `incident-${2020 + index}.pdf · p.${index * 3 + 1}-${index * 3 + 3}`,
        groupId: `incident-${2020 + index}.pdf`,
        messageIndex: index + 2,
        start: 0,
        end: text.length,
      };
    }),
  ];
  const selected = measuredIds(sources);
  sources.forEach((source) => { source.measured = selected.has(source.id); });
  const response = Array.from({ length: 42 }, (_, index) =>
    `Finding ${index + 1}\n`
    + `The reports agree that the latency increase followed deployment ${1200 + index}, but they `
    + `disagree on whether cache pressure or connection churn was primary. Evidence from regions `
    + `${String.fromCharCode(65 + index % 5)} and ${String.fromCharCode(65 + (index + 2) % 5)} remains unresolved.\n\n`,
  ).join("");
  return {
    name: "rag",
    model: "Gemma-3-4B",
    prompt: sources[1].text,
    response,
    sources,
    promptTokens: 24_576,
  };
}

function agentSpec(): FixtureSpec {
  const sources = Array.from({ length: 34 }, (_, index): SourceReading => {
    const role = index === 0 ? "system" : index === 1 ? "user" : index % 2 ? "assistant" : "tool";
    const kind = role === "tool" ? "tool_result" : role === "assistant" ? "tool_call" : "message";
    const text = role === "tool"
      ? JSON.stringify({
          command: `inspect_resource_${index}`,
          status: index % 7 === 0 ? "error" : "ok",
          rows: Array.from({ length: 18 }, (_, row) => ({ id: row, value: index * 100 + row })),
        }, null, 2)
      : role === "assistant"
        ? `Call inspect_resource_${index + 1} with the current run identifier and retain the result.`
        : index === 0
          ? "Use only declared tools. Report failed calls separately from successful results."
          : "Inspect every resource, reconcile the returned records, and produce a compact JSON report.";
    return {
      id: `stress.agent.event.${String(index).padStart(2, "0")}`,
      text,
      role,
      kind,
      label: role === "tool" ? `inspect_resource_${index}` : `${role} turn ${index}`,
      groupId: `event-${index}`,
      messageIndex: index,
      start: 0,
      end: text.length,
    };
  });
  const selected = measuredIds(sources);
  sources.forEach((source) => { source.measured = selected.has(source.id); });
  const response = JSON.stringify({
    status: "partial",
    inspected: Array.from({ length: 28 }, (_, index) => ({
      resource: `resource_${index}`,
      records: 18,
      state: index % 7 === 0 ? "failed" : "verified",
      evidence: [`tool_${index * 2}`, `tool_${index * 2 + 1}`],
    })),
    failed: [0, 7, 14, 21],
  }, null, 2);
  return {
    name: "agent",
    model: "Phi-4-mini-instruct",
    prompt: sources[1].text,
    response,
    sources,
    promptTokens: 12_288,
  };
}

function buildFixture(spec: FixtureSpec): ObservatoryData {
  const measured = spec.sources.filter((source) => source.measured);
  const tokens = decorateTokens(spec.response, measured);
  return {
    id: `stress_${spec.name}`,
    label: `${spec.name} stress fixture`,
    model: spec.model,
    quant: "Q4_K_M",
    createdAt: "FIXTURE",
    duration: "—",
    mode: "demo",
    prompt: spec.prompt,
    response: spec.response,
    flags: ["stress-fixture"],
    tokens,
    candidates: tokens[0]?.alternatives ?? [],
    sources: measured,
    contextSources: spec.sources,
    contextCoverage: {
      totalSources: spec.sources.length,
      measuredSources: measured.length,
      omittedSources: spec.sources.length - measured.length,
      measuredSpans: measured.length,
      complete: measured.length === spec.sources.length,
      strategy: "earliest_policy_then_recent_sources_proportional_chunks_v1",
      promptTokens: spec.promptTokens,
    },
    configuration: {
      activeDials: {},
      memoryCards: [],
      adapters: [],
      changes: [],
    },
  };
}

export function stressFixture(name: string | undefined): ObservatoryData | undefined {
  if (name === "thread") return buildFixture(threadSpec());
  if (name === "code") return buildFixture(codeSpec());
  if (name === "rag") return buildFixture(ragSpec());
  if (name === "agent") return buildFixture(agentSpec());
  return undefined;
}
