import type {
  ContextTension,
  InfluenceQuery,
  RunMessage,
  RunRecord,
  SpanAddress,
  SpanAddressDocument,
  SuggestedBreakpoints,
} from "../../data/contracts";
import type {
  ContextDocument,
  DecisionLocus,
  InfluenceSelection,
  LinkedReaderSpecimen,
  RelatedContextLocus,
  TextLocus,
} from "./model";

interface ReaderProjection {
  specimen: LinkedReaderSpecimen;
  addresses: SpanAddressDocument;
}

function codePointLength(value: string): number {
  return Array.from(value).length;
}

/** Convert the server's Unicode code-point coordinate into the browser's UTF-16 index. */
export function codePointOffset(value: string, offset: number): number | undefined {
  if (!Number.isInteger(offset) || offset < 0) return undefined;
  const points = Array.from(value);
  if (offset > points.length) return undefined;
  return points.slice(0, offset).join("").length;
}

function textLocus(address: SpanAddress, text: string): TextLocus | undefined {
  const canonical = address.resolution.canonical;
  if (!canonical || !["metadata_only", "exact"].includes(address.resolution.state)) return undefined;
  const start = codePointOffset(text, canonical.start);
  const end = codePointOffset(text, canonical.end);
  if (start === undefined || end === undefined || end <= start) return undefined;
  return { id: address.addressId, start, end, label: address.nativeRef.sourceLabel };
}

function phraseLoci(text: string, atomic: readonly TextLocus[]): TextLocus[] {
  const ordered = [...atomic].sort((left, right) => left.start - right.start || left.end - right.end);
  const out: TextLocus[] = [];
  let group: TextLocus[] = [];
  const flush = () => {
    if (!group.length) return;
    const first = group[0];
    const last = group.at(-1)!;
    out.push({ id: group.length === 1 ? first.id : `range:${first.id}:${last.id}`, start: first.start, end: last.end, memberIds: group.map((locus) => locus.id) });
    group = [];
  };
  for (const locus of ordered) {
    const previous = group.at(-1);
    if (previous && group.length >= 6 && /^\s/.test(text.slice(locus.start, locus.end))) flush();
    group.push(locus);
    if (/[.!?](?:["')\]]*)$/.test(text.slice(group[0].start, locus.end).trim())) flush();
  }
  flush();
  return out;
}

function contextMessages(run: RunRecord): readonly RunMessage[] {
  return run.assembledMessages ?? run.messages ?? [];
}

function documentId(message: RunMessage, index: number): string {
  return message.sourceId ? `source:${message.sourceId}:${index}` : `message:${index}`;
}

function documents(run: RunRecord): ContextDocument[] {
  const messages = contextMessages(run);
  if (!messages.length) {
    return [{
      id: "context-unavailable",
      label: "Recorded context",
      state: "unavailable",
      detail: "This run did not retain readable context messages.",
    }];
  }
  return messages.map((message, index) => ({
    id: documentId(message, index),
    label: message.sourceLabel?.trim() || message.role.toUpperCase(),
    text: message.content,
    state: "available",
  }));
}

export function projectLinkedReader(run: RunRecord, addresses: SpanAddressDocument): ReaderProjection {
  const answer = run.response ?? "";
  const atomicAnswerLoci = addresses.addresses
    .filter((address) => address.kind === "answer_span")
    .map((address) => textLocus(address, answer))
    .filter((locus): locus is TextLocus => locus !== undefined);
  const answerLoci = phraseLoci(answer, atomicAnswerLoci);
  return {
    specimen: { runId: run.id, answer, answerLoci, context: documents(run) },
    addresses,
  };
}

function addressDocument(
  address: SpanAddress,
  allAddresses: readonly SpanAddress[],
  messages: readonly RunMessage[],
  context: readonly ContextDocument[],
): ContextDocument | undefined {
  const bySource = (sourceId: string) => {
    const index = messages.findIndex((message) => message.sourceId === sourceId);
    return index >= 0 ? context[index] : undefined;
  };
  const byLabel = (label: string) => {
    const index = messages.findIndex((message) => message.sourceLabel === label || message.role === label);
    return index >= 0 ? context[index] : undefined;
  };
  if (address.nativeRef.clientSourceId) {
    const match = bySource(address.nativeRef.clientSourceId);
    if (match) return match;
  }
  if (address.nativeRef.parentId) {
    const parent = allAddresses.find((candidate) => candidate.nativeRef.id === address.nativeRef.parentId);
    if (parent) {
      const match = addressDocument(parent, allAddresses, messages, context);
      if (match) return match;
    }
  }
  if (address.nativeRef.sourceLabel) {
    const match = byLabel(address.nativeRef.sourceLabel);
    if (match) return match;
  }
  if (address.nativeRef.collection === "run.messages") {
    const match = /^message-(\d+)$/.exec(address.nativeRef.id);
    if (match) return context[Number(match[1])];
  }
  // A link's native context_index indexes measured prompt spans, not necessarily readable messages.
  // Only the single-document case is unambiguous; otherwise fail closed instead of highlighting the
  // wrong prose. Stable source ids, parents, and labels above remain the authoritative mappings.
  return context.length === 1 ? context[0] : undefined;
}

function selectionState(query: InfluenceQuery): InfluenceSelection["state"] {
  return query.measurement.state;
}

/** Collapse repeated backend links without losing opposing effects at the same source span. */
function distinctRelated(loci: readonly RelatedContextLocus[]): RelatedContextLocus[] {
  const strongest = new Map<string, RelatedContextLocus>();
  for (const locus of loci) {
    const key = `${locus.documentId}\u0000${locus.id}\u0000${locus.effect}`;
    const current = strongest.get(key);
    if (!current || Math.abs(locus.deltaNats) > Math.abs(current.deltaNats)) strongest.set(key, locus);
  }
  return [...strongest.values()];
}

export function projectInfluenceSelection(
  query: InfluenceQuery,
  run: RunRecord,
  projection: ReaderProjection,
): InfluenceSelection {
  const state = selectionState(query);
  if (state !== "available") return { state, reason: query.measurement.reason, related: [] };

  const allAddresses = projection.addresses.addresses;
  const messages = contextMessages(run);
  const related = query.links.flatMap((link): RelatedContextLocus[] => {
    const address = allAddresses.find((candidate) => candidate.addressId === link.sourceSpanId);
    if (!address) return [];
    const document = addressDocument(address, allAddresses, messages, projection.specimen.context);
    if (!document?.text) return [];
    const locus = textLocus(address, document.text);
    if (!locus) return [];
    return [{
      ...locus,
      documentId: document.id,
      effect: link.effect,
      deltaNats: link.deltaNats,
      evidenceState: link.evidenceState,
      answerLocusId: link.answerSpanId,
    }];
  });

  const methodName = typeof query.measurement.method?.name === "string"
    ? query.measurement.method.name
    : undefined;
  const floor = typeof query.measurement.thresholds?.measurement_floor_nats === "number"
    ? query.measurement.thresholds.measurement_floor_nats
    : undefined;
  return { state, method: methodName, floorNats: floor, related: distinctRelated(related) };
}

/** Project the backend's two-sided tension pairs onto the same readable source documents as influence. */
export function projectTensionSelections(
  tension: ContextTension,
  run: RunRecord,
  projection: ReaderProjection,
): Readonly<Record<string, InfluenceSelection>> {
  if (tension.measurement.state !== "available") {
    return Object.fromEntries(projection.specimen.answerLoci.map((locus) => [locus.id, {
      state: tension.measurement.state,
      reason: tension.measurement.reason,
      related: [],
    }]));
  }
  const allAddresses = projection.addresses.addresses;
  const messages = contextMessages(run);
  const byAnswer = new Map<string, RelatedContextLocus[]>();
  for (const pair of tension.tensions) {
    const current = byAnswer.get(pair.answerSpanId) ?? [];
    for (const side of [pair.supporting, pair.suppressing]) {
      const address = allAddresses.find((candidate) => candidate.addressId === side.sourceSpanId);
      if (!address) continue;
      const document = addressDocument(address, allAddresses, messages, projection.specimen.context);
      if (!document?.text) continue;
      const locus = textLocus(address, document.text);
      if (!locus) continue;
      if (!current.some((candidate) => candidate.id === locus.id && candidate.effect === side.effect)) {
        current.push({
          ...locus,
          documentId: document.id,
          effect: side.effect,
          deltaNats: side.deltaNats,
          evidenceState: side.evidenceState,
          answerLocusId: pair.answerSpanId,
        });
      }
    }
    byAnswer.set(pair.answerSpanId, current);
  }
  return Object.fromEntries(projection.specimen.answerLoci.flatMap((locus) => {
    const memberIds = locus.memberIds ?? [locus.id];
    const related = distinctRelated(memberIds.flatMap((answerId) => byAnswer.get(answerId) ?? []));
    return related.length ? [[locus.id, { state: "available" as const, method: "Recorded context tension", related }]] : [];
  }));
}

/** Keep every close-call locator returned by the backend, preserving its recorded token coordinate. */
export function projectDecisionLoci(breakpoints: SuggestedBreakpoints, run: RunRecord): DecisionLocus[] {
  return breakpoints.breakpoints.flatMap((breakpoint): DecisionLocus[] => {
    if (!breakpoint.closeCall) return [];
    const interval = breakpoint.tokenInterval;
    const start = interval ? codePointOffset(run.response ?? "", interval.start) : undefined;
    const end = interval ? codePointOffset(run.response ?? "", interval.end) : undefined;
    return [{
      id: breakpoint.breakpointId,
      position: breakpoint.position,
      start,
      end,
      emittedToken: run.responseTokens?.[breakpoint.position],
      emittedProbability: breakpoint.closeCall.emittedProbability,
      rivalTokenId: breakpoint.closeCall.rivalTokenId,
      rivalProbability: breakpoint.closeCall.rivalProbability,
      margin: breakpoint.closeCall.margin,
      meaningful: breakpoint.closeCall.meaningful,
    }];
  });
}

/** The UI indexes UTF-16 strings; the API accepts Unicode code-point ranges. */
export function locusQueryRange(answer: string, locus: TextLocus): { start: number; end: number } {
  return {
    start: codePointLength(answer.slice(0, locus.start)),
    end: codePointLength(answer.slice(0, locus.end)),
  };
}
