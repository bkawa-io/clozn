/**
 * Coordinates used to refer to the same investigated object across Studio
 * surfaces.  Labels intentionally do not participate in identity: they can
 * change as presentation changes, while these coordinates cannot.
 */
export type InvestigationSurface =
  | "runs"
  | "inspect"
  | "time-travel"
  | "compare"
  | "model-mri"
  | "test-this";

export interface AnswerSpanLocus {
  readonly kind: "answer-span";
  readonly runId: string;
  readonly answerId: string;
  readonly startChar: number;
  readonly endChar: number;
}

export interface AnswerTokenLocus {
  readonly kind: "answer-token";
  readonly runId: string;
  readonly answerId: string;
  readonly tokenIndex: number;
}

export interface ContextSpanLocus {
  readonly kind: "context-span";
  readonly runId: string;
  readonly sourceId: string;
  readonly startChar: number;
  readonly endChar: number;
}

export interface ConversationTurnLocus {
  readonly kind: "conversation-turn";
  readonly sessionId: string;
  readonly turnId: string;
  readonly turnIndex: number;
}

/** A boundary is before token 0 through after the final token. */
export interface TokenBoundaryLocus {
  readonly kind: "token-boundary";
  readonly runId: string;
  readonly sequenceId: string;
  readonly boundaryIndex: number;
}

export interface TokenLayerLocus {
  readonly kind: "token-layer";
  readonly runId: string;
  readonly sequenceId: string;
  readonly tokenIndex: number;
  readonly layerIndex: number;
}

export type InvestigationLocus =
  | AnswerSpanLocus
  | AnswerTokenLocus
  | ContextSpanLocus
  | ConversationTurnLocus
  | TokenBoundaryLocus
  | TokenLayerLocus;

export interface InvestigationSelection {
  readonly locus: InvestigationLocus;
  /** A locked selection is intentionally retained while the user navigates. */
  readonly locked: boolean;
  readonly origin: InvestigationSurface;
}

const encode = (value: string | number) => encodeURIComponent(String(value));

/** A serializable, display-independent identity suitable for query state. */
export function locusKey(locus: InvestigationLocus): string {
  switch (locus.kind) {
    case "answer-span":
      return `answer-span:${encode(locus.runId)}:${encode(locus.answerId)}:${locus.startChar}:${locus.endChar}`;
    case "answer-token":
      return `answer-token:${encode(locus.runId)}:${encode(locus.answerId)}:${locus.tokenIndex}`;
    case "context-span":
      return `context-span:${encode(locus.runId)}:${encode(locus.sourceId)}:${locus.startChar}:${locus.endChar}`;
    case "conversation-turn":
      return `conversation-turn:${encode(locus.sessionId)}:${encode(locus.turnId)}:${locus.turnIndex}`;
    case "token-boundary":
      return `token-boundary:${encode(locus.runId)}:${encode(locus.sequenceId)}:${locus.boundaryIndex}`;
    case "token-layer":
      return `token-layer:${encode(locus.runId)}:${encode(locus.sequenceId)}:${locus.tokenIndex}:${locus.layerIndex}`;
  }
}

export function sameLocus(left: InvestigationLocus | undefined, right: InvestigationLocus | undefined): boolean {
  return left !== undefined && right !== undefined && locusKey(left) === locusKey(right);
}

export function withSelection(
  locus: InvestigationLocus,
  origin: InvestigationSurface,
  locked = false,
): InvestigationSelection {
  return { locus, origin, locked };
}
