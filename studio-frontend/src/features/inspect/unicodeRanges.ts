/**
 * Text coordinates received from CLOZN are half-open Python `str` offsets: Unicode code points,
 * not JavaScript UTF-16 code units and not user-perceived grapheme clusters. Keep every conversion
 * between those coordinate systems explicit in this module.
 */

export interface CodePointRange {
  readonly start: number;
  readonly end: number;
}

export interface CodePointAnnotation extends CodePointRange {
  readonly id: string;
}

export interface AnnotatedTextSegment extends CodePointRange {
  readonly text: string;
  readonly annotationIds: readonly string[];
}

export type InvalidCodePointRangeReason =
  | "start_not_integer"
  | "end_not_integer"
  | "start_negative"
  | "end_before_start"
  | "end_out_of_bounds"
  | "empty_not_allowed";

export type CodePointRangeValidation =
  | { readonly valid: true; readonly codePointLength: number }
  | {
      readonly valid: false;
      readonly codePointLength: number;
      readonly reason: InvalidCodePointRangeReason;
    };

export interface ValidateCodePointRangeOptions {
  /** Half-open ranges may ordinarily be empty. Evidence-bearing callers can opt out explicitly. */
  readonly allowEmpty?: boolean;
}

/** Materialize the string using JavaScript's code-point iterator, which keeps surrogate pairs whole. */
export function toCodePoints(text: string): string[] {
  return Array.from(text);
}

/** Python `len(text)` semantics for Unicode strings. */
export function codePointLength(text: string): number {
  let length = 0;
  for (const _point of text) length += 1;
  return length;
}

/**
 * Validate a half-open range against the string's code-point length. The text itself is required so
 * callers cannot accidentally validate against JavaScript's UTF-16 `text.length`.
 */
export function validateCodePointRange(
  text: string,
  range: CodePointRange,
  options: ValidateCodePointRangeOptions = {},
): CodePointRangeValidation {
  const length = codePointLength(text);
  if (!Number.isSafeInteger(range.start)) {
    return { valid: false, codePointLength: length, reason: "start_not_integer" };
  }
  if (!Number.isSafeInteger(range.end)) {
    return { valid: false, codePointLength: length, reason: "end_not_integer" };
  }
  if (range.start < 0) {
    return { valid: false, codePointLength: length, reason: "start_negative" };
  }
  if (range.end < range.start) {
    return { valid: false, codePointLength: length, reason: "end_before_start" };
  }
  if (range.end > length) {
    return { valid: false, codePointLength: length, reason: "end_out_of_bounds" };
  }
  if (options.allowEmpty === false && range.start === range.end) {
    return { valid: false, codePointLength: length, reason: "empty_not_allowed" };
  }
  return { valid: true, codePointLength: length };
}

export function isValidCodePointRange(
  text: string,
  range: CodePointRange,
  options?: ValidateCodePointRangeOptions,
): boolean {
  return validateCodePointRange(text, range, options).valid;
}

export class InvalidCodePointRangeError extends RangeError {
  readonly name = "InvalidCodePointRangeError";

  constructor(
    readonly range: CodePointRange,
    readonly reason: InvalidCodePointRangeReason,
    readonly basisCodePointLength: number,
  ) {
    super(
      `Invalid Unicode code-point range [${range.start}, ${range.end}): ${reason} `
      + `(basis length ${basisCodePointLength})`,
    );
  }
}

export function assertValidCodePointRange(
  text: string,
  range: CodePointRange,
  options?: ValidateCodePointRangeOptions,
): void {
  const validation = validateCodePointRange(text, range, options);
  if (!validation.valid) {
    throw new InvalidCodePointRangeError(range, validation.reason, validation.codePointLength);
  }
}

/** Slice with Python Unicode code-point coordinates. Invalid ranges are rejected, never clamped. */
export function sliceCodePoints(text: string, start: number, end: number): string {
  const range = { start, end };
  assertValidCodePointRange(text, range);
  return toCodePoints(text).slice(start, end).join("");
}

/** Convert a Python code-point offset to a DOM/string UTF-16 index. */
export function codePointOffsetToUtf16Index(text: string, offset: number): number | null {
  if (!Number.isSafeInteger(offset) || offset < 0) return null;

  let codePointOffset = 0;
  let utf16Index = 0;
  if (offset === 0) return 0;
  for (const point of text) {
    utf16Index += point.length;
    codePointOffset += 1;
    if (codePointOffset === offset) return utf16Index;
  }
  return null;
}

/**
 * Convert a DOM/string UTF-16 index to a Python code-point offset. An index inside a surrogate pair
 * has no exact code-point boundary and returns `null` rather than rounding in either direction.
 */
export function utf16IndexToCodePointOffset(text: string, index: number): number | null {
  if (!Number.isSafeInteger(index) || index < 0 || index > text.length) return null;

  let codePointOffset = 0;
  let utf16Index = 0;
  if (index === 0) return 0;
  for (const point of text) {
    const nextUtf16Index = utf16Index + point.length;
    if (index < nextUtf16Index) return null;
    codePointOffset += 1;
    utf16Index = nextUtf16Index;
    if (index === utf16Index) return codePointOffset;
  }
  return null;
}

/**
 * Split text at every annotation boundary. Every non-empty segment appears once and contains every
 * annotation ID whose half-open range covers it. IDs are sorted so output does not depend on wire
 * order; overlapping and identical ranges are both represented without duplicating or dropping text.
 */
export function segmentAnnotatedText(
  text: string,
  annotations: readonly CodePointAnnotation[],
): AnnotatedTextSegment[] {
  const points = toCodePoints(text);
  const length = points.length;
  const boundaries = new Set<number>([0, length]);

  for (const annotation of annotations) {
    if (typeof annotation.id !== "string" || annotation.id.length === 0) {
      throw new TypeError("Annotation id must be a non-empty string");
    }
    assertValidCodePointRange(text, annotation);
    boundaries.add(annotation.start);
    boundaries.add(annotation.end);
  }

  const sortedBoundaries = [...boundaries].sort((left, right) => left - right);
  const segments: AnnotatedTextSegment[] = [];
  for (let index = 0; index < sortedBoundaries.length - 1; index += 1) {
    const start = sortedBoundaries[index];
    const end = sortedBoundaries[index + 1];
    if (start === end) continue;

    const annotationIds = [...new Set(
      annotations
        .filter((annotation) => annotation.start <= start && end <= annotation.end)
        .map((annotation) => annotation.id),
    )].sort((left, right) => left < right ? -1 : left > right ? 1 : 0);

    segments.push({
      start,
      end,
      text: points.slice(start, end).join(""),
      annotationIds,
    });
  }
  return segments;
}
