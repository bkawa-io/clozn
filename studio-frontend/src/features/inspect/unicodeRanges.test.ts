import { describe, expect, it } from "vitest";
import {
  InvalidCodePointRangeError,
  assertValidCodePointRange,
  codePointLength,
  codePointOffsetToUtf16Index,
  isValidCodePointRange,
  semanticTextRanges,
  segmentAnnotatedText,
  sliceCodePoints,
  toCodePoints,
  utf16IndexToCodePointOffset,
  validateCodePointRange,
} from "./unicodeRanges";

describe("Unicode code-point ranges", () => {
  const text = "A😀e\u0301Z";

  it("counts and slices Python Unicode code points rather than UTF-16 units or graphemes", () => {
    expect(text.length).toBe(6);
    expect(codePointLength(text)).toBe(5);
    expect(toCodePoints(text)).toEqual(["A", "😀", "e", "\u0301", "Z"]);
    expect(sliceCodePoints(text, 1, 2)).toBe("😀");
    expect(sliceCodePoints(text, 2, 4)).toBe("e\u0301");
    expect(sliceCodePoints(text, 3, 4)).toBe("\u0301");
    expect(sliceCodePoints(text, 5, 5)).toBe("");
  });

  it("converts only exact boundaries between code-point and UTF-16 coordinates", () => {
    expect([0, 1, 2, 3, 4, 5].map((offset) => codePointOffsetToUtf16Index(text, offset)))
      .toEqual([0, 1, 3, 4, 5, 6]);
    expect(codePointOffsetToUtf16Index(text, 6)).toBeNull();

    expect(utf16IndexToCodePointOffset(text, 0)).toBe(0);
    expect(utf16IndexToCodePointOffset(text, 1)).toBe(1);
    expect(utf16IndexToCodePointOffset(text, 2)).toBeNull();
    expect(utf16IndexToCodePointOffset(text, 3)).toBe(2);
    expect(utf16IndexToCodePointOffset(text, 6)).toBe(5);
    expect(utf16IndexToCodePointOffset(text, 7)).toBeNull();
  });

  it("validates against code-point length and rejects malformed ranges without clamping", () => {
    expect(validateCodePointRange("😀", { start: 0, end: 1 })).toEqual({
      valid: true,
      codePointLength: 1,
    });
    expect(validateCodePointRange("😀", { start: 0, end: 2 })).toMatchObject({
      valid: false,
      reason: "end_out_of_bounds",
      codePointLength: 1,
    });
    expect(validateCodePointRange(text, { start: 2, end: 1 })).toMatchObject({
      valid: false,
      reason: "end_before_start",
    });
    expect(validateCodePointRange(text, { start: 1.5, end: 2 })).toMatchObject({
      valid: false,
      reason: "start_not_integer",
    });
    expect(isValidCodePointRange(text, { start: 2, end: 2 })).toBe(true);
    expect(isValidCodePointRange(text, { start: 2, end: 2 }, { allowEmpty: false })).toBe(false);
    expect(() => assertValidCodePointRange(text, { start: -1, end: 1 }))
      .toThrow(InvalidCodePointRangeError);
    expect(() => sliceCodePoints(text, 0, 6)).toThrow(InvalidCodePointRangeError);
  });
});

describe("annotated text segmentation", () => {
  it("preserves all text and all covering IDs across overlapping and identical ranges", () => {
    const text = "A😀BCe\u0301F";
    const annotations = [
      { id: "middle", start: 3, end: 7 },
      { id: "outer", start: 1, end: 6 },
      { id: "left", start: 1, end: 4 },
      { id: "identical", start: 3, end: 7 },
    ];

    const segments = segmentAnnotatedText(text, annotations);
    expect(segments).toEqual([
      { start: 0, end: 1, text: "A", annotationIds: [] },
      { start: 1, end: 3, text: "😀B", annotationIds: ["left", "outer"] },
      {
        start: 3,
        end: 4,
        text: "C",
        annotationIds: ["identical", "left", "middle", "outer"],
      },
      { start: 4, end: 6, text: "e\u0301", annotationIds: ["identical", "middle", "outer"] },
      { start: 6, end: 7, text: "F", annotationIds: ["identical", "middle"] },
    ]);
    expect(segments.map((segment) => segment.text).join("")).toBe(text);
    expect(segmentAnnotatedText(text, [...annotations].reverse())).toEqual(segments);
  });

  it("honors half-open adjacency and keeps unannotated text in the stream", () => {
    expect(segmentAnnotatedText("abc", [
      { id: "left", start: 0, end: 1 },
      { id: "right", start: 1, end: 2 },
    ])).toEqual([
      { start: 0, end: 1, text: "a", annotationIds: ["left"] },
      { start: 1, end: 2, text: "b", annotationIds: ["right"] },
      { start: 2, end: 3, text: "c", annotationIds: [] },
    ]);
  });

  it("returns a complete plain segment without annotations and no segments for empty text", () => {
    expect(segmentAnnotatedText("😀e\u0301", [])).toEqual([
      { start: 0, end: 3, text: "😀e\u0301", annotationIds: [] },
    ]);
    expect(segmentAnnotatedText("", [{ id: "empty", start: 0, end: 0 }])).toEqual([]);
  });

  it("rejects invalid annotation geometry instead of clipping or dropping evidence", () => {
    expect(() => segmentAnnotatedText("A😀B", [{ id: "bad", start: 1, end: 4 }]))
      .toThrow(InvalidCodePointRangeError);
    expect(() => segmentAnnotatedText("abc", [{ id: "", start: 0, end: 1 }]))
      .toThrow(TypeError);
  });
});

describe("semantic text ranges", () => {
  it("uses sentence boundaries for readable phrases and covers the text exactly", () => {
    const text = "The first sentence explains one idea. The second sentence adds another useful detail. A final sentence closes it.";
    const ranges = semanticTextRanges(text, { minLength: 16, targetLength: 42, maxLength: 60 });

    expect(ranges.map((range) => sliceCodePoints(text, range.start, range.end))).toEqual([
      "The first sentence explains one idea. ",
      "The second sentence adds another useful detail. ",
      "A final sentence closes it.",
    ]);
    expect(ranges[0].start).toBe(0);
    expect(ranges.at(-1)?.end).toBe(codePointLength(text));
    expect(ranges.every((range, index) => index === 0 || ranges[index - 1].end === range.start)).toBe(true);
  });

  it("falls back to clause and word boundaries without splitting Unicode code points", () => {
    const text = "Emoji 🌕 stay whole while this deliberately long clause keeps moving, then another deliberately long clause follows without a sentence ending";
    const ranges = semanticTextRanges(text, { minLength: 18, targetLength: 34, maxLength: 46 });

    expect(ranges.length).toBeGreaterThan(1);
    expect(ranges.map((range) => sliceCodePoints(text, range.start, range.end)).join("")).toBe(text);
    expect(ranges.every((range) => range.end - range.start <= 46)).toBe(true);
    expect(ranges.some((range) => sliceCodePoints(text, range.start, range.end).includes("🌕"))).toBe(true);
  });

  it("keeps short prose as one phrase and rejects incoherent sizing", () => {
    expect(semanticTextRanges("Hi 🌕!")).toEqual([{ start: 0, end: 5 }]);
    expect(() => semanticTextRanges("text", { minLength: 10, targetLength: 5 }))
      .toThrow(RangeError);
  });

  it("does not split an ASCII abbreviation between its periods", () => {
    const text = "During the emergency, branches logged visits after five p.m., demonstrating how familiar access helped residents. The next sentence adds detail.";
    const ranges = semanticTextRanges(text, { minLength: 20, targetLength: 58, maxLength: 82 });
    const pieces = ranges.map((range) => sliceCodePoints(text, range.start, range.end));

    expect(pieces.join("")).toBe(text);
    expect(pieces.every((piece) => !piece.endsWith(" p.") && !piece.startsWith("m."))).toBe(true);
  });

  it("does not treat punctuation inside a number as a clause boundary", () => {
    const text = "During the emergency, branches logged 2,400 visits after five p.m., demonstrating how familiar access helped residents. The next sentence adds detail.";
    const ranges = semanticTextRanges(text, { minLength: 20, targetLength: 44, maxLength: 62 });
    const pieces = ranges.map((range) => sliceCodePoints(text, range.start, range.end));

    expect(pieces.join("")).toBe(text);
    expect(pieces.every((piece) => !piece.endsWith("2,") && !piece.startsWith("400"))).toBe(true);
  });
});
