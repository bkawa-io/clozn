import { describe, expect, it } from "vitest";
import {
  InvalidCodePointRangeError,
  assertValidCodePointRange,
  codePointLength,
  codePointOffsetToUtf16Index,
  isValidCodePointRange,
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
