import type { TokenReading } from "../../data/types";

export type AlignmentKind = "same" | "changed" | "a-only" | "b-only";

export interface AlignmentColumn {
  aIndex?: number;
  bIndex?: number;
  kind: AlignmentKind;
}

export interface TokenAlignment {
  columns: AlignmentColumn[];
  matched: number;
  hunks: number;
  firstChangedColumn: number;
  columnByA: Map<number, number>;
  columnByB: Map<number, number>;
}

interface Match {
  aIndex: number;
  bIndex: number;
}

function lcsMatches(a: TokenReading[], b: TokenReading[]): Match[] {
  const lengths = Array.from({ length: a.length + 1 }, () => new Uint16Array(b.length + 1));
  for (let aIndex = a.length - 1; aIndex >= 0; aIndex -= 1) {
    for (let bIndex = b.length - 1; bIndex >= 0; bIndex -= 1) {
      lengths[aIndex][bIndex] = a[aIndex].text === b[bIndex].text
        ? lengths[aIndex + 1][bIndex + 1] + 1
        : Math.max(lengths[aIndex + 1][bIndex], lengths[aIndex][bIndex + 1]);
    }
  }

  const matches: Match[] = [];
  let aIndex = 0;
  let bIndex = 0;
  while (aIndex < a.length && bIndex < b.length) {
    if (a[aIndex].text === b[bIndex].text) {
      matches.push({ aIndex, bIndex });
      aIndex += 1;
      bIndex += 1;
    } else if (lengths[aIndex + 1][bIndex] >= lengths[aIndex][bIndex + 1]) {
      aIndex += 1;
    } else {
      bIndex += 1;
    }
  }
  return matches;
}

export function alignTokens(a: TokenReading[], b: TokenReading[]): TokenAlignment {
  const columns: AlignmentColumn[] = [];
  let nextA = 0;
  let nextB = 0;

  const appendGap = (endA: number, endB: number) => {
    const aCount = endA - nextA;
    const bCount = endB - nextB;
    const width = Math.max(aCount, bCount);
    for (let offset = 0; offset < width; offset += 1) {
      const aIndex = offset < aCount ? nextA + offset : undefined;
      const bIndex = offset < bCount ? nextB + offset : undefined;
      columns.push({
        aIndex,
        bIndex,
        kind: aIndex == null ? "b-only" : bIndex == null ? "a-only" : "changed",
      });
    }
    nextA = endA;
    nextB = endB;
  };

  for (const match of [...lcsMatches(a, b), { aIndex: a.length, bIndex: b.length }]) {
    appendGap(match.aIndex, match.bIndex);
    if (match.aIndex < a.length && match.bIndex < b.length) {
      columns.push({ aIndex: match.aIndex, bIndex: match.bIndex, kind: "same" });
      nextA = match.aIndex + 1;
      nextB = match.bIndex + 1;
    }
  }

  const columnByA = new Map<number, number>();
  const columnByB = new Map<number, number>();
  let matched = 0;
  let hunks = 0;
  let inHunk = false;
  let firstChangedColumn = -1;
  columns.forEach((column, columnIndex) => {
    if (column.aIndex != null) columnByA.set(column.aIndex, columnIndex);
    if (column.bIndex != null) columnByB.set(column.bIndex, columnIndex);
    if (column.kind === "same") {
      matched += 1;
      inHunk = false;
    } else {
      if (firstChangedColumn < 0) firstChangedColumn = columnIndex;
      if (!inHunk) hunks += 1;
      inHunk = true;
    }
  });

  return {
    columns,
    matched,
    hunks,
    firstChangedColumn,
    columnByA,
    columnByB,
  };
}
