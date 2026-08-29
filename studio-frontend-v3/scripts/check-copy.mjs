import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

const root = new URL("../src/", import.meta.url).pathname;
const forbidden = [
  "Why did it say this?",
  "Recorded specimen",
  "Local only",
  "Your conversation history",
  "Explore what happened",
  "Evidence locus map",
];
const violations = [];

async function walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) await walk(path);
    else if (/\.(ts|tsx)$/.test(entry.name)) {
      const contents = await readFile(path, "utf8");
      for (const phrase of forbidden) if (contents.includes(phrase)) violations.push(`${path}: ${phrase}`);
    }
  }
}

await walk(root);
if (violations.length) {
  console.error(`Prototype narration is not allowed in v3 source:\n${violations.join("\n")}`);
  process.exitCode = 1;
}
