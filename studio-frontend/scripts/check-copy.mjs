import { readdir, readFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";

const sourceRoot = fileURLToPath(new URL("../src/", import.meta.url));
const banned = [
  "local" + " only",
  "no " + "cloud",
  "no " + "telemetry",
  "doesn't " + "phone home",
  "does not " + "phone home",
  "the river, not the " + "molecules",
  "you're an " + "angel",
  "you are an " + "angel",
];

async function filesIn(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  const files = await Promise.all(entries.map((entry) => {
    const path = join(dir, entry.name);
    return entry.isDirectory() ? filesIn(path) : [path];
  }));
  return files.flat();
}

const failures = [];
for (const file of await filesIn(sourceRoot)) {
  if (![".ts", ".tsx", ".css"].includes(extname(file))) continue;
  const text = (await readFile(file, "utf8")).toLowerCase();
  for (const phrase of banned) {
    if (text.includes(phrase)) failures.push(`${file}: "${phrase}"`);
  }
}

if (failures.length) {
  console.error("Studio copy check failed:\n" + failures.join("\n"));
  process.exit(1);
}

console.log("Studio copy check passed.");
