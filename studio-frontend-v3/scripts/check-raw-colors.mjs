import { readdir, readFile } from "node:fs/promises";
import { join, relative } from "node:path";

const root = new URL("../src/", import.meta.url);
const allowed = new Set(["src/styles/tokens.css"]);
const rawHex = /#[0-9a-fA-F]{3,8}\b/g;
const violations = [];

async function walk(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) await walk(path);
    else if (/\.(css|ts|tsx)$/.test(entry.name)) {
      const name = relative(new URL("../", import.meta.url).pathname, path);
      if (allowed.has(name)) continue;
      const contents = await readFile(path, "utf8");
      if (rawHex.test(contents)) violations.push(name);
      rawHex.lastIndex = 0;
    }
  }
}

await walk(new URL("../src/", import.meta.url).pathname);
if (violations.length) {
  console.error(`Raw hex colors are restricted to the approved token/theme files:\n${violations.join("\n")}`);
  process.exitCode = 1;
}
