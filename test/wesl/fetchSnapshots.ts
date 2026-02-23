import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";

const snapshotRepo =
  "https://github.com/patriciogonzalezvivo/lygia-snapshots.git";
const snapshotRevision = "3e5ed0a110c6c893735e290001bc11cdcef6a7c8";
const snapshotDir = path.resolve(import.meta.dirname, "__image_snapshots__");

/** vitest globalSetup: fetch snapshot images before tests run */
export async function setup(): Promise<void> {
  await fetchSnapshots(snapshotRepo, snapshotRevision, snapshotDir);
}

/** vitest globalTeardown: warn if snapshot images were modified */
export async function teardown(): Promise<void> {
  const result = spawnSync("git", ["status", "--porcelain"], {
    cwd: snapshotDir,
  });
  const output = result.stdout?.toString().trim();
  if (output) {
    const lines = output.split("\n");
    console.log(
      `\n⚠ ${lines.length} snapshot image(s) changed. ` +
        `Run \`pnpm snapshot:push\` to update the snapshots repo.\n`,
    );
  }
}

/** Clone or fetch+checkout a git repo to a local directory. */
async function fetchSnapshots(
  url: string,
  revision: string,
  targetDir: string,
): Promise<void> {
  if (await exists(targetDir)) {
    // fetch only if the pinned revision isn't already in the local clone
    if (!git(["cat-file", "-e", revision], targetDir)) {
      git(["fetch", "--quiet", "origin"], targetDir, `Fetching ${url}`);
    }
    // checkout is local-only, no network call
    git(
      ["checkout", "--quiet", revision],
      targetDir,
      `Checking out ${revision}`,
    );
  } else {
    // first run: shallow clone at the pinned revision
    const parent = path.dirname(targetDir);
    await fs.mkdir(parent, { recursive: true });
    git(
      ["clone", "--depth=1", "--revision", revision, url, targetDir],
      parent,
      `Cloning ${url} at ${revision}`,
    );
  }
}

function exists(p: string): Promise<boolean> {
  return fs.access(p).then(
    () => true,
    () => false,
  );
}

/** Run a git command synchronously. Throws with stderr on failure if msg is provided. */
function git(args: string[], cwd: string, msg?: string): boolean {
  const result = spawnSync("git", args, { cwd });
  if (msg && result.status !== 0) {
    throw new Error(`${msg} failed: ${result.stderr?.toString()}`);
  }
  return result.status === 0;
}
