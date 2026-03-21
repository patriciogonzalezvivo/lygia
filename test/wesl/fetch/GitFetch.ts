import { spawnSync } from "node:child_process";
import fs from "node:fs/promises";
import { fileURLToPath } from "node:url";

/**
 * Clone or fetch+checkout a git repo to a local directory.
 * @param url - git remote url
 * @param revision - commit hash or branch to checkout
 * @param targetDir - target directory for the repo
 */
export async function gitFetch(url: string, revision: string, targetDir: URL): Promise<void> {
  if (await exists(targetDir)) {
    if (gitHead(targetDir) === revision) return; // already at target
    if (!git(["cat-file", "commit", revision], targetDir)) {
      git(["fetch", "--quiet"], targetDir, `Fetching ${url} ${revision}`);
    }
    git(["checkout", "--quiet", revision], targetDir, `Checking out ${url} ${revision}`);
  } else {
    const parent = new URL(".", targetDir);
    await fs.mkdir(parent, { recursive: true });
    const target = fileURLToPath(targetDir);
    git(["clone", "--depth=1", url, "--revision", revision, target], parent, `Cloning ${url} ${revision}`);
  }
}

function gitHead(cwd: URL): string | undefined {
  const result = spawnSync("git", ["rev-parse", "HEAD"], { cwd });
  return result.status === 0 ? result.stdout?.toString().trim() : undefined;
}

function exists(path: URL): Promise<boolean> {
  return fs.access(path).then(() => true, () => false);
}

/** Run a git command synchronously. Throws with stderr on failure if msg is provided. */
function git(args: string[], cwd: URL | string, msg?: string): boolean {
  const result = spawnSync("git", args, { cwd });
  if (msg && result.status !== 0) {
    throw new Error(`${msg} failed ${result.stderr?.toString()}`);
  }
  return result.status === 0;
}
