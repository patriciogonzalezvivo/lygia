import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";
import { gitFetch } from "./GitFetch.ts";

const snapshotRepo =
  "https://github.com/patriciogonzalezvivo/lygia-snapshots.git";
const snapshotRevision = "3e5ed0a110c6c893735e290001bc11cdcef6a7c8";
const snapshotDir = new URL(
  "../__image_snapshots__/",
  pathToFileURL(`${import.meta.dirname}/`),
);

/** vitest globalSetup: fetch snapshot images before tests run */
export async function setup(): Promise<void> {
  await gitFetch(snapshotRepo, snapshotRevision, snapshotDir);
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
