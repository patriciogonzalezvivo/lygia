import { afterAll, beforeAll, test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import { destroySharedDevice, getGPUDevice } from "wesl-test";
import { lygiaExampleImage } from "./testUtil.ts";

imageMatcher();

let device: GPUDevice;

beforeAll(async () => {
  device = await getGPUDevice();
});

afterAll(() => {
  destroySharedDevice();
});

test("stroke and strokeEdge - grid pattern", async () => {
  // Renders a 4x4 grid of circles:
  // Top 2 rows: stroke() with varying widths (0.05 to 0.2)
  // Bottom 2 rows: strokeEdge() with varying edge smoothness (0.001 to 0.051)
  await lygiaExampleImage(device, "draw-stroke" );
});
