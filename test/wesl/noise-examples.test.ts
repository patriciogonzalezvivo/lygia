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

test("Perlin noise FBM pattern", async () => {
  await lygiaExampleImage(device, "perlin-noise-fbm", {
    size: [128, 128],
  });
});

test("Simplex noise FBM pattern", async () => {
  await lygiaExampleImage(device, "snoise-fbm", {
    size: [128, 128],
  });
});

test("Worley cellular pattern", async () => {
  await lygiaExampleImage(device, "worley-cellular", {
    size: [128, 128],
  });
});

test("Periodic noise tiling pattern", async () => {
  await lygiaExampleImage(device, "pnoise-tiling", {
    size: [128, 128],
  });
});

test("Wavelet vorticity pattern", async () => {
  await lygiaExampleImage(device, "wavelet-vorticity", {
    size: [128, 128],
  });
});
