import { afterAll, beforeAll, test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import {
  createSampler,
  destroySharedDevice,
  getGPUDevice,
  lemurTexture,
} from "wesl-test";
import { lygiaExampleImage } from "./testUtil.ts";

imageMatcher();

let device: GPUDevice;

beforeAll(async () => {
  device = await getGPUDevice();
});

afterAll(() => {
  destroySharedDevice();
});

test("Kaleidoscope with multiple segment counts (4, 6, 8, 12)", async () => {
  const inputTex = await lemurTexture(device);
  const sampler = createSampler(device);

  await lygiaExampleImage(device, "kaleidoscope", {
    size: [128, 128],
    inputTextures: [{ texture: inputTex, sampler }],
  });
});
