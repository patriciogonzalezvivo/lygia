import { afterAll, beforeAll, test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import {
  createSampler,
  destroySharedDevice,
  getGPUDevice,
  pngToTexture,
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

test("sprite animation with Mega Man sprite sheet", async () => {
  const spriteTex = await pngToTexture(
    device,
    "test/wesl/assets/sprite_megaman.png",
  );
  const sampler = createSampler(device);

  await lygiaExampleImage(device, "sprite-megaman", {
    size: [128, 128],
    textures: [spriteTex],
    samplers: [sampler],
    uniforms: { time: 22 },
  });
});
