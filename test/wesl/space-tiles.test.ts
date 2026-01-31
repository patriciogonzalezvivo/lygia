import { afterAll, beforeAll, test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import { destroySharedDevice, getGPUDevice } from "wgsl-test";
import { lygiaExampleImage } from "./testUtil.ts";

imageMatcher();

let device: GPUDevice;

beforeAll(async () => {
  device = await getGPUDevice();
});

afterAll(() => {
  destroySharedDevice();
});

test("Hexagonal tiling pattern", async () => {
  await lygiaExampleImage(device, "hex-tile", {
    size: [128, 128],
  });
});

test("Brick tiling pattern", async () => {
  await lygiaExampleImage(device, "brick-tile", {
    size: [128, 128],
  });
});

test("Checkerboard tiling pattern", async () => {
  await lygiaExampleImage(device, "checker-tile", {
    size: [128, 128],
  });
});

test("Mirror tiling pattern", async () => {
  await lygiaExampleImage(device, "mirror-tile", {
    size: [128, 128],
  });
});

test("Windmill tiling pattern", async () => {
  await lygiaExampleImage(device, "windmill-tile", {
    size: [128, 128],
  });
});

test("Triangular tiling pattern", async () => {
  await lygiaExampleImage(device, "tri-tile", {
    size: [128, 128],
  });
});
