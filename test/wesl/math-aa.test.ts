import { test } from "vitest";
import { imageMatcher } from "vitest-image-snapshot";
import { getGPUDevice } from "wgsl-test";
import { lygiaExampleImage } from "./testUtil.ts";

imageMatcher();

// Visual regression test verified against GLSL reference (lygia_examples/draw_aa.frag)
test("draw-aa - spiral pattern showing AA quality", async () => {
  const device = await getGPUDevice();
  await lygiaExampleImage(device, "draw-aa", {
    size: [512, 512],
    uniforms: {
      time: 0.1668, // Tuned to match GLSL snapshot
    },
  });
});
