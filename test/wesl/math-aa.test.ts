import { test } from "vitest";
import { lygiaExampleImage } from "./testUtil.ts";
import { getGPUDevice } from "wesl-test";
import { imageMatcher } from "vitest-image-snapshot";

imageMatcher();

// Visual regression test verified against GLSL reference (lygia_examples/draw_aa.frag)
test("draw-aa - spiral pattern showing AA quality", async () => {
  const device = await getGPUDevice();
  await lygiaExampleImage(device, "draw-aa", {
    size: [512, 512],
    uniforms: {
      time: .1668,  // Tuned to match GLSL snapshot
    },
  });
});
