import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute, lygiaTestWesl } from "./testUtil.ts";

await lygiaTestWesl("test/wesl/shaders/color_space_rgb_test");

test("hsv2ryb - FAST mode", async () => {
  const src = `
     import lygia::color::space::hsv2ryb::hsv2ryb;

     @compute @workgroup_size(1)
     fn foo() {
       let hsv = vec3f(0.3333, 0.8, 0.9); // Greenish HSV
       let result = hsv2ryb(hsv);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, {
    elem: "vec3f",
    conditions: { HSV2RYB_FAST: true },
  });
  // HSV -> RYB using fast CMY bias version
  expectCloseTo([0.9, 0.9, 0.18], result);
});
