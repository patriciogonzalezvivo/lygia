import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("blendHueOpacity", async () => {
  const src = `
     import lygia::color::blend::hue::blendHueOpacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.4, 0.2);
       let blend = vec3f(0.2, 0.6, 0.8);
       let result = blendHueOpacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend takes hue from blend
  // At 0.5: interpolate between base and blend result
  expectCloseTo([0.5, 0.5, 0.5], result);
});

test("blendSaturationOpacity", async () => {
  const src = `
     import lygia::color::blend::saturation::blendSaturationOpacity;

     @compute @workgroup_size(1)
     fn foo() {
       // Use saturated base and gray blend to show desaturation
       let base = vec3f(1.0, 0.0, 0.0);  // Pure red
       let blend = vec3f(0.5, 0.5, 0.5); // Gray
       let result = blendSaturationOpacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend desaturates to gray ~[1.0, 1.0, 1.0]
  // At opacity 0.5: halfway between base [1,0,0] and desaturated [1,1,1]
  // Result should be partially desaturated red
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[0]).toBeGreaterThan(result[2]);
  // Actual result: halfway to full desaturation
  expectCloseTo([1.0, 0.5, 0.5], result.slice(0, 3), 0.1);
});

test("blendLuminosityOpacity", async () => {
  const src = `
     import lygia::color::blend::luminosity::blendLuminosityOpacity;

     @compute @workgroup_size(1)
     fn foo() {
       // Use bright base and dark blend to show darkening
       let base = vec3f(1.0, 0.0, 0.0);  // Pure red (bright)
       let blend = vec3f(0.1, 0.1, 0.1); // Dark gray
       let result = blendLuminosityOpacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend darkens to ~[0.1, 0.1, 0.1]
  // At opacity 0.5: halfway between base [1,0,0] and darkened [0.1,0.1,0.1]
  // Result should be medium-dark red
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[0]).toBeGreaterThan(result[2]);
  expect(result[0]).toBeLessThan(0.7);
  expectCloseTo([0.55, 0.05, 0.05], result, 0.1);
});
