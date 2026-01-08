import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("blendOverlay3Opacity", async () => {
  const src = `
     import lygia::color::blend::overlay::blendOverlay3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.8);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendOverlay3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.24, 0.6, 0.88]
  // At 0.5: [0.24*0.5+0.4*0.5, 0.6*0.5+0.6*0.5, 0.88*0.5+0.8*0.5]
  expectCloseTo([0.32, 0.6, 0.84], result);
});

test("blendSoftLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::softLight::blendSoftLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.5, 0.6, 0.4);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendSoftLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.4, 0.6, 0.493]
  // At 0.5: [0.4*0.5+0.5*0.5, 0.6*0.5+0.6*0.5, 0.493*0.5+0.4*0.5]
  expectCloseTo([0.45, 0.6, 0.447], result, 0.01);
});

test("blendHardLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::hardLight::blendHardLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.8);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendHardLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.24, 0.6, 0.88]
  // At 0.5: [0.24*0.5+0.4*0.5, 0.6*0.5+0.6*0.5, 0.88*0.5+0.8*0.5]
  expectCloseTo([0.32, 0.6, 0.84], result);
});

test("blendVividLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::vividLight::blendVividLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.5, 0.6, 0.4);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendVividLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.167, 0.6, 0.667]
  // At 0.5: [0.167*0.5+0.5*0.5, 0.6*0.5+0.6*0.5, 0.667*0.5+0.4*0.5]
  expectCloseTo([0.334, 0.6, 0.534], result, 0.01);
});

test("blendPinLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::pinLight::blendPinLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       // Use values where pin light actually modifies output
       let base = vec3f(0.3, 0.7, 0.5);
       let blend = vec3f(0.1, 0.9, 0.5);
       let result = blendPinLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.2, 0.8, 0.5] (from blendPinLight3 test above)
  // At 0.5: [0.2*0.5+0.3*0.5, 0.8*0.5+0.7*0.5, 0.5*0.5+0.5*0.5]
  expectCloseTo([0.25, 0.75, 0.5], result);
});

test("blendLinearLight3Opacity", async () => {
  const src = `
     import lygia::color::blend::linearLight::blendLinearLight3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.5, 0.7);
       let result = blendLinearLight3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.0, 0.5, 1.0]
  // At 0.5: [0.0*0.5+0.4*0.5, 0.5*0.5+0.5*0.5, 1.0*0.5+0.6*0.5]
  expectCloseTo([0.2, 0.5, 0.8], result);
});

test("blendHardMix3Opacity", async () => {
  const src = `
     import lygia::color::blend::hardMix::blendHardMix3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.8);
       let blend = vec3f(0.3, 0.5, 0.2);
       let result = blendHardMix3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.0, 1.0, 1.0]
  // At 0.5: [0.0*0.5+0.4*0.5, 1.0*0.5+0.6*0.5, 1.0*0.5+0.8*0.5]
  expectCloseTo([0.2, 0.8, 0.9], result);
});
