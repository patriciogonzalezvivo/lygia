import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("blendAdd3Opacity - opacity 0.5", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.3, 0.5, 0.7);
       let blend = vec3f(0.8, 0.2, 0.4);
       let result = blendAdd3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // At opacity 0.5, result is halfway between base and full blend
  // Full blend: min(0.3+0.8,1)=1.0, min(0.5+0.2,1)=0.7, min(0.7+0.4,1)=1.0
  // Result: blend*0.5 + base*0.5 = [1.0*0.5+0.3*0.5, 0.7*0.5+0.5*0.5, 1.0*0.5+0.7*0.5]
  expectCloseTo([0.65, 0.6, 0.85], result);
});

test("blendAdd3Opacity - opacity 0", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.3, 0.5, 0.7);
       let blend = vec3f(0.8, 0.2, 0.4);
       let result = blendAdd3Opacity(base, blend, 0.0);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // At opacity 0, should return base unchanged
  expectCloseTo([0.3, 0.5, 0.7], result);
});

test("blendAdd3Opacity - opacity 1", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.3, 0.5, 0.7);
       let blend = vec3f(0.8, 0.2, 0.4);
       let result = blendAdd3Opacity(base, blend, 1.0);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // At opacity 1, should match non-opacity version
  expectCloseTo([1.0, 0.7, 1.0], result);
});

test("blendMultiply3Opacity", async () => {
  const src = `
     import lygia::color::blend::multiply::blendMultiply3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.6, 0.4);
       let blend = vec3f(0.5, 0.5, 0.5);
       let result = blendMultiply3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.4, 0.3, 0.2]
  // At 0.5: blend*0.5 + base*0.5 = [0.4*0.5+0.8*0.5, 0.3*0.5+0.6*0.5, 0.2*0.5+0.4*0.5]
  expectCloseTo([0.6, 0.45, 0.3], result);
});

test("blendScreen3Opacity", async () => {
  const src = `
     import lygia::color::blend::screen::blendScreenWithOpacity3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendScreenWithOpacity3(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.58, 0.7, 0.8]
  // At 0.5: [0.58*0.5+0.4*0.5, 0.7*0.5+0.5*0.5, 0.8*0.5+0.6*0.5]
  expectCloseTo([0.49, 0.6, 0.7], result);
});

test("blendAverage3Opacity", async () => {
  const src = `
     import lygia::color::blend::average::blendAverage3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.8);
       let blend = vec3f(0.4, 0.8, 0.2);
       let result = blendAverage3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.5, 0.6, 0.5]
  // At 0.5: [0.5*0.5+0.6*0.5, 0.6*0.5+0.4*0.5, 0.5*0.5+0.8*0.5]
  expectCloseTo([0.55, 0.5, 0.65], result);
});

test("blendLighten3Opacity", async () => {
  const src = `
     import lygia::color::blend::lighten::blendLighten3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.5);
       let blend = vec3f(0.3, 0.7, 0.5);
       let result = blendLighten3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.6, 0.7, 0.5]
  // At 0.5: [0.6*0.5+0.6*0.5, 0.7*0.5+0.4*0.5, 0.5*0.5+0.5*0.5]
  expectCloseTo([0.6, 0.55, 0.5], result);
});

test("blendDarken3Opacity", async () => {
  const src = `
     import lygia::color::blend::darken::blendDarken3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.5);
       let blend = vec3f(0.3, 0.7, 0.5);
       let result = blendDarken3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.3, 0.4, 0.5]
  // At 0.5: [0.3*0.5+0.6*0.5, 0.4*0.5+0.4*0.5, 0.5*0.5+0.5*0.5]
  expectCloseTo([0.45, 0.4, 0.5], result);
});
