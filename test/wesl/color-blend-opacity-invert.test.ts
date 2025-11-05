import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("blendDifference3Opacity", async () => {
  const src = `
     import lygia::color::blend::difference::blendDifference3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.3, 0.6);
       let blend = vec3f(0.5, 0.7, 0.4);
       let result = blendDifference3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.3, 0.4, 0.2]
  // At 0.5: [0.3*0.5+0.8*0.5, 0.4*0.5+0.3*0.5, 0.2*0.5+0.6*0.5]
  expectCloseTo([0.55, 0.35, 0.4], result);
});

test("blendDifference3Opacity - opacity 0 returns base", async () => {
  const src = `
     import lygia::color::blend::difference::blendDifference3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.3, 0.6);
       let blend = vec3f(0.5, 0.7, 0.4);
       let result = blendDifference3Opacity(base, blend, 0.0);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // At opacity 0, should return base unchanged
  expectCloseTo([0.8, 0.3, 0.6], result);
});

test("blendExclusion3Opacity", async () => {
  const src = `
     import lygia::color::blend::exclusion::blendExclusion3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.8);
       let blend = vec3f(0.3, 0.7, 0.2);
       let result = blendExclusion3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.54, 0.54, 0.68]
  // At 0.5: [0.54*0.5+0.6*0.5, 0.54*0.5+0.4*0.5, 0.68*0.5+0.8*0.5]
  expectCloseTo([0.57, 0.47, 0.74], result);
});

test("blendNegation3Opacity", async () => {
  const src = `
     import lygia::color::blend::negation::blendNegation3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.7, 0.5, 0.3);
       let blend = vec3f(0.4, 0.6, 0.8);
       let result = blendNegation3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.9, 0.9, 0.9]
  // At 0.5: [0.9*0.5+0.7*0.5, 0.9*0.5+0.5*0.5, 0.9*0.5+0.3*0.5]
  expectCloseTo([0.8, 0.7, 0.6], result);
});

test("blendPhoenix3Opacity", async () => {
  const src = `
     import lygia::color::blend::phoenix::blendPhoenix3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.7, 0.5, 0.3);
       let blend = vec3f(0.4, 0.6, 0.8);
       let result = blendPhoenix3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.7, 0.9, 0.5]
  // At 0.5: [0.7*0.5+0.7*0.5, 0.9*0.5+0.5*0.5, 0.5*0.5+0.3*0.5]
  expectCloseTo([0.7, 0.7, 0.4], result);
});

test("blendReflect3Opacity", async () => {
  const src = `
     import lygia::color::blend::reflect::blendReflect3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.2);
       let blend = vec3f(0.5, 0.3, 0.8);
       let result = blendReflect3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.32, 0.514, 0.2]
  // At 0.5: [0.32*0.5+0.4*0.5, 0.514*0.5+0.6*0.5, 0.2*0.5+0.2*0.5]
  expectCloseTo([0.36, 0.557, 0.2], result, 0.01);
});

test("blendSubtract3Opacity", async () => {
  const src = `
     import lygia::color::blend::subtract::blendSubtract3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.6, 0.5);
       let blend = vec3f(0.3, 0.4, 0.2);
       let result = blendSubtract3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.1, 0.0, 0.0]
  // At 0.5: [0.1*0.5+0.8*0.5, 0.0*0.5+0.6*0.5, 0.0*0.5+0.5*0.5]
  expectCloseTo([0.45, 0.3, 0.25], result);
});

test("blendGlow3Opacity", async () => {
  const src = `
     import lygia::color::blend::glow::blendGlow3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.2);
       let blend = vec3f(0.5, 0.3, 0.8);
       let result = blendGlow3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.417, 0.225, 0.8]
  // At 0.5: [0.417*0.5+0.4*0.5, 0.225*0.5+0.6*0.5, 0.8*0.5+0.2*0.5]
  expectCloseTo([0.409, 0.413, 0.5], result, 0.01);
});

test("blendColorBurn3Opacity", async () => {
  const src = `
     import lygia::color::blend::colorBurn::blendColorBurn3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.5, 0.4);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendColorBurn3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.0, 0.0, 0.0]
  // At 0.5: [0.0*0.5+0.6*0.5, 0.0*0.5+0.5*0.5, 0.0*0.5+0.4*0.5]
  // Relaxed precision for division-based blend mode with opacity
  expectCloseTo([0.3, 0.25, 0.2], result);
});

test("blendColorDodge3Opacity", async () => {
  const src = `
     import lygia::color::blend::colorDodge::blendColorDodge3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendColorDodge3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.571, 0.833, 1.0]
  // At 0.5: [0.571*0.5+0.4*0.5, 0.833*0.5+0.5*0.5, 1.0*0.5+0.6*0.5]
  expectCloseTo([0.486, 0.667, 0.8], result, 0.01);
});

test("blendLinearBurn3Opacity", async () => {
  const src = `
     import lygia::color::blend::linearBurn::blendLinearBurn3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.5, 0.7);
       let blend = vec3f(0.4, 0.3, 0.2);
       let result = blendLinearBurn3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.0, 0.0, 0.0]
  // At 0.5: [0.0*0.5+0.6*0.5, 0.0*0.5+0.5*0.5, 0.0*0.5+0.7*0.5]
  expectCloseTo([0.3, 0.25, 0.35], result);
});

test("blendLinearDodge3Opacity", async () => {
  const src = `
     import lygia::color::blend::linearDodge::blendLinearDodge3Opacity;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.2, 0.1);
       let result = blendLinearDodge3Opacity(base, blend, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Full blend: [0.7, 0.7, 0.7]
  // At 0.5: [0.7*0.5+0.4*0.5, 0.7*0.5+0.5*0.5, 0.7*0.5+0.6*0.5]
  expectCloseTo([0.55, 0.6, 0.65], result);
});
