import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("blendDifference3", async () => {
  const src = `
     import lygia::color::blend::difference::blendDifference3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.3, 0.6);
       let blend = vec3f(0.5, 0.7, 0.4);
       let result = blendDifference3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Difference mode: abs(base - blend)
  expectCloseTo([0.3, 0.4, 0.2], result);
});

test("blendExclusion3", async () => {
  const src = `
     import lygia::color::blend::exclusion::blendExclusion3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.8);
       let blend = vec3f(0.3, 0.7, 0.2);
       let result = blendExclusion3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Exclusion mode: base + blend - 2 * base * blend
  expectCloseTo([0.54, 0.54, 0.68], result);
});

test("blendNegation3", async () => {
  const src = `
     import lygia::color::blend::negation::blendNegation3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.7, 0.5, 0.3);
       let blend = vec3f(0.4, 0.6, 0.8);
       let result = blendNegation3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Negation mode: 1 - abs(1 - base - blend)
  expectCloseTo([0.9, 0.9, 0.9], result);
});

test("blendPhoenix3", async () => {
  const src = `
     import lygia::color::blend::phoenix::blendPhoenix3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.7, 0.5, 0.3);
       let blend = vec3f(0.4, 0.6, 0.8);
       let result = blendPhoenix3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Phoenix mode: min(base, blend) - max(base, blend) + 1
  // R: min(0.7,0.4) - max(0.7,0.4) + 1 = 0.4 - 0.7 + 1 = 0.7
  // G: min(0.5,0.6) - max(0.5,0.6) + 1 = 0.5 - 0.6 + 1 = 0.9
  // B: min(0.3,0.8) - max(0.3,0.8) + 1 = 0.3 - 0.8 + 1 = 0.5
  expectCloseTo([0.7, 0.9, 0.5], result);
});

test("blendReflect3", async () => {
  const src = `
     import lygia::color::blend::reflect::blendReflect3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.6, 0.2);
       let blend = vec3f(0.5, 0.3, 0.8);
       let result = blendReflect3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Reflect mode: base^2 / (1 - blend) clamped
  expectCloseTo([0.32, 0.514, 0.2], result, 0.01);
});

test("blendSubtract3", async () => {
  const src = `
     import lygia::color::blend::subtract::blendSubtract3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.6, 0.5);
       let blend = vec3f(0.3, 0.4, 0.2);
       let result = blendSubtract3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Subtract mode: max(base + blend - 1, 0)
  // R: max(0.8 + 0.3 - 1, 0) = max(0.1, 0) = 0.1
  // G: max(0.6 + 0.4 - 1, 0) = max(0.0, 0) = 0.0
  // B: max(0.5 + 0.2 - 1, 0) = max(-0.3, 0) = 0.0
  expectCloseTo([0.1, 0.0, 0.0], result);
});

test("blendDifference - f32", async () => {
  const src = `
     import lygia::color::blend::difference::blendDifference;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendDifference(0.8, 0.5);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.3], [result[0]]);
});

test("blendExclusion - f32", async () => {
  const src = `
     import lygia::color::blend::exclusion::blendExclusion;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendExclusion(0.6, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Exclusion: 0.6 + 0.3 - 2*0.6*0.3 = 0.9 - 0.36 = 0.54
  expectCloseTo([0.54], [result[0]]);
});

test("blendNegation - f32", async () => {
  const src = `
     import lygia::color::blend::negation::blendNegation;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendNegation(0.7, 0.4);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Negation: 1 - abs(1 - 0.7 - 0.4) = 1 - abs(-0.1) = 1 - 0.1 = 0.9
  expectCloseTo([0.9], [result[0]]);
});

test("blendPhoenix - f32", async () => {
  const src = `
     import lygia::color::blend::phoenix::blendPhoenix;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendPhoenix(0.7, 0.4);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Phoenix: min(0.7,0.4) - max(0.7,0.4) + 1 = 0.4 - 0.7 + 1 = 0.7
  expectCloseTo([0.7], [result[0]]);
});

test("blendReflect - f32", async () => {
  const src = `
     import lygia::color::blend::reflect::blendReflect;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendReflect(0.4, 0.5);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Reflect: 0.4^2 / (1-0.5) = 0.16/0.5 = 0.32
  expectCloseTo([0.32], [result[0]]);
});

test("blendSubtract - f32", async () => {
  const src = `
     import lygia::color::blend::subtract::blendSubtract;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendSubtract(0.8, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Subtract: max(0.8 + 0.3 - 1, 0) = max(0.1, 0) = 0.1
  expectCloseTo([0.1], [result[0]]);
});
