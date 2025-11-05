import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("blendAdd3", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.5, 0.3, 0.2);
       let blend = vec3f(0.2, 0.4, 0.6);
       let result = blendAdd3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Add mode: min(base + blend, 1.0)
  expectCloseTo([0.7, 0.7, 0.8], result);
});

test("blendMultiply3", async () => {
  const src = `
     import lygia::color::blend::multiply::blendMultiply3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.8, 0.6, 0.4);
       let blend = vec3f(0.5, 0.5, 0.5);
       let result = blendMultiply3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Multiply mode: base * blend
  expectCloseTo([0.4, 0.3, 0.2], result);
});

test("blendScreen3", async () => {
  const src = `
     import lygia::color::blend::screen::blendScreen3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendScreen3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Screen mode: 1 - (1 - base) * (1 - blend)
  expectCloseTo([0.58, 0.7, 0.8], result);
});

test("blendAverage3", async () => {
  const src = `
     import lygia::color::blend::average::blendAverage3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.8);
       let blend = vec3f(0.4, 0.8, 0.2);
       let result = blendAverage3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Average mode: (base + blend) / 2
  expectCloseTo([0.5, 0.6, 0.5], result);
});

test("blendLighten3", async () => {
  const src = `
     import lygia::color::blend::lighten::blendLighten3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.5);
       let blend = vec3f(0.3, 0.7, 0.5);
       let result = blendLighten3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Lighten mode: max(base, blend)
  expectCloseTo([0.6, 0.7, 0.5], result);
});

test("blendAdd - f32", async () => {
  const src = `
     import lygia::color::blend::add::blendAdd;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendAdd(0.5, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.8], [result[0]]);
});

test("blendMultiply - f32", async () => {
  const src = `
     import lygia::color::blend::multiply::blendMultiply;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendMultiply(0.8, 0.5);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.4], [result[0]]);
});

test("blendScreen - f32", async () => {
  const src = `
     import lygia::color::blend::screen::blendScreen;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendScreen(0.4, 0.5);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Screen: 1 - (1-0.4)*(1-0.5) = 1 - 0.6*0.5 = 1 - 0.3 = 0.7
  expectCloseTo([0.7], [result[0]]);
});

test("blendAverage - f32", async () => {
  const src = `
     import lygia::color::blend::average::blendAverage;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendAverage(0.6, 0.4);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5], [result[0]]);
});

test("blendLighten - f32", async () => {
  const src = `
     import lygia::color::blend::lighten::blendLighten;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendLighten(0.6, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.6], [result[0]]);
});
