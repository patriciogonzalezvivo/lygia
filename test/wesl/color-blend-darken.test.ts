import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("blendDarken3", async () => {
  const src = `
     import lygia::color::blend::darken::blendDarken3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.4, 0.5);
       let blend = vec3f(0.3, 0.7, 0.5);
       let result = blendDarken3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Darken mode: min(base, blend)
  expectCloseTo([0.3, 0.4, 0.5], result);
});

test("blendColorBurn3", async () => {
  const src = `
     import lygia::color::blend::colorBurn::blendColorBurn3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.5, 0.4);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendColorBurn3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Color burn mode: 1 - (1 - base) / blend
  // Relaxed precision for division-based blend mode
  expectCloseTo([0.0, 0.0, 0.0], result);
});

test("blendLinearBurn3", async () => {
  const src = `
     import lygia::color::blend::linearBurn::blendLinearBurn3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.6, 0.5, 0.7);
       let blend = vec3f(0.4, 0.3, 0.2);
       let result = blendLinearBurn3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Linear burn mode: max(base + blend - 1, 0)
  expectCloseTo([0.0, 0.0, 0.0], result);
});

test("blendDarken - f32", async () => {
  const src = `
     import lygia::color::blend::darken::blendDarken;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendDarken(0.6, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.3], [result[0]]);
});

test("blendColorBurn - f32", async () => {
  const src = `
     import lygia::color::blend::colorBurn::blendColorBurn;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendColorBurn(0.6, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Color burn: 1 - (1-0.6)/0.3 = 1 - 0.4/0.3 = 1 - 1.333 = clamped to 0
  // Relaxed precision for division-based blend mode
  expectCloseTo([0.0], [result[0]]);
});

test("blendLinearBurn - f32", async () => {
  const src = `
     import lygia::color::blend::linearBurn::blendLinearBurn;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendLinearBurn(0.6, 0.4);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Linear burn: max(0.6+0.4-1, 0) = 0
  expectCloseTo([0.0], [result[0]]);
});
