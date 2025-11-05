import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("blendColorDodge3", async () => {
  const src = `
     import lygia::color::blend::colorDodge::blendColorDodge3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.4, 0.5);
       let result = blendColorDodge3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Color dodge mode: base / (1 - blend)
  expectCloseTo([0.571, 0.833, 1.0], result, 0.01);
});

test("blendLinearDodge3", async () => {
  const src = `
     import lygia::color::blend::linearDodge::blendLinearDodge3;

     @compute @workgroup_size(1)
     fn foo() {
       let base = vec3f(0.4, 0.5, 0.6);
       let blend = vec3f(0.3, 0.2, 0.1);
       let result = blendLinearDodge3(base, blend);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Linear dodge mode: min(base + blend, 1)
  expectCloseTo([0.7, 0.7, 0.7], result);
});

test("blendColorDodge - f32", async () => {
  const src = `
     import lygia::color::blend::colorDodge::blendColorDodge;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendColorDodge(0.4, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Color dodge: 0.4/(1-0.3) = 0.4/0.7 = 0.571
  expectCloseTo([0.571], [result[0]], 0.01);
});

test("blendLinearDodge - f32", async () => {
  const src = `
     import lygia::color::blend::linearDodge::blendLinearDodge;

     @compute @workgroup_size(1)
     fn foo() {
       let result = blendLinearDodge(0.4, 0.3);
       test::results[0] = vec4f(result, 0.0, 0.0, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.7], [result[0]]);
});
