import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("gamma2linear", async () => {
  const src = `
     import lygia::color::space::gamma2linear::gamma2linear3;

     @compute @workgroup_size(1)
     fn foo() {
       let gamma = vec3f(0.5, 0.5, 0.5);
       let result = gamma2linear3(gamma);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // pow(0.5, 2.2) ≈ 0.2181 (standard gamma 2.2)
  expectCloseTo([0.2176, 0.2176, 0.2176], result);
});

test("linear2gamma", async () => {
  const src = `
     import lygia::color::space::linear2gamma::linear2gamma3;

     @compute @workgroup_size(1)
     fn foo() {
       let linear = vec3f(0.25, 0.25, 0.25);
       let result = linear2gamma3(linear);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // pow(0.25, 1/2.2) ≈ 0.5277 (standard gamma 2.2)
  expectCloseTo([0.5325, 0.5325, 0.5325], result);
});

test("gamma2linear - f32 overload", async () => {
  const src = `
     import lygia::color::space::gamma2linear::gamma2linear;

     @compute @workgroup_size(1)
     fn foo() {
       let gamma = 0.5;
       let result = gamma2linear(gamma);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // pow(0.5, 2.2) ≈ 0.2181 (standard gamma 2.2)
  expectCloseTo([0.2176], result);
});

test("gamma2linear4 - vec4 with alpha preservation", async () => {
  const src = `
     import lygia::color::space::gamma2linear::gamma2linear4;

     @compute @workgroup_size(1)
     fn foo() {
       let gamma = vec4f(0.5, 0.5, 0.5, 0.7);
       let result = gamma2linear4(gamma);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // pow(0.5, 2.2) ≈ 0.2181 for RGB, alpha unchanged
  expectCloseTo([0.2176, 0.2176, 0.2176, 0.7], result);
});

test("linear2gamma - f32 overload", async () => {
  const src = `
     import lygia::color::space::linear2gamma::linear2gamma;

     @compute @workgroup_size(1)
     fn foo() {
       let linear = 0.25;
       let result = linear2gamma(linear);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // pow(0.25, 1/2.2) ≈ 0.5277 (standard gamma 2.2)
  expectCloseTo([0.5325], result);
});

test("linear2gamma4 - vec4 with alpha preservation", async () => {
  const src = `
     import lygia::color::space::linear2gamma::linear2gamma4;

     @compute @workgroup_size(1)
     fn foo() {
       let linear = vec4f(0.25, 0.25, 0.25, 0.4);
       let result = linear2gamma4(linear);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // pow(0.25, 1/2.2) ≈ 0.5277 for RGB, alpha unchanged
  expectCloseTo([0.5325, 0.5325, 0.5325, 0.4], result);
});
