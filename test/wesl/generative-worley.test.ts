import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("worley2", async () => {
  const src = `
     import lygia::generative::worley::worley2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(1.5, 2.5); // Point in same cell

       let w1 = worley2(p1);
       let w2 = worley2(p2);
       let w3 = worley2(p3);

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different positions have different distances
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([0.7471], [result[0]]);
});

test("worley22", async () => {
  const src = `
     import lygia::generative::worley::worley22;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(3.0, 4.0); // Different point

       let w1 = worley22(p1);
       let w2 = worley22(p2);
       let w3 = worley22(p3);

       test::results[0] = vec4f(w1.x, w1.y, w2.x, w2.y);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Worley noise returns distances (F1, F2) in reasonable range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(1.5);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(1.5);
  // F1 should be less than or equal to F2 (closest point <= second closest)
  expect(result[0]).toBeLessThanOrEqual(result[1] + 0.001);
  // Regression: exact output value
  expectCloseTo([0.2529], [result[0]]);
});

test("worley3", async () => {
  const src = `
     import lygia::generative::worley::worley3;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point
       let p3 = vec3f(4.0, 5.0, 6.0); // Different point

       let w1 = worley3(p1);
       let w2 = worley3(p2);
       let w3 = worley3(p3);

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([0.3876], [result[0]]);
});

test("worley32", async () => {
  const src = `
     import lygia::generative::worley::worley32;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec3f(1.0, 2.0, 3.0);
       let p2 = vec3f(1.0, 2.0, 3.0); // Same point

       let w1 = worley32(p1);
       let w2 = worley32(p2);

       test::results[0] = vec4f(w1.x, w1.y, w2.x, w2.y);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0], result[1]], [result[2], result[3]]);
  // Worley noise returns distances (F1, F2) in reasonable range
  expect(result[0]).toBeGreaterThanOrEqual(0.0);
  expect(result[0]).toBeLessThanOrEqual(2.0);
  expect(result[1]).toBeGreaterThanOrEqual(0.0);
  expect(result[1]).toBeLessThanOrEqual(2.0);
  // F1 should be less than or equal to F2 (closest point <= second closest)
  expect(result[0]).toBeLessThanOrEqual(result[1] + 0.001);
  // Regression: exact output value
  expectCloseTo([0.6124], [result[0]]);
});
