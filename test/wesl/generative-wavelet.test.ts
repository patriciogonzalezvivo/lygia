import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("noised2", async () => {
  const src = `
     import lygia::generative::noised::noised2;

     @compute @workgroup_size(1)
     fn foo() {
       let h = 0.001;
       let p = vec2f(1.0, 2.0);

       // Get noise and analytical derivatives
       let nd = noised2(p);
       let noise_val = nd.x;
       let dx_analytical = nd.y;
       let dy_analytical = nd.z;

       // Compute numerical derivatives via finite differences
       let dx_numerical = (noised2(p + vec2f(h, 0.0)).x - noised2(p - vec2f(h, 0.0)).x) / (2.0 * h);
       let dy_numerical = (noised2(p + vec2f(0.0, h)).x - noised2(p - vec2f(0.0, h)).x) / (2.0 * h);

       test::results[0] = vec4f(dx_analytical, dx_numerical, dy_analytical, dy_numerical);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test that analytical derivatives match numerical derivatives
  // Note: precision=2 allows for numerical differentiation error (finite differences introduce ~0.01 error)
  expectCloseTo([result[0], result[2]], [result[1], result[3]], 2);
  // Derivatives should be in reasonable range
  expect(Math.abs(result[0])).toBeLessThan(5.0);
  expect(Math.abs(result[2])).toBeLessThan(5.0);
  // Regression: exact output value
  expectCloseTo([-0.3648], [result[0]]);
});

test("noised3", async () => {
  const src = `
     import lygia::generative::noised::noised3;

     @compute @workgroup_size(1)
     fn foo() {
       let h = 0.001;
       let p = vec3f(1.0, 2.0, 3.0);

       // Get noise and analytical derivatives
       let nd = noised3(p);
       let dx_analytical = nd.y;
       let dy_analytical = nd.z;
       let dz_analytical = nd.w;

       // Compute numerical derivatives via finite differences
       let dx_numerical = (noised3(p + vec3f(h, 0.0, 0.0)).x - noised3(p - vec3f(h, 0.0, 0.0)).x) / (2.0 * h);
       let dy_numerical = (noised3(p + vec3f(0.0, h, 0.0)).x - noised3(p - vec3f(0.0, h, 0.0)).x) / (2.0 * h);
       let dz_numerical = (noised3(p + vec3f(0.0, 0.0, h)).x - noised3(p - vec3f(0.0, 0.0, h)).x) / (2.0 * h);

       // Pack all results into a single vec4f (we only have one result slot)
       test::results[0] = vec4f(dx_analytical, dx_numerical, dy_analytical, dy_numerical);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test that analytical derivatives match numerical derivatives
  // Note: precision=2 allows for numerical differentiation error (finite differences introduce ~0.01 error)
  expectCloseTo([result[0], result[2]], [result[1], result[3]], 2);
  // Derivatives should be in reasonable range
  expect(Math.abs(result[0])).toBeLessThan(5.0);
  expect(Math.abs(result[2])).toBeLessThan(5.0);
  // Regression: exact output value
  expectCloseTo([-153 / 256], [result[0]]);
});

test("wavelet2", async () => {
  const src = `
     import lygia::generative::wavelet::wavelet2;

     @compute @workgroup_size(1)
     fn foo() {
       let p1 = vec2f(1.0, 2.0);
       let p2 = vec2f(1.0, 2.0); // Same point
       let p3 = vec2f(3.0, 4.0); // Different point

       let w1 = wavelet2(p1);
       let w2 = wavelet2(p2);
       let w3 = wavelet2(p3);

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different inputs produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([-0.1946], [result[0]]);
});

test("wavelet3", async () => {
  const src = `
     import lygia::generative::wavelet::wavelet3;

     @compute @workgroup_size(1)
     fn foo() {
       // Third component is the phase parameter
       let p = vec2f(1.0, 2.0);
       let phase1 = 0.0;
       let phase2 = 1.0;

       let w1 = wavelet3(vec3f(p, phase1));
       let w2 = wavelet3(vec3f(p, phase2));
       let w3 = wavelet3(vec3f(p, phase1)); // Same as w1

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same position and phase produce same output
  expectCloseTo([result[0]], [result[2]]);
  // Test that different phases produce different outputs
  expect(result[0]).not.toBeCloseTo(result[1], 1);
  // Regression: exact output value
  expectCloseTo([-0.1946], [result[0]]);
});

test("waveletScaled2", async () => {
  const src = `
     import lygia::generative::wavelet::waveletScaled2;

     @compute @workgroup_size(1)
     fn foo() {
       let p = vec2f(1.0, 2.0);
       let phase = 0.5;

       // Test with same phase but different scales - not frequency
       let w1 = waveletScaled2(p, phase);
       let w2 = waveletScaled2(p, phase); // Same
       let w3 = waveletScaled2(p * 2.0, phase); // Different position

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that scaling position changes output
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([-0.1114], [result[0]]);
});

test("waveletScaled3 - with custom scale parameter", async () => {
  const src = `
     import lygia::generative::wavelet::waveletScaled3;

     @compute @workgroup_size(1)
     fn foo() {
       // waveletScaled3 takes vec3f(position.xy, phase) and scale parameter
       let p = vec2f(1.0, 2.0);
       let phase = 0.5;
       let scale1 = 1.0;
       let scale2 = 2.0;

       // Test with same position/phase but different scales
       let w1 = waveletScaled3(vec3f(p, phase), scale1);
       let w2 = waveletScaled3(vec3f(p, phase), scale1); // Same
       let w3 = waveletScaled3(vec3f(p, phase), scale2); // Different scale

       test::results[0] = vec4f(w1, w2, w3, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different scales produce different outputs
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Regression: exact output value
  expectCloseTo([0.0945], [result[0]]);
});

test("wavelet - base function with custom phase and scale", async () => {
  const src = `
     import lygia::generative::wavelet::wavelet;

     @compute @workgroup_size(1)
     fn foo() {
       let p = vec2f(1.0, 2.0);
       let phase = 0.5;
       let scale = 1.5;

       let w1 = wavelet(p, phase, scale);
       let w2 = wavelet(p, phase, scale); // Same
       let w3 = wavelet(p, 1.0, scale); // Different phase
       let w4 = wavelet(p, phase, 2.0); // Different scale

       test::results[0] = vec4f(w1, w2, w3, w4);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Test determinism: same input produces same output
  expectCloseTo([result[0]], [result[1]]);
  // Test that different phase produces different output
  expect(result[0]).not.toBeCloseTo(result[2], 1);
  // Test that different scale produces different output
  expect(result[0]).not.toBeCloseTo(result[3], 1);
  // Regression: exact output value
  expectCloseTo([0.1884], [result[0]]);
});
