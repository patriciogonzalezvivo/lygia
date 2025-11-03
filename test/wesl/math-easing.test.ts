import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("cubic", async () => {
  const src = `
    import lygia::math::cubic::cubic;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = vec4f(cubic(0.0), cubic(0.5), cubic(1.0), 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // cubic(0) = 0, cubic(0.5) = 0.5, cubic(1) = 1
  expectCloseTo([0.0, 0.5, 1.0, 0.0], result);
});

test("quartic", async () => {
  const src = `
    import lygia::math::quartic::quartic;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = vec4f(quartic(0.0), quartic(0.5), quartic(1.0), 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // quartic(v) = v*v*(2-v*v), quartic(0.5) = 0.25 * 1.9375 H 0.4375
  expectCloseTo([0.0, 0.4375, 1.0, 0.0], result);
});

test("quintic", async () => {
  const src = `
    import lygia::math::quintic::quintic;
    @compute @workgroup_size(1)
    fn foo() {
      test::results[0] = vec4f(quintic(0.0), quintic(0.5), quintic(1.0), 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // quintic(0) = 0, quintic(0.5) = 0.5, quintic(1) = 1
  expectCloseTo([0.0, 0.5, 1.0, 0.0], result);
});

test("invCubic", async () => {
  const src = `
    import lygia::math::invCubic::invCubic;
    import lygia::math::cubic::cubic;
    @compute @workgroup_size(1)
    fn foo() {
      // invCubic should be the inverse of cubic
      let x = 0.3;
      let y = cubic(x);
      let xRecovered = invCubic(y);
      test::results[0] = vec4f(x, xRecovered, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Original x and recovered x should match
  expectCloseTo([0.3, 0.3], result.slice(0, 2), 0.01);
});

test("invQuartic", async () => {
  const src = `
    import lygia::math::invQuartic::invQuartic;
    import lygia::math::quartic::quartic;
    @compute @workgroup_size(1)
    fn foo() {
      // invQuartic should be the inverse of quartic
      let x = 0.7;
      let y = quartic(x);
      let xRecovered = invQuartic(y);
      test::results[0] = vec4f(x, xRecovered, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Original x and recovered x should match
  expectCloseTo([0.7, 0.7], result.slice(0, 2), 0.01);
});

test("gain", async () => {
  const src = `
    import lygia::math::gain::gain;
    @compute @workgroup_size(1)
    fn foo() {
      // gain(0.5, k) should always equal 0.5
      test::results[0] = vec4f(gain(0.5, 2.0), gain(0.25, 2.0), gain(0.75, 2.0), 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // gain(0.5, k) = 0.5 always, gain applies symmetric contrast boost
  expect(result[0]).toBeCloseTo(0.5, 2); // midpoint invariant

  // Exact values for regression detection
  expectCloseTo([0.5, 0.125, 0.875, 0.0], result);
});

test("parabola", async () => {
  const src = `
    import lygia::math::parabola::parabola;
    @compute @workgroup_size(1)
    fn foo() {
      // Test k=1.0: basic parabola, symmetric values at 0.25 and 0.75
      test::results[0] = vec4f(parabola(0.0, 1.0), parabola(0.25, 1.0), parabola(0.5, 1.0), parabola(0.75, 1.0));

      // Test k=2.0: sharper curve (narrower peak)
      test::results[1] = vec4f(parabola(0.25, 2.0), parabola(0.5, 2.0), parabola(0.75, 2.0), 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f", size: 2 });

  // k=1.0: edges at 0, peak at 0.5, symmetric at 0.25/0.75
  expectCloseTo([0.0, 0.75, 1.0, 0.75], result.slice(0, 4));

  // k=2.0: sharper peak - values at 0.25/0.75 drop to 0.5625 (0.75^2)
  expect(result[4]).toBeCloseTo(0.5625, 2); // sharper curve
  expect(result[5]).toBeCloseTo(1.0, 2); // peak always at 0.5
  expect(result[4]).toBeLessThan(result[1]); // k=2 gives lower values than k=1

  // Exact values for regression detection
  expectCloseTo([0.0, 0.75, 1.0, 0.75, 0.5625, 1.0, 0.5625, 0.0], result);
});

test("gaussian", async () => {
  const src = `
    import lygia::math::gaussian::gaussian;
    @compute @workgroup_size(1)
    fn foo() {
      // gaussian(0, sigma) should be 1.0
      // gaussian(sigma, sigma) should be exp(-0.5) H 0.606
      test::results[0] = vec4f(gaussian(0.0, 1.0), gaussian(1.0, 1.0), 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.6065], result.slice(0, 2));
});
