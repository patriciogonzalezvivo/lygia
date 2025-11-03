import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("hammersley", async () => {
  const src = `
    import lygia::math::hammersley::hammersley;
    @compute @workgroup_size(1)
    fn foo() {
      // Test known Hammersley sequence values
      // hammersley(i, N) = (i/N, radicalInverse_VdC(i))
      // For radicalInverse_VdC:
      //   0 -> 0.0
      //   1 -> 0.5 (binary 1 -> reversed -> 0.1 binary = 0.5)
      //   2 -> 0.25 (binary 10 -> reversed -> 0.01 binary = 0.25)
      //   3 -> 0.75 (binary 11 -> reversed -> 0.11 binary = 0.75)
      let h0 = hammersley(0u, 8);  // (0/8, 0.0) = (0.0, 0.0)
      let h1 = hammersley(1u, 8);  // (1/8, 0.5) = (0.125, 0.5)
      let h2 = hammersley(2u, 8);  // (2/8, 0.25) = (0.25, 0.25)
      let h3 = hammersley(3u, 8);  // (3/8, 0.75) = (0.375, 0.75)

      // Pack first two into result
      test::results[0] = vec4f(h0.x, h0.y, h1.x, h1.y);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify linear x component: h0.x = 0.0, h1.x = 0.125
  expectCloseTo([0.0, 0.125], [result[0], result[2]]);

  // Verify radical inverse y component (bit reversal): h0.y = 0.0, h1.y = 0.5
  expectCloseTo([0.0, 0.5], [result[1], result[3]]);

  // Exact values to catch regressions
  expectCloseTo([0.0, 0.0, 0.125, 0.5], result);
});

test("hammersley - bit reversal verification", async () => {
  const src = `
    import lygia::math::hammersley::hammersley;
    @compute @workgroup_size(1)
    fn foo() {
      // Test more bit reversal values
      let h2 = hammersley(2u, 8);  // (2/8, 0.25) = (0.25, 0.25)
      let h3 = hammersley(3u, 8);  // (3/8, 0.75) = (0.375, 0.75)

      test::results[0] = vec4f(h2.x, h2.y, h3.x, h3.y);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify linear x component: h2.x = 0.25, h3.x = 0.375
  expectCloseTo([0.25, 0.375], [result[0], result[2]]);

  // Verify radical inverse y component: h2.y = 0.25, h3.y = 0.75
  expectCloseTo([0.25, 0.75], [result[1], result[3]]);

  // Exact values to catch regressions
  expectCloseTo([0.25, 0.25, 0.375, 0.75], result);
});

test("nyquist", async () => {
  const src = `
    import lygia::math::nyquist::nyquist;
    @compute @workgroup_size(1)
    fn foo() {
      // nyquist filters based on frequency width:
      // - width < 0.5: no filtering (pass-through)
      // - width > 0.5: filters toward 0.5 (Nyquist limit)
      // The function attenuates signals based on how far they are from 0.5

      // Test 1: No filtering (width well below Nyquist)
      let v1 = nyquist(0.8, 0.1);  // width < 0.5, should pass through

      // Test 2: Partial filtering (width near Nyquist)
      let v2 = nyquist(0.8, 0.5);  // At Nyquist limit

      // Test 3: Heavy filtering (width above Nyquist)
      let v3 = nyquist(0.8, 1.0);  // Full filtering

      // Test 4: Value at midpoint (should always stay at 0.5)
      let v4 = nyquist(0.5, 0.8);  // 0.5 is unaffected by filtering

      test::results[0] = vec4f(v1, v2, v3, v4);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // 0 - No filtering: should pass through (0.8)
  expectCloseTo([0.8], [result[0]]);

  // 1 - Partial filtering: should be between 0.5 and 0.8
  expect(result[1]).toBeGreaterThan(0.5);
  expect(result[1]).toBeLessThan(0.8);

  // 2 - Heavy filtering: should be very close to 0.5 (heavily attenuated)
  expectCloseTo([0.5], [result[2]]);

  // 3 - Midpoint invariant: 0.5 should remain 0.5
  expectCloseTo([0.5], [result[3]]);

  expectCloseTo([0.8, 0.65, 0.5, 0.5], result);
});

test("permute", async () => {
  const src = `
     import lygia::math::permute::permute;

     @compute @workgroup_size(1)
     fn foo() {
       // permute(x) = mod289(((x * 34.0) + 1.0) * x)
       // Test with hand-calculated values:

       // permute(1.0) = mod289(((1 * 34) + 1) * 1) = mod289(35) = 35
       let p1 = permute(1.0);

       // permute(10.0) = mod289(((10 * 34) + 1) * 10) = mod289(3410) = 3410 % 289 = 231
       let p2 = permute(10.0);

       // permute(100.0) = mod289(((100 * 34) + 1) * 100) = mod289(340100) = 340100 % 289 = 236
       let p3 = permute(100.0);

       // Verify reproducibility
       let p1_repeat = permute(1.0);

       test::results[0] = vec4f(p1, p2, p3, p1_repeat);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  expectCloseTo([35.0, 231.0, 236.0, 35.0], result);
});

test("grad4 - noise gradient helper", async () => {
  const src = `
    import lygia::math::grad4::grad4;
    @compute @workgroup_size(1)
    fn foo() {
      // grad4 computes gradient vectors for 4D noise
      // It uses permutation value j and position p to generate gradients

      // Test 1: Reproducibility - same inputs give same output
      let g1a = grad4(123.456, vec4f(0.789, 0.234, 0.567, 0.891));
      let g1b = grad4(123.456, vec4f(0.789, 0.234, 0.567, 0.891));

      // Test 2: Different j values produce different gradients
      let g2 = grad4(42.0, vec4f(0.789, 0.234, 0.567, 0.891));

      // Test 3: Different positions produce different gradients
      let g3 = grad4(123.456, vec4f(0.33, 0.67, 0.89, 1.23));

      // Store first gradient for validation
      test::results[0] = g1a;
      test::results[1] = g1b;
      test::results[2] = g2;
      test::results[3] = g3;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f", size: 4 });

  const g1a = result.slice(0, 4);
  const g1b = result.slice(4, 8);
  const g2 = result.slice(8, 12);
  const g3 = result.slice(12, 16);

  // Property 1: Reproducibility - same inputs produce same output
  expectCloseTo(g1a, g1b);

  // Property 2: Different inputs should produce different gradients
  // (Note: hash collisions are possible, but unlikely with these specific inputs)
  expect(g1a).not.toEqual(g2);
  expect(g1a).not.toEqual(g3);

  // Exact values to catch regressions
  expectCloseTo([-0.866, 1.402, 1.402, -3.438], g1a);
});

test("grad4 - gradient range validation", async () => {
  const src = `
    import lygia::math::grad4::grad4;
    @compute @workgroup_size(1)
    fn foo() {
      // Test gradient at origin and other positions
      let g1 = grad4(50.0, vec4f(0.0, 0.0, 0.0, 0.0));
      let g2 = grad4(75.0, vec4f(0.2, 0.3, 0.4, 0.5));

      // Store y and z components for validation
      test::results[0] = vec4f(g1.y, g1.z, g2.y, g2.z);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // All gradient components should be in reasonable range
  for (let i = 0; i < 4; i++) {
    expect(Math.abs(result[i])).toBeLessThan(3.0);
  }

  // Exact values to catch regressions
  expectCloseTo([0.0, 0.0, -0.8, 0.0], result);
});

test("hemisphereCosSample - unit vector property", async () => {
  const src = `
    import lygia::math::hammersley::hemisphereCosSample;
    @compute @workgroup_size(1)
    fn foo() {
      // Test that output is a unit vector (or near unit length)
      let v1 = hemisphereCosSample(vec2f(0.0, 0.0));
      let v2 = hemisphereCosSample(vec2f(1.0, 1.0));
      let v3 = hemisphereCosSample(vec2f(0.5, 0.5));

      // Calculate lengths
      let len1 = length(v1);
      let len2 = length(v2);
      let len3 = length(v3);

      test::results[0] = vec4f(len1, len2, len3, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // All outputs should be unit vectors (length = 1.0)
  expectCloseTo([1.0, 1.0, 1.0], result.slice(0, 3));

  // Exact values to catch regressions
  expectCloseTo([1.0, 1.0, 1.0, 0.0], result);
});

test("hemisphereCosSample - positive hemisphere", async () => {
  const src = `
    import lygia::math::hammersley::hemisphereCosSample;
    @compute @workgroup_size(1)
    fn foo() {
      // Test that all outputs point into positive hemisphere (z >= 0)
      let v1 = hemisphereCosSample(vec2f(0.0, 0.0));
      let v2 = hemisphereCosSample(vec2f(1.0, 1.0));
      let v3 = hemisphereCosSample(vec2f(0.5, 0.5));
      let v4 = hemisphereCosSample(vec2f(0.25, 0.75));

      test::results[0] = vec4f(v1.z, v2.z, v3.z, v4.z);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Exact values to catch regressions
  expectCloseTo([1.0, 0.0, Math.SQRT1_2, 0.5], result);
});

test("hemisphereCosSample - known values", async () => {
  const src = `
    import lygia::math::hammersley::hemisphereCosSample;
    @compute @workgroup_size(1)
    fn foo() {
      // Test specific known values
      // u=(0,0): phi=0, cosTheta2=1, cosTheta=1, sinTheta=0  (0, 0, 1)
      let v1 = hemisphereCosSample(vec2f(0.0, 0.0));

      // u=(0,1): phi=0, cosTheta2=0, cosTheta=0, sinTheta=1  (1, 0, 0)
      let v2 = hemisphereCosSample(vec2f(0.0, 1.0));

      // Store z component from v1 and full v2
      test::results[0] = vec4f(v1.z, v2.x, v2.y, v2.z);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // u=(0,0) should give (0, 0, 1) - pointing straight up (testing z component)
  expectCloseTo([1.0], [result[0]]);

  // u=(0,1) should give (cos(0)*1, sin(0)*1, 0) = (1, 0, 0)
  expectCloseTo([1.0, 0.0, 0.0], result.slice(1, 4));

  // Exact values to catch regressions
  expectCloseTo([1.0, 1.0, 0.0, 0.0], result);
});

test("hemisphereCosSample - cosine distribution", async () => {
  const src = `
    import lygia::math::hammersley::hemisphereCosSample;
    @compute @workgroup_size(1)
    fn foo() {
      // Test that the distribution is cosine-weighted
      // Higher u.y values should produce samples closer to the horizon (smaller z)
      let v_low = hemisphereCosSample(vec2f(0.5, 0.1));   // Low u.y  higher z
      let v_mid = hemisphereCosSample(vec2f(0.5, 0.5));   // Mid u.y  mid z
      let v_high = hemisphereCosSample(vec2f(0.5, 0.9));  // High u.y  lower z

      test::results[0] = vec4f(v_low.z, v_mid.z, v_high.z, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Verify ordering: z decreases as u.y increases (cosine distribution property)
  expect(result[0]).toBeGreaterThan(result[1]); // low u.y has higher z than mid u.y
  expect(result[1]).toBeGreaterThan(result[2]); // mid u.y has higher z than high u.y

  // Exact values to catch regressions
  const invSqrt10 = 1 / Math.sqrt(10); // ≈ 0.31623
  expectCloseTo([0.94868, Math.SQRT1_2, invSqrt10, 0.0], result);
});
