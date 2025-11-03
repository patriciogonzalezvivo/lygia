import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("saturate", async () => {
  const src = `
    import lygia::math::saturate::saturate;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = saturate(-0.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([0.0], result);
});

test("saturate clamped upper", async () => {
  const src = `
    import lygia::math::saturate::saturate;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = saturate(1.5); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([1.0], result);
});

test("saturate3", async () => {
  const src = `
    import lygia::math::saturate::saturate3;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = saturate3(vec3f(-0.5, 0.5, 1.5)); }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  expectCloseTo([0.0, 0.5, 1.0], result);
});

test("pow2", async () => {
  const src = `
    import lygia::math::pow2::pow2;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow2(3.0); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([9.0], result);
});

test("pow22", async () => {
  const src = `
    import lygia::math::pow2::pow22;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow22(vec2f(2.0, 3.0)); }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec2f" });
  expectCloseTo([4.0, 9.0], result);
});

test("pow3", async () => {
  const src = `
    import lygia::math::pow3::pow3;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow3(2.0); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([8.0], result);
});

test("pow5", async () => {
  const src = `
    import lygia::math::pow5::pow5;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow5(2.0); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([32.0], result);
});

test("pow7", async () => {
  const src = `
    import lygia::math::pow7::pow7;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = pow7(2.0); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([128.0], result);
});

test("absi", async () => {
  const src = `
    import lygia::math::absi::absi;
    @compute @workgroup_size(1)
    fn foo() {
      // Test both positive and negative values
      test::results[0] = vec4f(
        f32(absi(5)),   // Positive
        f32(absi(-5)),  // Negative
        f32(absi(0)),   // Zero
        f32(absi(-12))  // Larger negative
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([5.0, 5.0, 0.0, 12.0], result);
});

test("absi negative", async () => {
  const src = `
    import lygia::math::absi::absi;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = f32(absi(-5)); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([5.0], result);
});

// Anti-aliased floor tests (require derivatives, use fragment shaders)

test("cubicMix", async () => {
  const src = `
     import lygia::math::cubicMix::cubicMix;

     @compute @workgroup_size(1)
     fn foo() {
       // Test cubic hermite interpolation at multiple points
       test::results[0] = vec4f(
         cubicMix(0.0, 1.0, 0.0),   // Start: should be 0
         cubicMix(0.0, 1.0, 0.25),  // Quarter: smooth curve
         cubicMix(0.0, 1.0, 0.75),  // Three-quarters
         cubicMix(0.0, 1.0, 1.0)    // End: should be 1
       );
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Cubic hermite: 3t² - 2t³
  // t=0.25: 3(0.0625) - 2(0.015625) = 0.1875 - 0.03125 = 0.15625
  // t=0.75: 3(0.5625) - 2(0.421875) = 1.6875 - 0.84375 = 0.84375
  expectCloseTo([0.0, 0.15625, 0.84375, 1.0], result);
});

test("smootherstep", async () => {
  const src = `
     import lygia::math::smootherstep::smootherstep;

     @compute @workgroup_size(1)
     fn foo() {
       // Test smoother step (6t⁵ - 15t⁴ + 10t³) at multiple points
       test::results[0] = vec4f(
         smootherstep(0.0, 1.0, 0.0),   // Start: should be 0
         smootherstep(0.0, 1.0, 0.25),  // Quarter: smooth acceleration
         smootherstep(0.0, 1.0, 0.75),  // Three-quarters: smooth deceleration
         smootherstep(0.0, 1.0, 1.0)    // End: should be 1
       );
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Smootherstep: 6t⁵ - 15t⁴ + 10t³
  // t=0.25: 6(0.00098) - 15(0.00391) + 10(0.01563) = 0.00586 - 0.05859 + 0.15625 = 0.10352
  // t=0.75: 6(0.23730) - 15(0.31641) + 10(0.42188) = 1.42383 - 4.74609 + 4.21875 = 0.89648
  expectCloseTo([0.0, 0.10352, 0.89648, 1.0], result);
});

test("fmod2", async () => {
  const src = `
     import lygia::math::fmod::fmod2;

     @compute @workgroup_size(1)
     fn foo() {
       // Test positive values
       let result1 = fmod2(vec2f(5.0, 7.0), vec2f(3.0, 4.0));
       test::results[0] = result1.x;
       test::results[1] = result1.y;

       // Test negative values - key difference from % operator
       let result2 = fmod2(vec2f(-5.0, -7.0), vec2f(3.0, 4.0));
       test::results[2] = result2.x;
       test::results[3] = result2.y;
     }
   `;
  const result = await lygiaTestCompute(src);
  // fmod(5.0, 3.0) = 2.0, fmod(7.0, 4.0) = 3.0
  expectCloseTo([2.0, 3.0], result.slice(0, 2));
  // fmod(-5.0, 3.0) = 1.0, fmod(-7.0, 4.0) = 1.0 (floored, not truncated)
  expectCloseTo([1.0, 1.0], result.slice(2, 4));
});

test("fmod3", async () => {
  const src = `
     import lygia::math::fmod::fmod3;

     @compute @workgroup_size(1)
     fn foo() {
       // Test with negative values to verify floor-based behavior
       let result = fmod3(vec3f(-5.5, 7.3, -2.1), vec3f(3.0, 4.0, 2.0));
       test::results[0] = result.x;
       test::results[1] = result.y;
       test::results[2] = result.z;
     }
   `;
  const result = await lygiaTestCompute(src);
  // fmod(-5.5, 3.0) ≈ 0.5, fmod(7.3, 4.0) ≈ 3.3, fmod(-2.1, 2.0) ≈ 1.9
  expectCloseTo([0.5, 3.3, 1.9], result);
});

test("fmod4", async () => {
  const src = `
     import lygia::math::fmod::fmod4;

     @compute @workgroup_size(1)
     fn foo() {
       let result = fmod4(vec4f(10.0, -10.0, 7.5, -7.5), vec4f(3.0, 3.0, 2.5, 2.5));
       test::results[0] = result.x;
       test::results[1] = result.y;
       test::results[2] = result.z;
       test::results[3] = result.w;
     }
   `;
  const result = await lygiaTestCompute(src);
  // fmod(10.0, 3.0) = 1.0, fmod(-10.0, 3.0) = 2.0, fmod(7.5, 2.5) = 0.0, fmod(-7.5, 2.5) = 0.0
  expectCloseTo([1.0, 2.0, 0.0, 0.0], result);
});

test("map - remap value between ranges", async () => {
  const src = `
    import lygia::math::map::map;
    @compute @workgroup_size(1)
    fn foo() {
      // Map 0.5 from [0,1] to [0,100]
      let result1 = map(0.5, 0.0, 1.0, 0.0, 100.0);
      // Map 5.0 from [0,10] to [100,200]
      let result2 = map(5.0, 0.0, 10.0, 100.0, 200.0);
      test::results[0] = vec4f(result1, result2, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([50.0, 150.0, 0.0, 0.0], result);
});

test("mirror - triangle wave", async () => {
  const src = `
    import lygia::math::mirror::mirror;
    @compute @workgroup_size(1)
    fn foo() {
      // mirror creates triangle wave: 0→1→0→1→0
      test::results[0] = vec4f(mirror(0.5), mirror(1.5), mirror(2.5), mirror(3.5));
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5, 0.5, 0.5, 0.5], result);
});

test("decimate - quantize value", async () => {
  const src = `
    import lygia::math::decimate::decimate;
    @compute @workgroup_size(1)
    fn foo() {
      // Decimate to 10 levels
      let result = decimate(0.567, 10.0);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src);
  // 0.567 * 10 = 5.67, floor = 5, 5/10 = 0.5
  expectCloseTo([0.5], result);
});

// Utility functions

test("taylorInvSqrt", async () => {
  const src = `
    import lygia::math::taylorInvSqrt::taylorInvSqrt;
    @compute @workgroup_size(1)
    fn foo() {
      // Test Taylor series approximation: 1.79284 - 0.85373 * r
      // This is a first-order approximation, accurate near r=1
      test::results[0] = vec4f(
        taylorInvSqrt(1.0),   // 1.793 - 0.854 * 1.0 = 0.939
        taylorInvSqrt(4.0),   // 1.793 - 0.854 * 4.0 = -1.622
        taylorInvSqrt(0.25),  // 1.793 - 0.854 * 0.25 = 1.579
        taylorInvSqrt(2.0)    // 1.793 - 0.854 * 2.0 = 0.085
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Exact values from the linear approximation formula
  expectCloseTo([0.9391, -1.6221, 1.5794, 0.0854], result);
});

// Anti-aliased functions (require derivatives, use fragment shaders)
test("adaptiveThreshold", async () => {
  const src = `
    import lygia::math::adaptiveThreshold::adaptiveThreshold;
    @compute @workgroup_size(1)
    fn foo() {
      // Test threshold comparison
      let result1 = adaptiveThreshold(0.8, 0.5, 0.1); // v > blur_v + b
      let result2 = adaptiveThreshold(0.4, 0.5, 0.1); // v < blur_v + b
      test::results[0] = vec4f(result1, result2, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0, 0.0, 0.0], result);
});

test("atan2Custom", async () => {
  const src = `
    import lygia::math::atan2::atan2Custom;
    import lygia::math::consts::PI;
    import lygia::math::consts::TAU;
    @compute @workgroup_size(1)
    fn foo() {
      // atan2Custom normalizes angles to [0, 2π] range
      // Formula: (atan2(y, x) + PI) % TAU
      // Note: This shifts by π, making atan2 output [0, 2π] instead of [-π, π]

      // Test common angles
      let angle1 = atan2Custom(1.0, 0.0);   // atan2(1,0) = π/2, +PI = 3π/2
      let angle2 = atan2Custom(0.0, 1.0);   // atan2(0,1) = 0, +PI = π
      let angle3 = atan2Custom(-1.0, 0.0);  // atan2(-1,0) = -π/2, +PI = π/2
      let angle4 = atan2Custom(0.0, -1.0);  // atan2(0,-1) = π, +PI = 2π % 2π = 0

      test::results[0] = vec4f(angle1, angle2, angle3, angle4);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  const PI = Math.PI;
  const TAU = 2 * PI;

  // Verify normalized angles [0, 2π]
  expectCloseTo([(3 * PI) / 2], [result[0]]); // 3π/2 ≈ 4.7124
  expectCloseTo([PI], [result[1]]); // π ≈ 3.1416
  expectCloseTo([PI / 2], [result[2]]); // π/2 ≈ 1.5708
  expectCloseTo([0.0], [result[3]]); // 0

  // All angles should be in [0, 2π) range
  for (let i = 0; i < 4; i++) {
    expect(result[i]).toBeGreaterThanOrEqual(0.0);
    expect(result[i]).toBeLessThan(TAU);
  }
});

test("atan2Custom - additional angles", async () => {
  const src = `
    import lygia::math::atan2::atan2Custom;
    import lygia::math::consts::PI;
    @compute @workgroup_size(1)
    fn foo() {
      // Test additional angle cases
      let angle5 = atan2Custom(1.0, 1.0);   // atan2(1,1) = π/4, +PI = 5π/4

      test::results[0] = vec4f(angle5, 0.0, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  const PI = Math.PI;

  expectCloseTo([(5 * PI) / 4], [result[0]]); // 5π/4 ≈ 3.927
});

test("bump", async () => {
  const src = `
    import lygia::math::bump::bump;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = bump(0.0, 0.0); // Should be 1.0
      let result2 = bump(1.0, 0.0); // Should be 0.0
      let result3 = bump(0.5, 0.0); // Should be 0.75
      test::results[0] = vec4f(result1, result2, result3, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0, 0.75], result.slice(0, 3));
});

test("bump2", async () => {
  const src = `
    import lygia::math::bump::bump2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = bump2(vec2f(0.0, 0.5), vec2f(0.0));
      test::results[0] = vec4f(result.x, result.y, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.75], result.slice(0, 2));
});

test("highPass", async () => {
  const src = `
    import lygia::math::highPass::highPass;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = highPass(0.8, 0.5); // Above threshold
      let result2 = highPass(0.3, 0.5); // Below threshold
      test::results[0] = vec4f(result1, result2, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.6, 0.0], result.slice(0, 2));
});

test("inside - scalar", async () => {
  const src = `
    import lygia::math::inside::inside;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = inside(5.0, 0.0, 10.0); // true
      let result2 = inside(-1.0, 0.0, 10.0); // false
      let result3 = inside(11.0, 0.0, 10.0); // false
      test::results[0] = vec4f(f32(result1), f32(result2), f32(result3), 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0, 0.0], result.slice(0, 3));
});

test("inside2", async () => {
  const src = `
    import lygia::math::inside::inside2;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = inside2(vec2f(5.0, 5.0), vec2f(0.0), vec2f(10.0)); // true
      let result2 = inside2(vec2f(-1.0, 5.0), vec2f(0.0), vec2f(10.0)); // false
      test::results[0] = vec4f(f32(result1), f32(result2), 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0], result.slice(0, 2));
});

test("mod2 - mutates pointer", async () => {
  const src = `
    import lygia::math::mod2::mod2;
    @compute @workgroup_size(1)
    fn foo() {
      var p = vec2f(7.0, 10.0);
      let c = mod2(&p, 3.0);
      test::results[0] = vec4f(p.x, p.y, c.x, c.y);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // p should be modified to centered remainder, c is the cell index
  expect(result[0]).toBeCloseTo(1.0, 1);
  expect(result[1]).toBeCloseTo(1.0, 1);
});

test("mod289", async () => {
  const src = `
    import lygia::math::mod289::mod289;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = mod289(300.0); // 300 % 289 = 11
      let result2 = mod289(289.0); // 289 % 289 = 0
      let result3 = mod289(100.0); // 100 % 289 = 100
      test::results[0] = vec4f(result1, result2, result3, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([11.0, 0.0, 100.0], result.slice(0, 3));
});

test("powFast", async () => {
  const src = `
    import lygia::math::powFast::powFast;
    @compute @workgroup_size(1)
    fn foo() {
      // powFast is a fast approximation: powFast(a, b) = a / ((1-b)*a + b)
      // This approximates pow(a, b) but is not exact

      // Test case 1: powFast(0.5, 0.5)
      // = 0.5 / ((1-0.5)*0.5 + 0.5)
      // = 0.5 / (0.5*0.5 + 0.5)
      // = 0.5 / (0.25 + 0.5)
      // = 0.5 / 0.75 = 0.6667
      // (True pow(0.5, 0.5) = 0.7071)
      let fast1 = powFast(0.5, 0.5);

      // Test case 2: powFast(0.8, 0.3)
      // = 0.8 / ((1-0.3)*0.8 + 0.3)
      // = 0.8 / (0.7*0.8 + 0.3)
      // = 0.8 / (0.56 + 0.3)
      // = 0.8 / 0.86 = 0.9302
      // (True pow(0.8, 0.3) = 0.9438)
      let fast2 = powFast(0.8, 0.3);

      // Test case 3: powFast(0.25, 0.75)
      // = 0.25 / ((1-0.75)*0.25 + 0.75)
      // = 0.25 / (0.25*0.25 + 0.75)
      // = 0.25 / (0.0625 + 0.75)
      // = 0.25 / 0.8125 = 0.3077
      // (True pow(0.25, 0.75) = 0.3536)
      let fast3 = powFast(0.25, 0.75);

      // Test edge case: powFast(1.0, x) should always be 1.0
      let edge1 = powFast(1.0, 0.5);

      test::results[0] = vec4f(fast1, fast2, fast3, edge1);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Verify approximation values (not exact pow, but close)
  expectCloseTo([0.6667], [result[0]]);
  expectCloseTo([0.9302], [result[1]]);
  expectCloseTo([0.3077], [result[2]]);

  // Edge case: powFast(1, x) = 1 for any x
  expectCloseTo([1.0], [result[3]]);
});

test("round", async () => {
  const src = `
    import lygia::math::round::round;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = round(2.3);
      let result2 = round(2.7);
      let result3 = round(-2.3);
      let result4 = round(-2.7);
      test::results[0] = vec4f(result1, result2, result3, result4);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([2.0, 3.0, -2.0, -3.0], result);
});

test("saturateMediump", async () => {
  const src = `
    import lygia::math::saturateMediump::saturateMediump;
    @compute @workgroup_size(1)
    fn foo() {
      // saturateMediump clamps to MEDIUMP_FLT_MAX (65504.0) on mobile
      // On desktop (TARGET_MOBILE=false), it's a pass-through
      // IMPORTANT: It does NOT clamp the lower bound to 0!

      let v1 = saturateMediump(-0.5);      // Negative passes through
      let v2 = saturateMediump(0.5);       // Normal value passes through
      let v3 = saturateMediump(1000.0);    // Below limit passes through
      let v4 = saturateMediump(100000.0);  // Above limit: desktop passes, mobile clamps to 65504

      test::results[0] = vec4f(v1, v2, v3, v4);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Test specific behavior
  expectCloseTo([-0.5], [result[0]]);
  expectCloseTo([0.5], [result[1]]);
  expectCloseTo([1000.0], [result[2]]);

  // v4: On desktop should be 100000, on mobile should be clamped to 65504
  // Test that it's either original or clamped (platform-dependent)
  const MEDIUMP_FLT_MAX = 65504.0;
  const isDesktop = Math.abs(result[3] - 100000.0) < 0.01;
  const isMobile = Math.abs(result[3] - MEDIUMP_FLT_MAX) < 0.01;
  expect(isDesktop || isMobile).toBe(true);
});

test("sum2", async () => {
  const src = `
    import lygia::math::sum::sum2;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple cases including negative values
      test::results[0] = vec4f(
        sum2(vec2f(3.0, 7.0)),      // Positive
        sum2(vec2f(-5.0, 8.0)),     // Mixed
        sum2(vec2f(-2.0, -3.0)),    // Negative
        sum2(vec2f(0.5, 0.25))      // Fractional
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([10.0, 3.0, -5.0, 0.75], result);
});

test("sum3", async () => {
  const src = `
    import lygia::math::sum::sum3;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple cases including negative and fractional values
      test::results[0] = vec4f(
        sum3(vec3f(3.0, 7.0, 5.0)),     // Positive
        sum3(vec3f(-2.0, 6.0, -1.0)),   // Mixed signs
        sum3(vec3f(-1.0, -2.0, -3.0)),  // All negative
        sum3(vec3f(0.25, 0.5, 0.75))    // Fractional
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([15.0, 3.0, -6.0, 1.5], result);
});

test("within - scalar", async () => {
  const src = `
    import lygia::math::within::within;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = within(5.0, 0.0, 10.0); // true -> 1.0
      let result2 = within(-1.0, 0.0, 10.0); // false -> 0.0
      let result3 = within(11.0, 0.0, 10.0); // false -> 0.0
      test::results[0] = vec4f(result1, result2, result3, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0, 0.0], result.slice(0, 3));
});

test("within2", async () => {
  const src = `
    import lygia::math::within::within2;
    @compute @workgroup_size(1)
    fn foo() {
      let result1 = within2(vec2f(5.0, 5.0), vec2f(0.0), vec2f(10.0)); // true -> 1.0
      let result2 = within2(vec2f(-1.0, 5.0), vec2f(0.0), vec2f(10.0)); // false -> 0.0
      test::results[0] = vec4f(result1, result2, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 0.0], result.slice(0, 2));
});
