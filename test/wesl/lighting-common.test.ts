import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("GGX", async () => {
  const src = `
     import lygia::lighting::common::ggx::GGX;

     @compute @workgroup_size(1)
     fn foo() {
       // GGX(NoH: f32, roughness: f32) -> f32
       // Test GGX distribution at perfect alignment (NoH=1.0) with roughness=0.5
       let NoH = 1.0;  // Perfect alignment between normal and half vector
       let roughness = 0.5;
       let result1 = GGX(NoH, roughness);

       // GGX should peak at NoH=1.0 and decrease as NoH decreases
       let result2 = GGX(0.8, roughness);

       // Lower roughness should produce sharper peak (higher value at NoH=1)
       let result3 = GGX(1.0, 0.1);

       test::results[0] = vec3f(result1, result2, result3);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // GGX distribution peaks at NoH=1.0
  expect(result[0]).toBeGreaterThan(0.3); // Peak value with roughness=0.5
  // GGX decreases as NoH decreases
  expect(result[1]).toBeLessThan(result[0]);
  // Lower roughness produces sharper, higher peak
  expect(result[2]).toBeGreaterThan(result[0]);
  // Exact values to catch regressions
  expectCloseTo([1.2732, 0.2943, 31.831], result.slice(0, 3));
});

test("GGXPrecise", async () => {
  const src = `
     import lygia::lighting::common::ggx::GGX;
     import lygia::lighting::common::ggx::GGXPrecise;

     @compute @workgroup_size(1)
     fn foo() {
       // GGXPrecise uses Lagrange's identity on TARGET_MOBILE for better mediump precision
       // It should match standard GGX but handle edge cases better
       let N = vec3f(0.0, 0.0, 1.0);
       let H = vec3f(0.0, 0.0, 1.0);  // Perfect alignment
       let NoH = dot(N, H);
       let roughness = 0.5;

       // Compare GGXPrecise with standard GGX
       let preciseFn = GGXPrecise(N, H, NoH, roughness);
       let standardFn = GGX(NoH, roughness);

       test::results[0] = vec2f(preciseFn, standardFn);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec2f" });
  // GGXPrecise should produce similar results to standard GGX (identical on desktop)
  expectCloseTo([result[0]], [result[1]]);
  // Both should be positive and reasonable
  expect(result[0]).toBeGreaterThan(0.3);
  // Exact values to catch regressions
  expectCloseTo([1.2732, 1.2732], result.slice(0, 2));
});

test("importanceSamplingGGX", async () => {
  const src = `
     import lygia::lighting::common::ggx::importanceSamplingGGX;

     @compute @workgroup_size(1)
     fn foo() {
       // importanceSamplingGGX generates direction in tangent space
       // With u=(0,0), should produce direction toward Z (cosTheta=1)
       let sample1 = importanceSamplingGGX(vec2f(0.0, 0.0), 0.5);

       // With u=(0.5,0.5), should produce mid-range direction
       let sample2 = importanceSamplingGGX(vec2f(0.5, 0.5), 0.5);

       // Lower roughness should bias samples toward Z axis
       let sample3 = importanceSamplingGGX(vec2f(0.5, 0.5), 0.1);

       test::results[0] = vec3f(sample1.z, length(sample2), sample3.z);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // u=(0,0) produces direction close to Z axis
  expectCloseTo([1.0], [result[0]]);
  // Samples should be normalized (unit length)
  expectCloseTo([1.0], [result[1]]);
  // Lower roughness biases toward Z
  expect(result[2]).toBeGreaterThan(0.7);
  // Exact values to catch regressions
  expectCloseTo([1.0, 1.0, 0.995], result.slice(0, 3));
});

test("schlick vec3f", async () => {
  const src = `
     import lygia::lighting::common::schlick::schlick;

     @compute @workgroup_size(1)
     fn foo() {
       // Schlick: f0 + (f90-f0)*(1-VoH)^5
       let f0 = vec3f(0.04, 0.04, 0.04);  // Dielectric base reflectivity
       let f90 = 1.0;  // Grazing angle reflectivity

       // At normal incidence (VoH=1.0), should equal f0
       let normalFn = schlick(f0, f90, 1.0);

       // At grazing angle (VoH=0.0), should equal f90
       let grazingFn = schlick(f0, f90, 0.0);

       // At mid-angle, should be between f0 and f90
       let midFn = schlick(f0, f90, 0.5);

       test::results[0] = vec3f(normalFn.x, grazingFn.x, midFn.x);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // At normal incidence, Fresnel equals f0
  expectCloseTo([0.04], [result[0]]);
  // At grazing angle, Fresnel approaches f90
  expectCloseTo([1.0], [result[1]]);
  // Mid-angle should be between f0 and f90
  expect(result[2]).toBeGreaterThan(0.04);
  expect(result[2]).toBeLessThan(1.0);
  // Exact values to catch regressions
  expectCloseTo([0.04, 1.0, 0.07], result.slice(0, 3));
});

test("schlickVec3", async () => {
  const src = `
     import lygia::lighting::common::schlick::schlickVec3;

     @compute @workgroup_size(1)
     fn foo() {
       // schlickVec3 with vec3f f90 for colored metals
       let f0 = vec3f(1.0, 0.71, 0.29);  // Gold-like base reflectivity
       let f90 = vec3f(1.0, 0.95, 0.9);  // Colored grazing reflectivity

       // At normal incidence, should equal f0
       let normalFn = schlickVec3(f0, f90, 1.0);

       // At grazing angle, should approach f90
       let grazingFn = schlickVec3(f0, f90, 0.0);

       test::results[0] = vec4f(normalFn, grazingFn.x);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // At normal incidence, equals f0
  expectCloseTo([1.0, 0.71, 0.29], [result[0], result[1], result[2]]);
  // At grazing angle, approaches f90
  expect(result[3]).toBeGreaterThan(0.85);
  // Exact values to catch regressions
  expectCloseTo([1.0, 0.71, 0.29, 1.0], result);
});

test("schlickF32", async () => {
  const src = `
     import lygia::lighting::common::schlick::schlickF32;

     @compute @workgroup_size(1)
     fn foo() {
       // schlickF32 scalar version
       let f0 = 0.04;
       let f90 = 1.0;

       // Test Schlick formula: f0 + (f90-f0)*(1-VoH)^5
       let normalFn = schlickF32(f0, f90, 1.0);   // VoH=1 -> f0
       let grazingFn = schlickF32(f0, f90, 0.0);  // VoH=0 -> f90
       let midFn = schlickF32(f0, f90, 0.5);      // VoH=0.5 -> interpolated

       test::results[0] = vec3f(normalFn, grazingFn, midFn);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // At normal incidence, equals f0
  expectCloseTo([0.04], [result[0]]);
  // At grazing, approaches f90
  expectCloseTo([1.0], [result[1]]);
  // Mid-angle between f0 and f90
  expect(result[2]).toBeGreaterThan(0.04);
  expect(result[2]).toBeLessThan(1.0);
  // Exact values to catch regressions
  expectCloseTo([0.04, 1.0, 0.07], result);
});

test("smithGGXCorrelated", async () => {
  const src = `
     import lygia::lighting::common::smithGGXCorrelated::smithGGXCorrelated;

     @compute @workgroup_size(1)
     fn foo() {
       // Smith GGX visibility term - increases with smoother surfaces
       let NoV = 0.8;
       let NoL = 0.7;

       // Higher roughness should reduce visibility (more masking/shadowing)
       let smoothFn = smithGGXCorrelated(NoV, NoL, 0.1);
       let roughFn = smithGGXCorrelated(NoV, NoL, 0.9);

       // Perfect alignment (NoV=1, NoL=1) should have high visibility
       let perfectFn = smithGGXCorrelated(1.0, 1.0, 0.5);

       test::results[0] = vec3f(smoothFn, roughFn, perfectFn);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Smooth surfaces have higher visibility
  expect(result[0]).toBeGreaterThan(result[1]);
  // Rough surfaces still have positive visibility
  expect(result[1]).toBeGreaterThan(0.0);
  // Perfect alignment has good visibility (visibility term is clamped)
  expect(result[2]).toBeGreaterThan(0.2);
  // Exact values to catch regressions
  expectCloseTo([0.4447, 0.3482, 0.25], result.slice(0, 3));
});

test("smithGGXCorrelated_Fast", async () => {
  const src = `
     import lygia::lighting::common::smithGGXCorrelated::{smithGGXCorrelated, smithGGXCorrelated_Fast};

     @compute @workgroup_size(1)
     fn foo() {
       // Fast approximation should be close to standard version
       let NoV = 0.8;
       let NoL = 0.7;
       let roughness = 0.5;

       let standardFn = smithGGXCorrelated(NoV, NoL, roughness);
       let fastFn = smithGGXCorrelated_Fast(NoV, NoL, roughness);

       // Test that fast version follows same trends
       let fastSmoothFn = smithGGXCorrelated_Fast(NoV, NoL, 0.1);
       let fastRoughFn = smithGGXCorrelated_Fast(NoV, NoL, 0.9);

       test::results[0] = vec4f(standardFn, fastFn, fastSmoothFn, fastRoughFn);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Fast approximation should be reasonably close to standard
  expect(Math.abs(result[0] - result[1])).toBeLessThan(0.1);
  // Fast version should also show smooth > rough
  expect(result[2]).toBeGreaterThan(result[3]);
  // Exact values to catch regressions
  expectCloseTo([0.4076, 0.3817, 0.4318, 0.342], result);
});
