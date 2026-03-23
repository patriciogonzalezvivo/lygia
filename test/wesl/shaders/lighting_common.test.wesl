import wgsl_test::{expect, expectNear, expectNearVec2, expectNearVec3, expectNearVec4};
import lygia::lighting::common::ggx::{GGX, GGXPrecise, importanceSamplingGGX};
import lygia::lighting::common::schlick::{schlick, schlickVec3, schlickF32};
import lygia::lighting::common::smithGGXCorrelated::{smithGGXCorrelated, smithGGXCorrelated_Fast};

@test fn testGGX() {
  let NoH = 1.0;
  let roughness = 0.5;
  let result1 = GGX(NoH, roughness);
  let result2 = GGX(0.8, roughness);
  let result3 = GGX(1.0, 0.1);

  // GGX distribution peaks at NoH=1.0
  expect(result1 > 0.3);
  // GGX decreases as NoH decreases
  expect(result2 < result1);
  // Lower roughness produces sharper, higher peak
  expect(result3 > result1);
  // Exact values to catch regressions
  expectNearVec3(vec3f(result1, result2, result3), vec3f(1.2732, 0.2943, 31.831));
}

@test fn testGGXPrecise() {
  let N = vec3f(0.0, 0.0, 1.0);
  let H = vec3f(0.0, 0.0, 1.0); // Perfect alignment
  let NoH = dot(N, H);
  let roughness = 0.5;

  let preciseFn = GGXPrecise(N, H, NoH, roughness);
  let standardFn = GGX(NoH, roughness);

  // GGXPrecise should produce similar results to standard GGX
  expectNear(preciseFn, standardFn);
  // Both should be positive and reasonable
  expect(preciseFn > 0.3);
  // Exact values to catch regressions
  expectNearVec2(vec2f(preciseFn, standardFn), vec2f(1.2732, 1.2732));
}

@test fn testImportanceSamplingGGX() {
  // With u=(0,0), should produce direction toward Z (cosTheta=1)
  let sample1 = importanceSamplingGGX(vec2f(0.0, 0.0), 0.5);
  // With u=(0.5,0.5), should produce mid-range direction
  let sample2 = importanceSamplingGGX(vec2f(0.5, 0.5), 0.5);
  // Lower roughness should bias samples toward Z axis
  let sample3 = importanceSamplingGGX(vec2f(0.5, 0.5), 0.1);

  // u=(0,0) produces direction close to Z axis
  expectNear(sample1.z, 1.0);
  // Samples should be normalized (unit length)
  expectNear(length(sample2), 1.0);
  // Lower roughness biases toward Z
  expect(sample3.z > 0.7);
  // Exact values to catch regressions
  expectNearVec3(vec3f(sample1.z, length(sample2), sample3.z), vec3f(1.0, 1.0, 0.995));
}

@test fn testSchlickVec3f() {
  let f0 = vec3f(0.04, 0.04, 0.04); // Dielectric base reflectivity
  let f90 = 1.0;

  // At normal incidence (VoH=1.0), should equal f0
  let normalFn = schlick(f0, f90, 1.0);
  // At grazing angle (VoH=0.0), should equal f90
  let grazingFn = schlick(f0, f90, 0.0);
  // At mid-angle, should be between f0 and f90
  let midFn = schlick(f0, f90, 0.5);

  expectNear(normalFn.x, 0.04);
  expectNear(grazingFn.x, 1.0);
  expect(midFn.x > 0.04);
  expect(midFn.x < 1.0);
  // Exact values to catch regressions
  expectNearVec3(vec3f(normalFn.x, grazingFn.x, midFn.x), vec3f(0.04, 1.0, 0.07));
}

@test fn testSchlickVec3() {
  let f0 = vec3f(1.0, 0.71, 0.29); // Gold-like base reflectivity
  let f90 = vec3f(1.0, 0.95, 0.9); // Colored grazing reflectivity

  // At normal incidence, should equal f0
  let normalFn = schlickVec3(f0, f90, 1.0);
  // At grazing angle, should approach f90
  let grazingFn = schlickVec3(f0, f90, 0.0);

  expectNearVec3(normalFn, vec3f(1.0, 0.71, 0.29));
  expect(grazingFn.x > 0.85);
  // Exact values to catch regressions
  expectNearVec4(vec4f(normalFn, grazingFn.x), vec4f(1.0, 0.71, 0.29, 1.0));
}

@test fn testSchlickF32() {
  let f0 = 0.04;
  let f90 = 1.0;

  let normalFn = schlickF32(f0, f90, 1.0);   // VoH=1 -> f0
  let grazingFn = schlickF32(f0, f90, 0.0);  // VoH=0 -> f90
  let midFn = schlickF32(f0, f90, 0.5);      // VoH=0.5 -> interpolated

  expectNear(normalFn, 0.04);
  expectNear(grazingFn, 1.0);
  expect(midFn > 0.04);
  expect(midFn < 1.0);
  // Exact values to catch regressions
  expectNearVec3(vec3f(normalFn, grazingFn, midFn), vec3f(0.04, 1.0, 0.07));
}

@test fn testSmithGGXCorrelated() {
  let NoV = 0.8;
  let NoL = 0.7;

  let smoothFn = smithGGXCorrelated(NoV, NoL, 0.1);
  let roughFn = smithGGXCorrelated(NoV, NoL, 0.9);
  let perfectFn = smithGGXCorrelated(1.0, 1.0, 0.5);

  // Smooth surfaces have higher visibility
  expect(smoothFn > roughFn);
  // Rough surfaces still have positive visibility
  expect(roughFn > 0.0);
  // Perfect alignment has good visibility
  expect(perfectFn > 0.2);
  // Exact values to catch regressions
  expectNearVec3(vec3f(smoothFn, roughFn, perfectFn), vec3f(0.4447, 0.3482, 0.25));
}

@test fn testSmithGGXCorrelatedFast() {
  let NoV = 0.8;
  let NoL = 0.7;
  let roughness = 0.5;

  let standardFn = smithGGXCorrelated(NoV, NoL, roughness);
  let fastFn = smithGGXCorrelated_Fast(NoV, NoL, roughness);

  let fastSmoothFn = smithGGXCorrelated_Fast(NoV, NoL, 0.1);
  let fastRoughFn = smithGGXCorrelated_Fast(NoV, NoL, 0.9);

  // Fast approximation should be reasonably close to standard
  expect(abs(standardFn - fastFn) < 0.1);
  // Fast version should also show smooth > rough
  expect(fastSmoothFn > fastRoughFn);
  // Exact values to catch regressions
  expectNearVec4(vec4f(standardFn, fastFn, fastSmoothFn, fastRoughFn), vec4f(0.4076, 0.3817, 0.4318, 0.342));
}
