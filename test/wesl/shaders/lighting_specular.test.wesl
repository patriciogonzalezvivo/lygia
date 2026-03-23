import wgsl_test::{expect, expectNear, expectNearVec3, expectNearVec4, expectWithin};
import lygia::lighting::fresnel::{fresnel, fresnelF32, fresnelFromVectors, fresnelRoughness};
import lygia::lighting::specular::cookTorrance::specularCookTorrance;
import lygia::lighting::toShininess::toShininess;

@test fn testFresnelVec3f() {
  let f0 = vec3f(0.04, 0.04, 0.04);
  let NoV = 1.0; // View perpendicular to surface
  let result = fresnel(f0, NoV);
  // Fresnel at normal incidence should be close to f0
  expectNear(result.x, 0.04);
}

@test fn testFresnelF32() {
  let f0 = 0.04;
  let NoV = 1.0;
  let result = fresnelF32(f0, NoV);
  expectNear(result, 0.04);
}

@test fn testFresnelFromVectors() {
  let f0 = vec3f(0.04, 0.04, 0.04);
  let normal = vec3f(0.0, 0.0, 1.0);
  let view = vec3f(0.0, 0.0, 1.0); // View perpendicular to surface
  let result = fresnelFromVectors(f0, normal, view);
  expectNear(result.x, 0.04);
}

@test fn testFresnelRoughness() {
  let f0 = vec3f(0.04, 0.04, 0.04);

  // At normal incidence (NoV=1.0), roughness has minimal effect
  let normalSmooth = fresnelRoughness(f0, 1.0, 0.1);

  // At grazing angle (NoV=0.1), roughness modulates the Fresnel peak
  let grazingSmooth = fresnelRoughness(f0, 0.1, 0.1);
  let grazingRough = fresnelRoughness(f0, 0.1, 0.9);

  // Mid-angle
  let midSmooth = fresnelRoughness(f0, 0.5, 0.1);

  // At normal incidence, should equal f0
  expectNear(normalSmooth.x, 0.04);

  // Smooth should be significantly elevated at grazing angle
  expect(grazingSmooth.x > 0.4);
  // Rough is attenuated
  expect(grazingRough.x < 0.3);
  // At least 1.5x difference
  expect(grazingSmooth.x > grazingRough.x * 1.5);

  // Mid-angle between normal and grazing
  expect(midSmooth.x > normalSmooth.x);
  expect(midSmooth.x < grazingSmooth.x);

  // Roughness effect should be stronger at grazing angles
  let grazingDiff = abs(grazingSmooth.x - grazingRough.x);
  expect(grazingDiff > 0.4);

  // Exact values to catch regressions
  expectNearVec4(vec4f(normalSmooth.x, grazingSmooth.x, grazingRough.x, midSmooth.x), vec4f(0.04, 0.54782, 0.07543, 0.06687));
}

@test fn testSpecularCookTorrance() {
  let specularColor = vec3f(0.04, 0.04, 0.04);
  let N = vec3f(0.0, 0.0, 1.0);

  // Test 1: Perfect specular reflection (H = N)
  let L1 = N;
  let H1 = N;
  let NoV1 = 1.0;
  let NoL1 = 1.0;
  let NoH1 = 1.0;
  let perfectSmooth = specularCookTorrance(L1, N, H1, NoV1, NoL1, NoH1, 0.1, specularColor);
  let perfectRough = specularCookTorrance(L1, N, H1, NoV1, NoL1, NoH1, 0.9, specularColor);

  // Test 2: Off-specular (H != N)
  let L3 = normalize(vec3f(0.5, 0.0, 1.0));
  let V3 = vec3f(0.0, 0.0, 1.0);
  let H3 = normalize(L3 + V3);
  let NoV3 = dot(N, V3);
  let NoL3 = dot(N, L3);
  let NoH3 = dot(N, H3);
  let offSpecSmooth = specularCookTorrance(L3, N, H3, NoV3, NoL3, NoH3, 0.1, specularColor);
  let offSpecRough = specularCookTorrance(L3, N, H3, NoV3, NoL3, NoH3, 0.9, specularColor);

  // Perfect alignment produces strongest specular
  expect(perfectSmooth.x > 0.1);
  expect(perfectSmooth.x > perfectRough.x);

  // Off-specular should be dimmer than perfect specular
  expect(offSpecSmooth.x < perfectSmooth.x);

  // Smooth surfaces have sharper falloff (larger ratio)
  let smoothRatio = perfectSmooth.x / (offSpecSmooth.x + 0.001);
  let roughRatio = perfectRough.x / (offSpecRough.x + 0.001);
  expect(smoothRatio > roughRatio);

  // Exact values to catch regressions
  expectNearVec4(vec4f(perfectSmooth.x, perfectRough.x, offSpecSmooth.x, offSpecRough.x), vec4f(0.31831, 0.00393, 0.00918, 0.00409));
}

@test fn testToShininess() {
  // Extremes of roughness (dielectric)
  let verySmooth = toShininess(0.0, 0.0);   // s = 0.95^4 * 240 ~ 194.4
  let veryRough = toShininess(1.0, 0.0);    // s = 0.45^4 * 240 ~ 9.8

  // Mid-roughness (dielectric)
  let midRough = toShininess(0.5, 0.0);     // s = 0.7^4 * 240 ~ 57.6

  // Metallic at same roughness
  let metallic = toShininess(0.3, 1.0);     // s = 0.8^4 * 80 ~ 32.77

  // Very smooth has highest shininess
  expect(verySmooth > 150.0);
  expectWithin(verySmooth, 194.4, 2.0, 1e-6);

  // Very rough has low shininess
  expect(veryRough < 15.0);
  expectWithin(veryRough, 9.8, 0.1, 1e-6);

  // Inverse relationship - smooth >> rough
  expect(verySmooth > veryRough * 10.0);

  // Mid-roughness is between extremes
  expect(midRough > veryRough);
  expect(midRough < verySmooth);
  expectWithin(midRough, 57.6, 0.1, 1e-6);

  // Metallic reduces shininess
  expect(metallic < 40.0);
  expect(metallic > 25.0);
  expectWithin(metallic, 32.77, 0.01, 1e-6);

  // All values should be in valid shininess range
  expect(verySmooth < 250.0);
  expect(veryRough > 0.0);
}
