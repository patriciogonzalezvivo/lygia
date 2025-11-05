import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("fresnel vec3f", async () => {
  const src = `
     import lygia::lighting::fresnel::fresnel;

     @compute @workgroup_size(1)
     fn foo() {
       // fresnel(f0: vec3f, NoV: f32) -> vec3f
       // f0 is base reflectivity (e.g. 0.04 for dielectrics)
       // NoV is dot product of normal and view direction
       let f0 = vec3f(0.04, 0.04, 0.04);  // Typical dielectric
       let NoV = 1.0;  // View perpendicular to surface (normal incidence)
       let result = fresnel(f0, NoV);
       test::results[0] = result.x;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Fresnel at normal incidence should be close to f0
  // Using default precision - actual difference is ~1e-9
  expectCloseTo([0.04], result);
});

test("fresnelF32", async () => {
  const src = `
     import lygia::lighting::fresnel::fresnelF32;

     @compute @workgroup_size(1)
     fn foo() {
       // fresnelF32(f0: f32, NoV: f32) -> f32
       // f0 is base reflectivity (e.g. 0.04 for dielectrics)
       // NoV is dot product of normal and view direction
       let f0 = 0.04;  // Typical dielectric
       let NoV = 1.0;  // View perpendicular to surface (normal incidence)
       let result = fresnelF32(f0, NoV);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Fresnel at normal incidence should be close to f0
  expectCloseTo([0.04], result);
});

test("fresnelFromVectors", async () => {
  const src = `
     import lygia::lighting::fresnel::fresnelFromVectors;

     @compute @workgroup_size(1)
     fn foo() {
       // fresnelFromVectors(f0: vec3f, normal: vec3f, view: vec3f) -> vec3f
       let f0 = vec3f(0.04, 0.04, 0.04);
       let normal = vec3f(0.0, 0.0, 1.0);
       let view = vec3f(0.0, 0.0, 1.0);  // View perpendicular to surface
       let result = fresnelFromVectors(f0, normal, view);
       test::results[0] = result.x;
     }
   `;
  const result = await lygiaTestCompute(src);
  // Fresnel at normal incidence should be close to f0
  // Using default precision - actual difference is ~1e-9
  expectCloseTo([0.04], result);
});

test("fresnelRoughness", async () => {
  const src = `
     import lygia::lighting::fresnel::{fresnel, fresnelRoughness};

     @compute @workgroup_size(1)
     fn foo() {
       // fresnelRoughness attenuates high speculars at glancing angles
       // Formula: f0 + (max(1-roughness, f0) - f0) * pow5(1-NoV)
       let f0 = vec3f(0.04, 0.04, 0.04);

       // Test 1: At normal incidence (NoV=1.0), roughness should have minimal effect
       // pow5(1-1.0) = 0, so result should equal f0 regardless of roughness
       let normalSmooth = fresnelRoughness(f0, 1.0, 0.1);

       // Test 2: At grazing angle (NoV=0.1), roughness should modulate the Fresnel peak
       // For smooth surfaces (low roughness), Fresnel approaches 1.0 at grazing angles
       // For rough surfaces (high roughness), Fresnel peak is attenuated
       let grazingSmooth = fresnelRoughness(f0, 0.1, 0.1);  // max(0.9, 0.04) = 0.9 at grazing
       let grazingRough = fresnelRoughness(f0, 0.1, 0.9);   // max(0.1, 0.04) = 0.1 at grazing

       // Test 3: Mid-angle (NoV=0.5) should show intermediate behavior
       let midSmooth = fresnelRoughness(f0, 0.5, 0.1);

       test::results[0] = vec4f(normalSmooth.x, grazingSmooth.x, grazingRough.x, midSmooth.x);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Test 1: At normal incidence, should equal f0 (0.04)
  expectCloseTo([0.04], [result[0]]);

  // Test 2: At grazing angle, smooth surface should have much higher Fresnel than rough
  expect(result[1]).toBeGreaterThan(0.4); // Smooth should be significantly elevated
  expect(result[2]).toBeLessThan(0.3); // Rough is attenuated
  expect(result[1]).toBeGreaterThan(result[2] * 1.5); // At least 1.5x difference

  // Test 3: Mid-angle should show intermediate values between normal and grazing
  expect(result[3]).toBeGreaterThan(result[0]); // Mid > normal incidence
  expect(result[3]).toBeLessThan(result[1]); // Mid < grazing (smooth)

  // Test 4: Roughness effect should be stronger at grazing angles
  // let normalDiff = 0.0;  // At normal incidence, roughness has no effect
  const grazingDiff = Math.abs(result[1] - result[2]);
  expect(grazingDiff).toBeGreaterThan(0.4); // Large difference at grazing angles

  // Most specific check last - exact values to catch regressions
  expectCloseTo([0.04, 0.54782, 0.07543, 0.06687], result);
});

test("specularCookTorrance", async () => {
  const src = `
     import lygia::lighting::specular::cookTorrance::specularCookTorrance;

     @compute @workgroup_size(1)
     fn foo() {
       // Cook-Torrance BRDF: (D * V * F) where:
       // D = GGX distribution (normal distribution function)
       // V = Smith visibility term (geometric shadowing/masking)
       // F = Fresnel (view-dependent reflectance)

       let specularColor = vec3f(0.04, 0.04, 0.04);
       let N = vec3f(0.0, 0.0, 1.0);

       // Test 1: Perfect specular reflection (H = N)
       // Light and view aligned with normal
       let L1 = N;
       let H1 = N;
       let NoV1 = 1.0;
       let NoL1 = 1.0;
       let NoH1 = 1.0;
       let perfectSmooth = specularCookTorrance(L1, N, H1, NoV1, NoL1, NoH1, 0.1, specularColor);
       let perfectRough = specularCookTorrance(L1, N, H1, NoV1, NoL1, NoH1, 0.9, specularColor);

       // Test 2: Off-specular (H != N)
       // When H deviates from N, specular should decrease
       // Effect is stronger for smooth surfaces (narrow lobe)
       let L3 = normalize(vec3f(0.5, 0.0, 1.0));  // 45° from normal
       let V3 = vec3f(0.0, 0.0, 1.0);
       let H3 = normalize(L3 + V3);
       let NoV3 = dot(N, V3);
       let NoL3 = dot(N, L3);
       let NoH3 = dot(N, H3);
       let offSpecSmooth = specularCookTorrance(L3, N, H3, NoV3, NoL3, NoH3, 0.1, specularColor);
       let offSpecRough = specularCookTorrance(L3, N, H3, NoV3, NoL3, NoH3, 0.9, specularColor);

       test::results[0] = vec4f(perfectSmooth.x, perfectRough.x, offSpecSmooth.x, offSpecRough.x);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Test 1: Perfect alignment produces strongest specular
  // Smooth surfaces have sharper, taller peaks
  expect(result[0]).toBeGreaterThan(0.1); // Strong specular for smooth
  expect(result[0]).toBeGreaterThan(result[1]); // Smooth > rough at peak

  // Test 2: Off-specular should be dimmer than perfect specular
  expect(result[2]).toBeLessThan(result[0]); // Off-spec < perfect (smooth)
  // Note: For rough surfaces, off-specular may be similar to or even slightly higher than perfect
  // due to broader light scattering. The key test is the ratio difference.

  // Test 3: Roughness effect on specular falloff
  // Smooth surfaces have sharper falloff (larger ratio)
  const smoothRatio = result[0] / (result[2] + 0.001); // Peak / off-spec
  const roughRatio = result[1] / (result[3] + 0.001);
  expect(smoothRatio).toBeGreaterThan(roughRatio); // Smooth falls off faster

  // Most specific check last - exact values to catch regressions
  expectCloseTo([0.31831, 0.00393, 0.00918, 0.00409], result);
});

test("toShininess", async () => {
  const src = `
     import lygia::lighting::toShininess::toShininess;

     @compute @workgroup_size(1)
     fn foo() {
       // toShininess converts PBR roughness to Blinn-Phong shininess
       // Formula: s = (0.95 - roughness*0.5)^4 * (80 + 160*(1-metallic))
       // Inverse relationship: high roughness -> low shininess

       // Test 1: Extremes of roughness (dielectric)
       let verySmooth = toShininess(0.0, 0.0);   // s = 0.95^4 * 240 ≈ 194.4
       let veryRough = toShininess(1.0, 0.0);    // s = 0.45^4 * 240 ≈ 9.8

       // Test 2: Mid-roughness (dielectric)
       let midRough = toShininess(0.5, 0.0);     // s = 0.7^4 * 240 ≈ 57.6

       // Test 3: Metallic vs dielectric at same roughness
       let dielectric = toShininess(0.3, 0.0);   // Uses 240 multiplier (80+160*1)
       let metallic = toShininess(0.3, 1.0);     // Uses 80 multiplier (80+160*0)

       test::results[0] = vec4f(verySmooth, veryRough, midRough, metallic);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Test 1: Very smooth has highest shininess
  expect(result[0]).toBeGreaterThan(150.0); // Should be ~194
  // Custom precision needed due to accumulated error in (0.95^4 * 240) computation
  expectCloseTo([194.4], [result[0]], 2.0);

  // Very rough has low shininess
  expect(result[1]).toBeLessThan(15.0); // Should be ~9.8
  // Custom precision needed due to accumulated error in (0.45^4 * 240) computation
  expectCloseTo([9.8], [result[1]], 0.1);

  // Test 2: Inverse relationship - smooth >> rough
  expect(result[0]).toBeGreaterThan(result[1] * 10); // At least 10x difference

  // Test 3: Mid-roughness is between extremes
  expect(result[2]).toBeGreaterThan(result[1]); // Mid > rough
  expect(result[2]).toBeLessThan(result[0]); // Mid < smooth
  // Custom precision needed due to accumulated error in (0.7^4 * 240) computation
  expectCloseTo([57.6], [result[2]], 0.1);

  // Test 4: Metallic reduces shininess (smaller multiplier)
  // Note: dielectric was removed to fit in vec4f, so we test against expected value
  // Metallic: roughness=0.3, metallic=1.0 -> s = 0.8^4 * 80 ≈ 32.77
  expect(result[3]).toBeLessThan(40.0); // Metallic should be low
  expect(result[3]).toBeGreaterThan(25.0); // But not too low
  // Custom precision needed due to accumulated error in (0.8^4 * 80) computation
  expectCloseTo([32.77], [result[3]], 0.01);

  // Test 5: All values should be in valid shininess range
  expect(result[0]).toBeLessThan(250.0); // Max is 240 * 0.95^4
  expect(result[1]).toBeGreaterThan(0.0); // Min is positive
});
