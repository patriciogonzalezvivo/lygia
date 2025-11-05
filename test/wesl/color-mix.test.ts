import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("mixOklab", async () => {
  const src = `
     import lygia::color::mixOklab::mixOklab;

     @compute @workgroup_size(1)
     fn foo() {
       let color1 = vec3f(1.0, 0.0, 0.0); // Red
       let color2 = vec3f(0.0, 0.0, 1.0); // Blue
       let result = mixOklab(color1, color2, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Mix red and blue in Oklab space - purple-ish result
  expectCloseTo([0.2637, 0.0866, 0.3628], result);
});

test("mixOklab4", async () => {
  const src = `
     import lygia::color::mixOklab::mixOklab4;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec4f(1.0, 0.0, 0.0, 0.8);
       let blue = vec4f(0.0, 0.0, 1.0, 0.4);
       let result = mixOklab4(red, blue, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Mix red and blue in Oklab space - purple-ish result
  // RGB should match mixOklab test, alpha should be 0.6 (mix of 0.8 and 0.4)
  expectCloseTo([0.2637, 0.0866, 0.3628, 0.6], result);
});

test("mixSpectral", async () => {
  const src = `
     import lygia::color::mixSpectral::mixSpectral;

     @compute @workgroup_size(1)
     fn foo() {
       let red = vec3f(1.0, 0.0, 0.0);
       let blue = vec3f(0.0, 0.0, 1.0);

       // Test 50% spectral mix of red and blue
       // Spectral mixing uses Kubelka-Munk theory - physically accurate paint mixing
       // This produces a different result than linear RGB interpolation
       let mixed = mixSpectral(red, blue, 0.5);

       // For comparison, store what linear mix would give
       let linearMix = mix(red, blue, 0.5); // Would be (0.5, 0, 0.5)

       test::results[0] = vec4f(mixed.r, mixed.g, mixed.b, linearMix.r);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Spectral mixing of red and blue produces a dark purple color
  // This is VERY different from linear RGB mixing which would give bright (0.5, 0, 0.5)
  // Spectral mixing uses Kubelka-Munk theory - physically accurate paint mixing
  // Real paint mixing produces darker, more muted colors than digital RGB mixing
  const mixed = [result[0], result[1], result[2]];

  // Spectral mixing produces a much darker result than linear mixing
  // All components should be quite low (realistic for physical pigment mixing)
  expect(mixed[0]).toBeGreaterThan(0.0);
  expect(mixed[0]).toBeLessThan(0.15); // Red is low

  expect(mixed[1]).toBeGreaterThan(0.0);
  expect(mixed[1]).toBeLessThan(0.1); // Green is very low

  expect(mixed[2]).toBeGreaterThan(0.0);
  expect(mixed[2]).toBeLessThan(0.1); // Blue is also low (darken effect from mixing)

  // Overall, the spectral mix should be notably darker than either input color
  const maxComponent = Math.max(...mixed);
  expect(maxComponent).toBeLessThan(0.2); // Dark purple, not bright

  // Verify linear mix comparison value is 0.5 (as expected for linear interpolation)
  expectCloseTo([0.5], [result[3]]);

  // Spectral mix should differ DRAMATICALLY from linear mix
  // (physical paint mixing produces darker colors than digital RGB mixing)
  expect(Math.abs(mixed[0] - result[3])).toBeGreaterThan(0.3); // 0.067 vs 0.5 = big difference!

  // Regression check - exact spectral mix values
  expectCloseTo([0.0673, 0.0093, 0.0241], mixed);
});

test("mixSpectral4", async () => {
  const src = `
     import lygia::color::mixSpectral::mixSpectral4;

     @compute @workgroup_size(1)
     fn foo() {
       let yellow = vec4f(1.0, 1.0, 0.0, 0.9);
       let cyan = vec4f(0.0, 1.0, 1.0, 0.5);
       let result = mixSpectral4(yellow, cyan, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Spectral mixing of yellow and cyan produces green
  // Alpha should be 0.7 (mix of 0.9 and 0.5)
  expect(result[1]).toBeGreaterThan(result[0]); // Green dominant
  expect(result[1]).toBeGreaterThan(result[2]); // Green > Blue
  expectCloseTo([0.7], [result[3]]); // Alpha

  // Regression check - exact spectral mix RGB values
  expectCloseTo([0.0782, 1.0272, 0.0596], [result[0], result[1], result[2]]);
});

test("mixSpectral_linear_to_reflectance", async () => {
  const src = `
     import lygia::color::mixSpectral::mixSpectral_linear_to_reflectance;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Pure red
       let reflectance = mixSpectral_linear_to_reflectance(rgb);

       // Sample a few wavelengths from the reflectance array
       // Red should have high reflectance in long wavelengths (red end of spectrum)
       // and low reflectance in short wavelengths (blue end)
       test::results[0] = vec4f(
         reflectance[0],   // Short wavelength (blue/violet)
         reflectance[19],  // Mid wavelength (green)
         reflectance[37],  // Long wavelength (red)
         1.0
       );
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // For pure red input:
  // - Short wavelengths (blue) should have low reflectance
  // - Long wavelengths (red) should have high reflectance
  expect(result[0]).toBeLessThan(0.2); // Blue end - low reflectance
  expect(result[2]).toBeGreaterThan(0.8); // Red end - high reflectance
  expect(result[2]).toBeGreaterThan(result[0]); // Red > Blue

  // Regression check - exact reflectance at sampled wavelengths
  expectCloseTo([0.0315, 0.0318, 0.9855], [result[0], result[1], result[2]]);
});

test("mixSpectral_reflectance_to_xyz", async () => {
  const src = `
     import lygia::color::mixSpectral::{mixSpectral_linear_to_reflectance, mixSpectral_reflectance_to_xyz};

     @compute @workgroup_size(1)
     fn foo() {
       // Test with a neutral gray
       let gray = vec3f(0.5, 0.5, 0.5);
       let reflectance = mixSpectral_linear_to_reflectance(gray);
       let xyz = mixSpectral_reflectance_to_xyz(reflectance);

       // For a neutral gray, XYZ values should be roughly equal
       // and Y (luminance) should be around 0.5
       test::results[0] = vec4f(xyz, 1.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // For neutral gray input:
  // - X, Y, Z should be similar (neutral color)
  // - Y (luminance) should be positive and reasonable
  const xyz = [result[0], result[1], result[2]];

  // All components should be positive
  expect(xyz[0]).toBeGreaterThan(0.0);
  expect(xyz[1]).toBeGreaterThan(0.0);
  expect(xyz[2]).toBeGreaterThan(0.0);

  // For gray, X and Z should be reasonably close to Y
  // (not exact due to spectral conversion, but within a reasonable range)
  expect(xyz[0]).toBeGreaterThan(xyz[1] * 0.5);
  expect(xyz[0]).toBeLessThan(xyz[1] * 1.5);
  expect(xyz[2]).toBeGreaterThan(xyz[1] * 0.5);
  expect(xyz[2]).toBeLessThan(xyz[1] * 1.5);

  // Regression check - exact XYZ values for gray input
  expectCloseTo([0.4751, 0.5, 0.5441], xyz);
});
