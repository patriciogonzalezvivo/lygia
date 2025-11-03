import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("desaturate", async () => {
  const src = `
     import lygia::color::desaturate::desaturate;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(1.0, 0.5, 0.0); // Orange color
       let result = desaturate(color, 0.5); // 50% desaturation
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Orange (1, 0.5, 0) with 50% desaturation should move toward gray (0.64, 0.64, 0.64)
  // Gray value is luminance: 1*0.3 + 0.5*0.59 + 0*0.11 = 0.3 + 0.295 = 0.595
  // 50% blend: (1+0.595)/2 = 0.7975, (0.5+0.595)/2 = 0.5475, (0+0.595)/2 = 0.2975
  expectCloseTo([0.7975, 0.5475, 0.2975], result);
});

test("desaturate4", async () => {
  const src = `
     import lygia::color::desaturate::desaturate4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(1.0, 0.5, 0.0, 0.8); // Orange color with alpha
       let result = desaturate4(color, 0.5); // 50% desaturation
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // RGB should be same as desaturate test, alpha should remain 0.8
  expectCloseTo([0.7975, 0.5475, 0.2975, 0.8], result);
});

test("brightnessMatrix", async () => {
  const src = `
     import lygia::color::brightnessMatrix::brightnessMatrix;

     @compute @workgroup_size(1)
     fn foo() {
       let matrix = brightnessMatrix(0.2); // 20% brightness increase
       // Test that the matrix translates colors correctly
       // Matrix should have brightness offset in the last column
       test::results[0] = vec4f(matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // The translation part should be (0.2, 0.2, 0.2, 1.0)
  expectCloseTo([0.2, 0.2, 0.2, 1.0], result);
});

test("contrast", async () => {
  const src = `
     import lygia::color::contrast::contrast;

     @compute @workgroup_size(1)
     fn foo() {
       let result = contrast(0.7, 1.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // (0.7 - 0.5) * 1.5 + 0.5 = 0.2 * 1.5 + 0.5 = 0.8
  expectCloseTo([0.8], result);
});

test("contrast3", async () => {
  const src = `
     import lygia::color::contrast::contrast3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.8, 0.6, 0.4);
       let result = contrast3(color, 2.0);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Each component: (v - 0.5) * 2.0 + 0.5
  // r: (0.8 - 0.5) * 2 + 0.5 = 1.1
  // g: (0.6 - 0.5) * 2 + 0.5 = 0.7
  // b: (0.4 - 0.5) * 2 + 0.5 = 0.3
  expectCloseTo([1.1, 0.7, 0.3], result);
});

test("contrastMatrix", async () => {
  const src = `
     import lygia::color::contrastMatrix::contrastMatrix;

     @compute @workgroup_size(1)
     fn foo() {
       let matrix = contrastMatrix(1.5);
       // Test diagonal and translation values
       test::results[0] = vec4f(matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][0]);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Diagonal should be 1.5, translation should be (1-1.5)*0.5 = -0.25
  expectCloseTo([1.5, 1.5, 1.5, -0.25], result);
});

test("brightnessContrast", async () => {
  const src = `
     import lygia::color::brightnessContrast::brightnessContrast;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.7;
       let brightness = 0.1;
       let contrast = 1.5;
       let result = brightnessContrast(value, brightness, contrast);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // (0.7 - 0.5) * 1.5 + 0.5 + 0.1 = 0.2 * 1.5 + 0.6 = 0.9
  expectCloseTo([0.9], result);
});

test("contrast4", async () => {
  const src = `
     import lygia::color::contrast::contrast4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.8, 0.6, 0.4, 0.9);
       let result = contrast4(color, 1.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Each RGB component: (v - 0.5) * 1.5 + 0.5
  // r: (0.8 - 0.5) * 1.5 + 0.5 = 0.95
  // g: (0.6 - 0.5) * 1.5 + 0.5 = 0.65
  // b: (0.4 - 0.5) * 1.5 + 0.5 = 0.35
  // a: preserved at 0.9
  expectCloseTo([0.95, 0.65, 0.35, 0.9], result);
});

test("exposure", async () => {
  const src = `
     import lygia::color::exposure::exposure;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.25;
       let amount = 2.0; // +2 stops = 4x brighter
       let result = exposure(value, amount);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src);
  // 0.25 * 2^2 = 0.25 * 4 = 1.0
  expectCloseTo([1.0], result);
});

test("hueShift", async () => {
  const src = `
     import lygia::color::hueShift::hueShift;
     import lygia::math::consts::TAU;

     @compute @workgroup_size(1)
     fn foo() {
       let rgb = vec3f(1.0, 0.0, 0.0); // Red
       let shifted = hueShift(rgb, TAU * 0.3333); // Shift by 120° (TAU/3 radians)
       test::results[0] = shifted;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Red shifted by 120° should become green
  // HSV color space conversion introduces small floating-point errors (~0.0002)
  expectCloseTo([0.0, 1.0, 0.0], result, 0.001);
});

test("vibrance", async () => {
  const src = `
     import lygia::color::vibrance::vibrance3;

     @compute @workgroup_size(1)
     fn foo() {
       // Orange color with low saturation (muted)
       let rgb = vec3f(0.6, 0.5, 0.4);
       // Increase vibrance by 0.5 (should increase saturation of muted colors)
       let result = vibrance3(rgb, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Vibrance formula: mix(vec3(luma), color, 1.0 + (v * 1.0 - sign(v) * sat))
  // max_color = 0.6, min_color = 0.4, sat = 0.2
  // luma ≈ 0.6*0.2126 + 0.5*0.7152 + 0.4*0.0722 = 0.5141
  // mix factor = 1.0 + (0.5 * 1.0 - sign(0.5) * 0.2) = 1.0 + 0.5 - 0.2 = 1.3
  // mix(0.5141, color, 1.3) means interpolate/extrapolate
  // r: 0.5141 + (0.6 - 0.5141) * 1.3 = 0.5141 + 0.1117 = 0.6258
  // g: 0.5141 + (0.5 - 0.5141) * 1.3 = 0.5141 - 0.0183 = 0.4958
  // b: 0.5141 + (0.4 - 0.5141) * 1.3 = 0.5141 - 0.1483 = 0.3658
  expectCloseTo([0.6258, 0.4958, 0.3658], result);
});

test("vibrance - selective saturation boost", async () => {
  const src = `
     import lygia::color::vibrance::vibrance3;

     @compute @workgroup_size(1)
     fn foo() {
       // Test 1: Muted color (low saturation) - should change significantly
       let muted = vec3f(0.6, 0.5, 0.4);  // sat = 0.2
       let muted_boosted = vibrance3(muted, 0.5);

       // Test 2: Saturated color (high saturation) - should change less
       let saturated = vec3f(1.0, 0.1, 0.0);  // sat = 0.9
       let saturated_boosted = vibrance3(saturated, 0.5);

       // Test 3: Negative vibrance should desaturate
       let color = vec3f(0.8, 0.4, 0.2);
       let desaturated = vibrance3(color, -0.5);

       // Calculate saturation change for each
       let muted_sat_change = (muted_boosted.r - muted_boosted.b) / (muted.r - muted.b);
       let saturated_sat_change = (saturated_boosted.r - saturated_boosted.g) / (saturated.r - saturated.g);

       test::results[0] = vec4f(muted_sat_change, saturated_sat_change, desaturated.r, desaturated.g);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });

  // Vibrance should increase muted saturation more than saturated colors
  // muted_sat_change should be > saturated_sat_change
  expect(result[0]).toBeGreaterThan(result[1]);

  // Muted color should have increased saturation (change > 1.0)
  expect(result[0]).toBeGreaterThan(1.0);

  // Negative vibrance should move colors toward gray
  expectCloseTo([0.833, 0.393], [result[2], result[3]]);
});
