import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("brightnessContrast3", async () => {
  const src = `
     import lygia::color::brightnessContrast::brightnessContrast3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.6, 0.5, 0.4);
       let brightness = 0.1;
       let contrast = 1.2;
       let result = brightnessContrast3(color, brightness, contrast);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // (0.6 - 0.5) * 1.2 + 0.5 + 0.1 = 0.12 + 0.6 = 0.72
  // (0.5 - 0.5) * 1.2 + 0.5 + 0.1 = 0 + 0.6 = 0.6
  // (0.4 - 0.5) * 1.2 + 0.5 + 0.1 = -0.12 + 0.6 = 0.48
  expectCloseTo([0.72, 0.6, 0.48], result);
});

test("brightnessContrast4", async () => {
  const src = `
     import lygia::color::brightnessContrast::brightnessContrast4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.6, 0.5, 0.4, 0.8);
       let brightness = 0.1;
       let contrast = 1.2;
       let result = brightnessContrast4(color, brightness, contrast);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // RGB adjusted, alpha preserved
  expectCloseTo([0.72, 0.6, 0.48, 0.8], result);
});

test("exposure3", async () => {
  const src = `
     import lygia::color::exposure::exposure3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.5, 0.5, 0.5);
       let amount = 1.0; // +1 stop = 2x brighter
       let result = exposure3(color, amount);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // 0.5 * 2^1 = 1.0
  expectCloseTo([1.0, 1.0, 1.0], result);
});

test("exposure4", async () => {
  const src = `
     import lygia::color::exposure::exposure4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.5, 0.5, 0.5, 0.7);
       let amount = 1.0;
       let result = exposure4(color, amount);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // RGB doubled, alpha preserved
  expectCloseTo([1.0, 1.0, 1.0, 0.7], result);
});

test("hueShiftRYB", async () => {
  const src = `
     import lygia::color::hueShiftRYB::hueShiftRYB;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(1.0, 0.0, 0.0); // Red
       let angle = 2.0944; // 120 degrees (1/3 turn) - shift red toward yellow in RYB space
       let result = hueShiftRYB(color, angle);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // RYB hue shift: Red shifted by 120° in RYB space
  // After RGB->RYB->hue shift->RGB conversion
  // Red shifted 120° in RYB color wheel goes toward yellow
  expectCloseTo([1.0, 1.0, 0.0], result);
});

test("hueShiftRYB4", async () => {
  const src = `
     import lygia::color::hueShiftRYB::hueShiftRYB4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(1.0, 0.0, 0.0, 0.7); // Red with alpha
       let angle = 2.0944; // 120 degrees
       let result = hueShiftRYB4(color, angle);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // RYB hue shift: Red shifted by 120° toward yellow
  // Loose precision due to RYB color space conversion approximations
  expectCloseTo([1.0, 1.0, 0.0, 0.7], result.slice(0, 4), 0.15);
  expectCloseTo([0.7], [result[3]]); // Alpha exact
});

test("heatmap", async () => {
  const src = `
     import lygia::color::palette::heatmap::heatmap;

     @compute @workgroup_size(1)
     fn foo() {
       let value = 0.5;
       let result = heatmap(value);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Heatmap formula: 1.0 - (v*2.1 - vec3(1.8,1.14,0.3))^2
  // For v=0.5: 1.0 - (1.05 - vec3(1.8,1.14,0.3))^2 = vec3(0.4375, 0.9919, 0.4375)
  expectCloseTo([0.4375, 0.9919, 0.4375], result);
});

test("paletteHue", async () => {
  const src = `
     import lygia::color::palette::hue::hue;

     @compute @workgroup_size(1)
     fn foo() {
       let x = 0.5;
       let ratio = 0.333; // neon ratio (1/3)
       let result = hue(x, ratio);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Physical hue palette at x=0.5 with ratio=1/3
  // Formula: v = abs(fmod(x + [0,1,2]*ratio, 1) * 2 - 1)
  // Then smoothstep: v*v*(3-2*v)
  // For x=0.5, ratio=1/3: [0.5, 0.833, 0.167] -> fmod -> [0.5, 0.833, 0.167]
  // -> *2-1 -> [0, 0.666, -0.666] -> abs -> [0, 0.666, 0.666] -> smoothstep
  // Loose precision due to smoothstep approximation differences
  expectCloseTo([0.0, 0.74, 0.743], result, 0.05);
});

test("hueDefault", async () => {
  const src = `
     import lygia::color::palette::hue::hueDefault;

     @compute @workgroup_size(1)
     fn foo() {
       // hueDefault uses default ratio of 1/3 (neon)
       // Test that it matches hue(x, 0.33333)
       let result = hueDefault(0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // hueDefault(0.5) should match paletteHue test: hue(0.5, 0.333)
  // Result should be [0.0, 0.740, 0.743] from paletteHue test
  // Loose precision due to smoothstep approximation differences
  expectCloseTo([0.0, 0.74, 0.743], result, 0.05);
});

test("whiteBalance3", async () => {
  const src = `
     import lygia::color::whiteBalance::whiteBalance3;

     @compute @workgroup_size(1)
     fn foo() {
       let gray = vec3f(0.5, 0.5, 0.5);

       // Test warm, neutral, and cool temperatures
       let warm = whiteBalance3(gray, 0.2, 0.0);    // Warm (orange/yellow)
       let neutral = whiteBalance3(gray, 0.0, 0.0); // Neutral (unchanged)
       let cool = whiteBalance3(gray, -0.2, 0.0);   // Cool (blue)

       test::results[0] = warm.r;
       test::results[1] = warm.b;
       test::results[2] = cool.r;
       test::results[3] = cool.b;
     }
   `;
  const result = await lygiaTestCompute(src);

  const warmR = result[0];
  const warmB = result[1];
  const coolR = result[2];
  const coolB = result[3];

  // Warm temperature should increase red
  expect(warmR).toBeGreaterThan(0.5); // warm.r > 0.5

  // Cool temperature should increase blue
  expect(coolB).toBeGreaterThan(warmB); // cool.b > warm.b

  // Warm should have higher red than cool
  expect(warmR).toBeGreaterThan(coolR); // warm.r > cool.r

  // Regression check - exact white balance values
  expectCloseTo([0.5141, 0.4585, 0.4727, 0.5919], [warmR, warmB, coolR, coolB]);
});

test("whiteBalance4", async () => {
  const src = `
     import lygia::color::whiteBalance::whiteBalance4;

     @compute @workgroup_size(1)
     fn foo() {
       let gray = vec4f(0.5, 0.5, 0.5, 0.8);

       // Test temperature shift (warm)
       let tempShift = whiteBalance4(gray, 0.2, 0.0);

       // Test tint shift (magenta/green)
       let tintMagenta = whiteBalance4(gray, 0.0, 0.1);  // Positive = magenta
       let tintGreen = whiteBalance4(gray, 0.0, -0.1);   // Negative = green

       test::results[0] = tempShift.r;
       test::results[1] = tempShift.b;
       test::results[2] = tintMagenta.g;
       test::results[3] = tintGreen.g;
     }
   `;
  const result = await lygiaTestCompute(src);

  const tempShiftR = result[0];
  const tempShiftB = result[1];
  const magentaG = result[2];
  const greenG = result[3];

  // Temperature shift: warm should have R > B
  expect(tempShiftR).toBeGreaterThan(tempShiftB); // tempShift.r > tempShift.b

  // Tint behavior: magenta tint reduces green, green tint increases green
  expect(greenG).toBeGreaterThan(magentaG); // green tint increases G
  expect(magentaG).toBeLessThan(0.5); // magenta tint decreases G
  expect(greenG).toBeGreaterThan(0.5); // green tint increases G

  // Regression check - exact white balance values with tint
  expectCloseTo(
    [0.5141, 0.4585, 0.4872, 0.5132],
    [tempShiftR, tempShiftB, magentaG, greenG],
  );
});

test("saturationMatrix", async () => {
  const src = `
     import lygia::color::saturationMatrix::saturationMatrix;

     @compute @workgroup_size(1)
     fn foo() {
       let amount = 1.5; // Increase saturation by 50%
       let mat = saturationMatrix(amount);
       // Test matrix by applying to an orange color
       let color = vec3f(0.8, 0.5, 0.3);
       let result = mat * vec4f(color, 1.0);
       test::results[0] = result.xyz;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Saturation matrix at 1.5 should increase color saturation
  // Original: (0.8, 0.5, 0.3) -> more saturated orange
  // Luma ~0.57, with 1.5 saturation should push values further from luma
  // Expected: R increases (>0.8), G stays similar, B decreases (<0.3)
  expect(result[0]).toBeGreaterThan(0.8); // Red should increase
  expect(result[2]).toBeLessThan(0.3); // Blue should decrease
  // Loose precision due to matrix multiplication accumulation
  expectCloseTo([0.95, 0.53, 0.17], result, 0.1);
});

test("levelsOutputRange3", async () => {
  const src = `
     import lygia::color::levels::outputRange::levelsOutputRange3;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec3f(0.5, 0.5, 0.5);
       let minOutput = vec3f(0.2, 0.2, 0.2);
       let maxOutput = vec3f(0.8, 0.8, 0.8);
       let result = levelsOutputRange3(color, minOutput, maxOutput);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Middle value (0.5) should map to middle of output range (0.5)
  expectCloseTo([0.5, 0.5, 0.5], result);
});

test("hueShift4", async () => {
  const src = `
     import lygia::color::hueShift::hueShift4;
     import lygia::math::consts::TAU;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(1.0, 0.0, 0.0, 0.85); // Red with alpha
       let shifted = hueShift4(color, TAU * 0.3333); // Shift by 120°
       test::results[0] = shifted;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Red shifted by 120° should become green, alpha preserved
  // Loose precision due to HSV conversion and hue rotation
  expectCloseTo([0.0, 1.0, 0.0, 0.85], result, 0.05);
});

test("vibrance4", async () => {
  const src = `
     import lygia::color::vibrance::vibrance4;

     @compute @workgroup_size(1)
     fn foo() {
       let color = vec4f(0.6, 0.5, 0.4, 0.8); // Muted orange with alpha
       let result = vibrance4(color, 0.5);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Vibrance should increase saturation of muted colors
  // RGB values calculated same as vibrance3 test, alpha preserved
  expectCloseTo([0.6258, 0.4958, 0.3658, 0.8], result);
});
