import wgsl_test::{expect, expectNear, expectNearVec3, expectNearVec4, expectWithin, expectWithinVec3, expectWithinVec4};
import lygia::color::brightnessContrast::{brightnessContrast3, brightnessContrast4};
import lygia::color::exposure::{exposure3, exposure4};
import lygia::color::hueShiftRYB::{hueShiftRYB, hueShiftRYB4};
import lygia::color::palette::heatmap::heatmap;
import lygia::color::palette::hue::{hue, hueDefault};
import lygia::color::whiteBalance::{whiteBalance3, whiteBalance4};
import lygia::color::saturationMatrix::saturationMatrix;
import lygia::color::levels::outputRange::levelsOutputRange3;
import lygia::color::hueShift::hueShift4;
import lygia::color::vibrance::vibrance4;
import lygia::math::consts::TAU;

@test fn testBrightnessContrast3() {
  let color = vec3f(0.6, 0.5, 0.4);
  let brightness = 0.1;
  let contrast = 1.2;
  let result = brightnessContrast3(color, brightness, contrast);
  // (0.6 - 0.5) * 1.2 + 0.5 + 0.1 = 0.72
  // (0.5 - 0.5) * 1.2 + 0.5 + 0.1 = 0.6
  // (0.4 - 0.5) * 1.2 + 0.5 + 0.1 = 0.48
  expectNearVec3(result, vec3f(0.72, 0.6, 0.48));
}

@test fn testBrightnessContrast4() {
  let color = vec4f(0.6, 0.5, 0.4, 0.8);
  let brightness = 0.1;
  let contrast = 1.2;
  let result = brightnessContrast4(color, brightness, contrast);
  // RGB adjusted, alpha preserved
  expectNearVec4(result, vec4f(0.72, 0.6, 0.48, 0.8));
}

@test fn testExposure3() {
  let color = vec3f(0.5, 0.5, 0.5);
  let amount = 1.0; // +1 stop = 2x brighter
  let result = exposure3(color, amount);
  // 0.5 * 2^1 = 1.0
  expectNearVec3(result, vec3f(1.0, 1.0, 1.0));
}

@test fn testExposure4() {
  let color = vec4f(0.5, 0.5, 0.5, 0.7);
  let amount = 1.0;
  let result = exposure4(color, amount);
  // RGB doubled, alpha preserved
  expectNearVec4(result, vec4f(1.0, 1.0, 1.0, 0.7));
}

@test fn testHueShiftRYB() {
  let color = vec3f(1.0, 0.0, 0.0); // Red
  let angle = 2.0944; // 120 degrees - shift red toward yellow in RYB space
  let result = hueShiftRYB(color, angle);
  // Red shifted 120 degrees in RYB color wheel goes toward yellow
  expectNearVec3(result, vec3f(1.0, 1.0, 0.0));
}

@test fn testHueShiftRYB4() {
  let color = vec4f(1.0, 0.0, 0.0, 0.7); // Red with alpha
  let angle = 2.0944; // 120 degrees
  let result = hueShiftRYB4(color, angle);
  // RYB hue shift: Red shifted by 120 degrees toward yellow
  // Loose precision due to RYB color space conversion approximations
  expectWithinVec4(result, vec4f(1.0, 1.0, 0.0, 0.7), 0.15, 1e-6);
  // Alpha exact
  expectNear(result.w, 0.7);
}

@test fn testHeatmap() {
  let value = 0.5;
  let result = heatmap(value);
  // Heatmap formula: 1.0 - (v*2.1 - vec3(1.8,1.14,0.3))^2
  expectNearVec3(result, vec3f(0.4375, 0.9919, 0.4375));
}

@test fn testPaletteHue() {
  let x = 0.5;
  let ratio = 0.333; // neon ratio (1/3)
  let result = hue(x, ratio);
  // Physical hue palette with smoothstep
  expectWithinVec3(result, vec3f(0.0, 0.74, 0.743), 0.05, 1e-6);
}

@test fn testHueDefault() {
  // hueDefault uses default ratio of 1/3 (neon)
  let result = hueDefault(0.5);
  // Should match hue(0.5, 0.333)
  expectWithinVec3(result, vec3f(0.0, 0.74, 0.743), 0.05, 1e-6);
}

@test fn testWhiteBalance3() {
  let gray = vec3f(0.5, 0.5, 0.5);

  let warm = whiteBalance3(gray, 0.2, 0.0);    // Warm (orange/yellow)
  let cool = whiteBalance3(gray, -0.2, 0.0);   // Cool (blue)

  // Warm temperature should increase red
  expect(warm.r > 0.5);

  // Cool temperature should increase blue
  expect(cool.b > warm.b);

  // Warm should have higher red than cool
  expect(warm.r > cool.r);

  // Regression check
  expectNear(warm.r, 0.5141);
  expectNear(warm.b, 0.4585);
  expectNear(cool.r, 0.4727);
  expectNear(cool.b, 0.5919);
}

@test fn testWhiteBalance4() {
  let gray = vec4f(0.5, 0.5, 0.5, 0.8);

  let tempShift = whiteBalance4(gray, 0.2, 0.0);
  let tintMagenta = whiteBalance4(gray, 0.0, 0.1);  // Positive = magenta
  let tintGreen = whiteBalance4(gray, 0.0, -0.1);   // Negative = green

  // Temperature shift: warm should have R > B
  expect(tempShift.r > tempShift.b);

  // Tint behavior: magenta tint reduces green, green tint increases green
  expect(tintGreen.g > tintMagenta.g);
  expect(tintMagenta.g < 0.5);
  expect(tintGreen.g > 0.5);

  // Regression check
  expectNear(tempShift.r, 0.5141);
  expectNear(tempShift.b, 0.4585);
  expectNear(tintMagenta.g, 0.4872);
  expectNear(tintGreen.g, 0.5132);
}

@test fn testSaturationMatrix() {
  let amount = 1.5; // Increase saturation by 50%
  let mat = saturationMatrix(amount);
  // Test matrix by applying to an orange color
  let color = vec3f(0.8, 0.5, 0.3);
  let result = mat * vec4f(color, 1.0);

  // Saturation matrix at 1.5 should increase color saturation
  expect(result.x > 0.8); // Red should increase
  expect(result.z < 0.3); // Blue should decrease

  // Loose precision due to matrix multiplication accumulation
  expectWithinVec3(result.xyz, vec3f(0.912, 0.462, 0.162), 0.1, 1e-6);
}

@test fn testLevelsOutputRange3() {
  let color = vec3f(0.5, 0.5, 0.5);
  let minOutput = vec3f(0.2, 0.2, 0.2);
  let maxOutput = vec3f(0.8, 0.8, 0.8);
  let result = levelsOutputRange3(color, minOutput, maxOutput);
  // Middle value (0.5) should map to middle of output range (0.5)
  expectNearVec3(result, vec3f(0.5, 0.5, 0.5));
}

@test fn testHueShift4() {
  let color = vec4f(1.0, 0.0, 0.0, 0.85); // Red with alpha
  let shifted = hueShift4(color, TAU * 0.3333); // Shift by 120 degrees
  // Red shifted by 120 degrees should become green, alpha preserved
  expectWithinVec4(shifted, vec4f(0.0, 1.0, 0.0, 0.85), 0.05, 1e-3);
}

@test fn testVibrance4() {
  let color = vec4f(0.6, 0.5, 0.4, 0.8); // Muted orange with alpha
  let result = vibrance4(color, 0.5);
  // Vibrance should increase saturation of muted colors, alpha preserved
  expectNearVec4(result, vec4f(0.6258, 0.4958, 0.3658, 0.8));
}
