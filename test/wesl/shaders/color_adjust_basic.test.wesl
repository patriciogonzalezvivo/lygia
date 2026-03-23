import wgsl_test::{expect, expectNear, expectNearVec3, expectNearVec4, expectWithinVec3};
import lygia::color::desaturate::{desaturate, desaturate4};
import lygia::color::brightnessMatrix::brightnessMatrix;
import lygia::color::contrast::{contrast, contrast3, contrast4};
import lygia::color::contrastMatrix::contrastMatrix;
import lygia::color::brightnessContrast::brightnessContrast;
import lygia::color::exposure::exposure;
import lygia::color::hueShift::hueShift;
import lygia::color::vibrance::vibrance3;
import lygia::math::consts::TAU;


@test fn testDesaturate() {
  let color = vec3f(1.0, 0.5, 0.0); // Orange color
  let result = desaturate(color, 0.5); // 50% desaturation
  // Gray value is luminance: 1*0.3 + 0.5*0.59 + 0*0.11 = 0.595
  // 50% blend: (1+0.595)/2 = 0.7975, (0.5+0.595)/2 = 0.5475, (0+0.595)/2 = 0.2975
  expectNearVec3(result, vec3f(0.7975, 0.5475, 0.2975));
}

@test fn testDesaturate4() {
  let color = vec4f(1.0, 0.5, 0.0, 0.8); // Orange with alpha
  let result = desaturate4(color, 0.5); // 50% desaturation
  // RGB same as desaturate, alpha should remain 0.8
  expectNearVec4(result, vec4f(0.7975, 0.5475, 0.2975, 0.8));
}

@test fn testBrightnessMatrix() {
  let matrix = brightnessMatrix(0.2); // 20% brightness increase
  // The translation part should be (0.2, 0.2, 0.2, 1.0)
  expectNearVec4(vec4f(matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]), vec4f(0.2, 0.2, 0.2, 1.0));
}

@test fn testContrast() {
  let result = contrast(0.7, 1.5);
  // (0.7 - 0.5) * 1.5 + 0.5 = 0.2 * 1.5 + 0.5 = 0.8
  expectNear(result, 0.8);
}

@test fn testContrast3() {
  let color = vec3f(0.8, 0.6, 0.4);
  let result = contrast3(color, 2.0);
  // r: (0.8 - 0.5) * 2 + 0.5 = 1.1
  // g: (0.6 - 0.5) * 2 + 0.5 = 0.7
  // b: (0.4 - 0.5) * 2 + 0.5 = 0.3
  expectNearVec3(result, vec3f(1.1, 0.7, 0.3));
}

@test fn testContrastMatrix() {
  let matrix = contrastMatrix(1.5);
  // Diagonal should be 1.5, translation should be (1-1.5)*0.5 = -0.25
  expectNearVec4(vec4f(matrix[0][0], matrix[1][1], matrix[2][2], matrix[3][0]), vec4f(1.5, 1.5, 1.5, -0.25));
}

@test fn testBrightnessContrast() {
  let value = 0.7;
  let brightness = 0.1;
  let cont = 1.5;
  let result = brightnessContrast(value, brightness, cont);
  // (0.7 - 0.5) * 1.5 + 0.5 + 0.1 = 0.2 * 1.5 + 0.6 = 0.9
  expectNear(result, 0.9);
}

@test fn testContrast4() {
  let color = vec4f(0.8, 0.6, 0.4, 0.9);
  let result = contrast4(color, 1.5);
  // r: (0.8 - 0.5) * 1.5 + 0.5 = 0.95
  // g: (0.6 - 0.5) * 1.5 + 0.5 = 0.65
  // b: (0.4 - 0.5) * 1.5 + 0.5 = 0.35
  // a: preserved at 0.9
  expectNearVec4(result, vec4f(0.95, 0.65, 0.35, 0.9));
}

@test fn testExposure() {
  let value = 0.25;
  let amount = 2.0; // +2 stops = 4x brighter
  let result = exposure(value, amount);
  // 0.25 * 2^2 = 0.25 * 4 = 1.0
  expectNear(result, 1.0);
}

@test fn testHueShift() {
  let rgb = vec3f(1.0, 0.0, 0.0); // Red
  let shifted = hueShift(rgb, TAU * 0.3333); // Shift by 120 degrees
  // Red shifted by 120 degrees should become green
  expectWithinVec3(shifted, vec3f(0.0, 1.0, 0.0), 0.001, 1e-3);
}

@test fn testVibrance() {
  // Orange color with low saturation (muted)
  let rgb = vec3f(0.6, 0.5, 0.4);
  let result = vibrance3(rgb, 0.5);
  // Vibrance formula applied
  expectNearVec3(result, vec3f(0.6258, 0.4958, 0.3658));
}

@test fn testVibranceSelective() {
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

  // Vibrance should increase muted saturation more than saturated colors
  expect(muted_sat_change > saturated_sat_change);

  // Muted color should have increased saturation (change > 1.0)
  expect(muted_sat_change > 1.0);

  // Negative vibrance should move colors toward gray
  expectNear(desaturated.r, 0.833);
  expectNear(desaturated.g, 0.393);
}
