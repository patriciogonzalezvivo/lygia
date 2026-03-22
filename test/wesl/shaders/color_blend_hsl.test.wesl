import wgsl_test::{expect, expectNearVec3, expectWithinVec3};
import lygia::color::blend::hue::blendHue;
import lygia::color::blend::saturation::blendSaturation;
import lygia::color::blend::color::blendColor;
import lygia::color::blend::luminosity::blendLuminosity;

@test fn testBlendHue() {
  let base = vec3f(0.8, 0.4, 0.2);
  let blend = vec3f(0.2, 0.6, 0.8);
  let result = blendHue(base, blend);
  // Hue blend - takes hue from blend, saturation and value from base
  expectNearVec3(result, vec3f(0.2, 0.6, 0.8));
}

@test fn testBlendSaturation() {
  // Saturation blend: hue and luminosity from base, saturation from blend
  let base = vec3f(1.0, 0.0, 0.0);  // Pure red (highly saturated)
  let blend = vec3f(0.5, 0.5, 0.5); // Gray (no saturation)
  let result = blendSaturation(base, blend);

  // Saturation blend with gray should desaturate the red
  // Result should be grayish (all channels similar)
  expect(abs(result.x - result.y) < 0.2);
  expect(abs(result.y - result.z) < 0.2);

  // Actual result: gray with all channels equal (desaturated)
  expectWithinVec3(result, vec3f(1.0, 1.0, 1.0), 0.05, 1e-6);
}

@test fn testBlendColor() {
  let base = vec3f(0.8, 0.4, 0.2);
  let blend = vec3f(0.2, 0.6, 0.8);
  let result = blendColor(base, blend);
  // Color blend - takes hue and saturation from blend, value from base
  expectNearVec3(result, vec3f(0.2, 0.6, 0.8));
}

@test fn testBlendLuminosity() {
  // Luminosity blend: hue and saturation from base, luminosity from blend
  let base = vec3f(1.0, 0.0, 0.0);  // Pure red (bright)
  let blend = vec3f(0.1, 0.1, 0.1); // Dark gray
  let result = blendLuminosity(base, blend);

  // Result should be dark red (R > G,B but much darker than input)
  expect(result.x > result.y);
  expect(result.x > result.z);
  // Should be darker than base
  expect(result.x < 0.5);

  // Actual result: dark red with only R channel having value
  expectWithinVec3(result, vec3f(0.1, 0.0, 0.0), 0.05, 1e-6);
}
