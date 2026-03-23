import wgsl_test::{expect, expectNear, expectNearVec3, expectWithinVec3, expectWithin};
import lygia::color::blend::hardLight::{blendHardLight, blendHardLight3};
import lygia::color::blend::overlay::{blendOverlay, blendOverlay3};
import lygia::color::blend::softLight::{blendSoftLight, blendSoftLight3};
import lygia::color::blend::vividLight::{blendVividLight, blendVividLight3};
import lygia::color::blend::pinLight::{blendPinLight, blendPinLight3};
import lygia::color::blend::linearLight::{blendLinearLight, blendLinearLight3};
import lygia::color::blend::hardMix::{blendHardMix, blendHardMix3};
import lygia::color::blend::glow::{blendGlow, blendGlow3};

// vec3 blend tests

@test fn testBlendHardLight3() {
  let base = vec3f(0.4, 0.6, 0.8);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendHardLight3(base, blend);
  // Hard light is overlay with base and blend swapped
  expectNearVec3(result, vec3f(0.24, 0.6, 0.88));
}

@test fn testBlendOverlay3() {
  let base = vec3f(0.4, 0.6, 0.8);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendOverlay3(base, blend);
  // Overlay mode: conditional multiply/screen
  expectNearVec3(result, vec3f(0.24, 0.6, 0.88));
}

@test fn testBlendSoftLight3() {
  let base = vec3f(0.5, 0.6, 0.4);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendSoftLight3(base, blend);
  // Soft light mode
  expectNearVec3(result, vec3f(0.4, 0.6, 0.493));
}

@test fn testBlendVividLight3() {
  let base = vec3f(0.5, 0.6, 0.4);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendVividLight3(base, blend);
  // Vivid light combines color dodge and color burn
  expectWithinVec3(result, vec3f(0.167, 0.6, 0.667), 0.01, 1e-6);
}

@test fn testBlendPinLight3() {
  // Pin light: if blend < 0.5: darken(base, 2*blend), else: lighten(base, 2*blend-1)
  let base = vec3f(0.3, 0.7, 0.5);
  let blend = vec3f(0.1, 0.9, 0.5);
  let result = blendPinLight3(base, blend);
  // R: blend=0.1<0.5 -> darken(0.3, 2*0.1) = min(0.3, 0.2) = 0.2
  // G: blend=0.9>0.5 -> lighten(0.7, 2*0.9-1) = max(0.7, 0.8) = 0.8
  // B: blend=0.5 -> edge case, should be close to base
  expectNearVec3(result, vec3f(0.2, 0.8, 0.5));
}

@test fn testBlendLinearLight3() {
  let base = vec3f(0.4, 0.5, 0.6);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendLinearLight3(base, blend);
  // Linear light is linear dodge + linear burn
  expectWithinVec3(result, vec3f(0.0, 0.5, 1.0), 1e-4, 1e-4);
}

@test fn testBlendHardMix3() {
  let base = vec3f(0.4, 0.6, 0.8);
  let blend = vec3f(0.3, 0.5, 0.2);
  let result = blendHardMix3(base, blend);
  // Hard mix produces posterized output
  expectWithinVec3(result, vec3f(0.0, 1.0, 1.0), 1e-4, 1e-4);
}

@test fn testBlendGlow3() {
  let base = vec3f(0.4, 0.6, 0.2);
  let blend = vec3f(0.5, 0.3, 0.8);
  let result = blendGlow3(base, blend);
  // Glow is reflect with base and blend swapped
  expectWithinVec3(result, vec3f(0.417, 0.225, 0.8), 0.01, 1e-6);
}

// f32 blend tests

@test fn testBlendOverlayF32() {
  let result = blendOverlay(0.4, 0.3);
  // Overlay: base<0.5: 2*base*blend = 2*0.4*0.3 = 0.24
  expectNear(result, 0.24);
}

@test fn testBlendSoftLightF32() {
  let result = blendSoftLight(0.5, 0.3);
  // Soft light: blend<0.5: 2*0.5*0.3 + 0.25*(1-0.6) = 0.3 + 0.1 = 0.4
  expectNear(result, 0.4);
}

@test fn testBlendHardLightF32() {
  let result = blendHardLight(0.4, 0.3);
  // Hard light: blend<0.5: 2*base*blend = 2*0.4*0.3 = 0.24
  expectNear(result, 0.24);
}

@test fn testBlendVividLightF32() {
  let result = blendVividLight(0.5, 0.3);
  // Vivid light: blend<0.5: colorBurn: 1-(1-0.5)/(2*0.3) = 1-0.5/0.6 = 0.167
  expectWithin(result, 0.167, 0.01, 1e-6);
}

@test fn testBlendPinLightF32() {
  // Pin light: blend < 0.5 ? darken : lighten
  let result1 = blendPinLight(0.3, 0.1);  // darken: min(0.3, 0.2) = 0.2
  let result2 = blendPinLight(0.7, 0.9);  // lighten: max(0.7, 0.8) = 0.8
  expectNear(result1, 0.2);
  expectNear(result2, 0.8);
}

@test fn testBlendLinearLightF32() {
  let result = blendLinearLight(0.4, 0.3);
  // Linear light: base + 2*blend - 1 = 0.4 + 0.6 - 1 = 0
  expectWithin(result, 0.0, 1e-4, 1e-4);
}

@test fn testBlendHardMixF32() {
  let result = blendHardMix(0.4, 0.3);
  // Hard mix produces 0 or 1
  expectWithin(result, 0.0, 1e-4, 1e-4);
}

@test fn testBlendGlowF32() {
  let result = blendGlow(0.4, 0.5);
  // Glow is reflect with swapped args: 0.5^2/(1-0.4) = 0.25/0.6 = 0.417
  expectWithin(result, 0.417, 0.01, 1e-6);
}
