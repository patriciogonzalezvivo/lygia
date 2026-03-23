import wgsl_test::{expectNear, expectNearVec3, expectWithinVec3};
import lygia::color::blend::difference::{blendDifference, blendDifference3};
import lygia::color::blend::exclusion::{blendExclusion, blendExclusion3};
import lygia::color::blend::negation::{blendNegation, blendNegation3};
import lygia::color::blend::phoenix::{blendPhoenix, blendPhoenix3};
import lygia::color::blend::reflect::{blendReflect, blendReflect3};
import lygia::color::blend::subtract::{blendSubtract, blendSubtract3};

// vec3 blend tests

@test fn testBlendDifference3() {
  let base = vec3f(0.8, 0.3, 0.6);
  let blend = vec3f(0.5, 0.7, 0.4);
  let result = blendDifference3(base, blend);
  // Difference mode: abs(base - blend)
  expectNearVec3(result, vec3f(0.3, 0.4, 0.2));
}

@test fn testBlendExclusion3() {
  let base = vec3f(0.6, 0.4, 0.8);
  let blend = vec3f(0.3, 0.7, 0.2);
  let result = blendExclusion3(base, blend);
  // Exclusion mode: base + blend - 2 * base * blend
  expectNearVec3(result, vec3f(0.54, 0.54, 0.68));
}

@test fn testBlendNegation3() {
  let base = vec3f(0.7, 0.5, 0.3);
  let blend = vec3f(0.4, 0.6, 0.8);
  let result = blendNegation3(base, blend);
  // Negation mode: 1 - abs(1 - base - blend)
  expectNearVec3(result, vec3f(0.9, 0.9, 0.9));
}

@test fn testBlendPhoenix3() {
  let base = vec3f(0.7, 0.5, 0.3);
  let blend = vec3f(0.4, 0.6, 0.8);
  let result = blendPhoenix3(base, blend);
  // Phoenix mode: min(base, blend) - max(base, blend) + 1
  // R: 0.4 - 0.7 + 1 = 0.7
  // G: 0.5 - 0.6 + 1 = 0.9
  // B: 0.3 - 0.8 + 1 = 0.5
  expectNearVec3(result, vec3f(0.7, 0.9, 0.5));
}

@test fn testBlendReflect3() {
  let base = vec3f(0.4, 0.6, 0.2);
  let blend = vec3f(0.5, 0.3, 0.8);
  let result = blendReflect3(base, blend);
  // Reflect mode: base^2 / (1 - blend) clamped
  expectWithinVec3(result, vec3f(0.32, 0.514, 0.2), 0.01, 1e-6);
}

@test fn testBlendSubtract3() {
  let base = vec3f(0.8, 0.6, 0.5);
  let blend = vec3f(0.3, 0.4, 0.2);
  let result = blendSubtract3(base, blend);
  // Subtract mode: max(base + blend - 1, 0)
  // R: max(0.1, 0) = 0.1
  // G: max(0.0, 0) = 0.0
  // B: max(-0.3, 0) = 0.0
  expectNearVec3(result, vec3f(0.1, 0.0, 0.0));
}

// f32 blend tests

@test fn testBlendDifferenceF32() {
  expectNear(blendDifference(0.8, 0.5), 0.3);
}

@test fn testBlendExclusionF32() {
  // Exclusion: 0.6 + 0.3 - 2*0.6*0.3 = 0.9 - 0.36 = 0.54
  expectNear(blendExclusion(0.6, 0.3), 0.54);
}

@test fn testBlendNegationF32() {
  // Negation: 1 - abs(1 - 0.7 - 0.4) = 1 - abs(-0.1) = 1 - 0.1 = 0.9
  expectNear(blendNegation(0.7, 0.4), 0.9);
}

@test fn testBlendPhoenixF32() {
  // Phoenix: min(0.7,0.4) - max(0.7,0.4) + 1 = 0.4 - 0.7 + 1 = 0.7
  expectNear(blendPhoenix(0.7, 0.4), 0.7);
}

@test fn testBlendReflectF32() {
  // Reflect: 0.4^2 / (1-0.5) = 0.16/0.5 = 0.32
  expectNear(blendReflect(0.4, 0.5), 0.32);
}

@test fn testBlendSubtractF32() {
  // Subtract: max(0.8 + 0.3 - 1, 0) = max(0.1, 0) = 0.1
  expectNear(blendSubtract(0.8, 0.3), 0.1);
}
