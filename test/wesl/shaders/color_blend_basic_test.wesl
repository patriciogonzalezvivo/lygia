// Native WESL tests for basic blend modes

import wgsl_test::{expectNear, expectNearVec3};
import lygia::color::blend::add::{blendAdd, blendAdd3};
import lygia::color::blend::multiply::{blendMultiply, blendMultiply3};
import lygia::color::blend::screen::{blendScreen, blendScreen3};
import lygia::color::blend::average::{blendAverage, blendAverage3};
import lygia::color::blend::lighten::{blendLighten, blendLighten3};

// vec3 blend tests

@test(blendAdd3_basic)
fn testBlendAdd3() {
  let base = vec3f(0.5, 0.3, 0.2);
  let blend = vec3f(0.2, 0.4, 0.6);
  // Add mode: min(base + blend, 1.0)
  expectNearVec3(blendAdd3(base, blend), vec3f(0.7, 0.7, 0.8));
}

@test(blendMultiply3_basic)
fn testBlendMultiply3() {
  let base = vec3f(0.8, 0.6, 0.4);
  let blend = vec3f(0.5, 0.5, 0.5);
  // Multiply mode: base * blend
  expectNearVec3(blendMultiply3(base, blend), vec3f(0.4, 0.3, 0.2));
}

@test(blendScreen3_basic)
fn testBlendScreen3() {
  let base = vec3f(0.4, 0.5, 0.6);
  let blend = vec3f(0.3, 0.4, 0.5);
  // Screen mode: 1 - (1 - base) * (1 - blend)
  expectNearVec3(blendScreen3(base, blend), vec3f(0.58, 0.7, 0.8));
}

@test(blendAverage3_basic)
fn testBlendAverage3() {
  let base = vec3f(0.6, 0.4, 0.8);
  let blend = vec3f(0.4, 0.8, 0.2);
  // Average mode: (base + blend) / 2
  expectNearVec3(blendAverage3(base, blend), vec3f(0.5, 0.6, 0.5));
}

@test(blendLighten3_basic)
fn testBlendLighten3() {
  let base = vec3f(0.6, 0.4, 0.5);
  let blend = vec3f(0.3, 0.7, 0.5);
  // Lighten mode: max(base, blend)
  expectNearVec3(blendLighten3(base, blend), vec3f(0.6, 0.7, 0.5));
}

// f32 blend tests

@test(blendAdd_f32)
fn testBlendAddF32() {
  expectNear(blendAdd(0.5, 0.3), 0.8);
}

@test(blendMultiply_f32)
fn testBlendMultiplyF32() {
  expectNear(blendMultiply(0.8, 0.5), 0.4);
}

@test(blendScreen_f32)
fn testBlendScreenF32() {
  // Screen: 1 - (1-0.4)*(1-0.5) = 1 - 0.6*0.5 = 1 - 0.3 = 0.7
  expectNear(blendScreen(0.4, 0.5), 0.7);
}

@test(blendAverage_f32)
fn testBlendAverageF32() {
  expectNear(blendAverage(0.6, 0.4), 0.5);
}

@test(blendLighten_f32)
fn testBlendLightenF32() {
  expectNear(blendLighten(0.6, 0.3), 0.6);
}
