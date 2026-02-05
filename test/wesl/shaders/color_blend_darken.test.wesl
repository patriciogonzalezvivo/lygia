// Native WESL tests for darken blend modes

import wgsl_test::{expectNear, expectNearVec3};
import lygia::color::blend::darken::{blendDarken, blendDarken3};
import lygia::color::blend::colorBurn::{blendColorBurn, blendColorBurn3};
import lygia::color::blend::linearBurn::{blendLinearBurn, blendLinearBurn3};

// vec3 blend tests

@test(blendDarken3_basic)
fn testBlendDarken3() {
  let base = vec3f(0.6, 0.4, 0.5);
  let blend = vec3f(0.3, 0.7, 0.5);
  // Darken mode: min(base, blend)
  expectNearVec3(blendDarken3(base, blend), vec3f(0.3, 0.4, 0.5));
}

@test(blendColorBurn3_basic)
fn testBlendColorBurn3() {
  let base = vec3f(0.6, 0.5, 0.4);
  let blend = vec3f(0.3, 0.4, 0.5);
  // Color burn mode: 1 - (1 - base) / blend
  expectNearVec3(blendColorBurn3(base, blend), vec3f(0.0, 0.0, 0.0));
}

@test(blendLinearBurn3_basic)
fn testBlendLinearBurn3() {
  let base = vec3f(0.6, 0.5, 0.7);
  let blend = vec3f(0.4, 0.3, 0.2);
  // Linear burn mode: max(base + blend - 1, 0)
  expectNearVec3(blendLinearBurn3(base, blend), vec3f(0.0, 0.0, 0.0));
}

// f32 blend tests

@test(blendDarken_f32)
fn testBlendDarkenF32() {
  expectNear(blendDarken(0.6, 0.3), 0.3);
}

@test(blendColorBurn_f32)
fn testBlendColorBurnF32() {
  // Color burn: 1 - (1-0.6)/0.3 = 1 - 0.4/0.3 = 1 - 1.333 = clamped to 0
  expectNear(blendColorBurn(0.6, 0.3), 0.0);
}

@test(blendLinearBurn_f32)
fn testBlendLinearBurnF32() {
  // Linear burn: max(0.6+0.4-1, 0) = 0
  expectNear(blendLinearBurn(0.6, 0.4), 0.0);
}
