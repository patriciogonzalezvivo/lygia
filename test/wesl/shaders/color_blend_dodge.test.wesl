// Native WESL tests for dodge blend modes

import wgsl_test::{expectNear, expectNearVec3};
import lygia::color::blend::colorDodge::{blendColorDodge, blendColorDodge3};
import lygia::color::blend::linearDodge::{blendLinearDodge, blendLinearDodge3};

// vec3 blend tests

@test(blendColorDodge3_basic)
fn testBlendColorDodge3() {
  let base = vec3f(0.4, 0.5, 0.6);
  let blend = vec3f(0.3, 0.4, 0.5);
  // Color dodge mode: base / (1 - blend) = 0.4/0.7, 0.5/0.6, 0.6/0.5
  expectNearVec3(blendColorDodge3(base, blend), vec3f(4.0/7.0, 5.0/6.0, 1.0));
}

@test(blendLinearDodge3_basic)
fn testBlendLinearDodge3() {
  let base = vec3f(0.4, 0.5, 0.6);
  let blend = vec3f(0.3, 0.2, 0.1);
  // Linear dodge mode: min(base + blend, 1)
  expectNearVec3(blendLinearDodge3(base, blend), vec3f(0.7, 0.7, 0.7));
}

// f32 blend tests

@test(blendColorDodge_f32)
fn testBlendColorDodgeF32() {
  // Color dodge: 0.4/(1-0.3) = 0.4/0.7 = 4/7
  expectNear(blendColorDodge(0.4, 0.3), 4.0/7.0);
}

@test(blendLinearDodge_f32)
fn testBlendLinearDodgeF32() {
  expectNear(blendLinearDodge(0.4, 0.3), 0.7);
}
