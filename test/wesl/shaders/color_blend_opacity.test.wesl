// Native WESL tests for blend modes with opacity

import wgsl_test::{expectNearVec3};
import lygia::color::blend::difference::blendDifference3Opacity;
import lygia::color::blend::exclusion::blendExclusion3Opacity;
import lygia::color::blend::negation::blendNegation3Opacity;
import lygia::color::blend::phoenix::blendPhoenix3Opacity;
import lygia::color::blend::reflect::blendReflect3Opacity;
import lygia::color::blend::subtract::blendSubtract3Opacity;
import lygia::color::blend::glow::blendGlow3Opacity;
import lygia::color::blend::colorBurn::blendColorBurn3Opacity;
import lygia::color::blend::colorDodge::blendColorDodge3Opacity;
import lygia::color::blend::linearBurn::blendLinearBurn3Opacity;
import lygia::color::blend::linearDodge::blendLinearDodge3Opacity;

@test(blendDifference3Opacity_basic)
fn testBlendDifference3Opacity() {
  let base = vec3f(0.8, 0.3, 0.6);
  let blend = vec3f(0.5, 0.7, 0.4);
  // Full blend: [0.3, 0.4, 0.2]
  // At 0.5: [0.3*0.5+0.8*0.5, 0.4*0.5+0.3*0.5, 0.2*0.5+0.6*0.5]
  expectNearVec3(blendDifference3Opacity(base, blend, 0.5), vec3f(0.55, 0.35, 0.4));
}

@test(blendDifference3Opacity_zero)
fn testBlendDifference3OpacityZero() {
  let base = vec3f(0.8, 0.3, 0.6);
  let blend = vec3f(0.5, 0.7, 0.4);
  // At opacity 0, should return base unchanged
  expectNearVec3(blendDifference3Opacity(base, blend, 0.0), vec3f(0.8, 0.3, 0.6));
}

@test(blendExclusion3Opacity_basic)
fn testBlendExclusion3Opacity() {
  let base = vec3f(0.6, 0.4, 0.8);
  let blend = vec3f(0.3, 0.7, 0.2);
  // Full blend: [0.54, 0.54, 0.68]
  // At 0.5: [0.54*0.5+0.6*0.5, 0.54*0.5+0.4*0.5, 0.68*0.5+0.8*0.5]
  expectNearVec3(blendExclusion3Opacity(base, blend, 0.5), vec3f(0.57, 0.47, 0.74));
}

@test(blendNegation3Opacity_basic)
fn testBlendNegation3Opacity() {
  let base = vec3f(0.7, 0.5, 0.3);
  let blend = vec3f(0.4, 0.6, 0.8);
  // Full blend: [0.9, 0.9, 0.9]
  // At 0.5: [0.9*0.5+0.7*0.5, 0.9*0.5+0.5*0.5, 0.9*0.5+0.3*0.5]
  expectNearVec3(blendNegation3Opacity(base, blend, 0.5), vec3f(0.8, 0.7, 0.6));
}

@test(blendPhoenix3Opacity_basic)
fn testBlendPhoenix3Opacity() {
  let base = vec3f(0.7, 0.5, 0.3);
  let blend = vec3f(0.4, 0.6, 0.8);
  // Full blend: [0.7, 0.9, 0.5]
  // At 0.5: [0.7*0.5+0.7*0.5, 0.9*0.5+0.5*0.5, 0.5*0.5+0.3*0.5]
  expectNearVec3(blendPhoenix3Opacity(base, blend, 0.5), vec3f(0.7, 0.7, 0.4));
}

@test(blendReflect3Opacity_basic)
fn testBlendReflect3Opacity() {
  let base = vec3f(0.4, 0.6, 0.2);
  let blend = vec3f(0.5, 0.3, 0.8);
  // Reflect: base^2 / (1 - blend)
  // Full blend: [0.16/0.5, 0.36/0.7, 0.04/0.2] = [0.32, 0.5142857, 0.2]
  // At 0.5: lerp(base, full, 0.5)
  let fullBlend = vec3f(0.16/0.5, 0.36/0.7, 0.04/0.2);
  expectNearVec3(blendReflect3Opacity(base, blend, 0.5), mix(base, fullBlend, vec3f(0.5)));
}

@test(blendSubtract3Opacity_basic)
fn testBlendSubtract3Opacity() {
  let base = vec3f(0.8, 0.6, 0.5);
  let blend = vec3f(0.3, 0.4, 0.2);
  // Full blend: [0.1, 0.0, 0.0] but actually [0.5, 0.2, 0.3] (max(base-blend,0))
  // At 0.5: [0.1*0.5+0.8*0.5, 0.0*0.5+0.6*0.5, 0.0*0.5+0.5*0.5]
  expectNearVec3(blendSubtract3Opacity(base, blend, 0.5), vec3f(0.45, 0.3, 0.25));
}

@test(blendGlow3Opacity_basic)
fn testBlendGlow3Opacity() {
  let base = vec3f(0.4, 0.6, 0.2);
  let blend = vec3f(0.5, 0.3, 0.8);
  // Glow: blend^2 / (1 - base)
  // Full blend: [0.25/0.6, 0.09/0.4, 0.64/0.8] = [0.4166..., 0.225, 0.8]
  // At 0.5: lerp(base, full, 0.5)
  let fullBlend = vec3f(0.25/0.6, 0.09/0.4, 0.64/0.8);
  expectNearVec3(blendGlow3Opacity(base, blend, 0.5), mix(base, fullBlend, vec3f(0.5)));
}

@test(blendColorBurn3Opacity_basic)
fn testBlendColorBurn3Opacity() {
  let base = vec3f(0.6, 0.5, 0.4);
  let blend = vec3f(0.3, 0.4, 0.5);
  // Full blend: [0.0, 0.0, 0.0]
  // At 0.5: [0.0*0.5+0.6*0.5, 0.0*0.5+0.5*0.5, 0.0*0.5+0.4*0.5]
  expectNearVec3(blendColorBurn3Opacity(base, blend, 0.5), vec3f(0.3, 0.25, 0.2));
}

@test(blendColorDodge3Opacity_basic)
fn testBlendColorDodge3Opacity() {
  let base = vec3f(0.4, 0.5, 0.6);
  let blend = vec3f(0.3, 0.4, 0.5);
  // Full blend: [0.571, 0.833, 1.0]
  // At 0.5: [0.571*0.5+0.4*0.5, 0.833*0.5+0.5*0.5, 1.0*0.5+0.6*0.5]
  // Use exact values: 4/7*0.5+0.4*0.5, 5/6*0.5+0.5*0.5, 1.0*0.5+0.6*0.5
  expectNearVec3(blendColorDodge3Opacity(base, blend, 0.5), vec3f(2.0/7.0+0.2, 5.0/12.0+0.25, 0.8));
}

@test(blendLinearBurn3Opacity_basic)
fn testBlendLinearBurn3Opacity() {
  let base = vec3f(0.6, 0.5, 0.7);
  let blend = vec3f(0.4, 0.3, 0.2);
  // Full blend: [0.0, 0.0, 0.0]
  // At 0.5: [0.0*0.5+0.6*0.5, 0.0*0.5+0.5*0.5, 0.0*0.5+0.7*0.5]
  expectNearVec3(blendLinearBurn3Opacity(base, blend, 0.5), vec3f(0.3, 0.25, 0.35));
}

@test(blendLinearDodge3Opacity_basic)
fn testBlendLinearDodge3Opacity() {
  let base = vec3f(0.4, 0.5, 0.6);
  let blend = vec3f(0.3, 0.2, 0.1);
  // Full blend: [0.7, 0.7, 0.7]
  // At 0.5: [0.7*0.5+0.4*0.5, 0.7*0.5+0.5*0.5, 0.7*0.5+0.6*0.5]
  expectNearVec3(blendLinearDodge3Opacity(base, blend, 0.5), vec3f(0.55, 0.6, 0.65));
}
