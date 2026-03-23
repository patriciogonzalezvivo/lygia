import wgsl_test::{expectNearVec3, expectWithinVec3};
import lygia::color::blend::overlay::blendOverlay3Opacity;
import lygia::color::blend::softLight::blendSoftLight3Opacity;
import lygia::color::blend::hardLight::blendHardLight3Opacity;
import lygia::color::blend::vividLight::blendVividLight3Opacity;
import lygia::color::blend::pinLight::blendPinLight3Opacity;
import lygia::color::blend::linearLight::blendLinearLight3Opacity;
import lygia::color::blend::hardMix::blendHardMix3Opacity;

@test fn testBlendOverlay3Opacity() {
  let base = vec3f(0.4, 0.6, 0.8);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendOverlay3Opacity(base, blend, 0.5);
  // Full blend: [0.24, 0.6, 0.88]
  // At 0.5: [0.24*0.5+0.4*0.5, 0.6*0.5+0.6*0.5, 0.88*0.5+0.8*0.5]
  expectNearVec3(result, vec3f(0.32, 0.6, 0.84));
}

@test fn testBlendSoftLight3Opacity() {
  let base = vec3f(0.5, 0.6, 0.4);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendSoftLight3Opacity(base, blend, 0.5);
  // Full blend: [0.4, 0.6, 0.493]
  // At 0.5: [0.45, 0.6, 0.447]
  expectWithinVec3(result, vec3f(0.45, 0.6, 0.447), 0.01, 1e-6);
}

@test fn testBlendHardLight3Opacity() {
  let base = vec3f(0.4, 0.6, 0.8);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendHardLight3Opacity(base, blend, 0.5);
  // Full blend: [0.24, 0.6, 0.88]
  // At 0.5: [0.32, 0.6, 0.84]
  expectNearVec3(result, vec3f(0.32, 0.6, 0.84));
}

@test fn testBlendVividLight3Opacity() {
  let base = vec3f(0.5, 0.6, 0.4);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendVividLight3Opacity(base, blend, 0.5);
  // Full blend: [0.167, 0.6, 0.667]
  // At 0.5: [0.334, 0.6, 0.534]
  expectWithinVec3(result, vec3f(0.334, 0.6, 0.534), 0.01, 1e-6);
}

@test fn testBlendPinLight3Opacity() {
  let base = vec3f(0.3, 0.7, 0.5);
  let blend = vec3f(0.1, 0.9, 0.5);
  let result = blendPinLight3Opacity(base, blend, 0.5);
  // Full blend: [0.2, 0.8, 0.5]
  // At 0.5: [0.25, 0.75, 0.5]
  expectNearVec3(result, vec3f(0.25, 0.75, 0.5));
}

@test fn testBlendLinearLight3Opacity() {
  let base = vec3f(0.4, 0.5, 0.6);
  let blend = vec3f(0.3, 0.5, 0.7);
  let result = blendLinearLight3Opacity(base, blend, 0.5);
  // Full blend: [0.0, 0.5, 1.0]
  // At 0.5: [0.2, 0.5, 0.8]
  expectNearVec3(result, vec3f(0.2, 0.5, 0.8));
}

@test fn testBlendHardMix3Opacity() {
  let base = vec3f(0.4, 0.6, 0.8);
  let blend = vec3f(0.3, 0.5, 0.2);
  let result = blendHardMix3Opacity(base, blend, 0.5);
  // Full blend: [0.0, 1.0, 1.0]
  // At 0.5: [0.2, 0.8, 0.9]
  expectNearVec3(result, vec3f(0.2, 0.8, 0.9));
}
