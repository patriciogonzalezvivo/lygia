// Native WESL tests for gamma/linear color space conversions

import wgsl_test::{expectNear, expectNearVec3, expectNearVec4};
import lygia::color::space::gamma2linear::{gamma2linear, gamma2linear3, gamma2linear4};
import lygia::color::space::linear2gamma::{linear2gamma, linear2gamma3, linear2gamma4};

@test
fn gamma2linearVec3() {
  let gamma = vec3f(0.5, 0.5, 0.5);
  // pow(0.5, 2.2) ≈ 0.2176
  expectNearVec3(gamma2linear3(gamma), vec3f(0.2176, 0.2176, 0.2176));
}

@test
fn linear2gammaVec3() {
  let linear = vec3f(0.25, 0.25, 0.25);
  // pow(0.25, 1/2.2) ≈ 0.5325
  expectNearVec3(linear2gamma3(linear), vec3f(0.5325, 0.5325, 0.5325));
}

@test
fn gamma2linearScalar() {
  // pow(0.5, 2.2) ≈ 0.2176
  expectNear(gamma2linear(0.5), 0.2176);
}

@test
fn gamma2linearVec4Alpha() {
  let gamma = vec4f(0.5, 0.5, 0.5, 0.7);
  // RGB converted, alpha unchanged
  expectNearVec4(gamma2linear4(gamma), vec4f(0.2176, 0.2176, 0.2176, 0.7));
}

@test
fn linear2gammaScalar() {
  // pow(0.25, 1/2.2) ≈ 0.5325
  expectNear(linear2gamma(0.25), 0.5325);
}

@test
fn linear2gammaVec4Alpha() {
  let linear = vec4f(0.25, 0.25, 0.25, 0.4);
  // RGB converted, alpha unchanged
  expectNearVec4(linear2gamma4(linear), vec4f(0.5325, 0.5325, 0.5325, 0.4));
}
