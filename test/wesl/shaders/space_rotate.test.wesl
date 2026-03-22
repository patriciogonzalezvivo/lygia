import wgsl_test::{expect, expectNear, expectNearVec2, expectNearVec4, expectWithin};
import lygia::math::consts::{HALF_PI, PI};
import lygia::space::rotateX::{rotateX3, rotateX3_c};
import lygia::space::rotateY::{rotateY3, rotateY3_c};
import lygia::space::rotateZ::{rotateZ3, rotateZ3_c};
import lygia::space::rotate::{rotate, rotate_c, rotate3};
import lygia::space::bracketing::{bracketing, BracketingResult};

@test fn testRotateX3() {
  let result = rotateX3(vec3f(1.0, 1.0, 0.0), HALF_PI); // 90 degrees
  // Rotating (1,1,0) around X by 90deg -> x stays 1, y->0, z->-1
  expectNearVec4(vec4f(result, 0.0), vec4f(1.0, 0.0, -1.0, 0.0));
}

@test fn testRotateY3() {
  let result = rotateY3(vec3f(1.0, 1.0, 0.0), HALF_PI); // 90 degrees
  expectNearVec4(vec4f(result, 0.0), vec4f(0.0, 1.0, -1.0, 0.0));
}

@test fn testRotateZ3() {
  let result = rotateZ3(vec3f(1.0, 0.0, 1.0), HALF_PI); // 90 degrees
  // Rotating (1,0,1) around Z by 90deg -> x->0, y->-1, z stays 1
  expectNearVec4(vec4f(result, 0.0), vec4f(0.0, -1.0, 1.0, 0.0));
}

@test fn testRotate2d() {
  // Rotate vec2 by 90 degrees around center (0.5, 0.5)
  let result = rotate(vec2f(1.0, 0.5), HALF_PI);
  // offset = (0.5, 0.0), rotated -> (0.0, 0.5), + center -> (0.5, 1.0)
  expectNearVec2(result, vec2f(0.5, 1.0));
}

@test fn testRotate2dCustomCenter() {
  // Rotate vec2 by 90 degrees around custom center (0, 0)
  let result = rotate_c(vec2f(1.0, 0.0), HALF_PI, vec2f(0.0));
  // Rotating (1.0, 0.0) by 90deg around (0, 0) -> (0, 1)
  expectNearVec2(result, vec2f(0.0, 1.0));
}

@test fn testRotate3d() {
  // Rotate vec3 by 90 degrees around Z axis from origin
  let result = rotate3(vec3f(1.0, 0.0, 0.0), HALF_PI, vec3f(0.0, 0.0, 1.0));
  let len = length(result);
  expectNear(len, 1.0);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.0, -1.0, 0.0, 0.0));
}

@test fn testRotate2dWithCenter2D() {
  // Rotate around custom center (0.3, 0.3) using _c variant
  let center = vec2f(0.3, 0.3);
  let result = rotate_c(vec2f(0.8, 0.3), HALF_PI, center);
  // Offset: (0.5, 0.0), rotated 90deg -> (0.0, 0.5), result: (0.3, 0.8)
  expectNearVec2(result, vec2f(0.3, 0.8));
}

@test fn testRotateX3CustomCenter() {
  let center = vec3f(0.5, 0.5, 0.5);
  let result = rotateX3_c(vec3f(1.0, 1.5, 0.5), HALF_PI, center);
  // Offset from center: (0.5, 1.0, 0.0)
  // Rotate X by 90deg: (0.5, 0.0, -1.0) + center = (1.0, 0.5, -0.5)
  expectNearVec4(vec4f(result, 0.0), vec4f(1.0, 0.5, -0.5, 0.0));
}

@test fn testRotateY3CustomCenter() {
  let center = vec3f(0.5, 0.5, 0.5);
  let result = rotateY3_c(vec3f(1.5, 1.0, 0.5), HALF_PI, center);
  // Offset from center: (1.0, 0.5, 0.0)
  // Rotate Y by 90deg: (0.0, 0.5, -1.0) + center = (0.5, 1.0, -0.5)
  expectNearVec4(vec4f(result, 0.0), vec4f(0.5, 1.0, -0.5, 0.0));
}

@test fn testRotateZ3CustomCenter() {
  let center = vec3f(0.5, 0.5, 0.5);
  let result = rotateZ3_c(vec3f(1.5, 0.5, 1.0), HALF_PI, center);
  // Offset from center: (1.0, 0.0, 0.5)
  // Rotate Z by 90deg: (0.0, -1.0, 0.5) + center = (0.5, -0.5, 1.0)
  expectNearVec4(vec4f(result, 0.0), vec4f(0.5, -0.5, 1.0, 0.0));
}

@test fn testBracketing() {
  // Test 1: Canonical angle (should snap exactly, blendAlpha ~ 0)
  let r1 = bracketing(vec2f(1.0, 0.0)); // angle = 0

  // Test 2: Between canonical angles (PI/40 is halfway between PI/20 steps)
  let angle_between = PI / 40.0;
  let r2 = bracketing(vec2f(cos(angle_between), sin(angle_between)));

  expectNear(r1.vAxis0.x, 1.0); // canonical axis
  expectWithin(r1.blendAlpha, 0.0, 0.01, 1e-4); // snaps to canonical
  expect(r2.blendAlpha > 0.2); // in valid range for in-between angle
  expect(r2.blendAlpha < 0.8);
  expect(abs(r2.vAxis0.x - r2.vAxis1.x) > 0.01); // vAxis0 and vAxis1 differ

  // Exact regression values
  expectNear(r1.vAxis0.x, 1.0);
  expectWithin(r1.blendAlpha, 0.0, 0.01, 1e-4);
  expectWithin(r2.blendAlpha, 0.5, 0.01, 1e-4);
  expectWithin(abs(r2.vAxis0.x - r2.vAxis1.x), 0.01231, 0.01, 1e-4);
}
