import wgsl_test::{expectNearVec2, expectNearVec4};
import lygia::space::scale::{scale2, scale2_f, scale2_c, scale3, scale3_c};
import lygia::math::scale2d::scale2dXY;

@test fn testScale2() {
  let result = scale2(vec2f(0.75, 0.25), vec2f(2.0, 0.5));
  // Scale (0.75, 0.25) by (2.0, 0.5) around center (0.5, 0.5)
  // (0.75 - 0.5) * 2.0 + 0.5 = 1.0
  // (0.25 - 0.5) * 0.5 + 0.5 = 0.375
  expectNearVec2(result, vec2f(1.0, 0.375));
}

@test fn testScale2f() {
  let result = scale2_f(vec2f(0.75, 0.25), 2.0);
  // Scale (0.75, 0.25) by 2.0 around center (0.5, 0.5)
  // (0.75 - 0.5) * 2.0 + 0.5 = 1.0
  // (0.25 - 0.5) * 2.0 + 0.5 = 0.0
  expectNearVec2(result, vec2f(1.0, 0.0));
}

@test fn testScale2dXY() {
  // Test non-uniform scale matrix (scale X by 2.0, Y by 3.0)
  let mat = scale2dXY(2.0, 3.0);
  let v = vec2f(4.0, 5.0);
  let result = mat * v;
  // Matrix should scale: (4*2, 5*3) = (8, 15)
  expectNearVec2(result, vec2f(8.0, 15.0));
}

@test fn testScale3() {
  let result = scale3(vec3f(0.75, 0.25, 0.5), vec3f(2.0, 0.5, 1.0));
  // Scale around (0.5, 0.5, 0.5)
  expectNearVec4(vec4f(result, 0.0), vec4f(1.0, 0.375, 0.5, 0.0));
}

@test fn testScale2CustomCenter() {
  // Scale around custom center point (0.3, 0.7) using _c variant
  let center = vec2f(0.3, 0.7);
  let result = scale2_c(vec2f(0.8, 0.9), vec2f(2.0, 2.0), center);
  // (0.8 - 0.3) * 2.0 + 0.3 = 0.5 * 2.0 + 0.3 = 1.3
  // (0.9 - 0.7) * 2.0 + 0.7 = 0.2 * 2.0 + 0.7 = 1.1
  expectNearVec2(result, vec2f(1.3, 1.1));
}

@test fn testScale3CustomCenter() {
  // Scale around custom center point (0.2, 0.3, 0.4) using _c variant
  let center = vec3f(0.2, 0.3, 0.4);
  let result = scale3_c(vec3f(0.7, 0.8, 0.9), vec3f(2.0, 3.0, 0.5), center);
  // (0.7 - 0.2) * 2.0 + 0.2 = 1.2
  // (0.8 - 0.3) * 3.0 + 0.3 = 1.8
  // (0.9 - 0.4) * 0.5 + 0.4 = 0.65
  expectNearVec4(vec4f(result, 0.0), vec4f(1.2, 1.8, 0.65, 0.0));
}
