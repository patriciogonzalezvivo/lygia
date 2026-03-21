import wgsl_test::{expectNearVec4, expectWithinVec4};
import lygia::math::toMat3::toMat3;
import lygia::math::toMat4::toMat4;
import lygia::math::inverse::inverse;
import lygia::math::scale2d::{scale2d, scale2dVec};
import lygia::math::scale3d::scale3d;
import lygia::math::scale4d::scale4d;
import lygia::math::translate4d::{translate4d, translate4dXYZ};

// toMat3
@test
fn testToMat3() {
  let m4 = mat4x4f(
    vec4f(1.0, 2.0, 3.0, 4.0),
    vec4f(5.0, 6.0, 7.0, 8.0),
    vec4f(9.0, 10.0, 11.0, 12.0),
    vec4f(13.0, 14.0, 15.0, 16.0)
  );
  let m3 = toMat3(m4);
  expectNearVec4(vec4f(m3[0][0], m3[1][1], m3[2][2], 0.0), vec4f(1.0, 6.0, 11.0, 0.0));
}

// toMat4
@test
fn testToMat4() {
  let m3 = mat3x3f(
    vec3f(1.0, 2.0, 3.0),
    vec3f(4.0, 5.0, 6.0),
    vec3f(7.0, 8.0, 9.0)
  );
  let m4 = toMat4(m3);
  expectNearVec4(vec4f(m4[0][0], m4[1][1], m4[2][2], m4[3][3]), vec4f(1.0, 5.0, 9.0, 1.0));
}

// inverse - mat3
@test
fn testInverse() {
  let m = mat3x3f(
    vec3f(1.0, 0.0, 0.0),
    vec3f(0.0, 2.0, 0.0),
    vec3f(0.0, 0.0, 3.0)
  );
  let mInv = inverse(m);
  // Inverse of diagonal matrix is 1/diagonal
  expectWithinVec4(vec4f(mInv[0][0], mInv[1][1], mInv[2][2], 0.0), vec4f(1.0, 0.5, 0.333, 0.0), 0.01, 1e-6);
}

// scale2d - uniform scale
@test
fn testScale2d() {
  let mat = scale2d(2.0);
  let v = vec2f(3.0, 4.0);
  let result = mat * v;
  expectNearVec4(vec4f(result.x, result.y, 0.0, 0.0), vec4f(6.0, 8.0, 0.0, 0.0));
}

// scale2dVec - non-uniform scale
@test
fn testScale2dVec() {
  let mat = scale2dVec(vec2f(2.0, 3.0));
  let v = vec2f(4.0, 5.0);
  let result = mat * v;
  expectNearVec4(vec4f(result.x, result.y, 0.0, 0.0), vec4f(8.0, 15.0, 0.0, 0.0));
}

// scale3d
@test
fn testScale3d() {
  let mat = scale3d(vec3f(2.0, 3.0, 4.0));
  let v = vec3f(1.0, 2.0, 3.0);
  let result = mat * v;
  expectNearVec4(vec4f(result.x, result.y, result.z, 0.0), vec4f(2.0, 6.0, 12.0, 0.0));
}

// scale4d
@test
fn testScale4d() {
  let mat = scale4d(vec3f(2.0, 3.0, 4.0));
  let v = vec4f(1.0, 2.0, 3.0, 1.0);
  let result = mat * v;
  expectNearVec4(result, vec4f(2.0, 6.0, 12.0, 1.0));
}

// translate4d
@test
fn testTranslate4d() {
  let mat = translate4d(vec3f(10.0, 20.0, 30.0));
  let v = vec4f(1.0, 2.0, 3.0, 1.0);
  let result = mat * v;
  expectNearVec4(result, vec4f(11.0, 22.0, 33.0, 1.0));
}

// translate4dXYZ
@test
fn testTranslate4dXYZ() {
  let mat = translate4dXYZ(5.0, 10.0, 15.0);
  let v = vec4f(1.0, 2.0, 3.0, 1.0);
  let result = mat * v;
  expectNearVec4(result, vec4f(6.0, 12.0, 18.0, 1.0));
}
