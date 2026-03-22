import wgsl_test::{expect, expectNear, expectNearVec4, expectWithinVec3};
import lygia::math::consts::{HALF_PI, INV_SQRT2};
import lygia::math::quat::add::quatAdd;
import lygia::math::quat::sub::quatSub;
import lygia::math::quat::mul::{quatMul, quatMulVec3};
import lygia::math::quat::conj::quatConj;
import lygia::math::quat::norm::quatNorm;
import lygia::math::quat::length::quatLength;
import lygia::math::quat::lengthSq::quatLengthSq;
import lygia::math::quat::identity::QUAT_IDENTITY;
import lygia::math::quat::lerp::quatLerp;
import lygia::math::quat::quat2mat3::quat2mat3;
import lygia::math::quat::quat2mat4::quat2mat4;
import lygia::math::quat::{quat, quatForward, quatForwardUp};
import lygia::math::quat::div::quatDiv;
import lygia::math::quat::neg::quatNeg;
import lygia::math::quat::inverse::quatInverse;

@test fn testQuatAdd() {
  let result = quatAdd(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(0.5, 0.5, 0.5, 0.5));
  expectNearVec4(result, vec4f(1.5, 2.5, 3.5, 4.5));
}

@test fn testQuatSub() {
  let result = quatSub(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(0.5, 0.5, 0.5, 0.5));
  expectNearVec4(result, vec4f(0.5, 1.5, 2.5, 3.5));
}

@test fn testQuatMul() {
  let q1 = normalize(vec4f(1.0, 0.0, 0.0, 1.0));
  let q2 = normalize(vec4f(0.0, 1.0, 0.0, 1.0));
  let result = quatMul(q1, q2);
  expectNearVec4(result, vec4f(0.5, 0.5, 0.5, 0.5));
}

@test fn testQuatConj() {
  let result = quatConj(vec4f(1.0, 2.0, 3.0, 4.0));
  expectNearVec4(result, vec4f(-1.0, -2.0, -3.0, 4.0));
}

@test fn testQuatNorm() {
  let result = quatNorm(vec4f(1.0, 2.0, 3.0, 4.0));
  expectNearVec4(result, vec4f(0.1826, 0.3651, 0.5477, 0.7303));
}

@test fn testQuatLength() {
  let result = quatLength(vec4f(1.0, 2.0, 3.0, 4.0));
  expectNear(result, 5.47723);
}

@test fn testQuatLengthSq() {
  let result = quatLengthSq(vec4f(1.0, 2.0, 3.0, 4.0));
  expectNear(result, 30.0);
}

@test fn testQuatIdentity() {
  expectNearVec4(QUAT_IDENTITY, vec4f(0.0, 0.0, 0.0, 1.0));
}

@test fn testQuatLerp() {
  let q1 = normalize(vec4f(1.0, 0.0, 0.0, 1.0));
  let q2 = normalize(vec4f(0.0, 1.0, 0.0, 1.0));
  let result = quatLerp(q1, q2, 0.5);
  expectNearVec4(result, vec4f(0.4082, 0.4082, 0.0, 0.8165));
}

@test fn testQuat2mat3() {
  let q = normalize(vec4f(0.0, 0.0, INV_SQRT2, INV_SQRT2));
  let m = quat2mat3(q);
  expectNearVec4(vec4f(m[0][0], m[1][1], m[2][2], 0.0), vec4f(0.0, 0.0, 1.0, 0.0));
}

@test fn testQuat2mat4() {
  let q = normalize(vec4f(0.0, 0.0, INV_SQRT2, INV_SQRT2));
  let m = quat2mat4(q);
  expectNearVec4(vec4f(m[0][0], m[1][1], m[2][2], m[3][3]), vec4f(0.0, 0.0, 1.0, 1.0));
}

@test fn testQuatFromAxisAngle() {
  let axis = normalize(vec3f(0.0, 1.0, 0.0));
  let angle = HALF_PI;
  let result = quat(axis, angle);
  // quat from Y-axis rotation of pi/2: (0, sin(pi/4), 0, cos(pi/4))
  expectNearVec4(result, vec4f(0.0, INV_SQRT2, 0.0, INV_SQRT2));
}

@test fn testQuatDiv() {
  let q = vec4f(2.0, 4.0, 6.0, 8.0);
  let result = quatDiv(q, 2.0);
  expectNearVec4(result, vec4f(1.0, 2.0, 3.0, 4.0));
}

@test fn testQuatNeg() {
  let q = vec4f(1.0, 2.0, 3.0, 4.0);
  let result = quatNeg(q);
  expectNearVec4(result, vec4f(-1.0, -2.0, -3.0, -4.0));
}

@test fn testQuatInverse() {
  let q = normalize(vec4f(1.0, 2.0, 3.0, 4.0));
  let qInv = quatInverse(q);
  // q * qInv should give identity quaternion (0,0,0,1)
  let identity = quatMul(q, qInv);
  expectNearVec4(identity, vec4f(0.0, 0.0, 0.0, 1.0));
}

@test fn testQuatForward() {
  // Create quaternion that rotates default forward to +X
  let forward = normalize(vec3f(1.0, 0.0, 0.0));
  let q = quatForward(forward);

  // Rotate default forward vector (0,0,1) using this quaternion
  let defaultForward = vec3f(0.0, 0.0, 1.0);
  let rotated = quatMulVec3(q, defaultForward);

  // Rotated vector should point in +X direction
  expectWithinVec3(vec3f(rotated.x, rotated.y, rotated.z), vec3f(1.0, 0.0, 0.0), 0.1, 1e-4);
  // Quaternion should be normalized
  let len = sqrt(dot(q, q));
  expectNear(len, 1.0);
}

@test fn testQuatForwardUp() {
  // Create quaternion with forward=+X and up=+Y
  let forward = normalize(vec3f(1.0, 0.0, 0.0));
  let up = normalize(vec3f(0.0, 1.0, 0.0));
  let q = quatForwardUp(forward, up);

  // Default forward (0,0,1) should rotate to our forward (+X)
  let defaultForward = vec3f(0.0, 0.0, 1.0);
  let rotatedForward = quatMulVec3(q, defaultForward);

  // Default up (0,1,0) should remain up (+Y)
  let defaultUp = vec3f(0.0, 1.0, 0.0);
  let rotatedUp = quatMulVec3(q, defaultUp);

  // Verify quaternion is normalized
  let len = sqrt(dot(q, q));

  expectNear(rotatedForward.x, 1.0);
  expectNear(rotatedUp.y, 1.0);
  expectNear(len, 1.0);
}
