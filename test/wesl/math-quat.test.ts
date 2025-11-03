import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("quatAdd", async () => {
  const src = `
    import lygia::math::quat::add::quatAdd;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatAdd(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(0.5, 0.5, 0.5, 0.5)); }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.5, 2.5, 3.5, 4.5], result);
});

test("quatSub", async () => {
  const src = `
    import lygia::math::quat::sub::quatSub;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatSub(vec4f(1.0, 2.0, 3.0, 4.0), vec4f(0.5, 0.5, 0.5, 0.5)); }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5, 1.5, 2.5, 3.5], result);
});

test("quatMul", async () => {
  const src = `
    import lygia::math::quat::mul::quatMul;
    @compute @workgroup_size(1)
    fn foo() {
      let q1 = normalize(vec4f(1.0, 0.0, 0.0, 1.0));
      let q2 = normalize(vec4f(0.0, 1.0, 0.0, 1.0));
      test::results[0] = quatMul(q1, q2);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.5, 0.5, 0.5, 0.5], result);
});

test("quatConj", async () => {
  const src = `
    import lygia::math::quat::conj::quatConj;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatConj(vec4f(1.0, 2.0, 3.0, 4.0)); }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([-1.0, -2.0, -3.0, 4.0], result);
});

test("quatNorm", async () => {
  const src = `
    import lygia::math::quat::norm::quatNorm;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatNorm(vec4f(1.0, 2.0, 3.0, 4.0)); }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.1826, 0.3651, 0.5477, 0.7303], result);
});

test("quatLength", async () => {
  const src = `
    import lygia::math::quat::length::quatLength;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatLength(vec4f(1.0, 2.0, 3.0, 4.0)); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([5.47723], result);
});

test("quatLengthSq", async () => {
  const src = `
    import lygia::math::quat::lengthSq::quatLengthSq;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = quatLengthSq(vec4f(1.0, 2.0, 3.0, 4.0)); }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([30.0], result);
});

test("quatIdentity", async () => {
  const src = `
    import lygia::math::quat::identity::QUAT_IDENTITY;
    @compute @workgroup_size(1)
    fn foo() { test::results[0] = QUAT_IDENTITY; }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.0, 0.0, 1.0], result);
});

test("quatLerp", async () => {
  const src = `
    import lygia::math::quat::lerp::quatLerp;
    @compute @workgroup_size(1)
    fn foo() {
      let q1 = normalize(vec4f(1.0, 0.0, 0.0, 1.0));
      let q2 = normalize(vec4f(0.0, 1.0, 0.0, 1.0));
      test::results[0] = quatLerp(q1, q2, 0.5);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.4082, 0.4082, 0.0, 0.8165], result);
});

test("quat2mat3", async () => {
  const src = `
    import lygia::math::consts::INV_SQRT2;
    import lygia::math::quat::quat2mat3::quat2mat3;
    @compute @workgroup_size(1)
    fn foo() {
      let q = normalize(vec4f(0.0, 0.0, INV_SQRT2, INV_SQRT2));
      let m = quat2mat3(q);
      test::results[0] = vec4f(m[0][0], m[1][1], m[2][2], 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.0, 1.0, 0.0], result);
});

test("quat2mat4", async () => {
  const src = `
    import lygia::math::consts::INV_SQRT2;
    import lygia::math::quat::quat2mat4::quat2mat4;
    @compute @workgroup_size(1)
    fn foo() {
      let q = normalize(vec4f(0.0, 0.0, INV_SQRT2, INV_SQRT2));
      let m = quat2mat4(q);
      test::results[0] = vec4f(m[0][0], m[1][1], m[2][2], m[3][3]);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([0.0, 0.0, 1.0, 1.0], result);
});

test("quat - create from axis and angle", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::quat::quat;
    @compute @workgroup_size(1)
    fn foo() {
      let axis = normalize(vec3f(0.0, 1.0, 0.0));
      let angle = HALF_PI; // π/2 radians (90 degrees)
      let result = quat(axis, angle);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // quat from Y-axis rotation of π/2: (0, sin(π/4), 0, cos(π/4)) ≈ (0, Math.SQT1_2, 0, Math.SQT1_2)
  expectCloseTo([0.0, Math.SQRT1_2, 0.0, Math.SQRT1_2], result);
});

test("quatDiv - divide quaternion by scalar", async () => {
  const src = `
    import lygia::math::quat::div::quatDiv;
    @compute @workgroup_size(1)
    fn foo() {
      let q = vec4f(2.0, 4.0, 6.0, 8.0);
      let result = quatDiv(q, 2.0);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 2.0, 3.0, 4.0], result);
});

test("quatNeg - negate quaternion", async () => {
  const src = `
    import lygia::math::quat::neg::quatNeg;
    @compute @workgroup_size(1)
    fn foo() {
      let q = vec4f(1.0, 2.0, 3.0, 4.0);
      let result = quatNeg(q);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([-1.0, -2.0, -3.0, -4.0], result);
});

test("quatInverse", async () => {
  const src = `
    import lygia::math::quat::inverse::quatInverse;
    import lygia::math::quat::mul::quatMul;
    @compute @workgroup_size(1)
    fn foo() {
      // Create a simple quaternion
      let q = normalize(vec4f(1.0, 2.0, 3.0, 4.0));
      let qInv = quatInverse(q);
      // Multiplying q * qInv should give identity quaternion (0,0,0,1)
      let identity = quatMul(q, qInv);
      test::results[0] = identity;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Identity quaternion is (0, 0, 0, 1)
  expectCloseTo([0.0, 0.0, 0.0, 1.0], result);
});

test("quatForward - create quat from forward vector", async () => {
  const src = `
    import lygia::math::quat::quatForward;
    import lygia::math::quat::quatConj;
    import lygia::math::quat::mul::quatMulVec3;
    @compute @workgroup_size(1)
    fn foo() {
      // Create quaternion that rotates default forward to +X
      let forward = normalize(vec3f(1.0, 0.0, 0.0));
      let q = quatForward(forward);

      // Test by rotating default forward vector (0,0,1) using this quaternion
      // It should rotate to point in the forward direction we specified (+X)
      let defaultForward = vec3f(0.0, 0.0, 1.0);
      let rotated = quatMulVec3(q, defaultForward);

      // Also verify quaternion is normalized
      let length = sqrt(dot(q, q));

      test::results[0] = vec4f(rotated.x, rotated.y, rotated.z, length);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Rotated vector should point in +X direction (our specified forward)
  expectCloseTo([1.0, 0.0, 0.0], result.slice(0, 3), 0.1);
  // Quaternion should be normalized
  expectCloseTo([1.0], [result[3]]);
});

test("quatForwardUp - create quat from forward and up vectors", async () => {
  const src = `
    import lygia::math::quat::quatForwardUp;
    import lygia::math::quat::mul::quatMulVec3;
    @compute @workgroup_size(1)
    fn foo() {
      // Create quaternion with forward=+X and up=+Y
      let forward = normalize(vec3f(1.0, 0.0, 0.0));
      let up = normalize(vec3f(0.0, 1.0, 0.0));
      let q = quatForwardUp(forward, up);

      // Test by rotating vectors
      // Default forward (0,0,1) should rotate to our forward (+X)
      let defaultForward = vec3f(0.0, 0.0, 1.0);
      let rotatedForward = quatMulVec3(q, defaultForward);

      // Default up (0,1,0) should remain up (+Y) since we specified that
      let defaultUp = vec3f(0.0, 1.0, 0.0);
      let rotatedUp = quatMulVec3(q, defaultUp);

      // Verify quaternion is normalized
      let length = sqrt(dot(q, q));

      test::results[0] = vec4f(rotatedForward.x, rotatedUp.y, length, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Rotated forward should point in +X direction
  expectCloseTo([1.0], [result[0]]);
  // Rotated up should still point in +Y direction
  expectCloseTo([1.0], [result[1]]);
  // Quaternion should be normalized
  expectCloseTo([1.0], [result[2]]);
});
