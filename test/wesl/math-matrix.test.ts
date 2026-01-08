import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

// Matrix conversion and operations

test("toMat3", async () => {
  const src = `
    import lygia::math::toMat3::toMat3;
    @compute @workgroup_size(1)
    fn foo() {
      let m4 = mat4x4f(
        vec4f(1.0, 2.0, 3.0, 4.0),
        vec4f(5.0, 6.0, 7.0, 8.0),
        vec4f(9.0, 10.0, 11.0, 12.0),
        vec4f(13.0, 14.0, 15.0, 16.0)
      );
      let m3 = toMat3(m4);
      test::results[0] = vec4f(m3[0][0], m3[1][1], m3[2][2], 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 6.0, 11.0, 0.0], result);
});

test("toMat4", async () => {
  const src = `
    import lygia::math::toMat4::toMat4;
    @compute @workgroup_size(1)
    fn foo() {
      let m3 = mat3x3f(
        vec3f(1.0, 2.0, 3.0),
        vec3f(4.0, 5.0, 6.0),
        vec3f(7.0, 8.0, 9.0)
      );
      let m4 = toMat4(m3);
      test::results[0] = vec4f(m4[0][0], m4[1][1], m4[2][2], m4[3][3]);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 5.0, 9.0, 1.0], result);
});

test("inverse - mat3", async () => {
  const src = `
    import lygia::math::inverse::inverse;
    @compute @workgroup_size(1)
    fn foo() {
      let m = mat3x3f(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 2.0, 0.0),
        vec3f(0.0, 0.0, 3.0)
      );
      let mInv = inverse(m);
      test::results[0] = vec4f(mInv[0][0], mInv[1][1], mInv[2][2], 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Inverse of diagonal matrix is 1/diagonal
  expectCloseTo([1.0, 0.5, 0.333], result.slice(0, 3), 0.01);
});

test("scale2d - uniform scale", async () => {
  const src = `
    import lygia::math::scale2d::scale2d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = scale2d(2.0);
      let v = vec2f(3.0, 4.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([6.0, 8.0, 0.0, 0.0], result);
});

test("scale2dVec - non-uniform scale", async () => {
  const src = `
    import lygia::math::scale2d::scale2dVec;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = scale2dVec(vec2f(2.0, 3.0));
      let v = vec2f(4.0, 5.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([8.0, 15.0, 0.0, 0.0], result);
});

test("scale3d", async () => {
  const src = `
    import lygia::math::scale3d::scale3d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = scale3d(vec3f(2.0, 3.0, 4.0));
      let v = vec3f(1.0, 2.0, 3.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([2.0, 6.0, 12.0, 0.0], result);
});

test("scale4d", async () => {
  const src = `
    import lygia::math::scale4d::scale4d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = scale4d(vec3f(2.0, 3.0, 4.0));
      let v = vec4f(1.0, 2.0, 3.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([2.0, 6.0, 12.0, 1.0], result);
});

test("translate4d", async () => {
  const src = `
    import lygia::math::translate4d::translate4d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = translate4d(vec3f(10.0, 20.0, 30.0));
      let v = vec4f(1.0, 2.0, 3.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([11.0, 22.0, 33.0, 1.0], result);
});

test("translate4dXYZ", async () => {
  const src = `
    import lygia::math::translate4d::translate4dXYZ;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = translate4dXYZ(5.0, 10.0, 15.0);
      let v = vec4f(1.0, 2.0, 3.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([6.0, 12.0, 18.0, 1.0], result);
});
