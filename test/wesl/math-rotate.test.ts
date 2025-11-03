import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("rotate2d - 90 degree rotation", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate2d::rotate2d;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate2d(HALF_PI); // /2 radians
      let v = vec2f(1.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, 0.0, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // 90 rotation of (1,0) should give approximately (0,1)
  expectCloseTo([0.0, 1.0, 0.0, 0.0], result);
});

test("rotate3d - rotation around axis", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate3d::rotate3d;
    @compute @workgroup_size(1)
    fn foo() {
      let axis = normalize(vec3f(0.0, 0.0, 1.0)); // Z-axis
      let mat = rotate3d(axis, HALF_PI); // /2 radians
      let v = vec3f(1.0, 0.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // 90 rotation around Z-axis of (1,0,0) - result is (0, -1, 0) due to matrix convention
  expectCloseTo([0.0, -1.0, 0.0, 0.0], result);
});

test("rotate3dX", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate3dX::rotate3dX;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate3dX(HALF_PI); // /2 radians
      let v = vec3f(0.0, 1.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // 90 rotation around X-axis of (0,1,0) should give (0,0,1)
  expectCloseTo([0.0, 0.0, 1.0], result.slice(0, 3));
});

test("rotate3dY", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate3dY::rotate3dY;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate3dY(HALF_PI); // /2 radians
      let v = vec3f(1.0, 0.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // 90 rotation around Y-axis of (1,0,0) should give (0,0,-1)
  expectCloseTo([0.0, 0.0, -1.0], result.slice(0, 3));
});

test("rotate3dZ", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate3dZ::rotate3dZ;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate3dZ(HALF_PI); // /2 radians
      let v = vec3f(1.0, 0.0, 0.0);
      let result = mat * v;
      test::results[0] = vec4f(result.x, result.y, result.z, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // 90 rotation around Z-axis of (1,0,0) - result depends on matrix convention
  expectCloseTo([0.0, -1.0, 0.0], result.slice(0, 3));
});

test("rotate4d - axis-angle rotation", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate4d::rotate4d;
    @compute @workgroup_size(1)
    fn foo() {
      let axis = normalize(vec3f(0.0, 0.0, 1.0));
      let mat = rotate4d(axis, HALF_PI); // /2 radians around Z
      let v = vec4f(1.0, 0.0, 0.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // 90 rotation around Z-axis of (1,0,0,1) - result depends on matrix convention
  expectCloseTo([0.0, -1.0, 0.0, 1.0], result);
});

test("rotate4dX", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate4dX::rotate4dX;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate4dX(HALF_PI); // /2 radians
      let v = vec4f(0.0, 1.0, 0.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // 90 rotation around X-axis of (0,1,0,1) - result depends on matrix convention
  expectCloseTo([0.0, 0.0, -1.0, 1.0], result);
});

test("rotate4dY", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate4dY::rotate4dY;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate4dY(HALF_PI); // /2 radians
      let v = vec4f(1.0, 0.0, 0.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // 90 rotation around Y-axis of (1,0,0,1) should give (0,0,-1,1)
  expectCloseTo([0.0, 0.0, -1.0, 1.0], result);
});

test("rotate4dZ", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::math::rotate4dZ::rotate4dZ;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = rotate4dZ(HALF_PI); // /2 radians
      let v = vec4f(1.0, 0.0, 0.0, 1.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // 90 rotation around Z-axis of (1,0,0,1) - result depends on matrix convention
  expectCloseTo([0.0, -1.0, 0.0, 1.0], result);
});
