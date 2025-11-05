import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("scale2", async () => {
  const src = `
    import lygia::space::scale::scale2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = scale2(vec2f(0.75, 0.25), vec2f(2.0, 0.5));
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec2f" });
  // Scale (0.75, 0.25) by (2.0, 0.5) around center (0.5, 0.5)
  // (0.75 - 0.5) * 2.0 + 0.5 = 0.25 * 2.0 + 0.5 = 1.0
  // (0.25 - 0.5) * 0.5 + 0.5 = -0.25 * 0.5 + 0.5 = 0.375
  expectCloseTo([1.0, 0.375], result);
});

test("scale2_f", async () => {
  const src = `
    import lygia::space::scale::scale2_f;
    @compute @workgroup_size(1)
    fn foo() {
      let result = scale2_f(vec2f(0.75, 0.25), 2.0);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec2f" });
  // Scale (0.75, 0.25) by 2.0 around center (0.5, 0.5)
  // (0.75 - 0.5) * 2.0 + 0.5 = 1.0
  // (0.25 - 0.5) * 2.0 + 0.5 = 0.0
  expectCloseTo([1.0, 0.0], result);
});

test("scale2dXY - matrix construction", async () => {
  const src = `
    import lygia::math::scale2d::scale2dXY;
    @compute @workgroup_size(1)
    fn foo() {
      // Test non-uniform scale matrix (scale X by 2.0, Y by 3.0)
      let mat = scale2dXY(2.0, 3.0);
      let v = vec2f(4.0, 5.0);
      let result = mat * v;
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec2f" });
  // Matrix should scale: (4*2, 5*3) = (8, 15)
  expectCloseTo([8.0, 15.0], result);
});

test("scale3", async () => {
  const src = `
    import lygia::space::scale::scale3;
    @compute @workgroup_size(1)
    fn foo() {
      let result = scale3(vec3f(0.75, 0.25, 0.5), vec3f(2.0, 0.5, 1.0));
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Scale around (0.5, 0.5, 0.5)
  expectCloseTo([1.0, 0.375, 0.5, 0.0], result);
});

test("scale2 - with custom CENTER_2D via constants", async () => {
  const src = `
    import lygia::space::scale::scale2;
    @compute @workgroup_size(1)
    fn foo() {
      // Scale around custom center point (0.3, 0.7)
      let result = scale2(vec2f(0.8, 0.9), vec2f(2.0, 2.0));
      test::results[0] = result;
    }
  `;
  // Test with custom CENTER_2D set via constants
  const result = await lygiaTestCompute(src, {
    elem: "vec2f",
    conditions: { CENTER_2D: true },
    constants: { CENTER_2D: "vec2f(0.3, 0.7)" },
  });
  // Scale (0.8, 0.9) by (2.0, 2.0) around center (0.3, 0.7)
  // (0.8 - 0.3) * 2.0 + 0.3 = 0.5 * 2.0 + 0.3 = 1.3
  // (0.9 - 0.7) * 2.0 + 0.7 = 0.2 * 2.0 + 0.7 = 1.1
  expectCloseTo([1.3, 1.1], result);
});

test("scale3 - with custom CENTER_3D via constants", async () => {
  const src = `
    import lygia::space::scale::scale3;
    @compute @workgroup_size(1)
    fn foo() {
      // Scale around custom center point (0.2, 0.3, 0.4)
      let result = scale3(vec3f(0.7, 0.8, 0.9), vec3f(2.0, 3.0, 0.5));
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  // Test with custom CENTER_3D set via constants
  const result = await lygiaTestCompute(src, {
    elem: "vec4f",
    conditions: { CENTER_3D: true },
    constants: { CENTER_3D: "vec3f(0.2, 0.3, 0.4)" },
  });
  // Scale (0.7, 0.8, 0.9) by (2.0, 3.0, 0.5) around center (0.2, 0.3, 0.4)
  // (0.7 - 0.2) * 2.0 + 0.2 = 0.5 * 2.0 + 0.2 = 1.2
  // (0.8 - 0.3) * 3.0 + 0.3 = 0.5 * 3.0 + 0.3 = 1.8
  // (0.9 - 0.4) * 0.5 + 0.4 = 0.5 * 0.5 + 0.4 = 0.65
  expectCloseTo([1.2, 1.8, 0.65, 0.0], result);
});
