import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute, lygiaTestWesl } from "./testUtil.ts";

await lygiaTestWesl("test/wesl/shaders/math_minmax_test");

test("mmax2", async () => {
  const src = `
    import lygia::math::mmax::mmax2;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple cases: positive, negative, mixed signs, equal values
      test::results[0] = vec4f(
        mmax2(vec2f(3.0, 7.0)),    // Max in second position
        mmax2(vec2f(9.0, 2.0)),    // Max in first position
        mmax2(vec2f(-5.0, -2.0)),  // Negative values
        mmax2(vec2f(-3.0, 4.0))    // Mixed signs
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([7.0, 9.0, -2.0, 4.0], result);
});

test("mmax3", async () => {
  const src = `
    import lygia::math::mmax::mmax3;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple cases with max in different positions
      test::results[0] = vec4f(
        mmax3(vec3f(3.0, 7.0, 5.0)),   // Max in middle
        mmax3(vec3f(9.0, 2.0, 4.0)),   // Max at start
        mmax3(vec3f(1.0, 3.0, 8.0)),   // Max at end
        mmax3(vec3f(-6.0, -2.0, -4.0)) // Negative values
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([7.0, 9.0, 8.0, -2.0], result);
});

test("mmin2", async () => {
  const src = `
    import lygia::math::mmin::mmin2;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple cases: min in different positions, negative, mixed
      test::results[0] = vec4f(
        mmin2(vec2f(3.0, 7.0)),    // Min in first position
        mmin2(vec2f(9.0, 2.0)),    // Min in second position
        mmin2(vec2f(-5.0, -2.0)),  // Negative values
        mmin2(vec2f(-3.0, 4.0))    // Mixed signs
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([3.0, 2.0, -5.0, -3.0], result);
});

test("mmin3", async () => {
  const src = `
    import lygia::math::mmin::mmin3;
    @compute @workgroup_size(1)
    fn foo() {
      // Test multiple cases with min in different positions
      test::results[0] = vec4f(
        mmin3(vec3f(3.0, 7.0, 5.0)),   // Min at start
        mmin3(vec3f(9.0, 2.0, 4.0)),   // Min in middle
        mmin3(vec3f(6.0, 8.0, 1.0)),   // Min at end
        mmin3(vec3f(-2.0, 5.0, -8.0))  // Mixed signs
      );
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([3.0, 2.0, 1.0, -8.0], result);
});
