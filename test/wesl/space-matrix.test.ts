import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("tbn", async () => {
  const src = `
    import lygia::space::tbn::tbn;
    @compute @workgroup_size(1)
    fn foo() {
      let t = vec3f(1.0, 0.0, 0.0);
      let b = vec3f(0.0, 1.0, 0.0);
      let n = vec3f(0.0, 0.0, 1.0);
      let mat = tbn(t, b, n);
      // Test that the matrix was created correctly by multiplying with a vector
      let v = mat * vec3f(1.0, 1.0, 1.0);
      test::results[0] = vec4f(v, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  expectCloseTo([1.0, 1.0, 1.0, 0.0], result);
});

test("perspective", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::space::perspective::perspective;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = perspective(HALF_PI, 16.0/9.0, 0.1, 100.0);
      // Just verify it creates a matrix (check one element)
      test::results[0] = mat[0][0];
    }
  `;
  const result = await lygiaTestCompute(src);
  // f / aspect where f = 1/tan(fov/2) = 1/tan(pi/4) = 1
  // so result = 1 / (16/9) = 9/16 = 0.5625
  expectCloseTo([0.5625], result);
});

test("orthographic", async () => {
  const src = `
    import lygia::space::orthographic::orthographic;
    @compute @workgroup_size(1)
    fn foo() {
      let mat = orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
      // Check the first element: 2/(r-l) = 2/(1-(-1)) = 2/2 = 1
      test::results[0] = mat[0][0];
    }
  `;
  const result = await lygiaTestCompute(src);
  expectCloseTo([1.0], result);
});

test("translate", async () => {
  const src = `
    import lygia::space::translate::translate;
    @compute @workgroup_size(1)
    fn foo() {
      // Create identity mat3 and add translation
      let m = mat3x3f(
        vec3f(1.0, 0.0, 0.0),
        vec3f(0.0, 1.0, 0.0),
        vec3f(0.0, 0.0, 1.0)
      );
      let result = translate(m, vec3f(10.0, 20.0, 30.0));
      // Extract translation component (4th column)
      test::results[0] = result[3];
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // Translation should be in the last column
  expectCloseTo([10.0, 20.0, 30.0, 1.0], result);
});
