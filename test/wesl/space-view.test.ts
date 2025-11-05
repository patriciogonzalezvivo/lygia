import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("eulerView", async () => {
  const src = `
    import lygia::math::consts::HALF_PI;
    import lygia::space::eulerView::eulerView;
    @compute @workgroup_size(1)
    fn foo() {
      // Test with camera at origin with 90° Y rotation
      let viewMatrix = eulerView(vec3f(0.0, 0.0, 0.0), vec3f(0.0, HALF_PI, 0.0));
      // Transform a point at (1, 0, 0) - should rotate around Y by 90°
      let testPoint = viewMatrix * vec4f(1.0, 0.0, 0.0, 1.0);
      test::results[0] = testPoint;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // eulerView creates view matrix from Euler angles (Y rotation by 90°)
  // Point at (1,0,0) rotated by 90° around Y should go to (0,0,-1)
  expectCloseTo([0.0, 0.0, -1.0, 1.0], result);
});

test("lookAt", async () => {
  const src = `
     import lygia::space::lookAt::lookAt;
     @compute @workgroup_size(1)
     fn foo() {
       // Test with camera looking down -Z axis with Y up
       let viewMatrix = lookAt(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 1.0, 0.0));
       // Transform a vector along the forward direction
       let testVec = viewMatrix * vec3f(0.0, 0.0, 1.0);
       test::results[0] = vec4f(testVec, 0.0);
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // lookAt creates orientation matrix from forward and up vectors
  // Looking down -Z (forward = (0,0,-1)), up = (0,1,0)
  // z-axis should be (0,0,-1), transforming (0,0,1) gives (0,0,-1)
  expectCloseTo([0.0, 0.0, -1.0, 0.0], result);
});

test("lookAtView", async () => {
  const src = `
     import lygia::space::lookAtView::lookAtView;
     @compute @workgroup_size(1)
     fn foo() {
       // Test with camera at (5,0,0) looking at origin with Y up
       let viewMatrix = lookAtView(vec3f(5.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));
       // Transform a point - the position should be embedded in the matrix
       let transformed = viewMatrix * vec4f(0.0, 0.0, 0.0, 1.0);
       test::results[0] = transformed;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // lookAtView creates 4x4 view matrix with position
  // Camera at (5,0,0) looking at origin should embed position in last column
  expectCloseTo([5.0, 0.0, 0.0, 1.0], result);
});

test("lookAtViewRoll", async () => {
  const src = `
     import lygia::math::consts::HALF_PI;
     import lygia::space::lookAtView::lookAtViewRoll;
     @compute @workgroup_size(1)
     fn foo() {
       // Test with camera at (0,5,0) looking at origin with 90° roll
       let viewMatrix = lookAtViewRoll(vec3f(0.0, 5.0, 0.0), vec3f(0.0, 0.0, 0.0), HALF_PI);
       // Extract position from the matrix (should be in last column)
       let position = viewMatrix * vec4f(0.0, 0.0, 0.0, 1.0);
       test::results[0] = position;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // lookAtViewRoll creates view matrix with roll parameter
  // Camera position should be (0,5,0) in the matrix
  expectCloseTo([0.0, 5.0, 0.0, 1.0], result);
});

test("lookAtViewFromDirection", async () => {
  const src = `
     import lygia::space::lookAtView::lookAtViewFromDirection;
     @compute @workgroup_size(1)
     fn foo() {
       // Test with camera at (3,0,0) looking in +X direction
       let viewMatrix = lookAtViewFromDirection(vec3f(3.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0));
       // Extract camera position from the matrix
       let position = viewMatrix * vec4f(0.0, 0.0, 0.0, 1.0);
       test::results[0] = position;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // lookAtViewFromDirection creates view matrix from position and direction
  // Camera position (3,0,0) should be embedded in the matrix
  expectCloseTo([3.0, 0.0, 0.0, 1.0], result);
});
