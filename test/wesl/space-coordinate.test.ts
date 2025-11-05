import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

const QTR_PI = Math.PI / 4;

test("cart2polar2", async () => {
  const src = `
    import lygia::space::cart2polar::cart2polar2;
    @compute @workgroup_size(1)
    fn foo() {
      let result = cart2polar2(vec2f(1.0, 1.0));
      test::results[0] = result; // vec2f: angle, radius
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec2f" });
  expectCloseTo([QTR_PI, Math.SQRT2], result); // atan2(1,1) = π/4, length = sqrt(2)
});

test("polar2cart", async () => {
  const src = `
    import lygia::math::consts::{QTR_PI, SQRT2};
    import lygia::space::polar2cart::polar2cart;
    @compute @workgroup_size(1)
    fn foo() {
      let result = polar2cart(vec2f(QTR_PI, SQRT2));
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec2f" });
  expectCloseTo([1.0, 1.0], result);
});

test("equirect2xyz", async () => {
  const src = `
    import lygia::space::equirect2xyz::equirect2xyz;
    @compute @workgroup_size(1)
    fn foo() {
      let result = equirect2xyz(vec2f(0.5, 0.5)); // Center of equirect map
      test::results[0] = vec4f(result, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec4f" });
  // At center (0.5, 0.5): Theta=PI, Phi=PI/2 -> direction pointing left (-X axis)
  expectCloseTo([-1.0, 0.0, 0.0, 0.0], result);
});

test("xyz2equirect", async () => {
  const src = `
    import lygia::space::xyz2equirect::xyz2equirect;
    @compute @workgroup_size(1)
    fn foo() {
      let result = xyz2equirect(vec3f(1.0, 0.0, 0.0)); // +X direction
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec2f" });
  // atan(0, 1) = 0, + PI = PI, / (2*PI) = 0.5
  // acos(0) = PI/2, / PI = 0.5
  expectCloseTo([0.5, 0.5], result);
});

test("fisheye2xyz", async () => {
  const src = `
     import lygia::space::fisheye2xyz::fisheye2xyz;

     @compute @workgroup_size(1)
     fn foo() {
       let uv = vec2f(0.75, 0.5);
       let result = fisheye2xyz(uv);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Fisheye projection should return a normalized direction vector
  const length = Math.sqrt(result[0] ** 2 + result[1] ** 2 + result[2] ** 2);
  expectCloseTo([length], [1.0]);
});

test("fisheye2xyz - division by zero at center", async () => {
  const src = `
     import lygia::space::fisheye2xyz::fisheye2xyz;

     @compute @workgroup_size(1)
     fn foo() {
       let uv = vec2f(0.5, 0.5); // Center point: R=0, triggers division by zero at line 14
       let result = fisheye2xyz(uv);
       test::results[0] = result;
     }
   `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  expectCloseTo([0.0, 1.0, 0.0], result.slice(0, 3));
});

test("nearest", async () => {
  const src = `
    import lygia::space::nearest::nearest;
    @compute @workgroup_size(1)
    fn foo() {
      // Test nearest-neighbor snapping for different UV coordinates
      let result1 = nearest(vec2f(0.7533, 0.2567), vec2f(1920.0, 1080.0));
      let result2 = nearest(vec2f(0.7500, 0.2500), vec2f(1920.0, 1080.0));
      test::results[0] = result1;
      test::results[1] = result2;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec2f" });
  // nearest snaps to pixel centers: floor(v*res)/res + offset
  // For 1920x1080: offset = 0.5/(1919, 1079) ≈ (0.00026, 0.00046)
  // (0.7533*1920, 0.2567*1080) = (1446.336, 277.236)
  // floor -> (1446, 277), /res -> (0.75313, 0.25648)
  // + offset -> (0.75339, 0.25694)
  expectCloseTo([0.7534, 0.2569, 0.75026, 0.25046], result);
});
