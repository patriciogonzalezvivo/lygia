import { expect, test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";


test("Triangle struct", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(3.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 4.0, 0.0);
      // Compute edge lengths to test struct functionality
      let ab = length(tri.b - tri.a);
      let bc = length(tri.c - tri.b);
      let ca = length(tri.a - tri.c);
      test::results[0] = vec3f(ab, bc, ca);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // 3-4-5 right triangle: edges are 3, 5, 4
  expectCloseTo([3.0, 5.0, 4.0], result);
});

test("area", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::area::area;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(3.0, 0.0, 1.0);
      tri.c = vec3f(0.0, 4.0, 1.0);
      let result = area(tri);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src);
  // Non-axis-aligned triangle: (B-A)×(C-A) = (-4, -3, 12), ||(−4,−3,12)|| = 13
  // Area = 0.5 * 13 = 6.5
  expectCloseTo([6.5], result);
});

test("barycentric - computes normalized coordinates", async () => {
  const src = `
    import lygia::geometry::triangle::barycentric::barycentric;

    @compute @workgroup_size(1)
    fn foo() {
      // Test with general non-axis-aligned triangle
      let a = vec3f(2.0, 1.0, -0.5);
      let b = vec3f(-1.0, 3.0, 0.5);
      let c = vec3f(1.5, -0.5, 2.0);

      let coords = barycentric(a, b, c);
      test::results[0] = coords;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });

  // Barycentric coordinates must sum to 1.0
  expect(result[0] + result[1] + result[2]).toBeCloseTo(1.0, 2);

  // Verify fundamental property: barycentric coords reconstruct a point in/on the triangle
  const a = [2.0, 1.0, -0.5];
  const b = [-1.0, 3.0, 0.5];
  const c = [1.5, -0.5, 2.0];
  const reconstructed = reconstructFromBarycentric(result, a, b, c);
  // Reconstructed point should be within triangle bounds
  expect(reconstructed[0]).toBeGreaterThan(-2.0);
  expect(reconstructed[0]).toBeLessThan(3.0);

  // Exact values to catch regressions (most specific test last)
  expectCloseTo([0.333, 0.333, 0.333], result, 2);
});

test("barycentric2 - Triangle struct wrapper", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric2;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(2.0, 1.0, -0.5);
      tri.b = vec3f(-1.0, 3.0, 0.5);
      tri.c = vec3f(1.5, -0.5, 2.0);
      let coords = barycentric2(tri);
      test::results[0] = coords;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });

  // Verify sum-to-1 property
  expect(result[0] + result[1] + result[2]).toBeCloseTo(1.0, 2);

  // Verify fundamental property: barycentric coords reconstruct a point in/on the triangle
  const a = [2.0, 1.0, -0.5];
  const b = [-1.0, 3.0, 0.5];
  const c = [1.5, -0.5, 2.0];
  const reconstructed = reconstructFromBarycentric(result, a, b, c);
  expect(reconstructed[0]).toBeGreaterThan(-2.0);
  expect(reconstructed[0]).toBeLessThan(3.0);

  // Should produce same result as barycentric(a, b, c)
  expectCloseTo([0.333, 0.333, 0.333], result, 2);
});

test("barycentric3 - point at vertex", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric3;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(1.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 1.0, 0.0);

      // Test point at vertex a - coordinate for a should be highest
      let result = barycentric3(tri, tri.a);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });

  // Point at vertex a should have dominant weight at a
  // Note: This function returns unnormalized coords (sum ≠ 1)
  expect(result[0]).toBeGreaterThan(result[1]);
  expect(result[0]).toBeGreaterThan(result[2]);
  expect(result[1]).toBeLessThan(0.01);
  expect(result[2]).toBeLessThan(0.01);
});

test("barycentric3 - edge midpoint", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::barycentric::barycentric3;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(2.0, 0.0, 0.0);
      tri.c = vec3f(0.0, 2.0, 0.0);

      // Point at midpoint of edge a-b
      let midpoint = vec3f(1.0, 0.0, 0.0);
      let result = barycentric3(tri, midpoint);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });

  // Midpoint of a-b should have equal weights for a and b, zero for c
  expect(Math.abs(result[0] - result[1])).toBeLessThan(0.01);
  expect(result[2]).toBeLessThan(0.01);
});

test("centroid", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::centroid::centroid;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(1.0, 2.0, -1.0);
      tri.b = vec3f(4.0, -1.0, 2.0);
      tri.c = vec3f(-2.0, 3.0, 1.0);
      let result = centroid(tri);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Centroid is average of vertices: ((1+4-2)/3, (2-1+3)/3, (-1+2+1)/3)
  expectCloseTo([1.0, 1.333, 0.667], result, 2);
});

test("normal", async () => {
  const src = `
    import lygia::geometry::triangle::triangle::Triangle;
    import lygia::geometry::triangle::normal::normal;

    @compute @workgroup_size(1)
    fn foo() {
      var tri: Triangle;
      tri.a = vec3f(0.0, 0.0, 0.0);
      tri.b = vec3f(1.0, 0.0, 1.0);
      tri.c = vec3f(0.0, 1.0, 1.0);
      let result = normal(tri);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });

  // Verify normal is unit length
  const length = Math.sqrt(result[0] ** 2 + result[1] ** 2 + result[2] ** 2);
  expect(length).toBeCloseTo(1.0, 2);

  // Tilted triangle: (B-A)×(C-A) = (1,0,1)×(0,1,1) = (-1,-1,1)
  // Normalized: (-1/√3, -1/√3, 1/√3)
  expectCloseTo([-0.577, -0.577, 0.577], result, 2);
});

/**
 * Reconstructs a 3D point from barycentric coordinates.
 * Verifies the fundamental property: u*a + v*b + w*c where (u,v,w) are barycentric coords.
 */
function reconstructFromBarycentric(
  baryCoords: number[],
  a: number[],
  b: number[],
  c: number[],
): number[] {
  return [
    baryCoords[0] * a[0] + baryCoords[1] * b[0] + baryCoords[2] * c[0],
    baryCoords[0] * a[1] + baryCoords[1] * b[1] + baryCoords[2] * c[1],
    baryCoords[0] * a[2] + baryCoords[1] * b[2] + baryCoords[2] * c[2],
  ];
}