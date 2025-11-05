import { test } from "vitest";
import { expectCloseTo, lygiaTestCompute } from "./testUtil.ts";

test("centroid", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::centroid::centroid;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(-2.0, -4.0, -6.0);
      box.max = vec3f(2.0, 4.0, 6.0);
      let result = centroid(box);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Centroid should be at (0, 0, 0)
  expectCloseTo([0.0, 0.0, 0.0], result);
});

test("contain", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::contain::contain;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(-1.0, -1.0, -1.0);
      box.max = vec3f(1.0, 1.0, 1.0);

      let inside = contain(box, vec3f(0.0, 0.0, 0.0));
      let outside = contain(box, vec3f(2.0, 0.0, 0.0));

      test::results[0] = vec3f(select(0.0, 1.0, inside), select(0.0, 1.0, outside), 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Inside should be true (1.0), outside should be false (0.0)
  expectCloseTo([1.0, 0.0, 0.0], result);
});

test("diagonal", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::diagonal::diagonal;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(-1.0, -2.0, -3.0);
      box.max = vec3f(1.0, 2.0, 3.0);
      let result = diagonal(box);
      test::results[0] = result;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Diagonal should be (2, 4, 6)
  expectCloseTo([2.0, 4.0, 6.0], result);
});

test("expand with scalar", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::expand::expand;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(-1.0, -1.0, -1.0);
      box.max = vec3f(1.0, 1.0, 1.0);
      expand(&box, 0.5);
      test::results[0] = vec3f(box.min.x, box.max.x, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Min expands to -1.5, max expands to 1.5
  expectCloseTo([-1.5, 1.5, 0.0], result);
});

test("expand2 with point", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::expand::expand2;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(-1.0, -1.0, -1.0);
      box.max = vec3f(1.0, 1.0, 1.0);
      expand2(&box, vec3f(2.0, -2.0, 0.5));
      test::results[0] = vec3f(box.min.y, box.max.x, 0.0);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Min.y expands to -2.0, max.x expands to 2.0
  expectCloseTo([-2.0, 2.0, 0.0], result);
});

test("expand3 with AABB", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::expand::expand3;

    @compute @workgroup_size(1)
    fn foo() {
      var box1: AABB;
      box1.min = vec3f(-1.0, -1.0, -1.0);
      box1.max = vec3f(1.0, 1.0, 1.0);

      var box2: AABB;
      box2.min = vec3f(0.0, -2.0, 0.0);
      box2.max = vec3f(2.0, 0.0, 2.0);

      expand3(&box1, box2);
      test::results[0] = vec3f(box1.min.y, box1.max.x, box1.max.z);
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // Min.y expands to -2.0, max.x expands to 2.0, max.z expands to 2.0
  expectCloseTo([-2.0, 2.0, 2.0], result);
});

test("square", async () => {
  const src = `
    import lygia::geometry::aabb::aabb::AABB;
    import lygia::geometry::aabb::square::square;

    @compute @workgroup_size(1)
    fn foo() {
      var box: AABB;
      box.min = vec3f(-1.0, -2.0, -0.5);
      box.max = vec3f(1.0, 2.0, 0.5);
      square(&box);
      let diag = box.max - box.min;
      test::results[0] = diag;
    }
  `;
  const result = await lygiaTestCompute(src, { elem: "vec3f" });
  // All dimensions should be equal to the largest dimension (4.0)
  expectCloseTo([4.0, 4.0, 4.0], result);
});
