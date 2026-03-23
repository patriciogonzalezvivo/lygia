import wgsl_test::expectNearVec3;
import lygia::geometry::aabb::aabb::AABB;
import lygia::geometry::aabb::centroid::centroid;
import lygia::geometry::aabb::contain::contain;
import lygia::geometry::aabb::diagonal::diagonal;
import lygia::geometry::aabb::expand::{expand, expand2, expand3};
import lygia::geometry::aabb::square::square;

@test fn centroidSymmetric() {
  var box: AABB;
  box.min = vec3f(-2.0, -4.0, -6.0);
  box.max = vec3f(2.0, 4.0, 6.0);
  expectNearVec3(centroid(box), vec3f(0.0, 0.0, 0.0));
}

@test fn containInsideOutside() {
  var box: AABB;
  box.min = vec3f(-1.0, -1.0, -1.0);
  box.max = vec3f(1.0, 1.0, 1.0);
  let inside = contain(box, vec3f(0.0, 0.0, 0.0));
  let outside = contain(box, vec3f(2.0, 0.0, 0.0));
  expectNearVec3(
    vec3f(select(0.0, 1.0, inside), select(0.0, 1.0, outside), 0.0),
    vec3f(1.0, 0.0, 0.0)
  );
}

@test fn diagonalExtent() {
  var box: AABB;
  box.min = vec3f(-1.0, -2.0, -3.0);
  box.max = vec3f(1.0, 2.0, 3.0);
  expectNearVec3(diagonal(box), vec3f(2.0, 4.0, 6.0));
}

@test fn expandScalar() {
  var box: AABB;
  box.min = vec3f(-1.0, -1.0, -1.0);
  box.max = vec3f(1.0, 1.0, 1.0);
  expand(&box, 0.5);
  expectNearVec3(vec3f(box.min.x, box.max.x, 0.0), vec3f(-1.5, 1.5, 0.0));
}

@test fn expandWithPoint() {
  var box: AABB;
  box.min = vec3f(-1.0, -1.0, -1.0);
  box.max = vec3f(1.0, 1.0, 1.0);
  expand2(&box, vec3f(2.0, -2.0, 0.5));
  expectNearVec3(vec3f(box.min.y, box.max.x, 0.0), vec3f(-2.0, 2.0, 0.0));
}

@test fn expandWithAABB() {
  var box1: AABB;
  box1.min = vec3f(-1.0, -1.0, -1.0);
  box1.max = vec3f(1.0, 1.0, 1.0);
  var box2: AABB;
  box2.min = vec3f(0.0, -2.0, 0.0);
  box2.max = vec3f(2.0, 0.0, 2.0);
  expand3(&box1, box2);
  expectNearVec3(vec3f(box1.min.y, box1.max.x, box1.max.z), vec3f(-2.0, 2.0, 2.0));
}

@test fn squareEqualize() {
  var box: AABB;
  box.min = vec3f(-1.0, -2.0, -0.5);
  box.max = vec3f(1.0, 2.0, 0.5);
  square(&box);
  let diag = box.max - box.min;
  expectNearVec3(diag, vec3f(4.0, 4.0, 4.0));
}
