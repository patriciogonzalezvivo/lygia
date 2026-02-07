// Native WESL tests for distance functions

import wgsl_test::expectNear;
import lygia::math::lengthSq::lengthSq2;
import lygia::math::lengthSq::lengthSq3;
import lygia::math::dist::distEuclidean2;
import lygia::math::dist::distManhattan2;

@test(lengthSq2_pythagorean)
fn testLengthSq2() {
  // 3^2 + 4^2 = 9 + 16 = 25
  expectNear(lengthSq2(vec2f(3.0, 4.0)), 25.0);
}

@test(lengthSq3_unit_vector)
fn testLengthSq3() {
  // 1^2 + 2^2 + 2^2 = 1 + 4 + 4 = 9
  expectNear(lengthSq3(vec3f(1.0, 2.0, 2.0)), 9.0);
}

@test(euclidean_3_4_5_triangle)
fn testDistEuclidean2() {
  // sqrt(3^2 + 4^2) = sqrt(25) = 5
  expectNear(distEuclidean2(vec2f(0.0, 0.0), vec2f(3.0, 4.0)), 5.0);
}

@test(manhattan_taxicab)
fn testDistManhattan2() {
  // |3| + |4| = 7
  expectNear(distManhattan2(vec2f(0.0, 0.0), vec2f(3.0, 4.0)), 7.0);
}
