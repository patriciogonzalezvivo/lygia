import wgsl_test::{expect, expectNear, expectNearVec2, expectNearVec3, expectNearVec4, expectWithin};
import lygia::math::consts::{QTR_PI, SQRT2, PI};
import lygia::space::cart2polar::cart2polar2;
import lygia::space::polar2cart::polar2cart;
import lygia::space::equirect2xyz::equirect2xyz;
import lygia::space::xyz2equirect::xyz2equirect;
import lygia::space::fisheye2xyz::fisheye2xyz;
import lygia::space::nearest::nearest;

@test fn testCart2polar2() {
  let result = cart2polar2(vec2f(1.0, 1.0));
  // atan2(1,1) = pi/4, length = sqrt(2)
  expectNearVec2(result, vec2f(QTR_PI, SQRT2));
}

@test fn testPolar2cart() {
  let result = polar2cart(vec2f(QTR_PI, SQRT2));
  expectNearVec2(result, vec2f(1.0, 1.0));
}

@test fn testEquirect2xyz() {
  let result = equirect2xyz(vec2f(0.5, 0.5)); // Center of equirect map
  // At center (0.5, 0.5): Theta=PI, Phi=PI/2 -> direction pointing left (-X axis)
  expectNearVec4(vec4f(result, 0.0), vec4f(-1.0, 0.0, 0.0, 0.0));
}

@test fn testXyz2equirect() {
  let result = xyz2equirect(vec3f(1.0, 0.0, 0.0)); // +X direction
  // atan(0, 1) = 0, + PI = PI, / (2*PI) = 0.5
  // acos(0) = PI/2, / PI = 0.5
  expectNearVec2(result, vec2f(0.5, 0.5));
}

@test fn testFisheye2xyz() {
  let uv = vec2f(0.75, 0.5);
  let result = fisheye2xyz(uv);
  // Fisheye projection should return a normalized direction vector
  let len = length(result);
  expectNear(len, 1.0);
}

@test fn testFisheye2xyzCenter() {
  let uv = vec2f(0.5, 0.5); // Center point: R=0, triggers division by zero edge case
  let result = fisheye2xyz(uv);
  expectNearVec3(result, vec3f(0.0, 1.0, 0.0));
}

@test fn testNearest() {
  // Test nearest-neighbor snapping for different UV coordinates
  let result1 = nearest(vec2f(0.7533, 0.2567), vec2f(1920.0, 1080.0));
  let result2 = nearest(vec2f(0.7500, 0.2500), vec2f(1920.0, 1080.0));
  // nearest snaps to pixel centers: floor(v*res)/res + offset
  expectNearVec2(result1, vec2f(0.7534, 0.2569));
  expectNearVec2(result2, vec2f(0.75026, 0.25046));
}
