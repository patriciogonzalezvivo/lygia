import wgsl_test::expectNearVec4;
import lygia::math::consts::HALF_PI;
import lygia::math::rotate2d::rotate2d;
import lygia::math::rotate3d::rotate3d;
import lygia::math::rotate3dX::rotate3dX;
import lygia::math::rotate3dY::rotate3dY;
import lygia::math::rotate3dZ::rotate3dZ;
import lygia::math::rotate4d::rotate4d;
import lygia::math::rotate4dX::rotate4dX;
import lygia::math::rotate4dY::rotate4dY;
import lygia::math::rotate4dZ::rotate4dZ;

@test fn rotate2d90() {
  let mat = rotate2d(HALF_PI);
  let result = mat * vec2f(1.0, 0.0);
  expectNearVec4(vec4f(result.x, result.y, 0.0, 0.0), vec4f(0.0, 1.0, 0.0, 0.0));
}

@test fn rotate3dAroundZ() {
  let axis = normalize(vec3f(0.0, 0.0, 1.0));
  let mat = rotate3d(axis, HALF_PI);
  let result = mat * vec3f(1.0, 0.0, 0.0);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.0, -1.0, 0.0, 0.0));
}

@test fn rotate3dXAxis() {
  let mat = rotate3dX(HALF_PI);
  let result = mat * vec3f(0.0, 1.0, 0.0);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.0, 0.0, 1.0, 0.0));
}

@test fn rotate3dYAxis() {
  let mat = rotate3dY(HALF_PI);
  let result = mat * vec3f(1.0, 0.0, 0.0);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.0, 0.0, -1.0, 0.0));
}

@test fn rotate3dZAxis() {
  let mat = rotate3dZ(HALF_PI);
  let result = mat * vec3f(1.0, 0.0, 0.0);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.0, -1.0, 0.0, 0.0));
}

@test fn rotate4dAroundZ() {
  let axis = normalize(vec3f(0.0, 0.0, 1.0));
  let mat = rotate4d(axis, HALF_PI);
  let result = mat * vec4f(1.0, 0.0, 0.0, 1.0);
  expectNearVec4(result, vec4f(0.0, -1.0, 0.0, 1.0));
}

@test fn rotate4dXAxis() {
  let mat = rotate4dX(HALF_PI);
  let result = mat * vec4f(0.0, 1.0, 0.0, 1.0);
  expectNearVec4(result, vec4f(0.0, 0.0, -1.0, 1.0));
}

@test fn rotate4dYAxis() {
  let mat = rotate4dY(HALF_PI);
  let result = mat * vec4f(1.0, 0.0, 0.0, 1.0);
  expectNearVec4(result, vec4f(0.0, 0.0, -1.0, 1.0));
}

@test fn rotate4dZAxis() {
  let mat = rotate4dZ(HALF_PI);
  let result = mat * vec4f(1.0, 0.0, 0.0, 1.0);
  expectNearVec4(result, vec4f(0.0, -1.0, 0.0, 1.0));
}
