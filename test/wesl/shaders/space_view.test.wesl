import wgsl_test::expectNearVec4;
import lygia::math::consts::HALF_PI;
import lygia::space::eulerView::eulerView;
import lygia::space::lookAt::lookAt;
import lygia::space::lookAtView::{lookAtView, lookAtViewRoll, lookAtViewFromDirection};

@test fn eulerViewRotateY90() {
  let viewMatrix = eulerView(vec3f(0.0, 0.0, 0.0), vec3f(0.0, HALF_PI, 0.0));
  let testPoint = viewMatrix * vec4f(1.0, 0.0, 0.0, 1.0);
  expectNearVec4(testPoint, vec4f(0.0, 0.0, -1.0, 1.0));
}

@test fn lookAtForward() {
  let viewMatrix = lookAt(vec3f(0.0, 0.0, -1.0), vec3f(0.0, 1.0, 0.0));
  let testVec = viewMatrix * vec3f(0.0, 0.0, 1.0);
  expectNearVec4(vec4f(testVec, 0.0), vec4f(0.0, 0.0, -1.0, 0.0));
}

@test fn lookAtViewPosition() {
  let viewMatrix = lookAtView(vec3f(5.0, 0.0, 0.0), vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));
  let transformed = viewMatrix * vec4f(0.0, 0.0, 0.0, 1.0);
  expectNearVec4(transformed, vec4f(5.0, 0.0, 0.0, 1.0));
}

@test fn lookAtViewWithRoll() {
  let viewMatrix = lookAtViewRoll(vec3f(0.0, 5.0, 0.0), vec3f(0.0, 0.0, 0.0), HALF_PI);
  let position = viewMatrix * vec4f(0.0, 0.0, 0.0, 1.0);
  expectNearVec4(position, vec4f(0.0, 5.0, 0.0, 1.0));
}

@test fn lookAtViewDirection() {
  let viewMatrix = lookAtViewFromDirection(vec3f(3.0, 0.0, 0.0), vec3f(1.0, 0.0, 0.0));
  let position = viewMatrix * vec4f(0.0, 0.0, 0.0, 1.0);
  expectNearVec4(position, vec4f(3.0, 0.0, 0.0, 1.0));
}
