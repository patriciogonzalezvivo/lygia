import wgsl_test::{expectNear, expectNearVec4};
import lygia::math::consts::HALF_PI;
import lygia::space::tbn::tbn;
import lygia::space::perspective::perspective;
import lygia::space::orthographic::orthographic;
import lygia::space::translate::translate;

@test fn tbnIdentity() {
  let t = vec3f(1.0, 0.0, 0.0);
  let b = vec3f(0.0, 1.0, 0.0);
  let n = vec3f(0.0, 0.0, 1.0);
  let mat = tbn(t, b, n);
  let v = mat * vec3f(1.0, 1.0, 1.0);
  expectNearVec4(vec4f(v, 0.0), vec4f(1.0, 1.0, 1.0, 0.0));
}

@test fn perspectiveMatrix() {
  let mat = perspective(HALF_PI, 16.0 / 9.0, 0.1, 100.0);
  // f / aspect where f = 1/tan(fov/2) = 1/tan(pi/4) = 1
  // so result = 1 / (16/9) = 9/16 = 0.5625
  expectNear(mat[0][0], 0.5625);
}

@test fn orthographicMatrix() {
  let mat = orthographic(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
  // 2/(r-l) = 2/(1-(-1)) = 2/2 = 1
  expectNear(mat[0][0], 1.0);
}

@test fn translateMatrix() {
  let m = mat3x3f(
    vec3f(1.0, 0.0, 0.0),
    vec3f(0.0, 1.0, 0.0),
    vec3f(0.0, 0.0, 1.0)
  );
  let result = translate(m, vec3f(10.0, 20.0, 30.0));
  expectNearVec4(result[3], vec4f(10.0, 20.0, 30.0, 1.0));
}
