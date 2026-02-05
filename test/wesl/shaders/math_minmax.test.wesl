// Native WESL tests for mmax/mmin functions

import wgsl_test::expectNearVec4;
import lygia::math::mmax::mmax2;
import lygia::math::mmax::mmax3;
import lygia::math::mmin::mmin2;
import lygia::math::mmin::mmin3;

@test
fn mmax2Cases() {
  expectNearVec4(
    vec4f(
      mmax2(vec2f(3.0, 7.0)),    // Max in second position
      mmax2(vec2f(9.0, 2.0)),    // Max in first position
      mmax2(vec2f(-5.0, -2.0)),  // Negative values
      mmax2(vec2f(-3.0, 4.0))    // Mixed signs
    ),
    vec4f(7.0, 9.0, -2.0, 4.0)
  );
}

@test
fn mmax3Cases() {
  expectNearVec4(
    vec4f(
      mmax3(vec3f(3.0, 7.0, 5.0)),   // Max in middle
      mmax3(vec3f(9.0, 2.0, 4.0)),   // Max at start
      mmax3(vec3f(1.0, 3.0, 8.0)),   // Max at end
      mmax3(vec3f(-6.0, -2.0, -4.0)) // Negative values
    ),
    vec4f(7.0, 9.0, 8.0, -2.0)
  );
}

@test
fn mmin2Cases() {
  expectNearVec4(
    vec4f(
      mmin2(vec2f(3.0, 7.0)),    // Min in first position
      mmin2(vec2f(9.0, 2.0)),    // Min in second position
      mmin2(vec2f(-5.0, -2.0)),  // Negative values
      mmin2(vec2f(-3.0, 4.0))    // Mixed signs
    ),
    vec4f(3.0, 2.0, -5.0, -3.0)
  );
}

@test
fn mmin3Cases() {
  expectNearVec4(
    vec4f(
      mmin3(vec3f(3.0, 7.0, 5.0)),   // Min at start
      mmin3(vec3f(9.0, 2.0, 4.0)),   // Min in middle
      mmin3(vec3f(6.0, 8.0, 1.0)),   // Min at end
      mmin3(vec3f(-2.0, 5.0, -8.0))  // Mixed signs
    ),
    vec4f(3.0, 2.0, 1.0, -8.0)
  );
}
