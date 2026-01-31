// Native WESL tests for space transform functions

import wgsl_test::{expectNear, expectNearVec2};
import lygia::space::center::{center, center2};
import lygia::space::uncenter::{uncenter, uncenter2};
import lygia::space::flipY::flipY2;
import lygia::space::aspect::aspect;
import lygia::space::ratio::ratio;
import lygia::space::unratio::unratio;

@test
fn centerScalar() {
  expectNear(center(0.5), 0.0);
}

@test
fn centerVec2() {
  expectNearVec2(center2(vec2f(0.5, 1.0)), vec2f(0.0, 1.0));
}

@test
fn uncenterScalar() {
  expectNear(uncenter(0.0), 0.5);
}

@test
fn uncenterVec2() {
  expectNearVec2(uncenter2(vec2f(-1.0, 1.0)), vec2f(0.0, 1.0));
}

@test
fn flipYVec2() {
  expectNearVec2(flipY2(vec2f(0.5, 0.25)), vec2f(0.5, 0.75));
}

@test
fn aspectWidescreen() {
  // 1920x1080 aspect ratio adjustment
  expectNearVec2(aspect(vec2f(0.5, 0.5), vec2f(1920.0, 1080.0)), vec2f(0.5 * 1920.0 / 1080.0, 0.5));
}

@test
fn ratioWidescreen() {
  // ratio keeps 0-1 range visible while correcting aspect
  expectNearVec2(ratio(vec2f(0.5, 0.5), vec2f(1920.0, 1080.0)), vec2f(0.5, 0.5));
}

@test
fn unratioWidescreen() {
  // unratio reverses the ratio adjustment
  expectNearVec2(unratio(vec2f(0.5, 0.5), vec2f(1920.0, 1080.0)), vec2f(0.5, 0.5));
}
