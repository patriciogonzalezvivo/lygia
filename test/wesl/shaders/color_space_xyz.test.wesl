import wgsl_test::{expectNearVec3, expectNearVec4, expectWithinVec3, expectWithinVec4};
import lygia::color::space::rgb2xyz::{rgb2xyz, rgb2xyz4};
import lygia::color::space::srgb2xyz::{srgb2xyz, srgb2xyz4};
import lygia::color::space::xyz2rgb::{xyz2rgb, xyz2rgb4};
import lygia::color::space::xyz2srgb::{xyz2srgb, xyz2srgb4};
import lygia::color::space::rgb2xyY::{rgb2xyY, rgb2xyY4};
import lygia::color::space::xyY2rgb::{xyY2rgb, xyY2rgb4};
import lygia::color::space::xyY2srgb::{xyY2srgb, xyY2srgb4};
import lygia::color::space::xyY2xyz::{xyY2xyz, xyY2xyz4};
import lygia::color::space::xyz2xyY::{xyz2xyY, xyz2xyY4};

@test
fn testRgb2xyz() {
  expectNearVec3(rgb2xyz(vec3f(0.8, 0.7, 0.5)), vec3f(67.0487, 70.6832, 57.4054));
}

@test
fn testSrgb2xyz() {
  let result = srgb2xyz(vec3f(1.0, 0.0, 0.0));
  expectNearVec3(result, vec3f(41.2456, 21.2673, 1.9334));
}

@test
fn testXyY2rgb() {
  let xyY = vec3f(0.64, 0.33, 21.26);
  expectWithinVec3(xyY2rgb(xyY), vec3f(1.0, 0.0, 0.0), 0.001, 1e-4);
}

@test
fn testRgb2xyY() {
  let result = rgb2xyY(vec3f(1.0, 0.0, 0.0));
  expectNearVec3(result, vec3f(0.64, 0.33, 21.2673));
}

@test
fn testXyY2srgb() {
  let xyY = vec3f(0.64, 0.33, 21.26);
  expectWithinVec3(xyY2srgb(xyY), vec3f(1.0, 0.0, 0.0), 0.001, 1e-4);
}

@test
fn testXyY2xyz() {
  let xyY = vec3f(0.3127, 0.3290, 1.0);
  expectWithinVec3(xyY2xyz(xyY), vec3f(0.9505, 1.0, 1.089), 0.01, 1e-6);
}

@test
fn testXyz2xyY() {
  let xyz = vec3f(0.9505, 1.0, 1.089);
  expectNearVec3(xyz2xyY(xyz), vec3f(0.3127, 0.329, 1.0));
}

@test
fn testXyz2srgb() {
  let xyz = vec3f(41.24, 21.26, 1.93);
  expectWithinVec3(xyz2srgb(xyz), vec3f(1.0, 0.0, 0.0), 0.001, 1e-4);
}

@test
fn testXyz2rgb() {
  let xyz = vec3f(41.24, 21.26, 1.93);
  expectWithinVec3(xyz2rgb(xyz), vec3f(1.0, 0.0, 0.0), 1e-3, 1e-4);
}

@test
fn testRgb2xyz4() {
  let rgb = vec4f(0.8, 0.7, 0.5, 0.2);
  expectNearVec4(rgb2xyz4(rgb), vec4f(67.0487, 70.6832, 57.4054, 0.2));
}

@test
fn testXyz2rgb4() {
  let xyz = vec4f(41.24, 21.26, 1.93, 0.6);
  expectWithinVec4(xyz2rgb4(xyz), vec4f(1.0, 0.0, 0.0, 0.6), 1e-3, 1e-4);
}

@test
fn testRgb2xyY4() {
  let rgb = vec4f(1.0, 0.0, 0.0, 0.4);
  expectWithinVec4(rgb2xyY4(rgb), vec4f(0.64, 0.33, 21.26, 0.4), 0.01, 1e-6);
}

@test
fn testSrgb2xyz4() {
  let srgb = vec4f(1.0, 0.0, 0.0, 0.65);
  expectWithinVec4(srgb2xyz4(srgb), vec4f(41.24, 21.26, 1.93, 0.65), 0.01, 1e-6);
}

@test
fn testXyY2rgb4() {
  let xyY = vec4f(0.64, 0.33, 21.26, 0.5);
  expectWithinVec4(xyY2rgb4(xyY), vec4f(1.0, 0.0, 0.0, 0.5), 0.001, 1e-4);
}

@test
fn testXyY2srgb4() {
  let xyY = vec4f(0.64, 0.33, 21.26, 0.85);
  expectWithinVec4(xyY2srgb4(xyY), vec4f(1.0, 0.0, 0.0, 0.85), 0.001, 1e-4);
}

@test
fn testXyY2xyz4() {
  let xyY = vec4f(0.3127, 0.3290, 1.0, 0.4);
  expectWithinVec4(xyY2xyz4(xyY), vec4f(0.9505, 1.0, 1.089, 0.4), 0.01, 1e-6);
}

@test
fn testXyz2srgb4() {
  let xyz = vec4f(41.24, 21.26, 1.93, 0.2);
  expectWithinVec4(xyz2srgb4(xyz), vec4f(1.0, 0.0, 0.0, 0.2), 0.001, 1e-4);
}

@test
fn testXyz2xyY4() {
  let xyz = vec4f(0.9505, 1.0, 1.089, 0.75);
  expectNearVec4(xyz2xyY4(xyz), vec4f(0.3127, 0.329, 1.0, 0.75));
}
