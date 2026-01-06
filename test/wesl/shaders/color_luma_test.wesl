// Native WESL tests for color luminance/luma functions

import wgsl_test::{expectNear, expectNearVec2};
import lygia::color::luminance::luminance;
import lygia::color::luminance::luminance4;
import lygia::color::luma::luma;
import lygia::color::luma::luma3;
import lygia::color::luma::luma4;

@test
fn luminanceOrange() {
  let color = vec3f(1.0, 0.5, 0.0);
  // Luminance: 1*0.2125 + 0.5*0.7154 + 0*0.0721 = 0.5702
  expectNear(luminance(color), 0.5702);
}

@test
fn luminance4Orange() {
  let color = vec4f(1.0, 0.5, 0.0, 0.8);
  // Alpha ignored, same as luminance
  expectNear(luminance4(color), 0.5702);
}

@test
fn luma3Orange() {
  let rgb = vec3f(1.0, 0.5, 0.0);
  // Luma using Rec709: 1.0*0.2126 + 0.5*0.7152 + 0.0*0.0722 = 0.5702
  expectNear(luma3(rgb), 0.5702);
}

@test
fn lumaGrayscaleConsistency() {
  let value = 0.75;
  let gray = vec3f(0.75, 0.75, 0.75);
  // For grayscale, both should give same result
  expectNearVec2(vec2f(luma(value), luma3(gray)), vec2f(0.75, 0.75));
}

@test
fn luma4Mixed() {
  let color = vec4f(0.8, 0.3, 0.1, 0.6);
  // Luma: 0.8*0.2126 + 0.3*0.7152 + 0.1*0.0722 = 0.39186
  expectNear(luma4(color), 0.39186);
}
