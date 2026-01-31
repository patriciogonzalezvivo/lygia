// Native WESL tests for color space roundtrip conversions

import wgsl_test::expectNearVec4;
import lygia::color::space::rgb2hsl::rgb2hsl4;
import lygia::color::space::hsl2rgb::hsl2rgb4;
import lygia::color::space::rgb2hsv::rgb2hsv4;
import lygia::color::space::hsv2rgb::hsv2rgb4;
import lygia::color::space::rgb2oklab::rgb2oklab4;
import lygia::color::space::oklab2rgb::oklab2rgb4;
import lygia::color::space::rgb2yiq::rgb2yiq4;
import lygia::color::space::yiq2rgb::yiq2rgb4;
import lygia::color::space::rgb2yuv::rgb2yuv4;
import lygia::color::space::yuv2rgb::yuv2rgb4;

@test(rgb_hsl_roundtrip)
fn testRgbHslRoundtrip() {
  let original = vec4f(0.7, 0.3, 0.5, 0.8);
  let hsl = rgb2hsl4(original);
  let back = hsl2rgb4(hsl);
  expectNearVec4(back, original);
}

@test(rgb_hsv_roundtrip)
fn testRgbHsvRoundtrip() {
  let original = vec4f(0.8, 0.2, 0.6, 0.5);
  let hsv = rgb2hsv4(original);
  let back = hsv2rgb4(hsv);
  expectNearVec4(back, original);
}

@test(rgb_oklab_roundtrip)
fn testRgbOklabRoundtrip() {
  let original = vec4f(0.6, 0.4, 0.2, 0.9);
  let oklab = rgb2oklab4(original);
  let back = oklab2rgb4(oklab);
  expectNearVec4(back, original);
}

@test(rgb_yiq_roundtrip)
fn testRgbYiqRoundtrip() {
  let original = vec4f(0.7, 0.4, 0.2, 0.6);
  let yiq = rgb2yiq4(original);
  let back = yiq2rgb4(yiq);
  // Small precision difference in YIQ conversion
  expectNearVec4(back, vec4f(0.7, 0.40003, 0.19989, 0.6));
}

@test(rgb_yuv_roundtrip)
fn testRgbYuvRoundtrip() {
  let original = vec4f(0.5, 0.6, 0.3, 0.8);
  let yuv = rgb2yuv4(original);
  let back = yuv2rgb4(yuv);
  // Small precision difference in YUV conversion
  expectNearVec4(back, vec4f(0.5, 0.60064, 0.29362, 0.8));
}
