// Native WESL tests for RGB-related color space conversions

import wgsl_test::{expectNear, expectNearVec3, expectNearVec4};
import lygia::color::space::hsl2rgb::{hsl2rgb, hsl2rgb4};
import lygia::color::space::rgb2hsl::{rgb2hsl, rgb2hsl4};
import lygia::color::space::hsv2rgb::{hsv2rgb, hsv2rgb4};
import lygia::color::space::rgb2hsv::{rgb2hsv, rgb2hsv4};
import lygia::color::space::hcy2rgb::{hcy2rgb, hcy2rgb4};
import lygia::color::space::rgb2hcy::{rgb2hcy, rgb2hcy4};
import lygia::color::space::hsv2ryb::hsv2ryb;
import lygia::color::space::rgb2luma::{rgb2luma, rgb2luma4};
import lygia::color::space::srgb2luma::{srgb2luma, srgb2luma4};
import lygia::color::space::rgb2hcv::{rgb2hcv, rgb2hcv4};
import lygia::color::space::rgb2hue::{rgb2hue, rgb2hue4};
import lygia::color::space::hue2rgb::hue2rgb;
import lygia::color::space::rgb2srgb::{rgb2srgb_mono, rgb2srgb, rgb2srgb4};
import lygia::color::space::srgb2rgb::{srgb2rgb_mono, srgb2rgb, srgb2rgb4};

// HSL conversions
@test
fn hsl2rgbCyan() {
  let hsl = vec3f(0.5, 0.8, 0.5);
  expectNearVec3(hsl2rgb(hsl), vec3f(0.1, 0.9, 0.9));
}

@test
fn rgb2hslCyan() {
  let rgb = vec3f(0.1, 0.9, 0.9);
  expectNearVec3(rgb2hsl(rgb), vec3f(0.5, 0.8, 0.5));
}

// HSV conversions
@test
fn hsv2rgbBlue() {
  let hsv = vec3f(0.6667, 1.0, 1.0);
  expectNearVec3(hsv2rgb(hsv), vec3f(0.0002, 0.0, 1.0));
}

@test
fn rgb2hsvBlue() {
  let rgb = vec3f(0.0, 0.0, 1.0);
  expectNearVec3(rgb2hsv(rgb), vec3f(0.6667, 1.0, 1.0));
}

// HCY conversions
@test
fn hcy2rgbRed() {
  let hcy = vec3f(0.0, 0.5, 0.5);
  expectNearVec3(hcy2rgb(hcy), vec3f(0.75, 0.3934, 0.3934));
}

@test
fn rgb2hcyRed() {
  let rgb = vec3f(1.0, 0.0, 0.0);
  expectNearVec3(rgb2hcy(rgb), vec3f(0.0, 1.0, 0.2989));
}

// RYB conversion
@test
fn hsv2rybRed() {
  let hsv = vec3f(0.0, 1.0, 1.0);
  expectNearVec3(hsv2ryb(hsv), vec3f(1.0, 0.0, 0.0));
}

// Luma
@test
fn rgb2lumaOrange() {
  let rgb = vec3f(1.0, 0.5, 0.0);
  // Rec709: 1.0*0.2126 + 0.5*0.7152 + 0.0*0.0722 = 0.5702
  expectNear(rgb2luma(rgb), 0.5702);
}

@test
fn srgb2lumaOrange() {
  let srgb = vec3f(1.0, 0.5, 0.0);
  // Rec601: 1.0*0.299 + 0.5*0.587 + 0.0*0.114 = 0.5925
  expectNear(srgb2luma(srgb), 0.5925);
}

// HCV and Hue
@test
fn rgb2hcvRed() {
  let rgb = vec3f(1.0, 0.0, 0.0);
  expectNearVec3(rgb2hcv(rgb), vec3f(0.0, 1.0, 1.0));
}

@test
fn rgb2hueGreen() {
  let rgb = vec3f(0.0, 1.0, 0.0);
  expectNear(rgb2hue(rgb), 0.3333);
}

@test
fn hue2rgbGreen() {
  expectNearVec3(hue2rgb(0.3333), vec3f(0.0002, 1.0, 0.0));
}

// Vec4 overloads with alpha preservation
@test
fn hcy2rgb4Alpha() {
  let hcy = vec4f(0.0, 0.5, 0.5, 0.6);
  expectNearVec4(hcy2rgb4(hcy), vec4f(0.75, 0.3934, 0.3934, 0.6));
}

@test
fn hsl2rgb4Alpha() {
  let hsl = vec4f(0.5, 0.8, 0.5, 0.9);
  expectNearVec4(hsl2rgb4(hsl), vec4f(0.1, 0.9, 0.9, 0.9));
}

@test
fn hsv2rgb4Alpha() {
  let hsv = vec4f(0.6667, 1.0, 1.0, 0.5);
  expectNearVec4(hsv2rgb4(hsv), vec4f(0.0002, 0.0, 1.0, 0.5));
}

@test
fn rgb2hcy4Alpha() {
  let rgb = vec4f(1.0, 0.0, 0.0, 0.1);
  expectNearVec4(rgb2hcy4(rgb), vec4f(0.0, 1.0, 0.2989, 0.1));
}

@test
fn rgb2hsl4Alpha() {
  let rgb = vec4f(0.1, 0.9, 0.9, 0.5);
  expectNearVec4(rgb2hsl4(rgb), vec4f(0.5, 0.8, 0.5, 0.5));
}

@test
fn rgb2hsv4Alpha() {
  let rgb = vec4f(0.0, 0.0, 1.0, 0.9);
  expectNearVec4(rgb2hsv4(rgb), vec4f(0.6667, 1.0, 1.0, 0.9));
}

@test
fn rgb2hcv4Alpha() {
  let rgb = vec4f(1.0, 0.5, 0.0, 0.6);
  expectNearVec4(rgb2hcv4(rgb), vec4f(0.0833, 1.0, 1.0, 0.6));
}

@test
fn rgb2hue4Alpha() {
  let rgb = vec4f(0.0, 1.0, 0.0, 0.75);
  expectNearVec4(rgb2hue4(rgb), vec4f(0.3333, 0.3333, 0.3333, 0.75));
}

@test
fn rgb2luma4Alpha() {
  let rgb = vec4f(1.0, 0.5, 0.0, 0.85);
  expectNearVec4(rgb2luma4(rgb), vec4f(0.5702, 0.5702, 0.5702, 0.85));
}

@test
fn srgb2luma4Alpha() {
  let srgb = vec4f(1.0, 0.5, 0.0, 0.95);
  expectNearVec4(srgb2luma4(srgb), vec4f(0.5925, 0.5925, 0.5925, 0.95));
}

// RGB <-> sRGB conversions
@test
fn rgb2srgbMono() {
  let low = rgb2srgb_mono(0.002);  // < 0.0031308 branch
  let high = rgb2srgb_mono(0.5);   // >= 0.0031308 branch
  expectNearVec4(vec4f(low, high, 0.0, 0.0), vec4f(0.02584, 0.73536, 0.0, 0.0));
}

@test
fn srgb2rgbMono() {
  let low = srgb2rgb_mono(0.03);   // < 0.04045 branch
  let high = srgb2rgb_mono(0.735); // >= 0.04045 branch
  expectNearVec4(vec4f(low, high, 0.0, 0.0), vec4f(0.00232, 0.49946, 0.0, 0.0));
}

@test
fn rgb2srgbVec3() {
  let rgb = vec3f(0.5, 0.3, 0.1);
  expectNearVec3(rgb2srgb(rgb), vec3f(0.7354, 0.5838, 0.3492));
}

@test
fn srgb2rgbVec3() {
  let srgb = vec3f(0.735, 0.584, 0.349);
  expectNearVec3(srgb2rgb(srgb), vec3f(0.4995, 0.3002, 0.0999));
}

@test
fn rgb2srgb4Alpha() {
  let rgb = vec4f(0.5, 0.3, 0.1, 0.4);
  expectNearVec4(rgb2srgb4(rgb), vec4f(0.7354, 0.5838, 0.3492, 0.4));
}

@test
fn srgb2rgb4Alpha() {
  let srgb = vec4f(0.735, 0.584, 0.349, 0.3);
  expectNearVec4(srgb2rgb4(srgb), vec4f(0.49946, 0.30019, 0.09989, 0.3));
}
