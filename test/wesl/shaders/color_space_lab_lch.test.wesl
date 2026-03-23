import wgsl_test::{expectNearVec3, expectNearVec4, expectWithinVec3, expectWithinVec4};
import lygia::color::space::lab2srgb::{lab2srgb, lab2srgb4};
import lygia::color::space::lab2rgb::{lab2rgb, lab2rgb4};
import lygia::color::space::srgb2lab::{srgb2lab, srgb2lab4};
import lygia::color::space::rgb2lab::{rgb2lab, rgb2lab4};
import lygia::color::space::lch2srgb::{lch2srgb, lch2srgb4};
import lygia::color::space::lch2rgb::{lch2rgb, lch2rgb4};
import lygia::color::space::srgb2lch::{srgb2lch, srgb2lch4};
import lygia::color::space::rgb2lch::{rgb2lch, rgb2lch4};
import lygia::color::space::lab2lch::{lab2lch, lab2lch4};
import lygia::color::space::lch2lab::{lch2lab, lch2lab4};
import lygia::color::space::lab2xyz::{lab2xyz, lab2xyz4};
import lygia::color::space::xyz2lab::{xyz2lab, xyz2lab4};

// vec3 variants

@test fn lab2srgbPurple() {
  expectNearVec3(lab2srgb(vec3f(50.0, 25.0, -25.0)), vec3f(0.5524, 0.413, 0.634));
}

@test fn lab2rgbRed() {
  expectWithinVec3(lab2rgb(vec3f(53.24, 80.09, 67.20)), vec3f(1.0, 0.0, 0.0), 1e-3, 1e-4);
}

@test fn srgb2labRed() {
  expectNearVec3(srgb2lab(vec3f(1.0, 0.0, 0.0)), vec3f(53.2408, 80.0925, 67.2032));
}

@test fn rgb2labGray() {
  expectWithinVec3(rgb2lab(vec3f(0.5, 0.5, 0.5)), vec3f(76.0693, 0.0, 0.00001), 1e-3, 1e-4);
}

@test fn lch2srgb3() {
  expectNearVec3(lch2srgb(vec3f(50.0, 30.0, 120.0)), vec3f(0.4277, 0.4903, 0.2895));
}

@test fn lch2rgbRed() {
  expectWithinVec3(lch2rgb(vec3f(53.24, 104.55, 40.0)), vec3f(1.0, 0.0, 0.0), 1e-3, 1e-4);
}

@test fn srgb2lchRed() {
  expectNearVec3(srgb2lch(vec3f(1.0, 0.0, 0.0)), vec3f(53.2408, 104.5518, 39.999));
}

@test fn rgb2lchRed() {
  expectNearVec3(rgb2lch(vec3f(1.0, 0.0, 0.0)), vec3f(53.2408, 104.5518, 39.999));
}

@test fn lab2lchConvert() {
  expectNearVec3(lab2lch(vec3f(50.0, 25.0, 25.0)), vec3f(50.0, 35.3553, 45.0));
}

@test fn lch2labConvert() {
  expectNearVec3(lch2lab(vec3f(50.0, 35.36, 45.0)), vec3f(50.0, 25.0033, 25.0033));
}

@test fn lab2xyzGray() {
  expectNearVec3(lab2xyz(vec3f(50.0, 0.0, 0.0)), vec3f(17.5061, 18.4186, 20.059));
}

@test fn xyz2labGray() {
  expectNearVec3(xyz2lab(vec3f(17.5, 18.4, 20.0)), vec3f(49.9777, 0.0615, 0.0653));
}

// vec4 variants (alpha preservation)

@test fn lab2lch4Alpha() {
  expectNearVec4(lab2lch4(vec4f(50.0, 25.0, 25.0, 0.75)), vec4f(50.0, 35.3553, 45.0, 0.75));
}

@test fn lab2rgb4Alpha() {
  expectWithinVec4(lab2rgb4(vec4f(53.24, 80.09, 67.20, 0.4)), vec4f(1.0, 0.0, 0.0, 0.4), 1e-3, 1e-4);
}

@test fn lab2srgb4Alpha() {
  expectNearVec4(lab2srgb4(vec4f(50.0, 25.0, -25.0, 0.85)), vec4f(0.5524, 0.413, 0.634, 0.85));
}

@test fn lab2xyz4Alpha() {
  expectNearVec4(lab2xyz4(vec4f(50.0, 0.0, 0.0, 0.3)), vec4f(17.5061, 18.4186, 20.059, 0.3));
}

@test fn lch2lab4Alpha() {
  expectNearVec4(lch2lab4(vec4f(50.0, 35.36, 45.0, 0.95)), vec4f(50.0, 25.0033, 25.0033, 0.95));
}

@test fn lch2rgb4Alpha() {
  expectWithinVec4(lch2rgb4(vec4f(53.24, 104.55, 40.0, 0.2)), vec4f(1.0, 0.0, 0.0, 0.2), 1e-3, 1e-4);
}

@test fn lch2srgb4Alpha() {
  expectNearVec4(lch2srgb4(vec4f(50.0, 30.0, 120.0, 0.65)), vec4f(0.4277, 0.4903, 0.2895, 0.65));
}

@test fn rgb2lab4Alpha() {
  expectWithinVec4(rgb2lab4(vec4f(0.5, 0.5, 0.5, 0.7)), vec4f(76.0693, 0.0, 0.00001, 0.7), 1e-3, 1e-4);
}

@test fn rgb2lch4Alpha() {
  expectNearVec4(rgb2lch4(vec4f(1.0, 0.0, 0.0, 0.8)), vec4f(53.2408, 104.5518, 39.999, 0.8));
}

@test fn srgb2lab4Alpha() {
  expectNearVec4(srgb2lab4(vec4f(1.0, 0.0, 0.0, 0.75)), vec4f(53.2408, 80.0925, 67.2032, 0.75));
}

@test fn srgb2lch4Alpha() {
  expectNearVec4(srgb2lch4(vec4f(1.0, 0.0, 0.0, 0.5)), vec4f(53.2408, 104.5518, 39.999, 0.5));
}

@test fn xyz2lab4Alpha() {
  expectNearVec4(xyz2lab4(vec4f(17.5, 18.4, 20.0, 0.9)), vec4f(49.9777, 0.06151, 0.06528, 0.9));
}
