import wgsl_test::{expectNearVec3, expectNearVec4, expectWithinVec3, expectWithinVec4};
import lygia::color::space::oklab2srgb::{oklab2srgb, oklab2srgb4};
import lygia::color::space::srgb2oklab::{srgb2oklab, srgb2oklab4};
import lygia::color::space::oklab2rgb::{oklab2rgb, oklab2rgb4};
import lygia::color::space::rgb2oklab::{rgb2oklab, rgb2oklab4};

@test fn oklab2srgbRed() {
  expectNearVec3(oklab2srgb(vec3f(0.628, 0.225, 0.126)), vec3f(1.0, 0.0, 0.0));
}

@test fn srgb2oklabRed() {
  expectNearVec3(srgb2oklab(vec3f(1.0, 0.0, 0.0)), vec3f(0.628, 0.2249, 0.1258));
}

@test fn oklab2rgbRed() {
  expectWithinVec3(oklab2rgb(vec3f(0.628, 0.225, 0.126)), vec3f(1.0008, -0.0002, -0.0002), 1e-3, 1e-4);
}

@test fn rgb2oklabRed() {
  expectNearVec3(rgb2oklab(vec3f(1.0, 0.0, 0.0)), vec3f(0.628, 0.2249, 0.1258));
}

@test fn oklab2rgb4Alpha() {
  expectWithinVec4(oklab2rgb4(vec4f(0.628, 0.225, 0.126, 0.45)), vec4f(1.0008, -0.0002, -0.0002, 0.45), 1e-3, 1e-4);
}

@test fn oklab2srgb4Alpha() {
  expectNearVec4(oklab2srgb4(vec4f(0.628, 0.225, 0.126, 0.35)), vec4f(1.0, 0.0, 0.0, 0.35));
}

@test fn rgb2oklab4Alpha() {
  expectNearVec4(rgb2oklab4(vec4f(1.0, 0.0, 0.0, 0.6)), vec4f(0.628, 0.2249, 0.1258, 0.6));
}

@test fn srgb2oklab4Alpha() {
  expectNearVec4(srgb2oklab4(vec4f(1.0, 0.0, 0.0, 0.85)), vec4f(0.62796, 0.22486, 0.12585, 0.85));
}
