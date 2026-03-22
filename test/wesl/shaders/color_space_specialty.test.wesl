import wgsl_test::{expect, expectNear, expectNearVec3, expectNearVec4};
import lygia::color::space::rgb2heat::{rgb2heat, rgb2heat4};
import lygia::color::space::cmyk2rgb::cmyk2rgb;
import lygia::color::space::rgb2cmyk::rgb2cmyk;
import lygia::color::space::k2rgb::k2rgb;
import lygia::color::space::rgb2lms::{rgb2lms, rgb2lms4};
import lygia::color::space::lms2rgb::{lms2rgb, lms2rgb4};
import lygia::color::space::rgb2ryb::{rgb2ryb, rgb2ryb4};
import lygia::color::space::ryb2rgb::{ryb2rgb, ryb2rgb4};

@test fn testRgb2heat() {
  let x = rgb2heat(vec3f(0.8, 0.7, 0.5));
  expectNear(x, 0.854);
}

@test fn testCmyk2rgb() {
  let cmyk = vec4f(0.0, 0.0, 0.0, 0.5); // 50% gray
  let result = cmyk2rgb(cmyk);
  // CMYK(0,0,0,0.5) -> RGB (50% gray)
  expectNearVec3(result, vec3f(0.5, 0.5, 0.5));
}

@test fn testRgb2cmyk() {
  let rgb = vec3f(1.0, 0.0, 0.0); // Red
  let result = rgb2cmyk(rgb);
  // RGB(1, 0, 0) -> CMYK(0, 1, 1, 0)
  expectNearVec4(result, vec4f(0.0, 1.0, 1.0, 0.0));
}

@test fn testK2rgb() {
  // Test that lower temps are warmer (more red, less blue)
  // and higher temps are cooler (less red, more blue)
  let warm = k2rgb(3000.0);  // Warm white
  let cool = k2rgb(8000.0);  // Cool white

  // Verify warm light has more red, less blue than cool light
  expect(warm.r > cool.r); // Warm has more red than cool
  expect(cool.b > warm.b); // Cool has more blue than warm
}

@test fn testRgb2lms() {
  let rgb = vec3f(1.0, 0.0, 0.0); // Red
  let result = rgb2lms(rgb);
  // RGB(1, 0, 0) -> LMS cone response (first column of RGB2LMS matrix)
  expectNearVec3(result, vec3f(17.8824, 3.45565, 0.02996));
}

@test fn testLms2rgb() {
  let lms = vec3f(0.3, 0.2, 0.1);
  let result = lms2rgb(lms);
  // LMS(0.3, 0.2, 0.1) -> RGB via LMS2RGB matrix multiplication
  expectNearVec3(result, vec3f(0.00985, -0.00363, 0.06842));
}

@test fn testLms2rgb4() {
  let lms = vec4f(0.3, 0.2, 0.1, 0.55);
  let result = lms2rgb4(lms);
  expectNearVec4(result, vec4f(0.00985, -0.00363, 0.06842, 0.55));
}

@test fn testRgb2lms4() {
  let rgb = vec4f(1.0, 0.0, 0.0, 0.3); // Red with alpha
  let result = rgb2lms4(rgb);
  expectNearVec4(result, vec4f(17.8824, 3.45565, 0.02996, 0.3));
}

@test fn testRgb2ryb() {
  let rgb = vec3f(1.0, 0.0, 0.0); // Red
  let result = rgb2ryb(rgb);
  expectNearVec3(result, vec3f(1.0, 0.0, 0.0));
}

@test fn testRgb2ryb4() {
  let rgb = vec4f(0.0, 1.0, 0.0, 0.5); // Green with alpha
  let result = rgb2ryb4(rgb);
  // Green in RGB -> RYB: Actual result (0.0, 1.0, 0.483)
  expectNearVec4(result, vec4f(0.0, 1.0, 0.483, 0.5));
}

@test fn testRyb2rgb() {
  let ryb = vec3f(1.0, 0.0, 0.0); // Red in RYB
  let result = ryb2rgb(ryb);
  expectNearVec3(result, vec3f(1.0, 0.0, 0.0));
}

@test fn testRyb2rgb4() {
  let ryb = vec4f(0.0, 1.0, 0.0, 0.75); // Yellow in RYB with alpha
  let result = ryb2rgb4(ryb);
  // Yellow in RYB -> RGB: (1.0, 1.0, 0.0)
  expectNearVec4(result, vec4f(1.0, 1.0, 0.0, 0.75));
}

@test fn testRgb2heat4() {
  let rgb = vec4f(0.8, 0.7, 0.5, 0.6); // Color with alpha
  let result = rgb2heat4(rgb);
  // Heat map conversion: heat value 0.854 replicated as (0.854, 0.854, 0.854, 0.6)
  expectNearVec4(result, vec4f(0.854, 0.854, 0.854, 0.6));
}
