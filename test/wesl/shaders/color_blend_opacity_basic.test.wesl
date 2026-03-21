import wgsl_test::expectNearVec3;
import lygia::color::blend::add::blendAdd3Opacity;
import lygia::color::blend::multiply::blendMultiply3Opacity;
import lygia::color::blend::screen::blendScreenWithOpacity3;
import lygia::color::blend::average::blendAverage3Opacity;
import lygia::color::blend::lighten::blendLighten3Opacity;
import lygia::color::blend::darken::blendDarken3Opacity;

// Add blend with opacity
@test fn blendAddHalf() {
  let base = vec3f(0.3, 0.5, 0.7);
  let blend = vec3f(0.8, 0.2, 0.4);
  expectNearVec3(blendAdd3Opacity(base, blend, 0.5), vec3f(0.65, 0.6, 0.85));
}

@test fn blendAddZero() {
  let base = vec3f(0.3, 0.5, 0.7);
  let blend = vec3f(0.8, 0.2, 0.4);
  expectNearVec3(blendAdd3Opacity(base, blend, 0.0), vec3f(0.3, 0.5, 0.7));
}

@test fn blendAddFull() {
  let base = vec3f(0.3, 0.5, 0.7);
  let blend = vec3f(0.8, 0.2, 0.4);
  expectNearVec3(blendAdd3Opacity(base, blend, 1.0), vec3f(1.0, 0.7, 1.0));
}

// Multiply blend with opacity
@test fn blendMultiplyHalf() {
  let base = vec3f(0.8, 0.6, 0.4);
  let blend = vec3f(0.5, 0.5, 0.5);
  expectNearVec3(blendMultiply3Opacity(base, blend, 0.5), vec3f(0.6, 0.45, 0.3));
}

// Screen blend with opacity
@test fn blendScreenHalf() {
  let base = vec3f(0.4, 0.5, 0.6);
  let blend = vec3f(0.3, 0.4, 0.5);
  expectNearVec3(blendScreenWithOpacity3(base, blend, 0.5), vec3f(0.49, 0.6, 0.7));
}

// Average blend with opacity
@test fn blendAverageHalf() {
  let base = vec3f(0.6, 0.4, 0.8);
  let blend = vec3f(0.4, 0.8, 0.2);
  expectNearVec3(blendAverage3Opacity(base, blend, 0.5), vec3f(0.55, 0.5, 0.65));
}

// Lighten blend with opacity
@test fn blendLightenHalf() {
  let base = vec3f(0.6, 0.4, 0.5);
  let blend = vec3f(0.3, 0.7, 0.5);
  expectNearVec3(blendLighten3Opacity(base, blend, 0.5), vec3f(0.6, 0.55, 0.5));
}

// Darken blend with opacity
@test fn blendDarkenHalf() {
  let base = vec3f(0.6, 0.4, 0.5);
  let blend = vec3f(0.3, 0.7, 0.5);
  expectNearVec3(blendDarken3Opacity(base, blend, 0.5), vec3f(0.45, 0.4, 0.5));
}
