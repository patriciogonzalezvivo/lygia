import wgsl_test::{expectNearVec3, expectNearVec4};
import lygia::color::tonemap::aces::{tonemapACES3, tonemapACES4};
import lygia::color::tonemap::debug::{tonemapDebug3, tonemapDebug4};
import lygia::color::tonemap::filmic::{tonemapFilmic3, tonemapFilmic4};
import lygia::color::tonemap::linear::{tonemapLinear3, tonemapLinear4};
import lygia::color::tonemap::reinhard::{tonemapReinhard3, tonemapReinhard4};
import lygia::color::tonemap::reinhardJodie::{tonemapReinhardJodie3, tonemapReinhardJodie4};
import lygia::color::tonemap::uncharted::{tonemapUncharted3, tonemapUncharted4, uncharted2Tonemap};
import lygia::color::tonemap::uncharted2::{tonemapUncharted23, tonemapUncharted24};
import lygia::color::tonemap::unreal::{tonemapUnreal3, tonemapUnreal4};

// vec3 variants

@test fn aces3() {
  expectNearVec3(tonemapACES3(vec3f(2.0, 1.5, 1.0)), vec3f(0.9149, 0.8768, 0.8038));
}

@test fn debug3() {
  expectNearVec3(tonemapDebug3(vec3f(1.5, 1.0, 0.5)), vec3f(0.5718, 0.9076, 0.0));
}

@test fn filmic3() {
  expectNearVec3(tonemapFilmic3(vec3f(2.0, 1.5, 1.0)), vec3f(0.9128, 0.8874, 0.8412));
}

@test fn linear3() {
  expectNearVec3(tonemapLinear3(vec3f(2.0, 1.5, 1.0)), vec3f(2.0, 1.5, 1.0));
}

@test fn reinhard3() {
  expectNearVec3(tonemapReinhard3(vec3f(2.0, 1.5, 1.0)), vec3f(0.7782, 0.5836, 0.3891));
}

@test fn reinhardJodie3() {
  expectNearVec3(tonemapReinhardJodie3(vec3f(2.0, 1.5, 1.0)), vec3f(0.7038, 0.5935, 0.4445));
}

@test fn uncharted3() {
  expectNearVec3(tonemapUncharted3(vec3f(2.0, 1.5, 1.0)), vec3f(0.7132, 0.6208, 0.4929));
}

@test fn uncharted2Raw() {
  expectNearVec3(uncharted2Tonemap(vec3f(2.0, 1.5, 1.0)), vec3f(0.3574, 0.2963, 0.2207));
}

@test fn uncharted2_3() {
  expectNearVec3(tonemapUncharted23(vec3f(2.0, 1.5, 1.0)), vec3f(0.4929, 0.4086, 0.3043));
}

@test fn unreal3() {
  expectNearVec3(tonemapUnreal3(vec3f(2.0, 1.5, 1.0)), vec3f(0.9457, 0.9236, 0.8823));
}

// vec4 variants (alpha preserved)

@test fn aces4() {
  expectNearVec4(tonemapACES4(vec4f(2.0, 1.5, 1.0, 0.8)), vec4f(0.9149, 0.8768, 0.8038, 0.8));
}

@test fn debug4() {
  expectNearVec4(tonemapDebug4(vec4f(1.5, 1.0, 0.5, 0.7)), vec4f(0.5718, 0.9076, 0.0, 0.7));
}

@test fn filmic4() {
  expectNearVec4(tonemapFilmic4(vec4f(2.0, 1.5, 1.0, 0.8)), vec4f(0.9128, 0.8874, 0.8412, 0.8));
}

@test fn linear4() {
  expectNearVec4(tonemapLinear4(vec4f(2.0, 1.5, 1.0, 0.5)), vec4f(2.0, 1.5, 1.0, 0.5));
}

@test fn reinhard4() {
  expectNearVec4(tonemapReinhard4(vec4f(2.0, 1.5, 1.0, 0.6)), vec4f(0.7782, 0.5836, 0.3891, 0.6));
}

@test fn reinhardJodie4() {
  expectNearVec4(tonemapReinhardJodie4(vec4f(2.0, 1.5, 1.0, 0.75)), vec4f(0.7038, 0.5935, 0.4445, 0.75));
}

@test fn uncharted4() {
  expectNearVec4(tonemapUncharted4(vec4f(2.0, 1.5, 1.0, 0.9)), vec4f(0.7132, 0.6208, 0.4929, 0.9));
}

@test fn uncharted2_4() {
  expectNearVec4(tonemapUncharted24(vec4f(2.0, 1.5, 1.0, 0.85)), vec4f(0.4929, 0.4086, 0.3043, 0.85));
}

@test fn unreal4() {
  expectNearVec4(tonemapUnreal4(vec4f(2.0, 1.5, 1.0, 0.65)), vec4f(0.9457, 0.9236, 0.8823, 0.65));
}
