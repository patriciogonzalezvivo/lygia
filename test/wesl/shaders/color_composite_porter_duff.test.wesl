import wgsl_test::expectNearVec4;
import lygia::color::composite::sourceOver::{compositeSourceOver4, compositeSourceOver3};
import lygia::color::composite::sourceIn::{compositeSourceIn4, compositeSourceIn3};
import lygia::color::composite::sourceOut::{compositeSourceOut4, compositeSourceOut3};
import lygia::color::composite::sourceAtop::{compositeSourceAtop4, compositeSourceAtop3};
import lygia::color::composite::destinationOver::{compositeDestinationOver4, compositeDestinationOver3};
import lygia::color::composite::destinationIn::{compositeDestinationIn4, compositeDestinationIn3};
import lygia::color::composite::destinationOut::{compositeDestinationOut4, compositeDestinationOut3};
import lygia::color::composite::destinationAtop::{compositeDestinationAtop4, compositeDestinationAtop3};
import lygia::color::composite::compositeXor::{compositeXor4, compositeXor3};

// vec4 variants

@test fn sourceOver4() {
  let s = vec4f(1.0, 0.0, 0.0, 0.5);
  let d = vec4f(0.0, 0.0, 1.0, 0.5);
  expectNearVec4(compositeSourceOver4(s, d), vec4f(0.5, 0.0, 0.25, 0.75));
}

@test fn sourceIn4() {
  let s = vec4f(1.0, 0.0, 0.0, 0.8);
  let d = vec4f(0.0, 0.0, 1.0, 0.5);
  expectNearVec4(compositeSourceIn4(s, d), vec4f(0.5, 0.0, 0.0, 0.4));
}

@test fn sourceOut4() {
  let s = vec4f(1.0, 0.0, 0.0, 0.8);
  let d = vec4f(0.0, 1.0, 0.0, 0.4);
  expectNearVec4(compositeSourceOut4(s, d), vec4f(0.6, 0.0, 0.0, 0.48));
}

@test fn sourceAtop4() {
  let s = vec4f(1.0, 0.0, 0.0, 0.6);
  let d = vec4f(0.0, 0.0, 1.0, 0.5);
  expectNearVec4(compositeSourceAtop4(s, d), vec4f(0.5, 0.0, 0.4, 0.5));
}

@test fn destinationOver4() {
  let s = vec4f(1.0, 0.0, 0.0, 0.5);
  let d = vec4f(0.0, 0.0, 1.0, 0.6);
  expectNearVec4(compositeDestinationOver4(s, d), vec4f(0.4, 0.0, 1.0, 0.8));
}

@test fn destinationIn4() {
  let s = vec4f(1.0, 0.0, 0.0, 0.6);
  let d = vec4f(0.0, 1.0, 0.0, 0.8);
  expectNearVec4(compositeDestinationIn4(s, d), vec4f(0.0, 0.6, 0.0, 0.48));
}

@test fn destinationOut4() {
  let s = vec4f(1.0, 0.0, 0.0, 0.3);
  let d = vec4f(0.0, 1.0, 0.0, 0.7);
  expectNearVec4(compositeDestinationOut4(s, d), vec4f(0.0, 0.7, 0.0, 0.49));
}

@test fn destinationAtop4() {
  let s = vec4f(1.0, 0.0, 0.0, 0.7);
  let d = vec4f(0.0, 0.0, 1.0, 0.5);
  expectNearVec4(compositeDestinationAtop4(s, d), vec4f(0.5, 0.0, 0.7, 0.7));
}

@test fn xor4() {
  let s = vec4f(1.0, 0.0, 0.0, 0.6);
  let d = vec4f(0.0, 0.0, 1.0, 0.4);
  expectNearVec4(compositeXor4(s, d), vec4f(0.6, 0.0, 0.4, 0.52));
}

// vec3 variants (with separate alpha parameters)

@test fn sourceOver3() {
  let result = compositeSourceOver3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), 0.5, 0.5);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.5, 0.0, 0.25, 0.0));
}

@test fn sourceIn3() {
  let result = compositeSourceIn3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), 0.8, 0.6);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.6, 0.0, 0.0, 0.0));
}

@test fn sourceOut3() {
  let result = compositeSourceOut3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), 0.8, 0.3);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.7, 0.0, 0.0, 0.0));
}

@test fn sourceAtop3() {
  let result = compositeSourceAtop3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), 0.6, 0.5);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.5, 0.0, 0.4, 0.0));
}

@test fn destinationOver3() {
  let result = compositeDestinationOver3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), 0.5, 0.6);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.4, 0.0, 1.0, 0.0));
}

@test fn destinationIn3() {
  let result = compositeDestinationIn3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), 0.7, 0.8);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.0, 0.7, 0.0, 0.0));
}

@test fn destinationOut3() {
  let result = compositeDestinationOut3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), 0.4, 0.7);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.0, 0.6, 0.0, 0.0));
}

@test fn destinationAtop3() {
  let result = compositeDestinationAtop3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), 0.7, 0.5);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.5, 0.0, 0.7, 0.0));
}

@test fn xor3() {
  let result = compositeXor3(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), 0.6, 0.4);
  expectNearVec4(vec4f(result, 0.0), vec4f(0.6, 0.0, 0.4, 0.0));
}
