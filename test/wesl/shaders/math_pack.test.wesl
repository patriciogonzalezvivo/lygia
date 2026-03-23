import wgsl_test::{expectNear, expectNearVec4, expectWithin, expectWithinVec4};
import lygia::math::pack::pack;
import lygia::math::unpack::{unpack4, unpack, unpack256, unpack8, unpack16, unpack32, unpack64, unpack128, unpackBase};

// pack/unpack roundtrip
@test
fn testPackUnpackRoundtrip() {
  let original = 0.12346;
  let packed = pack(original);
  let unpacked = unpack4(packed);
  expectNear(original, 0.12346);
  expectNear(unpacked, 0.12346);
}

// pack/unpack roundtrip - multiple values
@test
fn testPackUnpackMultiple() {
  let r1 = unpack4(pack(0.0));
  let r2 = unpack4(pack(0.25));
  let r3 = unpack4(pack(0.5));
  let r4 = unpack4(pack(0.75));
  expectWithin(r1, 0.0, 0.0001, 1e-4);
  expectNear(r2, 0.25);
  expectNear(r3, 0.5);
  expectNear(r4, 0.75);
}

// unpack256 - default base 256
@test
fn testUnpack256() {
  let v1 = vec3f(1.0, 0.0, 0.0);
  let v2 = vec3f(0.5, 0.5, 0.5);
  let v3 = vec3f(1.0, 1.0, 1.0);
  let r1 = unpack256(v1);
  let r2 = unpack256(v2);
  let r3 = unpack256(v3);
  // v1: 256 / 16581375 ~ 0.000015
  expectWithin(r1, 0.000015, 0.0001, 1e-4);
  // v2: ~ 0.50787
  expectNear(r2, 0.50787);
  // v3: ~ 1.01578
  expectNear(r3, 1.01578);
}

// unpack - alias for unpack256
@test
fn testUnpackAlias() {
  let v = vec3f(0.5, 0.5, 0.5);
  let r1 = unpack(v);
  let r2 = unpack256(v);
  // Both should produce the same value (~0.50787)
  expectNear(r1, 0.50787);
  expectNear(r2, 0.50787);
}

// unpack8 - base 8
@test
fn testUnpack8() {
  let r1 = unpack8(vec3f(1.0, 0.0, 0.0));
  let r2 = unpack8(vec3f(0.0, 1.0, 0.0));
  let r3 = unpack8(vec3f(0.0, 0.0, 1.0));
  let r4 = unpack8(vec3f(0.5, 0.5, 0.5));
  expectNearVec4(vec4f(r1, r2, r3, r4), vec4f(0.015625, 0.125, 1.0, 0.5703125));
}

// unpack16 - base 16
@test
fn testUnpack16() {
  let r1 = unpack16(vec3f(1.0, 0.0, 0.0));
  let r2 = unpack16(vec3f(0.0, 1.0, 0.0));
  let r3 = unpack16(vec3f(0.0, 0.0, 1.0));
  let r4 = unpack16(vec3f(0.5, 0.5, 0.5));
  expectNearVec4(vec4f(r1, r2, r3, r4), vec4f(0.00390625, 0.0625, 1.0, 0.533203125));
}

// unpack32 - base 32
@test
fn testUnpack32() {
  let r1 = unpack32(vec3f(1.0, 0.0, 0.0));
  let r2 = unpack32(vec3f(0.0, 1.0, 0.0));
  let r3 = unpack32(vec3f(0.0, 0.0, 1.0));
  let r4 = unpack32(vec3f(0.5, 0.5, 0.5));
  expectWithinVec4(vec4f(r1, r2, r3, r4), vec4f(0.00098, 0.03125, 1.0, 0.5161), 1e-3, 1e-4);
}

// unpack64 - base 64
@test
fn testUnpack64() {
  let r1 = unpack64(vec3f(1.0, 0.0, 0.0));
  let r2 = unpack64(vec3f(0.0, 1.0, 0.0));
  let r3 = unpack64(vec3f(0.0, 0.0, 1.0));
  let r4 = unpack64(vec3f(0.5, 0.5, 0.5));
  expectNearVec4(vec4f(r1, r2, r3, r4), vec4f(0.000244, 0.015625, 1.0, 0.507935));
}

// unpack128 - base 128
@test
fn testUnpack128() {
  let r1 = unpack128(vec3f(1.0, 0.0, 0.0));
  let r2 = unpack128(vec3f(0.0, 1.0, 0.0));
  let r3 = unpack128(vec3f(0.0, 0.0, 1.0));
  let r4 = unpack128(vec3f(0.5, 0.5, 0.5));
  expectNearVec4(vec4f(r1, r2, r3, r4), vec4f(0.000061, 0.0078125, 1.0, 0.503967));
}

// unpackBase - custom base
@test
fn testUnpackBase() {
  let base = 10.0;
  let r1 = unpackBase(vec3f(1.0, 0.0, 0.0), base);
  let r2 = unpackBase(vec3f(0.0, 1.0, 0.0), base);
  let r3 = unpackBase(vec3f(0.0, 0.0, 1.0), base);
  let r4 = unpackBase(vec3f(0.5, 0.5, 0.5), base);
  expectNearVec4(vec4f(r1, r2, r3, r4), vec4f(0.01, 0.1, 1.0, 0.555));
}

// unpack4 - vec4 unpacking (ThreeJS style)
@test
fn testUnpack4ThreeJS() {
  let r1 = unpack4(vec4f(1.0, 0.0, 0.0, 0.0));
  let r2 = unpack4(vec4f(0.0, 0.0, 0.0, 1.0));
  let r3 = unpack4(vec4f(0.5, 0.5, 0.5, 0.5));
  let r4 = unpack4(vec4f(1.0, 1.0, 1.0, 1.0));
  // r1: very small value ~5.96e-8
  expectWithin(r1, 5.960464e-8, 0.0001, 1e-4);
  // r2: 255/256 = 0.99609375
  expectNear(r2, 0.99609375);
  // r3: uniform 0.5
  expectNear(r3, 0.5);
  // r4: sum of all UnpackFactors ~ 1.0
  expectNear(r4, 1.0);
}
