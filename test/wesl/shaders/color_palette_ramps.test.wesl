// Native WESL tests for fire and water palette ramps

import wgsl_test::expectNearVec3;
import lygia::color::palette::fire::fire;
import lygia::color::palette::water::water;

// ============ fire ============
// Formula: vec3f(1.0, 0.25, 0.0625) * exp(4.0 * x - 1.0)

@test
fn fireCool() {
    // x=0.0: cool end (exp(-1) = 0.368)
    expectNearVec3(fire(0.0), vec3f(0.3679, 0.0920, 0.0230));
}

@test
fn fireNeutral() {
    // x=0.25: neutral (exp(0) = 1.0)
    expectNearVec3(fire(0.25), vec3f(1.0, 0.25, 0.0625));
}

@test
fn fireWarm() {
    // x=0.5: warm (exp(1) = 2.718)
    expectNearVec3(fire(0.5), vec3f(2.7183, 0.6796, 0.1699));
}

@test
fn fireHot() {
    // x=1.0: hot (exp(3) = 20.09)
    expectNearVec3(fire(1.0), vec3f(20.0855, 5.0214, 1.2553));
}

// ============ water ============
// Formula: pow(vec3f(0.1, 0.7, 0.8), vec3f(4.0 * saturate(1.0 - x)))

@test
fn waterDeep() {
    // x=0.0: deep water (power 4)
    expectNearVec3(water(0.0), vec3f(0.0001, 0.2401, 0.4096));
}

@test
fn waterMid() {
    // x=0.5: mid water (power 2)
    expectNearVec3(water(0.5), vec3f(0.01, 0.49, 0.64));
}

@test
fn waterSurface() {
    // x=1.0: surface (power 0 = 1.0)
    expectNearVec3(water(1.0), vec3f(1.0, 1.0, 1.0));
}

@test
fn waterClamped() {
    // x > 1.0: should clamp to surface due to saturate
    expectNearVec3(water(1.5), vec3f(1.0, 1.0, 1.0));
}
