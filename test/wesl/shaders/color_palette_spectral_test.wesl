// Native WESL tests for spectral palette functions

import wgsl_test::expectNearVec3;
import lygia::color::palette::spectral::geoffrey::spectral_geoffrey;
import lygia::color::palette::spectral::soft::spectral_soft;
import lygia::color::palette::spectral::gems::spectral_gems;
import lygia::color::palette::spectral::zucconi::spectral_zucconi;
import lygia::color::palette::spectral::zucconi6::spectral_zucconi6;
import lygia::color::palette::spectral::{spectral, spectralExpand};

// ============ spectral_geoffrey ============
// Formula: r = (t * 2.0 - 0.5) * 2.1 - vec3(1.8, 1.14, 0.3); return 0.99 - r*r

@test
fn geoffreyMid() {
    // t=0.5: green peak
    expectNearVec3(spectral_geoffrey(0.5), vec3f(0.4275, 0.9819, 0.4275));
}

@test
fn geoffreyViolet() {
    // t=0.0: violet end (out of gamut)
    expectNearVec3(spectral_geoffrey(0.0), vec3f(-7.1325, -3.8061, -0.8325));
}

@test
fn geoffreyRed() {
    // t=1.0: red end (out of gamut)
    expectNearVec3(spectral_geoffrey(1.0), vec3f(-0.8325, -3.0501, -7.1325));
}

// ============ spectral_soft ============
// Uses sin waves with phase offset, then pow(4)

@test
fn softMid() {
    // x=0.5: freq = PI/2, sin values near peak
    expectNearVec3(spectral_soft(0.5), vec3f(0.5931, 1.0, 0.5931));
}

@test
fn softStart() {
    // x=0.0: freq = 0
    expectNearVec3(spectral_soft(0.0), vec3f(0.0528, 0.0, 0.0528));
}

// ============ spectral_gems ============
// From GPU Gems, uses bump function

@test
fn gemsMid() {
    // x=0.5: pure green
    expectNearVec3(spectral_gems(0.5), vec3f(0.0, 1.0, 0.0));
}

@test
fn gemsQuarter() {
    // x=0.25: pure blue
    expectNearVec3(spectral_gems(0.25), vec3f(0.0, 0.0, 1.0));
}

@test
fn gemsThreeQuarter() {
    // x=0.75: pure red
    expectNearVec3(spectral_gems(0.75), vec3f(1.0, 0.0, 0.0));
}

// ============ spectral_zucconi ============
// Optimized bump-based spectrum

@test
fn zucconiMid() {
    // x=0.5: yellow-green region
    expectNearVec3(spectral_zucconi(0.5), vec3f(0.4964, 0.8404, 0.2163));
}

@test
fn zucconiRed() {
    // x=0.7: red region
    expectNearVec3(spectral_zucconi(0.7), vec3f(0.9765, 0.4925, 0.0));
}

// ============ spectral_zucconi6 ============
// Higher quality 6-term approximation

@test
fn zucconi6Mid() {
    // x=0.5: yellow-green region
    expectNearVec3(spectral_zucconi6(0.5), vec3f(0.4964, 0.8472, 0.1837));
}

@test
fn zucconi6Blue() {
    // x=0.3: cyan region
    expectNearVec3(spectral_zucconi6(0.3), vec3f(0.0, 0.5299, 0.4708));
}

// ============ spectral (main polynomial) ============

@test
fn spectralMid() {
    // x=0.5: yellow region
    expectNearVec3(spectral(0.5), vec3f(0.6734, 0.9020, 0.2002));
}

@test
fn spectralExpandMid() {
    // With l=0.5 expansion parameter
    expectNearVec3(spectralExpand(0.5, 0.5), vec3f(0.8044, 1.0, 0.8044));
}
