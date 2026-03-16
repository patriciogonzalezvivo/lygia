#include "lygia/generative/random.wgsl"

/*
contributors: [Morgan McGuire, Chris Wyman]
description: |
    hashed alpha testing we aim for quality equivalent to stochastic
    alpha testing while simultaneously achieving stability equivalent to
    traditional alpha testing
    https://cwyman.org/papers/i3d17_hashedAlpha.pdf
use:
    <float> alphaHashing(<vec3> pos[, <float> scale])
    <void> alphaHashingTest(<vec3> pos, <float> alpha[, <float> scale])
options:
    - ALPHAHASHING_FNC: haching/random function
    - ALPHAHASHING_OFFSET: time offset for hashing
*/

// #define ALPHAHASHING_FNC random

fn alphaHashing3(p: vec3f, s: f32) -> f32 {
    let maxDeriv = max(length(dpdx(p.xy)), length(dpdy(p.xy)));
    let pixScale = 1.0 / (s*maxDeriv);
    vec2 pixScales = vec2f(  exp2(floor(log2(pixScale))),
                            exp2(ceil(log2(pixScale))));
    vec2 alpha = vec2f(  ALPHAHASHING_FNC(floor(pixScales.x * p)),
                        ALPHAHASHING_FNC(floor(pixScales.y * p)));
    let lerpFactor = fract(log2(pixScale));
    let x = (1.0 - lerpFactor) * alpha.x + lerpFactor * alpha.y;
    let t = min(lerpFactor, 1.0 - lerpFactor);
    vec3 cases = vec3f(  x * x / (2.0 * t * (1.0 - t)),
                        (x - 0.5 * t) / (1.0 - t),
                        1.0 - (1.0 - x) * (1.0 - x) / (2.0 * t * (1.0 - t)));
    let threshold = (x < 1.0 - t) ? ((x < t) ? cases.x : cases.y) : cases.z;

    threshold = fract(threshold + ALPHAHASHING_OFFSET);

    return clamp(threshold, 1.0e-6, 1.0);
}

fn alphaHashingTest3(p: vec3f, a: f32, s: f32) {
    if (a < alphaHashing(p, s))
        discard;
}

fn alphaHashing3a(p: vec3f) -> f32 { return alphaHashing(p, 1.0); }
fn alphaHashingTest3a(p: vec3f, a: f32) { alphaHashingTest(p, a, 1.0); }
