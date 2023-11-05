#include "lygia/generative/random.glsl"

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

#ifndef ALPHAHASHING_FNC
#define ALPHAHASHING_FNC random
#endif


#ifndef FNC_ALPHAHASHING
#define FNC_ALPHAHASHING

float alphaHashing(vec3 p, float s) {
    float maxDeriv = max(length(dFdx(p.xy)), length(dFdy(p.xy)));
    float pixScale = 1.0 / (s*maxDeriv);
    vec2 pixScales = vec2(  exp2(floor(log2(pixScale))),
                            exp2(ceil(log2(pixScale))));
    vec2 alpha = vec2(  ALPHAHASHING_FNC(floor(pixScales.x * p)),
                        ALPHAHASHING_FNC(floor(pixScales.y * p)));
    float lerpFactor = fract(log2(pixScale));
    float x = (1.0 - lerpFactor) * alpha.x + lerpFactor * alpha.y;
    float t = min(lerpFactor, 1.0 - lerpFactor);
    vec3 cases = vec3(  x * x / (2.0 * t * (1.0 - t)),
                        (x - 0.5 * t) / (1.0 - t),
                        1.0 - (1.0 - x) * (1.0 - x) / (2.0 * t * (1.0 - t)));
    float threshold = (x < 1.0 - t) ? ((x < t) ? cases.x : cases.y) : cases.z;

    #ifdef ALPHAHASHING_OFFSET
    threshold = fract(threshold + ALPHAHASHING_OFFSET);
    #endif

    return clamp(threshold, 1.0e-6, 1.0);
}

void  alphaHashingTest(vec3 p, float a, float s) {
    if (a < alphaHashing(p, s))
        discard;
}

float alphaHashing(vec3 p) { return alphaHashing(p, 1.0); }
void  alphaHashingTest(vec3 p, float a) { alphaHashingTest(p, a, 1.0); }

#endif