#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: sample a texture with a looping flow animation, using a direction to push, an elapse time and a cycle.
use: sampleFlow(<SAMPLER_TYPE> tex, <vec2> st, <vec2> dir, <float> time, <float> cycle)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLEFLOW
#define FNC_SAMPLEFLOW
vec4 sampleFlow(SAMPLER_TYPE tex, vec2 st, vec2 dir, float time, float cycle) {
    float halfCycle = cycle * 0.5;

    float flowOffset0 = mod(time, cycle);
    float flowOffset1 = mod(time + halfCycle, cycle);

    float phase0 = flowOffset0;
    float phase1 = flowOffset1;

    // Sample normal map.
    vec4 A = SAMPLER_FNC(tex, (st + dir * phase0) );
    vec4 B = SAMPLER_FNC(tex, (st + dir * phase1) );

    float f = (abs(halfCycle - flowOffset0) / halfCycle);
    return mix( A, B, f );
}
#endif