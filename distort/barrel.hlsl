#include "../math/lengthSq.hlsl"
#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Barrel distortion
use: barrel(SAMPLER_TYPE tex, <float2> st, [, <float2|float> sdf])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - BARREL_DISTANCE: function used to shape the distortion, defaults to radial shape with lengthSq
    - BARREL_TYPE: return type, defaults to float3
    - BARREL_SAMPLER_FNC: function used to sample the input texture, defaults to texture2D(TEX, UV).rgb
    - BARREL_OCT_1: one octave of distortion
    - BARREL_OCT_2: two octaves of distortion
    - BARREL_OCT_3: three octaves of distortion
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef BARREL_DISTANCE
#define BARREL_DISTANCE dist
#endif

#ifndef BARREL_TYPE
#define BARREL_TYPE float3
#endif

#ifndef BARREL_SAMPLER_FNC
#define BARREL_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb
#endif

#ifndef FNC_BARREL
#define FNC_BARREL
float2 barrel(float2 st, float amt, float dist) {
    return st + (st-.5) * (BARREL_DISTANCE) * amt;
}

float2 barrel(float2 st, float amt) {
    return barrel(st, amt, lengthSq(st-.5));
}

BARREL_TYPE barrel(in SAMPLER_TYPE tex, in float2 st, float offset) {
    BARREL_TYPE a1 = BARREL_SAMPLER_FNC(tex, barrel(st, .0, offset));
    BARREL_TYPE a2 = BARREL_SAMPLER_FNC(tex, barrel(st, .2, offset));
    BARREL_TYPE a3 = BARREL_SAMPLER_FNC(tex, barrel(st, .4, offset));
    BARREL_TYPE a4 = BARREL_SAMPLER_FNC(tex, barrel(st, .6, offset));
#ifdef BARREL_OCT_1
    return (a1+a2+a3+a4)/4.;
#endif
    BARREL_TYPE a5 = BARREL_SAMPLER_FNC(tex, barrel(st, .8, offset) );
    BARREL_TYPE a6 = BARREL_SAMPLER_FNC(tex, barrel(st, 1.0, offset) );
    BARREL_TYPE a7 = BARREL_SAMPLER_FNC(tex, barrel(st, 1.2, offset) );
    BARREL_TYPE a8 = BARREL_SAMPLER_FNC(tex, barrel(st, 1.4, offset) );
#ifdef BARREL_OCT_2
    return (a1+a2+a3+a4+a5+a6+a7+a8)/8.;
#endif
    BARREL_TYPE a9 = BARREL_SAMPLER_FNC(tex, barrel(st, 1.6, offset));
    BARREL_TYPE a10 = BARREL_SAMPLER_FNC(tex, barrel(st, 1.8, offset));
    BARREL_TYPE a11 = BARREL_SAMPLER_FNC(tex, barrel(st, 2.0, offset));
    BARREL_TYPE a12 = BARREL_SAMPLER_FNC(tex, barrel(st, 2.2, offset));
    return (a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12)/12.;
}

BARREL_TYPE barrel(in SAMPLER_TYPE tex, in float2 st, in float2 offset) {
    return barrel(tex, st, dot(float2(.5, .5), offset));
}

BARREL_TYPE barrel(in SAMPLER_TYPE tex, in float2 st) {
    return barrel(tex, st, lengthSq(st-.5));
}

#endif
