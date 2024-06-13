#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: sampler shadowMap 
use: 
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <vec4|vec3> _coord)
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <vec2> _coord , float compare)
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <vec2> _shadowMapSize, <vec2> _coord , float compare)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SAMPLERSHADOW_FNC
#define SAMPLERSHADOW_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r
#endif

#ifndef FNC_SAMPLESHADOW
#define FNC_SAMPLESHADOW

float sampleShadow(in SAMPLER_TYPE shadowMap, in vec4 _coord) {
    vec3 shadowCoord = _coord.xyz / _coord.w;
    return SAMPLERSHADOW_FNC(shadowMap, shadowCoord.xy);
}

float sampleShadow(in SAMPLER_TYPE shadowMap, in vec3 _coord) {
    return sampleShadow(shadowMap, vec4(_coord, 1.0));
}

float sampleShadow(in SAMPLER_TYPE shadowMap, in vec2 uv, in float compare) {
    return step(compare, SAMPLERSHADOW_FNC(shadowMap, uv) );
}

float sampleShadow(in SAMPLER_TYPE shadowMap, in vec2 size, in vec2 uv, in float compare) {
    return sampleShadow(shadowMap, uv, compare);
}

#endif