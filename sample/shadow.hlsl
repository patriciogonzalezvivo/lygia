#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: sampler shadowMap 
use: 
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <float4|float3> _coord)
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <float2> _coord , float compare)
    - sampleShadow(<SAMPLER_TYPE> shadowMap, <float2> _shadowMapSize, <float2> _coord , float compare)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SAMPLESHADOW
#define FNC_SAMPLESHADOW

float sampleShadow(in SAMPLER_TYPE shadowMap, in float4 _coord) {
    float3 shadowCoord = _coord.xyz / _coord.w;
    return SAMPLER_FNC(shadowMap, shadowCoord.xy).r;
}

float sampleShadow(in SAMPLER_TYPE shadowMap, in float3 _coord) {
    return sampleShadow(shadowMap, float4(_coord, 1.0));
}

float sampleShadow(in SAMPLER_TYPE shadowMap, in float2 uv, in float compare) {
    return step(compare, SAMPLER_FNC(shadowMap, uv).r );
}

float sampleShadow(in SAMPLER_TYPE shadowMap, in float2 size, in float2 uv, in float compare) {
    return sampleShadow(shadowMap, uv, compare);
}

#endif