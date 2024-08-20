#include "../sampler.hlsl"
#include "../math/pow3.hlsl"

/*
contributors: Shadi El Hajj
description: Given a height map texture, calculate normal at point (s, t) 
use: normalFromHeightMap(<SAMPLER_TYPE> heightMap, <float2> st, <float> strength, <float> offset)
options:
    - SAMPLE_CHANNEL: texture channel to sample from. Defaults to 0 (red)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef SAMPLE_CHANNEL
#define SAMPLE_CHANNEL 0
#endif

float3 normalFromHeightMap(SAMPLER_TYPE heightMap, float2 st, float strength, float offset)
{
    offset = pow3(offset) * 0.1;
    
    float p = SAMPLER_FNC(heightMap, st)[SAMPLE_CHANNEL];
    float h = SAMPLER_FNC(heightMap, st + float2(offset, 0.0))[SAMPLE_CHANNEL];
    float v = SAMPLER_FNC(heightMap, st + float2(0.0, offset))[SAMPLE_CHANNEL];

    float3 a = float3(1, 0, (h - p) * strength);
    float3 b = float3(0, 1, (v - p) * strength);

    return normalize(cross(a, b));
}

float3 normalFromHeightMap(SAMPLER_TYPE heightMap, float2 st, float strength)
{
    return normalFromHeightMap(heightMap, st, strength, 0.5);

}
