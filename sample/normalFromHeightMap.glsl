#include "../sampler.glsl"
#include "../math/pow3.glsl"

/*
contributors: Shadi El Hajj
description: Given a height map texture, calculate normal at point (s, t) 
use: normalFromHeightMap(<SAMPLER_TYPE> heightMap, <vec2> st, <float> strength, <float> offset)
options:
    - SAMPLE_CHANNEL: texture channel to sample from. Defaults to 0 (red)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

#ifndef SAMPLE_CHANNEL
#define SAMPLE_CHANNEL 0
#endif

vec3 normalFromHeightMap(SAMPLER_TYPE heightMap, vec2 st, float strength, float offset)
{
    offset = pow3(offset) * 0.1;
    
    float p = SAMPLER_FNC(heightMap, st)[SAMPLE_CHANNEL];
    float h = SAMPLER_FNC(heightMap, st + vec2(offset, 0.0))[SAMPLE_CHANNEL];
    float v = SAMPLER_FNC(heightMap, st + vec2(0.0, offset))[SAMPLE_CHANNEL];

    vec3 a = vec3(1, 0, (h - p) * strength);
    vec3 b = vec3(0, 1, (v - p) * strength);

    return normalize(cross(a, b));
}

vec3 normalFromHeightMap(SAMPLER_TYPE heightMap, vec2 st, float strength)
{
    return normalFromHeightMap(heightMap, st, strength, 0.5);

}
