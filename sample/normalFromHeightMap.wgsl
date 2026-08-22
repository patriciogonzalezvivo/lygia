#include "../sampler.wgsl"
#include "../math/pow3.wgsl"

/*
contributors: Shadi El Hajj
description: Given a height map texture, calculate normal at point (s, t) 
use: normalFromHeightMap(<SAMPLER_TYPE> heightMap, <vec2> st, <float> strength, <float> offset)
options:
    - SAMPLE_CHANNEL: texture channel to sample from. Defaults to 0 (red)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn normalFromHeightMap(heightMap: SAMPLER_TYPE, st: vec2f, strength: f32, offset: f32) -> vec3f
{
    const SAMPLE_CHANNEL: f32 = 0;
    offset = pow3(offset) * 0.1;
    
    let p = SAMPLER_FNC(heightMap, st)[SAMPLE_CHANNEL];
    let h = SAMPLER_FNC(heightMap, st + vec2f(offset, 0.0))[SAMPLE_CHANNEL];
    let v = SAMPLER_FNC(heightMap, st + vec2f(0.0, offset))[SAMPLE_CHANNEL];

    let a = vec3f(1, 0, (h - p) * strength);
    let b = vec3f(0, 1, (v - p) * strength);

    return normalize(cross(a, b));
}

fn normalFromHeightMapa(heightMap: SAMPLER_TYPE, st: vec2f, strength: f32) -> vec3f
{
    return normalFromHeightMap(heightMap, st, strength, 0.5);

}
