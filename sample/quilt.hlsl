#include "../sampler.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    convertes QUILT of tiles into something the LookingGlass Volumetric display can render
use: sampleQuilt(<SAMPLER_TYPE> texture, <float4> calibration, <float3> tile, <float2> st, <float2> resolution)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SAMPLEQUILT_FLIPSUBP: null
    - SAMPLEQUILT_SAMPLER_FNC(POS_UV): Function used to sample into the normal map texture, defaults to texture2D(tex,POS_UV)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef SAMPLEQUILT_SAMPLER_FNC
#define SAMPLEQUILT_SAMPLER_FNC(UV) SAMPLER_FNC(tex, UV)
#endif

#ifndef FNC_QUILT
#define FNC_QUILT
float2 mapQuilt(float3 tile, float2 pos, float a) {
    float2 tile2 = tile.xy - 1.0;
    float2 dir = float2(-1.0, -1.0);

    a = frac(a) * tile.y;
    tile2.y += dir.y * floor(a);
    a = frac(a) * tile.x;
    tile2.x += dir.x * floor(a);
    return (tile2 + pos) / tile.xy;
}

float3 sampleQuilt(SAMPLER_TYPE tex, float4 calibration, float3 tile, float2 st, float2 resolution) {
    float pitch = -resolution.x / calibration.x  * calibration.y * sin(atan(abs(calibration.z)));
    float tilt = resolution.y / (resolution.x * calibration.z);
    float subp = 1.0 / (3.0 * resolution.x);
    float subp2 = subp * pitch;

    float a = (-st.x - st.y * tilt) * pitch - calibration.w;

    float3 color = float3(0.0, 0.0, 0.0);
    #ifdef SAMPLEQUILT_FLIPSUBP
    color.r = SAMPLEQUILT_SAMPLER_FNC( mapQuilt(tile, st, a-2.0*subp2) ).r;
    color.g = SAMPLEQUILT_SAMPLER_FNC( mapQuilt(tile, st, a-subp2) ).g;
    color.b = SAMPLEQUILT_SAMPLER_FNC( mapQuilt(tile, st, a) ).b;
    #else
    color.r = SAMPLEQUILT_SAMPLER_FNC( mapQuilt(tile, st, a) ).r;
    color.g = SAMPLEQUILT_SAMPLER_FNC( mapQuilt(tile, st, a-subp2) ).g;
    color.b = SAMPLEQUILT_SAMPLER_FNC( mapQuilt(tile, st, a-2.0*subp2) ).b;
    #endif
    return color;
}
#endif