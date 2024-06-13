/*
contributors: Patricio Gonzalez Vivo
description: Set of dither methods
use: <float4|float3|float> dither(<float4|float3|float> value[, <float> time])
options:
    - DITHER_FNC
    - TIME_SECS: elapsed time in seconds
    - RESOLUTION: view port resolution
    - BLUENOISE_TEXTURE_RESOLUTION
    - BLUENOISE_TEXTURE
    - DITHER_ANIMATED
    - DITHER_CHROMATIC
    - SAMPLER_FNC
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef TIME_SECS
#if defined(UNITY_COMPILER_HLSL)
#define TIME_SECS _Time.y
#endif
#endif

#ifndef RESOLUTION
#if defined(UNITY_COMPILER_HLSL)
#define RESOLUTION _ScreenParams
#endif
#endif

#include "dither/interleavedGradientNoise.hlsl"
#include "dither/vlachos.hlsl"
#include "dither/triangleNoise.hlsl"
#include "dither/blueNoise.hlsl"
#include "dither/shift.hlsl"

#ifndef DITHER_FNC
#ifdef TARGET_MOBILE
#define DITHER_FNC ditherInterleavedGradientNoise
#else
#define DITHER_FNC ditherVlachos
#endif
#endif

#ifndef FNC_DITHER
#define FNC_DITHER

float dither(float b, float2 fragcoord, const in float time) {
    return DITHER_FNC(b, fragcoord, time);
}

float3 dither(float3 rgb, float2 fragcoord, const in float time) {
    return DITHER_FNC(rgb, fragcoord, time);
}

float4 dither(float4 rgba, float2 fragcoord, const in float time) {
    return DITHER_FNC(rgba, fragcoord, time);
}

#ifdef TIME_SECS
float dither(float b, float2 fragcoord) {
    return dither(b, fragcoord, TIME_SECS);
}

float3 dither(float3 rgb, float2 fragcoord) {
    return dither(rgb, fragcoord, TIME_SECS);
}

float4 dither(float4 rgba, float2 fragcoord) {
    return dither(rgba, fragcoord, TIME_SECS);
}
#endif

#endif