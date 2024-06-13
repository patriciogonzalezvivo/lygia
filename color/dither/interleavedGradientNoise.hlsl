/*
contributors: Patricio Gonzalez Vivo
description: 'Jimenez 2014, "Next Generation Post-Processing in Call of Duty" http://advances.realtimerendering.com/s2014/index.html'
use: <float4|float3|float> interleavedGradientNoise(<float4|float3|float> value, <float> time)
options:
    - DITHER_INTERLEAVEDGRADIENTNOISE_ANIMATED
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifdef DITHER_ANIMATED
#define DITHER_INTERLEAVEDGRADIENTNOISE_ANIMATED
#endif

#ifndef FNC_DITHER_INTERLEAVEDGRADIENTNOISE
#define FNC_DITHER_INTERLEAVEDGRADIENTNOISE

float interleavedGradientNoise(const in float2 n) {
    return frac(52.982919 * frac(dot(float2(0.06711, 0.00584), n)));
}

float ditherInterleavedGradientNoise(float b, float2 fragcoord, const in float time) {
    #ifdef DITHER_INTERLEAVEDGRADIENTNOISE_ANIMATED
    fragcoord += 1337.0 * frac(time);
    #endif
    float noise = interleavedGradientNoise(fragcoord);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return b + noise / 255.0;
}

float3 ditherInterleavedGradientNoise(float3 rgb, float2 fragcoord, const in float time) {
    #ifdef DITHER_INTERLEAVEDGRADIENTNOISE_ANIMATED
    fragcoord += 1337.0 * frac(time);
    #endif
    float noise = interleavedGradientNoise(fragcoord);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return rgb.rgb + noise / 255.0;
}

float4 ditherInterleavedGradientNoise(float4 rgba, float2 fragcoord, const in float time) {
    return float4(ditherInterleavedGradientNoise(rgba.rgb, fragcoord, time), rgba.a);
}

#endif