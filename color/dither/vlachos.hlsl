/*
contributors: Patricio Gonzalez Vivo
description: 'Vlachos 2016, "Advanced VR Rendering" http://alex.vlachos.com/graphics/Alex_Vlachos_Advanced_VR_Rendering_GDC2015.pdf'
use: <float4|float3|float> ditherVlachos(<float4|float3|float> value, <float> time)
options:
    - DITHER_VLACHOS_ANIMATED
    - DITHER_VLACHOS_CHROMATIC
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifdef DITHER_ANIMATED
#define DITHER_VLACHOS_ANIMATED
#endif

#ifndef FNC_DITHER_VLACHOS
#define FNC_DITHER_VLACHOS

float ditherVlachos(float b, float2 fragcoord, const in float time) {
    #ifdef DITHER_VLACHOS_ANIMATED
    fragcoord += 1337.0*frac(time);
    #endif
    float noise = dot(float2(171.0, 231.0), fragcoord);
    noise = frac(noise / 71.0);
    // remap from [0..1[ to [-1..1[
    noise = (noise * 2.0) - 1.0;
    return b + (noise / 255.0);
}

float3 ditherVlachos(float3 rgb, float2 fragcoord, const in float time) {
    #ifdef DITHER_VLACHOS_ANIMATED
    fragcoord += 1337.0 * frac(time);
    #endif
    float n = dot(float2(171.0, 231.0), fragcoord);
    float3 noise = float3(n, n, n);
    noise = frac(noise / float3(103.0, 71.0, 97.0));
    return rgb.rgb + (noise / 255.0);
}

float4 ditherVlachos(float4 rgba, float2 fragcoord, const in float time) {
    return float4(ditherVlachos(rgba.rgb, fragcoord, time), rgba.a);
}

#endif