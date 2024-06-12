/*
contributors: Patricio Gonzalez Vivo
description: "2016 Banding in Games: A Noisy Rant"
use:
    - <float4|float3|float> ditherTriangleNoise(<float4|float3|float> value, <float2> fragcoord, <float> time)
    - <float4|float3|float> ditherTriangleNoise(<float4|float3|float> value, <float> time)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifdef DITHER_ANIMATED
#define DITHER_TRIANGLENOISE_ANIMATED
#endif

#ifdef DITHER_CHROMATIC
#define DITHER_TRIANGLENOISE_CHROMATIC
#endif

#ifndef RESOLUTION
#define RESOLUTION _ScreenParams
#endif

#ifndef FNC_DITHER_TRIANGLENOISE
#define FNC_DITHER_TRIANGLENOISE

float triangleNoise(in float2 n, const in float time) {
    // triangle noise, in [-1.0..1.0[ range
    #ifdef DITHER_TRIANGLENOISE_ANIMATED
    n += float2(0.07 * frac(time));
    #endif
    n  = frac(n * float2(5.3987, 5.4421));
    n += dot(n.yx, n.xy + float2(21.5351, 14.3137));

    float xy = n.x * n.y;
    // compute in [0..2[ and remap to [-1.0..1.0[
    return frac(xy * 95.4307) + frac(xy * 75.04961) - 1.0;
}

float ditherTriangleNoise(const in float b, in float2 fragcoord, const in float time) {
    fragcoord /= RESOLUTION;
    return b + triangleNoise(fragcoord, time) / 255.0;
}

float3 ditherTriangleNoise(const in float3 rgb, in float2 fragcoord, const in float time) {
    fragcoord /= RESOLUTION;

    #ifdef DITHER_TRIANGLENOISE_CHROMATIC 
    float3 dither = float3(
            triangleNoise(fragcoord, time),
            triangleNoise(fragcoord + 0.1337, time),
            triangleNoise(fragcoord + 0.3141, time));
    #else
    float n = triangleNoise(fragcoord, time);
    float3 dither = float3(n, n, n);
    #endif
            
    return rgb + dither / 255.0;
}

float4 ditherTriangleNoise(const in float4 rgba, in float2 fragcoord, const in float time) {
    fragcoord /= RESOLUTION;

    #ifdef DITHER_TRIANGLENOISE_CHROMATIC 
    float3 dither = float3(
            triangleNoise(fragcoord, time),
            triangleNoise(fragcoord + 0.1337, time),
            triangleNoise(fragcoord + 0.3141, time));
    #else
    float n = triangleNoise(fragcoord, time);
    float3 dither = float3(n, n, n);
    #endif
    return (rgba + float4(dither, dither.x)) / 255.0;
}

#endif