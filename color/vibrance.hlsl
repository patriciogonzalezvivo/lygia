/*
contributors: Christian Cann Schuldt Jensen ~ CeeJay.dk
description: |
    Vibrance is a smart-tool which cleverly increases the intensity of the more muted colors and leaves the already well-saturated colors alone. Prevents skin tones from becoming overly saturated and unnatural. 
    vibrance from https://github.com/CeeJayDK/SweetFX/blob/master/Shaders/Vibrance.fx 
use: <float3|float4> vibrance(<float3|float4> color, <float> v) 
license: MIT License (MIT) Copyright (c) 2014 CeeJayDK
*/

#include "../math/mmax.hlsl"
#include "../math/mmin.hlsl"
#include "luma.hlsl"

#ifndef FNC_VIBRANCE
#define FNC_VIBRANCE

float3 vibrance(in float3 color, in float v) {
    float max_color = mmax(color);
    float min_color = mmin(color);
    float sat = max_color - min_color;
    float lum = luma(color);
    return lerp(float3(lum, lum, lum), color, 1.0 + (v * 1.0 - (sign(v) * sat)));

}

float4 vibrance(in float4 color, in float v) { return float4( vibrance(color.rgb, v), color.a); }
#endif
