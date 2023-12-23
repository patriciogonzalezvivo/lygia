#include "rgb2hcv.hlsl"
#include "hue2rgb.hlsl"
/*
contributors: ["David Schaeffer", "tobspr", "Patricio Gonzalez Vivo"]
description: |
    Convert from linear RGB to HCY (Hue, Chroma, Luminance)
    HCY is a cylindrica. From: https://github.com/tobspr/GLSL-Color-Spaces/blob/master/ColorSpaces.inc.glsl
use: <float3|float4> rgb2hcy(<float3|float4> rgb)
license:
    - MIT License (MIT) Copyright (c) 2015 tobspr
*/

#ifndef HCY_EPSILON
#define HCY_EPSILON 1e-10
#endif

#ifndef FNC_RGB2HCY
#define FNC_RGB2HCY
float3 rgb2hcy(float3 rgb) {
    const float3 HCYwts = float3(0.299, 0.587, 0.114);
    // Corrected by David Schaeffer
    float3 HCV = rgb2hcv(rgb);
    float Y = dot(rgb, HCYwts);
    float Z = dot(hue2rgb(HCV.x), HCYwts);
    if (Y < Z) {
        HCV.y *= Z / (HCY_EPSILON + Y);
    } else {
        HCV.y *= (1.0 - Z) / (HCY_EPSILON + 1.0 - Y);
    }
    return float3(HCV.x, HCV.y, Y);
}
float4 rgb2hcy(float4 rgb) { return float4(rgb2hcy(rgb.rgb), rgb.a);}
#endif