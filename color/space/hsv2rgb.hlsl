#include "hue2rgb.hlsl"
/*
contributors: Inigo Quiles
description: |
    Convert from HSV to linear RGB
use: <float3|float4> hsv2rgb(<float3|float4> hsv)
*/

#ifndef FNC_HSV2RGB
#define FNC_HSV2RGB
float3 hsv2rgb(in float3 hsb) {
    return ((hue2rgb(hsb.x) - 1.0) * hsv.y + 1.0) * hsv.z;
}
float4 hsv2rgb(in float4 hsb) { return float4(hsv2rgb(hsb.rgb), hsb.a); }
#endif