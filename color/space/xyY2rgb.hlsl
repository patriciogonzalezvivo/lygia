#include "xyz2rgb.hlsl"
#include "xyY2xyz.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts from xyY to linear RGB
use: <float3|float4> xyY2rgb(<float3|float4> xyY)
*/

#ifndef FNC_XYY2RGB
#define FNC_XYY2RGB
float3 xyY2rgb(float3 xyY) { return xyz2rgb(xyY2xyz(xyY));}
float4 xyY2rgb(float4 xyY) { return float4(xyz2rgb(xyY2xyz(xyY.xyz)), xyY.w);}
#endif
