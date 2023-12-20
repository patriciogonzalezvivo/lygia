#include "xyz2rgb.hlsl"
#include "rgb2srgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a XYZ color to sRGB.
    From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
use: <float3|float4> xyz2srgb(<float3|float4> color)
*/

#ifndef FNC_XYZ2SRGB
#define FNC_XYZ2SRGB
float3 xyz2srgb(const in float3 xyz) { return rgb2srgb(xyz2rgb(xyz)); }
float4 xyz2srgb(const in float4 xyz) { return float4(xyz2srgb(xyz.rgb), xyz.a); }
#endif