#include "xyz2srgb.hlsl"
#include "srgb2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a XYZ color to linear RGB.
    From http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
use: xyz2rgb(<float3|float4> color)
*/

#ifndef FNC_XYZ2RGB
#define FNC_XYZ2RGB
float3 xyz2rgb(float3 xyz) { return srgb2rgb(xyz2srgb(xyz)); }
float4 xyz2rgb(float4 xyz) { return float4(xyz2rgb(xyz.rgb), xyz.a); }
#endif