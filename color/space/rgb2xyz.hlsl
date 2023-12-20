#include "srgb2xyz.hlsl"
#include "rgb2srgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB color to XYZ color space.
use: <float3|float4> rgb2xyz(<float3|float4> rgb)
*/

#ifndef FNC_RGB2XYZ
#define FNC_RGB2XYZ
float3 rgb2xyz(in float3 rgb) { return SRGB2XYZ * rgb2srgb(rgb);}
float4 rgb2xyz(in float4 rgb) { return float4(rgb2xyz(rgb.rgb),rgb.a); }
#endif