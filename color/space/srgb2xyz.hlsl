#include "rgb2xyz.glsl"
#include "srgb2rgb.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a sRGB color to XYZ
use: <float3|float4> srgb2xyz(<float3|float4> rgb)
*/

#ifndef FNC_RGB2XYZ
#define FNC_RGB2XYZ
float3 srgb2xyz(in float3 srgb) { return rgb2xyz(rgb2srgb(srgb));}
float4 srgb2xyz(in float4 srgb) { return float4(rgb2xyz(srgb.rgb),rgb.a); }
#endif
