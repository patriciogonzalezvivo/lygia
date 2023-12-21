#include "rgb2lab.hlsl"
#include "srgb2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a sRGB color to Lab
use: <float3|float4> srgb2lab(<float3|float4> rgb)
*/

#ifndef FNC_SRGB2LAB
#define FNC_SRGB2LAB
float3 srgb2lab(const in float3 srgb) { return rgb2lab(srgb2rgb(srgb));}
float4 srgb2lab(const in float4 srgb) { return float4(srgb2lab(srgb.rgb),rgb.a); }
#endif
