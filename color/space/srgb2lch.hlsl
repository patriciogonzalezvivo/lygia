#include "rgb2lch.hlsl"
#include "srgb2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a sRGB color to Lab
use: <float3|float4> srgb2lch(<float3|float4> rgb)
*/

#ifndef FNC_SRGB2LCH
#define FNC_SRGB2LCH
float3 srgb2lch(const in float3 srgb) { return rgb2lch(srgb2rgb(srgb));}
float4 srgb2lch(const in float4 srgb) { return float4(srgb2lch(srgb.rgb),srgb.a); }
#endif
