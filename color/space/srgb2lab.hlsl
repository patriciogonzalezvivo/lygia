#include "rgb2lab.hlsl"
#include "srgb2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a sRGB color to Lab
use: <float3|float4> srgb2lab(<float3|float4> rgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SRGB2LAB
#define FNC_SRGB2LAB
float3 srgb2lab(const in float3 srgb) { return rgb2lab(srgb2rgb(srgb));}
float4 srgb2lab(const in float4 srgb) { return float4(srgb2lab(srgb.rgb),rgb.a); }
#endif
