#include "rgb2xyz.hlsl"
#include "srgb2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a sRGB color to XYZ
use: <float3|float4> srgb2xyz(<float3|float4> rgb)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RGB2XYZ
#define FNC_RGB2XYZ
float3 srgb2xyz(in float3 srgb) { return rgb2xyz(rgb2srgb(srgb));}
float4 srgb2xyz(in float4 srgb) { return float4(rgb2xyz(srgb.rgb),rgb.a); }
#endif
