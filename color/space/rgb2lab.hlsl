#include "rgb2xyz.hlsl"
#include "xyz2lab.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB color to Lab color space.
use: <float3|float4> rgb2lab(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RGB2LAB
#define FNC_RGB2LAB
float3 rgb2lab(in float3 rgb) { return xyz2lab( rgb2xyz( rgb ) ); }
float4 rgb2lab(in float4 rgb) { return float4(rgb2lab(rgb.rgb),rgb.a); }
#endif