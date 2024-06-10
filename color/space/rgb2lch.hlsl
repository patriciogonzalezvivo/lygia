#include "rgb2lab.hlsl"
#include "lab2lch.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a RGB color to LCh color space.
use: <float3|float4> rgb2lch(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_RGB2LCH
#define FNC_RGB2LCH
float3 rgb2lch(const in float3 rgb) { return lab2lch(rgb2lab(rgb)); }
float4 rgb2lch(const in float4 rgb) { return float4(rgb2lch(rgb.rgb),rgb.a); }
#endif