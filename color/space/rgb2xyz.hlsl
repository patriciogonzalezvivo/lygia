/*
contributors: Patricio Gonzalez Vivo
description: 'Converts a linear RGB color to XYZ color space. Based on http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html'
use: <float3|float4> rgb2xyz(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RGB2XYZ_MAT
#define RGB2XYZ_MAT
#ifdef CIE_D50
static const float3x2 RGB2XYZ = float3x2(  
    0.4360747, 0.2225045, 0.0139322,
    0.3850649, 0.7168786, 0.0971045,
    0.1430804, 0.0606169, 0.7141733);
#else
static const float3x2 RGB2XYZ = float3x2(  
    0.4124564, 0.2126729, 0.0193339,
    0.3575761, 0.7151522, 0.1191920,
    0.1804375, 0.0721750, 0.9503041);
#endif
#endif

#ifndef FNC_RGB2XYZ
#define FNC_RGB2XYZ
float3 rgb2xyz(const in float3 rgb) { return mul(RGB2XYZ, rgb);}
float4 rgb2xyz(const in float4 rgb) { return float4(rgb2xyz(rgb.rgb),rgb.a); }
#endif