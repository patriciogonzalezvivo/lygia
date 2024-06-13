#include "lch2lab.hlsl"
#include "lab2rgb.hlsl"
/*
contributors: Patricio Gonzalez Vivo
description: "Converts a Lch to linear RGB color space. \nNote: LCh is simply Lab  but converted to polar coordinates (in degrees).\n"
use: <float3|float4> lch2rgb(<float3|float4> color)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LCH2RGB
#define FNC_LCH2RGB
float3 lch2rgb(float3 lch) { return lab2rgb( lch2lab(lch) ); }
float4 lch2rgb(float4 lch) { return float4(lch2rgb(lch.xyz),lch.a);}
#endif