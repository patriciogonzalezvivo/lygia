#include "lch2lab.hlsl"
#include "lab2rgb.hlsl"
/*
contributors: Patricio Gonzalez Vivo
description: |
    Converts a Lch to linear RGB color space. 
    Note: LCh is simply Lab but converted to polar coordinates (in degrees).
use: <float3|float4> lch2rgb(<float3|float4> color)
*/

#ifndef FNC_LCH2RGB
#define FNC_LCH2RGB
float3 lch2rgb(float3 lch) { return lab2rgb( lch2lab(lch) ); }
float4 lch2rgb(float4 lch) { return float4(lch2rgb(lch.xyz),lch.a);}
#endif