#include "space/hsl2rgb.hlsl"
#include "space/rgb2hsl.hlsl"

/*
contributors:
    - Johan Ismael
    - Patricio Gonzalez Vivo
description: Shifts color hue
use: hueShift(<float3|float4> color, <float> angle)
optionas:
    - HUESHIFT_AMOUNT: if defined, it uses a normalized value instead of an angle
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_HUESHIFT
#define FNC_HUESHIFT
float3 hueShift( float3 color, float a){
    float3 hsl = rgb2hsl(color);
#ifndef HUESHIFT_AMOUNT
    hsl.r = hsl.r * TAU + a;
    hsl.r = frac(hsl.r / TAU);
#else
    hsl.r += a;
#endif
    return hsl2rgb(hsl);
}

float4 hueShift(in float4 v, in float a) { return float4(hueShift(v.rgb, a), v.a);}
#endif