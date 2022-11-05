#include "rgb2xyz.hlsl"
#include "xyz2lab.hlsl"

/*
original_author: Patricio Gonzalez Vivo
description: Converts a RGB color to Lab color space.
use: rgb2lab(<float3|float4> color)
*/

#ifndef FNC_RGB2LAB
#define FNC_RGB2LAB
float3 rgb2lab(in float3 c) {
    float3 lab = xyz2lab( rgb2xyz( c ) );
    return float3(  lab.x / 100.0,
                    0.5 + 0.5 * (lab.y / 127.0),
                    0.5 + 0.5 * (lab.z / 127.0));
}

float4 rgb2lab(in float4 rgb) { return float4(rgb2lab(rgb.rgb),rgb.a); }
#endif