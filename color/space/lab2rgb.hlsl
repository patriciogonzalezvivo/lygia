#include "lab2xyz.hlsl"
#include "xyz2rgb.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: Converts a Lab color to RGB color space.
use: lab2rgb(<float3|float4> color)
*/

#ifndef FNC_LAB2RGB
#define FNC_LAB2RGB
float3 lab2rgb(in float3 lab) { return xyz2rgb( lab2xyz( lab ) ); }
float4 lab2rgb(in float4 lab) { return float4(lab2rgb(lab.rgb), lab.a); }
#endif