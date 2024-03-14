/*
contributors: Patricio Gonzalez Vivo
description: power of 7
use: <float|float2|float3|float4> pow7(<float|float2|float3|float4> x)
*/

#ifndef FNC_POW2
#define FNC_POW2

float pow7(in float x) { return x * x * x * x * x * x * x; }
float2 pow7(in float2 x) { return x * x * x * x * x * x * x; }
float3 pow7(in float3 x) { return x * x * x * x * x * x * x; }
float4 pow7(in float4 x) { return x * x * x * x * x * x * x; }

#endif
