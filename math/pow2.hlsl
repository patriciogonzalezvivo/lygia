/*
contributors: Patricio Gonzalez Vivo
description: power of 2
use: <float|float2|float3|float4> pow2(<float|float2|float3|float4> x)
*/

#ifndef FNC_POW2
#define FNC_POW2

float  pow2(in float x) { return x * x; }
float2 pow2(in float2 x) { return x * x; }
float3 pow2(in float3 x) { return x * x; }
float4 pow2(in float4 x) { return x * x; }

#endif
