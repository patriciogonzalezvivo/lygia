/*
contributors: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: <float|float2|float3|float4> lerp(<float|float2|float3|float4> a, <float|float2|float3|float4> b, <float|float2|float3|float4> c [, <float|float2|float3|float4> d], <float> pct)
*/

#ifndef FNC_LERP
#define FNC_LERP
#define lerp(A, B, PCT) mix(A, B, PCT) 
#endif