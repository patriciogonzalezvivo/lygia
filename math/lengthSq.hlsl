/*
contributors: Patricio Gonzalez Vivo
description: Squared length
use: <float2|float3|float4> lengthSq(<float2|float3|float4> v)
*/

#ifndef FNC_LENGTHSQ
#define FNC_LENGTHSQ

float lengthSq(in float2 v) { return dot(v, v); }
float lengthSq(in float3 v) { return dot(v, v); }
float lengthSq(in float4 v) { return dot(v, v); }

#endif
