/*
contributors: [Stefan Gustavson, Ian McEwan]
description: Fast, accurate inverse square root. 
use: <float|float2|float3|float4> taylorInvSqrt(<float|float2|float3|float4> x)
*/

#ifndef FNC_TAYLORINVSQRT
#define FNC_TAYLORINVSQRT
float taylorInvSqrt(in float r) { return 1.79284291400159 - 0.85373472095314 * r; }
float2 taylorInvSqrt(in float2 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float3 taylorInvSqrt(in float3 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float4 taylorInvSqrt(in float4 r) { return 1.79284291400159 - 0.85373472095314 * r; }
#endif