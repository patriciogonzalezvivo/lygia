/*
original_author: [Ian McEwan, Ashima Arts]
description: 
use: taylorInvSqrt(<float|float4> x)
*/

#ifndef FNC_TAYLORINVSQRT
#define FNC_TAYLORINVSQRT
float taylorInvSqrt(in float r) { return 1.79284291400159 - 0.85373472095314 * r; }
float2 taylorInvSqrt(in float2 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float3 taylorInvSqrt(in float3 r) { return 1.79284291400159 - 0.85373472095314 * r; }
float4 taylorInvSqrt(in float4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

#endif