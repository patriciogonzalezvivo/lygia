#include "mod289.hlsl"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: permute
use: <float|float2|float3|float4> permute(<float|float2|float3|float4> x)
*/

#ifndef FNC_PERMUTE
#define FNC_PERMUTE

float permute(in float x) { return mod289(((x * 34.0) + 1.0) * x); }
float2 permute(in float2 x) { return mod289(((x * 34.0) + 1.0) * x); }
float3 permute(in float3 x) { return mod289(((x * 34.0) + 1.0) * x); }
float4 permute(in float4 x) { return mod289(((x * 34.0) + 1.0) * x); }

#endif
