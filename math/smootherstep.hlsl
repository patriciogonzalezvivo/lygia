#include "quintic.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: quintic polynomial step function
use: <float|float2|float3|float4> smoothstep(<float|float2|float3|float4> in, <float|float2|float3|float4> out, <float|float2|float3|float4> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SMOOTHERSTEP
#define FNC_SMOOTHERSTEP
float smootherstep(float edge0, float edge1, float x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
float2 smootherstep(float2 edge0, float2 edge1, float2 x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
float3 smootherstep(float3 edge0, float3 edge1, float3 x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
float4 smootherstep(float4 edge0, float4 edge1, float4 x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
#endif