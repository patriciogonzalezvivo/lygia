/*
contributors: nan
description: Linear tonemap (no modifications are applied)
use: <float3|float4> tonemapLinear(<float3|float4> x)
*/


#ifndef FNC_TONEMAPLINEAR
#define FNC_TONEMAPLINEAR
float3 tonemapLinear(const float3 x) { return x; }
float4 tonemapLinear(const float4 x) { return x; }
#endif