/*
contributors: Inigo Quiles
description: inverse cubic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> invCubic(<float|float2|float3|float4> value);
*/

#ifndef FNC_INVCUBIC
#define FNC_INVCUBIC 
float   invCubic(const in float v)  { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
float2  invCubic(const in float2 v) { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
float3  invCubic(const in float3 v) { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
float4  invCubic(const in float4 v) { return 0.5-sin(asin(1.0-2.0*v)/3.0); }
#endif