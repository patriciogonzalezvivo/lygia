/*
contributors: Inigo Quiles
description: inverse quartic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> invQuartic(<float|float2|float3|float4> value);
*/

#ifndef FNC_INVQUARTIC
#define FNC_INVQUARTIC 
float   invQuartic(const in float v)    { return sqrt(1.0-sqrt(1.0-v)); }
float2  invQuartic(const in float2 v)   { return sqrt(1.0-sqrt(1.0-v)); }
float3  invQuartic(const in float3 v)   { return sqrt(1.0-sqrt(1.0-v)); }
float4  invQuartic(const in float4 v)   { return sqrt(1.0-sqrt(1.0-v)); }
#endif