/*
contributors: Inigo Quiles
description: quintic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> quintic(<float|float2|float3|float4> value);
*/

#ifndef FNC_QUINTIC
#define FNC_QUINTIC 

float   quintic(const in float v)   { return v*v*v*(v*(v*6.0-15.0)+10.0); }
float2  quintic(const in float2 v)  { return v*v*v*(v*(v*6.0-15.0)+10.0); }
float3  quintic(const in float3 v)  { return v*v*v*(v*(v*6.0-15.0)+10.0); }
float4  quintic(const in float4 v)  { return v*v*v*(v*(v*6.0-15.0)+10.0); }

#endif