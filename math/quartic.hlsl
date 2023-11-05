/*
contributors: Inigo Quiles
description: quartic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> quartic(<float|float2|float3|float4> value);
*/

#ifndef FNC_QUARTIC
#define FNC_QUARTIC

float   quartic(const in float v)   { return v*v*(2.0-v*v); }
float2  quartic(const in float2 v)  { return v*v*(2.0-v*v); }
float3  quartic(const in float3 v)  { return v*v*(2.0-v*v); }
float4  quartic(const in float4 v)  { return v*v*(2.0-v*v); }

#endif