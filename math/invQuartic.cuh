#include "operations.cuh"
#include "sqrt.cuh"

/*
contributors: Inigo Quiles
description: inverse quartic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> invQuartic(<float|float2|float3|float4> value);
*/

#ifndef FNC_INVQUARTIC
#define FNC_INVQUARTIC 
inline __host__ __device__ float   invQuartic(float v)    { return sqrt(1.0f - sqrt(1.0f - v)); }
inline __host__ __device__ float2  invQuartic(const float2& v)   { return sqrt(1.0f - sqrt(1.0f - v)); }
inline __host__ __device__ float3  invQuartic(const float3& v)   { return sqrt(1.0f - sqrt(1.0f - v)); }
inline __host__ __device__ float4  invQuartic(const float4& v)   { return sqrt(1.0f - sqrt(1.0f - v)); }
#endif
