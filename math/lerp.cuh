#include "../math.cuh"

/*
original_author: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: lerp(<float|float2|float3|float4> a, <float|float2|float3|float4> b, <float|float2|float3|float4> c [, <float|float2|float3|float4> d], <float> pct)
*/

#ifndef FNC_LERP
#define FNC_LERP

inline __device__ __host__ float lerp(float a, float b, float t) { return a + t*(b - a); }
inline __device__ __host__ float2 lerp(float2 a, float2 b, float t) { return a + t*(b - a); }
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t) { return a + t*(b - a); }
inline __device__ __host__ float4 lerp(float4 a, float4 b, float t) { return a + t*(b - a); }

#endif