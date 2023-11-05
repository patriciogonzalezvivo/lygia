#include "operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: lerp(<float|float2|float3|float4> a, <float|float2|float3|float4> b, <float|float2|float3|float4> c [, <float|float2|float3|float4> d], <float> pct)
*/

#ifndef FNC_LERP
#define FNC_LERP

inline __device__ __host__ float lerp(float a, float b, float t) { return a + t*(b - a); }

inline __device__ __host__ float2 lerp(const float2& a, const float2& b, float t) { return a + t*(b - a); }
inline __device__ __host__ float3 lerp(const float3& a, const float3& b, float t) { return a + t*(b - a); }
inline __device__ __host__ float4 lerp(const float4& a, const float4& b, float t) { return a + t*(b - a); }

inline __device__ __host__ float2 lerp(const float2& a, const float2& b, const float2& t) { return a + t*(b - a); }
inline __device__ __host__ float3 lerp(const float3& a, const float3& b, const float3& t) { return a + t*(b - a); }
inline __device__ __host__ float4 lerp(const float4& a, const float4& b, const float4& t) { return a + t*(b - a); }

#endif