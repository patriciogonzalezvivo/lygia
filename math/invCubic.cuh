#include "operations.cuh"
#include "sin.cuh"
#include "asin.cuh"

/*
contributors: Inigo Quiles
description: inverse cubic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> invCubic(<float|float2|float3|float4> value);
*/

#ifndef FNC_INVCUBIC
#define FNC_INVCUBIC 
inline __host__ __device__ float   invCubic(float v)  { return 0.5-sin(asin(1.0f - 2.0f * v) / 3.0f); }
inline __host__ __device__ float2  invCubic(const float2& v) { return 0.5-sin(asin(1.0f - 2.0f * v) / 3.0f); }
inline __host__ __device__ float3  invCubic(const float3& v) { return 0.5-sin(asin(1.0f - 2.0f * v) / 3.0f); }
inline __host__ __device__ float4  invCubic(const float4& v) { return 0.5-sin(asin(1.0f - 2.0f * v) / 3.0f); }
#endif