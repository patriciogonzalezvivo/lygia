#include "operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: power of 2
use: <float|float2|float3|float4> pow2(<float|float2|float3|float4> x)
*/

#ifndef FNC_POW2
#define FNC_POW2

inline __host__ __device__ float  pow2(float x) { return x * x; }
inline __host__ __device__ float2 pow2(const float2& x) { return x * x; }
inline __host__ __device__ float3 pow2(const float3& x) { return x * x; }
inline __host__ __device__ float4 pow2(const float4& x) { return x * x; }

#endif
