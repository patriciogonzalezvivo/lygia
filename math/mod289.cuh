#include "floor.cuh"
#include "operations.cuh"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: modulus of 289
use: <float|float2|float3|float4> mod289(<float|float2|float3|float4> x)
*/

#ifndef FNC_MOD289
#define FNC_MOD289

inline __device__ __host__ float mod289(float x) { return x - floor(x * (1.0f / 289.0f)) * 289.0f; }
inline __device__ __host__ float2 mod289(const float2& x) { return x - floor(x * (1.0f / 289.0f)) * 289.0f; }
inline __device__ __host__ float3 mod289(const float3& x) { return x - floor(x * (1.0f / 289.0f)) * 289.0f; }
inline __device__ __host__ float4 mod289(const float4& x) { return x - floor(x * (1.0f / 289.0f)) * 289.0f; }

#endif
