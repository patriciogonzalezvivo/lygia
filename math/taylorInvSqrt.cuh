#include "operations.cuh"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: Fast, accurate inverse square root. 
use: <float|float2|float3|float4> taylorInvSqrt(<float|float2|float3|float4> x)
*/

#ifndef FNC_TAYLORINVSQRT
#define FNC_TAYLORINVSQRT
inline __host__ __device__ float taylorInvSqrt(float r) { return 1.79284291400159f - 0.85373472095314f * r; }
inline __host__ __device__ float2 taylorInvSqrt(const float2& r) { return 1.79284291400159f - 0.85373472095314f * r; }
inline __host__ __device__ float3 taylorInvSqrt(const float3& r) { return 1.79284291400159f - 0.85373472095314f * r; }
inline __host__ __device__ float4 taylorInvSqrt(const float4& r) { return 1.79284291400159f - 0.85373472095314f * r; }
#endif