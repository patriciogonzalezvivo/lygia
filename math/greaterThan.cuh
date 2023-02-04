#include "step.cuh"

/*
original_author: [Ian McEwan, Ashima Arts]
description: grad4, used for snoise(float4 v)
use: grad4(<float> j, <float4> ip)

*/
#ifndef FNC_GRATERTHAN
#define FNC_GRATERTHAN

inline __host__ __device__ float greaterThan(float x, float y) { return step(y, x); }
inline __host__ __device__ float2 greaterThan(const float2& x, const float2& y) { return step(y, x); }
inline __host__ __device__ float3 greaterThan(const float3& x, const float3& y) { return step(y, x); }
inline __host__ __device__ float4 greaterThan(const float4& x, const float4& y) { return step(y, x); }
    
#endif
