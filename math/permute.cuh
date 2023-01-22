#include "mod289.cuh"

/*
original_author: [Ian McEwan, Ashima Arts]
description: permute
use: permute(<float|float2|float3|float4> x)
*/

#ifndef FNC_PERMUTE
#define FNC_PERMUTE

inline __device__ __host__ float permute(float x) { return mod289(((x * 34.0f) + 1.0f) * x); }
inline __device__ __host__ float2 permute(float2 x) { return mod289(((x * 34.0f) + 1.0f) * x); }
inline __device__ __host__ float3 permute(float3 x) { return mod289(((x * 34.0f) + 1.0f) * x); }
inline __device__ __host__ float4 permute(float4 x) { return mod289(((x * 34.0f) + 1.0f) * x); }

#endif
