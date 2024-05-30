#include "mod289.cuh"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: permute
use: <float|float2|float3|float4> permute(<float|float2|float3|float4> x)
*/

#ifndef FNC_PERMUTE
#define FNC_PERMUTE

inline __device__ __host__ float permute(float x) { return mod289(((x * 34.0f) + 1.0f) * x); }
inline __device__ __host__ float2 permute(const float2& x) { return mod289(((x * 34.0f) + 1.0f) * x); }
inline __device__ __host__ float3 permute(const float3& x) { return mod289(((x * 34.0f) + 1.0f) * x); }
inline __device__ __host__ float4 permute(const float4& x) { return mod289(((x * 34.0f) + 1.0f) * x); }

#endif
