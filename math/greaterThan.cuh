#include "step.cuh"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: greaterThan, returns 1 if x > y, 0 otherwise
use: greaterThan(<float|float2|float3|float4> x, y)
*/

#ifndef FNC_GREATERTHAN
#define FNC_GREATERTHAN
inline __host__ __device__ float greaterThan(float x, float y) { return step(y, x); }
inline __host__ __device__ float2 greaterThan(const float2& x, const float2& y) { return step(y, x); }
inline __host__ __device__ float3 greaterThan(const float3& x, const float3& y) { return step(y, x); }
inline __host__ __device__ float4 greaterThan(const float4& x, const float4& y) { return step(y, x); } 
#endif
