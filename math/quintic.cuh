#include "operations.cuh"

/*
contributors: Inigo Quiles
description: quintic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> quintic(<float|float2|float3|float4> value);
*/

#ifndef FNC_QUINTIC
#define FNC_QUINTIC 

inline __host__ __device__ float   quintic(float v)   { return v*v*v*(v*(v*6.0f-15.0f)+10.0f); }
inline __host__ __device__ float2  quintic(const float2& v)  { return v*v*v*(v*(v*6.0f-15.0f)+10.0f); }
inline __host__ __device__ float3  quintic(const float3& v)  { return v*v*v*(v*(v*6.0f-15.0f)+10.0f); }
inline __host__ __device__ float4  quintic(const float4& v)  { return v*v*v*(v*(v*6.0f-15.0f)+10.0f); }

#endif