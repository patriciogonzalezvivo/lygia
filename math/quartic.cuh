#include "operations.cuh"

/*
contributors: Inigo Quiles
description: quartic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> quartic(<float|float2|float3|float4> value);
*/

#ifndef FNC_QUARTIC
#define FNC_QUARTIC

inline __host__ __device__ float   quartic(float v)   { return v*v*(2.0f-v*v); }
inline __host__ __device__ float2  quartic(const float2& v)  { return v*v*(2.0f-v*v); }
inline __host__ __device__ float3  quartic(const float3& v)  { return v*v*(2.0f-v*v); }
inline __host__ __device__ float4  quartic(const float4& v)  { return v*v*(2.0f-v*v); }

#endif