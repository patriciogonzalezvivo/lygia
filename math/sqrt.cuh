#include "make.cuh"

/*
contributors: Inigo Quiles
description: inverse quartic polynomial https://iquilezles.org/articles/smoothsteps/
use: <float|float2|float3|float4> invQuartic(<float|float2|float3|float4> value);
*/

#ifndef FNC_SQRT
#define FNC_SQRT 

inline __host__ __device__ float2 sqrt(const float2& v) { return make_float2(sqrt(v.x), sqrt(v.y)); }
inline __host__ __device__ float3 sqrt(const float3& v) { return make_float3(sqrt(v.x), sqrt(v.y), sqrt(v.z)); }
inline __host__ __device__ float4 sqrt(const float4& v) { return make_float4(sqrt(v.x), sqrt(v.y), sqrt(v.z), sqrt(v.w)); }

#endif