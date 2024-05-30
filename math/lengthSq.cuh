#include "dot.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: Squared length
use: <float2|float3|float4> lengthSq(<float2|float3|float4> v)
*/

#ifndef FNC_LENGTHSQ
#define FNC_LENGTHSQ

inline __host__ __device__ float lengthSq(const float2& v) { return dot(v, v); }
inline __host__ __device__ float lengthSq(const float3& v) { return dot(v, v); }
inline __host__ __device__ float lengthSq(const float4& v) { return dot(v, v); }

#endif
