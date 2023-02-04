#include "saturate.cuh"
#include "quintic.cuh"

/*
original_author: Patricio Gonzalez Vivo
description: quintic polynomial step function 
use: smoothstep(<float> in, <float> out, <float> value)
*/

#ifndef FNC_SMOOTHERSTEP
#define FNC_SMOOTHERSTEP

inline __host__ __device__ float  smootherstep(float edge0, float edge1, float x) { return quintic( clamp( (x - edge0)/(edge1 - edge0), 0.0f, 1.0f )); }
inline __host__ __device__ float2 smootherstep(const float2& edge0, const float2& edge1, const float2& x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
inline __host__ __device__ float3 smootherstep(const float3& edge0, const float3& edge1, const float3& x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }
inline __host__ __device__ float4 smootherstep(const float4& edge0, const float4& edge1, const float4& x) { return quintic( saturate( (x - edge0)/(edge1 - edge0) )); }

#endif