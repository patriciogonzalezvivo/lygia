#include "clamp.cuh"
#include "operations.cuh"

/*
contributors:
description: 
use: 
*/

////////////////////////////////////////////////////////////////////////////////
// smoothstep
// - returns 0 if x < a
// - returns 1 if x > b
// - otherwise returns smooth interpolation between 0 and 1 based on x
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_SMOOTHSTEP
#define FNC_SMOOTHSTEP

inline __device__ __host__ float smoothstep(float a, float b, float x) {
	float y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(3.0f - (2.0f*y)));
}

inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x) {
	float2 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(make_float2(3.0f) - (make_float2(2.0f)*y)));
}

inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x) {
	float3 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(make_float3(3.0f) - (make_float3(2.0f)*y)));
}

inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x) {
	float4 y = clamp((x - a) / (b - a), 0.0f, 1.0f);
	return (y*y*(make_float4(3.0f) - (make_float4(2.0f)*y)));
}

#endif
