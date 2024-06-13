#include "dot.cuh"
#include "operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the normalize function for float2, float3, and float4 types, to match GLSL's behavior.
use: <float2|float3|float4> normalize(<float2|float3|float4> value);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_NORMALIZE
#define FNC_NORMALIZE

#ifndef __CUDACC__
#include <math.h>
inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif

inline __host__ __device__ float2 normalize(float2 v) {
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}

inline __host__ __device__ float3 normalize(float3 v) {
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}

inline __host__ __device__ float4 normalize(float4 v) {
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}

#endif