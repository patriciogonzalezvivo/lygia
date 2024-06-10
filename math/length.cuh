#include "dot.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the length function for float2, float3, and float4 types, to match GLSL's behavior.
use: <float> length(<float2|float3|float4> value);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_LENGTH
#define FNC_LENGTH

inline __host__ __device__ float length(const float2& v) { return sqrtf(dot(v, v)); }
inline __host__ __device__ float length(const float3& v) { return sqrtf(dot(v, v)); }
inline __host__ __device__ float length(const float4& v) { return sqrtf(dot(v, v)); }

#endif