#include "floor.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the floor function for float2, float3, and float4 types, to match GLSL's behavior.
use: <float2|float3|float4> fract(<float2|float3|float4> value);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FRACT
#define FNC_FRACT
inline __host__ __device__ float fract(float v) { return v - floorf(v); }
inline __host__ __device__ float2 fract(const float2& v) { return make_float2(fract(v.x), fract(v.y)); }
inline __host__ __device__ float3 fract(const float3& v) { return make_float3(fract(v.x), fract(v.y), fract(v.z)); }
inline __host__ __device__ float4 fract(const float4& v) { return make_float4(fract(v.x), fract(v.y), fract(v.z), fract(v.w)); }
#endif