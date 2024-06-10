#include "make.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the sqrt function for float2, float3, and float4 types, to match GLSL's behavior.
use: <float2|float3|float4> sqrt(<float2|float3|float4> value);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SQRT
#define FNC_SQRT 

inline __host__ __device__ float2 sqrt(const float2& v) { return make_float2(sqrt(v.x), sqrt(v.y)); }
inline __host__ __device__ float3 sqrt(const float3& v) { return make_float3(sqrt(v.x), sqrt(v.y), sqrt(v.z)); }
inline __host__ __device__ float4 sqrt(const float4& v) { return make_float4(sqrt(v.x), sqrt(v.y), sqrt(v.z), sqrt(v.w)); }

#endif