#include "make.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the sin function for float2, float3, and float4 types, to match GLSL's behavior.
use: <float2|float3|float4> sin(<float2|float3|float4> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_SIN
#define FNC_SIN
inline __host__ __device__ float2 sin(float2 v) { return make_float2(sin(v.x), sin(v.y)); }
inline __host__ __device__ float3 sin(float3 v) { return make_float3(sin(v.x), sin(v.y), sin(v.z)); }
inline __host__ __device__ float4 sin(float4 v) { return make_float4(sin(v.x), sin(v.y), sin(v.z), sin(v.w)); }
#endif