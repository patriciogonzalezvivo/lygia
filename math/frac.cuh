#include "floor.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the floor function for float2, float3, and float4 types, to match HLSL's behavior.
use: <float2|float3|float4> frac(<float2|float3|float4> value);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_FRAC
#define FNC_FRAC
inline __host__ __device__ float frac(float v) { return v - floorf(v); }
inline __host__ __device__ float2 frac(float2 v) { return make_float2(frac(v.x), frac(v.y)); }
inline __host__ __device__ float3 frac(float3 v) { return make_float3(frac(v.x), frac(v.y), frac(v.z)); }
inline __host__ __device__ float4 frac(float4 v) { return make_float4(frac(v.x), frac(v.y), frac(v.z), frac(v.w)); }
#endif