#include "make.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the asin function for float2, float3, and float4 types, to match GLSL's behavior.
use: <float2|float3|float4> asin(<float2|float3|float4> value)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_ASIN
#define FNC_ASIN
inline __host__ __device__ float2 asin(const float2& v) { return make_float2(asin(v.x), asin(v.y)); }
inline __host__ __device__ float3 asin(const float3& v) { return make_float3(asin(v.x), asin(v.y), asin(v.z)); }
inline __host__ __device__ float4 asin(const float4& v) { return make_float4(asin(v.x), asin(v.y), asin(v.z), asin(v.w)); }
#endif