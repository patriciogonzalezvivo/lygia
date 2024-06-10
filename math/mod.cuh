#include "make.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the mod function for float2, float3, and float4 types, to match GLSL's behavior.
use: 
    - <float2|float3|float4> mod(<float2|float3|float4> a, <float2|float3|float4> b);
    - <float2|float3|float4> mod(<float2|float3|float4> a, float b);
    - <float> mod(float a, float b);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MOD
#define FNC_MOD

inline __host__ __device__ float  mod(float a, float b) { return fmodf(a, b); }
inline __host__ __device__ float2 mod(const float2& a, const float2& b) { return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y)); }
inline __host__ __device__ float3 mod(const float3& a, const float3& b) { return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z)); }
inline __host__ __device__ float4 mod(const float4& a, const float4& b) { return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w)); }

inline __host__ __device__ float2 mod(const float2& a, float b) { return make_float2(fmodf(a.x, b), fmodf(a.y, b)); }
inline __host__ __device__ float3 mod(const float3& a, float b) { return make_float3(fmodf(a.x, b), fmodf(a.y, b), fmodf(a.z, b)); }
inline __host__ __device__ float4 mod(const float4& a, float b) { return make_float4(fmodf(a.x, b), fmodf(a.y, b), fmodf(a.z, b), fmodf(a.w, b)); }

#endif