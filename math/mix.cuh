#include "operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: expands mix to linearly mix more than two values
use: <float|float2|float3|float4> mix(<float|float2|float3|float4> a, <float|float2|float3|float4> b, <float|float2|float3|float4> c [, <float|float2|float3|float4> d], <float> pct)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MIX
#define FNC_MIX

inline __device__ __host__ float mix(float a, float b, float t) { return a + t*(b - a); }

inline __device__ __host__ float2 mix(const float2& a, const float2& b, float t) { return a + t*(b - a); }
inline __device__ __host__ float3 mix(const float3& a, const float3& b, float t) { return a + t*(b - a); }
inline __device__ __host__ float4 mix(const float4& a, const float4& b, float t) { return a + t*(b - a); }

inline __device__ __host__ float2 mix(const float2& a, const float2& b, const float2& t) { return a + t*(b - a); }
inline __device__ __host__ float3 mix(const float3& a, const float3& b, const float3& t) { return a + t*(b - a); }
inline __device__ __host__ float4 mix(const float4& a, const float4& b, const float4& t) { return a + t*(b - a); }

#endif