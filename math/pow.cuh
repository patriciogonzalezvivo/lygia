#include "make.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: pow for vectors
use: <float2|float3|float4> pow(<float2|float3|float4> value, <float|float2|float3|float4> e)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_POW
#define FNC_POW

inline __host__ __device__ float2 pow(const float2& v, float e) { return make_float2( pow(v.x, e), pow(v.y, e) ); }
inline __host__ __device__ float3 pow(const float3& v, float e) { return make_float3( pow(v.x, e), pow(v.y, e), pow(v.z, e) ); }
inline __host__ __device__ float4 pow(const float4& v, float e) { return make_float4( pow(v.x, e), pow(v.y, e), pow(v.z, e), pow(v.w, e) ); }

inline __host__ __device__ float2 pow(const float2& v, const float2& e) { return make_float2( pow(v.x, e.x), pow(v.y, e.y) ); }
inline __host__ __device__ float3 pow(const float3& v, const float3& e) { return make_float3( pow(v.x, e.x), pow(v.y, e.y), pow(v.z, e.z) ); }
inline __host__ __device__ float4 pow(const float4& v, const float4& e) { return make_float4( pow(v.x, e.x), pow(v.y, e.y), pow(v.z, e.z), pow(v.w, e.w) ); }

#endif