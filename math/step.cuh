#include "make.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the step function for float, float2, float3, and float4 types, to match GLSL's behavior.
use: 
    - <float> step(<float> a, <float> b);
    - <float2> step(<float2> a, <float2> b);
    - <float3> step(<float3> a, <float3> b);
    - <float4> step(<float4> a, <float4> b);
    - <float2> step(<float2> a, <float> b);
    - <float3> step(<float3> a, <float> b);
    - <float4> step(<float4> a, <float> b);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_STEP
#define FNC_STEP

inline __host__ __device__ float step(float a, float b) { return (a > b)? 1.0f : 0.0f; }

inline __host__ __device__ float2 step(float2 a, float b) { return make_float2(step(a.x, b), step(a.y, b)); }
inline __host__ __device__ float3 step(float3 a, float b) { return make_float3(step(a.x, b), step(a.y, b), step(a.z, b)); }
inline __host__ __device__ float4 step(float4 a, float b) { return make_float4(step(a.x, b), step(a.y, b), step(a.z, b), step(a.w, b)); }

inline __host__ __device__ float2 step(float2 a, float2 b) { return make_float2(step(a.x, b.x), step(a.y, b.y)); }
inline __host__ __device__ float3 step(float3 a, float3 b) { return make_float3(step(a.x, b.x), step(a.y, b.y), step(a.z, b.z)); }
inline __host__ __device__ float4 step(float4 a, float4 b) { return make_float4(step(a.x, b.x), step(a.y, b.y), step(a.z, b.z), step(a.w, b.w)); }

#endif