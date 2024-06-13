#include "make.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the dot function for float2, float3, and float4 types, to match GLSL's behavior.
use: <float|int|uint> dot(<float2|int2|uint2|float3|int3|uint3|float4|int4|uint4> a, <float2|int2|uint2|float3|int3|uint3|float4|int4|uint4> b);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_DOT
#define FNC_DOT
inline __host__ __device__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
inline __host__ __device__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __host__ __device__ float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

inline __host__ __device__ int dot(int2 a, int2 b) { return a.x * b.x + a.y * b.y; }
inline __host__ __device__ int dot(int3 a, int3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __host__ __device__ int dot(int4 a, int4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

inline __host__ __device__ uint dot(uint2 a, uint2 b) { return a.x * b.x + a.y * b.y; }
inline __host__ __device__ uint dot(uint3 a, uint3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __host__ __device__ uint dot(uint4 a, uint4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
#endif