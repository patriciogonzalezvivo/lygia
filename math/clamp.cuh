#include "min.cuh"
#include "max.cuh"

#ifndef FNC_CLAMP
#define FNC_CLAMP

/*
contributors: Patricio Gonzalez Vivo
description: this file contains the definition of the clamp function for float2, float3, and float4 types, to match GLSL's behavior.
use: 
    - <float2|float3|float4> clamp(<float2|float3|float4> value, float min, float max);
    - <float2|float3|float4> clamp(<float2|float3|float4> value, <float2|float3|float4> min, <float2|float3|float4> max);
    - <int2|int3|int4> clamp(<int2|int3|int4> value, int min, int max);
    - <int2|int3|int4> clamp(<int2|int3|int4> value, <int2|int3|int4> min, <int2|int3|int4> max);
    - <uint2|uint3|uint4> clamp(<uint2|uint3|uint4> value, uint min, uint max);
    - <uint2|uint3|uint4> clamp(<uint2|uint3|uint4> value, <uint2|uint3|uint4> min, <uint2|uint3|uint4> max);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

inline __device__ __host__ float clamp(float f, float a, float b) { return max(a, min(f, b)); }
inline __device__ __host__ int clamp(int f, int a, int b) { return max(a, min(f, b)); }
inline __device__ __host__ uint clamp(uint f, uint a, uint b) { return max(a, min(f, b)); }

inline __device__ __host__ float2 clamp(float2 v, float a, float b) { return make_float2(clamp(v.x, a, b), clamp(v.y, a, b)); }
inline __device__ __host__ float2 clamp(float2 v, float2 a, float2 b) { return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }

inline __device__ __host__ float3 clamp(float3 v, float a, float b) { return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)); }
inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b) { return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }

inline __device__ __host__ float4 clamp(float4 v, float a, float b) { return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)); }
inline __device__ __host__ float4 clamp(float4 v, float4 a, float4 b) { return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }

inline __device__ __host__ int2 clamp(int2 v, int a, int b) { return make_int2(clamp(v.x, a, b), clamp(v.y, a, b)); }
inline __device__ __host__ int2 clamp(int2 v, int2 a, int2 b) { return make_int2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }

inline __device__ __host__ int3 clamp(int3 v, int a, int b) { return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)); }
inline __device__ __host__ int3 clamp(int3 v, int3 a, int3 b) { return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }

inline __device__ __host__ int4 clamp(int4 v, int a, int b) { return make_int4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)); }
inline __device__ __host__ int4 clamp(int4 v, int4 a, int4 b) { return make_int4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }

inline __device__ __host__ uint2 clamp(uint2 v, uint a, uint b) { return make_uint2(clamp(v.x, a, b), clamp(v.y, a, b)); }
inline __device__ __host__ uint2 clamp(uint2 v, uint2 a, uint2 b) { return make_uint2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y)); }

inline __device__ __host__ uint3 clamp(uint3 v, uint a, uint b) { return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)); }
inline __device__ __host__ uint3 clamp(uint3 v, uint3 a, uint3 b) { return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z)); }

inline __device__ __host__ uint4 clamp(uint4 v, uint a, uint b) { return make_uint4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)); }
inline __device__ __host__ uint4 clamp(uint4 v, uint4 a, uint4 b) { return make_uint4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w)); }

#endif