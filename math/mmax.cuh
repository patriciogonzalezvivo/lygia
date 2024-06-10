#include "max.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: extend Max function to add more arguments
use:
    - <float> mmax(<float> A, <float> B, <float> C[, <float> D])
    - <float2|float3|float4> mmax(<float2|float3|float4> A)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_MMAX
#define FNC_MMAX

inline __device__ __host__ float mmax(float a, float b) { return max(a, b); }
inline __device__ __host__ float mmax(float a, float b, float c) { return max(a, max(b, c)); }
inline __device__ __host__ float mmax(float a, float b, float c, float d) { return max(max(a, b), max(c, d)); }

inline __device__ __host__ float mmax(const float2& v) { return max(v.x, v.y); }
inline __device__ __host__ float mmax(const float3& v) { return mmax(v.x, v.y, v.z); }
inline __device__ __host__ float mmax(const float4& v) { return mmax(v.x, v.y, v.z, v.w); }

#endif
