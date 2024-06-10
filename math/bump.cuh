#include "saturate.cuh"
#include "operations.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: bump in a range between -1 and 1
use: <float|float3> bump(<float|float3> x [, <float|float3> k])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_BUMP
#define FNC_BUMP

inline __host__ __device__ float bump(const float x) { return max(1.0f - x * x, 0.0f); }
inline __host__ __device__ float bump(float x, float k){ return clamp( (1.0 - x * x) - k, 0.0f, 1.0f); }

inline __host__ __device__ float3 bump(const float3& x) { return max(make_float3(1.0f) - x * x, make_float3(0.0f)); }
inline __host__ __device__ float3 bump(const float3& x, const float3& k){ return saturate( (1.0f - x * x) - k); }

#endif