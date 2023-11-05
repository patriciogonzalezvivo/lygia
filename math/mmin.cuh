#include "min.cuh"

/*
contributors: Patricio Gonzalez Vivo
description: extend min function to add more arguments
use: 
  - min(<float> A, <float> B, <float> C[, <float> D])
  - min(<float2|float3|float4> A)
*/

#ifndef FNC_MMIN
#define FNC_MMIN

inline __device__ __host__ float mmin(float a, float b) { return min(a, b); }
inline __device__ __host__ float mmin(float a, float b, float c) { return min(a, min(b, c)); }
inline __device__ __host__ float mmin(float a, float b, float c, float d) { return min(min(a,b), min(c, d)); }

inline __device__ __host__ float mmin(const float2& v) { return min(v.x, v.y); }
inline __device__ __host__ float mmin(const float3& v) { return mmin(v.x, v.y, v.z); }
inline __device__ __host__ float mmin(const float4& v) { return mmin(v.x, v.y, v.z, v.w); }

#endif
