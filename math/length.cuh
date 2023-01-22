////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_LENGTH
#define FNC_LENGTH

#include "dot.cuh"

inline __host__ __device__ float length(float2 v) { return sqrtf(dot(v, v)); }
inline __host__ __device__ float length(float3 v) { return sqrtf(dot(v, v)); }
inline __host__ __device__ float length(float4 v) { return sqrtf(dot(v, v)); }

#endif