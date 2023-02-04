#include "dot.cuh"

////////////////////////////////////////////////////////////////////////////////
// length
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_LENGTH
#define FNC_LENGTH

inline __host__ __device__ float length(const float2& v) { return sqrtf(dot(v, v)); }
inline __host__ __device__ float length(const float3& v) { return sqrtf(dot(v, v)); }
inline __host__ __device__ float length(const float4& v) { return sqrtf(dot(v, v)); }

#endif