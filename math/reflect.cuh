#include "dot.cuh"

////////////////////////////////////////////////////////////////////////////////
// reflect
// - returns reflection of incident ray I around surface normal N
// - N should be normalized, reflected vector's length is equal to length of I
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_REFLECT
#define FNC_REFLECT

inline __host__ __device__ float3 reflect(float3 i, float3 n) { return i - 2.0f * n * dot(n, i); }

#endif