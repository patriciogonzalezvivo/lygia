#include "floor.cuh"

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_FRAC
#define FNC_FRAC
inline __host__ __device__ float frac(float v) { return v - floorf(v); }
inline __host__ __device__ float2 frac(float2 v) { return make_float2(frac(v.x), frac(v.y)); }
inline __host__ __device__ float3 frac(float3 v) { return make_float3(frac(v.x), frac(v.y), frac(v.z)); }
inline __host__ __device__ float4 frac(float4 v) { return make_float4(frac(v.x), frac(v.y), frac(v.z), frac(v.w)); }
#endif