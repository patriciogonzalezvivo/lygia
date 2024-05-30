#include "floor.cuh"

////////////////////////////////////////////////////////////////////////////////
// frac - returns the fractional portion of a scalar or each vector component
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_FRACT
#define FNC_FRACT
inline __host__ __device__ float fract(float v) { return v - floorf(v); }
inline __host__ __device__ float2 fract(const float2& v) { return make_float2(fract(v.x), fract(v.y)); }
inline __host__ __device__ float3 fract(const float3& v) { return make_float3(fract(v.x), fract(v.y), fract(v.z)); }
inline __host__ __device__ float4 fract(const float4& v) { return make_float4(fract(v.x), fract(v.y), fract(v.z), fract(v.w)); }
#endif