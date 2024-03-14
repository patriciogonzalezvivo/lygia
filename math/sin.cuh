#include "make.cuh"

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_SIN
#define FNC_SIN
inline __host__ __device__ float2 sin(float2 v) { return make_float2(sin(v.x), sin(v.y)); }
inline __host__ __device__ float3 sin(float3 v) { return make_float3(sin(v.x), sin(v.y), sin(v.z)); }
inline __host__ __device__ float4 sin(float4 v) { return make_float4(sin(v.x), sin(v.y), sin(v.z), sin(v.w)); }
#endif