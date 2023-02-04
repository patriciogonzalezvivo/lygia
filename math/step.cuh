#include "make.cuh"

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_STEP
#define FNC_STEP

inline __host__ __device__ float step(float a, float b) { return (a > b)? 1.0f : 0.0f; }

inline __host__ __device__ float2 step(float2 a, float b) { return make_float2(step(a.x, b), step(a.y, b)); }
inline __host__ __device__ float3 step(float3 a, float b) { return make_float3(step(a.x, b), step(a.y, b), step(a.z, b)); }
inline __host__ __device__ float4 step(float4 a, float b) { return make_float4(step(a.x, b), step(a.y, b), step(a.z, b), step(a.w, b)); }

inline __host__ __device__ float2 step(float2 a, float2 b) { return make_float2(step(a.x, b.x), step(a.y, b.y)); }
inline __host__ __device__ float3 step(float3 a, float3 b) { return make_float3(step(a.x, b.x), step(a.y, b.y), step(a.z, b.z)); }
inline __host__ __device__ float4 step(float4 a, float4 b) { return make_float4(step(a.x, b.x), step(a.y, b.y), step(a.z, b.z), step(a.w, b.w)); }

#endif