#include "make.cuh"

////////////////////////////////////////////////////////////////////////////////
// fmod
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_MOD
#define FNC_MOD

inline __host__ __device__ float  mod(float a, float b) { return fmodf(a, b); }
inline __host__ __device__ float2 mod(const float2& a, const float2& b) { return make_float2(fmodf(a.x, b.x), fmodf(a.y, b.y)); }
inline __host__ __device__ float3 mod(const float3& a, const float3& b) { return make_float3(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z)); }
inline __host__ __device__ float4 mod(const float4& a, const float4& b) { return make_float4(fmodf(a.x, b.x), fmodf(a.y, b.y), fmodf(a.z, b.z), fmodf(a.w, b.w)); }

inline __host__ __device__ float2 mod(const float2& a, float b) { return make_float2(fmodf(a.x, b), fmodf(a.y, b)); }
inline __host__ __device__ float3 mod(const float3& a, float b) { return make_float3(fmodf(a.x, b), fmodf(a.y, b), fmodf(a.z, b)); }
inline __host__ __device__ float4 mod(const float4& a, float b) { return make_float4(fmodf(a.x, b), fmodf(a.y, b), fmodf(a.z, b), fmodf(a.w, b)); }

#endif