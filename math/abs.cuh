#include "make.cuh"

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_ABS
#define FNC_ABS

// inline __host__ __device__ float  abs(float  v) { return fabs(v); }
inline __host__ __device__ float2 abs(const float2& v) { return make_float2(fabs(v.x), fabs(v.y)); }
inline __host__ __device__ float3 abs(const float3& v) { return make_float3(fabs(v.x), fabs(v.y), fabs(v.z)); }
inline __host__ __device__ float4 abs(const float4& v) { return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w)); }

inline __host__ __device__ int2 abs(int2 v) { return make_int2(abs(v.x), abs(v.y)); }
inline __host__ __device__ int3 abs(int3 v) { return make_int3(abs(v.x), abs(v.y), abs(v.z)); }
inline __host__ __device__ int4 abs(int4 v) { return make_int4(abs(v.x), abs(v.y), abs(v.z), abs(v.w)); }

#endif