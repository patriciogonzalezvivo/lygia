#include "make.cuh"

////////////////////////////////////////////////////////////////////////////////
// absolute value
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_ASIN
#define FNC_ASIN
inline __host__ __device__ float2 asin(const float2& v) { return make_float2(asin(v.x), asin(v.y)); }
inline __host__ __device__ float3 asin(const float3& v) { return make_float3(asin(v.x), asin(v.y), asin(v.z)); }
inline __host__ __device__ float4 asin(const float4& v) { return make_float4(asin(v.x), asin(v.y), asin(v.z), asin(v.w)); }
#endif