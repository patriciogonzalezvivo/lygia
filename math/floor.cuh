#include "../math.cuh"

////////////////////////////////////////////////////////////////////////////////
// floor
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_FLOOR
#define FNC_FLOOR

inline __host__ __device__ float2 floorf(float2 v) { return make_float2(floorf(v.x), floorf(v.y)); }
inline __host__ __device__ float3 floorf(float3 v) { return make_float3(floorf(v.x), floorf(v.y), floorf(v.z)); }
inline __host__ __device__ float4 floorf(float4 v) { return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w)); }

#endif