#include "make.cuh"

////////////////////////////////////////////////////////////////////////////////
// max
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_MAX
#define FNC_MAX

#ifndef __CUDACC__
#include <math.h>
inline float max(float a, float b) { return a > b ? a : b; }
inline int max(int a, int b) { return a > b ? a : b; }
#endif

inline __host__ __device__ float2 max(float2 a, float2 b) { return make_float2(max(a.x, b.x), max(a.y, b.y)); }
inline __host__ __device__ float3 max(float3 a, float3 b) { return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
inline __host__ __device__ float4 max(float4 a, float4 b) { return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); }

inline __host__ __device__ int2 max(int2 a, int2 b) { return make_int2(max(a.x, b.x), max(a.y, b.y)); }
inline __host__ __device__ int3 max(int3 a, int3 b) { return make_int3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
inline __host__ __device__ int4 max(int4 a, int4 b) { return make_int4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); }

inline __host__ __device__ uint2 max(uint2 a, uint2 b) { return make_uint2(max(a.x, b.x), max(a.y, b.y)); }
inline __host__ __device__ uint3 max(uint3 a, uint3 b) { return make_uint3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
inline __host__ __device__ uint4 max(uint4 a, uint4 b) { return make_uint4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); }

#endif