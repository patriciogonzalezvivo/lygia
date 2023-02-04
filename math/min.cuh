#include "make.cuh"

////////////////////////////////////////////////////////////////////////////////
// min
////////////////////////////////////////////////////////////////////////////////

#ifndef FNC_MIN
#define FNC_MIN

#ifndef __CUDACC__
#include <math.h>
inline float min(float a, float b) { return a < b ? a : b; }
inline int min(int a, int b) { return a < b ? a : b; }
#endif

inline  __host__ __device__ float2 min(float2 a, float2 b) { return make_float2(min(a.x, b.x), min(a.y, b.y));}
inline __host__ __device__ float3 min(float3 a, float3 b) { return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));}
inline  __host__ __device__ float4 min(float4 a, float4 b) { return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));}

inline __host__ __device__ int2 min(int2 a, int2 b) { return make_int2(min(a.x, b.x), min(a.y, b.y));}
inline __host__ __device__ int3 min(int3 a, int3 b) { return make_int3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));}
inline __host__ __device__ int4 min(int4 a, int4 b) { return make_int4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));}

inline __host__ __device__ uint2 min(uint2 a, uint2 b) { return make_uint2(min(a.x, b.x), min(a.y, b.y));}
inline __host__ __device__ uint3 min(uint3 a, uint3 b) { return make_uint3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));}
inline __host__ __device__ uint4 min(uint4 a, uint4 b) { return make_uint4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));}

#endif